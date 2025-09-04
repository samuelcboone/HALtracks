import os
import json
import numpy as np

from pathlib import Path
from PIL import Image, ImageOps
import time

from torch.utils.data import Dataset, DataLoader, sampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchvision.transforms as T
import torch
torch.cuda.empty_cache()
torch.manual_seed(42)

from catalyst.dl import SupervisedRunner, DiceCallback, EarlyStoppingCallback
from catalyst.utils import onnx_export

BATCH_SIZE = 2
NUM_EPOCHS = 300

vfnames = []
with open('validation_list.txt', 'r') as fp:
    lst = fp.readlines()
    vfnames = [l.strip() for l in lst]
    print( len( vfnames ) )

X_path = 'inputs'
y_path = 'labels'
base_path = '/data/derived/'
logdir = "results2ws2000/logs/segmentation_script_%s" % NUM_EPOCHS
results = 'results2ws2000/results%s/' % NUM_EPOCHS

TRAIN_SPLIT = 80
TEST_SPLIT  = 20

use_classes = [
'01_Count', '025-1', '02_Count', '03_Count', '04_Count', '05_Count', '06_Count', '0705-19', 
'107-1', '109-2', '110-2', '111-2', '113-2', '115-1', '119-A', '123-A', '12_Grain', '1-2_Grain', 
'133-2', '13FT-5', '13_Grain', '142-1', '14_Grain', '15_Grain', '16_Grain', '18A', '1_Grain',
'2016_17-10-393', '2016_20-01L', '2016_23-01', '2016_395', '2016_LKA-15', '2016_LKA-38',
'2016_LKA-42', '21big', '21small', '2-2_AGE', '22_Grain', '23_little_apatite', '24_Grain',
'2_Grain', '3-1_AGE', '3_Grain', '404_13_41', '4D-3', '5_Grain', '64-1', '65-1', '66-1', '69-1',
'6_Grain', '6I-18', '6I-9', '70-1', '7-2', '8_Grain', '96Z', 'AF7', 'ag-10-02_G50', 'BRFT',
'C-1_C15-1', 'D2-UM', 'DN6A', 'DUR-1', 'Durango', 'Fish_Canyon', 'GD2', 'GOW', 'HM82-1', 'IR36',
'IR39', 'IR43', 'KDB21', 'KDB23', 'KLG-16', 'LS16', 'LX18', 'MD10', 'MD15', 'MD16', 'MD23',
'MICA', 'MIN006', 'MK18', 'MP-01', 'MTC007', 'OC9002', 'P67458nm', 'PAL33', 'PL18', 'PZ',
'QZ13', 'RH-11', 'S16', 'SAP_B', 'SH', 'STAV5', 'STB', 'TEL', 'Temora', 'TEMORA', 'TU13',
'TUB13', 'TUB15', 'TW650', 'UW98', 'W-127', 'W-34', 'w-70', 'W-96', 'X17L', 'XF1015',
'XF1018', 'XF1020', 'XF1021', 'XF1023', 'XF1034', 'XF1039', 'XF282', 'XL-4-3-1', 'Y10',
'Y12', 'Y15', 'Y17', 'Y19', 'Y20', 'Y23', 'Y25', 'Y29', 'Y2-A', 'Y30', 'Y32', 'Y40', 'Y43',
'Y46', 'ZB11'
]

#use_classes = ['Durango', '63-1_Age', '64-1', '65-1', '66-1_Age', '69-1_Age', '70-1_Age',
#    'MICA', 'ag', 'MtDromedary', 'Fish_Canyon', 'T12']
#use_classes = ['Durango']

def standardize_image( img ):
    trf = T.Compose([T.Resize(2000), T.CenterCrop(1800)])  #1200, 1000
    return trf(img)

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

unloader = T.ToPILImage()

def custom_replace(tensor, watershed):
    # we create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor <= watershed] = 0
    res[tensor > watershed ] = 1
    return res

def predb_to_mask(predb, idx):
    imag1 = ( predb['logits'][idx] ).cpu().clone()
    image = custom_replace(imag1, 0)
    image = image.squeeze(0)
    image = unloader(image)
    return image

class GrainDataset(Dataset):
    def __init__(self, image_path, label_path, is_test):
        super().__init__()

        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = self.combine_files( image_path, label_path, is_test )

    def combine_files(self, image_path, label_path, is_test):
        files = []
        for f in image_path.iterdir():
            if f.is_dir(): continue
            if 'Trans' in f.name: continue
            if f.name.startswith('._'): continue
            usage = False
            for clss in use_classes:
                if clss in f.name:
                    usage = True
            trans = f.name.replace('ReflStackFlat', 'TransFFT')

            #include stated files only for hold out
            if is_test:
                if trans in vfnames: usage = True
                else: usage = False
            else:
                if trans in vfnames: usage = False
                else: usage = True

            if not usage: continue

            if 'ReflStackFlat' in f.name:
                #use this
                #eg. images: Durango_1_Grain01_ReflStackFlat.tif and Durango_1_Grain01_TransFFT.tif
                # mask: Durango_1_Grain01_L.png
                trans = image_path/f.name.replace('ReflStackFlat', 'TransFFT')
                mask = label_path/f.name.replace('ReflStackFlat', 'L').replace('tif', 'png')
                if trans.exists and mask.exists:
                    file_set = {
                        'refl' : f,
                        'trans': trans,
                        'mask' : mask
                    }
                    files.append( file_set )

        return files

    def __len__(self):
        return len(self.files)

    def open_as_array(self, idx, invert=True ):
        raw_stack = np.stack([
            np.array( standardize_image( Image.open(self.files[idx]['refl'] ) ) ),
            np.array( standardize_image( Image.open(self.files[idx]['trans']) ) ),
            np.array( standardize_image( Image.open(self.files[idx]['trans']) ) )   #fool it into rgb?!
        ], axis=2)

        #invert
        if invert:
            raw_stack = raw_stack.transpose((2,0,1))

        # normalize
        return (raw_stack / np.iinfo(raw_stack.dtype).max)

    def get_name(self, idx):
        names = {
            'trans': self.files[idx]['trans'].name,
            'refl' : self.files[idx]['refl'].name,
            'label': self.files[idx]['mask'].name,
        }
        return names

    def open_mask(self, idx):
        raw_mask = np.array( standardize_image( Image.open(self.files[idx]['mask'] ) ) )
        return raw_mask

    def __getitem__(self, idx):
        x = torch.tensor(self.open_as_array(idx), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx), dtype=torch.float32)
        return x, y

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s

    def show_labels(self):
        lnames = [ f['mask'].name for f in self.files ]
        return lnames

X_fp = Path(base_path)
data = GrainDataset(X_fp/'inputs', X_fp/'labels', False)
data_len = len( data )
print( 'Length of use data', len(data) )
#print( data.show_labels() )

hold_dt = GrainDataset(X_fp/'inputs', X_fp/'labels', True)
print( 'Length of held back data', len(hold_dt) )
#print( hold_ds.show_labels() )

train_size = round(data_len*TRAIN_SPLIT/100); test_size = round(data_len*(TEST_SPLIT)/100)
train_ds, valid_ds = torch.utils.data.random_split(data, (train_size, test_size))
hold_ds, sorry_ds = torch.utils.data.random_split(hold_dt, (len(hold_dt)-2, 2))
print('Training size:', len(train_ds), 'Test size:', len(valid_ds), 'Hold out size', len(hold_ds) )

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
hold_dl  = DataLoader(hold_ds,  batch_size=BATCH_SIZE, shuffle=False)
print( 'Training batches:', len(train_dl), 'Validation batches:', len(valid_dl), 'Hold Out batches', len(hold_dl))

#homemade unet
from torch import nn
class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(128, 32, 3, 1)
        self.upconv1 = self.expand_block(64, out_channels, 3, 1)

    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        #print( 'conv', conv1.shape, conv2.shape, conv3.shape )
    
        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        #print( 'upconv', upconv3.shape, upconv2.shape, upconv1.shape )

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return expand

unet = UNET(3,1)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

loaders = {
    "train": train_dl,
    "valid": valid_dl
}

# model, criterion, optimizer
optimizer = torch.optim.Adam(unet.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
criterion = DiceLoss()
runner = SupervisedRunner()

torch.cuda.empty_cache()

#train
runner.train(
    model=unet,
    criterion=criterion,
    optimizer=optimizer,
    #scheduler=scheduler,
    loaders=loaders,
    #callbacks=[EarlyStoppingCallback(patience=10, loader_key="valid", metric_key="loss", minimize=True)],
    num_epochs=NUM_EPOCHS,
    logdir=logdir,
    verbose=True
)

j = 0
preds = []
idxs = list( hold_dl.dataset.indices )
for v1 in iter(hold_dl):
    xb, yb = v1    
    pred = runner.predict_batch( v1 )
    for i in range(BATCH_SIZE):
        try:
            pred_img = predb_to_mask(pred, i)
            idx = idxs[j+i]
            names = hold_dl.dataset.dataset.get_name(idx)
            predname = names['label'].replace('L', 'Predicted')
            pdict = {
               'trans': names['trans'], 'refl': names['refl'],
               'label': names['label'], 'pred': predname
            }
            preds.append( pdict )

            pred_img.save( results + predname )
        except:
            pass
    j += BATCH_SIZE

data = { 'results': preds }
with open( results+'result.json', 'w') as f:
   json.dump( data, f )

print( preds )
print('Result json:', results+'result.json')

#onnx export
features_batch = next(iter(loaders["valid"]))[0]
fname = "./results2ws2000/homeunet_epoch%s.onnx" % NUM_EPOCHS
onnx_export(model=runner.model.cpu(), batch=features_batch, file=fname, verbose=True)
print('Saved onnx model at', fname)
