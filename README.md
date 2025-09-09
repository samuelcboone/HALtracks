# HALtracks
A repository for the HALtracks 2D and HALtracks 3D automated digital fission-track convoluted neural networks of Boone et al. (2025).


About:

We introduce two deep-learning tools designed to detect and count fission tracks in apatite and mica automatically. The first, HALtracks 2D, analyses paired reflected- and transmitted-light images of polished grain surfaces. The second, HALtracks 3D, combines reflected images with a transmitted-light z-stack acquired at 0.5 μm spacing.

Both methods employ convolutional neural networks adapted from the U-Net framework originally described by Ronneberger et al. (2015). The HALtracks assigns each pixel to either a “track opening” or “non-track” class. Track geometries and counts are then obtained through the FastTracks software (Gleadow et al., 2009), which incorporates conventional particle analysis routines. Features inconsistent with track morphologies based on size, shape, or aspect ratio are automatically discarded through threshold filters, and overlapping openings are distinguished and tallied.

Training of HALtracks 2D and 3D drew upon a very large dataset (~2.5 terabytes) of 6,526 image sets of fission tracks in apatite. Due to its very large size, the training and testing datasets are not distributed within this repository; interested users should contact the authors directly.

Further technical details on model structure, optimisation, and evaluation are given in Boone et al. (2025). The trained HALtracks networks are provided as ONNX-formatted models, a standard exchange framework for neural networks that ensures compatibility across platforms and programming languages (e.g., Python, C++, Java). Both algorithms are now fully integrated within FastTracks V4, enabling automated fission-track analysis within a widely used digital platform.
