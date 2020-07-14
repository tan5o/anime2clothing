<br><br>

# Anime2Clothing
Pytorch official implementation of **Anime to Real Clothing: Cosplay Costume Generation via Image-to-Image Translation**.

<img src='imgs/purpose_of_paper.png' width="400px"/>

## Prerequisites
- Anaconda 3
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Training
`python train.py --project_name cosplay_synthesis  --dataset DATASET --checkpoints_dir "checkpoints" --g_conv normal --g_norm batch --d_conv normal --d_norm spectral --netG unet`

#### Continue Training 
Add continue_train otpion, and you can control starting epoch and resolution.

`--continue_train --start_epoch 0  --start_resolution 256`

### Testing


### Pre-trained model
In the future, we will release the pre-trained model when the copyright problem is cleared.

## Acknowledgments
Our code is inspired by [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
