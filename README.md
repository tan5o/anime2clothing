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
`python train.py --project_name cosplay_synthesis  --dataset DATASET --checkpoints_dir checkpoints --g_conv normal --g_norm batch --d_conv normal --d_norm spectral --netG unet`

#### Training dataset structure
```buildoutcfg
DATASET
├── train
│    ├── a
│    │   ├── 0.png
│    │   ├── 1.png
│    │      ︙
│    |   └── n.png
│    └── b
│        ├── 0.png
│        ├── 1.png
│           ︙
│        └── n.png
└── test
     ├── a
     │   ├── 0.png
     │   ├── 1.png
     │      ︙
     |   └── n.png
     └── b
         ├── 0.png
         ├── 1.png
            ︙
         └── n.png
```

#### Continue Training 
Add continue_train otpion, and you can control starting epoch and resolution.
Basically, model load from latest checkpoints. However, you can chose number of epoch if you use `--load_epoch` option.

`--continue_train --start_epoch 47  --start_resolution 256`

### Testing
`python test.py --model_path models/pretrained_model.pth --input_dir dataset/test/a --output_dir result`

### Pre-trained model
In the future, we will release the pre-trained model when the copyright problem is cleared.

## Acknowledgments
Our code is inspired by [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
