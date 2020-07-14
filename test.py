from __future__ import print_function

import argparse
import os
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils

from generator.unet.unet_model import UNet


# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--input_dir', required=True, help='input directory')
parser.add_argument('--model_path', type=str, required=True, help='generator model path')
parser.add_argument('--output_dir', type=str, required=True, help='output directory')
opt = parser.parse_args()
print(opt)

device = "cuda"

if __name__ == '__main__':

    net_g = UNet(n_channels=3, n_classes=3, conv_type="normal",
                 norm_type="batch", is_acgan=False, is_msg=False).to(device)
    net_g.load_state_dict(torch.load(opt.model_path))
    resolution = 256
    net_g.set_config(resolution, "stable", 1.0)
    net_g.eval()

    dirs = os.listdir(opt.input_dir)
    print('found {} files'.format(len(dirs)))
    for i, f in enumerate(dirs):
        img = Image.open(os.path.join(opt.input_dir, f))

        im_trans = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        img = im_trans(img)
        img = img.unsqueeze(0)

        input = img.to(device)
        out = net_g(input)
        out = torch.cat((img, out.cpu()), 3)

        vutils.save_image(out, os.path.join(opt.output_dir, f), nrow=4, normalize=True,
                          padding=0)

        print('{}/{}'.format(i, len(dirs)))