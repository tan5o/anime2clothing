import os
from os import listdir
from os.path import join
import random
import numpy as np
from PIL import Image, ImageFile, ImageOps, ImageDraw, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as f
from util.util import is_image_file, load_img
import cv2

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction, size, is_train = True):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        self.is_train = is_train
        self.size = size

        self.transform = transforms.Compose([
                          transforms.Resize((int(size*1.1), int(size*1.1))),
                          transforms.RandomRotation(3),
                          transforms.RandomCrop(size),
                          transforms.ColorJitter(hue=.05, saturation=.05),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform_input = transforms.Compose([
            transforms.Resize((int(256 * 1.1), int(256 * 1.1))),
            transforms.RandomRotation(3),
            transforms.RandomCrop(256),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transform_no_rotate = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        self.test_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.test_transform256 =  transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')

        while True:
            other_index = random.randrange(0, self.__len__())
            other_b = Image.open(join(self.b_path, self.image_filenames[other_index])).convert('RGB')

            if other_b != b:
                break

        dataset=[0]*10

        dataset[8] = self.image_filenames[index]
        if self.is_train:
            dataset[2] = self.transform_input(a)
            dataset[1] = self.transform_no_rotate(b)

            dataset[6] = self.transform(other_b)
        else:
            dataset[0] = self.test_transform(a)
            dataset[1] = self.test_transform(b)
            dataset[2] = self.test_transform256(a)
            dataset[3] = self.test_transform256(b)

        return dataset#, a_pose256, a_mask256, a_crop

    def __len__(self):
        return len(self.image_filenames)



def PositionCenter(img, mirror_padding=True):
    img_w, img_h = img.size
    #pixel_color = img.getpixel((0,0))
    pixel_color = (0,0,0)
    background = Image.new('RGB', (512, 512), pixel_color)
    bg_w, bg_h = background.size
    offset = (int((bg_w - img_w) / 2), int((bg_h - img_h) / 2))
    background.paste(img, offset)

    if mirror_padding:
        flip_img = ImageOps.mirror(img).resize((img.size[0]*3, img.size[1]))
        offset_mirror = (int((bg_w - img_w*7) / 2), int((bg_h - img_h) / 2))
        background.paste(flip_img, offset_mirror)
        offset_mirror = (int((bg_w - img_w) / 2) + img_w, int((bg_h - img_h) / 2))
        background.paste(flip_img, offset_mirror)

        mask = Image.new("L", background.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((offset[0] + 10, offset[1], offset[0] + img.size[0] - 10, offset[1] + img.size[1]), fill=255)
        mask_blur = mask.filter(ImageFilter.GaussianBlur(25))

        background_blur = background.filter(ImageFilter.GaussianBlur(100))

        background = Image.composite(background, background_blur, mask_blur)
    return background

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image#cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

if __name__ == '__main__':
    im = Image.open("dataset/cosplay/train/b/8.png")
    posi = PositionCenter(im, True)
    posi.save("test.png")