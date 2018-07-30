#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_loader.py
# @Author: Jehovah
# @Date  : 18-7-27
# @Desc  :

'''
load data by image list.txt
'''


import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import cv2
import numpy as np
from skimage.util import random_noise

IMG_EXTEND = ['.jpg', '.JPG', '.jpeg', '.JPEG',
              '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
              ]


def is_img_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTEND)


def make_dataset(dir, file,is_content=True):
    images = []

    file = os.path.join(dir,file)
    fimg = open(file, 'r')
    for line in fimg:
        line = line.strip('\n')
        line = line.rstrip()
        word = line.split("||")
        if is_content:
            images.append(os.path.join(dir, word[0]))
        else:
            images.append(os.path.join(dir, word[1]))

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(data.Dataset):
    def __init__(self, dataroot, datalist, is_content=True, is_train=True, loadSize=286, fineSize=256, transform=None, return_paths=None, loader=default_loader):
        super(MyDataset, self).__init__()

        img = make_dataset(dataroot, datalist,is_content)
        if len(img) == 0:
            raise (RuntimeError("Found 0 images in: " + dataroot + dir + "\n"
                                                                         "Supported image extensions are: " +
                                ",".join(IMG_EXTEND)))

        print len(img)
        self.imgs = img
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.is_train = is_train
        self.loadSize = loadSize
        self.fineSize = fineSize

    def __getitem__(self, index):

        path = self.imgs[index]
        img = Image.open(path).convert('RGB')

        # img = transforms.Resize(size=(512, 512))(img)
        # img = transforms.RandomCrop(256)(img)
        # img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(imgA)
        shape = img.size
        if self.is_train:
            # img = Image.fromarray(np.uint8(img))
            # n = torch.nn.ZeroPad2d(((self.loadSize-shape[2]) / 2,(self.loadSize-shape[2]) / 2,(self.loadSize-shape[1]) / 2,(self.loadSize-shape[1]) / 2))
            # img = n(img)
            img = transforms.RandomCrop(256, padding=((self.loadSize-shape[0]) / 2, (self.loadSize-shape[1]) / 2))(img)
            # img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        else:
            n = torch.nn.ZeroPad2d(((self.loadSize - shape[2]) / 2, (self.loadSize - shape[2]) / 2,
                                    (self.loadSize - shape[1]) / 2, (self.loadSize - shape[1]) / 2))
            img = n(img)
        img = transforms.ToTensor()(img)
        return img

    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':
    pass