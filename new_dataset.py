import cv2
import numpy as np
import torch
import random
import torchvision

from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset
from PIL import Image

class Mydata(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if hasattr(self, 'groups'):
            if index in self.groups:
                sample = self.mfc_transform[0](sample)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def get_all_files(self):
        img_name = self.full_filenames[0]
        data_list = [Image.open(self.full_filenames[idx]).convert('RGB') for idx in range(len(self.full_filenames))]
        label_list = [self.labels[idx] for idx in range(len(self.full_filenames))]
        return data_list, label_list
    
    def get_labels(self):
        return self.labels
    
    def update_transform(self, mfc_transform, transform, groups):
        self.mfc_transform = mfc_transform
        self.transform = transform
        self.groups = groups

# class Mydata(Dataset):
#     def __init__(self, file_list, labels, transform = None, mfc=False):
#         self.full_filenames = file_list
#         self.labels = labels
#         self.transform = transform
#         self.mfc = False
#         self.groups = []
    
#     def __len__(self):
#         return len(self.full_filenames)
    
#     def __getitem__(self, idx):
#         img_name = self.full_filenames[idx]
#         with open(img_name, 'rb') as f:
#             image = Image.open(f).convert('RGB')
#         # print(idx)
#         if idx in self.groups:
#             image = self.mfc_transform[0](image)
#         else:
#             image = self.transform(image)
#         label = self.labels[idx]
#         return image, label

#     def get_all_files(self):
#         img_name = self.full_filenames[0]
#         data_list = [Image.open(self.full_filenames[idx]).convert('RGB') for idx in range(len(self.full_filenames))]
#         label_list = [self.labels[idx] for idx in range(len(self.full_filenames))]
#         return data_list, label_list
    
#     def get_labels(self):
#         return self.labels
    
#     def update_transform(self, mfc_transform, transform, groups):
#         self.mfc_transform = mfc_transform
#         self.transform = transform
#         self.groups = groups


import logging
import os
import random
import torchvision
import torch
import pandas as pd
import numpy as np

from torchvision import transforms, datasets
from operator import itemgetter
from pathlib import Path
from PIL import Image
from torchvision.transforms import transforms
from augmentations import MyRandAugment, MyTrivialAugmentWide
from torchvision.transforms.autoaugment import RandAugment, TrivialAugmentWide
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

def get_data(strategy,dataset,magnification,dataroot,validation=False):
     
    if dataset == 'breakhis':
        resize_size = (448, 448)
    else:  # chestct
        resize_size = (224, 224)

    train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
    ])
    
    test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    if dataset == 'chestct':
        test_transform = transforms.Compose([
            transforms.Resize(resize_size),
            test_transform,
        ])
    
    if strategy == 'randaugment':
        train_transform = transforms.Compose([
            RandAugment(),
            train_transform,
        ])

    if strategy == 'trivialaugment':
        train_transform = transforms.Compose([
            TrivialAugmentWide(),
            train_transform,
        ])

    if strategy == 'randaugment_raw':
        train_transform = transforms.Compose([
            MyRandAugment(),
            train_transform,
        ])

    if strategy == 'trivialaugment_raw':
        train_transform = transforms.Compose([
            MyTrivialAugmentWide(),
            train_transform,
        ])
    
    # if dataset == 'chestct':
    #     train_transform = transforms.Compose([
    #         transforms.Grayscale(num_output_channels=3),
    #         train_transform,
    #     ])
    #     test_transform = transforms.Compose([
    #         transforms.Grayscale(num_output_channels=3),
    #         test_transform,
    #     ])
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.ppm', '.pgm'}
    label, data = [], []
    if 'breakhis' in dataset:
        all_folders = Path(dataroot).joinpath('BreakHis','histology_slides','breast')
        label_type = ['adenosis','fibroadenoma','phyllodes_tumor','tubular_adenoma','ductal_carcinoma','lobular_carcinoma','mucinous_carcinoma','papillary_carcinoma']
        for obj in all_folders.rglob('*'): 
            if obj.is_file() and obj.suffix.lower() in supported_extensions: 
                if magnification in str(obj.parent):
                    data.append(str(obj))
                    for i, label_name in enumerate(label_type):
                        if label_name in str(obj.parent):
                            label.append(i)
    if data != []:
        sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2)
        traintest_idx, test_idx = next(sss.split(data, label))
        getter = itemgetter(*test_idx)
        test_dataset = Mydata(getter(data), getter(label), test_transform)
        getter = itemgetter(*traintest_idx)
        traintest_dataset = Mydata(getter(data), getter(label), train_transform)

    if 'chestct' in dataset:
        data, label = [], []
        all_folders = Path(dataroot).joinpath('chest-ctscan-images_datasets','train')
        traintest_dataset = datasets.ImageFolder(str(all_folders), transform=train_transform)
        all_folders = Path(dataroot).joinpath('chest-ctscan-images_datasets','test')
        test_dataset = datasets.ImageFolder(str(all_folders), transform=test_transform)
    
    return traintest_dataset,test_dataset,resize_size,train_transform











