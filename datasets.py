import cv2
import numpy as np
import torch
import random
import torchvision

from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset, Subset
from PIL import Image
import logging
import os
import random
import torchvision
import torch
import pandas as pd
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from operator import itemgetter
from pathlib import Path
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms.autoaugment import RandAugment, TrivialAugmentWide
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold

def get_num_classes(dataset):
    return {
        'lymphoma':3,
        'breakhis':8,
        'lc25000':5,
        'rect':2,
        'chestct':4,
        'EndoscopicBladder':4,
        'corona':7,
        'kvasir-dataset':8,
        'PAD-UFES-20':6
    }[dataset]

class Mydata(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if hasattr(self, 'groups'):
            if index in self.groups:
                sample = self.mfc_transform[0](sample)
            else:
                sample = self.transform(sample)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def get_all_files(self):
        data_list = [self.loader(path) for path, target in self.samples]
        label_list = self.targets
        return data_list, label_list
    
    def get_labels(self):
        return self.labels
    
    def update_transform(self, mfc_transform, transform, groups):
        self.mfc_transform = mfc_transform
        self.transform = transform
        self.groups = groups

class Mydatasubset(Subset):
    def __getitem__(self, index):
        path, target = self.dataset.samples[self.indices[index]]
        sample = self.dataset.loader(path)
        if hasattr(self, 'groups'):
            if index in self.groups:
                sample = self.mfc_transform[0](sample)
            else:
                sample = self.transform(sample)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def get_all_files(self):
        data_list = [self.dataset.loader(self.dataset.samples[i][0]) for i in self.indices]
        label_list = [self.dataset.targets[i] for i in self.indices]
        return data_list, label_list
    
    def get_labels(self):
        return [self.dataset.targets[i] for i in self.indices]
    
    def update_transform(self, mfc_transform, transform, groups):
        self.mfc_transform = mfc_transform
        self.transform = transform
        self.groups = groups

def split_train_val_dataset(traintest_dataset, train_transform, test_transform, validation_folds=5, random_state=42):
    """
    从训练集中划分验证集
    
    Args:
        traintest_dataset: 原始训练集
        train_transform: 训练集数据增强
        test_transform: 测试集/验证集数据变换
        validation_folds: 交叉验证折数，1表示简单划分，>1表示K折交叉验证
        random_state: 随机种子
        
    Returns:
        tuple: (traintest_dataset, trainval_datasets, val_datasets) 
               完整训练集，用于交叉验证的训练集列表和验证集列表
    """
    # 获取训练集标签
    traintest_labels = traintest_dataset.get_labels()
    
    skf = StratifiedKFold(n_splits=validation_folds, shuffle=True, random_state=random_state)
    trainval_indices = list(range(len(traintest_dataset)))
    trainval_splits = list(skf.split(trainval_indices, traintest_labels))
    
    # 返回多个(train, val)数据集对用于交叉验证
    trainval_datasets = []
    val_datasets = []
    
    for train_idx, val_idx in trainval_splits:
        train_subset = Mydatasubset(traintest_dataset, train_idx)
        val_subset = Mydatasubset(traintest_dataset, val_idx)
        
        train_subset.transform = train_transform
        val_subset.transform = test_transform
        
        trainval_datasets.append(train_subset)
        val_datasets.append(val_subset)
        
    return traintest_dataset, trainval_datasets, val_datasets

def get_dataloaders(data_dir, 
                    batch_size, 
                    dataset_type,
                    magnification,
                    test_split=0.2,
                    validation=False,
                    validation_folds=5,
                    random_state=42,
                    resize=True):
     
    if dataset_type == 'breakhis':
        resize_size = (448, 448)
    elif 'kvasir-dataset' in dataset_type:
        # Kvasir dataset usually contains images of size varying, resizing to 224 is common for classification
        resize_size = (224, 224)
    else:  # chestct and others
        resize_size = (224, 224)

    if resize:
        train_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ])
        
        test_transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.ToTensor(),
            ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    if 'EndoscopicBladder' in dataset_type:
        # 加载 annotations.csv 文件
        annotations_file = os.path.join(data_dir, 'annotations.csv')
        df = pd.read_csv(annotations_file)
        
        # 创建类别到索引的映射
        classes = sorted(df['tissue type'].unique())
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # 根据 annotation.csv 划分训练集、验证集和测试集
        train_df = df[df['sub_dataset'] == 'train']
        val_df = df[df['sub_dataset'] == 'val']
        test_df = df[df['sub_dataset'] == 'test']
        
        # 创建自定义数据集
        full_dataset = Mydata(root=data_dir, transform=train_transform)
        
        # 根据 annotations.csv 中的文件名创建索引映射
        def create_index_mapping(dataset_df, class_to_idx):
            """创建文件名到 (路径，标签) 的映射"""
            mapping = {}
            for _, row in dataset_df.iterrows():
                filename = row.iloc[0]  # 第一列是文件名
                tissue_type = row['tissue type']
                class_idx = class_to_idx[tissue_type]
                
                # 构建完整的文件路径
                file_path = os.path.join(data_dir, tissue_type, filename)
                mapping[file_path] = (file_path, class_idx)
            return mapping
        
        # 为每个数据集创建样本列表
        train_mapping = create_index_mapping(train_df, class_to_idx)
        val_mapping = create_index_mapping(val_df, class_to_idx)
        test_mapping = create_index_mapping(test_df, class_to_idx)
        
        # 创建自定义的 Dataset 类来支持基于 annotation.csv 的加载
        class AnnotationDataset(Dataset):
            def __init__(self, root, file_mapping, transform=None):
                self.root = root
                self.file_mapping = file_mapping
                self.samples = list(file_mapping.values())
                self.targets = [sample[1] for sample in self.samples]
                self.transform = transform
                self.loader = lambda path: Image.open(path).convert('RGB')
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, index):
                path, target = self.samples[index]
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample, target, path
            
            def get_labels(self):
                return self.targets
        
        traintest_dataset = AnnotationDataset(data_dir, train_mapping, train_transform)
        test_dataset = AnnotationDataset(data_dir, test_mapping, test_transform)
        val_dataset = AnnotationDataset(data_dir, val_mapping, test_transform)
        
        # 如果需要验证集，则使用 annotation.csv 中的 val 划分
        if validation:         
            return traintest_dataset, test_dataset, resize_size, train_transform, [traintest_dataset], [val_dataset]
        else:
            train_loader = DataLoader(traintest_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
            return train_loader, test_loader, resize_size, train_transform, val_loader

    if 'breakhis' in dataset_type:
        root_dir = os.path.join(data_dir, magnification)
        # 加载完整数据集
        full_dataset = Mydata(root=root_dir, transform=train_transform)
        
        # 获取标签
        labels = [sample[1] for sample in full_dataset.samples]

        # 使用 StratifiedShuffleSplit 进行分层抽样
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split)
        train_indices, test_indices = next(sss.split(range(len(full_dataset)), labels))

        # 创建训练集和测试集的子集
        traintest_dataset = Mydatasubset(full_dataset, train_indices)
        test_dataset = Mydatasubset(full_dataset, test_indices)

        traintest_dataset.transform = train_transform
        test_dataset.transform = test_transform
        
        # 如果需要验证集，则从训练集中进一步划分
        if validation:         
            trainval_datasets, val_datasets = split_train_val_dataset(
                traintest_dataset, train_transform, test_transform, validation_folds, random_state
            )
            return full_traintest_dataset, test_dataset, resize_size, train_transform, trainval_datasets, val_datasets

    if 'kvasir' in dataset_type:
        # 假设数据目录结构为 data_dir/kvasir-dataset/class_name/image.jpg
        root_dir = os.path.join(data_dir, 'kvasir-dataset')
        full_dataset = Mydata(root=root_dir, transform=train_transform)
        
        # 获取标签
        labels = [sample[1] for sample in full_dataset.samples]

        # 使用 StratifiedShuffleSplit 进行分层抽样
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
        train_indices, test_indices = next(sss.split(range(len(full_dataset)), labels))

        # 创建训练集和测试集的子集
        traintest_dataset = Mydatasubset(full_dataset, train_indices)
        test_dataset = Mydatasubset(full_dataset, test_indices)

        traintest_dataset.transform = train_transform
        test_dataset.transform = test_transform
        
        # 如果需要验证集，则从训练集中进一步划分
        if validation:         
            full_traintest_dataset, trainval_datasets, val_datasets = split_train_val_dataset(
                traintest_dataset, train_transform, test_transform, validation_folds, random_state
            )
            return full_traintest_dataset, test_dataset, resize_size, train_transform, trainval_datasets, val_datasets

    if 'chestct' in dataset_type:
        data, label = [], []
        all_folders = Path(data_dir).joinpath('train')
        traintest_dataset = Mydata(str(all_folders), transform=train_transform)
        all_folders = Path(data_dir).joinpath('test')
        test_dataset = Mydata(str(all_folders), transform=test_transform)
        
        # 如果需要验证集，则从训练集中进一步划分
        if validation:
            full_traintest_dataset, trainval_datasets, val_datasets = split_train_val_dataset(
                traintest_dataset, train_transform, test_transform, validation_folds, random_state
            )
            return full_traintest_dataset, test_dataset, resize_size, train_transform, trainval_datasets, val_datasets
    
    train_loader = DataLoader(traintest_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    return train_loader, test_loader
