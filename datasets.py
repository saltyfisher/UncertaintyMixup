"""数据集处理模块，包含各种数据集的加载和处理函数"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit


def get_dataloaders(data_dir, batch_size=16, num_workers=4, dataset_type='chestct', magnification=None, test_split=0.2, random_state=42):
    """
    获取训练和测试数据加载器
    
    Args:
        data_dir (str): 数据集根目录路径
        batch_size (int): 批次大小
        num_workers (int): 数据加载器的工作进程数
        dataset_type (str): 数据集类型 ('chestct' 或 'breakhis')
        magnification (str): BreakHis数据集的放大倍数 ('40X', '100X', '200X', '400X')，默认为None表示使用所有倍数
        test_split (float): 测试集比例，默认为0.2
        random_state (int): 随机种子，默认为42
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # 根据数据集类型设置resize大小
    if dataset_type == 'breakhis':
        resize_size = (450, 450)
    else:  # chestct
        # resize_size = (320, 320)
        resize_size = (224, 224)
    
    if dataset_type == 'chestct':
        # 训练数据只做resize和ToTensor操作
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ])
        
        # 测试数据也只做resize和ToTensor操作
        test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ])
        
        # 测试数据也只做resize和ToTensor操作
        test_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ])
    
    # 根据数据集类型加载数据
    if dataset_type == 'breakhis':
        train_loader, test_loader = _get_breakhis_dataloaders(
            data_dir, batch_size, num_workers, train_transform, test_transform, magnification, test_split, random_state
        )
    else:
        train_loader, test_loader = _get_chestct_dataloaders(
            data_dir, batch_size, num_workers, train_transform, test_transform
        )
    
    return train_loader, test_loader


def _get_breakhis_dataloaders(data_dir, batch_size, num_workers, train_transform, test_transform, magnification=None, test_split=0.2, random_state=42):
    """
    获取BreakHis数据集的数据加载器
    
    Args:
        data_dir (str): BreakHis数据集根目录路径
        batch_size (int): 批次大小
        num_workers (int): 数据加载器的工作进程数
        train_transform: 训练数据预处理
        test_transform: 测试数据预处理
        magnification (str): 放大倍数 ('40X', '100X', '200X', '400X')，默认为None表示使用所有倍数
        test_split (float): 测试集比例，默认为0.2
        random_state (int): 随机种子，默认为42
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # 新的BreakHis数据集结构: histology_slides/breast/[benign|malignant]/SOB/*/*/magnification
    # 需要重新组织数据集结构，创建训练和测试目录
    root_dir = os.path.join(data_dir, 'histology_slides', 'breast')
    
    # 定义类别映射
    classes = {
        'adenosis': os.path.join(root_dir, 'benign', 'SOB', 'adenosis'),
        'fibroadenoma': os.path.join(root_dir, 'benign', 'SOB', 'fibroadenoma'),
        'phyllodes_tumor': os.path.join(root_dir, 'benign', 'SOB', 'phyllodes_tumor'),
        'tubular_adenoma': os.path.join(root_dir, 'benign', 'SOB', 'tubular_adenoma'),
        'ductal_carcinoma': os.path.join(root_dir, 'malignant', 'SOB', 'ductal_carcinoma'),
        'lobular_carcinoma': os.path.join(root_dir, 'malignant', 'SOB', 'lobular_carcinoma'),
        'mucinous_carcinoma': os.path.join(root_dir, 'malignant', 'SOB', 'mucinous_carcinoma'),
        'papillary_carcinoma': os.path.join(root_dir, 'malignant', 'SOB', 'papillary_carcinoma')
    }
    
    # 创建临时目录结构用于ImageFolder
    import tempfile
    temp_dir = tempfile.mkdtemp()
    dataset_dir = os.path.join(temp_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 为每个类别创建符号链接
    for class_name, class_path in classes.items():
        class_dir = os.path.join(dataset_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # 遍历该类别下的所有子目录和放大倍数
        if os.path.exists(class_path):
            sub_dirs = os.listdir(class_path)
            for sub_dir in sub_dirs:
                sub_dir_path = os.path.join(class_path, sub_dir)
                if os.path.isdir(sub_dir_path):
                    if magnification is not None:
                        mag_dir = os.path.join(sub_dir_path, magnification)
                        if os.path.exists(mag_dir):
                            # 创建符号链接到训练目录（简化处理，实际应用中可能需要更复杂的训练/测试分割）
                            try:
                                os.symlink(mag_dir, os.path.join(class_dir, f"{sub_dir}_{magnification}"))
                            except FileExistsError:
                                pass
                    else:
                        # 链接所有放大倍数
                        for mag in ['40X', '100X', '200X', '400X']:
                            mag_dir = os.path.join(sub_dir_path, mag)
                            if os.path.exists(mag_dir):
                                try:
                                    os.symlink(mag_dir, os.path.join(class_dir, f"{sub_dir}_{mag}"))
                                except FileExistsError:
                                    pass
    
    # 加载完整数据集
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transform)
    
    # 获取标签
    labels = [sample[1] for sample in full_dataset.samples]
    
    # 使用StratifiedShuffleSplit进行分层抽样
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
    train_indices, test_indices = next(sss.split(range(len(full_dataset)), labels))
    
    # 创建训练集和测试集的子集
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 为测试集设置不同的transform
    # 注意：Subset不直接支持transform，我们需要为子集中的每个样本手动应用测试transform
    # 这里我们创建一个包装类来处理不同的transform
    class TransformSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            # 检查x是否已经是Tensor，如果是则不需要再应用transform
            if self.transform and not isinstance(x, torch.Tensor):
                x = self.transform(x)
            elif self.transform and isinstance(x, torch.Tensor):
                # 如果x已经是Tensor，我们需要先将其转换回PIL Image再应用transform
                from torchvision.transforms import ToPILImage
                to_pil = ToPILImage()
                x = to_pil(x)
                x = self.transform(x)
            return x, y
        
        def __len__(self):
            return len(self.subset)
    
    # 应用不同的transform
    train_dataset = TransformSubset(train_dataset, train_transform)
    test_dataset = TransformSubset(test_dataset, test_transform)
    
    print(f"BreakHis dataset loaded with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    print(f"Classes: {full_dataset.classes}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def _get_chestct_dataloaders(data_dir, batch_size, num_workers, train_transform, test_transform):
    """
    获取ChestCT数据集的数据加载器
    
    Args:
        data_dir (str): ChestCT数据集根目录路径
        batch_size (int): 批次大小
        num_workers (int): 数据加载器的工作进程数
        train_transform: 训练数据预处理
        test_transform: 测试数据预处理
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # 原有的数据集加载方式
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def get_num_classes(dataset_type):
    """
    获取数据集的类别数
    
    Args:
        dataset_type (str): 数据集类型 ('chestct' 或 'breakhis')
    
    Returns:
        int: 类别数
    """
    if dataset_type == 'breakhis':
        # BreakHis有8个类别
        return 8
    else:
        # ChestCT有4个类别
        return 4