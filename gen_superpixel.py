import numpy as np
import cv2
import os
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from datasets import get_dataloaders
from tqdm import tqdm
import torch
import time
from skimage.filters import sobel
from concurrent.futures import ThreadPoolExecutor
import threading

def ct_feat(gray, grad_weight=0.5):
    I = gray.astype(np.float32)
    I = (I - I.min()) / (I.max() - I.min() + 1e-8)

    gx = sobel(I, axis=0)
    gy = sobel(I, axis=1)
    G = np.sqrt(gx * gx + gy * gy)
    G = (G - G.min()) / (G.max() - G.min() + 1e-8)
    feat = np.concatenate([I, grad_weight * G], axis=-1)  # HxWx2

    return feat

def he_feat(rgb):
    rgb = rgb.astype(np.float32) + 1.0
    od = -np.log(rgb / 255.0)

    HE = np.array([
        [0.650, 0.704, 0.286],
        [0.072, 0.990, 0.105],
        [0.268, 0.570, 0.776]
    ], dtype=np.float32)
    HE /= np.linalg.norm(HE, axis=1, keepdims=True)

    stains = od.reshape(-1, 3) @ np.linalg.inv(HE).T
    stains = stains.reshape((*rgb.shape[:2], 3))

    H = stains[..., 0]
    E = stains[..., 1]
    feat = np.stack([H, E], axis=-1)
    return feat

def _process_dataloader(dataloader, output_dir, dataset_type, magnification, n_segments=100, compactness=10, use_feat=False, max_workers=4):
    """
    处理数据加载器中的所有图像
    
    Args:
        dataloader: 数据加载器
        output_dir (str): 输出目录
        dataset_type (str): 数据集类型
        max_workers (int): 最大工作线程数
    """
    # 获取类别名称
    if hasattr(dataloader.dataset, 'dataset'):
        # 如果是Subset包装的数据集
        classes = dataloader.dataset.dataset.classes
    else:
        classes = dataloader.dataset.classes
    
    # 为每个类别创建目录
    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # for batch_idx, (images, labels, paths) in enumerate(tqdm(dataloader, desc=f"Processing {os.path.basename(output_dir)}")):
    #     _process_batch(images, labels, paths, classes, output_dir, dataset_type, magnification, n_segments, compactness, use_feat)
    # 使用ThreadPoolExecutor进行多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch_idx, (images, labels, paths) in enumerate(tqdm(dataloader, desc=f"Processing {os.path.basename(output_dir)}")):
            future = executor.submit(_process_batch, images, labels, paths, classes, output_dir, dataset_type, magnification, n_segments, compactness, use_feat)
            futures.append(future)
        
        # 等待所有任务完成
        for future in tqdm(futures, desc="Waiting for tasks to complete"):
            future.result()

def _process_batch(images, labels, paths, classes, output_dir, dataset_type, magnification, n_segments, compactness, use_feat):
    """处理一个批次的图像"""
    if isinstance(images, torch.Tensor):
        images = images.numpy()
    
    for i in range(images.shape[0]):
        image = np.transpose(images[i], (1, 2, 0))
        
        if use_feat:
            if dataset_type == 'chestct':
                image_feat = ct_feat(image)
                compactness = 0.1
            elif dataset_type == 'breakhis':
                image_feat = he_feat(image)
                if magnification == '40':
                    compactness = 0.01
                elif magnification == '100':
                    compactness = 0.01
                elif magnification == '200':
                    compactness = 0.01
                elif magnification == '400':
                    compactness = 0.01
            sigma = 0

        else:
            sigma = 1
        if use_feat:
            image_float = img_as_float(image_feat)
        else:
            image_float = img_as_float(image)
        
        segments = slic(image_float, n_segments=n_segments, compactness=compactness, sigma=sigma, channel_axis=-1)

        class_name = classes[labels[i].item()]
        filename = os.path.basename(paths[i])
        output_dir = output_dir + f"_{compactness}"
        output_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        image = (image * 255).astype(np.uint8)
        segmented_image = mark_boundaries(image, segments, color=(1, 0, 0))
        segmented_image = (segmented_image * 255).astype(np.uint8)
        cv2.imwrite(filepath, segmented_image)

def visualize_saliency(original_image, saliency_map, save_path=None):
    """
    可视化原始图像和显著图
    
    Args:
        original_image (numpy.ndarray): 原始图像
        saliency_map (numpy.ndarray): 显著图
        save_path (str): 保存路径，如果为None则显示图像
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示原始图像
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 显示显著图
    im = axes[1].imshow(saliency_map, cmap='jet')
    axes[1].set_title('Saliency Map')
    axes[1].axis('off')
    
    # 添加颜色条
    plt.colorbar(im, ax=axes[1])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def get_slic_superpixels(data_dir, output_dir, dataset_type='chestct', batch_size=1, num_workers=4, magnification=None, n_segments=100, compactness=10, use_feat=False):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数据加载器
    train_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        dataset_type=dataset_type,
        magnification=magnification
    )

    # 处理训练集
    _process_dataloader(train_loader, output_dir, dataset_type, magnification, n_segments, compactness, use_feat, num_workers)
    
    # 处理测试集
    _process_dataloader(test_loader, output_dir, dataset_type, magnification, n_segments, compactness, use_feat, num_workers)

if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate saliency maps using spectral residual method')
    parser.add_argument('--data_dir', type=str, default='/workspace/MedicalImageClassficationData/', help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, help='Path to save saliency maps')
    parser.add_argument('--dataset_type', type=str, default='chestct', 
                       choices=['chestct', 'breakhis', 'padufes'], 
                       help='Dataset type')
    parser.add_argument('--magnification', type=str, default=None,
                       choices=['40', '100', '200', '400'],
                       help='Magnification for BreakHis dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--feat', action='store_true')
    
    args = parser.parse_args()
    args.output_dir = f'./vis_results/superpixel/{args.dataset_type}'
    if args.dataset_type == 'breakhis':
        args.output_dir += f'_{args.magnification}'
    if args.feat:
        args.output_dir += '_feat'

    if args.dataset_type == 'breakhis':
        args.data_dir = '/workspace/MedicalImageClassficationData/BreakHis'
    elif args.dataset_type == 'chestct':
        args.data_dir = '/workspace/MedicalImageClassficationData/chest-ctscan-images_datasets'
    
    get_slic_superpixels(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        magnification=args.magnification,
        use_feat=args.feat
    )
