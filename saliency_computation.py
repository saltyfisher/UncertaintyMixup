"""使用谱残差显著图算法计算数据集显著图的模块"""

import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import get_dataloaders, get_num_classes
import argparse
import time
from tqdm import tqdm


def compute_spectral_residual_saliency(image):
    """
    使用谱残差算法计算图像的显著图
    
    Args:
        image: 输入图像，可以是numpy数组或PIL图像
    
    Returns:
        saliency_map: 显著图 (H, W)，数值范围[0, 1]
    """
    # 如果输入是PIL图像，转换为numpy数组
    if hasattr(image, 'convert'):
        image = np.array(image)
    
    # 确保图像是3通道的
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # 使用OpenCV的谱残差显著性检测
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)
    
    if not success:
        # 如果计算失败，返回全零图
        return np.zeros(image.shape[:2], dtype=np.float32)
    
    # 确保输出是float32类型
    saliency_map = saliency_map.astype(np.float32)
    
    # 归一化到[0, 1]范围
    if saliency_map.max() > 0:
        saliency_map = saliency_map / saliency_map.max()
    
    return saliency_map


def process_dataset_saliency(data_loader, dataset_name, output_dir, batch_size=1):
    """
    处理整个数据集，计算每张图像的显著图并保存
    
    Args:
        data_loader: 数据加载器
        dataset_name: 数据集名称
        output_dir: 输出目录
        batch_size: 批次大小（建议保持为1以便逐张处理）
    """
    print(f"\n开始处理 {dataset_name} 数据集...")
    
    # 创建输出目录
    dataset_output_dir = os.path.join(output_dir, dataset_name, 'saliency_maps')
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # 获取类别信息
    dataset = data_loader.dataset
    if hasattr(dataset, 'dataset'):
        # 如果是Subset包装的数据集
        classes = dataset.dataset.classes
    else:
        classes = dataset.classes
    
    print(f"数据集类别: {classes}")
    print(f"总样本数: {len(dataset)}")
    
    # 统计信息
    total_samples = 0
    class_counts = {}
    
    # 处理每个批次
    for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc=f"Processing {dataset_name}")):
        # 确保批次大小为1
        if images.size(0) != 1:
            print(f"警告: 批次大小不是1，当前为 {images.size(0)}")
            continue
            
        # 获取图像和标签
        image = images[0]  # 取第一个样本
        label = labels[0].item()
        class_name = classes[label]
        
        # 更新统计信息
        total_samples += 1
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
        
        # 将tensor转换为numpy数组
        if isinstance(image, torch.Tensor):
            # 如果是归一化的tensor，需要反归一化
            image_np = image.permute(1, 2, 0).cpu().numpy()
            
            # 检查是否需要反归一化（根据数值范围判断）
            if image_np.min() >= 0 and image_np.max() <= 1:
                # 假设是[0,1]范围的归一化数据
                image_np = (image_np * 255).astype(np.uint8)
            else:
                # 假设是原始像素值
                image_np = image_np.astype(np.uint8)
        else:
            image_np = np.array(image)
        
        # 计算显著图
        try:
            saliency_map = compute_spectral_residual_saliency(image_np)
            
            # 生成文件名
            filename = f"saliency_{batch_idx:06d}_class_{label}_{class_name}.npy"
            filepath = os.path.join(dataset_output_dir, filename)
            
            # 保存显著图为npy文件
            np.save(filepath, saliency_map)
            
        except Exception as e:
            print(f"处理图像 {batch_idx} 时出错: {e}")
            continue
    
    # 打印统计信息
    print(f"\n{dataset_name} 数据集处理完成:")
    print(f"总处理样本数: {total_samples}")
    print("各类别样本数:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    print(f"显著图保存位置: {dataset_output_dir}")


def main():
    parser = argparse.ArgumentParser(description='计算数据集的谱残差显著图')
    parser.add_argument('--data_root', type=str, default='../data',
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./saliency_results',
                        help='显著图输出目录')
    parser.add_argument('--datasets', nargs='+', default=['breakhis', 'chestct'],
                        choices=['breakhis', 'chestct', 'padufes'],
                        help='要处理的数据集列表')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小（建议保持为1）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数')
    parser.add_argument('--magnification', type=str, default=None,
                        choices=['40X', '100X', '200X', '400X'],
                        help='BreakHis数据集的放大倍数')
    
    args = parser.parse_args()
    
    print("=== 谱残差显著图计算程序 ===")
    print(f"数据集根目录: {args.data_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"处理的数据集: {args.datasets}")
    print(f"批次大小: {args.batch_size}")
    print(f"工作进程数: {args.num_workers}")
    if args.magnification:
        print(f"BreakHis放大倍数: {args.magnification}")
    
    start_time = time.time()
    
    # 为每个数据集创建显著图
    for dataset_name in args.datasets:
        print(f"\n{'='*50}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*50}")
        
        # 设置数据集特定的路径
        if dataset_name == 'breakhis':
            data_dir = os.path.join(args.data_root, 'breakhis')
        elif dataset_name == 'chestct':
            data_dir = os.path.join(args.data_root, 'chestct')
        elif dataset_name == 'padufes':
            data_dir = os.path.join(args.data_root, 'padufes')
        else:
            data_dir = os.path.join(args.data_root, dataset_name)
        
        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            print(f"警告: 数据目录 {data_dir} 不存在，跳过此数据集")
            continue
        
        try:
            # 获取数据加载器
            train_loader, test_loader = get_dataloaders(
                data_dir=data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dataset_type=dataset_name,
                magnification=args.magnification,
                resize=True  # 确保图像尺寸一致
            )
            
            # 处理训练集
            print(f"\n处理 {dataset_name} 训练集...")
            process_dataset_saliency(
                train_loader, 
                f"{dataset_name}_train", 
                args.output_dir, 
                args.batch_size
            )
            
            # 处理测试集
            print(f"\n处理 {dataset_name} 测试集...")
            process_dataset_saliency(
                test_loader, 
                f"{dataset_name}_test", 
                args.output_dir, 
                args.batch_size
            )
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")
            continue
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*50}")
    print("所有数据集处理完成!")
    print(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    print(f"结果保存在: {os.path.abspath(args.output_dir)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()