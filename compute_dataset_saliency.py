"""计算BreakHis和ChestCT数据集显著图的简化脚本"""

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
        image: 输入图像 (numpy array, HxWxC)
    
    Returns:
        saliency_map: 显著图 (H, W)，数值范围[0, 1]
    """
    # 确保图像是3通道的BGR格式（OpenCV要求）
    if len(image.shape) == 2:
        # 灰度图转RGB
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        # 单通道转RGB
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        # RGB转BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.shape[2] == 4:
        # RGBA转RGB再转BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"不支持的图像通道数: {image.shape[2]}")
    
    # 使用OpenCV的谱残差显著性检测
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image_bgr)
    
    if not success:
        print("警告: 谱残差显著性计算失败，返回零矩阵")
        return np.zeros(image.shape[:2], dtype=np.float32)
    
    # 确保输出是float32类型并归一化到[0, 1]
    saliency_map = saliency_map.astype(np.float32)
    if saliency_map.max() > 0:
        saliency_map = saliency_map / saliency_map.max()
    
    return saliency_map


def process_single_dataset(data_loader, dataset_name, output_dir):
    """
    处理单个数据集的所有图像
    
    Args:
        data_loader: 数据加载器
        dataset_name: 数据集名称（如 'breakhis_train'）
        output_dir: 输出目录
    """
    print(f"\n开始处理 {dataset_name} 数据集...")
    
    # 创建输出目录
    saliency_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(saliency_dir, exist_ok=True)
    
    # 获取基础数据集（处理Subset的情况）
    base_dataset = data_loader.dataset
    while hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset
    
    classes = base_dataset.classes
    print(f"类别列表: {classes}")
    print(f"总样本数: {len(data_loader.dataset)}")
    
    # 统计各类别样本数
    class_counter = {cls: 0 for cls in classes}
    processed_count = 0
    
    # 处理每张图像
    for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc=f"Processing {dataset_name}")):
        # 获取单张图像和标签
        image_tensor = images[0]  # shape: (C, H, W)
        label = labels[0].item()
        class_name = classes[label]
        
        # 转换tensor到numpy
        if isinstance(image_tensor, torch.Tensor):
            # 转换为(H, W, C)格式的numpy数组
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            
            # 处理归一化情况
            if image_np.max() <= 1.0 and image_np.min() >= 0:
                # [0,1]范围的归一化数据
                image_np = (image_np * 255).astype(np.uint8)
            elif image_np.max() <= 255 and image_np.min() >= 0:
                # [0,255]范围的原始数据
                image_np = image_np.astype(np.uint8)
            else:
                # 其他情况，强制转换到[0,255]
                image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype(np.uint8)
        else:
            image_np = np.array(image_tensor)
        
        try:
            # 计算显著图
            saliency_map = compute_spectral_residual_saliency(image_np)
            
            # 生成文件名
            filename = f"saliency_{processed_count:06d}_label_{label}_{class_name}.npy"
            filepath = os.path.join(saliency_dir, filename)
            
            # 保存为npy文件
            np.save(filepath, saliency_map)
            
            # 更新统计
            class_counter[class_name] += 1
            processed_count += 1
            
        except Exception as e:
            print(f"处理第 {processed_count} 张图像时出错: {e}")
            continue
    
    # 打印处理结果
    print(f"\n{dataset_name} 处理完成:")
    print(f"成功处理样本数: {processed_count}")
    print("各类别统计:")
    for class_name, count in class_counter.items():
        print(f"  {class_name}: {count} 张")
    print(f"显著图保存位置: {saliency_dir}")


def main():
    parser = argparse.ArgumentParser(description='计算BreakHis和ChestCT数据集的谱残差显著图')
    parser.add_argument('--breakhis_path', type=str, default='../data/breakhis',
                        help='BreakHis数据集路径')
    parser.add_argument('--chestct_path', type=str, default='../data/chestct',
                        help='ChestCT数据集路径')
    parser.add_argument('--output_dir', type=str, default='./saliency_maps',
                        help='显著图输出目录')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小（建议为1）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数')
    parser.add_argument('--magnification', type=str, default='40X',
                        choices=['40X', '100X', '200X', '400X'],
                        help='BreakHis数据集放大倍数')
    
    args = parser.parse_args()
    
    print("=== BreakHis和ChestCT数据集显著图计算 ===")
    print(f"BreakHis路径: {args.breakhis_path}")
    print(f"ChestCT路径: {args.chestct_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"放大倍数: {args.magnification}")
    
    start_time = time.time()
    
    # 处理BreakHis数据集
    if os.path.exists(args.breakhis_path):
        print(f"\n{'='*60}")
        print("处理 BreakHis 数据集")
        print(f"{'='*60}")
        
        try:
            # 获取BreakHis数据加载器
            train_loader, test_loader = get_dataloaders(
                data_dir=args.breakhis_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dataset_type='breakhis',
                magnification=args.magnification,
                resize=True
            )
            
            # 处理训练集和测试集
            process_single_dataset(train_loader, 'breakhis_train', args.output_dir)
            process_single_dataset(test_loader, 'breakhis_test', args.output_dir)
            
        except Exception as e:
            print(f"处理BreakHis数据集时出错: {e}")
    else:
        print(f"警告: BreakHis数据路径 {args.breakhis_path} 不存在")
    
    # 处理ChestCT数据集
    if os.path.exists(args.chestct_path):
        print(f"\n{'='*60}")
        print("处理 ChestCT 数据集")
        print(f"{'='*60}")
        
        try:
            # 获取ChestCT数据加载器
            train_loader, test_loader = get_dataloaders(
                data_dir=args.chestct_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dataset_type='chestct',
                resize=True
            )
            
            # 处理训练集和测试集
            process_single_dataset(train_loader, 'chestct_train', args.output_dir)
            process_single_dataset(test_loader, 'chestct_test', args.output_dir)
            
        except Exception as e:
            print(f"处理ChestCT数据集时出错: {e}")
    else:
        print(f"警告: ChestCT数据路径 {args.chestct_path} 不存在")
    
    # 总结
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print("所有处理完成!")
    print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"结果保存在: {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()