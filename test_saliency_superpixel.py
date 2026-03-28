#!/usr/bin/env python3
"""
测试谱残差显著图和基于显著值的超像素采样功能
"""

import torch
import numpy as np
import cv2
from utils import compute_spectral_residual_saliency, matting_cutmix_data
import matplotlib.pyplot as plt


def create_test_image():
    """创建测试图像"""
    # 创建一个简单的测试图像
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # 添加一些显著区域（亮色块）
    img[50:100, 50:100] = [255, 0, 0]  # 红色方块
    img[150:200, 150:200] = [0, 255, 0]  # 绿色方块
    img[100:150, 100:150] = [0, 0, 255]  # 蓝色方块
    
    # 添加背景噪声
    noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
    img = np.clip(img + noise, 0, 255)
    
    return img


def test_spectral_saliency():
    """测试谱残差显著图计算"""
    print("=== 测试谱残差显著图计算 ===")
    
    # 创建测试图像
    test_img = create_test_image()
    print(f"测试图像形状: {test_img.shape}")
    
    # 计算显著图
    saliency_map = compute_spectral_residual_saliency(test_img)
    print(f"显著图形状: {saliency_map.shape}")
    print(f"显著图数值范围: [{saliency_map.min():.4f}, {saliency_map.max():.4f}]")
    
    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(test_img)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    im = axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title('谱残差显著图')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('test_saliency_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("显著图测试完成，结果保存为 test_saliency_result.png")


def test_superpixel_weighted_sampling():
    """测试基于显著值的超像素采样"""
    print("\n=== 测试基于显著值的超像素采样 ===")
    
    # 创建批量测试数据
    batch_size = 4
    channels = 3
    height, width = 224, 224
    
    # 创建测试图像批次
    x = torch.randn(batch_size, channels, height, width)
    # 将其中一张图像替换为我们的测试图像
    test_img = create_test_image()
    x[0] = torch.from_numpy(test_img.transpose(2, 0, 1)).float() / 255.0
    
    # 创建标签
    y = torch.randint(0, 10, (batch_size,))
    
    print(f"输入张量形状: {x.shape}")
    print(f"标签: {y}")
    
    # 模拟一个简单的matting方法
    def dummy_matting_method(input_tensor):
        batch_size = input_tensor.shape[0]
        return torch.ones(batch_size, 64, 64) * 0.5
    
    try:
        # 测试修改后的函数
        mixed_x, out_lam, mixed_pairs = matting_cutmix_data(
            x=x,
            y=y,
            batch_idx=0,
            superpixel_nums=50,
            alpha=1.0,
            device='cpu',
            matting_method=dummy_matting_method,
            superpixel=True,
            alphalabel=True
        )
        
        print(f"混合后图像形状: {mixed_x.shape}")
        print(f"混合比例: {out_lam}")
        print(f"混合对数量: {len(mixed_pairs)}")
        
        # 检查混合对信息
        for i, pair in enumerate(mixed_pairs):
            if pair is not None:
                print(f"混合对 {i}: 源索引={pair['source_idx']}, "
                      f"目标索引={pair['target_idx']}, "
                      f"混合比例={pair['ratio']:.4f}")
        
        print("超像素加权采样测试成功!")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def analyze_superpixel_weights():
    """分析超像素权重分布"""
    print("\n=== 分析超像素权重分布 ===")
    
    # 创建测试图像
    test_img = create_test_image()
    test_img_float = test_img.astype(np.float32) / 255.0
    
    # 计算显著图
    saliency_map = compute_spectral_residual_saliency(test_img)
    
    # 生成超像素分割
    h, w, c = test_img.shape
    cluster = cv2.ximgproc.createSuperpixelSEEDS(w, h, c, 50, num_levels=4)
    cluster.iterate(test_img_float, 4)
    segments = cluster.getLabels()
    
    # 计算每个超像素的权重
    unique_labels = np.unique(segments)
    weights = []
    
    for label in unique_labels:
        mask = segments == label
        weight = np.sum(saliency_map[mask])
        weights.append(weight)
        pixel_count = np.sum(mask)
        print(f"超像素 {label}: 像素数={pixel_count}, 显著值之和={weight:.6f}")
    
    weights = np.array(weights)
    if np.sum(weights) > 0:
        normalized_weights = weights / np.sum(weights)
        print(f"\n权重统计:")
        print(f"最小权重: {np.min(normalized_weights):.6f}")
        print(f"最大权重: {np.max(normalized_weights):.6f}")
        print(f"平均权重: {np.mean(normalized_weights):.6f}")
        print(f"权重标准差: {np.std(normalized_weights):.6f}")
    
    # 可视化超像素分割和权重
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(test_img)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示超像素分割
    segments_vis = segments.copy()
    # 为每个超像素分配不同颜色
    for i, label in enumerate(unique_labels):
        segments_vis[segments == label] = i
    axes[1].imshow(segments_vis, cmap='tab20')
    axes[1].set_title('超像素分割')
    axes[1].axis('off')
    
    # 显示显著图
    im = axes[2].imshow(saliency_map, cmap='hot')
    axes[2].set_title('显著图')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('superpixel_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("超像素权重分析完成，结果保存为 superpixel_analysis.png")


if __name__ == "__main__":
    print("开始测试谱残差显著图和超像素加权采样功能...")
    
    # 测试谱残差显著图
    test_spectral_saliency()
    
    # 分析超像素权重
    analyze_superpixel_weights()
    
    # 测试完整的超像素加权采样流程
    test_superpixel_weighted_sampling()
    
    print("\n所有测试完成!")