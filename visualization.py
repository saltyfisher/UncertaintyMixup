import torch
import cv2
import numpy as np
import os
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import functional as F
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import get_model, get_target_layer
from dataset import get_tiny_imagenet_dataloader, denormalize


def visualize_heatmap(model, target_layers, image_tensor, image_path, method='gradcam'):
    """
    使用指定方法生成热力图
    
    Args:
        model: 模型
        target_layers: 目标层
        image_tensor: 输入图像张量
        image_path: 图像保存路径
        method: 可视化方法 ('gradcam', 'gradcam++', 'layercam')
    """
    # 将图像张量移到CPU并转换为numpy数组以进行可视化
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # 根据方法选择相应的CAM类
    if method == 'gradcam':
        cam = GradCAM(model=model, target_layers=[target_layers], reshape_transform=None)
    elif method == 'gradcam++':
        cam = GradCAMPlusPlus(model=model, target_layers=[target_layers], reshape_transform=None)
    elif method == 'layercam':
        cam = LayerCAM(model=model, target_layers=[target_layers], reshape_transform=None)
    else:
        raise ValueError("不支持的方法，支持的方法包括: 'gradcam', 'gradcam++', 'layercam'")
    
    # 计算热力图
    input_tensor = image_tensor.unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    
    # 打印调试信息
    print(f"热力图统计信息 - 最小值: {grayscale_cam.min():.6f}, 最大值: {grayscale_cam.max():.6f}, 平均值: {grayscale_cam.mean():.6f}")
    
    # 将热力图叠加到原始图像上
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    # 保存可视化结果
    cv2.imwrite(image_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    return visualization


def create_combined_heatmap_visualization(original_image, heatmaps, output_path, methods):
    """
    将原始图像和所有热力图组合成一张图
    
    Args:
        original_image: 原始图像 (numpy array)
        heatmaps: 热力图列表
        output_path: 输出路径
        methods: 方法名称列表
    """
    try:
        # 调整原始图像大小以适应布局
        h, w = original_image.shape[:2]
        
        # 创建一个大的画布来放置所有图像 (2行2列)
        combined_image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # 放置原始图像 (第一行第一列)
        original_vis = (original_image * 255).astype(np.uint8)
        combined_image[:h, :w] = original_vis
        
        # 放置热力图
        for i, (heatmap, method) in enumerate(zip(heatmaps, methods)):
            row = (i + 1) // 2
            col = (i + 1) % 2
            # 确保热力图是正确的格式
            if len(heatmap.shape) == 3 and heatmap.shape[2] == 3:
                # 检查热力图值范围并进行适当转换
                if heatmap.max() <= 1.0:
                    heatmap_vis = (heatmap * 255).astype(np.uint8)
                else:
                    heatmap_vis = heatmap.astype(np.uint8)
                combined_image[row*h:(row+1)*h, col*w:(col+1)*w] = cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR)
                print(f"热力图 {method} 值范围: [{heatmap_vis.min()}, {heatmap_vis.max()}]")
            else:
                print(f"警告: 热力图 {method} 格式不正确")
        
        # 添加文本标签
        cv2.putText(combined_image, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for i, method in enumerate(methods):
            row = (i + 1) // 2
            col = (i + 1) % 2
            cv2.putText(combined_image, method, (col*w + 10, row*h + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存组合图像
        cv2.imwrite(output_path, combined_image)
        print(f"成功保存组合图像到 {output_path}")
        return True
    except Exception as e:
        print(f"创建组合图像时出错: {e}")
        return False


def create_comparison_heatmap_visualization(original_image, heatmaps_dict, output_path, methods, target_class):
    """
    创建带dropout和不带dropout模型的对比热力图
    
    Args:
        original_image: 原始图像 (numpy array)
        heatmaps_dict: 包含两种模型热力图的字典 {'with_dropout': [...], 'without_dropout': [...]}
        output_path: 输出路径
        methods: 方法名称列表
        target_class: 目标类别名称
    """
    try:
        # 调整原始图像大小以适应布局
        h, w = original_image.shape[:2]
        
        # 创建一个大的画布来放置所有图像 (3行4列)
        # 第一行: 原始图像, 无dropout模型的三种热力图
        # 第二行: 空白, 有dropout模型的三种热力图
        # 第三行: 标题区域
        combined_image = np.zeros((h * 3, w * 4, 3), dtype=np.uint8)
        
        # 放置原始图像 (第一行第一列)
        original_vis = (original_image * 255).astype(np.uint8)
        combined_image[:h, :w] = original_vis
        
        # 放置无dropout模型的热力图 (第一行第2-4列)
        for i, (heatmap, method) in enumerate(zip(heatmaps_dict['without_dropout'], methods)):
            if len(heatmap.shape) == 3 and heatmap.shape[2] == 3:
                # 检查热力图值范围并进行适当转换
                if heatmap.max() <= 1.0:
                    heatmap_vis = (heatmap * 255).astype(np.uint8)
                else:
                    heatmap_vis = heatmap.astype(np.uint8)
                combined_image[:h, (i+1)*w:(i+2)*w] = cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR)
        
        # 放置有dropout模型的热力图 (第二行第2-4列)
        for i, (heatmap, method) in enumerate(zip(heatmaps_dict['with_dropout'], methods)):
            if len(heatmap.shape) == 3 and heatmap.shape[2] == 3:
                # 检查热力图值范围并进行适当转换
                if heatmap.max() <= 1.0:
                    heatmap_vis = (heatmap * 255).astype(np.uint8)
                else:
                    heatmap_vis = heatmap.astype(np.uint8)
                combined_image[h:2*h, (i+1)*w:(i+2)*w] = cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR)
        
        # 添加总标题（显示类别名称）
        cv2.putText(combined_image, f'Class: {target_class}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 添加子标题（显示模型类型和方法名称）
        for i, method in enumerate(methods):
            # 无dropout模型的子标题
            cv2.putText(combined_image, f'Without Dropout - {method}', ((i+1)*w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 有dropout模型的子标题
            cv2.putText(combined_image, f'With Dropout - {method}', ((i+1)*w + 10, h + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 保存组合图像
        cv2.imwrite(output_path, combined_image)
        print(f"成功保存对比图像到 {output_path}")
        return True
    except Exception as e:
        print(f"创建对比图像时出错: {e}")
        return False


def generate_heatmaps_for_model(model, target_layers, img_denorm, methods):
    """
    为指定模型生成所有热力图
    
    Args:
        model: 模型
        target_layers: 目标层
        img_denorm: 反归一化后的图像
        methods: 方法列表
    
    Returns:
        heatmaps: 热力图列表
    """
    heatmaps = []
    for method in methods:
        # 生成单个热力图
        input_tensor = img_denorm.unsqueeze(0)
        
        if method == 'gradcam':
            cam = GradCAM(model=model, target_layers=[target_layers], reshape_transform=None)
        elif method == 'gradcam++':
            cam = GradCAMPlusPlus(model=model, target_layers=[target_layers], reshape_transform=None)
        elif method == 'layercam':
            cam = LayerCAM(model=model, target_layers=[target_layers], reshape_transform=None)
        
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        print(f"{method} 热力图值范围: [{grayscale_cam.min():.6f}, {grayscale_cam.max():.6f}]")
        
        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        heatmaps.append(heatmap)
    
    return heatmaps


def run_visualization(data_dir, output_dir, model_type='without_dropout'):
    """
    运行热力图可视化
    
    Args:
        data_dir: 数据集目录
        output_dir: 输出目录
        model_type: 模型类型 ('with_dropout' 或 'without_dropout')
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取模型
    model = get_model(model_type)
    model.eval()
    
    # 获取目标层
    target_layers = get_target_layer(model, model_type)
    
    # 获取数据加载器
    dataloader, indices = get_tiny_imagenet_dataloader(data_dir, batch_size=10)
    
    # 创建模型特定的输出目录
    model_output_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 处理每个图像
    for batch_idx, (data, targets) in enumerate(dataloader):
        for i in range(data.size(0)):
            print(f"\n处理 {model_type} 模型的样本 {batch_idx * 10 + i}")
            image_tensor = data[i]
            target = targets[i]
            
            # 反归一化图像用于可视化
            img_denorm = denormalize(image_tensor)
            
            # 转换为numpy数组用于组合显示
            original_img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
            original_img_np = np.clip(original_img_np, 0, 1)
            
            # 打印原始图像信息
            print(f"原始图像值范围: [{original_img_np.min():.4f}, {original_img_np.max():.4f}]")
            
            # 为每种方法生成热力图并存储
            methods = ['gradcam', 'gradcam++', 'layercam']
            heatmaps = []
            
            for method in methods:
                try:
                    # 生成单个热力图
                    input_tensor = img_denorm.unsqueeze(0)
                    
                    if method == 'gradcam':
                        cam = GradCAM(model=model, target_layers=[target_layers], reshape_transform=None)
                    elif method == 'gradcam++':
                        cam = GradCAMPlusPlus(model=model, target_layers=[target_layers], reshape_transform=None)
                    elif method == 'layercam':
                        cam = LayerCAM(model=model, target_layers=[target_layers], reshape_transform=None)
                    
                    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
                    print(f"{method} 热力图值范围: [{grayscale_cam.min():.6f}, {grayscale_cam.max():.6f}]")
                    
                    img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
                    img_np = np.clip(img_np, 0, 1)
                    heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                    heatmaps.append(heatmap)
                    
                    # 也保存单独的热力图
                    output_path = os.path.join(
                        model_output_dir, 
                        f"sample_{batch_idx * 10 + i}_target_{target}_{method}.png"
                    )
                    # 确保热力图值范围正确
                    if heatmap.max() <= 1.0:
                        heatmap_vis = (heatmap * 255).astype(np.uint8)
                    else:
                        heatmap_vis = heatmap.astype(np.uint8)
                    cv2.imwrite(output_path, cv2.cvtColor(heatmap_vis, cv2.COLOR_RGB2BGR))
                    print(f"已保存 {model_type} 模型的 {method} 热力图到 {output_path}")
                except Exception as e:
                    print(f"生成 {method} 热力图时出错: {e}")
            
            # 创建组合图像
            combined_output_path = os.path.join(
                model_output_dir,
                f"sample_{batch_idx * 10 + i}_target_{target}_combined.png"
            )
            success = create_combined_heatmap_visualization(original_img_np, heatmaps, combined_output_path, methods)
            if success:
                print(f"已保存 {model_type} 模型的组合热力图到 {combined_output_path}")
            else:
                print(f"未能保存 {model_type} 模型的组合热力图到 {combined_output_path}")


def run_comparison_visualization(data_dir, output_dir):
    """
    运行带dropout和不带dropout模型的对比可视化
    
    Args:
        data_dir: 数据集目录
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数据加载器（使用不带dropout的模型获取数据）
    dataloader, indices = get_tiny_imagenet_dataloader(data_dir, batch_size=10)
    
    # 处理每个图像
    for batch_idx, (data, targets) in enumerate(dataloader):
        for i in range(data.size(0)):
            print(f"\n处理对比样本 {batch_idx * 10 + i}")
            image_tensor = data[i]
            target = targets[i]
            
            # 反归一化图像用于可视化
            img_denorm = denormalize(image_tensor)
            
            # 转换为numpy数组用于组合显示
            original_img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
            original_img_np = np.clip(original_img_np, 0, 1)
            
            # 打印原始图像信息
            print(f"原始图像值范围: [{original_img_np.min():.4f}, {original_img_np.max():.4f}]")
            
            # 获取两种模型
            model_without_dropout = get_model('without_dropout')
            model_without_dropout.eval()
            target_layers_without_dropout = get_target_layer(model_without_dropout, 'without_dropout')
            
            model_with_dropout = get_model('with_dropout')
            model_with_dropout.eval()
            target_layers_with_dropout = get_target_layer(model_with_dropout, 'with_dropout')
            
            # 为每种方法生成热力图并存储
            methods = ['gradcam', 'gradcam++', 'layercam']
            
            # 生成无dropout模型的热力图
            print("生成无dropout模型的热力图...")
            heatmaps_without_dropout = generate_heatmaps_for_model(
                model_without_dropout, target_layers_without_dropout, img_denorm, methods)
            
            # 生成有dropout模型的热力图
            print("生成有dropout模型的热力图...")
            heatmaps_with_dropout = generate_heatmaps_for_model(
                model_with_dropout, target_layers_with_dropout, img_denorm, methods)
            
            # 创建对比图像（只保存comparison图像，不保存单独的热力图）
            heatmaps_dict = {
                'without_dropout': heatmaps_without_dropout,
                'with_dropout': heatmaps_with_dropout
            }
            
            comparison_output_path = os.path.join(
                output_dir,
                f"sample_{batch_idx * 10 + i}_target_{target}_comparison.png"
            )
            success = create_comparison_heatmap_visualization(
                original_img_np, heatmaps_dict, comparison_output_path, methods, target)
            if success:
                print(f"已保存对比热力图到 {comparison_output_path}")
            else:
                print(f"未能保存对比热力图到 {comparison_output_path}")


if __name__ == "__main__":
    # Tiny ImageNet数据集路径
    data_dir = "../tiny-imagenet-200"
    
    # 输出目录
    output_dir = "./heatmaps"
    
    # 运行对比可视化
    print("运行带dropout和不带dropout模型的对比可视化...")
    run_comparison_visualization(data_dir, output_dir)
    
    print("\n所有对比热力图已生成完毕！")