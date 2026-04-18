import torch
import cv2
import numpy as np
import os
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, EigenCAM, FinerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import models

# 添加当前目录到 Python 路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models_ import get_model, get_target_layer


def visualize_heatmap_with_labels(
    model, 
    target_layers, 
    image_tensor, 
    image_path, 
    label,
    class_names=None,
    method='finercam',
    save_individual=True,
):
    """
    使用指定方法生成热力图并添加标签信息
    
    Args:
        model: 模型
        target_layers: 目标层
        image_tensor: 输入图像张量
        image_path: 图像保存路径
        label: 真实标签
        class_names: 类别名称列表
        method: 可视化方法 ('gradcam', 'gradcam++', 'layercam', 'eigencam', 'finercam')
    """
    # 将图像张量移到 CPU 并转换为 numpy 数组以进行可视化
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # 获取模型预测
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)
        pred_class = predicted.item()
        pred_confidence = probabilities[0, label].item()
    
    # 获取类别名称
    if class_names is None:
        true_label_name = f"Class {label}"
        pred_label_name = f"Class {pred_class}"
    else:
        true_label_name = class_names[label] if label < len(class_names) else f"Class {label}"
        pred_label_name = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"
    
    visualization = None
    # 根据方法选择相应的 CAM 类
    if save_individual:
        if method == 'gradcam':
            cam = GradCAM(model=model, target_layers=[target_layers], reshape_transform=None)
        elif method == 'gradcam++':
            cam = GradCAMPlusPlus(model=model, target_layers=[target_layers], reshape_transform=None)
        elif method == 'layercam':
            cam = LayerCAM(model=model, target_layers=[target_layers], reshape_transform=None)
        elif method == 'eigencam':
            cam = EigenCAM(model=model, target_layers=[target_layers], reshape_transform=None)
        elif method == 'finercam':
            cam = FinerCAM(model=model, target_layers=[target_layers], reshape_transform=None)
        else:
            raise ValueError(f"不支持的方法：{method}。支持的方法包括：'gradcam', 'gradcam++', 'layercam', 'eigencam', 'finercam'")
        
        # 计算热力图
        input_tensor = image_tensor.unsqueeze(0)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(label)])[0, :]
        
        # 打印调试信息
        print(f"热力图统计信息 - 最小值：{grayscale_cam.min():.6f}, 最大值：{grayscale_cam.max():.6f}, 平均值：{grayscale_cam.mean():.6f}")
        
        # 创建标题文本
        title_text = f"True: {true_label_name} | Pred: {pred_label_name} (Conf: {pred_confidence:.3f})"
        
        # 使用 matplotlib 创建图形
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 显示叠加了热力图的图像
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        ax.imshow(visualization)
        
        # 设置标题（包含真实标签、预测标签和置信度）
        ax.set_title(title_text, fontsize=18, fontweight='bold', pad=20)
        
        # 移除坐标轴
        ax.axis('off')
        
        # 调整布局以避免标题被裁剪
        plt.tight_layout()
        
        # 保存图像，使用较高的 DPI 以保证质量
        plt.savefig(image_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        
        # 关闭图形以释放内存
        plt.close(fig)
    
    return visualization, pred_class, pred_confidence


def extract_model_info_from_filename(filename):
    """
    从文件名中提取模型策略和数据集信息
    
    Args:
        filename: 图像文件名，如 'exp_mixed_compare_comix_mixed1.png'
    
    Returns:
        strategy: 策略名称 ('comix', 'guided', 'puzzlemix', 'uncertaintymixup')
        dataset_type: 数据集类型 ('chestct' 或 'breakhis')
        magnification: 放大倍数 (如果是 BreakHis 数据集)
    """
    # 默认配置
    dataset_type = 'chestct'  # 默认为 chestct
    magnification = None
    
    # 提取策略名称
    strategies = ['comix', 'guided', 'puzzlemix', 'uncertaintymixup_matting', 'uncertaintymixup']
    strategy = None
    
    for s in strategies:
        if s in filename.lower():
            strategy = s
            break
    
    if strategy is None:
        print(f"警告：无法从文件名 '{filename}' 中提取策略名称")
        return None, None, None
    
    # 检查是否为 BreakHis 数据集（包含放大倍数）
    if 'breakhis' in filename.lower():
        dataset_type = 'breakhis'
        # 尝试提取放大倍数
        for mag in ['40', '100', '200', '400']:
            if f'_{mag}' in filename or f'{mag}_' in filename:
                magnification = mag
                break
    
    return strategy, dataset_type, magnification


def generate_test_folder_heatmaps(
    args,
    test_dir,
    output_dir,
    model_arch='resnet18',
    model_type='without_dropout',
    methods=['finercam'],
):
    """
    为 test 文件夹中的每个样本生成热力图
    
    Args:
        args: 参数对象
        test_dir: test 文件夹路径
        output_dir: 输出目录
        model_arch: 模型架构
        model_type: 模型类型
        methods: 要使用的方法
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查 CUDA 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    # 获取类别名称映射
    class_names_dict = {
        'chestct': ['ADE', 'LARGE', 'NORMAL', 'SQU'],
        'breakhis': ['ADE', 'DUCT', 'FIBR', 'LOB', 'MUC', 'PAPI', 'PHY', 'TUB'],
        'bladder': ['Normal', 'Tumor'],
        'kvasir': ['Esophagitis', 'Barrett', 'Polyp', 'Cancer', 'Ulcerative Colitis', 'Crohn', 'Normal Z-line', 'Normal Pylorus'],
        'padufes': ['Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesion']
    }
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"发现 {len(image_files)} 个图像文件")
    
    # 按策略分组处理
    processed_count = 0
    error_count = 0
    
    for image_file in tqdm(image_files, desc="Processing images"):
        # 从文件名中提取模型信息
        strategy, dataset_type, magnification = extract_model_info_from_filename(image_file)
        strategy = 'uncertaintymixup'
        if strategy is None:
            print(f"跳过文件：{image_file} (无法识别策略)")
            continue
        
        # 构建模型路径
        model_path = f'best_model_{strategy}_{dataset_type}'
        if dataset_type == 'breakhis' and magnification:
            model_path += f'_{magnification}'
        model_path += '.pth'
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"警告：模型文件不存在：{model_path}，跳过 {image_file}")
            error_count += 1
            continue
        
        # 确定类别数量
        num_classes_map = {
            'lymphoma': 3,
            'breakhis': 8,
            'lc25000': 5,
            'rect': 2,
            'chestct': 4,
            'bladder': 4,
            'corona': 7,
            'kvasir': 8,
            'padufes': 6
        }
        num_classes = num_classes_map.get(dataset_type, 4)
        
        # 加载模型
        print(f"\n加载模型：{model_arch}, {model_type}, 策略：{strategy}")
        model = models.__dict__[model_arch](num_classes=num_classes)
        
        # 加载模型参数
        if os.path.isfile(model_path):
            print(f"加载模型参数：{model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # 处理不同的 checkpoint 格式
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 移除可能存在的'module.'前缀
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            print("模型参数加载成功！")
        else:
            print(f"错误：模型参数文件不存在：{model_path}")
            error_count += 1
            continue
        
        # 将模型移到设备上并设置为评估模式
        model = model.to(device)
        model.eval()
        
        # 禁用 dropout（如果有）
        if hasattr(model, 'disable_dropout'):
            model.disable_dropout()
        
        # 获取目标层
        target_layers = get_target_layer(model, model_type, model_arch)
        print(f"目标层：{target_layers}")
        
        # 获取类别名称
        class_names = class_names_dict.get(dataset_type, None)
        
        # 加载图像
        image_path = os.path.join(test_dir, image_file)
        try:
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            
            # 确定图像尺寸
            img_size = (448, 448) if dataset_type == 'breakhis' else (224, 224)
            
            # 图像预处理
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])
            image_tensor = transform(image).to(device)
            
            # 由于 test 文件夹中的图像已经是混合图像，我们假设标签为 0（或者可以根据需要调整）
            # 这里我们使用预测概率最高的类别作为标签
            with torch.no_grad():
                output = model(image_tensor.unsqueeze(0))
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted_label = torch.max(probabilities, 1)
                label = predicted_label.item()
            
            # 构建输出路径
            output_path = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}_{methods}.png')
            
            # 生成热力图
            _, pred_class, pred_confidence = visualize_heatmap_with_labels(
                model=model,
                target_layers=target_layers,
                image_tensor=image_tensor,
                image_path=output_path,
                label=label,
                class_names=class_names,
                method=methods,
                save_individual=True
            )
            
            print(f"  ✓ 已保存 {methods} 热力图到 {output_path}")
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ 处理 {image_file} 时出错：{e}")
            import traceback
            traceback.print_exc()
            error_count += 1
    
    print(f"\n处理完成！共处理 {processed_count} 个图像，{error_count} 个错误")
    print(f"结果保存在：{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为 test 文件夹中的图像生成 FinerCAM 热力图')
    
    # 模型相关参数
    parser.add_argument('--model_arch', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet50', 'resnet34', 'inception', 'wideresnet', 'efficientnet_b0'],
                       help='模型架构')
    parser.add_argument('--model_type', type=str, default='without_dropout',
                       choices=['with_dropout', 'without_dropout'],
                       help='模型类型（是否包含 dropout）')
    
    # 路径参数
    parser.add_argument('--test_dir', type=str, default='./test',
                       help='test 文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./test/finercam_results',
                       help='热力图输出目录')
    
    # 方法参数
    parser.add_argument('--methods', type=str, default='finercam', 
                       choices=['gradcam', 'gradcam++', 'layercam', 'eigencam', 'finercam'],
                       help='要使用的 Grad-CAM 方法')
    
    # 数据加载参数
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作进程数')
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("=" * 80)
    print("FinerCAM 显著图生成器 - Test 文件夹处理")
    print("=" * 80)
    print(f"模型架构：{args.model_arch}")
    print(f"模型类型：{args.model_type}")
    print(f"Test 文件夹：{args.test_dir}")
    print(f"输出目录：{args.output_dir}")
    print(f"使用方法：{args.methods}")
    print("=" * 80)
    
    # 执行热力图生成
    generate_test_folder_heatmaps(
        args,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_arch=args.model_arch,
        model_type=args.model_type,
        methods=args.methods,
    )
    
    print("\n" + "=" * 80)
    print("所有可视化完成！")
    print("=" * 80)
