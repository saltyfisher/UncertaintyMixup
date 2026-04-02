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

# 添加当前目录到 Python 路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models_ import get_model, get_target_layer
from datasets import get_dataloaders


def denormalize(image_tensor):
    """
    反归一化图像张量
    
    Args:
        image_tensor: 图像张量 (C, H, W)
    
    Returns:
        反归一化后的图像张量
    """
    # 由于数据集只做了 ToTensor 变换，没有做 Normalize，所以直接返回
    return image_tensor


def visualize_heatmap_with_labels(
    model, 
    target_layers, 
    image_tensor, 
    image_path, 
    label,
    class_names=None,
    method='gradcam'
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
    
    # 根据方法选择相应的 CAM 类
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


def generate_gradcam_heatmaps(
    args,
    model_path,
    data_dir,
    output_dir,
    dataset_type='chestct',
    model_arch='resnet18',
    model_type='without_dropout',
    batch_size=1,
    num_workers=4,
    magnification=None,
    methods=['gradcam', 'gradcam++', 'layercam'],
    save_individual=True,
    save_combined=True
):
    """
    使用 Grad-CAM 系列方法为训练集中的每个样本生成热力图
    
    Args:
        model_path: 模型参数文件路径
        data_dir: 数据集根目录
        output_dir: 输出目录
        dataset_type: 数据集类型 ('chestct', 'breakhis', etc.)
        model_arch: 模型架构 ('resnet18', 'resnet50', etc.)
        model_type: 模型类型 ('with_dropout' 或 'without_dropout')
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        magnification: BreakHis 数据集的放大倍数
        methods: 要使用的 Grad-CAM 方法列表
        save_individual: 是否保存单独的热力图
        save_combined: 是否保存组合的热力图
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查 CUDA 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    # 获取类别数量
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
    num_classes = num_classes_map.get(dataset_type, 2)

    # 加载模型
    print(f"加载模型：{model_arch}, {model_type}")
    model = get_model(model_type=model_type, pretrain=False, model_arch=model_arch, num_classes=num_classes)
    
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
        
        # 移除可能存在的'module.'前缀（DataParallel 包装的情况）
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("模型参数加载成功！")
    else:
        print(f"警告：模型参数文件不存在：{model_path}")
        print("将使用随机初始化的模型")
    
    # 将模型移到设备上并设置为评估模式
    model = model.to(device)
    model.eval()
    
    # 禁用 dropout（如果有）
    if hasattr(model, 'disable_dropout'):
        model.disable_dropout()
    
    # 获取目标层
    target_layers = get_target_layer(model, model_type, model_arch)
    print(f"目标层：{target_layers}")
    
    # 获取数据加载器
    print(f"加载数据集：{dataset_type}, {data_dir}")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224) if dataset_type != 'breakhis' else (448, 448)),
        transforms.ToTensor(),
    ])
    
    if args.dataset_type == 'breakhis':
        args.data_dir = '/workspace/MedicalImageClassficationData/BreakHis'
    elif args.dataset_type == 'chestct':
        args.data_dir = '/workspace/MedicalImageClassficationData/chest-ctscan-images_datasets'
    elif args.dataset_type == 'padufes':
        args.data_dir = '/workspace/MedicalImageClassficationData/PAD-UFES-20'
    elif args.dataset_type == 'bladder':
        args.data_dir = '/workspace/MedicalImageClassficationData/EndoscopicBladderTissue'
    elif args.dataset_type == 'kvasir':
        args.data_dir = '/workspace/MedicalImageClassficationData/kvasir-dataset'
    # 根据数据集类型获取数据加载器
    train_loader, test_loader = get_dataloaders(args.data_dir, batch_size, args.dataset_type, magnification, shuffle=False)

    print(f"开始生成热力图...")
    if args.dataset_type == 'breakhis':
        output_dir = os.path.join(output_dir, f'breakhis{args.magnification}')
    else:
        output_dir = os.path.join(output_dir, args.dataset_type)
    if args.use_augmentation:
        output_dir += f'_{args.strategy}'
    os.makedirs(output_dir, exist_ok=True)
    # 处理每个批次
    total_samples = 0
    
    # 定义类别名称映射（用于显示）
    class_names_dict = {
        'chestct': ['ADE', 'LARGE', 'NORMAL', 'SQU'],
        'breakhis': ['ADE', 'DUCT', 'FIBR', 'LOB', 'MUC', 'PAPI', 'PHY', 'TUB'],
        'bladder': ['Normal', 'Tumor'],
        'kvasir': ['Esophagitis', 'Barrett', 'Polyp', 'Cancer', 'Ulcerative Colitis', 'Crohn', 'Normal Z-line', 'Normal Pylorus'],
        'padufes': ['Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesion']
    }
    class_names = class_names_dict.get(dataset_type, None)

    for loader in [train_loader, test_loader]:
        for batch_idx, (images, labels, paths) in enumerate(tqdm(loader, desc="Processing batches")):
            # 将数据移到设备上
            images = images.to(device)
            
            # 处理批次中的每个样本
            for i in range(images.size(0)):
                image_tensor = images[i]
                label = labels[i]
                sample_output_name = paths[0].split('/')[-1].split('.')[0]
                output_path = os.path.join(
                            output_dir,
                            f'{sample_output_name}_{methods}.png'
                        )
                # if os.path.exists(output_path):
                #     continue
                
                # 反归一化图像用于可视化
                img_denorm = denormalize(image_tensor)
                
                # 转换为 numpy 数组
                original_img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
                original_img_np = np.clip(original_img_np, 0, 1)
                
                # 创建样本特定的输出目录
                sample_id = paths[0].split('/')[-1]
                
                
                
                # 存储所有方法的热力图
                heatmaps = []
                
                # 为每种方法生成热力图
                try:
                    # 调用 visualize_heatmap_with_labels 函数生成带标签的热力图
                    _, pred_class, pred_confidence = visualize_heatmap_with_labels(
                        model=model,
                        target_layers=target_layers,
                        image_tensor=img_denorm,
                        image_path=output_path,
                        label=label.item(),
                        class_names=class_names,
                        method=methods
                    )
                    print(f"  已保存 {methods} 热力图到 {output_path}")
                
                except Exception as e:
                    print(f"  生成 {methods} 热力图时出错：{e}")
                    import traceback
                    traceback.print_exc()
                
                total_samples += 1
    
    print(f"\n热力图生成完毕！共处理 {total_samples} 个样本")
    print(f"结果保存在：{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用 Grad-CAM 生成训练集样本的热力图')
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str, help='模型参数文件路径 (.pth)')
    parser.add_argument('--model_arch', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet50', 'resnet34', 'inception', 'wideresnet', 'efficientnet_b0'],
                       help='模型架构')
    parser.add_argument('--model_type', type=str, default='without_dropout',
                       choices=['with_dropout', 'without_dropout'],
                       help='模型类型（是否包含 dropout）')
    parser.add_argument('--use_augmentation', action='store_true', help='是否使用数据增强')
    parser.add_argument('--strategy', type=str, default='uncertaintymixup', choices=['uncertaintymixup', 'mixup', 'cutmixrand', 'puzzlemix', 'comix', 'guided'])
    
    # 数据集相关参数
    parser.add_argument('--data_dir', type=str, default='/workspace/MedicalImageClassficationData/', help='数据集根目录')
    parser.add_argument('--dataset_type', type=str, default='chestct',
                       choices=['chestct', 'breakhis', 'padufes', 'kvasir', 'bladder'],
                       help='数据集类型')
    parser.add_argument('--magnification', type=str, default=None,
                       choices=['40', '100', '200', '400'],
                       help='BreakHis 数据集的放大倍数')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./vis_results/gradcam_results',
                       help='热力图输出目录')
    
    # 方法参数
    parser.add_argument('--methods', type=str, default='gradcam++', choices=['gradcam', 'gradcam++', 'layercam', 'eigencam', 'finercam'],
                       help='要使用的 Grad-CAM 方法列表')
    parser.add_argument('--save_individual', action='store_true', default=True,
                       help='是否保存单独的热力图')
    parser.add_argument('--no_save_combined', action='store_true', default=False,
                       help='不保存组合的热力图')
    
    # 数据加载参数
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作进程数')
    
    args = parser.parse_args()
    args.model_path = 'best_model'
    if args.use_augmentation:
        args.model_path += args.strategy
    args.model_path += f'_{args.dataset_type}'
    if args.dataset_type == 'breakhis':
        args.model_path += f'_{args.magnification}'
    args.model_path += '.pth'
    # args.methods = 'layercam'
    magnifications = ['40', '100', '200', '400']
    # magnifications = ['40']
    datasets = ['chestct', 'breakhis']
    strategies = ['mixup', 'cutmixrand', 'puzzlemix', 'comix', 'guided', 'uncertaintymixup']
    args.output_dir = os.path.join(args.output_dir, args.methods)
    # 调用主函数
    args.dataset_type = 'chestct'
    generate_gradcam_heatmaps(
            args,
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            dataset_type=args.dataset_type,
            model_arch=args.model_arch,
            model_type=args.model_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            magnification=args.magnification,
            methods=args.methods,
            save_individual=args.save_individual,
            save_combined=not args.no_save_combined
        )
    
    for s in strategies:
        args.strategy = s
        args.use_augmentation = True
        generate_gradcam_heatmaps(
            args,
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            dataset_type=args.dataset_type,
            model_arch=args.model_arch,
            model_type=args.model_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            magnification=args.magnification,
            methods=args.methods,
            save_individual=args.save_individual,
            save_combined=not args.no_save_combined
        )

    args.dataset_type = 'breakhis'
    args.use_augmentation = False
    for m in magnifications:
        args.magnification = m
        generate_gradcam_heatmaps(
                args,
                model_path=args.model_path,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                dataset_type=args.dataset_type,
                model_arch=args.model_arch,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                magnification=args.magnification,
                methods=args.methods,
                save_individual=args.save_individual,
                save_combined=not args.no_save_combined
            )
    args.use_augmentation = True
    for m in magnifications:
        args.magnification = m
        for s in strategies:
            args.strategy = s
            generate_gradcam_heatmaps(
                args,
                model_path=args.model_path,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                dataset_type=args.dataset_type,
                model_arch=args.model_arch,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                magnification=args.magnification,
                methods=args.methods,
                save_individual=args.save_individual,
                save_combined=not args.no_save_combined
            )
