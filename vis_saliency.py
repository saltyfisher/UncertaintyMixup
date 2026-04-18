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
    method='gradcam',
    only_rank=False,
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
    save_combined=True,
    selected_samples=None
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
        selected_samples: 预筛选的样本列表，如果为 None 则处理所有样本
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
    
    # 存储所有样本的置信度信息用于排序
    all_samples_confidence = []

    for loader in [train_loader, test_loader]:
        for batch_idx, (images, labels, paths) in enumerate(tqdm(loader, desc="Processing batches")):
            # 将数据移到设备上
            images = images.to(device)
            
            # 处理批次中的每个样本
            for i in range(images.size(0)):
                image_tensor = images[i]
                label = labels[i]
                sample_output_name = paths[0].split('/')[-1].split('.')[0]
                
                # 如果提供了预筛选的样本列表，只处理这些样本
                if selected_samples is not None:
                    sample_id = paths[0].split('/')[-1]
                    if sample_id not in selected_samples:
                        continue
                
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
                        method=methods,
                        save_individual=save_individual
                    )
                    print(f"  已保存 {methods} 热力图到 {output_path}")
                    
                    # 保存样本信息用于排序
                    all_samples_confidence.append({
                        'filename': sample_id,
                        'pred_confidence': pred_confidence,
                        'true_label': label.item(),
                        'pred_label': pred_class,
                        'output_path': output_path
                    })
                
                except Exception as e:
                    print(f"  生成 {methods} 热力图时出错：{e}")
                    import traceback
                    traceback.print_exc()
                
                total_samples += 1
    
    # 如果启用了 only_rank 模式，只保存排序结果而不保存热力图
    if hasattr(args, 'only_rank') and args.only_rank:
        print(f"\n仅排名模式：不保存热力图，只保存排序结果")
    
    # 按照预测置信度从大到小排序
    all_samples_confidence.sort(key=lambda x: x['pred_confidence'], reverse=True)
    
    # 保存排序结果到文件
    rank_output_path = os.path.join(output_dir, f'samples_ranking_{methods}.txt')
    with open(rank_output_path, 'w', encoding='utf-8') as f:
        f.write("Rank\tFilename\tTrue Label\tPred Label\tPred Confidence\n")
        f.write("-" * 80 + "\n")
        for idx, sample_info in enumerate(all_samples_confidence, 1):
            f.write(f"{idx}\t{sample_info['filename']}\t{sample_info['true_label']}\t"
                   f"{sample_info['pred_label']}\t{sample_info['pred_confidence']:.6f}\n")
    
    print(f"\n热力图生成完毕！共处理 {total_samples} 个样本")
    print(f"排序结果已保存到：{rank_output_path}")
    print(f"结果保存在：{output_dir}")
    
    return all_samples_confidence


def load_selected_samples_from_file(ranking_file_path, top_k=50):
    """
    从排序文件中加载置信度最高和最低的样本
    
    Args:
        ranking_file_path: 排序结果文件路径
        top_k: 选择最高和最低各多少个样本
    
    Returns:
        筛选后的样本文件名集合
    """
    if not os.path.exists(ranking_file_path):
        return None
    
    selected_filenames = set()
    
    with open(ranking_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 跳过前两行（标题和分隔线）
        data_lines = lines[2:]
        
        if len(data_lines) < 2 * top_k:
            print(f"警告：文件中的样本数量不足 {2 * top_k} 个")
            return None
        
        # 取前 top_k 个（置信度最高）
        for i in range(top_k):
            parts = data_lines[i].strip().split('\t')
            if len(parts) >= 2:
                selected_filenames.add(parts[1])
        
        # 取后 top_k 个（置信度最低）
        for i in range(len(data_lines) - top_k, len(data_lines)):
            parts = data_lines[i].strip().split('\t')
            if len(parts) >= 2:
                selected_filenames.add(parts[1])
    
    print(f"从文件加载了 {len(selected_filenames)} 个筛选样本")
    return selected_filenames


def select_top_bottom_samples(all_samples_confidence, top_k=50):
    """
    选择置信度最高和最低的样本
    
    Args:
        all_samples_confidence: 所有样本的置信度列表
        top_k: 选择最高和最低各多少个样本
    
    Returns:
        筛选后的样本文件名集合
    """
    # 已经按置信度降序排序
    # 取前 top_k 个（置信度最高）
    top_samples = all_samples_confidence[:top_k]
    
    # 取后 top_k 个（置信度最低）
    bottom_samples = all_samples_confidence[-top_k:]
    
    # 合并并去重
    selected_filenames = set()
    for sample in top_samples + bottom_samples:
        selected_filenames.add(sample['filename'])
    
    print(f"\n已筛选样本：置信度最高的 {top_k} 个 + 置信度最低的 {top_k} 个，共 {len(selected_filenames)} 个唯一样本")
    
    return selected_filenames


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
    
    # 新增：only_rank 参数
    parser.add_argument('--only_rank', action='store_true', default=False,
                       help='仅保存排序结果，不生成热力图')
    
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
    # strategies = ['mixup', 'cutmixrand', 'puzzlemix', 'comix', 'guided', 'uncertaintymixup']
    strategies = ['puzzlemix', 'comix', 'guided', 'uncertaintymixup']
    args.output_dir = os.path.join(args.output_dir, args.methods)
    
    # 第一步：使用 uncertaintymixup 策略的模型筛选样本
    print("=" * 80)
    print("第一步：使用 UncertaintyMixup 模型筛选置信度最高和最低的各 50 个样本")
    print("=" * 80)
    
    args_copy = argparse.Namespace(**vars(args))  # 保存原始参数
    
    selected_samples_dict = {}  # 存储每个数据集和放大倍数的筛选样本
    temp_selection_dir = os.path.join(args.output_dir, 'temp_selection')
    
    # 检查 temp_selection 文件夹是否已有筛选结果
    need_selection = False
    if not os.path.exists(temp_selection_dir) or not os.listdir(temp_selection_dir):
        print("temp_selection 文件夹为空或不存在，需要重新筛选样本")
        need_selection = True
        os.makedirs(temp_selection_dir, exist_ok=True)
    else:
        print("发现已有的筛选结果，尝试从文件导入...")
        
        # 尝试从文件加载 chestct 的筛选样本
        chestct_ranking_file = os.path.join(temp_selection_dir, 'chestct', f'samples_ranking_{args.methods}.txt')
        if os.path.exists(chestct_ranking_file):
            selected_chestct = load_selected_samples_from_file(chestct_ranking_file, top_k=50)
            if selected_chestct:
                selected_samples_dict['chestct'] = {'samples': selected_chestct, 'magnification': None}
                print(f"ChestCT 数据集从文件加载了 {len(selected_chestct)} 个样本")
            else:
                need_selection = True
        else:
            need_selection = True
        
        # 尝试从文件加载 breakhis 各放大倍数的筛选样本
        for m in magnifications:
            breakhis_key = f'breakhis_{m}'
            breakhis_ranking_file = os.path.join(temp_selection_dir, f'breakhis{m}', f'samples_ranking_{args.methods}.txt')
            if os.path.exists(breakhis_ranking_file):
                selected_breakhis = load_selected_samples_from_file(breakhis_ranking_file, top_k=50)
                if selected_breakhis:
                    selected_samples_dict[breakhis_key] = {'samples': selected_breakhis, 'magnification': m}
                    print(f"BreakHis {m}x 数据集从文件加载了 {len(selected_breakhis)} 个样本")
                else:
                    need_selection = True
            else:
                need_selection = True
        
        if not need_selection:
            print("所有数据集的筛选样本已成功从文件导入！")
        else:
            print("部分筛选结果缺失，需要重新运行筛选")
    
    # 如果需要筛选，则运行筛选流程
    if need_selection:
        print("\n开始重新筛选样本...")
        
        # 处理 chestct 数据集
        args.dataset_type = 'chestct'
        args.use_augmentation = False
        args.strategy = 'uncertaintymixup'
        args.model_path = 'best_model_uncertaintymixup_chestct.pth'
        
        all_samples_confidence_chestct = generate_gradcam_heatmaps(
            args,
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=os.path.join(temp_selection_dir, 'chestct'),
            dataset_type=args.dataset_type,
            model_arch=args.model_arch,
            model_type=args.model_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            magnification=args.magnification,
            methods=args.methods,
            save_individual=False,  # 临时运行，不保存图片
            save_combined=False
        )
        
        # 筛选 chestct 的样本
        selected_chestct = select_top_bottom_samples(all_samples_confidence_chestct, top_k=50)
        selected_samples_dict['chestct'] = {'samples': selected_chestct, 'magnification': None}
        print(f"ChestCT 数据集筛选出 {len(selected_chestct)} 个样本")
        
        # 处理 breakhis 数据集的各个放大倍数
        for m in magnifications:
            args.dataset_type = 'breakhis'
            args.magnification = m
            args.use_augmentation = False
            args.strategy = 'uncertaintymixup'
            args.model_path = f'best_model_uncertaintymixup_breakhis_{m}.pth'
            
            all_samples_confidence_breakhis = generate_gradcam_heatmaps(
                args,
                model_path=args.model_path,
                data_dir=args.data_dir,
                output_dir=os.path.join(temp_selection_dir, f'breakhis{m}'),
                dataset_type=args.dataset_type,
                model_arch=args.model_arch,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                magnification=args.magnification,
                methods=args.methods,
                save_individual=False,  # 临时运行，不保存图片
                save_combined=False
            )
            
            # 筛选 breakhis 的样本
            selected_breakhis = select_top_bottom_samples(all_samples_confidence_breakhis, top_k=50)
            selected_samples_dict[f'breakhis_{m}'] = {'samples': selected_breakhis, 'magnification': m}
            print(f"BreakHis {m}x 数据集筛选出 {len(selected_breakhis)} 个样本")
    else:
        print("\n跳过筛选步骤，使用已保存的筛选结果")
    
    print("\n" + "=" * 80)
    print("第二步：使用所有策略的模型对筛选出的样本进行可视化")
    print("=" * 80)
    
    # 恢复原始参数
    args = args_copy
    
    # 处理 chestct 数据集的所有策略
    args.dataset_type = 'chestct'
    args.use_augmentation = False
    args.model_path = 'best_model_chestct.pth'
    
    print(f"\n处理 ChestCT 数据集...")
    # generate_gradcam_heatmaps(
    #     args,
    #     model_path=args.model_path,
    #     data_dir=args.data_dir,
    #     output_dir=args.output_dir,
    #     dataset_type=args.dataset_type,
    #     model_arch=args.model_arch,
    #     model_type=args.model_type,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     magnification=args.magnification,
    #     methods=args.methods,
    #     save_individual=args.save_individual,
    #     save_combined=not args.no_save_combined,
    #     selected_samples=selected_samples_dict['chestct']['samples']
    # )
    
    # 处理所有增强策略
    for s in strategies:
        args.strategy = s
        args.use_augmentation = True
        args.model_path = f'best_model_{s}_chestct.pth'
        print(f"\n处理 ChestCT 数据集 - 策略：{s}")
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
            selected_samples=selected_samples_dict['chestct']['samples']
        )

    # 处理 breakhis 数据集的所有放大倍数和策略
    args.use_augmentation = False
    for m in magnifications:
        args.magnification = m
        args.dataset_type = 'breakhis'
        args.model_path = f'best_model_breakhis_{m}.pth'
        
        key = f'breakhis_{m}'
        print(f"\n处理 BreakHis {m}x 数据集...")
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
            selected_samples=selected_samples_dict[key]['samples']
        )
        
        args.use_augmentation = True
        for s in strategies:
            args.strategy = s
            args.model_path = f'best_model_{s}_breakhis_{m}.pth'
            print(f"\n处理 BreakHis {m}x 数据集 - 策略：{s}")
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
                selected_samples=selected_samples_dict[key]['samples']
            )
    
    print("\n" + "=" * 80)
    print("所有可视化完成！")
    print("=" * 80)
