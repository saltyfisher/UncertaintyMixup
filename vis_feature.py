import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import models

# 添加当前目录到 Python 路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models_ import get_model, get_target_layer
from datasets import get_dataloaders


def extract_features(model, data_loader, device, layer_name='avgpool'):
    """
    从模型的指定层提取特征
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        layer_name: 要提取特征的层名称
        
    Returns:
        features: 提取的特征数组
        labels: 对应的标签数组
        paths: 对应的图像路径数组
    """
    features_list = []
    labels_list = []
    paths_list = []
    
    model.eval()
    
    # 注册 hook 来提取特征
    feature_hooks = {}
    
    def get_hook(name):
        def hook(module, input, output):
            feature_hooks[name] = output.detach()
        return hook
    
    # 找到目标层并注册 hook
    target_layer = None
    for name, module in model.named_modules():
        if layer_name in name or (hasattr(module, '__class__') and layer_name in module.__class__.__name__.lower()):
            target_layer = name
            module.register_forward_hook(get_hook(name))
            print(f"已注册 hook 到层：{name}")
            break
    
    if target_layer is None:
        print(f"警告：未找到包含 '{layer_name}' 的层，使用 avgpool 作为默认层")
        # 尝试直接访问 avgpool
        if hasattr(model, 'avgpool'):
            model.avgpool.register_forward_hook(get_hook('avgpool'))
            target_layer = 'avgpool'
        elif hasattr(model, '_avg_pooling'):
            model._avg_pooling.register_forward_hook(get_hook('_avg_pooling'))
            target_layer = '_avg_pooling'
    
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(tqdm(data_loader, desc="Extracting features")):
            images = images.to(device)
            
            # 前向传播
            _ = model(images)
            
            # 获取提取的特征
            if target_layer in feature_hooks:
                feature = feature_hooks[target_layer]
                # 展平特征
                feature = feature.view(feature.size(0), -1)
                features_list.append(feature.cpu().numpy())
                labels_list.append(labels.numpy())
                paths_list.extend(paths if isinstance(paths, list) else [paths] * len(images))
    
    # 合并所有批次的特征
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    
    return features, labels, paths_list


def apply_tsne(features, perplexity=30, n_components=2, learning_rate=200, n_iter=1000, random_state=42):
    """
    应用 t-SNE 降维
    
    Args:
        features: 高维特征数组
        perplexity: 困惑度
        n_components: 降维后的维度
        learning_rate: 学习率
        n_iter: 迭代次数
        random_state: 随机种子
        
    Returns:
        tsne_result: 降维后的结果
    """
    print(f"应用 t-SNE (perplexity={perplexity}, n_components={n_components})...")
    tsne = TSNE(
        n_components=n_components
    )
    tsne_result = tsne.fit_transform(features)
    print(f"t-SNE 完成，形状：{tsne_result.shape}")
    return tsne_result


def apply_umap(features, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42):
    """
    应用 UMAP 降维
    
    Args:
        features: 高维特征数组
        n_neighbors: 邻居数
        min_dist: 最小距离
        n_components: 降维后的维度
        metric: 距离度量方式
        random_state: 随机种子
        
    Returns:
        umap_result: 降维后的结果
    """
    print(f"应用 UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    umap_result = reducer.fit_transform(features)
    print(f"UMAP 完成，形状：{umap_result.shape}")
    return umap_result


def plot_embedding(embedding, labels, save_path, title='Embedding Visualization', 
                   class_names=None, figsize=(10, 8), alpha=0.6, s=50):
    """
    绘制降维可视化图
    
    Args:
        embedding: 降维后的嵌入向量 (N, 2) 或 (N, 3)
        labels: 每个样本的标签
        save_path: 保存路径
        title: 图标题
        class_names: 类别名称列表
        figsize: 图像大小
        alpha: 透明度
        s: 点的大小
    """
    print(f"绘制可视化图并保存到：{save_path}")
    
    # 确定是 2D 还是 3D
    if embedding.shape[1] == 2:
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, 
                            cmap='tab10', alpha=alpha, s=s, edgecolors='w', linewidth=0.5)
        
        if class_names:
            legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend1)
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab10(i), markersize=8) 
                      for i in range(len(class_names))]
            ax.legend(handles=handles, labels=class_names, title="Classes", loc='best')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
    elif embedding.shape[1] == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                            c=labels, cmap='tab10', alpha=alpha, s=s, edgecolors='w', linewidth=0.5)
        
        if class_names:
            legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend1)
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab10(i), markersize=8) 
                      for i in range(len(class_names))]
            ax.legend(handles=handles, labels=class_names, title="Classes", loc='best')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_zlabel('Component 3', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存可视化图到 {save_path}")


def visualize_tsne_umap(
    args,
    model_path,
    data_dir,
    output_dir,
    dataset_type='chestct',
    model_arch='resnet18',
    model_type='without_dropout',
    batch_size=32,
    num_workers=4,
    magnification=None,
    tsne_params=None,
    umap_params=None,
    save_both=True
):
    """
    使用 t-SNE 和 UMAP 可视化特征空间
    
    Args:
        args: 命令行参数
        model_path: 模型参数文件路径
        data_dir: 数据集根目录
        output_dir: 输出目录
        dataset_type: 数据集类型
        model_arch: 模型架构
        model_type: 模型类型
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        magnification: BreakHis 数据集的放大倍数
        tsne_params: t-SNE 参数字典
        umap_params: UMAP 参数字典
        save_both: 是否同时保存 t-SNE 和 UMAP 结果
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
        
        # 移除可能存在的'module.'前缀
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
    
    # 获取数据加载器
    print(f"加载数据集：{dataset_type}, {data_dir}")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224) if dataset_type != 'breakhis' else (448, 448)),
        transforms.ToTensor(),
    ])
    
    # 根据数据集类型设置数据路径
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
    
    # 提取特征
    print("\n开始提取特征...")
    features, labels, paths = extract_features(model, train_loader, device, layer_name='avgpool')
    print(f"特征提取完成，形状：{features.shape}")
       
    # 获取类别名称
    class_names = None
    if hasattr(train_loader.dataset, 'classes'):
        class_names = train_loader.dataset.classes
    else:
        class_names = [str(i) for i in range(num_classes)]
    
    # 应用 t-SNE
    if args.tsne:
        if tsne_params is None:
            tsne_params = {
                'perplexity': 30,
                'n_components': 2,
                'learning_rate': 200,
                'n_iter': 1000,
                'random_state': 42
            }
        
        tsne_result = apply_tsne(features, **tsne_params)
        
        # 保存 t-SNE 结果
        # 设置输出目录 - 参考 vis_saliency.py 的结构
        save_name = args.dataset_type
        if args.dataset_type == 'breakhis':
            save_name += f'_{args.magnification}'
        if args.use_augmentation:
            save_name += f'_{args.strategy}'
        os.makedirs(output_dir, exist_ok=True)
        tsne_save_path = os.path.join(output_dir, f'{save_name}.png')
        title_name = args.dataset_type
        if args.dataset_type == 'breakhis':
            title_name += f' {args.magnification}'
        if args.use_augmentation:
            title_name += f' {args.strategy}'
        else:
            title_name += ' Without Augmentation'
        plot_embedding(
            tsne_result, labels, tsne_save_path,
            title=f't-SNE Visualization - {title_name})',
            class_names=class_names
        )
        
        # 保存 t-SNE 数值结果
        # tsne_npz_path = os.path.join(output_dir, f'tsne_{model_type}_{model_arch}.npz')
        # np.savez(tsne_npz_path, 
        #         embedding=tsne_result, 
        #         labels=labels, 
        #         features=features,
        #         paths=np.array(paths))
        # print(f"  已保存 t-SNE 数值结果到 {tsne_npz_path}")
    
    # 应用 UMAP
    if args.umap:
        if umap_params is None:
            umap_params = {
                'n_neighbors': 15,
                'min_dist': 0.1,
                'n_components': 2,
                'metric': 'euclidean',
                'random_state': 42
            }
        
        umap_result = apply_umap(features, **umap_params)
        
        # 保存 UMAP 结果
        umap_save_path = os.path.join(output_dir, f'umap_{model_type}_{model_arch}.png')
        plot_embedding(
            umap_result, labels, umap_save_path,
            title=f'UMAP Visualization - {model_type} ({model_arch})',
            class_names=class_names
        )
        
        # 保存 UMAP 数值结果
        umap_npz_path = os.path.join(output_dir, f'umap_{model_type}_{model_arch}.npz')
        np.savez(umap_npz_path, 
                embedding=umap_result, 
                labels=labels, 
                features=features,
                paths=np.array(paths))
        print(f"  已保存 UMAP 数值结果到 {umap_npz_path}")
    
    print(f"\n可视化完成！结果保存在：{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用 t-SNE 和 UMAP 可视化特征空间')
    
    # 模型相关参数
    parser.add_argument('--model_path', type=str, help='模型参数文件路径 (.pth)')
    parser.add_argument('--model_arch', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet50', 'resnet34', 'inception', 'wideresnet', 'efficientnet_b0'],
                       help='模型架构')
    parser.add_argument('--model_type', type=str, default='without_dropout',
                       choices=['with_dropout', 'without_dropout'],
                       help='模型类型（是否包含 dropout）')
    parser.add_argument('--use_augmentation', action='store_true', help='是否使用数据增强')
    parser.add_argument('--strategy', type=str, default='uncertaintymixup', 
                       choices=['uncertaintymixup', 'mixup', 'cutmixrand', 'puzzlemix', 'comix', 'guided'])
    
    # 数据集相关参数
    parser.add_argument('--data_dir', type=str, default='/workspace/MedicalImageClassficationData/', 
                       help='数据集根目录')
    parser.add_argument('--dataset_type', type=str, default='chestct',
                       choices=['chestct', 'breakhis', 'padufes', 'kvasir', 'bladder'],
                       help='数据集类型')
    parser.add_argument('--magnification', type=str, default=None,
                       choices=['40', '100', '200', '400'],
                       help='BreakHis 数据集的放大倍数')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./vis_results/feature_results',
                       help='可视化输出目录')
    
    # t-SNE 参数
    parser.add_argument('--tsne',action='store_true', help='是否使用 t-SNE')
    parser.add_argument('--tsne_perplexity', type=int, default=30, help='t-SNE 困惑度')
    parser.add_argument('--tsne_iter', type=int, default=1000, help='t-SNE 迭代次数')
    parser.add_argument('--tsne_lr', type=int, default=200, help='t-SNE 学习率')
    
    # UMAP 参数
    parser.add_argument('--umap',action='store_true', help='是否使用 UMAP')
    parser.add_argument('--umap_neighbors', type=int, default=15, help='UMAP 邻居数')
    parser.add_argument('--umap_min_dist', type=float, default=0.1, help='UMAP 最小距离')
    parser.add_argument('--umap_metric', type=str, default='euclidean', 
                       choices=['euclidean', 'cosine', 'manhattan'],
                       help='UMAP 距离度量')
    
    # 数据加载参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作进程数')
    
    # 批量处理参数
    parser.add_argument('--batch_process', action='store_true', default=True,
                       help='是否批量处理所有数据集和策略')
    
    args = parser.parse_args()
    
    # 设置 t-SNE 和 UMAP 参数
    tsne_params = {
        'perplexity': args.tsne_perplexity,
        'n_components': 2,
        'learning_rate': args.tsne_lr,
        'n_iter': args.tsne_iter,
        'random_state': 42
    }
    
    umap_params = {
        'n_neighbors': args.umap_neighbors,
        'min_dist': args.umap_min_dist,
        'n_components': 2,
        'metric': args.umap_metric,
        'random_state': 42
    }
    
    # 定义要处理的数据集和策略
    magnifications = ['40', '100', '200', '400']
    strategies = ['mixup', 'cutmixrand', 'puzzlemix', 'comix', 'guided', 'uncertaintymixup']
    
    # 设置输出目录
    if args.tsne:
        base_output_dir = os.path.join(args.output_dir, 'tsne')
    elif args.umap:
        base_output_dir = os.path.join(args.output_dir, 'umap')
    else:
        base_output_dir = args.output_dir
    
    if args.batch_process:
        print("=" * 80)
        print("开始批量处理所有数据集和策略的 t-SNE/UMAP 可视化")
        print("=" * 80)
        
        # 处理 chestct 数据集
        print(f"\n{'='*60}")
        print(f"处理 ChestCT 数据集")
        print(f"{'='*60}")
        
        # 不带增强的模型
        args.dataset_type = 'chestct'
        args.use_augmentation = False
        args.model_path = 'best_model_chestct.pth'
        
        print(f"\n处理无增强模型...")
        visualize_tsne_umap(
            args,
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=base_output_dir,
            dataset_type=args.dataset_type,
            model_arch=args.model_arch,
            model_type=args.model_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            magnification=args.magnification,
            tsne_params=tsne_params,
            umap_params=umap_params,
            save_both=True
        )
        
        # 带增强的各个策略
        for s in strategies:
            args.strategy = s
            args.use_augmentation = True
            args.model_path = f'best_model_{s}_chestct.pth'
            
            print(f"\n处理策略：{s}")
            visualize_tsne_umap(
                args,
                model_path=args.model_path,
                data_dir=args.data_dir,
                output_dir=base_output_dir,
                dataset_type=args.dataset_type,
                model_arch=args.model_arch,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                magnification=args.magnification,
                tsne_params=tsne_params,
                umap_params=umap_params,
                save_both=True
            )
        
        # 处理 breakhis 数据集的各个放大倍数
        for m in magnifications:
            print(f"\n{'='*60}")
            print(f"处理 BreakHis {m}x 数据集")
            print(f"{'='*60}")
            
            args.magnification = m
            args.dataset_type = 'breakhis'
            
            # 不带增强的模型
            args.use_augmentation = False
            args.model_path = f'best_model_breakhis_{m}.pth'
            
            print(f"\n处理无增强模型...")
            visualize_tsne_umap(
                args,
                model_path=args.model_path,
                data_dir=args.data_dir,
                output_dir=base_output_dir,
                dataset_type=args.dataset_type,
                model_arch=args.model_arch,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                magnification=args.magnification,
                tsne_params=tsne_params,
                umap_params=umap_params,
                save_both=True
            )
            
            # 带增强的各个策略
            args.use_augmentation = True
            for s in strategies:
                args.strategy = s
                args.model_path = f'best_model_{s}_breakhis_{m}.pth'
                
                print(f"\n处理策略：{s}")
                visualize_tsne_umap(
                    args,
                    model_path=args.model_path,
                    data_dir=args.data_dir,
                    output_dir=base_output_dir,
                    dataset_type=args.dataset_type,
                    model_arch=args.model_arch,
                    model_type=args.model_type,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    magnification=args.magnification,
                    tsne_params=tsne_params,
                    umap_params=umap_params,
                    save_both=True
                )
        
        # 处理 padufes 数据集
        print(f"\n{'='*60}")
        print(f"处理 PAD-UFES-20 数据集")
        print(f"{'='*60}")
        
        args.dataset_type = 'padufes'
        args.magnification = None
        
        # 不带增强的模型
        args.use_augmentation = False
        args.model_path = 'best_model_padufes.pth'
        
        print(f"\n处理无增强模型...")
        visualize_tsne_umap(
            args,
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=base_output_dir,
            dataset_type=args.dataset_type,
            model_arch=args.model_arch,
            model_type=args.model_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            magnification=args.magnification,
            tsne_params=tsne_params,
            umap_params=umap_params,
            save_both=True
        )
        
        # 带增强的各个策略
        args.use_augmentation = True
        for s in strategies:
            args.strategy = s
            args.model_path = f'best_model_{s}_padufes.pth'
            
            print(f"\n处理策略：{s}")
            visualize_tsne_umap(
                args,
                model_path=args.model_path,
                data_dir=args.data_dir,
                output_dir=base_output_dir,
                dataset_type=args.dataset_type,
                model_arch=args.model_arch,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                magnification=args.magnification,
                tsne_params=tsne_params,
                umap_params=umap_params,
                save_both=True
            )
        
        # 处理 kvasir 数据集
        print(f"\n{'='*60}")
        print(f"处理 Kvasir 数据集")
        print(f"{'='*60}")
        
        args.dataset_type = 'kvasir'
        args.magnification = None
        
        # 不带增强的模型
        args.use_augmentation = False
        args.model_path = 'best_model_kvasir.pth'
        
        print(f"\n处理无增强模型...")
        visualize_tsne_umap(
            args,
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=base_output_dir,
            dataset_type=args.dataset_type,
            model_arch=args.model_arch,
            model_type=args.model_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            magnification=args.magnification,
            tsne_params=tsne_params,
            umap_params=umap_params,
            save_both=True
        )
        
        # 带增强的各个策略
        args.use_augmentation = True
        for s in strategies:
            args.strategy = s
            args.model_path = f'best_model_{s}_kvasir.pth'
            
            print(f"\n处理策略：{s}")
            visualize_tsne_umap(
                args,
                model_path=args.model_path,
                data_dir=args.data_dir,
                output_dir=base_output_dir,
                dataset_type=args.dataset_type,
                model_arch=args.model_arch,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                magnification=args.magnification,
                tsne_params=tsne_params,
                umap_params=umap_params,
                save_both=True
            )
        
        # 处理 bladder 数据集
        print(f"\n{'='*60}")
        print(f"处理 Bladder 数据集")
        print(f"{'='*60}")
        
        args.dataset_type = 'bladder'
        args.magnification = None
        
        # 不带增强的模型
        args.use_augmentation = False
        args.model_path = 'best_model_bladder.pth'
        
        print(f"\n处理无增强模型...")
        visualize_tsne_umap(
            args,
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=base_output_dir,
            dataset_type=args.dataset_type,
            model_arch=args.model_arch,
            model_type=args.model_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            magnification=args.magnification,
            tsne_params=tsne_params,
            umap_params=umap_params,
            save_both=True
        )
        
        # 带增强的各个策略
        args.use_augmentation = True
        for s in strategies:
            args.strategy = s
            args.model_path = f'best_model_{s}_bladder.pth'
            
            print(f"\n处理策略：{s}")
            visualize_tsne_umap(
                args,
                model_path=args.model_path,
                data_dir=args.data_dir,
                output_dir=base_output_dir,
                dataset_type=args.dataset_type,
                model_arch=args.model_arch,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                magnification=args.magnification,
                tsne_params=tsne_params,
                umap_params=umap_params,
                save_both=True
            )
        
        print(f"\n{'='*60}")
        print(f"所有可视化完成！")
        print(f"{'='*60}\n")
    else:
        # 单个处理模式
        # 设置模型路径
        args.model_path = 'best_model'
        if args.use_augmentation:
            args.model_path += args.strategy
        args.model_path += f'_{args.dataset_type}'
        if args.dataset_type == 'breakhis':
            args.model_path += f'_{args.magnification}'
        args.model_path += '.pth'
        
        if args.tsne:
            args.output_dir = os.path.join(args.output_dir, 'tsne')
        if args.umap:
            args.output_dir = os.path.join(args.output_dir, 'umap')
        
        # 调用主函数
        print(f"\n{'='*60}")
        print(f"开始 t-SNE 和 UMAP 可视化")
        print(f"{'='*60}\n")
        
        visualize_tsne_umap(
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
            tsne_params=tsne_params,
            umap_params=umap_params,
            save_both=True
        )
        
        print(f"\n{'='*60}")
        print(f"可视化完成！")
        print(f"{'='*60}\n")
