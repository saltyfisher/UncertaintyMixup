import torch
import gco
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import random
import os
from models import get_target_layer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def denormalize_image(image):
    """反归一化图像，将图像值范围从[0,1]转换为[0,255]"""
    # 确保图像值在[0,1]范围内
    image = np.clip(image, 0, 1)
    # 转换为[0,255]范围
    image = (image * 255).astype(np.uint8)
    return image


def mixup_data(x, y, alpha=1.0, device='cpu'):
    """标准Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    """生成随机边界框用于CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0, device='cpu'):
    """CutMix数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def get_mc_dropout_activations(model, images, targets, num_mc_samples=10):
    """使用MC Dropout获取多次类激活图"""
    # 保存模型的原始训练状态
    original_training_state = model.training
    
    # 临时启用训练模式并激活dropout
    model.train()
    model.enable_dropout()
    
    # 创建LayerCAM实例
    target_layers = [get_target_layer(model, 'with_dropout')]
    cam = LayerCAM(model=model, target_layers=target_layers)
    
    all_grayscale_cams = []
    
    # 获取MC样本
    for _ in range(num_mc_samples):
        # 确保输入张量需要梯度
        input_tensor = images.clone()
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.detach().requires_grad_(True)
            
        # 为整个批次创建目标
        targets_for_cam = [ClassifierOutputTarget(target.item()) for target in targets]
        
        # 批量处理获取类激活图
        grayscale_cam_batch = cam(input_tensor=input_tensor, targets=targets_for_cam)
        
        # 添加到结果中
        all_grayscale_cams.append(grayscale_cam_batch)
    
    # 转换为numpy数组
    all_grayscale_cams = np.array(all_grayscale_cams)
    
    # 恢复模型的原始训练状态并禁用dropout
    if not original_training_state:
        model.eval()
        model.disable_dropout()
    
    return all_grayscale_cams


def get_high_activation_areas_from_mc_dropout(all_grayscale_cams, activation_threshold=0.5, mean_threshold=0.7):
    """从MC Dropout的多次类激活图中选择重叠的高响应区域"""
    # 注意：这个函数不接收 dataset_type 参数，因为 mask 的生成与数据集类型无关
    #       mask 的尺寸适配在 paste_activation_regions 中完成
    high_activation_masks = []
    
    # 遍历每个样本
    for sample_idx in range(all_grayscale_cams.shape[1]):
        # 收集该样本的所有激活图
        sample_cams = all_grayscale_cams[:, sample_idx, :, :]
        
        # 计算所有激活图的平均值
        mean_cam = np.mean(sample_cams, axis=0)
        
        # 随机采样决定是否为该样本生成增广样本
        if np.random.random() > activation_threshold:
            # 生成空的mask（不进行增广）
            mask = np.zeros_like(mean_cam, dtype=np.uint8)
            high_activation_masks.append(mask)
            continue
        
        # 使用均值阈值过滤类激活图，小于该均值的设置为0
        threshold = np.max(mean_cam) * mean_threshold
        filtered_cam = np.where(mean_cam >= threshold, mean_cam, 0)
        
        # 选出面积最大的连通区域
        # 使用OpenCV查找连通组件
        _, labels, stats, centroids = cv2.connectedComponentsWithStats((filtered_cam > 0).astype(np.uint8), connectivity=8)
        
        # 如果没有连通组件，创建空的mask
        if labels.max() == 0:
            mask = np.zeros_like(mean_cam, dtype=np.uint8)
            high_activation_masks.append(mask)
            continue
        
        # 找到面积最大的连通组件（忽略背景）
        largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        
        # 创建mask，只保留最大连通区域
        mask = np.where(labels == largest_component, 1, 0).astype(np.uint8)
        
        high_activation_masks.append(mask)
    
    return high_activation_masks


def calculate_mixup_ratio(mask):
    """根据粘贴面积与原始图片的比值计算混合率"""
    # 计算mask中非零像素的数量
    masked_area = np.count_nonzero(mask)
    # 计算总像素数
    total_area = mask.shape[0] * mask.shape[1]
    # 返回混合率
    return masked_area / total_area if total_area > 0 else 0


def create_smooth_mask(mask, sigma=5):
    """创建平滑的掩码，使用高斯滤波器平滑边界"""
    from scipy.ndimage import gaussian_filter
    # 对mask进行高斯滤波以获得平滑边界
    smooth_mask = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    # 归一化到[0, 1]范围
    if smooth_mask.max() > 0:
        smooth_mask = smooth_mask / smooth_mask.max()
    return smooth_mask


def paste_activation_regions(images, targets, high_activation_masks, dataset_type='chestct'):
    """将高激活区域以SmoothMix方式粘贴到其他图像上，并计算混合率"""
    batch_size = images.size(0)
    mixed_images = images.clone()
    mixup_ratios = []  # 存储每个样本的混合率
    mixed_pairs = []  # 存储混合对的信息
    
    # 记录已经被用作混合目标的索引，避免混合后的数据再次被混合
    used_targets = set()
    
    # 根据数据集类型设置resize尺寸
    if dataset_type == 'breakhis':
        resize_size = (450, 450)
    else:  # chestct
        # resize_size = (320, 320)
        resize_size = (224, 224)
    
    for i in range(batch_size):
        # 检查mask是否为空（全零），如果为空则不进行混合
        if not np.any(high_activation_masks[i]):
            # 不进行混合，混合率为0
            mixup_ratios.append(0.0)
            mixed_pairs.append(None)
            continue
            
        # 随机选择一个不同类别的图像，且该图像未被用作混合目标
        other_classes = [j for j in range(batch_size) if targets[j] != targets[i] and j not in used_targets]
        if not other_classes:
            # 如果没有不同类别的图像，则不进行混合，混合率为0
            mixup_ratios.append(0.0)
            mixed_pairs.append(None)
            continue
            
        j = random.choice(other_classes)
        
        # 标记该目标已被使用，避免再次被选中
        used_targets.add(j)
        
        # 获取高激活区域mask
        mask_i = high_activation_masks[i]
        mask_i = cv2.resize(mask_i, resize_size)  # 根据数据集类型设置尺寸匹配
        
        # 创建平滑的mask用于SmoothMix
        smooth_mask_i = create_smooth_mask(mask_i, sigma=7)  # sigma控制平滑程度
        
        # 计算混合率
        ratio = calculate_mixup_ratio(mask_i)
        mixup_ratios.append(ratio)
        
        # 保存混合对信息
        mixed_pairs.append({
            'source_idx': i,
            'target_idx': j,
            'mask': mask_i,
            'smooth_mask': smooth_mask_i,
            'ratio': ratio
        })
        
        # 将图像转换为numpy进行处理
        img_i = images[i].cpu().numpy().transpose(1, 2, 0)
        img_j = mixed_images[j].cpu().numpy().transpose(1, 2, 0)  # 使用mixed_images而不是images
        
        # 使用平滑mask进行混合
        # 扩展平滑mask到3通道
        smooth_mask_3ch = np.stack([smooth_mask_i, smooth_mask_i, smooth_mask_i], axis=2)
        
        # 以平滑方式混合图像
        mixed_img = img_i * smooth_mask_3ch + img_j * (1 - smooth_mask_3ch)
        
        # 将混合后的图像放回mixed_images
        mixed_images[j] = torch.tensor(
            mixed_img.transpose(2, 0, 1),
            dtype=torch.float32,
            device=images.device  # 确保在同一设备上
        )
    
    return mixed_images, mixup_ratios, mixed_pairs


def save_mixed_results(images, mixed_images, all_grayscale_cams, mixed_pairs, epoch, batch_idx, save_dir, dataset_type='chestct', model=None):
    """保存混合结果的可视化"""
    # 创建保存目录
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # 只保存一个混合对（如果有混合对的话）
    saved_one = False
    for pair_idx, pair_info in enumerate(mixed_pairs):
        if pair_info is not None and not saved_one:
            source_idx = pair_info['source_idx']
            target_idx = pair_info['target_idx']
            mask = pair_info['mask']
            ratio = pair_info['ratio']
            
            # 获取源图像和目标图像
            source_img = images[source_idx].cpu().numpy().transpose(1, 2, 0)
            target_img = images[target_idx].cpu().numpy().transpose(1, 2, 0)
            mixed_img = mixed_images[target_idx].cpu().numpy().transpose(1, 2, 0)
            
            # 获取MC Dropout类激活图
            cam_source_mc = all_grayscale_cams[0][source_idx]  # 使用第一次MC采样的CAM
            cam_target_mc = all_grayscale_cams[0][target_idx]
            
            # 获取无Dropout类激活图（如果提供了模型）
            cam_source_no_dropout = None
            cam_target_no_dropout = None
            if model is not None:
                # 临时禁用dropout
                model.eval()
                model.disable_dropout()
                
                # 创建LayerCAM实例
                target_layers = [get_target_layer(model, 'with_dropout')]
                cam_no_dropout = LayerCAM(model=model, target_layers=target_layers)
                
                # 获取源图像的无Dropout类激活图
                input_tensor_source = images[source_idx].unsqueeze(0)
                targets_source = [ClassifierOutputTarget(pair_info.get('source_target', 0))]
                cam_source_no_dropout = cam_no_dropout(input_tensor=input_tensor_source)[0]
                
                # 获取目标图像的无Dropout类激活图
                input_tensor_target = images[target_idx].unsqueeze(0)
                targets_target = [ClassifierOutputTarget(pair_info.get('target_target', 0))]
                cam_target_no_dropout = cam_no_dropout(input_tensor=input_tensor_target)[0]
                
                # 恢复模型状态（启用dropout）
                model.train()
                model.enable_dropout()
            
            # 反归一化图像
            source_img = denormalize_image(source_img)
            target_img = denormalize_image(target_img)
            mixed_img = denormalize_image(mixed_img)
            
            # 创建可视化图 (2行4列)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
            # 根据数据集类型处理图像显示
            if dataset_type == 'chestct':
                # ChestCT是灰度图数据集
                axes[0, 0].imshow(source_img, cmap='gray')
                axes[0, 0].set_title(f'Source Image (Class: {source_idx})')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(target_img, cmap='gray')
                axes[0, 1].set_title(f'Target Image (Class: {target_idx})')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(mask, cmap='gray')
                axes[0, 2].set_title(f'Activation Mask (Ratio: {ratio:.3f})')
                axes[0, 2].axis('off')
                
                # 第二行：混合结果和类激活图
                axes[1, 0].imshow(mixed_img, cmap='gray')
                axes[1, 0].set_title('Mixed Image')
                axes[1, 0].axis('off')
                
                # 将CAM叠加到图像上（需要将灰度图转换为RGB格式）
                source_img_rgb = np.stack([source_img[:, :, 0]] * 3, axis=-1) if len(source_img.shape) == 3 else np.stack([source_img] * 3, axis=-1)
                target_img_rgb = np.stack([target_img[:, :, 0]] * 3, axis=-1) if len(target_img.shape) == 3 else np.stack([target_img] * 3, axis=-1)
                
                cam_on_source_mc = show_cam_on_image(source_img_rgb.astype(np.float32) / 255.0, cam_source_mc, use_rgb=True)
                cam_on_target_mc = show_cam_on_image(target_img_rgb.astype(np.float32) / 255.0, cam_target_mc, use_rgb=True)
                
                axes[1, 1].imshow(cam_on_source_mc)
                axes[1, 1].set_title('Source CAM (MC Dropout)')
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(cam_on_target_mc)
                axes[1, 2].set_title('Target CAM (MC Dropout)')
                axes[1, 2].axis('off')
                
                # 显示无Dropout类激活图或方差图
                if cam_source_no_dropout is not None and cam_target_no_dropout is not None:
                    cam_on_source_no_dropout = show_cam_on_image(source_img_rgb.astype(np.float32) / 255.0, cam_source_no_dropout, use_rgb=True)
                    cam_on_target_no_dropout = show_cam_on_image(target_img_rgb.astype(np.float32) / 255.0, cam_target_no_dropout, use_rgb=True)
                    
                    axes[0, 3].imshow(cam_on_source_no_dropout)
                    axes[0, 3].set_title('Source CAM (No Dropout)')
                    axes[0, 3].axis('off')
                    
                    axes[1, 3].imshow(cam_on_target_no_dropout)
                    axes[1, 3].set_title('Target CAM (No Dropout)')
                    axes[1, 3].axis('off')
                else:
                    # 如果没有提供模型，则显示方差图
                    sample_cams_source = all_grayscale_cams[:, source_idx, :, :]
                    sample_cams_target = all_grayscale_cams[:, target_idx, :, :]
                    cam_variance_source = np.var(sample_cams_source, axis=0)
                    cam_variance_target = np.var(sample_cams_target, axis=0)
                    
                    axes[0, 3].imshow(cam_variance_source, cmap='jet')
                    axes[0, 3].set_title('Source CAM Variance')
                    axes[0, 3].axis('off')
                    
                    axes[1, 3].imshow(cam_variance_target, cmap='jet')
                    axes[1, 3].set_title('Target CAM Variance')
                    axes[1, 3].axis('off')
            else:
                # BreakHis是彩色图像数据集
                axes[0, 0].imshow(source_img)
                axes[0, 0].set_title(f'Source Image (Class: {source_idx})')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(target_img)
                axes[0, 1].set_title(f'Target Image (Class: {target_idx})')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(mask, cmap='gray')
                axes[0, 2].set_title(f'Activation Mask (Ratio: {ratio:.3f})')
                axes[0, 2].axis('off')
                
                # 第二行：混合结果和类激活图
                axes[1, 0].imshow(mixed_img)
                axes[1, 0].set_title('Mixed Image')
                axes[1, 0].axis('off')
                
                # 将CAM叠加到图像上
                cam_on_source_mc = show_cam_on_image(source_img.astype(np.float32) / 255.0, cam_source_mc, use_rgb=True)
                cam_on_target_mc = show_cam_on_image(target_img.astype(np.float32) / 255.0, cam_target_mc, use_rgb=True)
                
                axes[1, 1].imshow(cam_on_source_mc)
                axes[1, 1].set_title('Source CAM (MC Dropout)')
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(cam_on_target_mc)
                axes[1, 2].set_title('Target CAM (MC Dropout)')
                axes[1, 2].axis('off')
                
                # 显示无Dropout类激活图或方差图
                if cam_source_no_dropout is not None and cam_target_no_dropout is not None:
                    cam_on_source_no_dropout = show_cam_on_image(source_img.astype(np.float32) / 255.0, cam_source_no_dropout, use_rgb=True)
                    cam_on_target_no_dropout = show_cam_on_image(target_img.astype(np.float32) / 255.0, cam_target_no_dropout, use_rgb=True)
                    
                    axes[0, 3].imshow(cam_on_source_no_dropout)
                    axes[0, 3].set_title('Source CAM (No Dropout)')
                    axes[0, 3].axis('off')
                    
                    axes[1, 3].imshow(cam_on_target_no_dropout)
                    axes[1, 3].set_title('Target CAM (No Dropout)')
                    axes[1, 3].axis('off')
                else:
                    # 如果没有提供模型，则显示方差图
                    sample_cams_source = all_grayscale_cams[:, source_idx, :, :]
                    sample_cams_target = all_grayscale_cams[:, target_idx, :, :]
                    cam_variance_source = np.var(sample_cams_source, axis=0)
                    cam_variance_target = np.var(sample_cams_target, axis=0)
                    
                    axes[0, 3].imshow(cam_variance_source, cmap='jet')
                    axes[0, 3].set_title('Source CAM Variance')
                    axes[0, 3].axis('off')
                    
                    axes[1, 3].imshow(cam_variance_target, cmap='jet')
                    axes[1, 3].set_title('Target CAM Variance')
                    axes[1, 3].axis('off')
            
            # 保存图像
            save_path = os.path.join(epoch_dir, f'batch_{batch_idx}_mixed_result.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # 标记已保存一个混合结果
            saved_one = True
            
            # 只保存一个混合结果，所以跳出循环
            break
    
    # 如果没有任何混合对，保存原始图像示例
    if not saved_one and len(images) > 0:
        # 保存第一个图像作为示例
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 获取第一个图像
        first_img = images[0].cpu().numpy().transpose(1, 2, 0)
        first_img = denormalize_image(first_img)
        
        if dataset_type == 'chestct':
            axes[0].imshow(first_img, cmap='gray')
            axes[0].set_title('Sample Image')
            axes[0].axis('off')
            
            # 显示该图像的CAM（如果可用）
            if len(all_grayscale_cams) > 0 and len(all_grayscale_cams[0]) > 0:
                cam_first = all_grayscale_cams[0][0]
                # 将灰度图转换为RGB格式
                first_img_rgb = np.stack([first_img[:, :, 0]] * 3, axis=-1) if len(first_img.shape) == 3 else np.stack([first_img] * 3, axis=-1)
                cam_on_first = show_cam_on_image(first_img_rgb.astype(np.float32) / 255.0, cam_first, use_rgb=True)
                axes[1].imshow(cam_on_first)
                axes[1].set_title('Sample CAM (MC Dropout)')
                axes[1].axis('off')
                
                # 显示无Dropout类激活图或方差图
                if model is not None:
                    # 临时禁用dropout
                    model.eval()
                    model.disable_dropout()
                    
                    # 创建LayerCAM实例
                    target_layers = [get_target_layer(model, 'with_dropout')]
                    cam_no_dropout = LayerCAM(model=model, target_layers=target_layers)
                    
                    # 获取无Dropout类激活图
                    input_tensor = images[0].unsqueeze(0)
                    targets = [ClassifierOutputTarget(0)]
                    cam_first_no_dropout = cam_no_dropout(input_tensor=input_tensor)[0]
                    
                    cam_on_first_no_dropout = show_cam_on_image(first_img_rgb.astype(np.float32) / 255.0, cam_first_no_dropout, use_rgb=True)
                    axes[2].imshow(cam_on_first_no_dropout)
                    axes[2].set_title('Sample CAM (No Dropout)')
                    axes[2].axis('off')
                    
                    # 恢复模型状态（启用dropout）
                    model.train()
                    model.enable_dropout()
                else:
                    # 显示方差图
                    sample_cams_first = all_grayscale_cams[:, 0, :, :]
                    cam_variance_first = np.var(sample_cams_first, axis=0)
                    axes[2].imshow(cam_variance_first, cmap='jet')
                    axes[2].set_title('Sample CAM Variance')
                    axes[2].axis('off')
            else:
                axes[1].axis('off')
                axes[2].axis('off')
        else:
            axes[0].imshow(first_img)
            axes[0].set_title('Sample Image')
            axes[0].axis('off')
            
            # 显示该图像的CAM（如果可用）
            if len(all_grayscale_cams) > 0 and len(all_grayscale_cams[0]) > 0:
                cam_first = all_grayscale_cams[0][0]
                cam_on_first = show_cam_on_image(first_img.astype(np.float32) / 255.0, cam_first, use_rgb=True)
                axes[1].imshow(cam_on_first)
                axes[1].set_title('Sample CAM (MC Dropout)')
                axes[1].axis('off')
                
                # 显示无Dropout类激活图或方差图
                if model is not None:
                    # 临时禁用dropout
                    model.eval()
                    model.disable_dropout()
                    
                    # 创建LayerCAM实例
                    target_layers = [get_target_layer(model, 'with_dropout')]
                    cam_no_dropout = LayerCAM(model=model, target_layers=target_layers)
                    
                    # 获取无Dropout类激活图
                    input_tensor = images[0].unsqueeze(0)
                    targets = [ClassifierOutputTarget(0)]
                    cam_first_no_dropout = cam_no_dropout(input_tensor=input_tensor)[0]
                    
                    cam_on_first_no_dropout = show_cam_on_image(first_img.astype(np.float32) / 255.0, cam_first_no_dropout, use_rgb=True)
                    axes[2].imshow(cam_on_first_no_dropout)
                    axes[2].set_title('Sample CAM (No Dropout)')
                    axes[2].axis('off')
                    
                    # 恢复模型状态（启用dropout）
                    model.train()
                    model.enable_dropout()
                else:
                    # 显示方差图
                    sample_cams_first = all_grayscale_cams[:, 0, :, :]
                    cam_variance_first = np.var(sample_cams_first, axis=0)
                    axes[2].imshow(cam_variance_first, cmap='jet')
                    axes[2].set_title('Sample CAM Variance')
                    axes[2].axis('off')
            else:
                axes[1].axis('off')
                axes[2].axis('off')
        
        save_path = os.path.join(epoch_dir, f'batch_{batch_idx}_sample.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def to_one_hot(inp, num_classes, device='cuda'):
    '''one-hot label'''
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    return y_onehot


def cost_matrix(width, device='cuda'):
    '''transport cost'''
    C = np.zeros([width**2, width**2], dtype=np.float32)

    for m_i in range(width**2):
        i1 = m_i // width
        j1 = m_i % width
        for m_j in range(width**2):
            i2 = m_j // width
            j2 = m_j % width
            C[m_i, m_j] = abs(i1 - i2)**2 + abs(j1 - j2)**2

    C = C / (width - 1)**2
    C = torch.tensor(C)
    if device == 'cuda':
        C = C.cuda()

    return C


cost_matrix_dict = {
    '2': cost_matrix(2, 'cuda').unsqueeze(0),
    '4': cost_matrix(4, 'cuda').unsqueeze(0),
    '8': cost_matrix(8, 'cuda').unsqueeze(0),
    '16': cost_matrix(16, 'cuda').unsqueeze(0)
}


def mixup_process(out,
                  target_reweighted,
                  hidden=0,
                  args=None,
                  grad=None,
                  noise=None,
                  adv_mask1=0,
                  adv_mask2=0,
                  mp=None,
                  visualize=False,
                  epoch=None,
                  batch=None,
                  vis_dir=None):
    '''various mixup process'''
    if args is not None:
        mixup_alpha = args.mixup_alpha
        in_batch = args.in_batch
        mean = args.mean
        std = args.std
        box = args.box
        graph = args.graph
        beta = args.beta
        gamma = args.gamma
        eta = args.eta
        neigh_size = args.neigh_size
        n_labels = args.n_labels
        transport = args.transport
        t_eps = args.t_eps
        t_size = args.t_size

    block_num = 2**np.random.randint(1, 5)
    indices = np.random.permutation(out.size(0))

    lam = get_lambda(mixup_alpha)

    if hidden:
        # Manifold Mixup
        out = out * lam + out[indices] * (1 - lam)
        ratio = torch.ones(out.shape[0], device='cuda') * lam
    else:
        if box:
            # CutMix
            out, ratio = mixup_box(out, out[indices], alpha=lam, device='cuda')
        elif graph:
            # PuzzleMix
            if block_num > 1:
                out, ratio = mixup_graph(out,
                                         grad,
                                         indices,
                                         block_num=block_num,
                                         alpha=lam,
                                         beta=beta,
                                         gamma=gamma,
                                         eta=eta,
                                         neigh_size=neigh_size,
                                         n_labels=n_labels,
                                         mean=mean,
                                         std=std,
                                         transport=transport,
                                         t_eps=t_eps,
                                         t_size=t_size,
                                         noise=noise,
                                         adv_mask1=adv_mask1,
                                         adv_mask2=adv_mask2,
                                         mp=mp,
                                         device='cuda',
                                         visualize=visualize,
                                         epoch=epoch,
                                         batch=batch,
                                         vis_dir=vis_dir)
            else:
                ratio = torch.ones(out.shape[0], device='cuda')
        else:
            # Input Mixup
            out = out * lam + out[indices] * (1 - lam)
            ratio = torch.ones(out.shape[0], device='cuda') * lam

    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * ratio.unsqueeze(-1) + target_shuffled_onehot * (
        1 - ratio.unsqueeze(-1))

    return out, target_reweighted


def get_lambda(alpha=1.0, alpha2=None):
    '''Return lambda'''
    if alpha > 0.:
        if alpha2 is None:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = np.random.beta(alpha + 1e-2, alpha2 + 1e-2)
    else:
        lam = 1.
    return lam


def graphcut_multi(unary1, unary2, pw_x, pw_y, alpha, beta, eta, n_labels=2, eps=1e-8):
    '''alpha-beta swap algorithm'''
    block_num = unary1.shape[0]

    large_val = 1000 * block_num**2

    if n_labels == 2:
        prior = np.array([-np.log(alpha + eps), -np.log(1 - alpha + eps)])
    elif n_labels == 3:
        prior = np.array([
            -np.log(alpha**2 + eps), -np.log(2 * alpha * (1 - alpha) + eps),
            -np.log((1 - alpha)**2 + eps)
        ])
    elif n_labels == 4:
        prior = np.array([
            -np.log(alpha**3 + eps), -np.log(3 * alpha**2 * (1 - alpha) + eps),
            -np.log(3 * alpha * (1 - alpha)**2 + eps), -np.log((1 - alpha)**3 + eps)
        ])

    prior = eta * prior / block_num**2
    unary_cost = (large_val * np.stack([(1 - lam) * unary1 + lam * unary2 + prior[i]
                                        for i, lam in enumerate(np.linspace(0, 1, n_labels))],
                                       axis=-1)).astype(np.int32)
    pairwise_cost = np.zeros(shape=[n_labels, n_labels], dtype=np.float32)

    for i in range(n_labels):
        for j in range(n_labels):
            pairwise_cost[i, j] = (i - j)**2 / (n_labels - 1)**2

    pw_x = (large_val * (pw_x + beta)).astype(np.int32)
    pw_y = (large_val * (pw_y + beta)).astype(np.int32)
    labels = 1.0 - gco.cut_grid_graph(unary_cost, pairwise_cost, pw_x, pw_y,
                                      algorithm='swap') / (n_labels - 1)
    mask = labels.reshape(block_num, block_num)

    return mask


def neigh_penalty(input1, input2, k):
    '''data local smoothness term'''
    pw_x = input1[:, :, :-1, :] - input2[:, :, 1:, :]
    pw_y = input1[:, :, :, :-1] - input2[:, :, :, 1:]

    pw_x = pw_x[:, :, k - 1::k, :]
    pw_y = pw_y[:, :, :, k - 1::k]

    pw_x = F.avg_pool2d(pw_x.abs().mean(1), kernel_size=(1, k))
    pw_y = F.avg_pool2d(pw_y.abs().mean(1), kernel_size=(k, 1))

    return pw_x, pw_y


def mixup_box(input1, input2, alpha=0.5, device='cuda'):
    '''CutMix'''
    batch_size, _, height, width = input1.shape
    ratio = np.zeros([batch_size])

    rx = np.random.uniform(0, height)
    ry = np.random.uniform(0, width)
    rh = np.sqrt(1 - alpha) * height
    rw = np.sqrt(1 - alpha) * width
    x1 = int(np.clip(rx - rh / 2, a_min=0., a_max=height))
    x2 = int(np.clip(rx + rh / 2, a_min=0., a_max=height))
    y1 = int(np.clip(ry - rw / 2, a_min=0., a_max=width))
    y2 = int(np.clip(ry + rw / 2, a_min=0., a_max=width))
    input1[:, :, x1:x2, y1:y2] = input2[:, :, x1:x2, y1:y2]
    ratio += 1 - (x2 - x1) * (y2 - y1) / (width * height)

    ratio = torch.tensor(ratio, dtype=torch.float32)
    if device == 'cuda':
        ratio = ratio.cuda()

    return input1, ratio


def mixup_graph(input1,
                grad1,
                indices,
                block_num=2,
                alpha=0.5,
                beta=0.,
                gamma=0.,
                eta=0.2,
                neigh_size=2,
                n_labels=2,
                mean=None,
                std=None,
                transport=False,
                t_eps=10.0,
                t_size=16,
                noise=None,
                adv_mask1=0,
                adv_mask2=0,
                device='cuda',
                mp=None,
                visualize=False,
                epoch=None,
                batch=None,
                vis_dir=None):
    '''Puzzle Mix'''
    input2 = input1[indices].clone()

    batch_size, _, _, width = input1.shape
    block_size = width // block_num
    neigh_size = min(neigh_size, block_size)
    t_size = min(t_size, block_size)

    # normalize
    beta = beta / block_num / 16

    # unary term
    grad1_pool = F.avg_pool2d(grad1, block_size)
    unary1_torch = grad1_pool / grad1_pool.reshape(batch_size, -1).sum(1).reshape(batch_size, 1, 1)
    unary2_torch = unary1_torch[indices]

    # calculate pairwise terms
    input1_pool = F.avg_pool2d(input1 * std + mean, neigh_size)
    input2_pool = input1_pool[indices]

    # 计算k值
    k = max(1, block_size // neigh_size)  # 确保k至少为1
    
    # 先调用neigh_penalty获取实际的尺寸
    pw_x_00, pw_y_00 = neigh_penalty(input2_pool, input2_pool, k)
    pw_x_01, pw_y_01 = neigh_penalty(input2_pool, input1_pool, k)
    pw_x_10, pw_y_10 = neigh_penalty(input1_pool, input2_pool, k)
    pw_x_11, pw_y_11 = neigh_penalty(input1_pool, input1_pool, k)
    
    # 根据实际返回的尺寸创建pw_x和pw_y
    batch_size_actual = pw_x_00.shape[0]
    pw_x = torch.zeros([batch_size_actual, 2, 2, pw_x_00.shape[-2], pw_x_00.shape[-1]], device=device)
    pw_y = torch.zeros([batch_size_actual, 2, 2, pw_y_00.shape[-2], pw_y_00.shape[-1]], device=device)

    if k > 0:
        pw_x[:, 0, 0], pw_y[:, 0, 0] = pw_x_00, pw_y_00
        pw_x[:, 0, 1], pw_y[:, 0, 1] = pw_x_01, pw_y_01
        pw_x[:, 1, 0], pw_y[:, 1, 0] = pw_x_10, pw_y_10
        pw_x[:, 1, 1], pw_y[:, 1, 1] = pw_x_11, pw_y_11

    pw_x = beta * gamma * pw_x
    pw_y = beta * gamma * pw_y

    # re-define unary and pairwise terms to draw graph
    unary1 = unary1_torch.clone()
    unary2 = unary2_torch.clone()

    target_h, target_w = unary1.shape[-2], unary1.shape[-1]

    # 调整pw_x和pw_y的尺寸以匹配unary张量
    if pw_x.shape[-2] != target_h - 1 or pw_x.shape[-1] != target_w:
        # 保持前3个维度不变，只对最后2个空间维度进行处理
        pw_x_reshaped = pw_x.view(pw_x.shape[0], -1, pw_x.shape[-2], pw_x.shape[-1])
        target_size = (target_h - 1, target_w)
        
        # 如果当前尺寸大于目标尺寸，则裁剪；如果小于目标尺寸，则插值
        if pw_x_reshaped.shape[-2] > target_size[0] or pw_x_reshaped.shape[-1] > target_size[1]:
            # 裁剪到目标尺寸
            pw_x_reshaped = pw_x_reshaped[:, :, :target_size[0], :target_size[1]]
        elif pw_x_reshaped.shape[-2] < target_size[0] or pw_x_reshaped.shape[-1] < target_size[1]:
            # 插值到目标尺寸
            pw_x_reshaped = F.interpolate(pw_x_reshaped, size=target_size, mode='bilinear', align_corners=False)
        
        pw_x = pw_x_reshaped.view(pw_x.shape[0], pw_x.shape[1], pw_x.shape[2], target_h - 1, target_w)

    # 调整pw_y的尺寸以匹配unary张量
    if pw_y.shape[-2] != target_h or pw_y.shape[-1] != target_w - 1:
        # 保持前3个维度不变，只对最后2个空间维度进行处理
        pw_y_reshaped = pw_y.view(pw_y.shape[0], -1, pw_y.shape[-2], pw_y.shape[-1])
        target_size = (target_h, target_w - 1)
        
        # 如果当前尺寸大于目标尺寸，则裁剪；如果小于目标尺寸，则插值
        if pw_y_reshaped.shape[-2] > target_size[0] or pw_y_reshaped.shape[-1] > target_size[1]:
            # 裁剪到目标尺寸
            pw_y_reshaped = pw_y_reshaped[:, :, :target_size[0], :target_size[1]]
        elif pw_y_reshaped.shape[-2] < target_size[0] or pw_y_reshaped.shape[-1] < target_size[1]:
            # 插值到目标尺寸
            pw_y_reshaped = F.interpolate(pw_y_reshaped, size=target_size, mode='bilinear', align_corners=False)
        
        pw_y = pw_y_reshaped.view(pw_y.shape[0], pw_y.shape[1], pw_y.shape[2], target_h, target_w - 1)

    unary2[:, :-1, :] += (pw_x[:, 1, 0] + pw_x[:, 1, 1]) / 2.
    unary1[:, :-1, :] += (pw_x[:, 0, 1] + pw_x[:, 0, 0]) / 2.
    unary2[:, 1:, :] += (pw_x[:, 0, 1] + pw_x[:, 1, 1]) / 2.
    unary1[:, 1:, :] += (pw_x[:, 1, 0] + pw_x[:, 0, 0]) / 2.

    unary2[:, :, :-1] += (pw_y[:, 1, 0] + pw_y[:, 1, 1]) / 2.
    unary1[:, :, :-1] += (pw_y[:, 0, 1] + pw_y[:, 0, 0]) / 2.
    unary2[:, :, 1:] += (pw_y[:, 0, 1] + pw_y[:, 1, 1]) / 2.
    unary1[:, :, 1:] += (pw_y[:, 1, 0] + pw_y[:, 0, 0]) / 2.

    pw_x = (pw_x[:, 1, 0] + pw_x[:, 0, 1] - pw_x[:, 1, 1] - pw_x[:, 0, 0]) / 2
    pw_y = (pw_y[:, 1, 0] + pw_y[:, 0, 1] - pw_y[:, 1, 1] - pw_y[:, 0, 0]) / 2

    unary1 = unary1.detach().cpu().numpy()
    unary2 = unary2.detach().cpu().numpy()
    pw_x = pw_x.detach().cpu().numpy()
    pw_y = pw_y.detach().cpu().numpy()

    # solve graphcut
    if mp is None:
        mask = []
        for i in range(batch_size):
            mask.append(
                graphcut_multi(unary2[i], unary1[i], pw_x[i], pw_y[i], alpha, beta, eta, n_labels))
    else:
        input_mp = []
        for i in range(batch_size):
            input_mp.append((unary2[i], unary1[i], pw_x[i], pw_y[i], alpha, beta, eta, n_labels))
        mask = mp.starmap(graphcut_multi, input_mp)

    # optimal mask
    mask = torch.tensor(mask, dtype=torch.float32, device=device)
    mask = mask.unsqueeze(1)

    # add adversarial noise
    if adv_mask1 == 1.:
        input1 = input1 * std + mean + noise
        input1 = torch.clamp(input1, 0, 1)
        input1 = (input1 - mean) / std

    if adv_mask2 == 1.:
        input2 = input2 * std + mean + noise[indices]
        input2 = torch.clamp(input2, 0, 1)
        input2 = (input2 - mean) / std

    # tranport
    if transport:
        if t_size == -1:
            t_block_num = block_num
            t_size = block_size
        elif t_size < block_size:
            # block_size % t_size should be 0
            t_block_num = width // t_size
            mask = F.interpolate(mask, size=t_block_num)
            grad1_pool = F.avg_pool2d(grad1, t_size)
            unary1_torch = grad1_pool / grad1_pool.reshape(batch_size, -1).sum(1).reshape(
                batch_size, 1, 1)
            unary2_torch = unary1_torch[indices]
        else:
            t_block_num = block_num

        input1_vis = input1.clone()
        input2_vis = input2.clone()
        # input1
        plan = mask_transport(mask, unary1_torch, eps=t_eps)
        input1 = transport_image(input1, plan, batch_size, t_block_num, t_size)

        # input2
        plan = mask_transport(1 - mask, unary2_torch, eps=t_eps)
        input2 = transport_image(input2, plan, batch_size, t_block_num, t_size)

    # final mask and mixed ratio
    mask = F.interpolate(mask, size=width)
    ratio = mask.reshape(batch_size, -1).mean(-1)

    return mask * input1 + (1 - mask) * input2, ratio

def mask_transport(mask, grad_pool, eps=0.01):
    '''optimal transport plan'''
    batch_size = mask.shape[0]
    block_num = mask.shape[-1]

    n_iter = int(block_num)
    C = cost_matrix_dict[str(block_num)]

    z = (mask > 0).float()
    cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)

    # row and col
    for _ in range(n_iter):
        row_best = cost.min(-1)[1]
        plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

        # column resolve
        cost_fight = plan * cost
        col_best = cost_fight.min(-2)[1]
        plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
        plan_lose = (1 - plan_win) * plan

        cost += plan_lose

    return plan_win


def transport_image(img, plan, batch_size, block_num, block_size):
    '''apply transport plan to images'''
    input_patch = img.reshape([batch_size, 3, block_num, block_size,
                               block_num * block_size]).transpose(-2, -1)
    input_patch = input_patch.reshape([batch_size, 3, block_num, block_num, block_size,
                                       block_size]).transpose(-2, -1)
    input_patch = input_patch.reshape([batch_size, 3, block_num**2, block_size,
                                       block_size]).permute(0, 1, 3, 4, 2).unsqueeze(-1)

    input_transport = plan.transpose(
        -2, -1).unsqueeze(1).unsqueeze(1).unsqueeze(1).matmul(input_patch).squeeze(-1).permute(
            0, 1, 4, 2, 3)
    input_transport = input_transport.reshape(
        [batch_size, 3, block_num, block_num, block_size, block_size])
    input_transport = input_transport.transpose(-2, -1).reshape(
        [batch_size, 3, block_num, block_num * block_size, block_size])
    input_transport = input_transport.transpose(-2, -1).reshape(
        [batch_size, 3, block_num * block_size, block_num * block_size])

    return input_transport