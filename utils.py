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
from joblib import Parallel, delayed
# from models import get_target_layer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from datetime import datetime
from scipy.ndimage import convolve1d
import torchvision.transforms as transforms

def to_one_hot(inp, num_classes, device='cuda'):
    '''one-hot label'''
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    return y_onehot

def visualize_multi_class_cam(model, image, label, num_classes, epoch, batch_idx, device='cpu', save_path=None):
    """
    可视化同一个样本对不同类别的CAM热力图对比
    
    Args:
        model: 训练好的模型
        image: 输入图像 (C, H, W)
        num_classes: 类别总数
        device: 设备类型
        save_path: 保存路径，如果为None则不保存
    
    Returns:
        matplotlib figure对象
    """
    # 确保模型处于评估模式
    model.eval()
    model.disable_dropout()
    epoch_dir = os.path.join(save_path, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    # 确保图像有批次维度
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # 添加批次维度
    
    image = image.to(device)
    
    # 创建目标层
    target_layers = [get_target_layer(model, 'with_dropout')]
    
    # 创建LayerCAM实例
    cam = LayerCAM(model=model, target_layers=target_layers)
    
    # 准备图像用于可视化 (转换为HxWxC格式)
    image_np = image[0].cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    label_np = label[0].cpu().numpy()
    
    # 反归一化图像
    image_vis = denormalize_image(image_np)
    
    # 转换为0-1范围的浮点数用于CAM可视化
    image_float = image_vis.astype(np.float32) / 255.0
    
    # 如果是灰度图，转换为RGB
    if image_float.shape[2] == 1:
        image_float = np.repeat(image_float, 3, axis=2)
        image_vis = np.repeat(image_vis, 3, axis=2)
    
    # 为每个类别生成CAM
    cam_images = []
    for class_idx in range(min(num_classes, 9)):  # 限制最多显示9个类别
        grayscale_cam = get_mc_dropout_activations(model, image[0,:].unsqueeze(0), torch.tensor(class_idx).unsqueeze(0))
        grayscale_cam = np.mean(grayscale_cam, axis=0)[0,:]
        # targets = [ClassifierOutputTarget(class_idx)]
        
        # # 生成灰度CAM
        # grayscale_cam = cam(input_tensor=image, targets=targets)[0, :]
        
        # 将CAM叠加到原图上
        cam_image = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)
        cam_images.append(cam_image)
    
    # 创建子图显示所有结果
    cols = min(3, len(cam_images))
    rows = (len(cam_images) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    fig.suptitle(f'Class {label_np} CAM Visulization', fontsize=16)
    
    # 处理axes可能是一维或二维的情况
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # 显示原始图像和各类别的CAM
    for i in range(len(cam_images)+1):
        if i == 0:
            axes[i].imshow(image_vis)
            axes[i].set_title('Original Image')
        else:
            axes[i].imshow(cam_images[i-1])
            axes[i].set_title(f'Class {i-1} CAM')
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(cam_images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        save_path = os.path.join(epoch_dir, f'batch_{batch_idx}_mixed_result.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def denormalize_image(image):
    """反归一化图像，将图像值范围从[0,1]转换为[0,255]"""
    # 确保图像值在[0,1]范围内
    image = np.clip(image, 0, 1)
    # 转换为[0,255]范围
    image = (image * 255).astype(np.uint8)
    return image


def mixup_data(args, x, y, alpha=1.0, device='cpu'):
    """标准Mixup数据增强"""
    batch_size = x.size(0)
    if alpha > 0:
        if args.strategy == 'mixuprand':
            lam = np.random.beta(alpha, alpha, batch_size)
        else:
            lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    index = torch.randperm(batch_size).to(device)
    mixed_x = x.clone()
    if args.strategy == 'mixuprand':
        for i in range(batch_size):
            mixed_x[i] = lam[i] * x[i] + (1 - lam[i]) * x[index[i]]
    else:
        lam = np.repeat(lam, batch_size)
        mixed_x = lam[0] * x + (1 - lam[0]) * x[index, :]
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


def matting_cutmix_data(x, y, batch_idx, superpixel_nums, alpha=1.0, trimap_alpha=10, device='cpu', matting_method=None, superpixel=True,random_superpixel=False,alphalabel=True,trimap_gen='graph',matting=True):
    """
    基于显著性图和抠图技术的CutMix数据增强
    
    Args:
        x: 输入图像数据
        y: 标签
        alpha: beta分布参数
        device: 设备类型
        matting_method: 抠图方法函数，接受图像、显著性图和Trimap作为输入，返回分割遮罩
    
    Returns:
        mixed_x: 混合后的图像
        y_a, y_b: 混合的标签
        out_lam: 混合比例
        index: 混合样本索引
        mixed_pairs: 混合对信息
    """
    batch_size = x.size(0)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = torch.randperm(batch_size).to(device)
    
    m_size = 64
    b, c, h, w = x.shape
    mixed_x = x.clone()
    out_lam = np.repeat(lam, y.shape[0])
    mixed_pairs = []
    all_results = {'input':[],'superpixels':[],'selected':[],'trimap':[],'alpha':[]}
    if superpixel:
        if random_superpixel:
            num_superpixels = max(50, int(superpixel_nums*lam))
        else:
            num_superpixels = superpixel_nums

        cluster = cv2.ximgproc.createSuperpixelSEEDS(
            w, h, c, 
            num_superpixels,
            num_levels=4 
        )        
    matting_trimap = []
    refined_mask = []
    if superpixel:
        matting_input = torch.zeros(b, x.shape[1]+1, m_size, m_size)
        recorder = {'pos':[],'mask':[]}
    else:
        matting_input = torch.zeros(b, x.shape[1]+1, w, h)
    for i in range(batch_size):
        j = index[i]
        input = x[j].detach().cpu().numpy().transpose(1, 2, 0)
        all_results['input'].append(input)
        if superpixel:
            cluster.iterate(input, 4)
            segement = cluster.getLabels()
            all_results['superpixels'].append(segement)
            
            # 使用谱残差算法计算显著图
            saliency_map = generate_saliency_map((input * 255).astype(np.uint8))
            
            # 计算每个超像素块的显著值之和作为权重
            unique_labels = np.unique(segement)
            weights = []
            for label in unique_labels:
                mask = segement == label
                weight = np.sum(saliency_map[mask])
                weights.append(weight)
            
            # 归一化权重
            weights = np.array(weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                # 如果所有权重都为0，使用均匀分布
                weights = np.ones_like(weights) / len(weights)
            
            # 根据权重随机选择超像素块
            selected_label = np.random.choice(unique_labels, p=weights)
            
            mask = segement == selected_label
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            if (max_row <= min_row) or (max_col <= min_col):
                recorder['pos'].append(None)
                recorder['mask'].append(None)
                matting_trimap.append(np.zeros((m_size,m_size)))
                continue
            else:
                recorder['pos'].append([min_col, max_col, min_row, max_row])
                recorder['mask'].append(mask)
            input = (input * 255).astype(np.uint8)
            cropped_region = input[min_row:max_row, min_col:max_col]
            all_results['selected'].append(cropped_region)
            # cropped_saliency_map = saliency_map[min_row:max_row, min_col:max_col]
            # cropped_saliency_map = cv2.resize(cropped_saliency_map, (m_size, m_size), interpolation=cv2.INTER_CUBIC)
            # cropped_saliency_map = cropped_saliency_map / np.max(cropped_saliency_map)
            cropped_region = cv2.resize(cropped_region, (m_size, m_size), interpolation=cv2.INTER_CUBIC)
            cropped_saliency_map = generate_saliency_map(cropped_region)
        else:
            saliency_map = generate_saliency_map((input * 255).astype(np.uint8))
            cropped_saliency_map = saliency_map
            cropped_region = input
        if matting:
            matting_input[j, :c] = torch.from_numpy(cropped_region.transpose(2, 0, 1) / 255.)
            # saliency_map = generate_saliency_map(cropped_region)
            trimap = trimap_generate(cropped_region, cropped_saliency_map, trimap_alpha,trimap_gen)
            
            # 保存cropped_region, cropped_saliency_map, trimap在同一张图上
            # os.makedirs('./vis_results/trimaps_gen', exist_ok=True)
            # # 转换为可视化格式
            # vis_cropped_region = cropped_region.copy()
            # vis_saliency_map = (cropped_saliency_map * 255).astype(np.uint8)
            # vis_saliency_map = np.repeat(vis_saliency_map[:, :, np.newaxis], 3, axis=2)
            # vis_trimap = np.repeat(trimap[:, :, np.newaxis], 3, axis=2)
            
            # # 水平拼接三张图像
            # combined_img = np.hstack([vis_cropped_region, vis_saliency_map, vis_trimap])
            # cv2.imwrite(os.path.join('./vis_results/trimaps_gen', f'trimap_combined_{batch_idx}_{i}.png'), combined_img)
            
            matting_trimap.append(trimap)
            matting_input[j,:3] = torch.from_numpy(trimap/255.).to(device)
        else:
            refined_mask.append(cropped_saliency_map)
        # 使用提供的抠图方法生成最终遮罩
    if matting:
        matting_trimap = np.stack(matting_trimap, axis=0)
        with torch.no_grad():
            refined_mask = matting_method(matting_input.to(device))
        refined_mask = refined_mask.cpu().numpy()
        refined_mask[matting_trimap == 0] = 0.0
        refined_mask[matting_trimap == 255] = 1.0
        refined_mask = refined_mask * 255
        refined_mask = refined_mask.astype(np.uint8)
        matting_trimap = [matting_trimap[i] for i in range(batch_size)]
    else:
        refined_mask = np.stack(refined_mask, axis=0)
        # 加了这两步后为什么效果这么好
        # refined_mask[refined_mask > 0.5] = 1.0
        # refined_mask[refined_mask <= 0.5] = 0.0
    for i in range(batch_size):
    # 使用遮罩的信息熵作为ratio
        mask = np.zeros(x.shape[2:])
        trimap = np.zeros(x.shape[2:])
        if superpixel:
            if recorder['pos'][i] == None:
                mixed_pairs.append(None)
            else:
                out = refined_mask[i]
                pos = recorder['pos'][i]
                min_col, max_col, min_row, max_row = pos
                cropped_region = cv2.resize(out, (max_col-min_col, max_row-min_row), interpolation=cv2.INTER_CUBIC)
                all_results['alpha'].append(cropped_region)
                mask[min_row:max_row, min_col:max_col] = cropped_region
                mask = mask * recorder['mask'][i]
                # cropped_triamp = matting_trimap[i]
                # cropped_triamp = cv2.resize(cropped_triamp, (max_col-min_col, max_row-min_row), interpolation=cv2.INTER_CUBIC)
                # all_results['trimap'].append(cropped_triamp)
                # trimap[min_row:max_row, min_col:max_col] = cropped_triamp
                # trimap = trimap * recorder['mask'][i]
                # matting_trimap[i] = trimap
        else:
            mask = refined_mask[i]
        if alphalabel:
            if superpixel:
                out_lam[i] = out_lam[i]*np.sum(refined_mask[i]/255.)/(refined_mask.shape[1]*refined_mask.shape[2])
            else:
                out_lam[i] = np.sum(refined_mask[i]/255.)/(refined_mask.shape[1]*refined_mask.shape[2])
        else:
            mask = out_lam[i] * mask
        j = index[i]
        # 使用抠图遮罩混合图像
        mask = mask/255.
        mixed_x[i] = apply_mask_based_mixup(x[i], x[j], mask)
        
        # os.makedirs('./vis_results/mixed_gen', exist_ok=True)
        # save_mixed = mixed_x[i].detach().cpu().numpy().transpose(1, 2, 0)
        # save_mixed = (save_mixed * 255).astype(np.uint8)
        # save_mask = (mask * 255).astype(np.uint8)
        # save_mask = np.repeat(save_mask[:,:,np.newaxis],3,axis=2)
        # save_trimap = np.repeat(matting_trimap[i][:, :, np.newaxis], 3, axis=2)
        # save_mixed = np.hstack([save_mixed, save_trimap, save_mask])
        # cv2.imwrite(os.path.join('./vis_results/mixed_gen', f'mixed_{batch_idx}_{i}.png'), save_mixed)

        mixed_pairs.append({
            'source_idx': j,
            'target_idx': i,
            # 'trimap':matting_trimap[i],
            'trimap':None,
            'mask': mask,
            'ratio': out_lam[i]
        })
    
    # 保存all_results中的结果到vis文件夹
    # save_dir = 'vis'
    # os.makedirs(save_dir, exist_ok=True)
    
    # for key, values in all_results.items():
    #     for idx, value in enumerate(values):
    #         file_name = f"{idx}_{key}"
    #         if isinstance(value, np.ndarray):
    #             # 特殊处理superpixels数据
    #             if key == 'superpixels':
    #                 # 保存原始superpixel标签数据
    #                 np.save(os.path.join(save_dir, f"{file_name}.npy"), value)
                    
    #                 # 创建overlay可视化效果
    #                 if idx < len(all_results['input']):
    #                     # 获取对应的原始图像
    #                     original_img = all_results['input'][idx]
    #                     original_img = (original_img * 255).astype(np.uint8)
                        
    #                     # 确保图像是3通道的
    #                     if len(original_img.shape) == 2 or original_img.shape[2] == 1:
    #                         original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
                        
    #                     segments = all_results['superpixels'][idx]

    #                     result = original_img.copy()
    #                     color=[255, 0, 0]
    #                     # 绘制超像素边界
    #                     for label in np.unique(segments):
    #                         # 创建当前标签的掩码
    #                         mask = (segments == label).astype(np.uint8) * 255
                            
    #                         # 查找轮廓
    #                         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
    #                         # 绘制轮廓
    #                         cv2.drawContours(result, contours, -1, color, 1)
                        
    #                     cv2.imwrite(os.path.join(save_dir, f"{file_name}_superpixel_overlay.png"), result)
    #             elif key in ['input', 'selected']:
    #                 # 这些已经是图像数据，确保是uint8格式
    #                 if value.dtype != np.uint8:
    #                     img_data = (value * 255).astype(np.uint8)
    #                 else:
    #                     img_data = value
    #                 cv2.imwrite(os.path.join(save_dir, f"{file_name}.png"), img_data)
    #             elif key in ['trimap', 'alpha']:
    #                 # 这些是单通道图像，需要转换为3通道
    #                 if len(value.shape) == 2:
    #                     img_data = np.repeat(value[:, :, np.newaxis], 3, axis=2)
    #                 else:
    #                     img_data = value
    #                 # 确保是uint8格式
    #                 if img_data.dtype != np.uint8:
    #                     img_data = (img_data * 255).astype(np.uint8) if img_data.max() <= 1 else img_data.astype(np.uint8)
    #                 cv2.imwrite(os.path.join(save_dir, f"{file_name}.png"), img_data)
    #             else:
    #                 # 其他numpy数组保存为.npy文件
    #                 np.save(os.path.join(save_dir, f"{file_name}.npy"), value)
    #         else:
    #             # 保存其他类型数据为文本文件
    #             with open(os.path.join(save_dir, f"{file_name}.txt"), 'w') as f:
    #                 f.write(str(value))
    
    y_a, y_b = y, y[index]
    return mixed_x, out_lam, mixed_pairs

def generate_saliency_map(image):
    """
    生成图像的显著性图
    
    Args:
        image: 输入图像 (C, H, W)
    
    Returns:
        saliency_map: 显著性图 (H, W)
    """
    # 转换为OpenCV格式 (H, W, C)
    if image.shape[0] == 3:  # 如果是通道优先
        img = np.transpose(image, (1, 2, 0))
    else:
        img = image
    
    # 转换为灰度图
    # if len(img.shape) == 3 and img.shape[2] == 3:
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # else:
    #     gray = img
    
    # # 使用Sobel算子计算梯度
    # grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # # 计算梯度幅值
    # saliency_map = np.sqrt(grad_x**2 + grad_y**2)
    
    sailency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map = sailency.computeSaliency(img)
    # 归一化到0-1范围
    if saliency_map.max() > 0:
        saliency_map = saliency_map / saliency_map.max()
    
    return saliency_map


def extract_connected_components(saliency_map, threshold=0.5):
    """
    从显著性图中提取连通区域，并使用形态学方法生成三分图
    
    Args:
        saliency_map: 显著性图
        threshold: 二值化阈值
    
    Returns:
        labeled_components: 标记的连通区域
        trimap: 三分图，前景是255，背景是0，未知区域是128
    """
    # 二值化
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    binary_map = cv2.threshold(saliency_map, threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1].astype(np.uint8)
    # binary_map = (saliency_map > threshold).astype(np.uint8)
    
    # 使用形态学操作生成三分图
    # 膨胀操作扩展前景区域
    k_size = random.choice(range(1, 5))
    iterations = np.random.randint(1, 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(binary_map, kernel, iterations=iterations)    
    # 腐蚀操作收缩背景区域
    eroded = cv2.erode(binary_map, kernel, iterations=iterations)
    # 创建Trimap: 前景是255，背景是0，未知区域是128
    trimap = np.zeros_like(binary_map, dtype=np.uint8)
    trimap.fill(128)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0

    return trimap


def apply_mask_based_mixup(target_img, source_img, mask):
    """
    根据遮罩混合两张图像
    
    Args:
        target_img: 目标图像
        source_img: 源图像
        mask: 遮罩 (与图像具有相同的形状)
    
    Returns:
        mixed_img: 混合后的图像
    """
    # 确保遮罩有正确的形状
    if mask.shape[0] != target_img.shape[0]:
        # 如果遮罩是单通道但图像不是，扩展遮罩通道
        mask = np.repeat(mask[np.newaxis,:,:], 3, axis=0)
    
    # 转换为torch tensor
    mask_tensor = torch.from_numpy(mask).to(target_img.device, dtype=target_img.dtype)
    
    # 使用遮罩混合图像
    mixed_img = target_img * (1 - mask_tensor) + source_img * mask_tensor
    
    return mixed_img


def cutmix_data(args, x, y, alpha=1.0, device='cpu'):
    """CutMix数据增强"""
    batch_size = x.size(0)
    if alpha > 0:
        if args.strategy == 'cutmixrand':
            lam = np.random.beta(alpha, alpha, batch_size)
        else:
            lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = torch.randperm(batch_size).to(device)

    mixed_pairs = None
    mixed_x = x.clone()
    if args.strategy == 'cutmixrand':
        for i in range(batch_size):
            bbx1, bby1, bbx2, bby2 = rand_bbox(x[i].unsqueeze(0).size(), lam[i])
            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index[i], :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam[i] = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (mixed_x.size()[-1] * mixed_x.size()[-2]))
    else:
        lam = np.repeat(lam, x.shape[0])
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam[0])
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam[:] = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index, mixed_pairs


def get_mc_dropout_activations(model, images, targets, num_mc_samples=5):
    """使用MC Dropout获取多次类激活图"""
    # 保存模型的原始训练状态
    
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
    
    return all_grayscale_cams


def get_high_activation_areas_from_mc_dropout(all_grayscale_cams, activation_threshold=0.5, mean_threshold=0.7, guided_filter=True, radius=5, eps=1e-8, alpha=1):
    """从MC Dropout的多次类激活图中选择重叠的高响应区域"""
    # 注意：这个函数不接收 dataset_type 参数，因为 mask 的生成与数据集类型无关
    #       mask 的尺寸适配在 paste_activation_regions 中完成
    high_activation_masks = []
    lam = np.random.beta(alpha, alpha)
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
        mean_threshold = np.max(mean_cam) * mean_threshold
        filtered_cam = np.where(mean_cam >= mean_threshold, mean_cam, 0)
        
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
        
        # 应用引导滤波平滑mask边缘
        if guided_filter:
            # 将mask转换为0-1之间的浮点数
            mask_float = mask.astype(np.float32)
            # 使用引导滤波平滑mask，使用mean_cam作为引导图像
            guided_mask = cv2.ximgproc.guidedFilter(mean_cam.astype(np.float32), mask_float, radius, eps)
            # 将结果转换回二值mask
            mask = (guided_mask > 0.5).astype(np.uint8)
        
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
    
    # 记录可用的mask不全零的样本
    available_indices = [i for i in range(batch_size) if np.any(high_activation_masks[i])]

    # 如果可用的mask数量小于2，则不进行混合
    if len(available_indices) < 2:
        for i in range(batch_size):
            mixup_ratios.append(0.0)
            mixed_pairs.append(None)
        return mixed_images, mixup_ratios, mixed_pairs
    
    resize_size = images.shape[2:]
    # j = random.choice(available_indices)
    for i in range(batch_size):
        # 从可用的mask中随机选择一个，并从中选择前景
        # 从可用的mask中随机选择不同类的样本
        idx = [j for j in available_indices if targets[j] != targets[i]]
        if idx == []:
            mixup_ratios.append(0.0)
            mixed_pairs.append(None)
            continue
        j = random.choice(idx)
        while j == i:
            j = random.choice(idx)
        # j = random.choice(available_indices)
        # while j == i:
        #     j = random.choice(available_indices)
        mask_j = high_activation_masks[j]
        mask_j = cv2.resize(mask_j, resize_size)  # 根据数据集类型设置尺寸匹配
        
        # 创建平滑的mask用于SmoothMix
        smooth_mask_j = create_smooth_mask(mask_j, sigma=7)  # sigma控制平滑程度
        
        # 计算混合率
        ratio = calculate_mixup_ratio(mask_j)
        mixup_ratios.append(ratio)

        # 将图像转换为numpy进行处理
        img_i = images[i].cpu().numpy().transpose(1, 2, 0)
        img_j = images[j].cpu().numpy().transpose(1, 2, 0)
        
        # 使用平滑mask进行混合
        # 扩展平滑mask到3通道
        smooth_mask_3ch = np.stack([smooth_mask_j, smooth_mask_j, smooth_mask_j], axis=2)
        
        # 以平滑方式混合图像
        mixed_img = img_i * (1 - smooth_mask_3ch) + img_j * smooth_mask_3ch
        
        # 将混合后的图像放回mixed_images
        mixed_images[i] = torch.tensor(
            mixed_img.transpose(2, 0, 1),
            dtype=torch.float32,
            device=images.device  # 确保在同一设备上
        )

        mixed_pairs.append({
            'source_idx': j,
            'target_idx': i,
            'mask': mask_j,
            'smooth_mask': smooth_mask_j,
            'ratio': ratio
        })

    mixup_ratios = torch.tensor(np.array(mixup_ratios), dtype=torch.float32,
            device=images.device)  # 确保在同一设备上)
    return mixed_images, mixup_ratios, mixed_pairs


def paste_activation_regions_v2(images, targets, high_activation_masks, dataset_type='chestct'):
    """将高激活区域粘贴到与目标图像中高响应区域相近但不重叠的位置"""
    batch_size = images.size(0)
    mixed_images = images.clone()
    mixup_ratios = []  # 存储每个样本的混合率
    mixed_pairs = []  # 存储混合对的信息
    
    # 记录可用的mask不全零的样本
    available_indices = [i for i in range(batch_size) if np.any(high_activation_masks[i])]

    # 如果可用的mask数量小于2，则不进行混合
    if len(available_indices) < 2:
        for i in range(batch_size):
            mixup_ratios.append(0.0)
            mixed_pairs.append(None)
        return mixed_images, mixup_ratios, mixed_pairs
    
    resize_size = images.shape[2:]
    
    for i in range(batch_size):
        # 从可用的mask中随机选择不同类的样本作为源
        idx = [j for j in available_indices if targets[j] != targets[i]]
        if idx == []:
            mixup_ratios.append(0.0)
            mixed_pairs.append(None)
            continue
            
        j = random.choice(idx)
        while j == i:  # 确保不是同一个样本
            j = random.choice(idx)
            
        mask_j = high_activation_masks[j]
        mask_j = cv2.resize(mask_j, resize_size)
        
        # 获取目标图像的高响应区域
        mask_i = high_activation_masks[i]
        mask_i = cv2.resize(mask_i, resize_size)
        
        # 创建平滑的mask用于SmoothMix
        smooth_mask_j = create_smooth_mask(mask_j, sigma=7)
        
        # 计算混合率
        ratio = calculate_mixup_ratio(mask_j)
        mixup_ratios.append(ratio)
        
        # 查找目标图像中与高响应区域相近但不重叠的位置
        target_position = find_non_overlapping_position(mask_i, mask_j, resize_size)
        
        # 如果找到了合适的位置，则在该位置粘贴源图像的高响应区域
        if target_position is not None:
            # 在指定位置粘贴mask_j
            positioned_mask = np.zeros_like(mask_j)
            x_offset, y_offset = target_position
            h, w = mask_j.shape
            
            # 确保不会越界
            x_end = min(x_offset + h, resize_size[0])
            y_end = min(y_offset + w, resize_size[1])
            
            positioned_mask[x_offset:x_end, y_offset:y_end] = mask_j[:x_end-x_offset, :y_end-y_offset]
            smooth_positioned_mask = create_smooth_mask(positioned_mask, sigma=7)
        else:
            # 如果找不到合适位置，使用原始方法
            positioned_mask = mask_j
            smooth_positioned_mask = smooth_mask_j
            
        # 将图像转换为numpy进行处理
        img_i = images[i].cpu().numpy().transpose(1, 2, 0)
        img_j = images[j].cpu().numpy().transpose(1, 2, 0)
        
        # 使用平滑mask进行混合
        smooth_mask_3ch = np.stack([smooth_positioned_mask, smooth_positioned_mask, smooth_positioned_mask], axis=2)
        
        # 以平滑方式混合图像
        mixed_img = img_i * (1 - smooth_mask_3ch) + img_j * smooth_mask_3ch
        
        # 将混合后的图像放回mixed_images
        mixed_images[i] = torch.tensor(
            mixed_img.transpose(2, 0, 1),
            dtype=torch.float32,
            device=images.device
        )

        mixed_pairs.append({
            'source_idx': j,
            'target_idx': i,
            'original_mask': mask_j,
            'positioned_mask': positioned_mask,
            'smooth_mask': smooth_positioned_mask,
            'ratio': ratio
        })

    mixup_ratios = torch.tensor(np.array(mixup_ratios), dtype=torch.float32, device=images.device)
    return mixed_images, mixup_ratios, mixed_pairs


def find_non_overlapping_position(target_mask, source_mask, resize_size, max_attempts=50):
    """
    在目标图像中查找与高响应区域相近但不重叠的位置
    
    Args:
        target_mask: 目标图像的高响应区域mask
        source_mask: 源图像的高响应区域mask
        resize_size: 图像尺寸
        max_attempts: 最大尝试次数
    
    Returns:
        position: (x, y)位置坐标，如果找不到则返回None
    """
    # 获取目标mask的边界框
    target_coords = np.where(target_mask > 0)
    if len(target_coords[0]) == 0:
        return None
        
    target_min_x, target_max_x = np.min(target_coords[0]), np.max(target_coords[0])
    target_min_y, target_max_y = np.min(target_coords[1]), np.max(target_coords[1])
    
    source_h, source_w = source_mask.shape
    
    # 在目标高响应区域附近寻找位置
    search_radius = max(source_h, source_w) // 2
    
    for _ in range(max_attempts):
        # 在目标区域附近随机选择一个中心点
        center_x = np.random.randint(
            max(0, target_min_x - search_radius),
            min(resize_size[0], target_max_x + search_radius + 1)
        )
        center_y = np.random.randint(
            max(0, target_min_y - search_radius),
            min(resize_size[1], target_max_y + search_radius + 1)
        )
        
        # 计算放置位置（使中心点对齐）
        x_offset = max(0, min(center_x - source_h // 2, resize_size[0] - source_h))
        y_offset = max(0, min(center_y - source_w // 2, resize_size[1] - source_w))
        
        # 检查这个位置是否与目标高响应区域重叠
        region_slice = (slice(x_offset, x_offset + source_h), slice(y_offset, y_offset + source_w))
        # overlap = np.any(target_mask[region_slice] > 0)
        overlap = np.sum(target_mask[region_slice] < np.sum(target_mask)/2)
        
        # 如果不重叠，返回这个位置
        if not overlap:
            return (x_offset, y_offset)
    
    # 如果尝试多次仍未找到合适位置，返回None
    return None


def save_mixed_results(images, mixed_images, mixed_pairs, epoch, batch_idx, save_dir, dataset_type='chestct', model=None, images_paths=None):
    """保存混合结果的可视化"""
    # 创建保存目录
    epoch_dir = save_dir
    os.makedirs(epoch_dir, exist_ok=True)

    for pair_idx, pair_info in enumerate(mixed_pairs):
        if pair_info is not None:
            source_idx = pair_info['source_idx']
            target_idx = pair_info['target_idx']
            mask = pair_info['mask']
            ratio = pair_info['ratio']
            if pair_info['trimap'] is not None:
                trimap = pair_info['trimap']/255
            else:
                trimap = None
            
            # 获取源图像和目标图像
            source_img = images[source_idx].cpu().numpy().transpose(1, 2, 0)
            target_img = images[target_idx].cpu().numpy().transpose(1, 2, 0)
            mixed_img = mixed_images[target_idx].cpu().numpy().transpose(1, 2, 0)
            source_img_name = images_paths[source_idx].split('/')[-1].split('.')[0]
            # 反归一化图像
            source_img = denormalize_image(source_img)
            target_img = denormalize_image(target_img)
            mixed_img = denormalize_image(mixed_img)          
            
            save_function(source_img, epoch_dir, f'{source_img_name}.png')
            save_function(target_img, epoch_dir, f'{source_img_name}_target.png')
            save_function(mixed_img, epoch_dir, f'{source_img_name}_mixed.png')
            save_function(mask, epoch_dir, f'{source_img_name}_mask.png')
            if trimap is not None:
                save_function(trimap, epoch_dir, f'{source_img_name}_trimap.png')

def save_function(images, save_dir, name):
    """保存输入图像"""
    save_dir = os.path.join(save_dir, name)
    plt.imsave(save_dir, images, dpi=300)

def calculate_mask_entropy(mask):
    """
    计算遮罩的信息熵
    
    Args:
        mask: 输入遮罩数组
        
    Returns:
        entropy: 信息熵值
    """
    # 将遮罩转换为一维数组
    flat_mask = mask.flatten()
    
    # 计算直方图
    hist, _ = np.histogram(flat_mask, bins=256, range=(0, 256), density=True)
    
    # 避免log(0)的情况
    hist = hist[hist > 0]
    
    # 计算信息熵
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy

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

def calculate_pairwise_costs_vectorized(img, sigma):
    """
    (This is the same vectorized function from a previous answer)
    Returns p, q indices and their corresponding scalar weights.
    """
    h, w, _ = img.shape
    num_pixels = w * h
    img_flat = img.reshape(-1, 3).astype(np.float32)
    img_flat = img_flat.mean(axis=1)

    # Horizontal edges
    p_h = np.arange(num_pixels).reshape(h, w)[:, :-1].ravel()
    q_h = np.arange(num_pixels).reshape(h, w)[:, 1:].ravel()
    
    diffs_h = np.sum((img_flat[p_h] - img_flat[q_h])**2, axis=1)
    
    # Vertical edges
    p_v = np.arange(num_pixels).reshape(h, w)[:-1, :].ravel()
    q_v = np.arange(num_pixels).reshape(h, w)[1:, :].ravel()
    diffs_v = np.sum((img_flat[p_v] - img_flat[q_v])**2, axis=1)

    # Combine indices and differences
    p_indices = np.concatenate([p_h, p_v])
    q_indices = np.concatenate([q_h, q_v])
    diffs = np.concatenate([diffs_h, diffs_v])

    # Calculate scalar weights
    weights = (np.exp(-diffs / (2 * sigma)))
    
    # Return structure and weights separately
    edges_structure = np.stack([p_indices, q_indices], axis=1)
    return edges_structure, weights

def calculate_pairwise_costs_with_convolve1d(img, pool_size=3):
    """
    使用convolve1d实现方向性池化
    
    Args:
        img: 输入图像 (H, W, C)
        pool_size: 池化窗口大小
        
    Returns:
        pairwise_h: 水平方向平滑项 (H, W-1)
        pairwise_v: 垂直方向平滑项 (H-1, W)
    """
    
    h, w, c = img.shape
    num_pixels = h * w
    img_float = img.astype(np.float32)
    
    # 创建平均池化核
    kernel = np.ones(pool_size) / pool_size
    
    # 水平方向：计算相邻像素差异
    horizontal_diff = np.mean(np.abs(img_float[:, :-1, :] - img_float[:, 1:, :]), axis=2)
    
    # 垂直方向：计算相邻像素差异
    vertical_diff = np.mean(np.abs(img_float[:-1, :, :] - img_float[1:, :, :]), axis=2)
    
    # 对水平方向差异使用水平convolve1d（沿axis=1，即列方向）
    pooled_horizontal = convolve1d(horizontal_diff, kernel, axis=1, mode='constant', cval=0.0)
    
    # 对垂直方向差异使用垂直convolve1d（沿axis=0，即行方向）
    pooled_vertical = convolve1d(vertical_diff, kernel, axis=0, mode='constant', cval=0.0)
    
    # Horizontal edges
    p_h = np.arange(num_pixels).reshape(h, w)[:, :-1].ravel()
    q_h = np.arange(num_pixels).reshape(h, w)[:, 1:].ravel()
    
    # Vertical edges
    p_v = np.arange(num_pixels).reshape(h, w)[:-1, :].ravel()
    q_v = np.arange(num_pixels).reshape(h, w)[1:, :].ravel()

    diffs_h = pooled_horizontal.ravel()
    diffs_v = pooled_vertical.ravel()
    # Combine indices and differences
    p_indices = np.concatenate([p_h, p_v])
    q_indices = np.concatenate([q_h, q_v])
    weights = np.concatenate([diffs_h, diffs_v])
    
    edges_structure = np.stack([p_indices, q_indices], axis=1)
    return edges_structure, weights

def calculate_grid_pairwise_costs(img, sigma):
    """
    Calculates horizontal and vertical pairwise costs for a grid graph.

    Args:
        img (np.ndarray): Input image (H, W, 3).
        LAMBDA (float): Smoothness term weight.
        SIGMA (float): Color sensitivity parameter.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - pairwise_h: Horizontal costs, shape (H, W-1)
            - pairwise_v: Vertical costs, shape (H-1, W)
    """
    h, w, _ = img.shape
    img_float = img.astype(np.float32)

    # 1. Calculate Horizontal Pairwise Costs
    # Difference between a pixel and its right neighbor
    diffs_h = np.sum((img_float[:, :-1, :] - img_float[:, 1:, :])**2, axis=2)
    pairwise_h = (np.exp(-diffs_h / (2 * sigma)))
    
    # an alternative form, often used and more stable
    # pairwise_h = (LAMBDA / (1 + diffs_h)).astype(np.int32)

    # 2. Calculate Vertical Pairwise Costs
    # Difference between a pixel and its bottom neighbor
    diffs_v = np.sum((img_float[:-1, :, :] - img_float[1:, :, :])**2, axis=2)
    pairwise_v = (np.exp(-diffs_v / (2 * sigma)))
    
    # an alternative form
    # pairwise_v = (LAMBDA / (1 + diffs_v)).astype(np.int32)

    return pairwise_h, pairwise_v

def calculate_grid_pairwise_costs_convolved(img, sigma, pool_size):
    """
    Calculates horizontal, vertical, and diagonal pairwise costs for a grid graph.

    Args:
        img (np.ndarray): Input image (H, W, 3).
        sigma (float): Color sensitivity parameter.
        pool_size (int): Pooling window size for convolution.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - pairwise_h: Horizontal costs, shape (H, W-1)
            - pairwise_v: Vertical costs, shape (H-1, W)
            - pairwise_dr: Diagonal right costs, shape (H-1, W-1)
            - pairwise_dl: Diagonal left costs, shape (H-1, W-1)
    """
    h, w, _ = img.shape
    img_float = img.astype(np.float32)
    
    kernel = np.ones(pool_size) / pool_size
        
    # 水平方向：计算相邻像素差异
    horizontal_diff = np.mean(np.abs(img_float[:, :-1, :] - img_float[:, 1:, :]), axis=2)
    
    # 垂直方向：计算相邻像素差异
    vertical_diff = np.mean(np.abs(img_float[:-1, :, :] - img_float[1:, :, :]), axis=2)
    
    # 对角线方向：计算相邻像素差异
    # 右下对角线 (i,j) 和 (i+1,j+1)
    diag_right_diff = np.mean(np.abs(img_float[:-1, :-1, :] - img_float[1:, 1:, :]), axis=2)
    
    # 左下对角线 (i,j+1) 和 (i+1,j)
    diag_left_diff = np.mean(np.abs(img_float[:-1, 1:, :] - img_float[1:, :-1, :]), axis=2)
    
    # 对水平方向差异使用水平convolve1d（沿axis=1，即列方向）
    pairwise_h = convolve1d(horizontal_diff, kernel, axis=1, mode='constant', cval=0.0)
    
    # 对垂直方向差异使用垂直convolve1d（沿axis=0，即行方向）
    pairwise_v = convolve1d(vertical_diff, kernel, axis=0, mode='constant', cval=0.0)
    
    # 对对角线方向差异使用convolve1d
    pairwise_dr = convolve1d(convolve1d(diag_right_diff, kernel, axis=1, mode='constant', cval=0.0), 
                             kernel, axis=0, mode='constant', cval=0.0)
    pairwise_dl = convolve1d(convolve1d(diag_left_diff, kernel, axis=1, mode='constant', cval=0.0), 
                             kernel, axis=0, mode='constant', cval=0.0)

    # 归一化到[0,1]范围
    # if pairwise_h.max() > 1:
    #     pairwise_h = pairwise_h / pairwise_h.max()
    # if pairwise_v.max() > 1:
    #     pairwise_v = pairwise_v / pairwise_v.max()
    # if pairwise_dr.max() > 1:
    #     pairwise_dr = pairwise_dr / pairwise_dr.max()
    # if pairwise_dl.max() > 1:
    #     pairwise_dl = pairwise_dl / pairwise_dl.max()

    return pairwise_h, pairwise_v, pairwise_dr, pairwise_dl

def graph_cut_with_index(index, unary, pairwise_h, pairwise_v, pairwise_cost):
    result = gco.cut_grid_graph(unary, pairwise_h, pairwise_v, pairwise_cost)
    return index, result

def trimap_generate(input,
                saliency,
                trimap_alpha=10,
                trimap_gen='graph',
                sigma1=None,
                sigma2=None,
                lam1=None,
                lam2=None,
                mp=None):
    
    if trimap_gen == 'graph':
        large_val_pairwise = 10
        large_val_unary = 100
        h, w, c = input.shape
        unary = np.zeros((3, h, w))
        # saliency = cv2.GaussianBlur(saliency, (3, 3), 0)
        # 0 for background, 2 for foreground, 1 for unknown
        unary[0] = -1*np.log(1-saliency+1e-8)
        unary[2] = -1*np.log(saliency+1e-8)
        # unary[0] = saliency
        # unary[1] = 1-saliency
        unary[1] = trimap_alpha * (saliency - 0.5)**2

        pairwise_cost = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                pairwise_cost[i, j] = (i - j)**2 / (3 - 1)**2
        
        # pairwise_cost = np.ones((3,3), dtype=np.float32)
        # np.fill_diagonal(pairwise_cost, 0)
        # pairwise_cost = np.array([
        #     [0, lam1, lam2],
        #     [lam1, 0, lam2],
        #     [lam2, lam2, 0]
        # ])

        # input = input.transpose(1, 2, 0)
        unary = unary.transpose(1, 2, 0)
        # pairwise_h, pairwise_v = calculate_grid_pairwise_costs(input, sigma2)
        pairwise_h, pairwise_v, pairwise_dr, pairwise_dl = calculate_grid_pairwise_costs_convolved(input, sigma1, 3)
        pairwise_h = np.nan_to_num(pairwise_h, nan=0.0, posinf=0.0, neginf=0.0)
        pairwise_v = np.nan_to_num(pairwise_v, nan=0.0, posinf=0.0, neginf=0.0)
        pairwise_dr = np.nan_to_num(pairwise_dr, nan=0.0, posinf=0.0, neginf=0.0)
        pairwise_dl = np.nan_to_num(pairwise_dl, nan=0.0, posinf=0.0, neginf=0.0)
        pairwise_h = (pairwise_h * large_val_pairwise).astype(np.int32)
        pairwise_v = (pairwise_v * large_val_pairwise).astype(np.int32)
        pairwise_dr = (pairwise_dr * large_val_pairwise).astype(np.int32)
        pairwise_dl = (pairwise_dl * large_val_pairwise).astype(np.int32)

        # unary[:,:-1] += np.repeat(pairwise_h[:,:,np.newaxis],3,axis=2)
        # unary[:-1,:] += np.repeat(pairwise_v[:,:,np.newaxis],3,axis=2)
        unary = np.nan_to_num(unary, nan=0.0, posinf=large_val_unary, neginf=-large_val_unary)
        # unary = (unary * large_val_unary).astype(np.int32)
        mask = gco.cut_grid_graph(unary, pairwise_cost, pairwise_v, pairwise_h, pairwise_dr, pairwise_dl, algorithm='swap')

        # edges, weights = _calculate_pairwise_costs_vectorized(input, sigma2)
        # edges, weights = calculate_pairwise_costs_with_convolve1d(input)
        # unary = unary.reshape(h*w, 3)
        # unary = np.nan_to_num(unary, nan=0.0, posinf=large_val, neginf=-large_val)
        # unary = (unary * large_val).astype(np.int32)
        # weights = np.nan_to_num(weights, nan=0.0, posinf=0, neginf=0)
        # weights = (weights * large_val).astype(np.int32)
        # mask = gco.cut_general_graph(edges, weights, unary, pairwise_cost, algorithm='swap')

        mask = mask.reshape(h, w)
        mask[mask == 0] = 255
        mask[mask == 1] = 128
        mask[mask == 2] = 0
    elif trimap_gen == 'stats':
        bg_th = trimap_alpha / 100
        mask = saliency.copy()
        # _, mask1 = cv2.threshold((saliency*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t_bg = np.quantile(saliency, bg_th)
        t_fg = np.quantile(saliency, 1-bg_th)
        mask[mask<=t_bg] = 0
        mask[mask>=t_fg] = 1
        mask[(mask>t_bg)&(mask<t_fg)] = 0.5
        mask = (mask * 255).astype(np.uint8)
    # if mp is None:
    #     mask = []
    #     for i in range(b):
    #         # 将数据移动到正确的设备上
    #         mask_i = gco.cut_grid_graph(unary[i], pairwise_h[i], pairwise_v[i], pairwise_cost)
    #         mask.append(mask_i)
    # else:
    #     # 使用joblib替代multiprocessing，并传递索引确保输出与输入数据索引对齐
    #     inputs = [(i, unary[i].detach(), 
    #               pairwise_h[i], 
    #               pairwise_v[i], 
    #               pairwise_cost) 
    #              for i in range(b)]
    #     results = Parallel(n_jobs=1, backend='loky')(delayed(gco.cut_grid_graph)(*inp) for inp in inputs)
        
    #     # 根据索引重新排序结果
    #     sorted_results = sorted(results, key=lambda x: x[0])
    #     mask = [result[1] for result in sorted_results]

    mask = mask.astype(np.uint8)
    return mask