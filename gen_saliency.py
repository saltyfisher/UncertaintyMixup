"""使用谱残差法生成图像显著图"""
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import get_dataloaders
from scipy.ndimage import gaussian_filter
import concurrent

def normalize_map(S):
    S = S - S.min()
    if S.max() > 0:
        S = S / S.max()
    return S

import numpy as np
import cv2

def rgb_to_hematoxylin(img_rgb):
    """
    Convert RGB H&E image to Hematoxylin channel
    """
    img = img_rgb.astype(np.float32) + 1.0
    OD = -np.log(img / 255.0)

    # H&E stain matrix (Ruifrok)
    HE = np.array([
        [0.650, 0.704, 0.286],  # Hematoxylin
        [0.072, 0.990, 0.105],  # Eosin
        [0.268, 0.570, 0.776]   # residual
    ])

    HE_inv = np.linalg.pinv(HE)
    C = np.dot(OD.reshape((-1, 3)), HE_inv.T)
    H = C[:, 0].reshape(img_rgb.shape[:2])

    H = cv2.normalize(H, None, 0, 1, cv2.NORM_MINMAX)
    return H

def structure_constraint(H, sigma=2):
    """
    LoG response to emphasize nuclei-like blobs
    """
    H_blur = gaussian_filter(H, sigma)
    log = gaussian_filter(H_blur, sigma, order=2)
    log = np.abs(log)
    return normalize_map(log)

def entropy_weight(S):
    hist, _ = np.histogram(S.flatten(), bins=256, range=(0,1), density=True)
    hist += 1e-8
    entropy = -np.sum(hist * np.log(hist))
    return 1.0 / entropy

def multi_scale_spectral_residual(
    img,
    scales=(1, 2, 4, 8),
    smooth_sigma=2,
    eps=1e-8
):
    """
    Multi-scale Spectral Residual Saliency for medical images

    Parameters
    ----------
    img : ndarray
        Input grayscale image (H, W), uint8 or float
    scales : tuple
        Gaussian kernel sigmas in frequency domain
    smooth_sigma : float
        Spatial smoothing for final saliency map
    eps : float
        Numerical stability

    Returns
    -------
    saliency : ndarray
        Saliency map normalized to [0, 1]
    """

    # ---- 1. 预处理 ----
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)

    # 可选：医学图像对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply((img * 255).astype(np.uint8)) / 255.0

    # ---- 2. FFT ----
    F = np.fft.fft2(img)
    A = np.abs(F)
    P = np.angle(F)

    # ---- 3. Log amplitude ----
    L = np.log(A + eps)

    saliency_maps = []

    # ---- 4. 多尺度谱残差 ----
    for sigma in scales:
        background = gaussian_filter(L, sigma=sigma)
        residual = L - background

        S = np.fft.ifft2(np.exp(residual + 1j * P))
        S = np.abs(S) ** 2

        S = gaussian_filter(S, sigma=smooth_sigma)
        S = normalize_map(S)

        saliency_maps.append(S)
    
    # ---- 5. 多尺度融合 ----
    weights = [entropy_weight(S) for S in saliency_maps]
    weights = np.array(weights) / np.sum(weights)
    saliency = sum(w * S for w, S in zip(weights, saliency_maps))
    # saliency = np.mean(saliency_maps, axis=0)
    saliency = normalize_map(saliency)

    return saliency

def spectral_residual_saliency(image):
    """
    使用OpenCV中的谱残差法计算图像的显著图
    
    Args:
        image (numpy.ndarray): 输入图像，形状为(H, W, C)
    
    Returns:
        numpy.ndarray: 显著图，形状为(H, W)
    """
    # 将图像转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 使用OpenCV的谱残差显著性检测
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(gray)
    
    if not success:
        raise RuntimeError("Failed to compute saliency map")
    
    # OpenCV返回的是浮点型数组，范围在[0, 1]
    return saliency_map

def generate_saliency_maps(data_dir, output_dir, dataset_type='chestct', batch_size=1, num_workers=4, magnification=None, multiscale=False, deconv=False):
    """
    为数据集中的所有图像生成显著图
    
    Args:
        data_dir (str): 数据集根目录路径
        output_dir (str): 显著图保存目录
        dataset_type (str): 数据集类型
        batch_size (int): 批次大小
        num_workers (int): 数据加载器的工作进程数
        magnification (str): BreakHis数据集的放大倍数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取数据加载器
    train_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        dataset_type=dataset_type,
        magnification=magnification
    )
    
    # 处理训练集
    _process_dataloader(train_loader, output_dir, dataset_type, multiscale, deconv)
    
    # 处理测试集
    _process_dataloader(test_loader, output_dir, dataset_type, multiscale, deconv)

def _process_single_image(image_data, output_dir, classes, dataset_type, multiscale, deconv):
    """
    处理单张图像并保存结果
    """
    image, label, path = image_data
    # 转换图像格式 (B, C, H, W) -> (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    
    # 将像素值范围从[0, 1]转换为[0, 255]
    original_image = (image * 255).astype(np.uint8)
    
    if dataset_type == 'breakhis':
        if deconv:
            image = rgb_to_hematoxylin(original_image)
            image = gaussian_filter(image, sigma=1)
    # 生成显著图
    if multiscale:
        if dataset_type == 'breakhis':
            saliency_map = multi_scale_spectral_residual(image)
        else:
            saliency_map = multi_scale_spectral_residual(image)
    else:
        saliency_map = spectral_residual_saliency(image)
    
    saliency_map = normalize_map(saliency_map)
    
    # 将显著图转换为热力图
    heatmap = cv2.applyColorMap((saliency_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # 将热力图叠加到原图上
    overlay = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)
    
    # 获取类别名称
    class_name = classes[label.item()]
    
    # 生成文件名
    filename = os.path.basename(path)
    filepath = os.path.join(output_dir, class_name, filename)
    
    # 保存叠加后的图像
    cv2.imwrite(filepath, overlay)

def _process_dataloader(dataloader, output_dir, dataset_type, multiscale=False, deconv=False):
    """
    处理数据加载器中的所有图像
    
    Args:
        dataloader: 数据加载器
        output_dir (str): 输出目录
        dataset_type (str): 数据集类型
    """
    # 获取类别名称
    if hasattr(dataloader.dataset, 'dataset'):
        # 如果是Subset包装的数据集
        classes = dataloader.dataset.dataset.classes
    else:
        classes = dataloader.dataset.classes
    
    # 为每个类别创建目录
    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # 使用 ThreadPoolExecutor 进行多线程处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        # 处理每个批次
        for batch_idx, (images, labels, paths) in enumerate(tqdm(dataloader, desc=f"Processing {os.path.basename(output_dir)}")):
            # 将tensor转换为numpy数组
            if isinstance(images, torch.Tensor):
                images = images.numpy()
            
            # 提交每个图像的处理任务到线程池
            for i in range(images.shape[0]):
                image_data = (images[i], labels[i], paths[i])
                future = executor.submit(_process_single_image, image_data, output_dir, classes, dataset_type, multiscale, deconv)
                futures.append(future)
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing image: {e}")

def visualize_saliency(original_image, saliency_map, save_path=None):
    """
    可视化原始图像和显著图
    
    Args:
        original_image (numpy.ndarray): 原始图像
        saliency_map (numpy.ndarray): 显著图
        save_path (str): 保存路径，如果为None则显示图像
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示原始图像
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 显示显著图
    im = axes[1].imshow(saliency_map, cmap='jet')
    axes[1].set_title('Saliency Map')
    axes[1].axis('off')
    
    # 添加颜色条
    plt.colorbar(im, ax=axes[1])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate saliency maps using spectral residual method')
    parser.add_argument('--data_dir', type=str, default='/workspace/MedicalImageClassficationData/', help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, help='Path to save saliency maps')
    parser.add_argument('--dataset_type', type=str, default='chestct', 
                       choices=['chestct', 'breakhis', 'padufes'], 
                       help='Dataset type')
    parser.add_argument('--magnification', type=str, default=None,
                       choices=['40', '100', '200', '400'],
                       help='Magnification for BreakHis dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--deconv', action='store_true')
    
    args = parser.parse_args()
    args.output_dir = f'./vis_results/saliency/{args.dataset_type}'
    if args.multiscale:
        args.output_dir += '_multiscale'
    if args.deconv:
        args.output_dir += '_deconv'
    if args.dataset_type == 'breakhis':
        args.output_dir += f'_{args.magnification}'
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dataset_type == 'breakhis':
        args.data_dir = '/workspace/MedicalImageClassficationData/BreakHis'
    elif args.dataset_type == 'chestct':
        args.data_dir = '/workspace/MedicalImageClassficationData/chest-ctscan-images_datasets'
    generate_saliency_maps(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        magnification=args.magnification,
        multiscale=args.multiscale,
        deconv=args.deconv
    )