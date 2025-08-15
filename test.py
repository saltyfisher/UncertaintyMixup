import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt


class ResNet18WithMCDropout(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(ResNet18WithMCDropout, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
        
        # 在每个残差块后添加dropout
        self.resnet18.layer1 = self.add_dropout_to_layer(self.resnet18.layer1, dropout_rate)
        self.resnet18.layer2 = self.add_dropout_to_layer(self.resnet18.layer2, dropout_rate)
        self.resnet18.layer3 = self.add_dropout_to_layer(self.resnet18.layer3, dropout_rate)
        self.resnet18.layer4 = self.add_dropout_to_layer(self.resnet18.layer4, dropout_rate)
        
        # 替换最后的全连接层
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def add_dropout_to_layer(self, layer, dropout_rate):
        # 在每个残差块后添加dropout
        for block in layer:
            block.dropout = nn.Dropout2d(dropout_rate)
        return layer
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        
        x = self.process_layer_with_dropout(self.resnet18.layer1, x)
        x = self.process_layer_with_dropout(self.resnet18.layer2, x)
        x = self.process_layer_with_dropout(self.resnet18.layer3, x)
        x = self.process_layer_with_dropout(self.resnet18.layer4, x)
        
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet18.fc(x)
        return x
    
    def process_layer_with_dropout(self, layer, x):
        for block in layer:
            x = block(x)
            if hasattr(block, 'dropout'):
                x = block.dropout(x)
        return x
    
    def enable_dropout(self):
        """启用所有dropout层，即使在eval模式下"""
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


def get_test_dataloader(data_dir, batch_size=16, num_workers=4):
    """获取测试数据加载器"""
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return test_loader


def mc_dropout_evaluation(model, images, num_mc_samples=10):
    """使用MC Dropout进行多次前向传播以评估不确定性"""
    model.enable_dropout()
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_mc_samples):
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            predictions.append(probs.cpu().numpy())
    
    predictions = np.array(predictions)
    mean_predictions = np.mean(predictions, axis=0)
    uncertainty = np.var(predictions, axis=0)
    
    return mean_predictions, uncertainty


def generate_gradcam_maps(model, images, targets):
    """使用GradCAM++生成类激活图"""
    target_layers = [model.resnet18.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    grayscale_cams = []
    for i in range(images.size(0)):
        input_tensor = images[i].unsqueeze(0)
        targets_for_cam = [ClassifierOutputTarget(targets[i].item())]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets_for_cam)[0, :]
        grayscale_cams.append(grayscale_cam)
    
    return np.array(grayscale_cams)


def get_high_activation_areas(grayscale_cams):
    """从类激活图中选择面积最大的高响应区域"""
    high_activation_masks = []
    
    for cam in grayscale_cams:
        # 使用Otsu阈值法确定阈值
        _, thresh = cv2.threshold((cam * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓并选择面积最大的区域
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 选择面积最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 创建mask
            mask = np.zeros_like(cam, dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 1, -1)
        else:
            # 如果没有找到轮廓，使用高于平均值的区域
            mask = (cam > np.mean(cam)).astype(np.uint8)
        
        high_activation_masks.append(mask)
    
    return high_activation_masks


def visualize_results(images, labels, mean_predictions, uncertainty, grayscale_cams, high_activation_masks, class_names, output_dir='results'):
    """可视化结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 反归一化图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(min(10, images.size(0))):  # 只可视化前10个样本
        # 反归一化图像
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(std * img + mean, 0, 1)
        
        # 预测类别
        pred_class = np.argmax(mean_predictions[i])
        true_class = labels[i].item()
        
        # 创建可视化图
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # 原始图像
        axes[0].imshow(img)
        axes[0].set_title(f'Original\nTrue: {class_names[true_class]}\nPred: {class_names[pred_class]}')
        axes[0].axis('off')
        
        # 类激活图
        cam_on_image = show_cam_on_image(img, grayscale_cams[i], use_rgb=True)
        axes[1].imshow(cam_on_image)
        axes[1].set_title('GradCAM++')
        axes[1].axis('off')
        
        # 高激活区域
        axes[2].imshow(high_activation_masks[i], cmap='gray')
        axes[2].set_title('High Activation Area')
        axes[2].axis('off')
        
        # 不确定性图
        axes[3].imshow(np.sum(uncertainty[i], axis=0), cmap='jet')
        axes[3].set_title('Uncertainty (MC Dropout)')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i}_true_{class_names[true_class]}_pred_{class_names[pred_class]}.png'))
        plt.close()


def test_model_with_visualizations(model_path, data_dir, device, num_mc_samples=10):
    """测试模型并生成可视化结果"""
    # 类别名称
    class_names = ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma']
    
    # 加载模型
    model = ResNet18WithMCDropout(num_classes=4, dropout_rate=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 获取测试数据
    test_loader = get_test_dataloader(data_dir, batch_size=8)
    
    # 处理一个批次的数据进行可视化
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # MC Dropout评估
        mean_predictions, uncertainty = mc_dropout_evaluation(model, images, num_mc_samples)
        
        # 生成GradCAM++图
        grayscale_cams = generate_gradcam_maps(model, images, labels)
        
        # 获取高激活区域
        high_activation_masks = get_high_activation_areas(grayscale_cams)
        
        # 可视化结果
        visualize_results(images, labels, mean_predictions, uncertainty, grayscale_cams, high_activation_masks, class_names)
        
        print(f"Visualizations saved for a batch of {len(images)} samples")
        break  # 只处理一个批次进行演示


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ResNet18 with MC Dropout and GradCAM++')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='/workspace/MedicalImageClassficationData/chest-ctscan-images_datasets',
                        help='Path to chest CT scan images dataset')
    parser.add_argument('--num_mc_samples', type=int, default=10, help='Number of MC samples for uncertainty estimation')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 测试模型并生成可视化结果
    test_model_with_visualizations(args.model_path, args.data_dir, device, args.num_mc_samples)


if __name__ == '__main__':
    main()