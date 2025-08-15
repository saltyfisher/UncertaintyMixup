import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import random
import os
import argparse
import time
from datetime import datetime
from models import ResNet18WithDropout, get_target_layer
from datasets import get_dataloaders, get_num_classes
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import (denormalize_image, mixup_data, mixup_criterion, rand_bbox, cutmix_data, 
                   get_mc_dropout_activations, get_high_activation_areas_from_mc_dropout,
                   calculate_mixup_ratio, save_mixed_results, paste_activation_regions)


def train_with_uncertaintymixup(model, train_loader, device, optimizer, epoch, save_dir=None, dataset_type='chestct', activation_threshold=0.5, mean_threshold=0.7):
    """使用UncertaintyMixup进行训练"""
    # 确保模型在正常训练过程中处于训练模式，但不启用dropout
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 用于计算precision, recall和F1-score
    all_predictions = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 使用MC Dropout获取类激活图（10次）
        # get_mc_dropout_activations函数内部会临时启用dropout
        all_grayscale_cams = get_mc_dropout_activations(model, images, labels, num_mc_samples=10)
        
        # 重新确保模型回到正常的训练模式（不带dropout）
        model.train()
        
        # 从MC Dropout的多次类激活图中选择重叠的高响应区域
        high_activation_masks = get_high_activation_areas_from_mc_dropout(all_grayscale_cams, activation_threshold, mean_threshold)
        
        # 粘贴高响应区域到不同类别的其他数据，并计算混合率
        mixed_images, mixup_ratios, mixed_pairs = paste_activation_regions(images, labels, high_activation_masks, dataset_type)
        
        # 保存混合结果（仅在需要保存的epoch）
        if save_dir is not None:
            save_mixed_results(images, mixed_images, all_grayscale_cams, mixed_pairs, 
                             epoch, batch_idx, save_dir, dataset_type, model)
        
        # 使用混合图像进行前向传播
        outputs = model(mixed_images)
        
        # 创建损失函数实例
        criterion = nn.CrossEntropyLoss()
        
        # 计算损失 - 使用Mixup损失函数
        loss = 0
        for i in range(images.size(0)):
            # 对于每个样本，如果有混合则使用Mixup损失，否则使用标准损失
            if mixup_ratios[i] > 0:
                # 根据mixed_pairs找到与当前样本混合的目标样本索引
                target_idx = None
                for pair_info in mixed_pairs:
                    if pair_info is not None and pair_info['source_idx'] == i:
                        target_idx = pair_info['target_idx']
                        break
                
                if target_idx is not None:
                    # 使用Mixup损失函数
                    loss += mixup_criterion(criterion, outputs[target_idx].unsqueeze(0), labels[target_idx].unsqueeze(0), labels[i].unsqueeze(0), 1 - mixup_ratios[i])
                else:
                    loss += criterion(outputs[i].unsqueeze(0), labels[i].unsqueeze(0))
            else:
                loss += criterion(outputs[i].unsqueeze(0), labels[i].unsqueeze(0))
        
        # 对损失取平均值（除以批次大小）
        loss = loss / images.size(0)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累计损失和准确率
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 保存预测结果和真实标签，用于计算precision, recall和F1-score
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    
    # 使用sklearn计算准确率、精确率、召回率和F1分数
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def train_with_mixup(model, train_loader, device, optimizer, alpha=1.0):
    """使用标准Mixup进行训练"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 用于计算precision, recall和F1-score
    all_predictions = []
    all_labels = []
    all_labels_a = []
    all_labels_b = []
    all_lam = []
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 应用Mixup
        mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha, device)
        
        # 前向传播
        outputs = model(mixed_images)
        
        # 计算损失
        criterion = nn.CrossEntropyLoss()
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        
        # 优化器步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (lam * (predicted == labels_a).sum().float() + (1 - lam) * (predicted == labels_b).sum().float()).item()
        
        # 保存预测结果和真实标签，用于计算precision, recall和F1-score
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_labels_a.extend(labels_a.cpu().numpy())
        all_labels_b.extend(labels_b.cpu().numpy())
        all_lam.extend([lam] * labels.size(0))
    
    avg_loss = total_loss / total
    
    # 使用sklearn计算准确率、精确率、召回率和F1分数
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def train_with_cutmix(model, train_loader, device, optimizer, alpha=1.0):
    """使用CutMix进行训练"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 用于计算precision, recall和F1-score
    all_predictions = []
    all_labels = []
    all_labels_a = []
    all_labels_b = []
    all_lam = []
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 应用CutMix
        mixed_images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha, device)
        
        # 前向传播
        outputs = model(mixed_images)
        
        # 计算损失
        criterion = nn.CrossEntropyLoss()
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        
        # 优化器步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (lam * (predicted == labels_a).sum().float() + (1 - lam) * (predicted == labels_b).sum().float()).item()
        
        # 保存预测结果和真实标签，用于计算precision, recall和F1-score
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_labels_a.extend(labels_a.cpu().numpy())
        all_labels_b.extend(labels_b.cpu().numpy())
        all_lam.extend([lam] * labels.size(0))
    
    avg_loss = total_loss / total
    
    # 使用sklearn计算准确率、精确率、召回率和F1分数
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, strategy='uncertaintymixup', dataset_type='chestct', activation_threshold=0.5, mean_threshold=0.7, save_mixed_results=False):
    """训练模型的主函数"""
    model.to(device)
    
    # 在log_dir名称中添加日期、策略名称和激活阈值
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{strategy}_{dataset_type}_{activation_threshold}_{mean_threshold}_{timestamp}"
    
    # 创建tensorboard写入器
    writer = SummaryWriter(logdir=log_dir)
    
    # 创建混合结果保存目录（仅在需要保存时创建）
    mixed_results_dir = None
    if save_mixed_results:
        mixed_results_dir = f"mixed_results/{strategy}_{dataset_type}_{activation_threshold}_{mean_threshold}_{timestamp}"
    
    best_test_acc = 0.0
    best_epoch_test_f1 = 0.0  # 跟踪最佳测试F1-score
    
    # 以F1-score为标准，保存最佳F1-score时的其他指标
    best_f1_epoch_metrics = (0.0, 0.0, 0.0, 0.0)  # (acc, precision, recall, f1)
    
    for epoch in range(num_epochs):
        # 根据策略选择训练方法
        if strategy == 'uncertaintymixup':
            # 检查是否需要保存混合结果（每20个epoch以及第一个和最后一个epoch）
            save_dir = None
            if save_mixed_results and ((epoch + 1) % 20 == 0 or epoch == 0 or epoch == num_epochs - 1):
                save_dir = mixed_results_dir
                # 确保目录存在
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                
            train_result = train_with_uncertaintymixup(model, train_loader, device, optimizer, epoch, save_dir, dataset_type, activation_threshold, mean_threshold)
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_result
        elif strategy == 'mixup':
            train_result = train_with_mixup(model, train_loader, device, optimizer)
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_result
        elif strategy == 'cutmix':
            train_result = train_with_cutmix(model, train_loader, device, optimizer)
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_result
        else:
            # 标准训练
            model.train()
            train_loss = 0.0
            train_corrects = 0
            train_total = 0
            
            # 用于计算precision, recall和F1-score
            all_predictions = []
            all_labels = []
            
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_corrects += (predicted == labels).sum().item()
                
                # 保存预测结果和真实标签，用于计算precision, recall和F1-score
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # 计算训练损失和准确率
            train_loss = train_loss / train_total
            
            # 使用sklearn计算准确率、精确率、召回率和F1分数
            train_acc = accuracy_score(all_labels, all_predictions)
            train_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
            train_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
            train_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
            
            print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            
        # 只在需要保存混合结果的epoch进行测试
        if strategy in ['uncertaintymixup', 'mixup', 'cutmix'] and ((epoch + 1) % 20 == 0 or epoch == 0 or epoch == num_epochs - 1):
            # 测试模型
            test_loss, test_acc, test_precision, test_recall, test_f1 = test_model(model, test_loader, device, criterion)
            
            # 以F1-score为标准，保存最佳F1-score时的其他指标
            if test_f1 > best_epoch_test_f1:
                best_epoch_test_f1 = test_f1
                # 保存这一组指标作为最佳F1时的指标
                best_f1_epoch_metrics = (test_acc, test_precision, test_recall, test_f1)
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Test', test_acc, epoch)
            writer.add_scalar('Precision/Train', train_precision, epoch)
            writer.add_scalar('Precision/Test', test_precision, epoch)
            writer.add_scalar('Recall/Train', train_recall, epoch)
            writer.add_scalar('Recall/Test', test_recall, epoch)
            writer.add_scalar('F1-score/Train', train_f1, epoch)
            writer.add_scalar('F1-score/Test', test_f1, epoch)
            
            # 保存最佳模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), f'best_model_{strategy}_{dataset_type}.pth')
            
            print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        elif strategy in ['uncertaintymixup', 'mixup', 'cutmix']:
            # 仅记录训练结果到TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Precision/Train', train_precision, epoch)
            writer.add_scalar('Recall/Train', train_recall, epoch)
            writer.add_scalar('F1-score/Train', train_f1, epoch)
            print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        elif epoch == 0 or epoch == num_epochs - 1:
            # 对于标准训练，在第一个和最后一个epoch也进行测试
            test_loss, test_acc, test_precision, test_recall, test_f1 = test_model(model, test_loader, device, criterion)
            
            # 以F1-score为标准，保存最佳F1-score时的其他指标
            if test_f1 > best_epoch_test_f1:
                best_epoch_test_f1 = test_f1
                # 保存这一组指标作为最佳F1时的指标
                best_f1_epoch_metrics = (test_acc, test_precision, test_recall, test_f1)
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Test', test_acc, epoch)
            writer.add_scalar('Precision/Train', train_precision, epoch)
            writer.add_scalar('Precision/Test', test_precision, epoch)
            writer.add_scalar('Recall/Train', train_recall, epoch)
            writer.add_scalar('Recall/Test', test_recall, epoch)
            writer.add_scalar('F1-score/Train', train_f1, epoch)
            writer.add_scalar('F1-score/Test', test_f1, epoch)
            
            # 保存最佳模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), f'best_model_{strategy}_{dataset_type}.pth')
            
            print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    # 关闭tensorboard写入器
    writer.close()
    
    print('Training completed.')
    # 返回以F1-score为标准的最佳指标
    return model, best_f1_epoch_metrics[0], best_f1_epoch_metrics[1], best_f1_epoch_metrics[2], best_f1_epoch_metrics[3]


def test_model(model, test_loader, device, criterion=None):
    """在测试集上评估模型性能"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # 用于计算precision, recall和F1-score
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # 计算测试损失
            if criterion is not None:
                test_loss += criterion(outputs, labels).item() * images.size(0)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 保存预测结果和真实标签，用于计算precision, recall和F1-score
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均测试损失
    if criterion is not None:
        test_loss = test_loss / total
    
    # 计算accuracy, precision, recall和F1-score
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    print(f'测试集准确率: {accuracy:.4f}')
    print(f'测试集Precision: {precision:.4f}')
    print(f'测试集Recall: {recall:.4f}')
    print(f'测试集F1-score: {f1:.4f}')
    
    if criterion is not None:
        return test_loss, accuracy, precision, recall, f1
    else:
        return accuracy, precision, recall, f1


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Medical Image Classification with UncertaintyMixup')
    parser.add_argument('--data_dir', type=str, default='/workspace/MedicalImageClassficationData/chest-ctscan-images_datasets', 
                        help='数据集目录路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=180, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='优化器类型')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD优化器的动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减系数')
    parser.add_argument('--strategy', type=str, default='uncertaintymixup', 
                        choices=['standard', 'mixup', 'cutmix', 'uncertaintymixup', 'uncertaintymixup_v2'],
                        help='训练策略')
    parser.add_argument('--dataset', type=str, default='chestct', choices=['chestct', 'breakhis'],
                        help='数据集类型')
    parser.add_argument('--magnification', type=str, default=None, choices=['40X', '100X', '200X', '400X', None],
                        help='BreakHis数据集的放大倍数，None表示使用所有倍数')
    parser.add_argument('--test_split', type=float, default=0.2, help='BreakHis数据集的测试集比例')
    parser.add_argument('--num_trials', type=int, default=1, help='独立实验次数')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备号')
    parser.add_argument('--activation_threshold', type=float, default=0.5, help='激活阈值，大于该阈值时生成增广样本')
    parser.add_argument('--mean_threshold', type=float, default=0.7, help='均值阈值，用于过滤类激活图')
    parser.add_argument('--save_mixed_results', action='store_true', help='是否保存混合结果可视化图像')
    
    args = parser.parse_args()
    
    # 存储每次实验的结果
    trial_results = []
    
    # 进行多次独立实验
    for trial in range(args.num_trials):
        print(f"\n{'='*50}")
        print(f"开始第 {trial + 1}/{args.num_trials} 次独立实验")
        print(f"{'='*50}")
        
        # 根据数据集类型设置数据目录和类别数
        if args.dataset == 'breakhis':
            args.data_dir = '/workspace/MedicalImageClassficationData/BreakHis'
            num_classes = get_num_classes('breakhis')
        else:
            args.data_dir = '/workspace/MedicalImageClassficationData/chest-ctscan-images_datasets'
            num_classes = get_num_classes('chestct')
        
        print(f"Using {args.dataset} dataset")
        print(f"Data directory: {args.data_dir}")
        if args.magnification:
            print(f"Using magnification: {args.magnification}")
        if args.dataset == 'breakhis':
            print(f"Test split: {args.test_split}")
        
        # 设置设备
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            print(f"Using GPU device {args.gpu}")
        else:
            device = torch.device("cpu")
            print("CUDA is not available, using CPU")
        print(f"Using device: {device}")
        
        # 获取数据加载器
        train_loader, test_loader = get_dataloaders(
            args.data_dir, 
            batch_size=args.batch_size, 
            dataset_type=args.dataset,
            magnification=args.magnification,
            test_split=args.test_split
        )
        
        # 创建模型
        model = ResNet18WithDropout(num_classes=num_classes, dropout_rate=args.dropout_rate)
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 选择优化器
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            print(f"Using SGD optimizer with learning rate {args.lr} and momentum {args.momentum}")
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            print(f"Using Adam optimizer with learning rate {args.lr}")
        
        # 训练模型
        print("Starting training...")
        print(f"Training strategy: {args.strategy}")
        print(f"Dataset: {args.dataset}")
        print(f"Activation threshold: {args.activation_threshold}")
        print(f"Mean threshold: {args.mean_threshold}")
        print(f"Save mixed results: {args.save_mixed_results}")
        model, best_test_acc, best_test_precision, best_test_recall, best_test_f1 = train_model(model, train_loader, test_loader, criterion, optimizer, args.num_epochs, device, strategy=args.strategy, dataset_type=args.dataset, activation_threshold=args.activation_threshold, mean_threshold=args.mean_threshold, save_mixed_results=args.save_mixed_results)
        
        # 测试模型
        print("Testing model...")
        test_results = test_model(model, test_loader, device, criterion)
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = test_results
        # 使用训练过程中各指标的最佳值而不是最终测试值
        trial_results.append((best_test_acc, best_test_precision, best_test_recall, best_test_f1))
        
        print(f"第 {trial + 1} 次实验的最佳测试结果:")
        print(f"  准确率: {best_test_acc:.4f}")
        print(f"  Precision: {best_test_precision:.4f}")
        print(f"  Recall: {best_test_recall:.4f}")
        print(f"  F1-score: {best_test_f1:.4f}")
    
    # 输出所有实验的统计结果
    if args.num_trials > 1:
        print(f"\n{'='*50}")
        print(f"独立实验统计结果 (共 {args.num_trials} 次实验)")
        print(f"{'='*50}")
        
        # 分别提取各项指标
        accuracies = [result[0] for result in trial_results]
        precisions = [result[1] for result in trial_results]
        recalls = [result[2] for result in trial_results]
        f1_scores = [result[3] for result in trial_results]
        
        print(f"准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"F1-score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        
        print(f"\n详细统计:")
        print(f"  最高准确率: {np.max(accuracies):.4f}")
        print(f"  最低准确率: {np.min(accuracies):.4f}")
        print(f"  准确率方差: {np.var(accuracies):.4f}")
        
        print(f"  最高Precision: {np.max(precisions):.4f}")
        print(f"  最低Precision: {np.min(precisions):.4f}")
        print(f"  Precision方差: {np.var(precisions):.4f}")
        
        print(f"  最高Recall: {np.max(recalls):.4f}")
        print(f"  最低Recall: {np.min(recalls):.4f}")
        print(f"  Recall方差: {np.var(recalls):.4f}")
        
        print(f"  最高F1-score: {np.max(f1_scores):.4f}")
        print(f"  最低F1-score: {np.min(f1_scores):.4f}")
        print(f"  F1-score方差: {np.var(f1_scores):.4f}")
        
        print(f"\n所有实验结果:")
        for i, (acc, prec, rec, f1) in enumerate(trial_results):
            print(f"  实验 {i+1}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    
    # 如果只进行了一次实验，也输出简单的结果
    else:
        accuracy, precision, recall, f1 = trial_results[0]
        print(f"\n最终测试结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")


if __name__ == '__main__':
    main()