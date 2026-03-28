import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
# from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import random
import os
import argparse
import json
import time
import models
from datetime import datetime
from new_dataset import get_data
# from models import get_model
from datasets import get_dataloaders, get_num_classes
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from DIM_model import DIMModel
from utils import (denormalize_image, mixup_data, mixup_criterion, rand_bbox, cutmix_data, 
                   get_mc_dropout_activations, get_high_activation_areas_from_mc_dropout,
                   calculate_mixup_ratio, save_mixed_results, paste_activation_regions, paste_activation_regions_v2,  to_one_hot, visualize_multi_class_cam, matting_cutmix_data)


def train_with_uncertaintymixup(args, model, train_loader, device, optimizer, epoch, save_dir=None, dataset_type='chestct', activation_threshold=1, mean_threshold=0.7, use_guided_filter=True, guided_radius=5, guided_eps=1e-8,matting_model=None,superpixel=True,alphalabel=True,sigma1=0.01,sigma2=0.1,lam1=10000,lam2=10):
    """使用UncertaintyMixup进行训练"""
    # 确保模型在正常训练过程中处于训练模式，但不启用dropout
    
    total_loss = 0.0
    
    # 用于计算precision, recall和F1-score
    all_predictions = []
    all_labels = []
    
    # 创建日志文件
    log_filename = f"./training_log/training_log_epoch_{epoch}.txt"
    log_file = open(log_filename, 'w')
    log_file.write("epoch\tbatch\tratio\tmixed_label\tloss\n")  # 写入表头
    
    num_classes = get_num_classes(dataset_type)
    for batch_idx, (images, labels, _) in enumerate(train_loader):      
        images = images.to(device)
        labels = labels.to(device)
        
        if args.matting:
            mixed_images, mixup_ratios, mixed_pairs = matting_cutmix_data(images, labels, batch_idx,superpixel_nums=args.superpixel_nums, trimap_alpha=args.trimap_alpha, device=device, matting_method=matting_model,superpixel=superpixel,alphalabel=alphalabel,trimap_gen=args.trimap_gen,random_superpixel=args.random_superpixel)
        else:
        # 使用MC Dropout获取类激活图
        # get_mc_dropout_activations函数内部会临时启用dropout
        # all_high_activation_masks = np.zeros((len(images), num_classes, images.shape[2], images.shape[3]))
        # for l in range(num_classes):
        #     all_grayscale_cams = get_mc_dropout_activations(model, images, torch.tensor([l]*images.shape[0]), num_mc_samples=5)
        #     high_activation_masks = get_high_activation_areas_from_mc_dropout(
        #         all_grayscale_cams, 
        #         activation_threshold, 
        #         mean_threshold,
        #         guided_filter=use_guided_filter,
        #         radius=guided_radius,
        #         eps=guided_eps
        #     )
        #     all_high_activation_masks[:, l, :, :] = np.stack(high_activation_masks, axis=0)

        # for i in range(images.shape[0]):
        #     true_cam = all_high_activation_masks[i, labels[i].item(), :, :]
        #     other_cam = np.delete(all_high_activation_masks[i], labels[i].item(), axis=0)
        #     other_cam = np.clip(np.sum(other_cam, axis=0), 0, 1)
        #     high_activation_masks[i] = np.clip(true_cam - other_cam, 0, 1)
            

            all_grayscale_cams = get_mc_dropout_activations(model, images, labels, num_mc_samples=5)        
            
            # 从MC Dropout的多次类激活图中选择重叠的高响应区域
            high_activation_masks = get_high_activation_areas_from_mc_dropout(
                all_grayscale_cams, 
                activation_threshold, 
                mean_threshold,
                guided_filter=use_guided_filter,
                radius=guided_radius,
                eps=guided_eps
            )
            
            # 粘贴高响应区域到不同类别的其他数据，并计算混合率
            mixed_images, mixup_ratios, mixed_pairs = paste_activation_regions(images, labels, high_activation_masks, dataset_type)
        # mixed_images, mixup_ratios, mixed_pairs = paste_activation_regions_v2(images, labels, high_activation_masks, dataset_type)
        
        # 保存混合结果（仅在需要保存的epoch）
        if save_dir is not None:
            save_mixed_results(images, mixed_images, mixed_pairs, 
                             epoch, batch_idx, save_dir, dataset_type, model)
        
        # 重新确保模型回到正常的训练模式（不带dropout）
        model.train()
        # model.disable_dropout()
        # 使用混合图像进行前向传播
        outputs = model(mixed_images)
        if args.model == 'googlenet':
            outputs = outputs[0]
        # 创建损失函数实例
        criterion = nn.CrossEntropyLoss()
        
        # 计算损失 - 使用Mixup损失函数
        labels_output = labels.clone()
        labels = to_one_hot(labels, num_classes=get_num_classes(dataset_type), device=device)
        loss = 0
        # mixup_ratios[:] = mixup_ratios.mean()
        for i in range(images.size(0)):
            if mixed_pairs[i] is None:
                loss += criterion(outputs[i], labels[i])
            else:
                source_idx = mixed_pairs[i]['source_idx']
                target_idx = mixed_pairs[i]['target_idx']
                label_b = labels[source_idx]
                label_a = labels[target_idx]
                ratio = mixup_ratios[i]
                prob = F.softmax(outputs[i], dim=0)
                # ratio = ratio * prob[label_b==1]
                # if ratio < 0.1:
                #     ratio = 0
                ratio = 1 - ratio
                mixed_label = ratio * label_a + (1 - ratio) * label_b
                
                # 记录日志信息
                # label_a_idx = torch.argmax(label_a)
                # label_b_idx = torch.argmax(label_b)
                
                # label = ratio * label_a + (1 - ratio) * label_b
                # loss_ = torch.sum(-label * F.log_softmax(outputs[i], dim=0))
                loss_ = mixup_criterion(criterion, outputs[i], label_a, label_b, ratio)
                loss += loss_
                log_file.write(f"{epoch}\t{batch_idx}\t{ratio.item():.3f}\t{mixed_label.max().item():.3f}\t{loss_.item():.3f}\n")
        # # # 对损失取平均值（除以批次大小）
        loss = loss / images.size(0)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累计损失和准确率
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        
        # 保存预测结果和真实标签，用于计算precision, recall和F1-score
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels_output.cpu().numpy())
    
    # 关闭日志文件
    log_file.close()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    
    # 使用sklearn计算准确率、精确率、召回率和F1分数
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def train_with_mixup(args, model, train_loader, device, optimizer, alpha=1.0):
    """使用标准Mixup进行训练"""
    model.train()
    # model.disable_dropout()
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
        mixed_images, labels_a, labels_b, lam = mixup_data(args, images, labels, alpha, device)
        
        # 前向传播
        outputs = model(mixed_images)
        
        # 计算损失
        criterion = nn.CrossEntropyLoss()
        loss = 0
        for i in range(outputs.shape[0]):
            loss += mixup_criterion(criterion, outputs[i], labels_a[i], labels_b[i], lam[i])
        loss = loss/outputs.shape[0]
        # loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        
        # 优化器步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
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


def train_with_cutmix(args, dataset_type, epoch, model, train_loader, device, optimizer, alpha=1.0, save_dir=None):
    """使用CutMix进行训练"""
    model.train()
    # model.disable_dropout()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 用于计算precision, recall和F1-score
    all_predictions = []
    all_labels = []
    all_labels_a = []
    all_labels_b = []
    all_lam = []
    
    log_filename = f"training_log/training_log_epoch_{epoch}.txt"
    log_file = open(log_filename, 'w')
    log_file.write("epoch\tbatch\tratio\tmixed_label\tloss\n")  # 写入表头
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 应用CutMix
        mixed_images, labels_a, labels_b, lam, index, mixed_pairs = cutmix_data(args, images, labels, alpha, device)
        
        # 前向传播
        outputs = model(mixed_images)
        la = to_one_hot(labels, get_num_classes(dataset_type), device=device)
        # 计算损失
        criterion = nn.CrossEntropyLoss()
        loss = 0
        for i in range(outputs.shape[0]):
            mixed_label = la[i] * lam[i] + la[index[i]] * (1 - lam[i])
            loss_ = mixup_criterion(criterion, outputs[i], labels_a[i], labels_b[i], lam[i])
            loss += loss_
            # log_file.write(f"{epoch}\t{batch_idx}\t{lam[i]:.3f}\t{mixed_label.max().item():.3f}\t{loss_.item():.3f}\n")
        loss = loss/outputs.shape[0]
        # loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    
        # 优化器步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        # 保存预测结果和真实标签，用于计算precision, recall和F1-score
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_labels_a.extend(labels_a.cpu().numpy())
        all_labels_b.extend(labels_b.cpu().numpy())
        all_lam.extend([lam] * labels.size(0))
    
    log_file.close()
    avg_loss = total_loss / total
    
    # 使用sklearn计算准确率、精确率、召回率和F1分数
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def train_model(args, model, train_loader, test_loader, criterion, optimizer, num_epochs, device, strategy='uncertaintymixup', dataset_type='chestct', activation_threshold=0.5, mean_threshold=0.7, save_mixed_results=False, use_guided_filter=True, guided_radius=5, guided_eps=1e-8, matting_model=None,uncertaintymixup=False, superpixel=True, alphalabel=True,model_arch='resnet18'):
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
        mixed_results_dir = f"mixed_results/{strategy}_{args.superpixel_nums}_{args.trimap_alpha}_{timestamp}"
        os.makedirs(mixed_results_dir, exist_ok=True)
    
    best_test_acc = 0.0
    best_epoch_test_f1 = 0.0  # 跟踪最佳测试F1-score
    
    # 以F1-score为标准，保存最佳F1-score时的其他指标
    best_f1_epoch_metrics = (0.0, 0.0, 0.0, 0.0)  # (acc, precision, recall, f1)
    
    for epoch in range(num_epochs):
        save_dir = None
        if save_mixed_results and ((epoch + 1) % 20 == 0 or epoch == 0 or epoch == num_epochs - 1):
            save_dir = mixed_results_dir
            # 确保目录存在
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
        # 根据策略选择训练方法
        if uncertaintymixup:
            # 检查是否需要保存混合结果（每20个epoch以及第一个和最后一个epoch）                
            train_result = train_with_uncertaintymixup(
                args, model, train_loader, device, optimizer, epoch, save_dir, dataset_type, 
                activation_threshold, mean_threshold,
                use_guided_filter=use_guided_filter,
                guided_radius=guided_radius,
                guided_eps=guided_eps,
                matting_model=matting_model,
                superpixel=superpixel,
                alphalabel=alphalabel,
            )
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_result
        elif 'mixup' == strategy:
            train_result = train_with_mixup(args, model, train_loader, device, optimizer)
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_result
        elif 'cutmix' == strategy:
            train_result = train_with_cutmix(args, dataset_type, epoch, model, train_loader, device, optimizer, save_dir=save_dir)
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_result
        else:
            # 标准训练
            model.train()
            # model.disable_dropout()
            train_loss = 0.0
            train_corrects = 0
            train_total = 0
            
            # 用于计算precision, recall和F1-score
            all_predictions = []
            all_labels = []
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                if save_dir is not None:
                    visualize_multi_class_cam(model, images, labels, get_num_classes(dataset_type), epoch, batch_idx, save_path=save_dir)
                optimizer.zero_grad()
                outputs = model(images)
                if args.model == 'googlenet':
                    outputs = outputs[0]
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
        if ((epoch + 1) % 1 == 0 or epoch == 0 or epoch == num_epochs - 1):
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
    
    # 保存实验结果到results文件夹
    experiment_results = {
        "best_test_accuracy": best_test_acc,
        "best_f1_accuracy": best_f1_epoch_metrics[0],
        "best_f1_precision": best_f1_epoch_metrics[1],
        "best_f1_recall": best_f1_epoch_metrics[2],
        "best_f1_score": best_f1_epoch_metrics[3],
        "final_train_loss": train_loss,
        "final_train_accuracy": train_acc,
        "final_train_precision": train_precision,
        "final_train_recall": train_recall,
        "final_train_f1": train_f1,
        "dataset_type": dataset_type,
        "strategy": strategy,
        "model_arch": model_arch,
        "num_epochs": num_epochs,
        "activation_threshold": activation_threshold,
        "mean_threshold": mean_threshold
    }
    
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
        for images, labels, _ in test_loader:
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
    
    # print(f'测试集准确率: {accuracy:.4f}')
    # print(f'测试集Precision: {precision:.4f}')
    # print(f'测试集Recall: {recall:.4f}')
    # print(f'测试集F1-score: {f1:.4f}')
    
    if criterion is not None:
        return test_loss, accuracy, precision, recall, f1
    else:
        return accuracy, precision, recall, f1


def main():
    augmentation_methods = ['mixup', 'cutmix','mixuprand', 'cutmixrand','randaugment','trivialaugment','randaugment_raw','trivialaugment_raw','default']
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Medical Image Classification with UncertaintyMixup')
    parser.add_argument('--data_dir', type=str, default='/workspace/MedicalImageClassficationData/', 
                        help='数据集目录路径')
    parser.add_argument('--model', type=str, default='resnet18', help='模型选择')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=180, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='优化器类型')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD优化器的动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减系数')
    parser.add_argument('--strategy', type=str, default='uncertaintymixup', 
                        choices=augmentation_methods,
                        help='训练策略')
    parser.add_argument('--dataset', type=str, default='chestct', choices=['chestct', 'breakhis', 'padufes'],
                        help='数据集类型')
    parser.add_argument('--magnification', type=str, default=None, choices=['40', '100', '200', '400', None],
                        help='BreakHis数据集的放大倍数，None表示使用所有倍数')
    parser.add_argument('--test_split', type=float, default=0.2, help='BreakHis数据集的测试集比例')
    parser.add_argument('--num_trials', type=int, default=10, help='独立实验次数')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备号')
    parser.add_argument('--activation_threshold', type=float, default=1, help='激活阈值，大于该阈值时生成增广样本')
    parser.add_argument('--mean_threshold', type=float, default=0.7, help='均值阈值，用于过滤类激活图')
    parser.add_argument('--save_mixed_results', action='store_true', help='是否保存混合结果可视化图像')
    parser.add_argument('--pretrain', action='store_true', help='是否使用预训练权重')
    parser.add_argument('--test_all', action='store_true', help='是否测试所有方法')
    parser.add_argument('--resize', action='store_true', help='是否缩放测试集')
    parser.add_argument('--use_guided_filter', action='store_true', default=False, help='是否使用引导滤波处理高激活区域')
    parser.add_argument('--guided_radius', type=int, default=5, help='引导滤波的半径参数')
    parser.add_argument('--guided_eps', type=float, default=1e-8, help='引导滤波的正则化参数')
    parser.add_argument('--matting', action='store_true', help='是否使用抠图处理高激活区域')
    parser.add_argument('--uncertaintymixup', action='store_true', help='是否使用uncertaintymixup')
    parser.add_argument('--superpixel', action='store_true', help='聚类')
    parser.add_argument('--superpixel_nums', type=int, default=100, help='聚类数量')
    parser.add_argument('--random_superpixel', action='store_true')
    parser.add_argument('--trimap_alpha', type=int, default=20, help='trimap生成的alpha参数')
    parser.add_argument('--trimap_gen', type=str, default='graph', help='trimap生成的方法',choices=['graph', 'stats'])
    parser.add_argument('--alphalabel', action='store_true', help='标签混合')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    
    # 存储每次实验的结果
    trial_results = []
    # all_superpixel_nums = [100, 200, 500, 1000]
    all_superpixel_nums = [10, 20, 50]
    all_trimap_alpha = [10, 20, 40, 80]
    args.random_superpixel = True
    for superpixel_nums in all_superpixel_nums:
        for trimap_alpha in all_trimap_alpha:
            if superpixel_nums == 100 and trimap_alpha == 10:
                continue
            args.superpixel_nums = superpixel_nums
            args.trimap_alpha = trimap_alpha
            for trial in range(args.num_trials):
                print(f"\n{'='*50}")
                print(f"开始第 {trial + 1}/{args.num_trials} 次独立实验")
                print(f"{'='*50}")
                
                # 根据数据集类型设置数据目录和类别数
                if args.dataset == 'breakhis':
                    args.data_dir = '/workspace/MedicalImageClassficationData/BreakHis'
                    num_classes = get_num_classes('breakhis')
                elif args.dataset == 'chestct':
                    args.data_dir = '/workspace/MedicalImageClassficationData/chest-ctscan-images_datasets'
                    num_classes = get_num_classes('chestct')
                elif args.dataset == 'padufes':
                    args.data_dir = '/workspace/MedicalImageClassficationData/PAD-UFES-20'
                    num_classes = get_num_classes('padufes')
                elif args.dataset == 'bladdertissue':
                    args.data_dir = '/workspace/MedicalImageClassficationData/EndoscopicBladderTissue'
                    num_classes = get_num_classes('bladdertissue')            
                
                print(f"Using {args.dataset} dataset")
                print(f"Data directory: {args.data_dir}")
                if args.magnification:
                    print(f"Using magnification: {args.magnification}")
                if args.dataset == 'breakhis':
                    print(f"Test split: {args.test_split}")
                print(f"Pretrain: {args.pretrain}")
                if 'breakhis' in args.dataset:
                    args.resize = False
                # 设置设备
                if torch.cuda.is_available():
                    device = torch.device(f"cuda:{args.gpu}")
                    print(f"Using GPU device {args.gpu}")
                else:
                    device = torch.device("cpu")
                    print("CUDA is not available, using CPU")
                print(f"Using device: {device}")
                matting_model = None
                if args.matting:
                    print(f"Matting: {args.matting}")
                    checkpoint = 'BEST_params_DIM.pth'
                    matting_model = DIMModel()
                    matting_model.load_state_dict(torch.load(checkpoint))
                    matting_model = matting_model.to(device)
                    matting_model.eval()

                # 获取数据加载器
                if args.check:
                    traintest_dataset,test_dataset,resize_size,transform_train = get_data(args.strategy,args.dataset,args.magnification,'/workspace/MedicalImageClassficationData/')

                    train_loader = DataLoader(traintest_dataset, batch_size=args.batch_size, shuffle=True, persistent_workers=True, num_workers=4)
                    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, persistent_workers=True, shuffle=False, num_workers=4)
                else:
                    train_loader, test_loader = get_dataloaders(
                        args.data_dir, 
                        batch_size=args.batch_size, 
                        dataset_type=args.dataset,
                        magnification=args.magnification,
                        test_split=args.test_split,
                    )
                
                model = models.__dict__[args.model](num_classes=num_classes)
                # 创建模型
                # model = get_model(
                #     model_type='with_dropout',
                #     pretrain=args.pretrain,
                #     model_arch=args.model,
                #     num_classes=get_num_classes(args.dataset),
                # )
                print(f"model: {args.model}")
                # 定义损失函数
                criterion = nn.CrossEntropyLoss()
                
                # 选择优化器
                if args.optimizer == 'sgd':
                    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                    print(f"Using SGD optimizer with learning rate {args.lr} and momentum {args.momentum}")
                else:
                    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    print(f"Using Adam optimizer with learning rate {args.lr}")
                # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
                print(f"Pretrain: {args.pretrain}")
                
                # 训练模型
                print("Starting training...")
                print(f"Training strategy: {args.strategy}")
                print(f"Dataset: {args.dataset}")
                print(f"Activation threshold: {args.activation_threshold}")
                print(f"Mean threshold: {args.mean_threshold}")
                print(f"Save mixed results: {args.save_mixed_results}")
                print(f"Pretrain: {args.pretrain}")
                print(f"Superpixel Nums: {args.superpixel_nums}")
                print(f"Trimap Alpha: {args.trimap_alpha}")
                print(f"Trimap Gen:{args.trimap_gen}")
                if args.random_superpixel:
                    print(f"Random Superpixel True")
                model, best_test_acc, best_test_precision, best_test_recall, best_test_f1 = train_model(
                    args, model, train_loader, test_loader, criterion, optimizer, args.num_epochs, device, 
                    strategy=args.strategy, dataset_type=args.dataset, 
                    activation_threshold=args.activation_threshold, mean_threshold=args.mean_threshold, 
                    save_mixed_results=args.save_mixed_results,
                    use_guided_filter=args.use_guided_filter,
                    guided_radius=args.guided_radius,
                    guided_eps=args.guided_eps,
                    matting_model=matting_model,
                    uncertaintymixup = args.uncertaintymixup,
                    superpixel=args.superpixel,
                    alphalabel=args.alphalabel,
                )
                
                # 测试模型
                print("Testing model...")
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
                
                print(f"准确率: {np.mean(accuracies):.3f}({np.std(accuracies):.3f})")
                print(f"Precision: {np.mean(precisions):.3f}({np.std(precisions):.3f})")
                print(f"Recall: {np.mean(recalls):.3f}({np.std(recalls):.3f})")
                print(f"F1-score: {np.mean(f1_scores):.3f}({np.std(f1_scores):.3f})")
                
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
                
                # 保存统计结果到日志文件
                trial_stats = {
                    "num_trials": args.num_trials,
                    "mean_accuracy": float(np.mean(accuracies)),
                    "std_accuracy": float(np.std(accuracies)),
                    "mean_precision": float(np.mean(precisions)),
                    "std_precision": float(np.std(precisions)),
                    "mean_recall": float(np.mean(recalls)),
                    "std_recall": float(np.std(recalls)),
                    "mean_f1_score": float(np.mean(f1_scores)),
                    "std_f1_score": float(np.std(f1_scores)),
                    "max_accuracy": float(np.max(accuracies)),
                    "min_accuracy": float(np.min(accuracies)),
                    "var_accuracy": float(np.var(accuracies)),
                    "max_precision": float(np.max(precisions)),
                    "min_precision": float(np.min(precisions)),
                    "var_precision": float(np.var(precisions)),
                    "max_recall": float(np.max(recalls)),
                    "min_recall": float(np.min(recalls)),
                    "var_recall": float(np.var(recalls)),
                    "max_f1_score": float(np.max(f1_scores)),
                    "min_f1_score": float(np.min(f1_scores)),
                    "var_f1_score": float(np.var(f1_scores)),
                    "all_results": [{"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1_score": float(f1)} 
                                for acc, prec, rec, f1 in trial_results]
                }
                
                # 使用save_experiment_results函数保存结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if args.dataset == 'breakhis':
                    filename = f"{args.model}_{args.strategy}_{args.dataset}_{args.magnification}"
                else:    
                    filename = f"{args.model}_{args.strategy}_{args.dataset}"
                filename += f"_{args.trimap_gen}_{args.superpixel_nums}_{args.trimap_alpha}"
                if args.random_superpixel:
                    filename += "_rs"
                
                # 保存为JSON格式
                # with open(f"results/{filename}.json", 'w') as f:
                #     json.dump(trial_stats, f, indent=4)
                
                # 保存为CSV格式
                with open(f"results/{filename}.csv", 'w') as f:
                    f.write("metric,mean(var),std,min,max,var\n")
                    f.write(f"accuracy,{np.mean(accuracies):.3f}({np.std(accuracies):.3f}),{np.std(accuracies):.4f},{np.min(accuracies):.4f},{np.max(accuracies):.4f},{np.var(accuracies):.4f}\n")
                    f.write(f"precision,{np.mean(precisions):.3f}({np.std(precisions):.3f}),{np.std(precisions):.4f},{np.min(precisions):.4f},{np.max(precisions):.4f},{np.var(precisions):.4f}\n")
                    f.write(f"recall,{np.mean(recalls):.3f}({np.std(recalls):.3f}),{np.std(recalls):.4f},{np.min(recalls):.4f},{np.max(recalls):.4f},{np.var(recalls):.4f}\n")
                    f.write(f"f1_score,{np.mean(f1_scores):.3f}({np.std(f1_scores):.3f}),{np.std(f1_scores):.4f},{np.min(f1_scores):.4f},{np.max(f1_scores):.4f},{np.var(f1_scores):.4f}\n")
                    f.write("\nDetailed results:\n")
                    f.write("trial,accuracy,precision,recall,f1_score\n")
                    for i, (acc, prec, rec, f1) in enumerate(trial_results):
                        f.write(f"{i+1},{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f}\n")
                
                print(f"\n实验统计结果已保存至:")
                print(f"  - results/{filename}.json")
                print(f"  - results/{filename}.csv")
            
            # 如果只进行了一次实验，也输出简单的结果
            else:
                accuracy, precision, recall, f1 = trial_results[0]
                print(f"\n最终测试结果:")
                print(f"  准确率: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-score: {f1:.4f}")
                
                # 保存单次实验结果到日志文件
                single_result = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1)
                }
                
                # 使用save_experiment_results函数保存结果
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"single_result_{args.model}_{args.strategy}_{args.dataset}_{timestamp}"
                
                # 保存为JSON格式
                with open(f"results/{filename}.json", 'w') as f:
                    json.dump(single_result, f, indent=4)
                
                # 保存为CSV格式
                with open(f"results/{filename}.csv", 'w') as f:
                    f.write("metric,value\n")
                    f.write(f"accuracy,{accuracy:.4f}\n")
                    f.write(f"precision,{precision:.4f}\n")
                    f.write(f"recall,{recall:.4f}\n")
                    f.write(f"f1_score,{f1:.4f}\n")
                
                print(f"\n实验结果已保存至:")
                print(f"  - results/{filename}.json")
                print(f"  - results/{filename}.csv")


if __name__ == '__main__':
    main()