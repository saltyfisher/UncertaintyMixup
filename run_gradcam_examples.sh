#!/bin/bash

# Grad-CAM 热力图生成示例脚本

# 示例 1: ChestCT 数据集 - 不带 Dropout 的模型
echo "=== 示例 1: ChestCT 数据集 - 不带 Dropout ==="
python vis_saliency.py \
    --model_path best_model_chestct.pth \
    --data_dir ../chest-ctscan-images_datasets \
    --dataset_type chestct \
    --model_arch resnet18 \
    --model_type without_dropout \
    --output_dir ./gradcam_results/chestct_without_dropout \
    --batch_size 1 \
    --num_workers 4

# 示例 2: ChestCT 数据集 - 带 Dropout 的模型（UncertaintyMixup）
echo "\n=== 示例 2: ChestCT 数据集 - 带 Dropout (UncertaintyMixup) ==="
python vis_saliency.py \
    --model_path best_model_uncertaintymixup_chestct.pth \
    --data_dir ../chest-ctscan-images_datasets \
    --dataset_type chestct \
    --model_arch resnet18 \
    --model_type with_dropout \
    --output_dir ./gradcam_results/chestct_with_dropout \
    --batch_size 1 \
    --num_workers 4

# 示例 3: BreakHis 数据集 - 40X 放大倍数
echo "\n=== 示例 3: BreakHis 数据集 - 40X ==="
python vis_saliency.py \
    --model_path best_model_breakhis_40.pth \
    --data_dir ../BreakHis \
    --dataset_type breakhis \
    --model_arch resnet18 \
    --model_type without_dropout \
    --magnification 40X \
    --output_dir ./gradcam_results/breakhis_40X \
    --batch_size 1 \
    --num_workers 4

# 示例 4: 只使用 Grad-CAM 方法
echo "\n=== 示例 4: 只使用 Grad-CAM ==="
python vis_saliency.py \
    --model_path best_model_chestct.pth \
    --data_dir ../chest-ctscan-images_datasets \
    --dataset_type chestct \
    --methods gradcam \
    --output_dir ./gradcam_results/chestct_gradcam_only \
    --batch_size 1 \
    --num_workers 4

# 示例 5: 使用所有方法（默认）
echo "\n=== 示例 5: 使用所有三种方法 ==="
python vis_saliency.py \
    --model_path best_model_chestct.pth \
    --data_dir ../chest-ctscan-images_datasets \
    --dataset_type chestct \
    --methods gradcam gradcam++ layercam \
    --output_dir ./gradcam_results/chestct_all_methods \
    --batch_size 1 \
    --num_workers 4

echo "\n=== 所有示例执行完毕 ==="
