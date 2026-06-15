#!/bin/bash

# 检查是否提供了 strategy 参数
if [ -z "$1" ]; then
    echo "用法: ./run.sh <strategy>"
    echo "示例: ./run.sh mixup"
    echo "      ./run.sh cutmixrand"
    echo ""
    echo "可用的策略: mixup, cutmixrand, uncertaintymixup 等"
    exit 1
fi

STRATEGY=$1
MODEL=$2
GPU=$3

echo "=========================================="
echo "开始训练，使用策略: $STRATEGY，使用模型: $MODEL"
echo "=========================================="

# 定义数据集列表
DATASETS=("chestct" "breakhis")

# 定义放大倍数列表（仅适用于 breakhis）
MAGNIFICATIONS=(40 100 200 400)

# 遍历所有数据集
for DATASET in "${DATASETS[@]}"; do
    if [ "$DATASET" == "chestct" ]; then
        # chestct 只使用 magnification 400
        echo ""
        echo "----------------------------------------"
        echo "数据集: $DATASET"
        echo "----------------------------------------"
        python train.py --model $MODEL --dataset $DATASET --strategy $STRATEGY --matting --superpixel --alphalabel --random_superpixel --gpu $GPU
        
    elif [ "$DATASET" == "breakhis" ]; then
        # breakhis 遍历所有放大倍数
        for MAG in "${MAGNIFICATIONS[@]}"; do
            echo ""
            echo "----------------------------------------"
            echo "数据集: $DATASET, 放大倍数: $MAG"
            echo "----------------------------------------"
            python train.py --model $MODEL --dataset $DATASET --magnification $MAG --strategy $STRATEGY --matting --superpixel --alphalabel --random_superpixel --gpu $GPU
        done  # <--- 确保这个 done 存在
    fi        # <--- 确保这个 fi 存在，用于闭合 if/elif
done          # <--- 外层循环的 done

echo ""
echo "=========================================="
echo "所有训练完成！"
echo "=========================================="