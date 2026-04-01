#!/bin/bash

# t-SNE 和 UMAP 特征可视化运行脚本
# 使用方法：bash run_tsne_umap.sh

# 设置参数
DATASET_TYPE="chestct"  # 数据集类型：chestct, breakhis, padufes, kvasir, bladder
MODEL_ARCH="resnet18"   # 模型架构：resnet18, resnet50, resnet34
MODEL_TYPE="without_dropout"  # 模型类型：with_dropout, without_dropout
BATCH_SIZE=32           # 批次大小
NUM_WORKERS=4           # 数据加载工作进程数

# 输出目录
OUTPUT_DIR="./vis_results/tsne_umap_results"

# t-SNE 参数
TSNE_PERPLEXITY=30      # t-SNE 困惑度，建议范围 5-50
TSNE_ITER=1000          # t-SNE 迭代次数
TSNE_LR=200             # t-SNE 学习率

# UMAP 参数
UMAP_NEIGHBORS=15       # UMAP 邻居数，建议范围 5-50
UMAP_MIN_DIST=0.1       # UMAP 最小距离，建议范围 0.0-1.0
UMAP_METRIC="euclidean" # UMAP 距离度量：euclidean, cosine, manhattan

# 是否使用数据增强
USE_AUGMENTATION=false
STRATEGY="uncertaintymixup"  # 增强策略：uncertaintymixup, mixup, cutmixrand, puzzlemix, comix, guided

# BreakHis 放大倍数（仅当 DATASET_TYPE=breakhis 时使用）
MAGNIFICATION=null  # 可选值：40, 100, 200, 400

echo "======================================"
echo "t-SNE & UMAP Feature Visualization"
echo "======================================"
echo "Dataset: $DATASET_TYPE"
echo "Model: $MODEL_ARCH ($MODEL_TYPE)"
echo "Output Dir: $OUTPUT_DIR"
echo "======================================"

# 构建命令
CMD="python vis_feature.py \
    --dataset_type $DATASET_TYPE \
    --model_arch $MODEL_ARCH \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --output_dir $OUTPUT_DIR \
    --tsne_perplexity $TSNE_PERPLEXITY \
    --tsne_iter $TSNE_ITER \
    --tsne_lr $TSNE_LR \
    --umap_neighbors $UMAP_NEIGHBORS \
    --umap_min_dist $UMAP_MIN_DIST \
    --umap_metric $UMAP_METRIC"

# 如果使用数据增强
if [ "$USE_AUGMENTATION" = true ]; then
    CMD="$CMD --use_augmentation --strategy $STRATEGY"
fi

# 如果是 BreakHis 数据集
if [ "$DATASET_TYPE" = "breakhis" ] && [ "$MAGNIFICATION" != "null" ]; then
    CMD="$CMD --magnification $MAGNIFICATION"
fi

# 执行命令
echo "Running command:"
echo $CMD
echo ""
eval $CMD

echo ""
echo "======================================"
echo "Visualization completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "======================================"
