# t-SNE & UMAP Feature Visualization

## 概述

本工具用于可视化深度学习模型提取的特征，通过 t-SNE 和 UMAP 两种降维技术，将高维特征空间映射到 2D 或 3D 空间，帮助理解模型的表征学习能力和类别分离效果。

## 功能特性

- **特征提取**: 从训练好的模型中提取中间层特征（如 avgpool 层）
- **t-SNE 可视化**: 使用 t-Distributed Stochastic Neighbor Embedding 进行非线性降维
- **UMAP 可视化**: 使用 Uniform Manifold Approximation and Projection 进行流形学习
- **多数据集支持**: 支持 ChestCT、BreakHis、PAD-UFES-20、Kvasir、Bladder 等医学图像数据集
- **多模型架构**: 支持 ResNet18、ResNet50、ResNet34、Inception、WideResNet、EfficientNet-B0
- **对比分析**: 可对比带/不带 Dropout 的模型特征分布差异
- **结果保存**: 同时保存可视化图片和数值结果（.npz 格式）

## 安装依赖

除了项目基础依赖外，还需要安装以下库：

```bash
pip install scikit-learn umap-learn matplotlib seaborn
```

或者更新 requirements.txt：

```bash
pip install -r requirements.txt
```

确保 requirements.txt 包含：
- scikit-learn >= 0.24.0
- umap-learn >= 0.5.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## 使用方法

### 方法 1: 使用运行脚本（推荐）

```bash
# 基本用法
bash run_tsne_umap.sh

# 修改参数后运行
# 编辑 run_tsne_umap.sh 中的参数，然后执行
bash run_tsne_umap.sh
```

### 方法 2: 直接运行 Python 脚本

#### 基本示例

```bash
# ChestCT 数据集，ResNet18，不带 Dropout
python vis_feature.py \
    --dataset_type chestct \
    --model_arch resnet18 \
    --model_type without_dropout \
    --batch_size 32 \
    --output_dir ./vis_results/tsne_umap_results
```

#### 使用数据增强训练的模型

```bash
# 使用 UncertaintyMixup 策略训练的模型
python vis_feature.py \
    --dataset_type chestct \
    --model_arch resnet18 \
    --model_type without_dropout \
    --use_augmentation \
    --strategy uncertaintymixup \
    --batch_size 32 \
    --output_dir ./vis_results/tsne_umap_results
```

#### BreakHis 数据集

```bash
# BreakHis 数据集，放大倍数 200x
python vis_feature.py \
    --dataset_type breakhis \
    --model_arch resnet18 \
    --model_type with_dropout \
    --magnification 200 \
    --batch_size 32 \
    --output_dir ./vis_results/tsne_umap_results
```

#### 调整 t-SNE 参数

```bash
# 调整困惑度和迭代次数
python vis_feature.py \
    --dataset_type chestct \
    --model_arch resnet18 \
    --model_type without_dropout \
    --tsne_perplexity 50 \
    --tsne_iter 1500 \
    --tsne_lr 300 \
    --batch_size 32
```

#### 调整 UMAP 参数

```bash
# 调整邻居数和最小距离
python vis_feature.py \
    --dataset_type chestct \
    --model_arch resnet18 \
    --model_type without_dropout \
    --umap_neighbors 30 \
    --umap_min_dist 0.05 \
    --umap_metric cosine \
    --batch_size 32
```

## 参数说明

### 模型相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | 自动生成 | 模型参数文件路径 (.pth) |
| `--model_arch` | str | resnet18 | 模型架构 (resnet18/resnet50/resnet34/inception/wideresnet/efficientnet_b0) |
| `--model_type` | str | without_dropout | 模型类型 (with_dropout/without_dropout) |
| `--use_augmentation` | flag | False | 是否使用数据增强训练的模型 |
| `--strategy` | str | uncertaintymixup | 数据增强策略 (uncertaintymixup/mixup/cutmixrand/puzzlemix/comix/guided) |

### 数据集相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | /workspace/MedicalImageClassficationData/ | 数据集根目录 |
| `--dataset_type` | str | chestct | 数据集类型 (chestct/breakhis/padufes/kvasir/bladder) |
| `--magnification` | str | None | BreakHis 数据集的放大倍数 (40/100/200/400) |

### t-SNE 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--tsne_perplexity` | int | 30 | 困惑度，建议范围 5-50 |
| `--tsne_iter` | int | 1000 | 迭代次数，越大结果越稳定但计算越慢 |
| `--tsne_lr` | int | 200 | 学习率，建议范围 10-1000 |

### UMAP 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--umap_neighbors` | int | 15 | 邻居数，越小关注局部结构，越大关注全局结构 |
| `--umap_min_dist` | float | 0.1 | 最小距离，控制嵌入点的紧密程度 (0.0-1.0) |
| `--umap_metric` | str | euclidean | 距离度量 (euclidean/cosine/manhattan) |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--batch_size` | int | 32 | 批次大小 |
| `--num_workers` | int | 4 | 数据加载工作进程数 |
| `--output_dir` | str | ./vis_results/tsne_umap_results | 输出目录 |

## 输出结果

程序会在指定的输出目录下生成以下文件：

```
output_dir/
├── chestct/                          # 按数据集分类
│   ├── tsne_without_dropout_resnet18.png      # t-SNE 可视化图
│   ├── tsne_without_dropout_resnet18.npz      # t-SNE 数值结果
│   ├── umap_without_dropout_resnet18.png      # UMAP 可视化图
│   └── umap_without_dropout_resnet18.npz      # UMAP 数值结果
├── breakhis200/                      # BreakHis 特定放大倍数
│   ├── tsne_with_dropout_resnet18.png
│   └── ...
└── ...
```

### 数值结果文件 (.npz)

每个 .npz 文件包含以下数组：

- `embedding`: 降维后的坐标 (N, 2) 或 (N, 3)
- `labels`: 每个样本的类别标签 (N,)
- `features`: 原始高维特征 (N, D)
- `paths`: 每个样本的图像路径 (N,)

可以使用以下代码加载：

```python
import numpy as np

data = np.load('tsne_result.npz')
embedding = data['embedding']
labels = data['labels']
features = data['features']
paths = data['paths']
```

## 结果解读

### t-SNE 可视化

- **簇的分离**: 不同颜色的点代表不同类别，簇间距离越大表示类别分离越好
- **簇的紧密度**: 簇内点越密集，表示该类别的样本特征越一致
- **困惑度影响**: 
  - 较小的 perplexity（如 5-10）：关注局部结构，可能产生多个小簇
  - 较大的 perplexity（如 30-50）：关注全局结构，可能合并相邻簇

### UMAP 可视化

- **全局结构**: UMAP 更好地保持全局数据结构，簇间关系更有意义
- **计算速度**: 通常比 t-SNE 更快，尤其适用于大数据集
- **参数影响**:
  - `n_neighbors`: 较小值（5-10）关注局部细节，较大值（30-50）关注全局结构
  - `min_dist`: 较小值（0.0-0.1）使点更聚集，较大值（0.5-1.0）使点更分散

### 对比分析

通过比较不同模型（如 with_dropout vs without_dropout）的可视化结果，可以分析：

- **Dropout 的影响**: Dropout 是否提高了特征的判别性？
- **数据增强的影响**: UncertaintyMixup 等策略是否改善了类别分离？
- **模型架构的影响**: 不同深度的网络（ResNet18 vs ResNet50）学习到的特征有何差异？

## 示例命令

### 对比带/不带 Dropout 的模型

```bash
# Without Dropout
python vis_feature.py --model_type without_dropout --dataset_type chestct

# With Dropout
python vis_feature.py --model_type with_dropout --dataset_type chestct
```

### 对比不同数据增强策略

```bash
# No augmentation
python vis_feature.py --dataset_type chestct

# Mixup
python vis_feature.py --dataset_type chestct --use_augmentation --strategy mixup

# UncertaintyMixup
python vis_feature.py --dataset_type chestct --use_augmentation --strategy uncertaintymixup
```

### 对比不同模型架构

```bash
# ResNet18
python vis_feature.py --model_arch resnet18 --dataset_type chestct

# ResNet50
python vis_feature.py --model_arch resnet50 --dataset_type chestct
```

## 故障排除

### 问题 1: CUDA out of memory

**解决方案**: 减小 batch_size

```bash
python vis_feature.py --batch_size 16  # 或更小
```

### 问题 2: t-SNE 结果不稳定

**解决方案**: 
- 增加迭代次数：`--tsne_iter 1500`
- 固定随机种子（代码中已设置为 42）
- 尝试不同的 perplexity 值

### 问题 3: 模型参数文件不存在

**解决方案**: 
- 检查 `--model_path` 是否正确
- 确保模型已经训练并保存
- 程序会自动构建模型路径：`best_model[_strategy][_magnification].pth`

### 问题 4: 特征维度太高导致内存不足

**解决方案**: 
- 使用更小的模型架构
- 减少数据量（修改数据加载器）
- 使用 PCA 先降维到较低维度（如 50D），再用 t-SNE/UMAP

## 高级用法

### 自定义特征层

修改 `extract_features` 函数中的 `layer_name` 参数：

```python
features, labels, paths = extract_features(
    model, data_loader, device, 
    layer_name='layer3'  # 提取 layer3 的特征
)
```

### 3D 可视化

修改 t-SNE 和 UMAP 的参数：

```bash
python vis_feature.py --tsne_3d  # 需要在代码中设置 n_components=3
```

### 批量处理多个数据集

创建批处理脚本：

```bash
#!/bin/bash
for dataset in chestct breakhis padufes; do
    python vis_feature.py --dataset_type $dataset
done
```

## 参考文献

1. van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. JMLR.
2. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. arXiv.
3. Grad-CAM: https://github.com/jacobgil/pytorch-grad-cam

## 许可证

与主项目保持一致。
