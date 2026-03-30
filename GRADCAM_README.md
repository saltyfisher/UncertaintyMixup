# Grad-CAM 热力图生成说明

## 功能概述

`vis_saliency.py` 脚本用于导入训练好的模型参数，使用 Grad-CAM、Grad-CAM++ 和 LayerCAM 等方法为训练集中的每个样本生成热力图，并保存可视化结果。

## 使用方法

### 基本用法

```bash
python vis_saliency.py --model_path <模型路径> --data_dir <数据集路径> --dataset_type <数据集类型>
```

### 完整参数示例

#### 1. ChestCT 数据集 (ResNet18)

```bash
python vis_saliency.py \
    --model_path best_model_chestct.pth \
    --data_dir /path/to/chestct/data \
    --dataset_type chestct \
    --model_arch resnet18 \
    --model_type without_dropout \
    --output_dir ./gradcam_results_chestct \
    --batch_size 1 \
    --num_workers 4
```

#### 2. BreakHis 数据集 (不同放大倍数)

```bash
python vis_saliency.py \
    --model_path best_model_breakhis_40X.pth \
    --data_dir /path/to/breakhis/data \
    --dataset_type breakhis \
    --magnification 40X \
    --model_arch resnet18 \
    --model_type without_dropout \
    --output_dir ./gradcam_results_breakhis_40X \
    --batch_size 1 \
    --num_workers 4
```

#### 3. 使用特定的 Grad-CAM 方法

```bash
python vis_saliency.py \
    --model_path best_model.pth \
    --data_dir /path/to/data \
    --dataset_type chestct \
    --methods gradcam \
    --output_dir ./gradcam_results_only_gradcam
```

## 参数说明

### 必需参数

- `--model_path`: 模型参数文件路径 (.pth 格式)
- `--data_dir`: 数据集根目录路径

### 可选参数

#### 模型相关

- `--model_arch`: 模型架构
  - 可选值：`resnet18`, `resnet50`, `resnet34`, `inception`, `wideresnet`, `efficientnet_b0`
  - 默认：`resnet18`

- `--model_type`: 模型类型
  - 可选值：`with_dropout`, `without_dropout`
  - 默认：`without_dropout`

#### 数据集相关

- `--dataset_type`: 数据集类型
  - 可选值：`chestct`, `breakhis`, `padufes`, `kvasir`, `bladder`
  - 默认：`chestct`

- `--magnification`: BreakHis 数据集的放大倍数
  - 可选值：`40X`, `100X`, `200X`, `400X`
  - 默认：`None`

#### 输出相关

- `--output_dir`: 热力图输出目录
  - 默认：`./gradcam_results`

- `--methods`: 要使用的 Grad-CAM 方法列表
  - 可选值：`gradcam`, `gradcam++`, `layercam`
  - 默认：`['gradcam', 'gradcam++', 'layercam']`

- `--save_individual`: 是否保存单独的热力图
  - 默认：`True`

- `--no_save_combined`: 不保存组合的热力图
  - 默认：`False`（即保存组合图）

#### 数据加载相关

- `--batch_size`: 批次大小
  - 默认：`1`

- `--num_workers`: 数据加载工作进程数
  - 默认：`4`

## 输出格式

### 目录结构

对于每个样本，会在输出目录中创建一个子目录，命名格式为：
```
sample_{序号:06d}_label_{标签编号}/
```

例如：
```
gradcam_results/
├── sample_000000_label_0/
│   ├── gradcam.png          # Grad-CAM 热力图
│   ├── gradcam++.png        # Grad-CAM++ 热力图
│   ├── layercam.png         # LayerCAM 热力图
│   └── combined.png         # 组合所有方法的对比图
├── sample_000001_label_1/
│   ├── gradcam.png
│   ├── gradcam++.png
│   ├── layercam.png
│   └── combined.png
└── ...
```

### 图像说明

1. **单独热力图** (`gradcam.png`, `gradcam++.png`, `layercam.png`)
   - 每张图都是原始图像与对应方法生成的热力图的叠加
   - 红色区域表示模型关注的重点区域

2. **组合热力图** (`combined.png`)
   - 横向排列：原始图像 + 各种方法的热力图
   - 便于直观对比不同方法的注意力区域差异

## 使用示例

### 示例 1: 比较带/不带 Dropout 的模型

```bash
# 不带 Dropout 的模型
python vis_saliency.py \
    --model_path best_model_without_dropout.pth \
    --data_dir ./data/chestct \
    --dataset_type chestct \
    --model_type without_dropout \
    --output_dir ./results_without_dropout

# 带 Dropout 的模型
python vis_saliency.py \
    --model_path best_model_with_dropout.pth \
    --data_dir ./data/chestct \
    --dataset_type chestct \
    --model_type with_dropout \
    --output_dir ./results_with_dropout
```

### 示例 2: 只使用 Grad-CAM 方法

```bash
python vis_saliency.py \
    --model_path best_model.pth \
    --data_dir ./data/chestct \
    --dataset_type chestct \
    --methods gradcam \
    --output_dir ./gradcam_only
```

### 示例 3: 处理 BreakHis 数据集

```bash
# 40X 放大倍数
python vis_saliency.py \
    --model_path best_model_breakhis_40X.pth \
    --data_dir ./data/BreakHis \
    --dataset_type breakhis \
    --magnification 40X \
    --output_dir ./breakhis_40X_gradcam

# 100X 放大倍数
python vis_saliency.py \
    --model_path best_model_breakhis_100X.pth \
    --data_dir ./data/BreakHis \
    --dataset_type breakhis \
    --magnification 100X \
    --output_dir ./breakhis_100X_gradcam
```

## 注意事项

1. **GPU 要求**: Grad-CAM 计算需要 GPU 支持以获得更快的速度
   - 脚本会自动检测并使用可用的 GPU
   - 如果没有 GPU，将使用 CPU（速度较慢）

2. **内存使用**: 
   - 建议保持 `batch_size=1` 以避免内存溢出
   - 可以根据显存大小调整 `num_workers`

3. **运行时间**: 
   - 取决于样本数量和分辨率
   - 对于大数据集，建议在 GPU 服务器上运行
   - 可以使用 `tqdm` 进度条监控处理进度

4. **模型参数格式**: 
   - 支持多种 checkpoint 格式（state_dict, model_state_dict 等）
   - 自动处理 DataParallel 包装的模型（移除'module.'前缀）

5. **输出文件大小**:
   - 每个样本会保存多张图像，注意磁盘空间
   - 可以通过 `--methods` 参数减少生成的方法数量来节省空间

## 依赖要求

确保安装了必要的依赖包：

```bash
pip install torch torchvision opencv-python numpy tqdm pytorch-grad-cam
```

或使用项目的 requirements.txt:

```bash
pip install -r requirements.txt
```

## 故障排除

### 常见问题

1. **CUDA out of memory**
   - 减少 `batch_size` 到 1
   - 减少 `num_workers`

2. **模型加载失败**
   - 检查模型路径是否正确
   - 确认模型参数文件格式正确
   - 检查模型架构和数据集类别数是否匹配

3. **ImportError: No module named 'models_'**
   - 确保在项目根目录下运行脚本
   - 或添加项目路径到 PYTHONPATH

4. **Grad-CAM 热力图全黑或异常**
   - 检查模型是否正确加载
   - 确认目标层选择是否正确
   - 验证输入图像的归一化方式

## 结果分析

### 热力图解读

- **红色/暖色调区域**: 模型高度关注的区域，对分类决策贡献大
- **蓝色/冷色调区域**: 模型较少关注的区域

### 对比分析

通过比较不同模型（带/不带 Dropout）的热力图，可以：
- 观察不确定性建模对模型注意力的影响
- 分析模型是否关注到了正确的病理特征
- 评估模型的可解释性和可靠性

## 扩展应用

生成的热力图可以用于：
1. **模型诊断**: 识别模型是否关注到错误区域
2. **特征分析**: 理解不同训练策略的影响
3. **结果展示**: 在论文或报告中可视化模型决策依据
4. **不确定性量化**: 结合 MC Dropout 分析预测置信度
