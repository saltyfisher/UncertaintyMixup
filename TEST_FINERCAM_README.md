# Test 文件夹 FinerCAM 显著图生成指南

## 功能说明

`vis_saliency_test.py` 脚本用于为 `test` 文件夹中的图像生成 FinerCAM 热力图。它会根据图像文件名中包含了模型策略名称（如 `comix`、`guided`、`puzzlemix`、`uncertaintymixup`）自动加载对应的预训练模型。

## 使用方法

### 基本用法

```bash
python vis_saliency_test.py
```

这将使用默认参数处理 `./test` 文件夹中的所有图像，并将结果保存到 `./test/finercam_results`。

### 自定义参数

```bash
python vis_saliency_test.py \
    --model_arch resnet18 \
    --model_type without_dropout \
    --test_dir ./test \
    --output_dir ./test/finercam_results \
    --methods finercam
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_arch` | `resnet18` | 模型架构（resnet18/resnet50/resnet34/inception/wideresnet/efficientnet_b0） |
| `--model_type` | `without_dropout` | 模型类型（with_dropout/without_dropout） |
| `--test_dir` | `./test` | test 文件夹路径 |
| `--output_dir` | `./test/finercam_results` | 热力图输出目录 |
| `--methods` | `finercam` | 可视化方法（gradcam/gradcam++/layercam/eigencam/finercam） |

## 文件名格式要求

脚本会从图像文件名中自动提取模型信息。支持的文件名格式示例：

- `exp_mixed_compare_comix_mixed1.png` → 使用 `comix` 策略模型
- `exp_mixed_compare_guided_mixed2.png` → 使用 `guided` 策略模型
- `exp_mixed_compare_puzzlemix_mixed3.png` → 使用 `puzzlemix` 策略模型
- `exp_mixed_compare_uncertaintymixup_matting_mixed4.png` → 使用 `uncertaintymixup_matting` 策略模型

对于 BreakHis 数据集，文件名中应包含放大倍数（如 `breakhis_40`、`breakhis_100` 等）。

## 模型文件命名规范

脚本会根据文件名自动构建模型路径：

- ChestCT 数据集：`best_model_{strategy}_chestct.pth`
- BreakHis 数据集：`best_model_{strategy}_breakhis_{magnification}.pth`

示例：
- `best_model_comix_chestct.pth`
- `best_model_guided_breakhis_100.pth`

## 输出说明

对于每个输入图像，脚本会生成一个热力图文件，并在图中显示：

- **真实标签** (True Label)
- **预测标签** (Pred Label)
- **预测置信度** (Prediction Confidence)

输出文件名格式：`{原文件名}_{method}.png`

例如：
- 输入：`exp_mixed_compare_comix_mixed1.png`
- 输出：`exp_mixed_compare_comix_mixed1_finercam.png`

## 注意事项

1. **确保模型文件存在**：脚本会根据文件名自动查找对应的模型文件，如果模型文件不存在，会跳过该图像并显示警告。

2. **GPU 加速**：如果系统中有可用的 GPU，脚本会自动使用 CUDA 进行加速。

3. **批量处理**：脚本会依次处理 test 文件夹中的所有图像，建议使用后台运行以避免中断。

4. **磁盘空间**：生成的热力图为高分辨率图像（300 DPI），请确保有足够的磁盘空间。

## 运行示例

### 后台运行（推荐）

```bash
nohup python vis_saliency_test.py > finercam_generation.log 2>&1 &
```

### 查看进度

```bash
tail -f finercam_generation.log
```

## 支持的可视化方法

- **Grad-CAM**: 经典的梯度加权类激活映射
- **Grad-CAM++**: Grad-CAM 的改进版本，更好的定位能力
- **Layer-CAM**: 从不同层提取特征的热力图方法
- **Eigen-CAM**: 基于特征分解的热力图方法
- **Finer-CAM**: 更精细的类激活映射方法（默认）

## 故障排除

### 问题：模型文件不存在

**解决方案**：检查模型文件是否在当前目录下，或者修改脚本中的模型路径。

### 问题：无法从文件名中提取策略

**解决方案**：确保文件名中包含以下策略之一：`comix`, `guided`, `puzzlemix`, `uncertaintymixup`, `uncertaintymixup_matting`

### 问题：内存不足

**解决方案**：减少同时处理的图像数量，或使用更小的批次大小。
