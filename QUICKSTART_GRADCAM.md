# Grad-CAM 热力图生成 - 快速开始

## 最简单的使用方式

假设你已经有一个训练好的模型，想要为训练集样本生成热力图：

```bash
python vis_saliency.py \
    --model_path best_model_chestct.pth \
    --data_dir /path/to/your/chestct/data \
    --dataset_type chestct \
    --output_dir ./gradcam_results
```

就这么简单！脚本会自动：
1. 加载你的模型参数
2. 加载训练集数据
3. 使用 Grad-CAM、Grad-CAM++、LayerCAM 三种方法生成热力图
4. 保存每个样本的单独热力图和组合对比图

## 常用场景

### 场景 1: ChestCT 数据集

```bash
python vis_saliency.py \
    --model_path best_model_chestct.pth \
    --data_dir ../chest-ctscan-images_datasets \
    --dataset_type chestct \
    --output_dir ./chestct_gradcam
```

### 场景 2: BreakHis 数据集（40X 放大）

```bash
python vis_saliency.py \
    --model_path best_model_breakhis_40.pth \
    --data_dir ../BreakHis \
    --dataset_type breakhis \
    --magnification 40X \
    --output_dir ./breakhis_40x_gradcam
```

### 场景 3: 只使用 Grad-CAM（不生成其他方法）

```bash
python vis_saliency.py \
    --model_path best_model.pth \
    --data_dir /path/to/data \
    --dataset_type chestct \
    --methods gradcam \
    --output_dir ./gradcam_only
```

## 查看帮助信息

```bash
python vis_saliency.py --help
```

## 输出结果说明

对于每个训练集样本，你会得到类似这样的目录结构：

```
./gradcam_results/
├── sample_000000_label_0/      # 第 1 个样本，类别 0
│   ├── gradcam.png             # Grad-CAM 热力图
│   ├── gradcam++.png           # Grad-CAM++ 热力图
│   ├── layercam.png            # LayerCAM 热力图
│   └── combined.png            # 四种方法对比图
├── sample_000001_label_1/      # 第 2 个样本，类别 1
│   ├── gradcam.png
│   ├── gradcam++.png
│   ├── layercam.png
│   └── combined.png
└── ...
```

## 推荐参数

- **batch_size**: 保持为 1（避免内存问题）
- **num_workers**: 根据你的 CPU 核心数设置（推荐 4）
- **methods**: 如果只需要一种方法，可以指定 `--methods gradcam`

## 运行时间估算

- 每张图像约 1-3 秒（取决于 GPU 性能）
- 1000 张图像约需 15-50 分钟
- 建议使用 GPU 加速

## 常见问题

**Q: 没有 GPU 怎么办？**  
A: 脚本会自动使用 CPU，但速度会慢很多。

**Q: 显存不足怎么办？**  
A: 确保 `--batch_size 1`，这是默认值。

**Q: 如何中断处理？**  
A: 按 `Ctrl+C` 可以随时中断，已处理的样本会保留。

**Q: 输出目录已存在怎么办？**  
A: 脚本会自动在现有目录中追加新样本，不会覆盖。
