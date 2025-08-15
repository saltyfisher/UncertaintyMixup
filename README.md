# UncertaintyMixup 热力图可视化

本项目实现了使用 Grad-CAM、Grad-CAM++ 和 LayerCAM 对 ResNet18 模型进行热力图可视化，比较带有和不带有 Dropout 层的模型效果。

## 项目结构

```
UncertaintyMixup/
├── models.py          # 模型定义文件
├── datasets.py        # 数据加载和预处理
├── visualization.py   # 热力图可视化实现
├── main.py            # 主程序入口
├── train.py           # 训练脚本
├── test.py            # 测试脚本
├── utils.py           # 工具函数
├── requirements.txt   # 项目依赖
└── README.md          # 说明文档
```

## 环境依赖

- Python 3.6+
- PyTorch 1.7+
- torchvision
- OpenCV
- numpy
- Pillow
- pytorch-grad-cam
- tensorboardX
- scikit-learn

安装依赖:
```bash
pip install -r requirements.txt
```

## 使用方法

### 热力图可视化:

```bash
cd UncertaintyMixup
python main.py
```

可选参数:
- `--data_dir`: Tiny ImageNet 数据集路径 (默认: ../tiny-imagenet-200)
- `--output_dir`: 热力图输出目录 (默认: ./heatmaps)

### 模型训练:

```bash
python train.py --strategy uncertaintymixup --dataset chestct
```

可选参数:
- `--data_dir`: 数据集目录路径
- `--batch_size`: 批次大小 (默认: 16)
- `--num_epochs`: 训练轮数 (默认: 180)
- `--lr`: 学习率 (默认: 0.001)
- `--dropout_rate`: Dropout率 (默认: 0.5)
- `--optimizer`: 优化器类型 (默认: adam)
- `--strategy`: 训练策略 (默认: uncertaintymixup)
- `--dataset`: 数据集类型 (默认: chestct)
- `--num_trials`: 独立实验次数 (默认: 1)
- `--gpu`: GPU设备号 (默认: 0)
- `--save_mixed_results`: 是否保存混合结果可视化图像

### 模型测试:

```bash
python test.py --model_path best_model.pth
```

## 功能说明

1. 加载 PyTorch 预训练的 ResNet18 模型
2. 分别构建带 Dropout 和不带 Dropout 的模型版本
3. 从 Tiny ImageNet 数据集中随机选择 10 个样本
4. 使用以下三种方法生成热力图:
   - Grad-CAM
   - Grad-CAM++
   - LayerCAM
5. 分别对带 Dropout 和不带 Dropout 的模型进行可视化
6. 结果保存在 `heatmaps/with_dropout` 和 `heatmaps/without_dropout` 目录中

## 输出示例

输出文件命名格式:
```
sample_{序号}_target_{类别}_{方法}.png
```

例如:
- `sample_0_target_45_gradcam.png`
- `sample_3_target_123_gradcam++.png`
- `sample_7_target_67_layercam.png`

## 实现细节

- 使用 ImageNet 预训练的 ResNet18 权重
- Dropout 率设置为 0.5
- 图像预处理采用标准的 ImageNet 归一化参数
- 可视化目标层为 ResNet 的 layer4[-1] (最后一个残差块)