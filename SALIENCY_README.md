# 数据集显著图计算说明

本项目提供了计算BreakHis和ChestCT数据集谱残差显著图的功能。

## 文件说明

- `saliency_computation.py`: 通用的显著图计算脚本，支持多个数据集
- `compute_dataset_saliency.py`: 专门为BreakHis和ChestCT设计的简化脚本
- `SALIENCY_README.md`: 本说明文档

## 算法原理

使用**谱残差显著性检测算法**(Spectral Residual Saliency Detection)：
- 基于傅里叶变换的频域分析
- 通过计算图像频谱的对数残差来检测显著区域
- 能够有效识别图像中的异常或突出区域

## 使用方法

### 方法1: 使用简化脚本（推荐）

```bash
python compute_dataset_saliency.py \
    --breakhis_path /path/to/breakhis/data \
    --chestct_path /path/to/chestct/data \
    --output_dir ./saliency_results \
    --magnification 40X
```

### 方法2: 使用通用脚本

```bash
python saliency_computation.py \
    --data_root /path/to/data/root \
    --output_dir ./saliency_results \
    --datasets breakhis chestct \
    --magnification 40X
```

## 参数说明

### compute_dataset_saliency.py 参数

- `--breakhis_path`: BreakHis数据集路径（默认: ../data/breakhis）
- `--chestct_path`: ChestCT数据集路径（默认: ../data/chestct）
- `--output_dir`: 显著图输出目录（默认: ./saliency_maps）
- `--batch_size`: 批次大小（默认: 1，建议保持1）
- `--num_workers`: 数据加载工作进程数（默认: 4）
- `--magnification`: BreakHis放大倍数（40X, 100X, 200X, 400X，默认: 40X）

### saliency_computation.py 参数

- `--data_root`: 数据集根目录
- `--output_dir`: 输出目录
- `--datasets`: 要处理的数据集列表
- `--batch_size`: 批次大小
- `--num_workers`: 工作进程数
- `--magnification`: BreakHis放大倍数

## 输出格式

显著图以 `.npy` 格式保存，文件命名规则：
```
saliency_{序号:06d}_label_{标签编号}_{类别名}.npy
```

例如：
- `saliency_000000_label_0_adenosis.npy`
- `saliency_000123_label_2_normal.npy`

## 输出目录结构

```
saliency_results/
├── breakhis_train/
│   ├── saliency_000000_label_0_adenosis.npy
│   ├── saliency_000001_label_1_fibroadenoma.npy
│   └── ...
├── breakhis_test/
│   ├── saliency_000000_label_0_adenosis.npy
│   └── ...
├── chestct_train/
│   ├── saliency_000000_label_0_covid.npy
│   └── ...
└── chestct_test/
    ├── saliency_000000_label_0_covid.npy
    └── ...
```

## 数据加载说明

### BreakHis数据集
- 支持指定放大倍数（40X, 100X, 200X, 400X）
- 自动处理数据集的层次结构
- 8个类别：4个良性（adenosis, fibroadenoma, phyllodes_tumor, tubular_adenoma）+ 4个恶性（ductal_carcinoma, lobular_carcinoma, mucinous_carcinoma, papillary_carcinoma）

### ChestCT数据集
- 4个类别：COVID, Normal, Lung_Opacity, Viral_Pneumonia
- 标准的train/test目录结构

## 使用示例

### 基本使用
```bash
# 使用默认路径
python compute_dataset_saliency.py

# 指定自定义路径
python compute_dataset_saliency.py \
    --breakhis_path /data/BreaKHis_v1 \
    --chestct_path /data/chest_xray \
    --output_dir /results/saliency_maps
```

### 查看帮助
```bash
python compute_dataset_saliency.py --help
python saliency_computation.py --help
```

## 注意事项

1. **内存使用**: 显著图计算相对轻量，但大量图像处理仍需注意内存使用
2. **运行时间**: 取决于图像数量和分辨率，建议在GPU服务器上运行
3. **数据格式**: 脚本会自动处理不同格式的输入图像（灰度、RGB、RGBA）
4. **错误处理**: 遇到无法处理的图像会跳过并记录错误信息

## 加载显著图示例

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载显著图
saliency_map = np.load('saliency_000000_label_0_adenosis.npy')

# 可视化
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(saliency_map, cmap='hot')
plt.title('Saliency Map')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.hist(saliency_map.flatten(), bins=50)
plt.title('Saliency Distribution')
plt.show()
```

## 依赖要求

确保安装了必要的依赖包：
```bash
pip install opencv-python torch torchvision numpy tqdm
```

## 故障排除

常见问题及解决方案：

1. **OpenCV错误**: 确保安装了opencv-contrib-python包
2. **内存不足**: 减少batch_size或num_workers
3. **路径错误**: 检查数据集路径是否正确
4. **权限问题**: 确保输出目录有写入权限

## 性能优化建议

1. 使用SSD存储显著图文件以提高I/O性能
2. 在多核CPU上增加num_workers参数
3. 对于大数据集，考虑分批处理
4. 使用进度条监控处理进度