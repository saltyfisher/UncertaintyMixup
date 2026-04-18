#!/bin/bash

# Test 文件夹 FinerCAM 显著图生成脚本
# 该脚本会自动处理 test 文件夹中的所有图像，并根据文件名中的模型名称加载对应模型

echo "=================================================="
echo "FinerCAM 显著图生成器 - Test 文件夹"
echo "=================================================="

# 设置参数
TEST_DIR="./test"
OUTPUT_DIR="./test/finercam_results"
MODEL_ARCH="resnet18"
MODEL_TYPE="without_dropout"
METHOD="finercam"

# 检查 test 文件夹是否存在
if [ ! -d "$TEST_DIR" ]; then
    echo "错误：test 文件夹不存在：$TEST_DIR"
    exit 1
fi

# 统计图像文件数量
IMAGE_COUNT=$(find "$TEST_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
echo "发现 $IMAGE_COUNT 个图像文件"

if [ $IMAGE_COUNT -eq 0 ]; then
    echo "警告：test 文件夹中没有找到图像文件"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo ""
echo "配置信息："
echo "  - Test 文件夹：$TEST_DIR"
echo "  - 输出目录：$OUTPUT_DIR"
echo "  - 模型架构：$MODEL_ARCH"
echo "  - 模型类型：$MODEL_TYPE"
echo "  - 可视化方法：$METHOD"
echo ""

# 运行 Python 脚本
echo "开始生成 FinerCAM 热力图..."
echo "=================================================="

python vis_saliency_test.py \
    --model_arch "$MODEL_ARCH" \
    --model_type "$MODEL_TYPE" \
    --test_dir "$TEST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --methods "$METHOD"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ FinerCAM 热力图生成完成！"
    echo "=================================================="
    echo ""
    echo "结果保存在：$OUTPUT_DIR"
    echo ""
    
    # 统计生成的文件数量
    OUTPUT_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*.png" | wc -l)
    echo "共生成 $OUTPUT_COUNT 个热力图文件"
else
    echo ""
    echo "=================================================="
    echo "✗ 错误：FinerCAM 热力图生成失败"
    echo "=================================================="
    exit 1
fi
