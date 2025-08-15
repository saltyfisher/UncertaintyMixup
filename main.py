import torch
import os
import sys
import argparse
import shutil

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization import run_visualization, run_comparison_visualization


def main():
    parser = argparse.ArgumentParser(description='ResNet18热力图可视化')
    parser.add_argument('--data_dir', type=str, default='../tiny-imagenet-200',
                        help='Tiny ImageNet数据集路径')
    parser.add_argument('--output_dir', type=str, default='./heatmaps',
                        help='热力图输出目录')
    parser.add_argument('--mode', type=str, default='comparison',
                        choices=['individual', 'comparison'],
                        help='运行模式: individual(分别生成) 或 comparison(对比生成)')
    
    args = parser.parse_args()
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录 {args.data_dir} 不存在")
        return
    
    # 在运行前删除之前保存的结果
    if os.path.exists(args.output_dir):
        print(f"删除之前的输出目录: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    if args.mode == 'individual':
        # 为不带dropout的模型生成热力图
        print("为不带dropout的ResNet18模型生成热力图...")
        run_visualization(args.data_dir, args.output_dir, model_type='without_dropout')
        
        # 为带dropout的模型生成热力图
        print("为带dropout的ResNet18模型生成热力图...")
        run_visualization(args.data_dir, args.output_dir, model_type='with_dropout')
        
        print("\n所有热力图已生成完毕！")
        print(f"结果保存在 {os.path.abspath(args.output_dir)} 目录中")
    else:
        # 运行对比可视化
        print("运行带dropout和不带dropout模型的对比可视化...")
        run_comparison_visualization(args.data_dir, args.output_dir)
        
        print("\n所有对比热力图已生成完毕！")
        print(f"结果保存在 {os.path.abspath(args.output_dir)} 目录中")


if __name__ == "__main__":
    main()