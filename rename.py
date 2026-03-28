import os
import re

def rename_files_in_results():
    # 检查results目录是否存在
    if not os.path.exists('results'):
        print("results目录不存在")
        return
    
    # 遍历results目录中的所有文件
    for filename in os.listdir('results'):
        # 匹配trial_results<name>_<timestamp>的模式
        # 假设时间戳是下划线后的一串数字和可能的点号(用于小数秒)
        match = re.match(r'trial_stats_(.+)_\d+_\d+\.csv$', filename)
        
        if match:
            # 提取中间部分作为新文件名
            new_name = match.group(1) + '.csv'
            old_path = os.path.join('results', filename)
            new_path = os.path.join('results', new_name)
            
            # 重命名文件
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_name}")
        else:
            # 如果不匹配模式，跳过该文件
            print(f"跳过文件: {filename} (不符合命名模式)")

if __name__ == "__main__":
    rename_files_in_results()