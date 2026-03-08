import os
import sys
from pathlib import Path
from collections import defaultdict


def get_all_filenames(source_folder):
    """
    读取源文件夹中的所有文件名
    """
    filenames = set()
    source_path = Path(source_folder)
    
    if not source_path.exists():
        print(f"源文件夹不存在: {source_folder}")
        return filenames
    
    for item in source_path.iterdir():
        if item.is_file():
            filenames.add(item.name)
    
    return filenames


def find_duplicate_files(target_directory, target_filenames):
    """
    在指定目录及其子目录中递归查找相同文件名的文件
    """
    found_files = defaultdict(list)
    target_path = Path(target_directory)
    
    if not target_path.exists():
        print(f"目标目录不存在: {target_directory}")
        return found_files
    
    for root, dirs, files in os.walk(target_directory):
        for file in files:
            if file in target_filenames:
                file_path = os.path.join(root, file)
                found_files[file].append(file_path)
    
    return found_files


def main():
    # 配置参数
    if len(sys.argv) == 3:
        source_folder = sys.argv[1]
        target_directory = sys.argv[2]
    else:
        try:
            source_folder = input("请输入源文件夹路径: ").strip()
            target_directory = input("请输入搜索目标目录路径: ").strip()
        except EOFError:
            print("使用方法: python find_duplicate_files.py <源文件夹路径> <目标搜索目录>")
            print("或者直接运行脚本并按提示输入路径")
            return
    
    print(f"\n正在读取源文件夹: {source_folder}")
    source_filenames = get_all_filenames(source_folder)
    
    if not source_filenames:
        print("源文件夹中没有找到任何文件")
        return
    
    print(f"找到 {len(source_filenames)} 个文件")
    
    print(f"\n正在搜索目标目录: {target_directory}")
    duplicate_files = find_duplicate_files(target_directory, source_filenames)
    
    if not duplicate_files:
        print("没有找到重复的文件")
        return
    
    print(f"\n找到 {len(duplicate_files)} 个重复文件:")
    print("=" * 50)
    
    for filename, paths in duplicate_files.items():
        print(f"\n文件名: {filename}")
        print(f"找到 {len(paths)} 个匹配:")
        for path in paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()