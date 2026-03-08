import os
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def parse_filename(filename):
    """
    解析文件名，提取日期、序号等关键信息
    文件名格式: MAX_20240604_MAX_0168_Color_D.png
    或: MAX_20240604_001_MAX_0168_Color_D.png (带后缀)
    或: MAX_20240425_MAX_0001_850nm_D.tif
    """
    # 匹配模式: MAX_日期_[后缀_]MAX_序号_其他信息.扩展名
    pattern = r'MAX_(\d{8})(?:_(\d{3}))?_MAX_(\d{4})_(.+)'
    match = re.match(pattern, filename)
    
    if match:
        date = match.group(1)  # 日期，如 20240604
        suffix = match.group(2)  # 后缀，如 001，可能为None
        sequence = match.group(3)  # 序号，如 0168
        rest = match.group(4)  # 剩余部分，如 Color_D.png
        
        return {
            'date': date,
            'suffix': suffix,
            'sequence': sequence,
            'rest': rest,
            'full_date': f"{date}_{suffix}" if suffix else date
        }
    
    return None


def create_file_key(parsed_info):
    """
    根据解析的信息创建匹配键
    匹配规则：日期 + 日期后缀（如果有）+ 序号
    """
    if parsed_info:
        return f"{parsed_info['full_date']}_{parsed_info['sequence']}"
    return None


def find_date_folders(search_root, target_dates):
    """
    根据日期在搜索目录中找到可能包含目标文件的文件夹
    """
    date_folders = []
    search_path = Path(search_root)
    
    if not search_path.exists():
        print(f"搜索根目录不存在: {search_root}")
        return date_folders
    
    # 递归查找包含目标日期的文件夹
    print("正在扫描目录结构...")
    all_dirs = []
    for root, dirs, files in os.walk(search_root):
        if files:  # 只处理包含文件的目录
            all_dirs.append((root, files))
    
    # 使用进度条显示扫描进度
    for root, files in tqdm(all_dirs, desc="扫描文件夹", unit="folder"):
        # 检查当前文件夹中是否有匹配日期的文件
        for file in files:
            parsed = parse_filename(file)
            if parsed and parsed['date'] in target_dates:
                if root not in date_folders:
                    date_folders.append(root)
                break
    
    return date_folders


def find_matching_files(source_folder, search_root):
    """
    查找匹配的文件
    返回: (matched_groups, unmatched_files)
    """
    source_path = Path(source_folder)
    if not source_path.exists():
        print(f"源文件夹不存在: {source_folder}")
        return {}, []
    
    # 解析源文件夹中的所有文件
    source_files = {}
    target_dates = set()
    
    for file in source_path.iterdir():
        if file.is_file():
            parsed = parse_filename(file.name)
            if parsed:
                key = create_file_key(parsed)
                if key:
                    source_files[key] = {
                        'file_path': file,
                        'parsed_info': parsed
                    }
                    target_dates.add(parsed['date'])
    
    print(f"源文件夹中找到 {len(source_files)} 个有效文件")
    print(f"涉及日期: {sorted(target_dates)}")
    
    # 根据日期筛选搜索文件夹
    print("正在筛选包含目标日期的文件夹...")
    date_folders = find_date_folders(search_root, target_dates)
    print(f"找到 {len(date_folders)} 个可能包含匹配文件的文件夹")
    
    # 在筛选后的文件夹中查找匹配文件
    matched_groups = defaultdict(list)
    
    for folder in tqdm(date_folders, desc="查找匹配文件", unit="folder"):
        folder_path = Path(folder)
        
        for file in folder_path.iterdir():
            if file.is_file():
                parsed = parse_filename(file.name)
                if parsed:
                    key = create_file_key(parsed)
                    if key in source_files:
                        matched_groups[key].append(file)
    
    # 找出没有匹配的源文件
    unmatched_files = []
    for key, source_info in source_files.items():
        if key not in matched_groups:
            unmatched_files.append(source_info['file_path'])
    
    return matched_groups, unmatched_files


def copy_files_to_same_folder(source_folder, matched_groups):
    """
    将匹配的文件复制到same文件夹
    """
    source_path = Path(source_folder)
    same_folder = source_path.parent / "same"
    same_folder.mkdir(exist_ok=True)
    
    copied_count = 0
    
    for key, matched_files in tqdm(matched_groups.items(), desc="复制匹配文件", unit="group"):
        # 为每个匹配组创建子文件夹
        group_folder = same_folder / key
        group_folder.mkdir(exist_ok=True)
        
        # 复制源文件
        source_file = None
        for file in source_path.iterdir():
            if file.is_file():
                parsed = parse_filename(file.name)
                if parsed and create_file_key(parsed) == key:
                    source_file = file
                    break
        
        if source_file:
            dest_path = group_folder / source_file.name
            shutil.copy2(source_file, dest_path)
            copied_count += 1
        
        # 复制匹配的文件
        for matched_file in matched_files:
            dest_path = group_folder / matched_file.name
            shutil.copy2(matched_file, dest_path)
            copied_count += 1
    
    print(f"\n总共复制了 {copied_count} 个文件到 {same_folder}")


def move_unmatched_files(source_folder, unmatched_files):
    """
    将没有匹配的文件移动到no-compare文件夹
    """
    if not unmatched_files:
        print("所有文件都找到了匹配")
        return
    
    source_path = Path(source_folder)
    no_compare_folder = source_path.parent / "no-compare"
    no_compare_folder.mkdir(exist_ok=True)
    
    moved_count = 0
    
    for file_path in tqdm(unmatched_files, desc="移动未匹配文件", unit="file"):
        dest_path = no_compare_folder / file_path.name
        shutil.move(str(file_path), str(dest_path))
        moved_count += 1
    
    print(f"\n总共移动了 {moved_count} 个未匹配文件到 {no_compare_folder}")


def main():
    if len(sys.argv) != 3:
        print("使用方法: python match_spectral_files.py <源文件夹路径> <搜索根目录>")
        print("示例:")
        print("  源文件夹: \\\\192.168.3.155\\高光谱测试样本库\\原油检测\\00大庆现场测试\\03标注数据以及模型文件\\00数据和标签\\dataset_zcc\\整理\\train\\feedback\\images")
        print("  搜索目录: \\\\192.168.3.155\\高光谱测试样本库\\原油检测\\00大庆现场测试\\01数据\\01_S800")
        return
    
    source_folder = sys.argv[1]
    search_root = sys.argv[2]
    
    print("=" * 80)
    print("高光谱文件匹配工具")
    print("=" * 80)
    print(f"源文件夹: {source_folder}")
    print(f"搜索目录: {search_root}")
    print()
    
    # 查找匹配文件
    matched_groups, unmatched_files = find_matching_files(source_folder, search_root)
    
    print("\n" + "=" * 50)
    print("匹配结果统计:")
    print(f"找到匹配的文件组: {len(matched_groups)}")
    print(f"未匹配的源文件: {len(unmatched_files)}")
    
    if matched_groups:
        print("\n匹配详情:")
        for key, files in matched_groups.items():
            print(f"  {key}: {len(files)} 个匹配文件")
    
    if unmatched_files:
        print("\n未匹配的文件:")
        for file_path in unmatched_files:
            print(f"  {file_path.name}")
    
    # 询问是否继续处理
    print("\n" + "=" * 50)
    response = input("是否继续复制匹配文件和移动未匹配文件? (y/n): ").strip().lower()
    
    if response in ['y', 'yes', '是']:
        if matched_groups:
            copy_files_to_same_folder(source_folder, matched_groups)
        
        if unmatched_files:
            move_unmatched_files(source_folder, unmatched_files)
        
        print("\n处理完成!")
    else:
        print("操作已取消")


if __name__ == "__main__":
    main()