import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def get_file_extension(filename):
    """
    获取文件扩展名（小写）
    """
    return Path(filename).suffix.lower()


def scan_folders_for_files(same_folder, no_compare_folder):
    """
    扫描same和no-compare文件夹，收集所有文件
    返回按扩展名分类的文件字典
    """
    files_by_extension = defaultdict(list)
    
    # 扫描same文件夹
    if Path(same_folder).exists():
        print(f"正在扫描same文件夹: {same_folder}")
        same_files = []
        for root, dirs, files in os.walk(same_folder):
            for file in files:
                file_path = Path(root) / file
                same_files.append(file_path)
        
        for file_path in tqdm(same_files, desc="处理same文件夹文件", unit="file"):
            if file_path.is_file():
                ext = get_file_extension(file_path.name)
                files_by_extension[ext].append({
                    'path': file_path,
                    'source': 'same'
                })
    else:
        print(f"same文件夹不存在: {same_folder}")
    
    # 扫描no-compare文件夹
    if Path(no_compare_folder).exists():
        print(f"正在扫描no-compare文件夹: {no_compare_folder}")
        no_compare_files = []
        for root, dirs, files in os.walk(no_compare_folder):
            for file in files:
                file_path = Path(root) / file
                no_compare_files.append(file_path)
        
        for file_path in tqdm(no_compare_files, desc="处理no-compare文件夹文件", unit="file"):
            if file_path.is_file():
                ext = get_file_extension(file_path.name)
                files_by_extension[ext].append({
                    'path': file_path,
                    'source': 'no-compare'
                })
    else:
        print(f"no-compare文件夹不存在: {no_compare_folder}")
    
    return files_by_extension


def create_extension_folders(output_folder, extensions):
    """
    创建按扩展名分类的文件夹
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    created_folders = {}
    for ext in extensions:
        # 去掉点号，如果扩展名为空则使用"no_extension"
        folder_name = ext[1:] if ext.startswith('.') and len(ext) > 1 else "no_extension"
        if folder_name == "":
            folder_name = "no_extension"
        
        ext_folder = output_path / folder_name
        ext_folder.mkdir(exist_ok=True)
        created_folders[ext] = ext_folder
    
    return created_folders


def copy_files_by_extension(files_by_extension, output_folder):
    """
    将文件按扩展名复制到对应文件夹
    """
    print(f"\n正在创建输出文件夹: {output_folder}")
    extension_folders = create_extension_folders(output_folder, files_by_extension.keys())
    
    copy_stats = defaultdict(int)
    total_files = sum(len(files) for files in files_by_extension.values())
    
    with tqdm(total=total_files, desc="复制文件", unit="file") as pbar:
        for ext, files in files_by_extension.items():
            target_folder = extension_folders[ext]
            
            for file_info in files:
                source_path = file_info['path']
                source_type = file_info['source']
                
                # 生成目标文件名，避免重名冲突
                target_name = source_path.name
                target_path = target_folder / target_name
                
                # 如果文件已存在，添加序号后缀
                counter = 1
                while target_path.exists():
                    name_parts = source_path.stem, counter, source_path.suffix
                    target_name = f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                    target_path = target_folder / target_name
                    counter += 1
                
                # 复制文件
                try:
                    shutil.copy2(source_path, target_path)
                    copy_stats[ext] += 1
                    copy_stats[f"{ext}_from_{source_type}"] += 1
                except Exception as e:
                    print(f"复制文件失败: {source_path} -> {target_path}, 错误: {e}")
                
                pbar.update(1)
    
    return copy_stats


def print_statistics(files_by_extension, copy_stats):
    """
    打印统计信息
    """
    print("\n" + "=" * 60)
    print("文件分类统计:")
    print("=" * 60)
    
    total_files = 0
    for ext, files in sorted(files_by_extension.items()):
        if not ext:
            ext_name = "无扩展名"
        else:
            ext_name = ext
        
        same_count = sum(1 for f in files if f['source'] == 'same')
        no_compare_count = sum(1 for f in files if f['source'] == 'no-compare')
        
        print(f"{ext_name:>15}: {len(files):>4} 个文件 (same: {same_count}, no-compare: {no_compare_count})")
        total_files += len(files)
    
    print("-" * 60)
    print(f"{'总计':>15}: {total_files:>4} 个文件")
    
    print("\n" + "=" * 60)
    print("复制结果统计:")
    print("=" * 60)
    
    for ext in sorted(files_by_extension.keys()):
        copied = copy_stats.get(ext, 0)
        if copied > 0:
            ext_name = ext if ext else "无扩展名"
            print(f"{ext_name:>15}: {copied:>4} 个文件已复制")


def main():
    if len(sys.argv) != 4:
        print("使用方法: python merge_and_classify_files.py <same文件夹路径> <no-compare文件夹路径> <输出文件夹路径>")
        print("示例:")
        print("  same文件夹: \\\\192.168.3.155\\高光谱测试样本库\\原油检测\\00大庆现场测试\\03标注数据以及模型文件\\00数据和标签\\dataset_zcc\\整理\\train\\feedback\\same")
        print("  no-compare文件夹: \\\\192.168.3.155\\高光谱测试样本库\\原油检测\\00大庆现场测试\\03标注数据以及模型文件\\00数据和标签\\dataset_zcc\\整理\\train\\feedback\\no-compare")
        print("  输出文件夹: \\\\192.168.3.155\\高光谱测试样本库\\原油检测\\00大庆现场测试\\03标注数据以及模型文件\\00数据和标签\\dataset_zcc\\整理\\val")
        return
    
    same_folder = sys.argv[1]
    no_compare_folder = sys.argv[2]
    output_folder = sys.argv[3]
    
    print("=" * 80)
    print("文件合并和分类工具")
    print("=" * 80)
    print(f"same文件夹: {same_folder}")
    print(f"no-compare文件夹: {no_compare_folder}")
    print(f"输出文件夹: {output_folder}")
    print()
    
    # 扫描文件夹，按扩展名分类
    files_by_extension = scan_folders_for_files(same_folder, no_compare_folder)
    
    if not files_by_extension:
        print("没有找到任何文件")
        return
    
    # 显示统计信息
    print_statistics(files_by_extension, {})
    
    # 询问是否继续
    print("\n" + "=" * 50)
    response = input("是否继续复制文件到按扩展名分类的文件夹? (y/n): ").strip().lower()
    
    if response in ['y', 'yes', '是']:
        # 复制文件
        copy_stats = copy_files_by_extension(files_by_extension, output_folder)
        
        # 显示最终统计
        print_statistics(files_by_extension, copy_stats)
        print(f"\n文件已成功分类复制到: {output_folder}")
    else:
        print("操作已取消")


if __name__ == "__main__":
    main()