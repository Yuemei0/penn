import shutil
from pathlib import Path

# def copy_dataset_structure(src_dir, dst_dir):
#     """
#     递归复制 src_dir 下的所有文件到 dst_dir，保持原目录结构。
    
#     Args:
#         src_dir (str or Path): 原始数据根目录
#         dst_dir (str or Path): 目标数据根目录
#     """
#     src_dir = Path(src_dir)
#     dst_dir = Path(dst_dir)
    
#     # 遍历源目录下所有文件
#     for file_path in src_dir.rglob("*"):
#         if file_path.is_file():
#             # 计算相对路径
#             rel_path = file_path.relative_to(src_dir)
#             # 构建目标文件路径
#             dst_file_path = dst_dir / rel_path
#             # 创建目标目录
#             dst_file_path.parent.mkdir(parents=True, exist_ok=True)
#             # 复制文件
#             shutil.copy2(file_path, dst_file_path)
#             print(f"Copied {file_path} -> {dst_file_path}")

# # 使用示例
# copy_dataset_structure("/data/lym/F0_SOTA/penn/data/AUG_datasets/ptdb/SPEECH DATA/MALE/REF_NEW", "/data/lym/F0_SOTA/penn/data/AUG_datasets/ptdb/SPEECH DATA/MALE/REF")



# from pathlib import Path

# def delete_f0_files(target_dir):
#     """
#     删除 target_dir 下（包括子目录）的所有 .f0 文件
#     """
#     target_dir = Path(target_dir)
#     for file_path in target_dir.rglob("*.f0"):
#         try:
#             file_path.unlink()
#             print(f"Deleted: {file_path}")
#         except Exception as e:
#             print(f"Failed to delete {file_path}: {e}")

# # 使用示例
# delete_f0_files("/data/lym/F0_SOTA/penn/data/AUG_datasets/ptdb/SPEECH DATA")

# from pathlib import Path

# def delete_aug_files(target_dir):
#     """
#     删除 target_dir 下（包括子目录）所有以 '_aug' 结尾的文件
#     """
#     target_dir = Path(target_dir)
#     for file_path in target_dir.rglob("*_aug.wav"):
#         try:
#             file_path.unlink()
#             print(f"Deleted: {file_path}")
#         except Exception as e:
#             print(f"Failed to delete {file_path}: {e}")

# # 使用示例
# delete_aug_files("/data/lym/F0_SOTA/penn/data/AUG_datasets/mdb/audio_stems")

# 直接复制ref_new存在命名不一致，将Mic前缀替换
# import os
# from pathlib import Path

# def rename_files_recursively(target_dir):
#     # 确保路径是 Path 对象
#     target_dir = Path(target_dir)
    
#     # 使用 rglob 递归搜索所有子文件夹中以 ref_ 开头的文件
#     files = list(target_dir.rglob('ref_*'))
    
#     if not files:
#         print("未找到任何以 ref_ 开头的文件。")
#         return

#     print(f"在 {target_dir} 及其子目录下找到 {len(files)} 个文件，准备重命名...")

#     for file_path in files:
#         old_name = file_path.name
        
#         # 仅替换文件名前缀
#         new_name = old_name.replace('ref_', 'mic_', 1)
        
#         # 构造新路径（保持在原有的子文件夹内）
#         new_path = file_path.with_name(new_name)
        
#         try:
#             file_path.rename(new_path)
#             # 打印相对路径，方便观察
#             print(f"重命名: {file_path.relative_to(target_dir)} -> {new_name}")
#         except Exception as e:
#             print(f"跳过文件 {old_name}，原因: {e}")

# if __name__ == "__main__":
#     # 替换为你实际的顶层文件夹路径
#     TARGET_DIRECTORY = '/data/lym/F0_SOTA/penn/data/AUG_datasets/ptdb/SPEECH DATA/MALE/REF'
#     rename_files_recursively(TARGET_DIRECTORY)


# 给aug版本的数据统一复制标注
import shutil
from pathlib import Path
from tqdm import tqdm

def backup_and_rename_labels(target_dir):
    """
    遍历目录及其子目录，将不以 _aug.csv 结尾的文件
    复制一份并命名为 原文件名_aug.csv
    """
    target_dir = Path(target_dir)
    if not target_dir.exists():
        print(f"错误：路径 {target_dir} 不存在")
        return

    # rglob('*') 遍历所有文件
    # 过滤掉已经是 _aug.csv 结尾的文件，以及文件夹本身
    files_to_process = [
        f for f in target_dir.rglob('*') 
        if f.is_file() and not f.name.endswith('_aug.csv')
    ]

    print(f"找到 {len(files_to_process)} 个待处理文件...")

    for file_path in tqdm(files_to_process, desc="复制并重命名"):
        # 构造新文件名：原文件名 + _aug.csv
        # 注意：这里使用 .name 包含原始后缀（如 ref_01.csv -> ref_01.csv_aug.csv）
        # 如果你想去掉原后缀（如 ref_01.csv -> ref_01_aug.csv），请使用 .stem
        new_name = f"{file_path.stem}_aug.csv"
        new_path = file_path.with_name(new_name)

        # 检查目标文件是否已存在，防止覆盖
        if not new_path.exists():
            try:
                shutil.copy2(file_path, new_path) # copy2 保留元数据
            except Exception as e:
                print(f"无法复制文件 {file_path.name}: {e}")
        else:
            # 如果已经存在同名 _aug.csv，则跳过
            pass

if __name__ == "__main__":
    # 设置你的目标根目录
    TARGET_DIR = '/data/lym/F0_SOTA/penn/data/AUG_datasets/ptdb/SPEECH DATA/MALE/REF'
    backup_and_rename_labels(TARGET_DIR)