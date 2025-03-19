import os
import random
import string
from multiprocessing import Pool, cpu_count, Manager
import time

def generate_random_string(length=20):
    """生成指定长度的随机字符串，由小写字母和数字组成"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def rename_video(file_info):
    """重命名单个视频文件"""
    old_path, root, progress_dict = file_info
    extension = os.path.splitext(old_path)[1]
    new_name = f"v1{generate_random_string(20)}{extension}"
    new_path = os.path.join(root, new_name)
    os.rename(old_path, new_path)
    
    # 更新进度
    with progress_dict["lock"]:
        progress_dict["completed"] += 1
        print(f"\r已完成: {progress_dict['completed']} / {progress_dict['total']} 个文件", end='', flush=True)
    
    return f"重命名: {old_path} -> {new_path}"

def collect_video_files(folder_path, video_extensions):
    """收集所有视频文件路径"""
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append((os.path.join(root, file), root))
    return video_files

def main(folder_path):
    # 支持的视频文件扩展名
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.m4v')

    # 收集所有视频文件路径
    video_files = collect_video_files(folder_path, video_extensions)
    total_files = len(video_files)
    print(f"找到 {total_files} 个视频文件，开始重命名...")

    if total_files == 0:
        print("没有需要重命名的视频文件。")
        return

    # 共享变量用于进度更新
    manager = Manager()
    progress_dict = manager.dict()
    progress_dict["completed"] = 0
    progress_dict["total"] = total_files
    progress_dict["lock"] = manager.Lock()

    start_time = time.time()  # 开始计时

    # 包装文件信息以包含进度字典
    video_files_with_progress = [(file[0], file[1], progress_dict) for file in video_files]

    # 多进程并行处理
    with Pool(processes=cpu_count()) as pool:
        pool.map(rename_video, video_files_with_progress)

    end_time = time.time()  # 结束计时

    print("\n重命名完成！")
    total_time = end_time - start_time
    avg_time = total_time / total_files if total_files > 0 else 0
    print(f"总耗时: {total_time:.2f} 秒，平均每个文件耗时: {avg_time:.4f} 秒")

if __name__ == "__main__":
    # 请将 'your_folder_path' 替换为目标文件夹的路径
    folder_path = r"/storage/zhubin/Janus-MoE/videos_clip_v6_20241203"
    if os.path.isdir(folder_path):
        main(folder_path)
    else:
        print(f"路径无效: {folder_path}")
