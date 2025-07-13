# 文件: utils/io.py
import open3d as o3d
import os
import time

def create_pcd_stream_from_files(folder_path, start_index=200, frame_delay=0.1):
    """
    一个生成器，模拟从传感器传来的实时点云流。
    """
    try:
        pcd_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pcd')])
        if not pcd_files or start_index >= len(pcd_files):
            print(f"警告: 在'{folder_path}'中没有找到足够多的PCD文件。")
            return
    except FileNotFoundError:
        print(f"错误: 找不到PCD文件夹 '{folder_path}'。")
        return

    for pcd_file in pcd_files[start_index:]:
        pcd = o3d.io.read_point_cloud(pcd_file)
        if not pcd.has_points():
            continue
        yield pcd
        if frame_delay > 0:
            time.sleep(frame_delay)