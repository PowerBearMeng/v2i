# 文件: utils/io.py
import open3d as o3d
import os
import time
import numpy as np


def read_ply_file(file_path):
    """
    使用 Open3D 读取 .ply 文件，自动处理 ASCII 和二进制格式。
    
    返回:
        o3d.geometry.PointCloud 对象，如果失败则返回 None。
    """
    try:
        # Open3D 的 read_point_cloud 会自动处理二进制和ASCII格式
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            print(f"[警告] 文件 {os.path.basename(file_path)} 为空或读取失败。")
            return None
        return pcd
    except Exception as e:
        print(f"[错误] 使用 Open3D 读取文件 {file_path} 失败: {e}")
        return None
    

def create_ply_stream_from_files(folder_path, start_index=0, frame_delay=0.1):
    """
    一个生成器，使用 Open3D 读取 .ply 文件来模拟实时点云流。
    """
    try:
        # 筛选出 .ply 文件
        ply_files = sorted([
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) if f.endswith('.ply')
        ])
        if not ply_files or start_index >= len(ply_files):
            print(f"警告: 在'{folder_path}'中没有找到足够多的PLY文件。")
            return
    except FileNotFoundError:
        print(f"错误: 找不到PLY文件夹 '{folder_path}'。")
        return

    print(f"找到 {len(ply_files)} 个PLY文件，将从第 {start_index} 个开始处理...")
    for ply_file in ply_files[start_index:]:
        # --- 使用我们新的、正确的PLY读取函数 ---
        pcd = read_ply_file(ply_file)
        
        if pcd is None:  # 如果读取失败或文件为空，则跳过
            continue
            
        yield pcd
        
        if frame_delay > 0:
            time.sleep(frame_delay)