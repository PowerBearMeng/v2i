# 文件: utils/io.py
import open3d as o3d
import os
import time
import numpy as np

def read_bin_to_xyzi(file_path):
    """
    读取 KITTI 格式的二进制 .bin 文件 (float32, N*4)。
    返回 N x 4 的 NumPy 数组 [x, y, z, intensity]。
    """
    try:
        # 读取原始二进制数据
        points = np.fromfile(file_path, dtype=np.float32)
        # 重塑为 N x 4 (x, y, z, intensity)
        return points.reshape(-1, 4)
    except Exception as e:
        print(f"[错误] 读取 BIN 文件 {file_path} 失败: {e}")
        return None
    
# --- USER'S FUNCTION ADDED HERE ---
def read_pcd_with_intensity(file_path):
    """
    手动读取包含 XYZI 的二进制 PCD 文件，返回 N x 4 的 NumPy 数组。
    """
    try:
        with open(file_path, 'rb') as f:
            header = []
            while True:
                line = f.readline().decode('ascii').strip()
                header.append(line)
                if line.startswith('DATA binary'):
                    break
            
            # 从头部解析点数
            points_line = [line for line in header if line.startswith('POINTS')]
            num_points = int(points_line[0].split(' ')[1])

            # 读取二进制数据
            data = np.fromfile(f, dtype=np.float32)
            # 根据点数和字段数（XYZI=4）重塑数组
            if data.size == num_points * 4:
                return data.reshape(num_points, 4)
            else:
                print(f"[警告] 文件 {os.path.basename(file_path)} 的数据大小与头部信息不匹配。跳过此文件。")
                return None

    except Exception as e:
        print(f"[错误] 读取文件 {file_path} 失败: {e}")
        return None

# --- NEW HELPER FUNCTION to convert NumPy array to Open3D PointCloud ---
def _numpy_array_to_pcd(xyzi_data):
    """
    将一个 N x 4 的 NumPy 数组 (XYZI) 转换为 Open3D 点云对象。
    强度信息将被存储在颜色通道中以便后续处理和保存。
    """
    if xyzi_data is None or xyzi_data.shape[1] != 4:
        return o3d.geometry.PointCloud()

    pcd = o3d.geometry.PointCloud()
    
    # XYZ是前三列
    pcd.points = o3d.utility.Vector3dVector(xyzi_data[:, :3])
    
    # 强度是第四列
    intensity = xyzi_data[:, 3]
    
    # Open3D 使用 .colors (Vector3dVector) 来存储每个点的颜色。
    # 我们将强度值放入第一个颜色通道（R），以便我们的保存函数可以提取它。
    # 创建一个 N x 3 的零矩阵，然后将强度填充到第一列。
    colors = np.zeros((intensity.shape[0], 3))
    colors[:, 0] = intensity / 255.0  # 假设强度范围是0-255，归一化到0-1
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

# --- MODIFIED GENERATOR FUNCTION ---
def create_pcd_stream_from_files(folder_path, start_index=200, frame_delay=0.1, type_filter='.bin'):
    """
    一个生成器，使用自定义的XYZI读取器来模拟实时点云流。
    """
    try:
        all_files = sorted(os.listdir(folder_path))
        point_cloud_files = [
            os.path.join(folder_path, f) 
            for f in all_files 
            if f.endswith(type_filter)
        ]
        if not point_cloud_files or start_index >= len(point_cloud_files):
            print(f"警告: 在'{folder_path}'中没有找到足够多的点云文件{type_filter}。")
            return
    except FileNotFoundError:
        print(f"错误: 找不到PCD文件夹 '{folder_path}'。")
        return

    for pcd_file in point_cloud_files[start_index:]:
        # --- MODIFIED LOGIC: Use the new reader and converter ---
        xyzi_data = None
        if type_filter == '.bin':
            xyzi_data = read_bin_to_xyzi(pcd_file)
        elif type_filter == '.pcd': 
            xyzi_data = read_pcd_with_intensity(pcd_file)
        if xyzi_data is None:
            continue
            
        pcd = _numpy_array_to_pcd(xyzi_data)
        # --------------------------------------------------------
        
        if not pcd.has_points():
            continue
            
        yield pcd
        
        if frame_delay > 0:
            time.sleep(frame_delay)

# --- The save function remains the same as before ---
# def save_pcd_with_intensity(file_path, pcd):
#     """
#     Saves an Open3D point cloud to a .pcd file in binary format with XYZ and Intensity fields.
#     """
#     if not pcd.has_points():
#         return

#     points = np.asarray(pcd.points)
#     num_points = points.shape[0]

#     if pcd.has_colors():
#         # 我们的加载器已经将强度放在了第一个颜色通道并归一化了
#         # 现在我们把它取出来并恢复（如果需要的话）
#         intensity = np.asarray(pcd.colors)[:, 0:1] * 255.0 # 乘以255恢复
        
#         xyzi = np.hstack((points, intensity)).astype(np.float32)
        
#         header = f"""# .PCD v0.7 - Point Cloud Data file format
# VERSION 0.7
# FIELDS x y z intensity
# SIZE 4 4 4 4
# TYPE F F F F
# COUNT 1 1 1 1
# WIDTH {num_points}
# HEIGHT 1
# VIEWPOINT 0 0 0 1 0 0 0
# POINTS {num_points}
# DATA binary
# """
#     else:
#         # Fallback if no color/intensity data exists
#         intensity = np.zeros((num_points, 1), dtype=np.float32)
#         xyzi = np.hstack((points, intensity)).astype(np.float32)
        
#         header = f"""# .PCD v0.7 - Point Cloud Data file format
# VERSION 0.7
# FIELDS x y z intensity
# SIZE 4 4 4 4
# TYPE F F F F
# COUNT 1 1 1 1
# WIDTH {num_points}
# HEIGHT 1
# VIEWPOINT 0 0 0 1 0 0 0
# POINTS {num_points}
# DATA binary
# """
#     try:
#         with open(file_path, 'wb') as f:
#             f.write(header.encode('ascii'))
#             f.write(xyzi.tobytes())
#     except Exception as e:
#         print(f"[Error] Failed to write PCD file {file_path}: {e}")