import open3d as o3d
import numpy as np
import argparse
import os
import glob

def pcd_to_bin(pcd_file, bin_file):
    """
    单个.pcd文件を読み込み、Kitti形式の.binファイルに変換します。
    点群データは (x, y, z, intensity) の形式で保存されます。
    intensity情報がない場合は、0で埋められます。

    Args:
        pcd_file (str): 入力となるPCDファイルのパス
        bin_file (str): 出力となるBINファイルのパス
    """
    try:
        pcd = o3d.io.read_point_cloud(pcd_file)
    except Exception as e:
        print(f"错误：读取PCD文件失败 {pcd_file}: {e}")
        return

    points = np.asarray(pcd.points)

    if pcd.has_colors():
        # 将颜色信息作为强度（这里我们取第一个颜色通道）
        intensities = np.asarray(pcd.colors)[:, 0:1]
    else:
        # 如果没有强度信息，则用0填充
        intensities = np.zeros((points.shape[0], 1), dtype=np.float32)

    # 将 (x, y, z) 和 intensity 合并
    point_cloud_data = np.hstack((points, intensities)).astype(np.float32)

    # 写入二进制文件
    point_cloud_data.tofile(bin_file)
    print(f"成功转换: {pcd_file} -> {bin_file}")

def convert_folder(input_folder, output_folder):
    """
    一个文件夹中所有的 .pcd 文件转换为 .bin 文件。

    Args:
        input_folder (str): 包含 .pcd 文件的输入文件夹路径。
        output_folder (str): 保存 .bin 文件的输出文件夹路径。
    """
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 查找输入文件夹中所有的 .pcd 文件
    pcd_files = glob.glob(os.path.join(input_folder, '*.pcd'))

    if not pcd_files:
        print(f"在文件夹 {input_folder} 中未找到任何 .pcd 文件。")
        return

    print(f"找到 {len(pcd_files)} 个 .pcd 文件，开始转换...")

    for pcd_file in pcd_files:
        # 构建输出文件名
        base_name = os.path.basename(pcd_file)
        file_name_without_ext = os.path.splitext(base_name)[0]
        bin_file = os.path.join(output_folder, f"{file_name_without_ext}.bin")

        # 执行转换
        pcd_to_bin(pcd_file, bin_file)

    print("\n所有文件转换完成！")


if __name__ == '__main__':


    pcd_input_folder = '/home/yty/mfh/mot/my_mot/pcd/0708_sta_01'  # <--- 修改这里

    # 2. 设置您希望保存BIN文件的文件夹路径
    bin_output_folder = '/home/yty/mfh/mos/bin_data/sequences/00/velodyne'   # <--- 修改这里

    convert_folder(pcd_input_folder, bin_output_folder)
