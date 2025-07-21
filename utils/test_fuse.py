# 文件名: fuse_with_imu.py
# Description: 自动对齐和融合两个点云文件，支持GPS坐标转换、IMU初始旋转和ICP精细配准。
# 版本: v3 (集成了真实的IMU数据处理)

import open3d as o3d
import numpy as np
import os
import pymap3d as pm
import matplotlib.pyplot as plt
import copy
import time
# --- 新增依赖 ---
# Scipy库用于处理复杂的旋转，特别是四元数运算
# 如果尚未安装，请运行: pip install scipy
from scipy.spatial.transform import Rotation as R

class PointCloudFuser:
    """
    一个用于自动对齐和融合两个点云的类。

    该类封装了从加载、预处理、基于GPS的粗对齐、
    基于IMU的旋转初始对齐、ICP精细配准到最终融合的整个流程。
    """

    # --- 类常量定义 ---
    COLORMAP = plt.get_cmap("jet")

    def __init__(self, ply_path_a, ply_path_b, gps_a, gps_b, imu_quat_a, imu_quat_b, output_path, voxel_size=0.2, save_intermediate=False):
        """
        初始化点云融合器。

        Args:
            ply_path_a (str): 点云A的文件路径。
            ply_path_b (str): 点云B的文件路径。
            gps_a (tuple): 点云A的GPS坐标 (纬度, 经度, 高度)。
            gps_b (tuple): 点云B的GPS坐标 (纬度, 经度, 高度)。
            imu_quat_a (list or np.ndarray): 点云A的IMU姿态四元数 [x, y, z, w]。
            imu_quat_b (list or np.ndarray): 点云B的IMU姿态四元数 [x, y, z, w]。
            output_path (str): 融合后点云的保存路径。
            voxel_size (float): 用于下采样和配准的体素大小。
            save_intermediate (bool): 是否保存中间结果。
        """
        self.ply_path_a = ply_path_a
        self.ply_path_b = ply_path_b
        self.lat_a, self.lon_a, self.alt_a = gps_a
        self.lat_b, self.lon_b, self.alt_b = gps_b
        # 新增：存储IMU四元数
        self.imu_quat_a = imu_quat_a
        self.imu_quat_b = imu_quat_b
        self.output_path = output_path
        self.voxel_size = voxel_size
        self.save_intermediate = save_intermediate

        # 用于存储中间数据的属性
        self.pcd_a_obj = None
        self.pcd_b_obj = None
        self.fused_pcd = None


    def _separate_ground(self, pcd, voxel_size=0.05, distance_threshold=0.1, ransac_n=3, num_iterations=1000, normal_angle_threshold_deg=15):
        """
        使用预处理和法线约束的RANSAC算法，更鲁棒地将点云分离为地面和非地面点。
        """
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

        plane_model, inliers_indices = pcd_down.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        if not inliers_indices:
            return o3d.geometry.PointCloud(), pcd

        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        z_axis = np.array([0, 0, 1])
        cos_theta = np.abs(np.dot(normal, z_axis) / (np.linalg.norm(normal) + 1e-9))
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_deg = np.rad2deg(angle_rad)

        if angle_deg > normal_angle_threshold_deg:
            return o3d.geometry.PointCloud(), pcd

        all_points = np.asarray(pcd.points)
        distances = np.abs(a * all_points[:, 0] + b * all_points[:, 1] + c * all_points[:, 2] + d)
        
        original_inliers_indices = np.where(distances < distance_threshold)[0]

        ground_cloud = pcd.select_by_index(original_inliers_indices)
        objects_cloud = pcd.select_by_index(original_inliers_indices, invert=True)
        
        return ground_cloud, objects_cloud

    @staticmethod
    def _compute_translation(lat_a, lon_a, alt_a, lat_b, lon_b, alt_b):
        """
        计算将 A 的坐标中心转换到 B 的坐标中心所需的平移向量（使用 ENU 坐标系）。
        """
        e, n, u = pm.geodetic2enu(lat_a, lon_a, alt_a, lat_b, lon_b, alt_b)
        return np.array([e, n, u])
    
    # --- 新增核心函数 ---
    @staticmethod
    def _compute_imu_rotation_matrix(quat_a, quat_b):
        """
        根据两个IMU的姿态四元数，计算将A的姿态转换到B的姿态所需的3x3旋转矩阵。
        
        Args:
            quat_a (list or np.ndarray): 源姿态（点云A）的四元数 [x, y, z, w]。
            quat_b (list or np.ndarray): 目标姿态（点云B）的四元数 [x, y, z, w]。
            
        Returns:
            np.ndarray: 3x3的旋转矩阵。
        """
        # 使用scipy.spatial.transform.Rotation处理四元数
        # from_quat期望的格式是 [x, y, z, w]
        rot_a = R.from_quat(quat_a)
        rot_b = R.from_quat(quat_b)
        
        # 计算从姿态A到姿态B的相对旋转
        # 公式: R_relative = R_final * R_initial.inv()
        # 这里 B 是 final, A 是 initial
        relative_rotation = rot_b * rot_a.inv()
        
        # 将相对旋转转换为3x3矩阵
        return relative_rotation.as_matrix()


    def _refine_with_icp(self, source_pcd, target_pcd, initial_transform):
        """
        使用 ICP 对 source_pcd 和 target_pcd 进行精细配准。
        """
        print(f"{len(source_pcd.points)} points in source, {len(target_pcd.points)} points in target")
        threshold = self.voxel_size * 0.8
        
        source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))
        target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))

        result_icp = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd,
            threshold,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        return result_icp.transformation

    @staticmethod
    def _colorize_point_cloud_by_height(pcd):
        """根据Z轴高度为点云着色"""
        points = np.asarray(pcd.points)
        if points.size == 0:
            return pcd
        
        z_coords = points[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        
        if z_max == z_min:
            return pcd.paint_uniform_color([0.5, 0.5, 0.5])

        normalized_z = (z_coords - z_min) / (z_max - z_min + 1e-9)
        colors = PointCloudFuser.COLORMAP(normalized_z)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    @staticmethod
    def _visualize_point_clouds(pcds, window_name="Point Cloud Visualization"):
        """可视化点云列表"""
        o3d.visualization.draw_geometries(
            pcds,
            window_name=window_name,
            width=1600,
            height=900,
            point_show_normal=False
        )

    def fuse(self):
        """
        执行完整的自动对齐和融合流程，并打印各阶段耗时。
        """
        print("="*50)
        print("开始点云融合流程 (GPS+IMU+ICP)...")
        total_start_time = time.time()

        # --- 步骤 1: 加载并预处理点云 ---
        print("\n[步骤 1/5] 加载并预处理点云...")
        step1_start = time.time()
        pcd_a_full = o3d.io.read_point_cloud(self.ply_path_a)
        _, self.pcd_a_obj = self._separate_ground(pcd_a_full)
        pcd_b_full = o3d.io.read_point_cloud(self.ply_path_b)
        _, self.pcd_b_obj = self._separate_ground(pcd_b_full)
        step1_end = time.time()
        print(f"路侧点云 (A) 处理后点数: {len(self.pcd_a_obj.points)}")
        print(f"车载点云 (B) 处理后点数: {len(self.pcd_b_obj.points)}")
        print(f"-> 耗时: {(step1_end - step1_start) * 1000:.2f} ms")

        # --- 步骤 2: 根据GPS计算粗略平移 ---
        print("\n[步骤 2/5] 根据GPS计算粗略平移...")
        step2_start = time.time()
        translation = self._compute_translation(self.lat_a, self.lon_a, self.alt_a, self.lat_b, self.lon_b, self.alt_b)
        gps_transform = np.identity(4)
        gps_transform[0:3, 3] = translation
        pcd_a_translated = copy.deepcopy(self.pcd_a_obj)
        pcd_a_translated.transform(gps_transform)
        step2_end = time.time()
        print(f"GPS平移向量 (e,n,u): {translation.round(2)}")
        print(f"-> 耗时: {(step2_end - step2_start) * 1000:.2f} ms")
        # --- 可视化检查 GPS 对齐效果 ---
        print("\n--- 显示 GPS 粗对齐效果 ---")
        pcd_a_gps_aligned = copy.deepcopy(pcd_a_translated)
        pcd_a_gps_aligned.paint_uniform_color([1, 0, 0])  # 红色表示点云A（路侧）
        pcd_b_copy = copy.deepcopy(self.pcd_b_obj).paint_uniform_color([0, 0, 1])  # 蓝色表示点云B（车载）
        self._visualize_point_clouds([pcd_a_gps_aligned, pcd_b_copy], "GPS Alignment Only")
        
        # --- 步骤 3: 根据IMU计算初始旋转变换 ---
        print("\n[步骤 3/5] 根据IMU计算初始旋转变换...")
        step3_start = time.time()
        
        # --- 改动开始 ---
        # 不再使用单位矩阵，而是调用函数根据真实的IMU数据计算旋转矩阵
        R_imu = self._compute_imu_rotation_matrix(self.imu_quat_a, self.imu_quat_b)
        
        # 构建4x4的初始变换矩阵（只包含IMU的旋转）
        # 这个矩阵将作为ICP的初始猜测
        initial_transform = np.identity(4)
        initial_transform[:3, :3] = R_imu
        # --- 改动结束 ---
        
        step3_end = time.time()
        print("IMU初始旋转矩阵已根据输入数据计算生成。")
        print("IMU Rotation Matrix (R_imu):\n", R_imu.round(4))
        print(f"-> 耗时: {(step3_end - step3_start) * 1000:.2f} ms")
        
        # --- 可视化检查初始对齐效果 ---
        print("\n--- 显示IMU初始对齐效果 (ICP前) ---")
        pcd_a_initial_aligned = copy.deepcopy(pcd_a_translated).transform(initial_transform)
        pcd_a_initial_aligned.paint_uniform_color([1, 0, 0]) # 点云A (路侧) 设置为红色
        pcd_b_copy = copy.deepcopy(self.pcd_b_obj).paint_uniform_color([0, 0, 1]) # 点云B (车载) 设置为蓝色
        self._visualize_point_clouds([pcd_a_initial_aligned, pcd_b_copy], "Initial Alignment (GPS + IMU)")

        # --- 步骤 4: ICP精细配准 ---
        print("\n[步骤 4/5] ICP精细配准...")
        step4_start = time.time()
        # 使用GPS平移后的点云A和点云B，以及IMU提供的初始旋转进行ICP
        # initial_transform 现在包含了真实的IMU旋转信息，能给ICP一个更好的起点
        final_icp_transform = self._refine_with_icp(pcd_a_translated, self.pcd_b_obj, initial_transform)
        step4_end = time.time()
        print("ICP配准完成。")
        print(f"-> 耗时: {(step4_end - step4_start) * 1000:.2f} ms")

        # --- 步骤 5: 融合并保存结果 ---
        print("\n[步骤 5/5] 融合并保存结果...")
        step5_start = time.time()
        pcd_a_final = copy.deepcopy(self.pcd_a_obj)
        # 最终变换 = ICP精细变换 * GPS平移变换
        final_transform_on_original_a = np.dot(final_icp_transform, gps_transform)
        pcd_a_final.transform(final_transform_on_original_a)
        
        self.fused_pcd = pcd_a_final + self.pcd_b_obj
        fused_pcd_down = self.fused_pcd.voxel_down_sample(voxel_size=self.voxel_size/2)
        
        o3d.io.write_point_cloud(self.output_path, fused_pcd_down)
            
        step5_end = time.time()
        print(f'融合后总点数 (下采样后): {len(fused_pcd_down.points)}')
        print(f"融合后的点云已保存至: {self.output_path}")
        print(f"-> 耗时: {(step5_end - step5_start) * 1000:.2f} ms")

        total_end_time = time.time()
        print("\n" + "="*50)
        print(f"流程结束。总耗时: {(total_end_time - total_start_time) * 1000:.2f} ms ({(total_end_time - total_start_time):.2f} 秒)")

        # 可视化最终结果
        print("\n--- 显示最终融合效果 ---")
        fused_pcd_colorized = self._colorize_point_cloud_by_height(fused_pcd_down)
        self._visualize_point_clouds([fused_pcd_colorized], "Final Fused Point Cloud")


# ==============================================================================
# 示例入口 (请在此处配置您的数据)
# ==============================================================================
if __name__ == "__main__":
    # --- 1. 输入点云文件路径 ---
    # 假设A是固定的路侧激光雷达，B是移动的车辆激光雷达
    ply_path_a = "pcd/0717_veh_01/frame_00150.pcd" # 路侧点云
    ply_path_b = "pcd/0717_veh_02/frame_00150.pcd" # 车载点云

    # --- 2. A和B点云对应的GPS坐标 ---
    lat_a, lon_a, alt_a = 39.958805084228516, 116.35052490234375, 41.07400131225586
    lat_b, lon_b, alt_b = 39.95876693725586, 116.35079956054688, 41.056999206842975

    # lat_a, lon_a, alt_a = 39.958805084228516, 116.35052490234375, 41.07400131225586
    # lat_b, lon_b, alt_b = 39.95876693725586, 116.35079956054688, 41.056999206842975
    # imu_a = [0.0022620478448701306, 0.022189015508610025,-0.026345076599445613, 0.9994040562601673] 
    # imu_b = [-0.0047880017929742055, 0.019167718566032242,0.022272667896620864,0.9995567026780275]

    # --- 3. A和B点云对应的IMU姿态四元数 [x, y, z, w] ---
    # !!! 关键步骤: 请在这里填入您自己真实的IMU数据 !!!
    # 这里的数值是示例，需要您用自己的数据替换。
    # 假设路侧单元(A)是水平安装的，没有旋转。四元数为 [0, 0, 0, 1]
    imu_quat_a = [0.0022620478448701306, 0.022189015508610025,0.026345076599445613, 0.9994040562601673] 
    imu_quat_b = [0.0047880017929742055, 0.019167718566032242,0.022272667896620864,0.9995567026780275]


    # yaw_deg, pitch_deg, roll_deg = 5.0, -2.0, 0.0
    # r_b_example = R.from_euler('zyx', [yaw_deg, pitch_deg, roll_deg], degrees=True)
    # imu_quat_b = r_b_example.as_quat() # 得到 [x, y, z, w] 格式的四元数
    # print(f"示例车载IMU四元数 (B): {imu_quat_b.round(4)}")


    # --- 4. 输出路径 ---
    output_path = "./lidar_fused_imu_icp.pcd"

    # --- 实例化并执行融合 ---
    # 您可以调整 voxel_size 来观察不同精度下的效果
    fuser = PointCloudFuser(
        ply_path_a=ply_path_a,
        ply_path_b=ply_path_b,
        gps_a=(lat_a, lon_a, alt_a),
        gps_b=(lat_b, lon_b, alt_b),
        # 传入IMU数据
        imu_quat_a=imu_quat_a,
        imu_quat_b=imu_quat_b,
        output_path=output_path,
        voxel_size=0.05, 
        save_intermediate=True
    )
    
    fuser.fuse()
