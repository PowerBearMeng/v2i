# 文件名: test_fuse_modified.py
# Description: 对齐两个点云。流程包含GPS粗对齐、IMU初始旋转，并可选择性执行ICP精细配准。
# 版本: v4 (移除地面分割，分离可视化步骤，ICP可选)

import open3d as o3d
import numpy as np
import os
import pymap3d as pm
import matplotlib.pyplot as plt
import copy
import time
from scipy.spatial.transform import Rotation as R

class PointCloudFuser:
    """
    一个用于对齐和融合两个点云的类。
    流程：GPS粗对齐 -> IMU初始旋转 -> (可选)ICP精细配准。
    """

    COLORMAP = plt.get_cmap("jet")

    def __init__(self, ply_path_a, ply_path_b, gps_a, gps_b, imu_quat_a, imu_quat_b, output_path, voxel_size=0.2):
        """
        初始化点云融合器。
        """
        self.ply_path_a = ply_path_a
        self.ply_path_b = ply_path_b
        self.lat_a, self.lon_a, self.alt_a = gps_a
        self.lat_b, self.lon_b, self.alt_b = gps_b
        self.imu_quat_a = imu_quat_a
        self.imu_quat_b = imu_quat_b
        self.output_path = output_path
        self.voxel_size = voxel_size
        self.pcd_a_obj = None
        self.pcd_b_obj = None
        self.fused_pcd = None

    # --- 方法 _separate_ground 已被移除 ---

    @staticmethod
    def _compute_translation(lat_a, lon_a, alt_a, lat_b, lon_b, alt_b):
        """
        计算将 A 的坐标中心转换到 B 的坐标中心所需的平移向量（使用 ENU 坐标系）。
        """
        e, n, u = pm.geodetic2enu(lat_a, lon_a, alt_a, lat_b, lon_b, alt_b)
        return np.array([e, n, u])
    
    @staticmethod
    def _compute_imu_rotation_matrix(quat_a, quat_b):
        """
        根据两个IMU的姿态四元数，计算将A的姿态转换到B的姿态所需的3x3旋转矩阵。
        """
        rot_a = R.from_quat(quat_a)
        rot_b = R.from_quat(quat_b)
        relative_rotation = rot_b * rot_a.inv()
        return relative_rotation.as_matrix()

    def _refine_with_icp(self, source_pcd, target_pcd, initial_transform):
        """
        使用 ICP 对 source_pcd 和 target_pcd 进行精细配准。
        """
        print(f"为ICP准备点云... 源点云: {len(source_pcd.points)} 个点, 目标点云: {len(target_pcd.points)} 个点")
        threshold = self.voxel_size * 0.8
        
        # 为ICP降采样以提高效率
        source_down = source_pcd.voxel_down_sample(self.voxel_size)
        target_down = target_pcd.voxel_down_sample(self.voxel_size)

        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))

        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down,
            threshold,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        return result_icp.transformation

    @staticmethod
    def _colorize_point_cloud_by_height(pcd):
        """根据Z轴高度为点云着色"""
        points = np.asarray(pcd.points)
        if points.size == 0: return pcd
        z_coords = points[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        if z_max == z_min: return pcd.paint_uniform_color([0.5, 0.5, 0.5])
        normalized_z = (z_coords - z_min) / (z_max - z_min + 1e-9)
        colors = PointCloudFuser.COLORMAP(normalized_z)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    @staticmethod
    def _visualize_point_clouds(pcds, window_name="Point Cloud Visualization"):
        """可视化点云列表"""
        print(f"\n--- 显示窗口: {window_name} ---")
        print("--- (关闭此窗口以继续下一步) ---")
        o3d.visualization.draw_geometries(
            pcds,
            window_name=window_name,
            width=1600,
            height=900,
            point_show_normal=False
        )

    def run_alignment_visualizations(self):
        """
        执行分步对齐，并可视化每一步的结果。
        """
        print("="*50)
        print("开始点云对齐流程...")
        total_start_time = time.time()

        # --- 步骤 1: 加载点云 (不再进行地面分割) ---
        print("\n[步骤 1/3] 加载原始点云...")
        # --- 已修改 ---
        self.pcd_a_obj = o3d.io.read_point_cloud(self.ply_path_a)
        self.pcd_b_obj = o3d.io.read_point_cloud(self.ply_path_b)
        
        if not self.pcd_a_obj.has_points() or not self.pcd_b_obj.has_points():
            print("错误：一个或两个点云文件为空或无法加载。")
            return

        print(f"路侧点云 (A) 点数: {len(self.pcd_a_obj.points)}")
        print(f"车载点云 (B) 点数: {len(self.pcd_b_obj.points)}")

        # --- 新增: 可视化原始点云 ---
        pcd_a_orig = copy.deepcopy(self.pcd_a_obj).paint_uniform_color([1, 0, 0])  # A为红色
        pcd_b_orig = copy.deepcopy(self.pcd_b_obj).paint_uniform_color([0, 0, 1])  # B为蓝色
        self._visualize_point_clouds([pcd_a_orig, pcd_b_orig], "1. Original Point Clouds (Unaligned)")

        # --- 步骤 2: 根据GPS计算并应用平移 ---
        print("\n[步骤 2/3] GPS对齐 (平移)...")
        translation = self._compute_translation(self.lat_a, self.lon_a, self.alt_a, self.lat_b, self.lon_b, self.alt_b)
        gps_transform = np.identity(4)
        gps_transform[0:3, 3] = translation
        
        pcd_a_translated = copy.deepcopy(self.pcd_a_obj)
        pcd_a_translated.transform(gps_transform)
        
        print(f"GPS平移向量 (e,n,u): {translation.round(2)}")
        
        # --- 可视化GPS对齐效果 ---
        pcd_a_gps_aligned_vis = copy.deepcopy(pcd_a_translated).paint_uniform_color([1, 0, 0])
        pcd_b_vis = copy.deepcopy(self.pcd_b_obj).paint_uniform_color([0, 0, 1])
        self._visualize_point_clouds([pcd_a_gps_aligned_vis, pcd_b_vis], "2. GPS Alignment Only")
        
        # --- 步骤 3: 根据IMU计算并应用旋转 ---
        print("\n[步骤 3/3] IMU对齐 (旋转)...")
        R_imu = self._compute_imu_rotation_matrix(self.imu_quat_a, self.imu_quat_b)
        
        initial_transform = np.identity(4)
        initial_transform[:3, :3] = R_imu
        
        print("IMU初始旋转矩阵已计算生成。")
        print("IMU Rotation Matrix (R_imu):\n", R_imu.round(4))
        
        # --- 可视化IMU对齐效果 ---
        pcd_a_initial_aligned = copy.deepcopy(pcd_a_translated)
        pcd_a_initial_aligned.transform(initial_transform) # 在GPS平移的基础上应用旋转
        pcd_a_initial_aligned.paint_uniform_color([1, 0, 0])
        self._visualize_point_clouds([pcd_a_initial_aligned, pcd_b_vis], "3. Initial Alignment (GPS + IMU)")

        print("\n" + "="*50)
        print("所有初始对齐步骤的可视化已完成。")
        print("如需进行ICP精细配准和融合，请取消下面代码块的注释。")
        
        # --- (可选) 步骤 4 & 5: ICP精细配准、融合与保存 ---
        # --- 如果需要运行ICP，请取消下面的注释 ---
        
        print("\n[步骤 4/5] (可选) ICP精细配准...")
        step4_start = time.time()
        
        # ICP的输入是GPS平移后的点云A，和原始点云B，以及IMU提供的初始变换
        final_icp_transform = self._refine_with_icp(pcd_a_translated, self.pcd_b_obj, initial_transform)
        
        step4_end = time.time()
        print("ICP配准完成。")
        print(f"-> 耗时: {(step4_end - step4_start) * 1000:.2f} ms")

        print("\n[步骤 5/5] (可选) 融合并保存结果...")
        step5_start = time.time()
        pcd_a_final = copy.deepcopy(self.pcd_a_obj)
        
        # 最终变换 = ICP精细变换 * GPS平移变换
        # 注意：这里我们使用 pcd_a_translated 作为 source，所以 final_icp_transform 是从 pcd_a_translated 到 pcd_b 的变换。
        # 因此，我们需要将这个变换应用在 pcd_a_translated 上，或者将它与 gps_transform 结合起来应用在原始 pcd_a 上。
        # 最终变换矩阵 = ICP变换 * GPS平移矩阵
        final_transform_on_original_a = np.dot(final_icp_transform, gps_transform)
        
        pcd_a_final.transform(final_transform_on_original_a)
        
        self.fused_pcd = pcd_a_final + self.pcd_b_obj
        fused_pcd_down = self.fused_pcd.voxel_down_sample(voxel_size=self.voxel_size / 2)
        
        # o3d.io.write_point_cloud(self.output_path, fused_pcd_down)
            
        step5_end = time.time()
        print(f'融合后总点数 (下采样后): {len(fused_pcd_down.points)}')
        print(f"融合后的点云已保存至: {self.output_path}")
        print(f"-> 耗时: {(step5_end - step5_start) * 1000:.2f} ms")

        total_end_time = time.time()
        print("\n" + "="*50)
        print(f"流程结束。总耗时: {(total_end_time - total_start_time) * 1000:.2f} ms ({(total_end_time - total_start_time):.2f} 秒)")

        # 可视化最终结果
        print("\n--- 显示最终融合效果 (ICP后) ---")
        fused_pcd_colorized = self._colorize_point_cloud_by_height(fused_pcd_down)
        self._visualize_point_clouds([fused_pcd_colorized], "Final Fused Point Cloud (after ICP)")
        

# ==============================================================================
# 示例入口 (请在此处配置您的数据)
# ==============================================================================
if __name__ == "__main__":
    # --- 1. 输入点云文件路径 ---
    ply_path_a = "pcd/0825/test_i_02/1756111744843747948.pcd" # 侧点云 红

    lat_a, lon_a, alt_a = 39.961151123046875, 116.354804992675781, 43.147529220581057
    imu_quat_a = [0.008161994768286, -0.000838012162088, -0.014805852943061,
 -0.999856722883658]

    ply_path_b = "pcd/0825/test_v_02/1756111744889027843.pcd" # 车载点云 蓝
    lat_b, lon_b, alt_b = 39.960941314697266, 116.354812622070312, 41.584999084472656
    imu_quat_b = [0.001378136236529, -0.002015504309174, -0.034069549808985,
 -0.999416481882657] 

    # --- 4. 输出路径 ---
    # ply_path_b = "pcd/0730_test_04/frame_00200.pcd" # 车载点云
    # lat_b, lon_b, alt_b = 39.958728790283203, 116.350135803222656, 40.618050003051756
    # imu_quat_b = [0.021994095371043, 0.017969071406821, -0.700374404788033, 0.713210393474040]
    output_path = "./lidar_fused_imu_icp.pcd"

    fuser = PointCloudFuser(
        ply_path_a=ply_path_a,
        ply_path_b=ply_path_b,
        gps_a=(lat_a, lon_a, alt_a),
        gps_b=(lat_b, lon_b, alt_b),
        imu_quat_a=imu_quat_a,
        imu_quat_b=imu_quat_b,
        output_path=output_path,
        voxel_size=0.1  # ICP的体素大小，可以根据点云密度调整
    )
    
    # --- 已修改 ---
    # 调用新的主函数，该函数只执行可视化对齐步骤
    fuser.run_alignment_visualizations()