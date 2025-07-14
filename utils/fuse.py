# 文件名: fuse.py
# Description: 自动对齐和融合两个点云文件，支持GPS坐标转换和ICP精细配准。


import open3d as o3d
import numpy as np
import os
import pymap3d as pm
import matplotlib.pyplot as plt
import copy
import time

class PointCloudFuser:
    """
    一个用于自动对齐和融合两个点云的类。

    该类封装了从加载、预处理、基于GPS的粗对齐、
    基于特征的初始变换估计、ICP精细配准到最终融合的整个流程。
    """

    # --- 类常量定义 ---
    COLORMAP = plt.get_cmap("jet")

    def __init__(self, ply_path_a, ply_path_b, gps_a, gps_b, output_path, voxel_size=0.2, save_intermediate=False):
        """
        初始化点云融合器。

        Args:
            ply_path_a (str): 点云A的文件路径。
            ply_path_b (str): 点云B的文件路径。
            gps_a (tuple): 点云A的GPS坐标 (纬度, 经度, 高度)。
            gps_b (tuple): 点云B的GPS坐标 (纬度, 经度, 高度)。
            output_path (str): 融合后点云的保存路径。
            voxel_size (float): 用于下采样和配准的体素大小。
            save_intermediate (bool): 是否保存中间结果。
        """
        self.ply_path_a = ply_path_a
        self.ply_path_b = ply_path_b
        self.lat_a, self.lon_a, self.alt_a = gps_a
        self.lat_b, self.lon_b, self.alt_b = gps_b
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

    def _estimate_initial_transform_with_features(self, source_pcd, target_pcd):
        """
        使用FPFH特征进行全局配准，来估计一个粗略的初始变换矩阵。
        """
        print("--- 正在使用特征匹配估计初始变换 ---")
        source_down = source_pcd.voxel_down_sample(self.voxel_size)
        target_down = target_pcd.voxel_down_sample(self.voxel_size)
        print(f"源点云点数: {len(source_down.points)}, 目标点云点数: {len(target_down.points)}")

        radius_normal = self.voxel_size * 2
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        distance_threshold = self.voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        
        if result.transformation.trace() == 4.0:
            return np.identity(4)
            
        return result.transformation

    def _refine_with_icp(self, source_pcd, target_pcd, initial_transform):
        """
        使用 ICP 对 source_pcd 和 target_pcd 进行精细配准。
        """
        print(f"{len(source_pcd.points)} points in source, {len(target_pcd.points)} points in target")
        threshold = self.voxel_size * 0.8
        result_icp = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd,
            threshold,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
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
        print("开始自动点云融合流程...")
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

        # --- 步骤 3: 自动估计初始变换 ---
        print("\n[步骤 3/5] 自动估计初始变换 (FPFH 特征匹配)...")
        step3_start = time.time()
        initial_transform = self._estimate_initial_transform_with_features(pcd_a_translated, self.pcd_b_obj)
        step3_end = time.time()
        print(f"-> 耗时: {(step3_end - step3_start) * 1000:.2f} ms")
        
        pcd_a_initial_aligned = copy.deepcopy(pcd_a_translated).transform(initial_transform)
        pcd_a_initial_aligned.paint_uniform_color([1, 0, 0])
        pcd_b_copy = copy.deepcopy(self.pcd_b_obj).paint_uniform_color([0, 1, 0])
        print("--- 显示自动初始对齐效果 (特征匹配后，ICP前) ---")
        self._visualize_point_clouds([pcd_a_initial_aligned, pcd_b_copy], "Initial Alignment (Feature-Based)")

        # --- 步骤 4: ICP精细配准 ---
        print("\n[步骤 4/5] ICP精细配准...")
        step4_start = time.time()
        final_icp_transform = self._refine_with_icp(pcd_a_translated, self.pcd_b_obj, initial_transform)
        step4_end = time.time()
        print(f"-> 耗时: {(step4_end - step4_start) * 1000:.2f} ms")

        # --- 步骤 5: 融合并保存结果 ---
        print("\n[步骤 5/5] 融合并保存结果...")
        step5_start = time.time()
        pcd_a_final = copy.deepcopy(self.pcd_a_obj)
        final_transform_on_original_a = np.dot(final_icp_transform, gps_transform)
        pcd_a_final.transform(final_transform_on_original_a)
        
        self.fused_pcd = pcd_a_final + self.pcd_b_obj
        fused_pcd_down = self.fused_pcd.voxel_down_sample(voxel_size=self.voxel_size/2)
        
        if self.save_intermediate:
            o3d.io.write_point_cloud(self.output_path, self.fused_pcd)
            
        step5_end = time.time()
        print(f'融合后总点数: {len(self.fused_pcd.points)}')
        print(f"融合后的点云已保存至: {self.output_path}")
        print(f"-> 耗时: {(step5_end - step5_start) * 1000:.2f} ms")

        total_end_time = time.time()
        print("\n" + "="*50)
        print(f"流程结束。总耗时: {(total_end_time - total_start_time) * 1000:.2f} ms ({(total_end_time - total_start_time):.2f} 秒)")

        # 可视化最终结果
        print("\n--- 显示最终融合效果 ---")
        fused_pcd_colorized = self._colorize_point_cloud_by_height(fused_pcd_down)
        self._visualize_point_clouds([fused_pcd_colorized], "Final Fused Point Cloud")

    @staticmethod
    def create_dummy_data(folder_path):
        """如果文件不存在，则创建示例数据"""
        path_a = os.path.join(folder_path, "roadside.pcd")
        path_b = os.path.join(folder_path, "vehicle.pcd")

        if os.path.exists(path_a) and os.path.exists(path_b):
            print("找到现有数据，将使用它们进行融合。")
            return

        print("未找到示例数据，正在创建...")
        os.makedirs(folder_path, exist_ok=True)

        mesh_b = o3d.geometry.TriangleMesh.create_box(width=5, height=2, depth=1)
        mesh_b2 = o3d.geometry.TriangleMesh.create_box(width=1, height=2, depth=3)
        mesh_b2.translate([0, 0, 1])
        pcd_b = (mesh_b + mesh_b2).sample_points_poisson_disk(number_of_points=5000)
        o3d.io.write_point_cloud(path_b, pcd_b)
        print(f"已创建: {path_b}")

        pcd_a = copy.deepcopy(pcd_b)
        rotation_angle_deg = 30
        R = pcd_a.get_rotation_matrix_from_xyz((0, 0, np.deg2rad(rotation_angle_deg)))
        true_translation = np.array([10, 5, 0])
        
        transform_a_to_world = np.identity(4)
        transform_a_to_world[:3, :3] = R
        transform_a_to_world[:3, 3] = true_translation
        
        pcd_a.transform(np.linalg.inv(transform_a_to_world))
        o3d.io.write_point_cloud(path_a, pcd_a)
        print(f"已创建: {path_a}")
        print("示例数据创建完成。")


# 示例入口
if __name__ == "__main__":
    # --- 输入参数配置 ---
    folder_path = './cloud_points'
    
    # 检查并创建示例数据
    # PointCloudFuser.create_dummy_data(folder_path)

    ply_path_a = os.path.join(folder_path, "roadside.pcd")
    ply_path_b = os.path.join(folder_path, "vehicle.pcd")

    lat_a, lon_a, alt_a = 39.9591273, 116.3493549, 38.0
    lat_b, lon_b, alt_b = 39.9590883, 116.3492560, 38.0
    # 输出路径
    output_path = "./lidar_fused_auto_aligned_class.pcd"

    # --- 实例化并执行融合 ---
    # 您可以调整 voxel_size 来观察不同精度下的效果
    fuser = PointCloudFuser(
        ply_path_a=ply_path_a,
        ply_path_b=ply_path_b,
        gps_a=(lat_a, lon_a, alt_a),
        gps_b=(lat_b, lon_b, alt_b),
        output_path=output_path,
        voxel_size=0.3, # 较大的体素尺寸可以加快粗对齐速度
        save_intermediate=True
    )
    
    fuser.fuse()
