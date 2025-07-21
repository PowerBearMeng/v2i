import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import time

class MDetector:
    """
    一个独立的M-detector实现，用于从点云序列中检测移动物体。
    该方法基于论文 "Moving Event Detection from LiDAR Point Streams"。
    【已扩展以包含Case 1, 2, 3的检测逻辑】
    """
    def __init__(self, config):
        self.config = config
        self.hor_res = np.deg2rad(config['hor_resolution_deg'])
        self.ver_res = np.deg2rad(config['ver_resolution_deg'])
        self.fov_up = np.deg2rad(config['fov_up'])
        self.fov_down = np.deg2rad(config['fov_down'])

        self.depth_map_width = int(np.ceil(2 * np.pi / self.hor_res))
        self.depth_map_height = int(np.ceil((self.fov_up - self.fov_down) / self.ver_res))
        
        # 历史记录现在包含更丰富的信息
        self.history = []
        self.max_history_length = config['max_depth_map_num']
        self.timings = {}

    def _project_to_spherical(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        depth = np.linalg.norm(points, axis=1)
        depth[depth == 0] = 1e-6
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(z / depth)
        u = ((np.pi - azimuth) / self.hor_res).astype(int)
        v = ((self.fov_up - elevation) / self.ver_res).astype(int)
        u = np.clip(u, 0, self.depth_map_width - 1)
        v = np.clip(v, 0, self.depth_map_height - 1)
        return np.stack([u, v, depth], axis=1)

    def _create_history_frame(self, spherical_coords):
        """
        创建并返回一个包含多种深度图的历史帧字典。
        """
        u, v, depth = spherical_coords[:, 0].astype(int), spherical_coords[:, 1].astype(int), spherical_coords[:, 2]
        
        # 1. min_depth_map: 每个像素最近的深度 (用于Case 1和3)
        min_depth_map = np.full((self.depth_map_height, self.depth_map_width), np.inf, dtype=np.float32)
        np.minimum.at(min_depth_map, (v, u), depth)
        
        # 2. max_depth_map: 每个像素最远的深度 (用于Case 2)
        max_depth_map = np.full((self.depth_map_height, self.depth_map_width), -np.inf, dtype=np.float32)
        np.maximum.at(max_depth_map, (v, u), depth)
        max_depth_map[max_depth_map == -np.inf] = np.inf # 将未击中的点设为无穷远

        return {"min_depth": min_depth_map, "max_depth": max_depth_map}

    def _find_dynamic_points_case1(self, spherical_coords):
        """向量化实现 Case 1: 检测新进入视野的前景物体。"""
        u, v, current_depths = spherical_coords[:, 0].astype(int), spherical_coords[:, 1].astype(int), spherical_coords[:, 2]
        occlusion_counts = np.zeros(len(current_depths), dtype=np.uint8)
        
        for frame in self.history:
            historic_min_depths = frame["min_depth"][v, u]
            is_occluded = current_depths < historic_min_depths - self.config['enter_min_thr1']
            occlusion_counts += is_occluded
            
        return np.where(occlusion_counts >= self.config['occluded_map_thr1'])[0]

    def _find_dynamic_points_case2(self, spherical_coords):
        """向量化实现 Case 2: 检测因被遮挡而消失的背景点背后的物体。"""
        u, v, current_depths = spherical_coords[:, 0].astype(int), spherical_coords[:, 1].astype(int), spherical_coords[:, 2]
        occlusion_counts = np.zeros(len(current_depths), dtype=np.uint8)

        for frame in self.history:
            historic_max_depths = frame["max_depth"][v, u]
            # 如果当前点的深度比历史最远点还要远，说明历史最远点可能被遮挡了
            is_occluded = current_depths > historic_max_depths + self.config['occ_depth_thr2']
            occlusion_counts += is_occluded
            
        return np.where(occlusion_counts >= self.config['occluded_times_thr2'])[0]

    def _find_dynamic_points_case3(self, spherical_coords):
        """向量化实现 Case 3: 检测正在遮挡背景的前景物体。"""
        u, v, current_depths = spherical_coords[:, 0].astype(int), spherical_coords[:, 1].astype(int), spherical_coords[:, 2]
        occlusion_counts = np.zeros(len(current_depths), dtype=np.uint8)

        for frame in self.history:
            historic_min_depths = frame["min_depth"][v, u]
            # 如果当前点比历史最近点还要近，说明它可能正在遮挡
            is_occluding = current_depths < historic_min_depths - self.config['occ_depth_thr3']
            occlusion_counts += is_occluding

        return np.where(occlusion_counts >= self.config['occluding_times_thr3'])[0]

    def process_frame(self, pcd):
        t0 = time.perf_counter()
        points = np.asarray(pcd.points)
        distances = np.linalg.norm(points, axis=1)
        valid_indices = distances > self.config['blind_dis']
        points = points[valid_indices]
        t1 = time.perf_counter()
        self.timings['A_preprocess'] = (t1 - t0) * 1000

        spherical_coords = self._project_to_spherical(points)
        t2 = time.perf_counter()
        self.timings['B_projection'] = (t2 - t1) * 1000

        if not self.history:
             # 如果历史记录为空，则跳过检测，只更新历史
            final_dynamic_points = np.array([])
            final_static_points = points
            t4 = time.perf_counter()
            self.timings['C_find_potential'] = 0
            self.timings['D_cluster_and_separate'] = 0
        else:
            # --- 2. 分别执行三种情况的检测 ---
            indices_c1 = self._find_dynamic_points_case1(spherical_coords)
            indices_c2 = self._find_dynamic_points_case2(spherical_coords)
            indices_c3 = self._find_dynamic_points_case3(spherical_coords)
            
            # 合并所有检测到的动态点索引并去重
            potential_dynamic_indices = np.unique(np.concatenate([indices_c1, indices_c2, indices_c3]))
            t3 = time.perf_counter()
            self.timings['C_find_potential'] = (t3 - t2) * 1000
            # --- 3. 聚类去噪与动静态分离 ---
            if potential_dynamic_indices.size > self.config['cluster_min_points']:
                potential_dynamic_points = points[potential_dynamic_indices]
                clustering = DBSCAN(eps=self.config['cluster_eps'], 
                                    min_samples=self.config['cluster_min_points']).fit(potential_dynamic_points)
                valid_cluster_mask = clustering.labels_ != -1
                final_dynamic_indices = potential_dynamic_indices[valid_cluster_mask]
                
                is_dynamic_mask = np.zeros(len(points), dtype=bool)
                is_dynamic_mask[final_dynamic_indices] = True
                
                final_dynamic_points = points[is_dynamic_mask]
                final_static_points = points[~is_dynamic_mask]
            else:
                final_dynamic_points = np.array([])
                final_static_points = points
            
            t4 = time.perf_counter()
            self.timings['D_cluster_and_separate'] = (t4 - t3) * 1000

        # --- 4. 创建Open3D对象 ---
        static_pcd = o3d.geometry.PointCloud()
        static_pcd.points = o3d.utility.Vector3dVector(final_static_points)

        dynamic_pcd = o3d.geometry.PointCloud()
        if final_dynamic_points.any():
            dynamic_pcd.points = o3d.utility.Vector3dVector(final_dynamic_points)
        t5 = time.perf_counter()
        self.timings['E_create_pcd_objects'] = (t5 - t4) * 1000

        # --- 5. 更新历史记录 ---
        current_frame_maps = self._create_history_frame(spherical_coords)
        self.history.append(current_frame_maps)
        if len(self.history) > self.max_history_length:
            self.history.pop(0)
        t6 = time.perf_counter()
        self.timings['F_update_history'] = (t6 - t5) * 1000
        
        self.timings['G_total'] = (t6 - t0) * 1000
        return static_pcd, dynamic_pcd
    
    def process_points(self, pcd):
        """
        处理单帧点云，并同时返回静态点、原始动态点(point-out)和聚类后动态点(frame-out)。
        """
        t0 = time.perf_counter()
        points = np.asarray(pcd.points)
        distances = np.linalg.norm(points, axis=1)
        valid_indices = distances > self.config['blind_dis']
        points = points[valid_indices]
        t1 = time.perf_counter()
        self.timings['A_preprocess'] = (t1 - t0) * 1000

        spherical_coords = self._project_to_spherical(points)
        t2 = time.perf_counter()
        self.timings['B_projection'] = (t2 - t1) * 1000

        # 初始化返回值，以防没有历史记录
        point_out_points = np.array([])
        frame_out_points = np.array([])
        final_static_points = points

        if self.history:
            # --- 2. 分别执行三种情况的检测 ---
            indices_c1 = self._find_dynamic_points_case1(spherical_coords)
            indices_c2 = self._find_dynamic_points_case2(spherical_coords)
            indices_c3 = self._find_dynamic_points_case3(spherical_coords)
            
            # 合并所有检测到的动态点索引并去重
            potential_dynamic_indices = np.unique(np.concatenate([indices_c1, indices_c2, indices_c3]))
            t3 = time.perf_counter()
            self.timings['C_find_potential'] = (t3 - t2) * 1000

            # 【获取 Point-out 结果】
            # 这就是未经聚类去噪的、逐点判断的原始动态点集合
            point_out_points = points[potential_dynamic_indices]

            # --- 3. 聚类去噪与动静态分离 ---
            if potential_dynamic_indices.size > self.config['cluster_min_points']:
                clustering = DBSCAN(eps=self.config['cluster_eps'], 
                                    min_samples=self.config['cluster_min_points']).fit(point_out_points)
                
                valid_cluster_mask = clustering.labels_ != -1
                
                # 【获取 Frame-out 结果】
                # 这是聚类和去噪后的、更干净的动态点
                frame_out_points = point_out_points[valid_cluster_mask]
                
                # 【高效分离静态点】
                # 我们需要找到 frame_out_points 在原始 points 数组中的索引
                # 注意：直接比较浮点数数组可能不稳定，但对于从同一源提取的数据是可行的
                # 一个更鲁棒的方法是使用KDTree，但为了效率我们先用布尔掩码
                is_dynamic_mask = np.zeros(len(points), dtype=bool)
                
                # 为了找到原始索引，我们需要一个映射
                final_dynamic_indices_in_original = potential_dynamic_indices[valid_cluster_mask]
                is_dynamic_mask[final_dynamic_indices_in_original] = True
                
                final_static_points = points[~is_dynamic_mask]
            else:
                # 如果没有足够的潜在动态点，则所有点都是静态的
                frame_out_points = np.array([])
                final_static_points = points
            
            t4 = time.perf_counter()
            self.timings['D_cluster_and_separate'] = (t4 - t3) * 1000
        else:
             # 如果历史记录为空，则所有计时为0
            t4 = time.perf_counter()
            self.timings['C_find_potential'] = 0
            self.timings['D_cluster_and_separate'] = 0


        # --- 4. 创建Open3D对象 ---
        static_pcd = o3d.geometry.PointCloud()
        static_pcd.points = o3d.utility.Vector3dVector(final_static_points)
        
        # 为 point-out 创建对象
        point_out_pcd = o3d.geometry.PointCloud()
        if point_out_points.any():
            point_out_pcd.points = o3d.utility.Vector3dVector(point_out_points)

        # 为 frame-out 创建对象
        frame_out_pcd = o3d.geometry.PointCloud()
        if frame_out_points.any():
            frame_out_pcd.points = o3d.utility.Vector3dVector(frame_out_points)
            
        t5 = time.perf_counter()
        self.timings['E_create_pcd_objects'] = (t5 - t4) * 1000

        # --- 5. 更新历史记录 ---
        current_frame_maps = self._create_history_frame(spherical_coords)
        self.history.append(current_frame_maps)
        if len(self.history) > self.max_history_length:
            self.history.pop(0)
        t6 = time.perf_counter()
        self.timings['F_update_history'] = (t6 - t5) * 1000
        
        self.timings['G_total'] = (t6 - t0) * 1000
        
        # 【关键改动】返回三种点云
        return static_pcd, point_out_pcd, frame_out_pcd

    def get_timings(self):
        return self.timings