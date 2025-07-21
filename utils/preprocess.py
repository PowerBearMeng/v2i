# 文件: utils/preprocess.py

import open3d as o3d
import numpy as np

class PointCloudPreprocessor:
    """
    一个用于点云预处理的工具类。
    封装了计算地面模型、移除地面/过高点、按距离/高度过滤等常用功能。
    """
    def __init__(self, model_pcd_path=None, 
                 ground_dist_threshold=0.1,
                 num_iterations=1000,
                 verbose=True):
        self.plane_model = None
        self.verbose = verbose
        if model_pcd_path:
            self.plane_model, _ = self.compute_ground_plane(
                model_pcd_path,
                distance_threshold=ground_dist_threshold,
                num_iterations=num_iterations,
                verbose=self.verbose
            )

    @staticmethod
    def compute_ground_plane(pcd_path, distance_threshold=0.1, num_iterations=1000, verbose=True):
        if verbose:
            print("正在从模型点云计算地面方程...")
        pcd = o3d.io.read_point_cloud(pcd_path)
        if not pcd.has_points():
            raise ValueError(f"模型文件 '{pcd_path}' 无效或为空。")
        # --------
        points = np.asarray(pcd.points)
        # 创建一个布尔掩码，只保留 X >= 0 的点
        mask = points[:, 0] >= 0
        # 使用掩码选择要保留的点的索引
        indices_to_keep = np.where(mask)[0]
        
        # 创建一个只包含 X >= 0 点的新点云对象
        pcd_for_plane_fitting = pcd.select_by_index(indices_to_keep)
        # ---------
        
        plane_model, inliers = pcd_for_plane_fitting.segment_plane(distance_threshold, ransac_n=3, num_iterations=num_iterations)
        if plane_model[2] < 0:
            plane_model = -plane_model
            if verbose:
                print("检测到平面法向量朝下，已自动翻转。")
        if verbose:
            a, b, c, d = plane_model
            print(f"计算完成！地面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        return plane_model, inliers

    def _filter_by_roi(self, pcd, config):
        """
        根据XYZ坐标范围 (ROI - Region of Interest) 过滤点云。
        """
        points = np.asarray(pcd.points)
        
        # 从配置中获取每个轴的范围，如果没有提供，则默认为无限大
        x_min = config.get('x_min', -np.inf)
        x_max = config.get('x_max', np.inf)
        y_min = config.get('y_min', -np.inf)
        y_max = config.get('y_max', np.inf)
        z_min = config.get('z_min', -np.inf)
        z_max = config.get('z_max', np.inf)
        
        # 创建每个轴的布尔掩码
        mask_x = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
        mask_y = (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        mask_z = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        
        # 合并所有掩码
        final_mask = mask_x & mask_y & mask_z
        
        return pcd.select_by_index(np.where(final_mask)[0])

    def _filter_by_ground_and_height(self, pcd, config):
        if self.plane_model is None:
            raise RuntimeError("平面参数未设置。")
        points = np.asarray(pcd.points)
        signed_distances = points @ self.plane_model[:3] + self.plane_model[3]
        mask_not_ground = np.abs(signed_distances) > config.get('ground_dist_threshold', 0.1)
        mask_not_ceiling = signed_distances < config.get('max_height_above_ground', 3.0)
        final_mask = np.logical_and(mask_not_ground, mask_not_ceiling)
        return pcd.select_by_index(np.where(final_mask)[0])

    def _filter_by_distance(self, pcd, config):
        points = np.asarray(pcd.points)
        distances_xy = np.linalg.norm(points[:, :2], axis=1)
        mask = distances_xy < config.get('max_dist', 50.0)
        return pcd.select_by_index(np.where(mask)[0])

    def preprocess(self, pcd, config):
        processed_pcd = pcd
        voxel_size = config.get('voxel_size')
        if voxel_size and voxel_size > 0:
            processed_pcd = processed_pcd.voxel_down_sample(voxel_size)
            if self.verbose:
                print(f"-> 体素下采样后 ({voxel_size}m): {len(processed_pcd.points)} 点")
        if self.verbose:
            print(f"\n开始预处理，原始点数: {len(processed_pcd.points)}")
        if 'filter_by_roi' in config:
            params = config['filter_by_roi']
            processed_pcd = self._filter_by_roi(processed_pcd, params)
            if self.verbose:
                print(f"-> ROI过滤后 ({params}): {len(processed_pcd.points)} 点")
        # if 'filter_by_distance' in config:
        #     params = config['filter_by_distance']
        #     processed_pcd = self._filter_by_distance(processed_pcd, params)
        #     if self.verbose:
        #         print(f"-> 距离过滤后 ({params}): {len(processed_pcd.points)} 点")
        if 'filter_by_ground_plane' in config:
            params = config['filter_by_ground_plane']
            processed_pcd = self._filter_by_ground_and_height(processed_pcd, params)
            if self.verbose:
                print(f"-> 地面/过高点过滤后 ({params}): {len(processed_pcd.points)} 点")
        if self.verbose:
            print("预处理完成。")
        return processed_pcd

# --- 用法示例 ---
if __name__ == "__main__":
    # --- 1. 初始化预处理器 (离线步骤) ---
    # 在初始化时，从一个静态场景点云文件自动计算并存储地面模型。
    # 这个文件应该是清晰的、能代表一般场景地面的。
    MODEL_PATH = "./pcd/0718_i_01/frame_00100.pcd"
    try:
        preprocessor = PointCloudPreprocessor(model_pcd_path=MODEL_PATH)
    except ValueError as e:
        print(e)
        preprocessor = None

    if preprocessor:
        # --- 2. 定义实时处理的配置 ---
        # 你可以在这里灵活地组合和调整参数，而无需修改类代码。
        processing_config = {
        'filter_by_ground_plane': {
        'ground_dist_threshold': 0.3,      # 距离地面此距离内的点被视为地面
        'max_height_above_ground': 3.0      # 保留离地此高度以下的物体
        },              # 体素网格滤波器的体素大小
        'filter_by_roi': {
        'x_min': -40.0,     # 只看雷达前方 (X轴正方向)
        'x_max': 40.0,    # 最远看到50米
        'y_min': -50.0,   # Y轴方向，比如左右各10米宽的范围
        'y_max': -5.0,
        }
}
        
        # --- 3. 加载一个新帧并进行处理 (实时步骤) ---
        frame_path = "./pcd/0718_i_01/frame_00100.pcd" 
        frame_to_process = o3d.io.read_point_cloud(frame_path)

        # 调用统一的preprocess方法
        clean_pcd = preprocessor.preprocess(frame_to_process, processing_config)
        
        # --- 4. 可视化结果 ---
        # 将处理后的点云染成红色，以便和原始点云对比
        frame_to_process.paint_uniform_color([0.7, 0.7, 0.7]) # 原始点云为灰色
        clean_pcd.paint_uniform_color([1.0, 0, 0])          # 处理后点云为红色

        o3d.visualization.draw_geometries(
            [frame_to_process, clean_pcd],
            window_name="红色: 预处理后 | 灰色: 原始点云"
        )