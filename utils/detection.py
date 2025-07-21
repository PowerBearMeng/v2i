# 文件: utils/detection.py
import open3d as o3d
import numpy as np
from collections import deque 
from utils.config import VISUALIZATION_CONFIG
import time

class DynamicObjectDetector:
    def __init__(self, config):
        """
        初始化动态物体检测器。
        :param config: 包含检测参数的字典，例如DETECTION_CONFIG。
        """
        self.distance_threshold = config.get('distance_threshold', 0.2)
        self.dbscan_eps = config.get('dbscan_eps', 1.5)
        self.dbscan_min_points = config.get('dbscan_min_points', 15)

        # --- 新增代码 ---
        # 用于存储上一帧稳定后的静态包圍盒
        self.last_stable_static_boxes = [] 
        # 平滑系数 alpha。值越小，平滑效果越强，但响应越慢。0.4是一个不错的起始值。
        self.smoothing_alpha = 0.4  

    def _cluster_and_filter(self, pcd):
        """内部方法：使用DBSCAN对点云进行聚类并过滤噪声。"""
        if not pcd.has_points():
            return o3d.geometry.PointCloud()
        
        labels = np.array(pcd.cluster_dbscan(eps=self.dbscan_eps, min_points=self.dbscan_min_points, print_progress=False))
        good_indices = np.where(labels >= 0)[0]
        
        if good_indices.size == 0:
            return o3d.geometry.PointCloud()
        
        return pcd.select_by_index(good_indices)

    def _expand_clusters_iteratively(self, pcd_full, pcd_seed):
        """
        使用迭代的区域生长方法来扩展动态点集。
        :param pcd_full: 完整的当前帧点云。
        :param pcd_seed: 初始的、高可信度的动态点云“种子”。
        :return: 包含所有扩展后动态点的索引数组。
        """
        if not pcd_full.has_points() or not pcd_seed.has_points():
            return np.array([], dtype=int)

        # 1. 为完整的当前帧点云构建KDTree，用于快速邻近搜索
        kdtree_full = o3d.geometry.KDTreeFlann(pcd_full)
        
        # 2. 初始化我们的数据结构
        #    - `processed_indices`: 使用集合(set)来存储所有已确认的动态点索引，查询速度快且自动去重。
        #    - `queue`: 使用双端队列(deque)作为“待办列表”，从左侧弹出，从右侧添加。
        seed_points = np.asarray(pcd_seed.points)
        processed_indices = set()
        queue = deque()

        # 3. 将所有种子点加入队列和已处理集合
        for point in seed_points:
            # 找到每个种子点在完整点云中的索引
            [k, idx, _] = kdtree_full.search_knn_vector_3d(point, 1)
            if k > 0:
                index = idx[0]
                if index not in processed_indices:
                    processed_indices.add(index)
                    queue.append(index)
        
        # 4. 开始迭代扩展（区域生长）
        while queue:
            # 从待办列表中取出一个点的索引
            current_index = queue.popleft()
            current_point = pcd_full.points[current_index]
            
            # 搜索这个点的所有邻居
            # 使用self.dbscan_eps作为半径，因为它定义了“什么是同一个物体”的距离标准
            [k, neighbor_indices, _] = kdtree_full.search_radius_vector_3d(current_point, self.dbscan_eps)
            
            # 遍历所有邻居
            for neighbor_index in neighbor_indices:
                # 如果这个邻居还没有被处理过
                if neighbor_index not in processed_indices:
                    # 将它标记为已处理（动态点）
                    processed_indices.add(neighbor_index)
                    # 并将它加入到待办列表的末尾，以便继续从它开始扩展
                    queue.append(neighbor_index)
                    
        # 5. 返回所有被标记为动态的点的索引
        return np.array(list(processed_indices), dtype=int)
    
    def detect(self, pcd_curr, pcd_prev):
        """
        通过比较当前帧和前一帧来检测运动中的物体。
        :return: (静态点云, 动态点云)
        """
        if not pcd_curr.has_points() or not pcd_prev.has_points():
            return pcd_curr, o3d.geometry.PointCloud()

        kdtree_prev = o3d.geometry.KDTreeFlann(pcd_prev)
        curr_points = np.asarray(pcd_curr.points)

        dists_sq = np.array([kdtree_prev.search_knn_vector_3d(pt, 1)[2][0] for pt in curr_points])
        motion_indices = np.where(np.sqrt(dists_sq) > self.distance_threshold)[0]

        if motion_indices.size == 0:
            return pcd_curr, o3d.geometry.PointCloud()

        final_motion_pcd = pcd_curr.select_by_index(motion_indices)

        if not final_motion_pcd.has_points():
            return pcd_curr, o3d.geometry.PointCloud()

        kdtree_final_motion = o3d.geometry.KDTreeFlann(final_motion_pcd)

        dists_sq_map_list = [kdtree_final_motion.search_knn_vector_3d(pt, 1)[2][0] for pt in np.asarray(pcd_curr.points)]
        dists_sq_map = np.array(dists_sq_map_list)
        final_motion_indices = np.where(np.sqrt(dists_sq_map) < self.dbscan_eps / 2.0)[0]

        static_pcd = pcd_curr.select_by_index(final_motion_indices, invert=True)
        dynamic_pcd = pcd_curr.select_by_index(final_motion_indices)
        
        return static_pcd, dynamic_pcd

    def detect_new(self, pcd_curr, pcd_prev):
        """
        检测动态物体（增强版）：
        - 使用双向最近邻一致性过滤误报
        - 加入多帧确认机制稳定动态判断
        :return: (静态点云, 动态点云)
        """
        if not pcd_curr.has_points() or not pcd_prev.has_points():
            return pcd_curr, o3d.geometry.PointCloud()

        curr_points = np.asarray(pcd_curr.points)
        prev_points = np.asarray(pcd_prev.points)

        # KD-Tree 构建
        kdtree_prev = o3d.geometry.KDTreeFlann(pcd_prev)
        kdtree_curr = o3d.geometry.KDTreeFlann(pcd_curr)

        # ========== STEP 1: 双向最近邻一致性判断 ==========
        motion_indices = []
        for i, pt in enumerate(curr_points):
            # 当前点在上一帧中的最近邻
            k1, idx1, _ = kdtree_prev.search_knn_vector_3d(pt, 1)
            if k1 == 0:
                continue
            matched_pt = prev_points[idx1[0]]

            # 反向最近邻判断
            k2, idx2, _ = kdtree_curr.search_knn_vector_3d(matched_pt, 1)
            if k2 == 0:
                continue

            if idx2[0] == i:  # 一致
                dist = np.linalg.norm(pt - matched_pt)
                if dist > self.distance_threshold:
                    motion_indices.append(i)

        motion_indices = np.array(motion_indices, dtype=int)

        # ========== STEP 2: 多帧确认机制 ==========
        if not hasattr(self, 'dynamic_index_buffer'):
            from collections import deque
            self.dynamic_index_buffer = deque(maxlen=3)

        self.dynamic_index_buffer.append(set(motion_indices))

        confirmed_indices = []
        if len(self.dynamic_index_buffer) >= self.dynamic_index_buffer.maxlen:
            for idx in motion_indices:
                count = sum(idx in prev for prev in self.dynamic_index_buffer)
                if count >= 2:  # 至少出现2次才认为是真动态
                    confirmed_indices.append(idx)
            motion_indices = np.array(confirmed_indices, dtype=int)

        if motion_indices.size == 0:
            return pcd_curr, o3d.geometry.PointCloud()

        # ========== STEP 3: 最终动态点聚类与分离 ==========
        final_motion_pcd = pcd_curr.select_by_index(motion_indices)

        if not final_motion_pcd.has_points():
            return pcd_curr, o3d.geometry.PointCloud()

        # 为了聚类扩张，这一步保留也可以不做
        kdtree_final_motion = o3d.geometry.KDTreeFlann(final_motion_pcd)

        dists_sq_map_list = [kdtree_final_motion.search_knn_vector_3d(pt, 1)[2][0] for pt in curr_points]
        dists_sq_map = np.array(dists_sq_map_list)
        final_motion_indices = np.where(np.sqrt(dists_sq_map) < self.dbscan_eps / 2.0)[0]

        static_pcd = pcd_curr.select_by_index(final_motion_indices, invert=True)
        dynamic_pcd = pcd_curr.select_by_index(final_motion_indices)

        return static_pcd, dynamic_pcd

    
    def detect3(self, pcd_curr, pcd_prev):
        """
        使用迭代式区域生长来更完整地检测动态物体。
        :return: (静态点云, 动态点云, 包围盒列表)
        """
        # 步骤A: 寻找“疑似”动态点 (不变)
        if not pcd_curr.has_points() or not pcd_prev.has_points():
            return pcd_curr, o3d.geometry.PointCloud()

        kdtree_prev = o3d.geometry.KDTreeFlann(pcd_prev)
        dists_sq = np.array([kdtree_prev.search_knn_vector_3d(pt, 1)[2][0] for pt in np.asarray(pcd_curr.points)])
        motion_indices = np.where(np.sqrt(dists_sq) > self.distance_threshold)[0]

        if motion_indices.size == 0:
            return pcd_curr, o3d.geometry.PointCloud()

        candidate_motion_pcd = pcd_curr.select_by_index(motion_indices)

        # 步骤B: 聚类“疑似”动态点，得到“种子” (不变)
        seed_motion_pcd = self._cluster_and_filter(candidate_motion_pcd)

        if not seed_motion_pcd.has_points():
            return pcd_curr, o3d.geometry.PointCloud()
            
        # 步骤C (新): 调用迭代扩展方法
        final_motion_indices = self._expand_clusters_iteratively(pcd_curr, seed_motion_pcd)

        if final_motion_indices.size == 0:
            return pcd_curr, o3d.geometry.PointCloud()

        # 步骤D: 最终分类 (不变)
        static_pcd = pcd_curr.select_by_index(final_motion_indices, invert=True)
        dynamic_pcd = pcd_curr.select_by_index(final_motion_indices)
        
        return static_pcd, dynamic_pcd
    
    def detect_box(self, pcd_curr, pcd_prev):
        """
        通过比较当前帧和前一帧来检测运动中的物体，并为每个物体生成包围盒。
        
        :return: (静态点云, 动态点云, 包围盒列表)
        """
        if not pcd_curr.has_points() or not pcd_prev.has_points():
            return pcd_curr, o3d.geometry.PointCloud(), []

        # 步骤 1: 找出候选的移动点 (逻辑不变)
        kdtree_prev = o3d.geometry.KDTreeFlann(pcd_prev)
        curr_points = np.asarray(pcd_curr.points)
        dists_sq = np.array([kdtree_prev.search_knn_vector_3d(pt, 1)[2][0] for pt in curr_points])
        motion_indices = np.where(np.sqrt(dists_sq) > self.distance_threshold)[0]

        if motion_indices.size == 0:
            return pcd_curr, o3d.geometry.PointCloud(), []

        candidate_motion_pcd = pcd_curr.select_by_index(motion_indices)

        # 步骤 2: 对候选点进行DBSCAN聚类
        if not candidate_motion_pcd.has_points():
            return pcd_curr, o3d.geometry.PointCloud(), []
        
        labels = np.array(candidate_motion_pcd.cluster_dbscan(
            eps=self.dbscan_eps,
            min_points=self.dbscan_min_points,
            print_progress=False
        ))
        
        unique_labels = np.unique(labels[labels >= 0]) # 忽略噪声点(-1)
        
        bounding_boxes = []
        
        good_indices = np.where(labels >= 0)[0]
        if good_indices.size == 0:
            return pcd_curr, o3d.geometry.PointCloud(), []
        
        final_motion_pcd_for_mapping = candidate_motion_pcd.select_by_index(good_indices)

        # 步骤 3: 遍历每一个簇，为它们生成包围盒
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_pcd = candidate_motion_pcd.select_by_index(cluster_indices)
            
            # 增加点数检查，防止为过小的簇生成包围盒时程序崩溃
            if len(cluster_pcd.points) >= 4:
                bbox = cluster_pcd.get_oriented_bounding_box()
                bbox.color = [0, 0, 1]  # 设置包围盒颜色为绿色
                bounding_boxes.append(bbox)
        
        
        # 步骤 4: 映射回原始点云以获得饱满的动态点云 (逻辑不变)
        kdtree_final_motion = o3d.geometry.KDTreeFlann(final_motion_pcd_for_mapping)
        dists_sq_map_list = [kdtree_final_motion.search_knn_vector_3d(pt, 1)[2][0] for pt in np.asarray(pcd_curr.points)]
        dists_sq_map = np.array(dists_sq_map_list)
        final_motion_indices = np.where(np.sqrt(dists_sq_map) < self.dbscan_eps / 2.0)[0]

        static_pcd = pcd_curr.select_by_index(final_motion_indices, invert=True)
        dynamic_pcd = pcd_curr.select_by_index(final_motion_indices)
        
        # 步骤 5: 返回所有结果
        return static_pcd, dynamic_pcd, bounding_boxes
        # 在 utils/detection.py 中添加这个高效的统一检测函数

# 在 utils/detection.py 的 DynamicObjectDetector 类中

    def detect_all_unified(self, pcd_curr, pcd_prev):
        """
        通过“先聚类，后分类”的高效方法，并对静态框进行时间滤波。
        """
        if not pcd_curr.has_points() or not pcd_prev.has_points():
            return o3d.geometry.PointCloud(), o3d.geometry.PointCloud(), [], [] 

        # --- 步骤 1: 对当前帧完整聚类 (逻辑不变) ---
        labels = np.array(pcd_curr.cluster_dbscan(
            eps=self.dbscan_eps,
            min_points=self.dbscan_min_points,
            print_progress=False
        ))
        unique_labels = np.unique(labels[labels >= 0])  

        # --- 步骤 2: 准备判断 (逻辑不变) ---
        kdtree_prev = o3d.geometry.KDTreeFlann(pcd_prev)

        # 临时变量，用于存储当前帧的原始检测结果
        current_raw_static_boxes = []
        dynamic_boxes = []
        final_dynamic_pcd = o3d.geometry.PointCloud()
        final_static_pcd = o3d.geometry.PointCloud() # 这个也会在后面重新生成   

        # --- 步骤 3: 遍历所有簇，初步判断其动静属性 (逻辑不变) ---
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_pcd = pcd_curr.select_by_index(cluster_indices)

            if not cluster_pcd.has_points() or len(cluster_pcd.points) < 4:
                continue    

            cluster_center = cluster_pcd.get_center()
            [k, idx, dist_sq] = kdtree_prev.search_knn_vector_3d(cluster_center, 1)

            bbox = cluster_pcd.get_oriented_bounding_box()

            if np.sqrt(dist_sq[0]) > self.distance_threshold:
                # --- 判定为动态物体 ---
                final_dynamic_pcd += cluster_pcd
                bbox.color = [1, 0, 0]  # 红色
                dynamic_boxes.append(bbox)
            else:
                # --- 判定为静态物体 ---
                # 先不着急放入最终列表，而是放入临时列表，等待滤波
                current_raw_static_boxes.append(bbox)   

        # --- 步骤 4: 对静态包围盒进行时间滤波 ---

        # 如果上一帧没有任何稳定的静态框，就直接信任当前帧的结果
        if not self.last_stable_static_boxes:
            self.last_stable_static_boxes = current_raw_static_boxes

        stable_boxes_for_this_frame = []
        matched_last_frame_indices = set()  

        # 为了快速匹配，为上一帧的稳定框中心点构建KDTree
        if self.last_stable_static_boxes:
            last_centers = np.array([box.get_center() for box in self.last_stable_static_boxes])
            kdtree_last_boxes = o3d.geometry.KDTreeFlann(np.transpose(last_centers))    

            # 遍历当前帧检测到的每一个原始静态框
            for current_box in current_raw_static_boxes:
                current_center = current_box.get_center()

                # 在上一帧的稳定框中寻找最近的匹配对象
                # 这里的匹配距离阈值可以根据您的场景大小调整
                [k, idx, dist_sq] = kdtree_last_boxes.search_radius_vector_3d(current_center, radius=0.5)   

                if k > 0:
                    # --- 找到了匹配！进行平滑处理 ---
                    # 获取匹配到的上一帧的稳定框
                    matched_last_box = self.last_stable_static_boxes[idx[0]]

                    # 使用指数移动平均(EMA)来平滑中心点和尺寸
                    alpha = self.smoothing_alpha
                    smoothed_center = alpha * current_center + (1 - alpha) * matched_last_box.get_center()
                    smoothed_extent = alpha * current_box.extent + (1 - alpha) * matched_last_box.extent

                    # 创建一个新的、平滑后的包围盒 (旋转用最新的，中心和尺寸用平滑后的)
                    smoothed_box = o3d.geometry.OrientedBoundingBox(smoothed_center, current_box.R, smoothed_extent)
                    smoothed_box.color = [0, 1, 0] # 绿色
                    stable_boxes_for_this_frame.append(smoothed_box)

                    # 记录下这个上一帧的框已经被匹配过了
                    matched_last_frame_indices.add(idx[0])
                else:
                    # --- 没有找到匹配，说明是新出现的静态物体 ---
                    # 直接添加，不进行平滑
                    current_box.color = [0, 1, 0] # 绿色
                    stable_boxes_for_this_frame.append(current_box)
        else:
            # 如果上一帧没有数据，直接使用当前帧的原始数据
            for box in current_raw_static_boxes:
                box.color = [0, 1, 0]
            stable_boxes_for_this_frame = current_raw_static_boxes  


        # 更新类的状态，为下一帧做准备
        self.last_stable_static_boxes = stable_boxes_for_this_frame 

        # 重新根据最终的稳定框生成静态点云 (可选，但为了显示一致性推荐)
        # 这一步比较耗时，如果对性能要求极致，可以只返回框
        if stable_boxes_for_this_frame:
            # 从pcd_curr中裁剪出所有在稳定静态框内的点
            # 注意: crop会返回一个新的点云，所以不能用 +=
            temp_static_pcds = [pcd_curr.crop(bbox) for bbox in stable_boxes_for_this_frame]
            if temp_static_pcds:
                 final_static_pcd = temp_static_pcds[0]
                 for i in range(1, len(temp_static_pcds)):
                     final_static_pcd += temp_static_pcds[i]    

        # --- 步骤 5: 返回所有结果 ---
        return final_static_pcd, final_dynamic_pcd, dynamic_boxes, stable_boxes_for_this_frame
    
    def detect_efficient(self, pcd_curr, pcd_prev):
        """
        一个更高效的检测方法，采用“先聚类，再分类”的策略。
        它只返回动态和静态点云，不处理包围盒，以获得最快的速度。

        :param pcd_curr: 当前帧的预处理后点云。
        :param pcd_prev: 前一帧的预处理后点云。
        :return: (静态点云, 动态点云)
        """
        if not pcd_curr.has_points() or not pcd_prev.has_points():
            return o3d.geometry.PointCloud(), o3d.geometry.PointCloud()

        # --- 步骤 1: 对当前帧的完整点云进行DBSCAN聚类 ---
        # 这是计算瓶颈，因此输入点云的数量至关重要。
        # 建议在预处理时使用体素下采样来减少点数。
        time_1 = time.time()
        labels = np.array(pcd_curr.cluster_dbscan(
            eps=self.dbscan_eps,
            min_points=self.dbscan_min_points,
            print_progress=False
        ))
        time_2 = time.time()
        print(f"DBSCAN聚类耗时: {(time_2 - time_1) * 1000:.2f} ms")
        # 获取所有有效簇的标签（忽略噪声点，其标签为-1）
        unique_labels = np.unique(labels[labels >= 0])
        
        # 如果没有形成任何簇，则认为所有点都是静态的
        if unique_labels.size == 0:
            return pcd_curr, o3d.geometry.PointCloud()

        # --- 步骤 2: 为前一帧点云构建KDTree，用于快速查询 ---
        kdtree_prev = o3d.geometry.KDTreeFlann(pcd_prev)

        # --- 步骤 3: 遍历每一个簇，判断其动/静属性 ---
        dynamic_cluster_labels = []

        for label in unique_labels:
            # 找出属于当前簇的所有点的索引
            cluster_indices = np.where(labels == label)[0]
            
            # 为了提高效率，我们只使用簇的中心点来代表整个簇进行判断
            cluster_pcd = pcd_curr.select_by_index(cluster_indices)
            cluster_center = cluster_pcd.get_center()

            # 在前一帧的KDTree中搜索离簇中心最近的点
            [k, idx, dist_sq] = kdtree_prev.search_knn_vector_3d(cluster_center, 1)

            # 如果最近邻的距离大于阈值，则认为该簇是动态的
            if k > 0 and np.sqrt(dist_sq[0]) > self.distance_threshold:
                dynamic_cluster_labels.append(label)
        
        # --- 步骤 4: 根据分类结果，一次性分离动静态点云 ---
        if not dynamic_cluster_labels:
            # 如果没有检测到任何动态簇
            return pcd_curr, o3d.geometry.PointCloud()
        
        # 使用np.isin高效地找出所有属于动态簇的点的索引
        dynamic_indices = np.where(np.isin(labels, dynamic_cluster_labels))[0]

        # 使用索引一次性分离点云
        dynamic_pcd = pcd_curr.select_by_index(dynamic_indices)
        static_pcd = pcd_curr.select_by_index(dynamic_indices, invert=True)

        return static_pcd, dynamic_pcd
    
    # 请将这个新函数添加到您的 utils/detection.py 文件中
    # 您可以保留旧的 detect3，或者直接用这个覆盖它
    def detect3_optimized(self, pcd_curr, pcd_prev):
        """
        detect3的高效重构版本。
        采用“先聚类，再分类”的策略来获得完整的动态物体，避免了逐点搜索和迭代扩展。
    
        :return: (静态点云, 动态点云)
        """
        # 步骤A: 检查输入是否有效 (不变)
        if not pcd_curr.has_points() or not pcd_prev.has_points():
            return pcd_curr, o3d.geometry.PointCloud()
    
        # =======================================================================
        # 核心优化点: 不再逐点搜索，而是先对整帧进行一次聚类
        # =======================================================================
        
        # 步骤B: 对当前帧的完整点云进行DBSCAN聚类
        # 这是新的计算瓶颈，速度取决于输入点云的数量。
        # 务必在预处理阶段使用体素下采样（Voxel Downsampling）来控制点数！
        labels = np.array(pcd_curr.cluster_dbscan(
            eps=self.dbscan_eps,
            min_points=self.dbscan_min_points,
            print_progress=False
        ))
        
        # 获取所有有效物体的标签（忽略噪声点，其标签为-1）
        unique_labels = np.unique(labels[labels >= 0])
        
        if unique_labels.size == 0:
            # 如果没有形成任何物体簇，说明全是静态或噪声
            return pcd_curr, o3d.geometry.PointCloud()
    
        # 步骤C: 为前一帧构建KDTree，用于快速查询
        kdtree_prev = o3d.geometry.KDTreeFlann(pcd_prev)
    
        # =======================================================================
        # 核心优化点: 不再对“点”进行动静判断，而是对“物体”进行判断
        # =======================================================================
    
        # 步骤D: 遍历每一个“物体簇”，判断其动/静属性
        dynamic_cluster_labels = []
        for label in unique_labels:
            # 找出属于当前簇的所有点的索引
            cluster_indices = np.where(labels == label)[0]
            
            # 使用簇的中心点来代表整个簇进行判断，计算量极小
            cluster_center = pcd_curr.select_by_index(cluster_indices).get_center()
    
            # 在前一帧中搜索离簇中心最近的点
            [k, idx, dist_sq] = kdtree_prev.search_knn_vector_3d(cluster_center, 1)
    
            # 如果簇中心的移动距离大于阈值，则将整个簇标记为动态
            if k > 0 and np.sqrt(dist_sq[0]) > self.distance_threshold:
                dynamic_cluster_labels.append(label)
        
        # 步骤E: 根据分类结果，一次性分离动静态点云
        if not dynamic_cluster_labels:
            # 如果没有检测到任何动态簇
            return pcd_curr, o3d.geometry.PointCloud()
        
        # 使用NumPy的isin函数，一次性找出所有动态簇包含的点的索引
        dynamic_indices = np.where(np.isin(labels, dynamic_cluster_labels))[0]
    
        # 使用索引直接生成最终的动态和静态点云
        dynamic_pcd = pcd_curr.select_by_index(dynamic_indices)
        static_pcd = pcd_curr.select_by_index(dynamic_indices, invert=True)
    
        return static_pcd, dynamic_pcd