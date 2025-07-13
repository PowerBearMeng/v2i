# 这是一个可以添加到你的 detection.py 中的示例函数

import open3d as o3d
import numpy as np
from utils.config import (PCD_FOLDER, MODEL_PATH, PREPROCESSING_CONFIG, 
                    DETECTION_CONFIG, VISUALIZATION_CONFIG, APP_CONFIG)
from preprocess import PointCloudPreprocessor
from visualization import RealtimeVisualizer
from utils.io import create_pcd_stream_from_files

def detect_objects_with_clustering(pcd_clean):
    """
    使用DBSCAN聚类来检测物体，并用简单的规则进行分类。
    :param pcd_clean: 移除了地面的点云。
    :return: 包含边界框和标签的列表。
    """
    
    # 1. 使用DBSCAN进行聚类
    # eps: 两个点被视为邻居的最大距离 (单位：米)。需要根据你的点云密度调整。
    # min_points: 形成一个簇所需的最小点数。
    # print_progress: 打印进度条。
    labels = np.array(pcd_clean.cluster_dbscan(eps=0.8, min_points=10, print_progress=False))

    max_label = labels.max()
    print(f"检测到 {max_label + 1} 个聚类。")

    detected_objects = []
    
    # 2. 遍历每一个聚类
    for i in range(max_label + 1):
        # 提取属于当前聚类的点
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd_clean.select_by_index(cluster_indices)
        
        # 3. 为聚类拟合一个边界框
        # 使用Oriented Bounding Box可以得到物体的朝向
        bbox = cluster_pcd.get_oriented_bounding_box()
        bbox.color = (0, 1, 0) # 默认为绿色
        
        # 4. 提取特征并进行规则分类
        dims = bbox.extent  # [length, width, height]
        height = dims[2]
        width = min(dims[0], dims[1])
        length = max(dims[0], dims[1])
        num_points = len(cluster_pcd.points)

        label = "未知"
        # 简单的规则分类器
        if 1.2 < height < 2.2 and width < 1.2 and length < 1.2 and num_points > 10:
            label = "行人"
            bbox.color = (1, 0, 0) # 红色代表行人
        elif 1.0 < height < 3.5 and 1.0 < width < 3.0 and 2.0 < length < 8.0 and num_points > 20:
            label = "车辆"
            bbox.color = (0, 0, 1) # 蓝色代表车辆

        if label :
            print(f"聚类 {i}: 检测为 {label} | 尺寸: {length:.2f}x{width:.2f}x{height:.2f} | 点数: {num_points}")
            detected_objects.append(bbox)
            
    return detected_objects

if __name__ == "__main__":
    preprocess = PointCloudPreprocessor(model_pcd_path=MODEL_PATH)
    preprocessor = PointCloudPreprocessor(model_pcd_path=MODEL_PATH)

    visualizer = RealtimeVisualizer(VISUALIZATION_CONFIG["window_name"])
    pcd_stream = create_pcd_stream_from_files(
            PCD_FOLDER, 
            start_index=APP_CONFIG["pcd_start_index"], 
            frame_delay=APP_CONFIG["frame_delay_sec"]
        )
    for pcd in pcd_stream:
        pcd_clean = preprocess.preprocess(pcd, PREPROCESSING_CONFIG)
        detected_objects = detect_objects_with_clustering(pcd_clean)
        pcd_clean.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色表示预处理后的点云
        visualizer.update([pcd_clean] + detected_objects)
    # 这将显示预处理后的点云和检测到的物体边界