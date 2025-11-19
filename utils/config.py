# 文件: config.py

import os
import numpy as np
# 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
# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCD_FOLDER = "pcd/0825/test_i_02_renamed"  # PCD文件所在的文件夹
# PCD_FOLDER = "/home/mfh/driving/data/carla/64_range_50m/test"  # PCD文件所在的文件夹
# 用于计算地面模型的静态场景PCD文件
MODEL_PATH = os.path.join(PCD_FOLDER, "i_0010.pcd") 


# --- 预处理配置 ---
PREPROCESSING_CONFIG = {
    'filter_by_ground_plane': {
        'ground_dist_threshold': 0.2,      # 距离地面此距离内的点被视为地面
        'max_height_above_ground': 3.0      # 保留离地此高度以下的物体
    },
    'voxel_size': 0.1,                  # 体素网格滤波器的体素大小
    'filter_by_roi': {
        'x_min': -np.inf,     # 只看雷达前方 (X轴正方向)
        'x_max': np.inf,    # 最远看到50米
        'y_min': -np.inf,
        'y_max': np.inf,
    },
    # 'filter_by_roi': {
    #     'x_min': -10.0,     # 只看雷达前方 (X轴正方向)
    #     'x_max': 20.0,    # 最远看到50米
    #     'y_min': -64.0,
    #     'y_max': 64.0,
    #     'z_min': -4.0,
    #     'z_max': 1.0
    # }
}

# --- 动态检测配置 ---
DETECTION_CONFIG = {
    'distance_threshold': 0.2,  # 两帧间点被视为“移动”的距离阈值
    'dbscan_eps': 1.5,          # DBSCAN聚类半径
    'dbscan_min_points': 10     # 形成动态物体的最少点数
}

# --- 可视化配置 ---
VISUALIZATION_CONFIG = {
    "window_name": "Real-time 3D Motion Detection",
    "static_color": [0.5, 0.5, 0.5], # 灰色
    "dynamic_color": [1.0, 0.0, 0.0] # 红色
}

# --- 应用配置 ---
APP_CONFIG = {
    "pcd_start_index": 0, # 从第几帧开始处理
    "frame_delay_sec": 0.1  # 模拟雷达帧率的延迟
}

BEV_CONFIG = {
    # BEV图像的像素尺寸 [height, width]
    "grid_size": [512, 512],
    # 对应的真实世界范围 [X_min, X_max, Y_min, Y_max] (米)
    "lidar_range": [-51.2, 51.2, -51.2, 51.2],
}
