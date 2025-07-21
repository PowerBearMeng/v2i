# 文件: main_mdetector_standalone.py

import time
import open3d as o3d
import numpy as np

# 从utils中导入所需的模块
from utils.config import (PCD_FOLDER, MODEL_PATH, PREPROCESSING_CONFIG, 
                          VISUALIZATION_CONFIG, APP_CONFIG)
from utils.preprocess import PointCloudPreprocessor
from utils.io import create_pcd_stream_from_files
from utils.visualization import RealtimeVisualizer
from m_detect.m_detection import MDetector  # <--- 导入新的独立MDetector类
from utils.logger import PerformanceLogger

def run():
    """
    使用独立的M-detector类进行动态检测的主执行函数。
    """
    print("--- 启动基于独立M-Detector的实时3D动态物体检测系统 ---")
    verbose = False
    
    m_detector_config = {
        # --- Case 1 & Sensor Params ---
        'hor_resolution_deg': 0.2,
        'ver_resolution_deg': 2.0,
        'fov_up': 15.0,
        'fov_down': -15.0,
        'blind_dis': 0.2,
        'max_depth_map_num': 5,
        'enter_min_thr1': 0.5,
        'occluded_map_thr1': 2,

        # --- 新增：Case 2 Params (被遮挡) ---
        'occ_depth_thr2': 0.15,         # 判断被遮挡的深度阈值(米)
        'occluded_times_thr2': 2,       # 触发Case 2的最小连续帧数

        # --- 新增：Case 3 Params (正在遮挡) ---
        'occ_depth_thr3': 0.15,         # 判断正在遮挡的深度阈值(米)
        'occluding_times_thr3': 2,      # 触发Case 3的最小连续帧数

        # --- Post-processing Params ---
        'cluster_eps': 1.0,
        'cluster_min_points': 10
    }

    # --- 1. 初始化 ---
    print("正在初始化模块...")
    try:
        # 预处理器仍然可以用于移除地面等操作，以简化M-detector的输入
        preprocessor = PointCloudPreprocessor(model_pcd_path=MODEL_PATH, verbose=False)
        
        # 初始化独立的MDetector
        detector = MDetector(config=m_detector_config)
        
        visualizer = RealtimeVisualizer("Standalone M-Detector Visualization")
        pcd_stream = create_pcd_stream_from_files(
            PCD_FOLDER, 
            start_index=APP_CONFIG["pcd_start_index"], 
            frame_delay=APP_CONFIG["frame_delay_sec"]
        )
        if verbose:
            logger = PerformanceLogger(log_folder="logs_mdetector_standalone")
            
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    frame_index = 0
    print("初始化完成，开始处理数据流...")

    # --- 2. 实时处理循环 ---
    try:
        for pcd_current_raw in pcd_stream:
            start_time = time.time()
            frame_index += 1

            # 步骤 A: (可选但推荐) 使用预处理器简化输入
            pcd_current_clean = preprocessor.preprocess(pcd_current_raw, PREPROCESSING_CONFIG)
            
            # 步骤 B: 调用M-detector的处理方法
            static_pcd, dynamic_pcd = detector.process_frame(pcd_current_clean)
            # 2. 获取并打印耗时
            timings = detector.get_timings()
            print(f"\n--- 帧 {frame_index} 性能分析 ---")
            for step, duration in timings.items():
                print(f"{step:<25s}: {duration:8.2f} ms")
            print("--------------------------")

            detect_time = (time.time() - start_time) * 1000

            # 步骤 C: 准备可视化
            static_pcd.paint_uniform_color(VISUALIZATION_CONFIG["static_color"])
            dynamic_pcd.paint_uniform_color(VISUALIZATION_CONFIG["dynamic_color"])

            # 步骤 D: 更新可视化窗口
            visualizer.update([static_pcd, dynamic_pcd])
            
            # 打印统计信息
            total_points = len(dynamic_pcd.points) + len(static_pcd.points)
            percent = (len(dynamic_pcd.points) / total_points * 100) if total_points > 0 else 0
            size_bytes = len(np.asarray(dynamic_pcd.points, dtype=np.float32).tobytes())

            print(f"\r帧 {frame_index}: 检测耗时: {detect_time:.1f} ms | 动态点数: {len(dynamic_pcd.points)} | 占比: {percent:.2f}%  ", end="")
            
            if verbose:
                logger.log(frame_index, detect_time, len(dynamic_pcd.points), len(static_pcd.points), percent, size_bytes)

    except KeyboardInterrupt:
        print("\n用户中断，正在关闭程序...")
    finally:
        # --- 3. 清理 ---
        visualizer.close()
        if 'logger' in locals():
            logger.close()
        print("\n程序已安全退出。")

if __name__ == '__main__':
    run()