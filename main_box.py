# 文件: main.py

import time
import open3d as o3d
import numpy as np
# 从各个模块导入所需的类和配置
from utils.config import (PCD_FOLDER, MODEL_PATH, PREPROCESSING_CONFIG, 
                    DETECTION_CONFIG, VISUALIZATION_CONFIG, APP_CONFIG)
from utils.preprocess import PointCloudPreprocessor
from utils.detection import DynamicObjectDetector
from utils.io import create_pcd_stream_from_files
from utils.visualization import RealtimeVisualizer
from utils.logger import PerformanceLogger  

def run():
    """
    主执行函数，串联整个处理流程。
    """
    print("--- 启动实时3D动态物体检测系统 ---")
    verbose = False
    # --- 1. 初始化 ---
    print("正在初始化模块...")
    try:
        preprocessor = PointCloudPreprocessor(model_pcd_path=MODEL_PATH)
        detector = DynamicObjectDetector(DETECTION_CONFIG)
        visualizer = RealtimeVisualizer(VISUALIZATION_CONFIG["window_name"])
        pcd_stream = create_pcd_stream_from_files(
            PCD_FOLDER, 
            start_index=APP_CONFIG["pcd_start_index"], 
            frame_delay=APP_CONFIG["frame_delay_sec"]
        )
        if verbose:
            logger = PerformanceLogger()  
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    frame_index = 0
    pcd_prev_clean = None
    print("初始化完成，开始处理数据流...")

    # --- 2. 实时处理循环 ---
    try:
        for pcd_current_raw in pcd_stream:
            start_time = time.time()

            # 步骤 A: 预处理
            pcd_current_clean = preprocessor.preprocess(pcd_current_raw, PREPROCESSING_CONFIG)
            print(f"原始点数: {len(pcd_current_raw.points)} -> 预处理后点数: {len(pcd_current_clean.points)}")
            mid_time = time.time()
            preprocess_time = (mid_time - start_time) * 1000  # 转换为毫秒
            if pcd_prev_clean is None:
                pcd_prev_clean = pcd_current_clean
                continue
            
            # 步骤 B: 动态检测
            static_pcd, dynamic_pcd, dynamic_boxes, static_boxes = detector.detect_all_unified(pcd_current_clean, pcd_prev_clean)
            detect_time = (time.time() - mid_time) * 1000
            # 步骤 C: 准备可视化
            static_pcd.paint_uniform_color(VISUALIZATION_CONFIG["static_color"])
            dynamic_pcd.paint_uniform_color(VISUALIZATION_CONFIG["dynamic_color"])

            # 步骤 D: 更新可视化窗口
            geometries_to_draw = [static_pcd, dynamic_pcd] + dynamic_boxes + static_boxes
            visualizer.update(geometries_to_draw)

            # 步骤 E: 更新上一帧状态
            pcd_prev_clean = pcd_current_clean
            size_bytes = len(np.asarray(dynamic_pcd.points, dtype=np.float32).tobytes())
            percent = len(dynamic_pcd.points) / (len(dynamic_pcd.points)+len(static_pcd.points)) 
            
            print(f"帧 {frame_index}: 处理时间: {preprocess_time:.2f} ms, 动态点数: {len(dynamic_pcd.points)}, "
                  f"占比: {percent:.2%}, 检测时间：{detect_time:.2f}, 数据大小: {size_bytes / 1024:.2f} KB"
                  f"检测到物体数量 -> 动态框: {len(dynamic_boxes)} 个, 静态框: {len(static_boxes)} 个")
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