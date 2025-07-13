# 文件: main.py

import time
import open3d as o3d
import numpy as np
import cv2 # 导入OpenCV

# 从各个模块导入所需的类和配置
from utils.config import (PCD_FOLDER, MODEL_PATH, PREPROCESSING_CONFIG, 
                    VISUALIZATION_CONFIG, APP_CONFIG, BEV_CONFIG) # 导入BEV_CONFIG
from utils.preprocess import PointCloudPreprocessor
from utils.bev import BEVProcessor # 导入新的BEV处理器
from utils.io import create_pcd_stream_from_files
from utils.visualization import RealtimeVisualizer

def run_bev_detection():
    """
    使用BEV进行动态检测的主执行函数。
    """
    print("--- 启动基于BEV的实时3D动态物体检测系统 ---")

    # --- 1. 初始化 ---
    print("正在初始化模块...")
    try:
        preprocessor = PointCloudPreprocessor(model_pcd_path=MODEL_PATH, verbose=False)
        bev_processor = BEVProcessor(BEV_CONFIG) # 初始化BEV处理器
        visualizer_3d = RealtimeVisualizer(VISUALIZATION_CONFIG["window_name"])
        pcd_stream = create_pcd_stream_from_files(
            PCD_FOLDER, 
            start_index=APP_CONFIG["pcd_start_index"], 
            frame_delay=APP_CONFIG["frame_delay_sec"]
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    bev_prev = None
    print("初始化完成，开始处理数据流...")

    # --- 2. 实时处理循环 ---
    try:
        for pcd_current_raw in pcd_stream:
            start_time = time.time()

            # 步骤 A: 3D预处理
            pcd_current_clean = preprocessor.preprocess(pcd_current_raw, PREPROCESSING_CONFIG)

            # 步骤 B: 转换为BEV图像
            bev_curr = bev_processor.point_cloud_to_bev(pcd_current_clean)

            if bev_prev is None:
                bev_prev = bev_curr
                continue
            
            # 步骤 C: 在BEV上进行动态检测
            motion_mask, bboxes_2d = bev_processor.detect_motion(bev_curr, bev_prev)

            # 步骤 D: 可视化
            # 3D 可视化：显示预处理后的点云
            pcd_current_clean.paint_uniform_color([0.5, 0.5, 0.5]) # 灰色静态场景
            visualizer_3d.update([pcd_current_clean])

            # 2D 可视化：显示BEV图和检测结果
            # 将单通道的BEV图转为三通道的彩色图以便绘制
            bev_display = cv2.cvtColor(bev_curr, cv2.COLOR_GRAY2BGR)
            motion_mask_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

            # 在BEV图上绘制红色的2D包围盒
            for (x, y, w, h) in bboxes_2d:
                cv2.rectangle(bev_display, (x, y), (x + w, y + h), (0, 0, 255), 2) # 红色包围盒

            # 将两张图水平拼接起来显示
            combined_display = np.hstack([bev_display, motion_mask_display])
            cv2.imshow("BEV Detection (Left: Scene | Right: Motion)", combined_display)
            
            # 按 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 步骤 E: 更新上一帧状态
            bev_prev = bev_curr

            # 打印统计信息
            proc_time_ms = (time.time() - start_time) * 1000
            print(f"帧处理耗时: {proc_time_ms:.1f} ms | "
                  f"检测到动态物体: {len(bboxes_2d)} 个", end="\r")

    except KeyboardInterrupt:
        print("\n用户中断，正在关闭程序...")
    finally:
        # --- 3. 清理 ---
        visualizer_3d.close()
        cv2.destroyAllWindows()
        print("\n程序已安全退出。")

if __name__ == '__main__':
    run_bev_detection()