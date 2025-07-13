#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 文件名: ros2_humble_listener.py

# 导入 ROS 2 Python 客户端库
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

# 导入消息类型和转换工具
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

# 导入标准库和第三方库
import numpy as np
import open3d as o3d
import time

# --- 导入您项目已有的模块 ---
# 确保这些文件与此脚本在同一Python包或路径下
from utils.config import PREPROCESSING_CONFIG, DETECTION_CONFIG, MODEL_PATH
from utils.preprocess import PointCloudPreprocessor
from utils.detection import DynamicObjectDetector
from utils.logger import PerformanceLogger

def pointcloud2_to_open3d(pointcloud2_msg: PointCloud2) -> o3d.geometry.PointCloud:
    """
    将 ROS 2 的 PointCloud2 消息转换为 Open3D 的 PointCloud 对象。
    这个函数与ROS 1版本基本兼容，因为消息结构相似。
    """
    # 从 PointCloud2 消息中读取点云数据
    points_list = []
    # 使用 sensor_msgs_py.point_cloud2 读取数据
    for point in pc2.read_points(pointcloud2_msg, skip_nans=True, field_names=("x", "y", "z")):
        points_list.append([point[0], point[1], point[2]])

    if not points_list:
        return o3d.geometry.PointCloud()

    points_np = np.array(points_list, dtype=np.float64)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    
    return pcd

class PointCloudProcessorNode(Node):
    """
    一个监听、处理并发布点云数据的 ROS 2 节点。
    """
    def __init__(self):
        # 1. 初始化父类 Node，并设置节点名
        super().__init__('pointcloud_processor_node')
        
        # 声明参数，使其可以在启动时配置
        self.declare_parameter('subscribe_topic', '/rslidar_points')
        topic_name = self.get_parameter('subscribe_topic').get_parameter_value().string_value
        
        exact_match_qos_profile = QoSProfile(
            # 可靠性：与发布者一致，设置为 RELIABLE
            reliability=QoSReliabilityPolicy.RELIABLE,

            # 历史策略：与发布者一致，设置为 KEEP_LAST
            history=QoSHistoryPolicy.KEEP_LAST,

            # 队列深度：与发布者一致，设置为 10
            depth=10,

            # 持久性：与发布者一致，设置为 VOLATILE
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.get_logger().info("ROS 2 节点初始化...")

        # 2. 实例化您已有的处理模块
        try:
            self.get_logger().info("正在初始化点云预处理器...")
            self.preprocessor = PointCloudPreprocessor(model_pcd_path=MODEL_PATH, verbose=False)
            self.get_logger().info("正在初始化动态物体检测器...")
            self.detector = DynamicObjectDetector(DETECTION_CONFIG)
            # self.get_logger().info("正在初始化性能记录器...")
            # self.logger = PerformanceLogger(log_folder="ros2_logs")
        except (ValueError, RuntimeError) as e:
            self.get_logger().error(f"初始化处理模块失败: {e}")
            # 在ROS 2中，我们不直接关闭，而是让主程序处理
            raise e
            
        # 3. 初始化状态变量
        self.pcd_prev_clean = None
        self.frame_index = 0

        # 4. 创建 ROS 2 订阅者
        self.subscription = self.create_subscription(
            PointCloud2,
            topic_name,
            self.lidar_callback,
            qos_profile=exact_match_qos_profile # 使用为传感器数据优化的QoS配置
        )
        

    def lidar_callback(self, msg: PointCloud2):
        """
        处理从雷达话题接收到的每一帧点云数据的回调函数。
        """
        self.get_logger().info(f"--- 收到第 {self.frame_index} 帧数据 ---")

        # 1. 格式转换: ROS2 PointCloud2 -> Open3D PointCloud
        proc_start_time = time.time()
        pcd_current_raw = pointcloud2_to_open3d(msg)
        if not pcd_current_raw.has_points():
            self.get_logger().warn("收到的点云为空，跳过处理。")
            return

        # 2. 调用预处理器
        pcd_current_clean = self.preprocessor.preprocess(pcd_current_raw, PREPROCESSING_CONFIG)

        # 3. 如果是第一帧，则存储并等待下一帧
        if self.pcd_prev_clean is None or not self.pcd_prev_clean.has_points():
            self.pcd_prev_clean = pcd_current_clean
            self.frame_index += 1
            self.get_logger().info("已存储第一帧作为参考，等待下一帧进行动态检测。")
            return

        # 4. 调用动态检测器
        static_pcd, dynamic_pcd = self.detector.detect(pcd_current_clean, self.pcd_prev_clean)
        proc_end_time = time.time()
        # 5. 更新前一帧以备下次使用
        self.pcd_prev_clean = pcd_current_clean

        proc_time_ms = (proc_end_time - proc_start_time) * 1000
        
        # 6. 日志记录
        dyn_points_count = len(dynamic_pcd.points)
        stat_points_count = len(static_pcd.points)
        total_points_count = dyn_points_count + stat_points_count
        percent = (dyn_points_count / total_points_count * 100) if total_points_count > 0 else 0
        
        print(f"处理耗时: {proc_time_ms:.2f} ms | "
              f"动态点数: {dyn_points_count} | "
              f"静态点数: {stat_points_count} | "
              f"动态点占比: {percent:.2f}%")

        self.frame_index += 1

    def destroy_node(self):
        """在节点销毁时关闭日志文件。"""
        # self.get_logger().info("正在关闭日志文件...")
        # self.logger.close()
        super().destroy_node()

def main(args=None):
    # 初始化 rclpy
    rclpy.init(args=args)
    
    processor_node = None
    try:
        # 创建并实例化节点
        processor_node = PointCloudProcessorNode()
        # 保持节点运行，等待回调函数被调用
        rclpy.spin(processor_node)
    except Exception as e:
        if processor_node:
            processor_node.get_logger().error(f"节点运行时发生未处理的异常: {e}")
    finally:
        # 销毁节点并关闭 rclpy
        if processor_node:
            processor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()