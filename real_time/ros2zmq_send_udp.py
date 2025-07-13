#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

import zmq
import msgpack
import msgpack_numpy as m

import numpy as np
import open3d as o3d
import time
import threading

from utils.config import PREPROCESSING_CONFIG, DETECTION_CONFIG, MODEL_PATH
from utils.preprocess import PointCloudPreprocessor
from utils.detection import DynamicObjectDetector


def pointcloud2_to_open3d(pointcloud2_msg: PointCloud2, roi_bounds: dict) -> o3d.geometry.PointCloud:
    """
    Efficiently converts PointCloud2 to Open3D, performing ROI filtering in NumPy.
    This version correctly handles plain (N, 3) NumPy arrays.
    """
    # Step 1: Efficiently read point cloud data into a plain NumPy array.
    # The output is a standard (N, 3) array.
    points_np = pc2.read_points_numpy(
        pointcloud2_msg,
        field_names=("x", "y", "z"),
        skip_nans=True
    )

    if points_np.shape[0] == 0:
        return o3d.geometry.PointCloud()

    # Step 2: Perform fast ROI filtering using integer indices for columns.
    # THIS IS THE PART THAT WAS FIXED
    mask = (
        (points_np[:, 0] >= roi_bounds['x_min']) &  # Use index 0 for 'x'
        (points_np[:, 0] <= roi_bounds['x_max']) &
        (points_np[:, 1] >= roi_bounds['y_min']) &  # Use index 1 for 'y'
        (points_np[:, 1] <= roi_bounds['y_max']) &
        (points_np[:, 2] >= roi_bounds['z_min']) &  # Use index 2 for 'z'
        (points_np[:, 2] <= roi_bounds['z_max'])
    )
    points_filtered_np = points_np[mask]

    if points_filtered_np.shape[0] == 0:
        return o3d.geometry.PointCloud()

    # Step 3: Copy the filtered, smaller NumPy array to Open3D.
    pcd = o3d.geometry.PointCloud()
    old_time = time.time()
    pcd.points = o3d.utility.Vector3dVector(points_filtered_np)
    new_time = time.time()
    # print(f"PointCloud conversion took {(new_time - old_time)*1000:.2f} seconds for {points_filtered_np.shape[0]} points.")
    return pcd

class RosToZmqBridgeNode(Node):
    """
    一个监听ROS点云话题，处理后通过ZMQ发布的桥接节点。
    """
    def __init__(self):
        super().__init__('ros_to_zmq_bridge_node')

        # --- 1. 设置 ZMQ 发布者 (MODIFIED PART) ---
        self.get_logger().info("正在初始化 ZeroMQ RADIO Socket...")
        self.zmq_context = zmq.Context()
        self.zmq_radio = self.zmq_context.socket(zmq.RADIO)
        receiver_ip = "10.129.138.20"  # 这里必须是接收端电脑的明确 IP 地址
        port = 5555
        connection_str = f"udp://{receiver_ip}:{port}" # <--- 修改
        self.zmq_radio.connect(connection_str) # <--- 修改
        self.get_logger().info(f"ZMQ RADIO 已连接到 {connection_str}") 

        # --- 2. 设置 ROS 2 订阅者
        self.declare_parameter('subscribe_topic', '/rslidar_points')
        topic_name = self.get_parameter('subscribe_topic').get_parameter_value().string_value
        
        exact_match_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1, 
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.subscription = self.create_subscription(
            PointCloud2, topic_name, self.lidar_callback, qos_profile=exact_match_qos_profile
        )
        self.get_logger().info(f"已成功订阅ROS话题: '{topic_name}'")

        # --- 3. 实例化处理模块
        self.preprocessor = PointCloudPreprocessor(model_pcd_path=MODEL_PATH, verbose=False)
        self.detector = DynamicObjectDetector(DETECTION_CONFIG)
        
        self.pcd_prev_clean = None
        self.frame_index = 0
        self.my_roi = {
        'x_min': 2.0, 'x_max': 80.0,
        'y_min': -30.0, 'y_max': 30.0,
        'z_min': -3.0,  'z_max': 5.0
    }
        
        # --- 4. 添加处理锁 ---
        self.processing_lock = threading.Lock()


    def lidar_callback(self, msg: PointCloud2):
        # --- 核心改动：检查锁的状态 ---
        # 尝试获取锁，但不要阻塞等待。如果锁已被其他线程占用（意味着处理仍在进行），则立即返回。
        if not self.processing_lock.acquire(blocking=False):
            self.get_logger().warn("处理速度跟不上，丢弃一帧数据以保持实时性。")
            return

        try:
            # --- 以下是您的原始处理逻辑 ---
            # proc_start_time = time.time()
            pcd_current_raw = pointcloud2_to_open3d(msg, self.my_roi)
            if not pcd_current_raw.has_points(): 
                self.processing_lock.release()
                return 
            proc_start_time = time.time()
            pcd_current_clean = self.preprocessor.preprocess(pcd_current_raw, PREPROCESSING_CONFIG)
            if self.pcd_prev_clean is None:
                self.pcd_prev_clean = pcd_current_clean
                self.frame_index += 1
                self.processing_lock.release()
                return 
                
            static_pcd, dynamic_pcd = self.detector.detect(pcd_current_clean, self.pcd_prev_clean)
            self.pcd_prev_clean = pcd_current_clean
            proc_end_time = time.time()
            
            self.frame_index += 1 # 保证每处理一帧，ID都增加
            if dynamic_pcd.has_points():
                dyn_points_np = np.asarray(dynamic_pcd.points, dtype=np.float32)
                
                send_timestamp = time.time()
                message_to_send = {
                    'frame_id': self.frame_index,
                    'proc_time_ms': (proc_end_time - proc_start_time) * 1000,
                    'dynamic_points': dyn_points_np,
                    'static_points_count': len(static_pcd.points),
                    'send_timestamp': send_timestamp
                }
                
                packed_message = msgpack.packb(message_to_send, default=m.encode)
                self.zmq_radio.send(packed_message, group=b'lidar')
                self.get_logger().info(f"已处理第 {self.frame_index} 帧, 时间为{(proc_end_time - proc_start_time) * 1000} ,并已通过ZMQ发送 {len(dyn_points_np)} 个动态点。")
            else:
                self.get_logger().info(f"已处理第 {self.frame_index} 帧, 耗时 {(proc_end_time - proc_start_time) * 1000:.2f} ms, 未检测到动态点。")
        
        
        finally:
             if self.processing_lock.locked():
                self.processing_lock.release()


    def destroy_node(self):
        self.get_logger().info("正在关闭 ZeroMQ Publisher...")
        self.zmq_radio.close()
        self.zmq_context.term()
        self.get_logger().info("ZMQ 资源已安全关闭。")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    bridge_node = None
    try:
        bridge_node = RosToZmqBridgeNode()
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        pass
    finally:
        if bridge_node:
            bridge_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()