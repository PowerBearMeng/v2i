#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zmq
import msgpack
import msgpack_numpy as m
import numpy as np
import open3d as o3d
import time

class ZmqClientVisualizer:
    """
    一个通过ZMQ接收点云数据并使用Open3D进行实时可视化的客户端。
    """
    def __init__(self, server_ip="localhost", port=5555):
        """
        初始化ZMQ连接和Open3D可视化窗口。

        :param server_ip: 服务器的IP地址。
        :param port: 服务器的端口号。
        """
        # --- 1. 设置 ZMQ 订阅者 ---
        print("正在初始化 ZeroMQ Subscriber...")
        self.zmq_context = zmq.Context()
        # 创建一个 SUB (订阅) 类型的套接字
        self.zmq_subscriber = self.zmq_context.socket(zmq.SUB)
        # 连接到服务器的地址和端口
        connection_str = f"tcp://{server_ip}:{port}"
        self.zmq_subscriber.connect(connection_str)
        self.zmq_subscriber.subscribe(b"")
        print(f"ZMQ Subscriber 已连接到 {connection_str}")

        
        # --- 2. 设置 Open3D 可视化器 ---
        print("正在初始化 Open3D Visualizer...")
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Dynamic PointCloud Viewer", width=800, height=600)
        # 创建一个空的点云对象，后续会用接收到的数据更新它
        self.pcd = o3d.geometry.PointCloud()
        self.is_geom_added = False # 标志位，用于判断几何体是否已添加到可视化窗口
        print("可视化窗口已准备就绪。")
        
        # --- 3. 设置坐标轴 ---
        # 创建一个坐标轴几何体，尺寸可以根据你的场景调整
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        self.vis.add_geometry(coordinate_frame)


    def run(self):
        """
        启动主循环，接收数据并更新可视化。
        """
        print("\n客户端已启动，正在等待从服务器接收数据...")
        try:
            while True:
                # --- 4. 接收并解包数据 ---
                # 阻塞等待，直到接收到一条消息
                packed_message = self.zmq_subscriber.recv()
                # recv_timestamp = time.time()
                # 使用 msgpack 和 msgpack_numpy 进行解包
                message = msgpack.unpackb(packed_message, object_hook=m.decode)
                
                # --- 5. 处理和显示数据 ---
                frame_id = message['frame_id']
                proc_time = message['proc_time_ms']
                dyn_points_np = message['dynamic_points']
                static_count = message['static_points_count']
                # send_timestamp = message['send_timestamp']
                # latency = (recv_timestamp - send_timestamp ) * 1000  # 转换为毫秒

                print(f"接收到第 {frame_id} 帧: "
                      f"动态点 {len(dyn_points_np)} 个, "
                      f"静态点 {static_count} 个, "
                      f"服务器处理耗时: {proc_time:.2f} ms, "
                      )

                # --- 6. 更新 Open3D 点云 ---
                if dyn_points_np.size > 0:
                    # 将接收到的numpy数组更新到点云对象的points属性
                    self.pcd.points = o3d.utility.Vector3dVector(dyn_points_np)
                    # 为点云上色，方便观察 (例如，红色)
                    self.pcd.paint_uniform_color([1.0, 0, 0])
                    
                    if not self.is_geom_added:
                        # 如果是第一次接收到点云，则将其添加到可视化窗口
                        self.vis.add_geometry(self.pcd)
                        self.is_geom_added = True
                    else:
                        # 如果已经添加过，则只更新几何体
                        self.vis.update_geometry(self.pcd)
                else:
                    # 如果没有动态点，则清空点云显示
                    self.pcd.clear()
                    if self.is_geom_added:
                        self.vis.update_geometry(self.pcd)

                # 处理窗口事件，并刷新视图
                self.vis.poll_events()
                self.vis.update_renderer()

        except KeyboardInterrupt:
            print("\n检测到用户中断 (Ctrl+C)。")
        finally:
            # --- 7. 清理资源 ---
            self.close()

    def close(self):
        """
        安全地关闭ZMQ套接字和Open3D窗口。
        """
        print("正在关闭 ZeroMQ Subscriber...")
        self.zmq_subscriber.close()
        self.zmq_context.term()
        print("正在销毁 Open3D 窗口...")
        self.vis.destroy_window()
        print("客户端已成功关闭。")

def main():
    server_ip = "10.29.73.37"
    client = ZmqClientVisualizer(server_ip=server_ip)
    client.run()
    
if __name__ == '__main__':
    main()