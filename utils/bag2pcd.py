import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

import sqlite3
import yaml
import numpy as np
import open3d as o3d
import os
from pathlib import Path

# === 用户参数 ===
bag_path = "/home/yty/mfh/robosense/0709_sta_03"  # 替换成你的bag路径
topic_name = "/rslidar_points"
output_dir = "/home/yty/mfh/mot/my_mot/pcd/"+ bag_path.split('/')[-1]

# === 读取 metadata.yaml 得到 topic_type ===
def get_topic_type(metadata_path, topic_name):
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
        topics = metadata['rosbag2_bagfile_information']['topics_with_message_count']
        for topic in topics:
            if topic['topic_metadata']['name'] == topic_name:
                return topic['topic_metadata']['type']
    return None

def pointcloud2_to_xyz_array(msg):
    cloud_points = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    xyz_list = []
    for p in cloud_points:
        xyz_list.append([p[0], p[1], p[2], p[3]])
    return np.array(xyz_list, dtype=np.float32)


# === 存储为pcd文件 ===
def save_pcd(xyzi_array, frame_id):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzi_array[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(xyzi_array[:, 3:4] / 255.0, (1, 3)))  # 以强度作为灰度颜色
    pcd_path = os.path.join(output_dir, f"frame_{frame_id:05d}.pcd")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved: {pcd_path}")

def main():
    db3_path = Path(bag_path) / (bag_path.split('/')[-1] + "_0.db3")
    print(f"正在处理数据库: {db3_path}")
    metadata_path = Path(bag_path) / "metadata.yaml"
    os.makedirs(output_dir, exist_ok=True)

    topic_type = get_topic_type(metadata_path, topic_name)
    if topic_type != "sensor_msgs/msg/PointCloud2":
        print(f"错误: Topic {topic_name} 类型不是 PointCloud2, 实际是 {topic_type}")
        return

    conn = sqlite3.connect(str(db3_path))
    cursor = conn.cursor()

    cursor.execute("SELECT id, topic_id, data, timestamp FROM messages")
    topic_dict = {}
    cursor.execute("SELECT id, name FROM topics")
    for topic_id, name in cursor.fetchall():
        topic_dict[topic_id] = name

    frame_id = 0
    cursor.execute("SELECT topic_id, data FROM messages")
    for topic_id, data in cursor.fetchall():
        if topic_dict[topic_id] == topic_name:
            msg = deserialize_message(data, PointCloud2)
            xyz_array = pointcloud2_to_xyz_array(msg)
            if xyz_array.shape[0] > 0:
                save_pcd(xyz_array, frame_id)
                frame_id += 1

    conn.close()
    print("✅ 全部完成！")

if __name__ == '__main__':
    main()
