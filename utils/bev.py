# 文件: utils/bev.py

import open3d as o3d
import numpy as np
import cv2 # 需要OpenCV库: pip install opencv-python

class BEVProcessor:
    def __init__(self, config):
        """
        初始化鸟瞰图处理器。
        :param config: 包含BEV参数的字典，例如BEV_CONFIG。
        """
        # BEV图像的尺寸（像素）
        self.grid_size = config.get('grid_size', [512, 512])
        # 真实世界范围（米），例如[-25.6m, 25.6m]
        self.lidar_range = config.get('lidar_range', [-25.6, 25.6, -25.6, 25.6])
        # 每个像素代表的真实尺寸（米/像素）
        self.cell_size = (self.lidar_range[1] - self.lidar_range[0]) / self.grid_size[0]

    def point_cloud_to_bev(self, pcd):
        """
        将3D点云转换为2D鸟瞰图（BEV）。
        """
        # 创建一个空的黑色图像
        bev_image = np.zeros(self.grid_size, dtype=np.uint8)
        
        points = np.asarray(pcd.points)

        # 筛选出在定义范围内的点
        mask = (points[:, 0] > self.lidar_range[0]) & (points[:, 0] < self.lidar_range[1]) & \
               (points[:, 1] > self.lidar_range[2]) & (points[:, 1] < self.lidar_range[3])
        points_in_range = points[mask]

        # 将点云坐标转换为图像像素坐标
        # 这里我们假设雷达在(0,0)，图像中心对应(0,0)
        x_coords = ((points_in_range[:, 0] - self.lidar_range[0]) / self.cell_size).astype(np.int32)
        y_coords = ((points_in_range[:, 1] - self.lidar_range[2]) / self.cell_size).astype(np.int32)

        # 在BEV图像上将对应的像素点亮为白色
        bev_image[y_coords, x_coords] = 255
        
        return bev_image

    def detect_motion(self, bev_curr, bev_prev):
        """
        在两张BEV图像上通过帧间差分检测运动。
        """
        # 1. 计算两帧的差异
        diff_image = cv2.absdiff(bev_curr, bev_prev)
        
        # 2. 对差异图像进行二值化
        _, thresh_image = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)
        
        # 3. 使用形态学操作去噪并连接物体
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(thresh_image, kernel, iterations=2)
        
        # 4. 查找运动物体的轮廓
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. 为每个轮廓生成2D包围盒
        bboxes_2d = [cv2.boundingRect(c) for c in contours]
        
        return dilated_image, bboxes_2d