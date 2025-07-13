# 文件: utils/logger.py

import csv
import os
from datetime import datetime

class PerformanceLogger:
    """
    一个用于记录性能指标到 CSV 文件的工具类。
    """
    def __init__(self, log_folder="logs"):
        """
        初始化logger，创建一个带时间戳的日志文件并写入表头。
        :param log_folder: 存放日志文件的文件夹。
        """
        # 确保日志文件夹存在
        os.makedirs(log_folder, exist_ok=True)
        
        # 使用当前时间创建唯一的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_folder, f"performance_log_{timestamp}.csv")
        
        # 打开文件句柄，准备写入。newline='' 是写入csv文件的标准做法
        self.file_handle = open(self.log_path, 'w', newline='', encoding='utf-8')
        
        # 创建csv写入器
        self.writer = csv.writer(self.file_handle)
        
        # 写入表头
        self.writer.writerow(['frame_index', 'proc_time_ms', 'dynamic_points_count','static_points_count', 'percent', 'size_bytes'])
        
        print(f"性能日志将记录在: {self.log_path}")

    def log(self, frame_index, proc_time_ms, dynamic_points_count, static_points_count, percent, size_bytes):
        self.writer.writerow([frame_index, proc_time_ms, dynamic_points_count, static_points_count, percent, size_bytes])

    def close(self):
        """
        安全地关闭文件句柄。
        """
        if self.file_handle:
            self.file_handle.close()
            print(f"\n性能日志已保存: {self.log_path}")