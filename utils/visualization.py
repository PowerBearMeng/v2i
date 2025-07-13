# 文件: utils/visualization.py
import open3d as o3d

class RealtimeVisualizer:
    def __init__(self, window_name="RealtimeVisualizer"):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name)
        self.is_first_frame = True

    def update(self, geometries):
        """
        更新可视化窗口中的几何体。
        :param geometries: 一个包含所有要显示的o3d.geometry.Geometry3D对象的列表。
        """
        self.vis.clear_geometries()
        for geom in geometries:
            if not geom.is_empty():
                self.vis.add_geometry(geom, reset_bounding_box=self.is_first_frame)
        
        if self.is_first_frame and any(g.has_points() for g in geometries):
             self.is_first_frame = False

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()