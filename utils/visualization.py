# 文件: utils/visualization.py
import open3d as o3d

class RealtimeVisualizer:
    def __init__(self, window_name="RealtimeVisualizer"):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name)
        self.is_first_frame = True
        self._geometries = {}

    def update(self, geometries):
        # """
        # 更新可视化窗口中的几何体。
        # :param geometries: 一个包含所有要显示的o3d.geometry.Geometry3D对象的列表。
        # """
        # self.vis.clear_geometries()
        # for geom in geometries:
        #     if not geom.is_empty():
        #         self.vis.add_geometry(geom, reset_bounding_box=self.is_first_frame)
        
        # if self.is_first_frame and any(g.has_points() for g in geometries):
        #      self.is_first_frame = False

        # self.vis.poll_events()
        # self.vis.update_renderer()
# --- 1. 找出需要 新增/移除/更新 的几何体 ---
        
        # 将传入的几何体列表转换为以内存ID为键的字典，方便快速查找
        incoming_geoms = {id(g): g for g in geometries if g is not None}
        
        # 找出需要移除的几何体 (存在于旧的集合，但不存在于新的集合)
        geoms_to_remove = [self._geometries[gid] for gid in self._geometries if gid not in incoming_geoms]
        
        # 找出需要新增的几何体 (存在于新的集合，但不存在于旧的集合)
        geoms_to_add = [incoming_geoms[gid] for gid in incoming_geoms if gid not in self._geometries]
        
        # 找出需要更新的几何体 (同时存在于新旧集合中)
        geoms_to_update = [incoming_geoms[gid] for gid in incoming_geoms if gid in self._geometries]

        # --- 2. 执行操作 ---
        
        # 移除
        for geom in geoms_to_remove:
            self.vis.remove_geometry(geom, reset_bounding_box=False)

        # 新增
        for geom in geoms_to_add:
            self.vis.add_geometry(geom, reset_bounding_box=self.is_first_frame)
            # 仅在第一次添加几何体时重置视角
            if self.is_first_frame:
                self.is_first_frame = False

        # 更新
        for geom in geoms_to_update:
            self.vis.update_geometry(geom)

        # --- 3. 更新内部状态并渲染 ---
        self._geometries = incoming_geoms
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()