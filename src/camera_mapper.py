#!/usr/bin/env python3
import numpy as np

class CameraAngleMapper:
    def __init__(self, h_fov_deg: float, v_fov_deg: float):
        self.h_fov = np.radians(h_fov_deg)
        self.v_fov = np.radians(v_fov_deg)

    def pixel_to_angle(self, px: float, py: float, frame_w: int, frame_h: int) -> tuple:
        """
        动态计算角度，确保无论分辨率如何变化，中心点永远准确。
        """
        cx, cy = frame_w / 2.0, frame_h / 2.0
        fx = cx / np.tan(self.h_fov / 2.0)
        fy = cy / np.tan(self.v_fov / 2.0)
        
        # Yaw: 目标在右侧 (px > cx) -> 需向右转 (Yaw > 0)
        yaw = np.degrees(np.arctan2(px - cx, fx))
        
        # Pitch: 目标在下侧 (py > cy) -> 需向下转 (Pitch < 0)
        # 注意：这里加了负号，因为图像 Y 轴向下，而云台 Pitch 轴通常向上为正
        pitch = -np.degrees(np.arctan2(py - cy, fy))
        
        return yaw, pitch

    def angle_to_pixel(self, yaw_deg: float, pitch_deg: float) -> tuple:
        """反向映射：角度 → 像素坐标（用于可视化）"""
        yaw_rad = np.radians(yaw_deg)
        pitch_rad = np.radians(pitch_deg)
        px = self.cx + self.fx * np.tan(yaw_rad)
        py = self.cy - self.fy * np.tan(pitch_rad)  # Y轴反向
        return px, py