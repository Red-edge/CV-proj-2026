#!/usr/bin/env python3
"""
Camera-to-Angle Mapper for Gimbal Tracking
Converts pixel coordinates (bbox center) to Yaw/Pitch angles relative to optical axis.
"""
import numpy as np

class CameraAngleMapper:
    def __init__(self, h_fov_deg: float = 45.0, v_fov_deg: float = 30.0, 
                 img_w: int = 1280, img_h: int = 720):
        self.h_fov = np.radians(h_fov_deg)
        self.v_fov = np.radians(v_fov_deg)
        self.cx = img_w / 2.0
        self.cy = img_h / 2.0
        # 等效焦距 (像素单位)
        self.fx = self.cx / np.tan(self.h_fov / 2.0)
        self.fy = self.cy / np.tan(self.v_fov / 2.0)
        print(f"📐 Mapper init: {img_w}x{img_h}, HFOV={h_fov_deg}°, VFOV={v_fov_deg}°")

    def pixel_to_angle(self, px: float, py: float) -> tuple:
        """
        将像素坐标转换为相对于光轴的角度 (度)
        yaw: 水平偏角 (右正左负)
        pitch: 垂直偏角 (上正下负)
        """
        # 图像Y轴向下，世界坐标系Y轴向上，需取反
        yaw = np.degrees(np.arctan2(px - self.cx, self.fx))
        pitch = -np.degrees(np.arctan2(py - self.cy, self.fy))
        return yaw, pitch

    def angle_to_pixel(self, yaw_deg: float, pitch_deg: float) -> tuple:
        """反向映射：角度 → 像素坐标（用于可视化）"""
        yaw_rad = np.radians(yaw_deg)
        pitch_rad = np.radians(pitch_deg)
        px = self.cx + self.fx * np.tan(yaw_rad)
        py = self.cy - self.fy * np.tan(pitch_rad)  # Y轴反向
        return px, py