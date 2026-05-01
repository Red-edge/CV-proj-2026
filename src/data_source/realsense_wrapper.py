"""
RealSense T265 Data Source Wrapper
Handles camera initialization, frame capture, and VIO pose streaming.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import time
import pyrealsense2 as rs
import cv2

class RealSenseT265:
    def __init__(self, width=848, height=800, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.is_initialized = False

    def initialize(self):
        try:
            # 1. 配置双目前视鱼眼 (Y8 灰度)
            self.cfg.enable_stream(rs.stream.fisheye, 1, self.width, self.height, rs.format.y8, self.fps)
            self.cfg.enable_stream(rs.stream.fisheye, 2, self.width, self.height, rs.format.y8, self.fps)
            
            # 2. 配置位姿流 (✅ 关键修复：不传任何格式/尺寸参数)
            self.cfg.enable_stream(rs.stream.pose)
            
            # 3. 启动管线
            self.pipe.start(self.cfg)
            self.is_initialized = True
            print(f"✅ T265 初始化成功: {self.width}x{self.height} @ {self.fps}fps")
            return True
        except Exception as e:
            print(f"⚠ T265 初始化失败: {e}")
            self.is_initialized = False
            return False

    def get_frame(self):
        if not self.is_initialized: return None, None
        try:
            frames = self.pipe.wait_for_frames()
            fisheye = frames.get_fisheye_frame(1)
            if not fisheye: return None, None
            
            img = np.asanyarray(fisheye.get_data())
            
            # 🔥 提取 T265 IMU 角速度 (rad/s)
            ang_vel = None
            pose_frame = frames.get_pose_frame()
            if pose_frame:
                pd = pose_frame.get_pose_data()
                # (wx, wy, wz) 对应 Pitch, Yaw, Roll 角速度
                ang_vel = (pd.angular_velocity.x, pd.angular_velocity.y, pd.angular_velocity.z)
                
            return img, ang_vel
        except Exception as e:
            print(f"⚠ 获取 T265 帧失败: {e}")
            return None, None

    def stop(self):
        if self.is_initialized:
            self.pipe.stop()
            self.is_initialized = False
            print("🛑 T265 已停止")
    
    def get_intrinsics(self) -> Dict[str, float]:
        """Get camera intrinsics."""
        return self.intrinsics.copy()


class VideoFallback:
    """
    Fallback to video file or webcam when T265 is not available.
    """
    
    def __init__(self, source: str = '0'):
        """
        Args:
            source: '0' for webcam, or path to video file
        """
        self.source = source
        self.cap = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize video capture."""
        import cv2
        
        try:
            if self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                print(f"⚠ Failed to open video source: {self.source}")
                return False
            
            self.is_initialized = True
            print(f"✓ Video fallback initialized: {self.source}")
            return True
            
        except Exception as e:
            print(f"⚠ Error initializing video: {e}")
            return False
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], None]:
        """
        Get a frame from video source.
        
        Returns:
            frame: BGR image (H, W, 3) or None
            pose: Always None for fallback
        """
        if not self.is_initialized or self.cap is None:
            return None, None
        
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        return frame, None
    
    def stop(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
            self.is_initialized = False
            print("Video capture stopped")
    
    def get_intrinsics(self) -> Dict[str, float]:
        """Return dummy intrinsics for fallback mode."""
        return {'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240}
