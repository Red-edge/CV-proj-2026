"""
RealSense T265 Data Source Wrapper
Handles camera initialization, frame capture, and VIO pose streaming.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import time


class RealSenseT265:
    """
    Wrapper for Intel RealSense T265 camera.
    Provides fisheye images and 6DoF pose data.
    """
    
    def __init__(self, 
                 width: int = 848, 
                 height: int = 800, 
                 fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        
        self.pipeline = None
        self.config = None
        self.is_initialized = False
        
        # Camera intrinsics (will be populated after initialization)
        self.intrinsics = {
            'fx': 0, 'fy': 0,
            'cx': 0, 'cy': 0
        }
        
    def initialize(self) -> bool:
        """
        Initialize the T265 camera.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import pyrealsense2 as rs
            
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable fisheye streams (T265 has two fisheye cameras)
            self.config.enable_stream(
                rs.stream.fisheye, 1, 
                rs.format.y8, 
                self.width, self.height, self.fps
            )
            self.config.enable_stream(
                rs.stream.fisheye, 2,
                rs.format.y8,
                self.width, self.height, self.fps
            )
            
            # Enable pose stream for VIO data
            self.config.enable_stream(rs.stream.pose, rs.format.sixdof)
            
            # Start pipeline
            self.pipeline.start(self.config)
            
            # Get intrinsics from first fisheye camera
            profile = self.pipeline.get_active_profile()
            fisheye_profile = profile.get_stream(rs.stream.fisheye, 1)
            intrinsics_rs = fisheye_profile.as_video_stream_profile().get_intrinsics()
            
            self.intrinsics = {
                'fx': intrinsics_rs.fx,
                'fy': intrinsics_rs.fy,
                'cx': intrinsics_rs.ppx,
                'cy': intrinsics_rs.ppy
            }
            
            self.is_initialized = True
            print(f"✓ T265 initialized: {self.width}x{self.height}@{self.fps}fps")
            print(f"  Intrinsics: fx={self.intrinsics['fx']:.1f}, fy={self.intrinsics['fy']:.1f}, "
                  f"cx={self.intrinsics['cx']:.1f}, cy={self.intrinsics['cy']:.1f}")
            
            return True
            
        except ImportError:
            print("⚠ pyrealsense2 not installed")
            return False
        except Exception as e:
            print(f"⚠ Failed to initialize T265: {e}")
            return False
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Get a fisheye frame and corresponding pose data.
        
        Returns:
            frame: Grayscale image (H, W) or None
            pose: Dictionary with pose data or None
        """
        if not self.is_initialized or self.pipeline is None:
            return None, None
        
        try:
            import pyrealsense2 as rs
            
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            
            # Get fisheye frame (using left camera, stream 1)
            fisheye_frame = frames.get_fisheye_frame(1)
            if fisheye_frame is None:
                return None, None
            
            frame = np.asanyarray(fisheye_frame.get_data())
            
            # Get pose data
            pose_frame = frames.first_or_default(rs.stream.pose, rs.format.sixdof)
            pose_data = None
            
            if pose_frame:
                pose = pose_frame.as_pose_frame().get_pose_data()
                pose_data = {
                    'translation': {
                        'x': pose.translation.x,
                        'y': pose.translation.y,
                        'z': pose.translation.z
                    },
                    'rotation': {
                        'x': pose.rotation.x,
                        'y': pose.rotation.y,
                        'z': pose.rotation.z,
                        'w': pose.rotation.w
                    },
                    'velocity': {
                        'x': pose.velocity.x,
                        'y': pose.velocity.y,
                        'z': pose.velocity.z
                    },
                    'angular_velocity': {
                        'x': pose.angular_velocity.x,
                        'y': pose.angular_velocity.y,
                        'z': pose.angular_velocity.z
                    },
                    'tracker_confidence': pose.tracker_confidence,
                    'frame_number': pose_frame.frame_number,
                    'timestamp': pose_frame.get_timestamp()
                }
            
            return frame, pose_data
            
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None, None
    
    def stop(self):
        """Stop the camera pipeline."""
        if self.pipeline:
            self.pipeline.stop()
            self.is_initialized = False
            print("T265 stopped")
    
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
