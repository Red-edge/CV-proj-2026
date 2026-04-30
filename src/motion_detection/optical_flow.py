"""
Motion Detection Module
Implements optical flow computation and ego-motion compensation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any


class OpticalFlowCalculator:
    """
    Computes optical flow using Lucas-Kanade or Farneback methods.
    """
    
    def __init__(self, 
                 flow_type: str = 'sparse',
                 max_features: int = 500,
                 quality_level: float = 0.01,
                 min_distance: int = 10):
        """
        Args:
            flow_type: 'sparse' (Lucas-Kanade) or 'dense' (Farneback)
            max_features: Maximum number of corners to track
            quality_level: Quality level for corner detection
            min_distance: Minimum distance between corners
        """
        self.flow_type = flow_type
        self.max_features = max_features
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Feature detector parameters
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7,
            useHarrisDetector=False,
            k=0.04
        )
        
        # State
        self.old_frame: Optional[np.ndarray] = None
        self.old_points: Optional[np.ndarray] = None
        self.colors = np.random.randint(0, 255, (max_features, 3), dtype=np.uint8)
    
    def compute(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Compute optical flow for a frame.
        
        Args:
            frame: Input image (grayscale or BGR)
            
        Returns:
            flow_vis: Visualization image with flow vectors
            flow_vectors: List of flow vector dictionaries
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        if self.old_frame is None:
            self.old_frame = gray.copy()
            self.old_points = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), []
        
        if self.flow_type == 'sparse':
            flow_vis, flow_vectors = self._sparse_flow(gray)
        else:
            flow_vis, flow_vectors = self._dense_flow(gray)
        
        self.old_frame = gray.copy()
        return flow_vis, flow_vectors
    
    def _sparse_flow(self, gray: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Compute sparse optical flow using Lucas-Kanade."""
        p0 = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        
        if p0 is None or self.old_points is None:
            self.old_points = p0
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), []
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_frame, gray, self.old_points, None, **self.lk_params
        )
        
        if p1 is None:
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), []
        
        # Select good points
        good_new = p1[st == 1]
        good_old = self.old_points[st == 1]
        
        # Draw flow vectors
        flow_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        flow_vectors = []
        
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            magnitude = np.sqrt((a - c)**2 + **(b - d)2)
            flow_vectors.append({
                'start': (int(c), int(d)),
                'end': (int(a), int(b)),
                'magnitude': magnitude
            })
            
            # Draw vector
            color = self.colors[i].tolist()
            flow_vis = cv2.line(flow_vis, (int(a), int(b)), (int(c), int(d)), color, 1)
            flow_vis = cv2.circle(flow_vis, (int(a), int(b)), 3, color, -1)
        
        # Update previous points
        self.old_points = good_new.reshape(-1, 1, 2)
        
        return flow_vis, flow_vectors
    
    def _dense_flow(self, gray: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Compute dense optical flow using Farneback method."""
        flow = cv2.calcOpticalFlowFarneback(
            self.old_frame, gray, None,
            pyr_scale=0.5, levels=3, winsize=21,
            iterations=3, poly_n=7, poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        
        # Convert to visualization
        hsv = np.zeros_like(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Extract flow vectors (sampled)
        flow_vectors = []
        step = 10
        h, w = gray.shape[:2]
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                mag = np.sqrt(fx**2 + fy**2)
                if mag > 1:
                    flow_vectors.append({
                        'start': (x, y),
                        'end': (int(x + fx), int(y + fy)),
                        'magnitude': mag
                    })
        
        return flow_vis, flow_vectors
    
    def reset(self):
        """Reset the calculator state."""
        self.old_frame = None
        self.old_points = None


class EgoMotionCompensator:
    """
    Compensates for camera ego-motion using VIO data from T265.
    Uses homography estimation when VIO is not available.
    """
    
    def __init__(self):
        self.last_pose: Optional[Dict] = None
        self.intrinsics: Optional[Dict[str, float]] = None
        self.last_frame: Optional[np.ndarray] = None
    
    def set_intrinsics(self, intrinsics: Dict[str, float]):
        """Set camera intrinsics."""
        self.intrinsics = intrinsics.copy()
    
    def compensate(self, 
                   flow_vectors: List[Dict],
                   pose_current: Optional[Dict],
                   frame: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Compensate flow vectors for ego-motion.
        
        Args:
            flow_vectors: Original flow vectors
            pose_current: Current pose from T265
            frame: Current frame (for homography estimation if no pose)
            
        Returns:
            compensated_flows: Flow vectors with ego-motion removed
        """
        if not flow_vectors:
            return []
        
        # Method 1: Use VIO pose data (preferred)
        if pose_current and self.last_pose and self.intrinsics:
            return self._compensate_with_vio(flow_vectors, pose_current)
        
        # Method 2: Estimate ego-motion from image (fallback)
        elif frame is not None and self.last_frame is not None:
            return self._compensate_with_homography(flow_vectors, frame)
        
        # No compensation possible
        return flow_vectors
    
    def _compensate_with_vio(self, 
                             flow_vectors: List[Dict],
                             pose_current: Dict) -> List[Dict]:
        """
        Compensate flow using VIO pose delta.
        Simplified implementation - full version would project 3D motion.
        """
        # Calculate pose delta
        t_last = self.last_pose['translation']
        t_curr = pose_current['translation']
        
        dx = t_curr['x'] - t_last['x']
        dy = t_curr['y'] - t_last['y']
        dz = t_curr['z'] - t_last['z']
        
        # Simplified: estimate ego-flow magnitude from translation
        # In practice, this would require depth information
        ego_magnitude = np.sqrt(dx**2 + dy**2 + dz**2) * 100  # Scale factor
        
        # Subtract ego component from each flow vector
        compensated = []
        for flow in flow_vectors:
            residual_mag = max(0, flow['magnitude'] - ego_magnitude)
            compensated.append({
                'start': flow['start'],
                'end': flow['end'],
                'magnitude': residual_mag,
                'original_magnitude': flow['magnitude']
            })
        
        self.last_pose = pose_current
        return compensated
    
    def _compensate_with_homography(self,
                                    flow_vectors: List[Dict],
                                    frame: np.ndarray) -> List[Dict]:
        """
        Estimate and remove global motion using homography.
        Assumes most of the scene is static background.
        """
        if self.last_frame is None:
            self.last_frame = frame.copy()
            return flow_vectors
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY) \
            if len(self.last_frame.shape) == 3 else self.last_frame
        
        # Find matches between frames
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(last_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            self.last_frame = frame.copy()
            return flow_vectors
        
        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 4:
            self.last_frame = frame.copy()
            return flow_vectors
        
        # Get matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        
        if H is None:
            self.last_frame = frame.copy()
            return flow_vectors
        
        # Estimate global motion magnitude from homography
        # Translation component
        tx = H[0, 2]
        ty = H[1, 2]
        global_motion = np.sqrt(tx**2 + ty**2)
        
        # Subtract global motion from flow vectors
        compensated = []
        for flow in flow_vectors:
            residual_mag = max(0, flow['magnitude'] - global_motion)
            compensated.append({
                'start': flow['start'],
                'end': flow['end'],
                'magnitude': residual_mag,
                'original_magnitude': flow['magnitude']
            })
        
        self.last_frame = frame.copy()
        return compensated
    
    def update_frame(self, frame: np.ndarray):
        """Update reference frame for homography estimation."""
        self.last_frame = frame.copy()
    
    def reset(self):
        """Reset compensator state."""
        self.last_pose = None
        self.last_frame = None
