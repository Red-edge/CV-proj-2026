"""
ROI Generator Module
Generates dynamic Regions of Interest from motion data.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque


class ROIGenerator:
    """
    Generates ROI boxes from optical flow vectors.
    Includes clustering, filtering, and temporal smoothing.
    """
    
    def __init__(self,
                 grid_size: int = 30,
                 min_points_per_cell: int = 2,
                 magnitude_threshold: float = 2.0,
                 iou_threshold: float = 0.3,
                 temporal_window: int = 5):
        """
        Args:
            grid_size: Size of grid cells for clustering (pixels)
            min_points_per_cell: Minimum flow vectors to consider a cell active
            magnitude_threshold: Minimum flow magnitude to consider as motion
            iou_threshold: IoU threshold for merging overlapping boxes
            temporal_window: Number of frames for temporal smoothing
        """
        self.grid_size = grid_size
        self.min_points_per_cell = min_points_per_cell
        self.magnitude_threshold = magnitude_threshold
        self.iou_threshold = iou_threshold
        
        # Temporal smoothing
        self.temporal_window = temporal_window
        self.roi_history: deque = deque(maxlen=temporal_window)
        
        # State
        self.last_rois: List[Tuple[int, int, int, int]] = []
    
    def generate(self, 
                 flow_vectors: List[Dict],
                 frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Generate ROI boxes from flow vectors.
        
        Args:
            flow_vectors: List of flow vector dictionaries with 'start', 'end', 'magnitude'
            frame_shape: Shape of the input frame (H, W) or (H, W, C)
            
        Returns:
            roi_boxes: List of (x, y, w, h) bounding boxes
        """
        if not flow_vectors:
            self.last_rois = []
            return []
        
        h, w = frame_shape[:2]
        
        # Filter by magnitude
        significant_flows = [
            v for v in flow_vectors 
            if v.get('magnitude', 0) > self.magnitude_threshold
        ]
        
        if not significant_flows:
            self.last_rois = []
            return []
        
        # Grid-based clustering
        grid = {}
        for flow in significant_flows:
            x, y = flow['start']
            gx, gy = x // self.grid_size, y // self.grid_size
            key = (gx, gy)
            
            if key not in grid:
                grid[key] = []
            grid[key].append(flow)
        
        # Generate boxes from clusters
        roi_boxes = []
        for (gx, gy), flows in grid.items():
            if len(flows) >= self.min_points_per_cell:
                # Get bounding box of cluster
                x_min = min(f['start'][0] for f in flows)
                y_min = min(f['start'][1] for f in flows)
                x_max = max(f['end'][0] for f in flows)
                y_max = max(f['end'][1] for f in flows)
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                roi_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        # Merge overlapping boxes
        roi_boxes = self._merge_overlapping_boxes(roi_boxes)
        
        # Apply temporal smoothing
        roi_boxes = self._temporal_smooth(roi_boxes, frame_shape)
        
        self.last_rois = roi_boxes
        return roi_boxes
    
    def _merge_overlapping_boxes(self, 
                                  boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes using NMS-like approach."""
        if not boxes:
            return []
        
        # Sort by x coordinate
        boxes = sorted(boxes, key=lambda b: b[0])
        merged = [boxes[0]]
        
        for current in boxes[1:]:
            last = merged[-1]
            
            # Check overlap
            if (current[0] < last[0] + last[2] and 
                current[1] < last[1] + last[3]):
                # Merge
                x = min(last[0], current[0])
                y = min(last[1], current[1])
                w = max(last[0] + last[2], current[0] + current[2]) - x
                h = max(last[1] + last[3], current[1] + current[3]) - y
                merged[-1] = (x, y, w, h)
            else:
                merged.append(current)
        
        return merged
    
    def _temporal_smooth(self, 
                         current_rois: List[Tuple[int, int, int, int]],
                         frame_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Smooth ROI positions over time using exponential moving average.
        """
        self.roi_history.append(current_rois)
        
        if len(self.roi_history) < 2:
            return current_rois
        
        # For each ROI, find corresponding ROIs in history and average
        # Simple approach: average all ROIs (assumes consistent number)
        if len(self.roi_history[0]) == len(current_rois):
            smoothed = []
            for i in range(len(current_rois)):
                xs = [rois[i][0] for rois in self.roi_history if i < len(rois)]
                ys = [rois[i][1] for rois in self.roi_history if i < len(rois)]
                ws = [rois[i][2] for rois in self.roi_history if i < len(rois)]
                hs = [rois[i][3] for rois in self.roi_history if i < len(rois)]
                
                if xs and ys and ws and hs:
                    avg_x = int(np.mean(xs))
                    avg_y = int(np.mean(ys))
                    avg_w = int(np.mean(ws))
                    avg_h = int(np.mean(hs))
                    smoothed.append((avg_x, avg_y, avg_w, avg_h))
            
            return smoothed if smoothed else current_rois
        
        return current_rois
    
    def create_attention_mask(self, 
                              roi_boxes: List[Tuple[int, int, int, int]],
                              frame_shape: Tuple[int, int],
                              blur_radius: int = 15) -> np.ndarray:
        """
        Create an attention mask from ROI boxes.
        
        Args:
            roi_boxes: List of (x, y, w, h) boxes
            frame_shape: Shape of the frame
            blur_radius: Radius for Gaussian blur
            
        Returns:
            mask: Soft attention mask (H, W) with values in [0, 1]
        """
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Draw filled rectangles for each ROI
        for (x, y, bw, bh) in roi_boxes:
            cv2.rectangle(mask, (x, y), (x + bw, y + bh), 1.0, -1)
        
        # Apply Gaussian blur for soft edges
        if blur_radius > 0:
            kernel_size = blur_radius * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        # Normalize to [0, 1]
        max_val = mask.max()
        if max_val > 0:
            mask = mask / max_val
        
        return mask
    
    def reset(self):
        """Reset generator state."""
        self.roi_history.clear()
        self.last_rois = []


class SaliencyMapGenerator:
    """
    Generates motion saliency maps from optical flow.
    """
    
    def __init__(self, 
                 decay_rate: float = 0.9,
                 threshold_method: str = 'otsu'):
        """
        Args:
            decay_rate: Decay rate for temporal accumulation
            threshold_method: 'otsu' or 'fixed'
        """
        self.decay_rate = decay_rate
        self.threshold_method = threshold_method
        self.saliency_map: Optional[np.ndarray] = None
    
    def update(self, 
               flow_vectors: List[Dict],
               frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Update saliency map with new flow vectors.
        
        Args:
            flow_vectors: List of flow vector dictionaries
            frame_shape: Shape of the frame
            
        Returns:
            saliency_map: Current saliency map (H, W)
        """
        h, w = frame_shape[:2]
        
        # Initialize saliency map
        if self.saliency_map is None:
            self.saliency_map = np.zeros((h, w), dtype=np.float32)
        
        # Decay previous map
        self.saliency_map *= self.decay_rate
        
        # Add new motion contributions
        for flow in flow_vectors:
            x, y = flow['start']
            mag = flow.get('magnitude', 1.0)
            
            if 0 <= y < h and 0 <= x < w:
                # Add motion magnitude at this location
                self.saliency_map[y, x] = min(1.0, self.saliency_map[y, x] + mag / 10.0)
        
        # Apply Gaussian blur for spatial smoothing
        self.saliency_map = cv2.GaussianBlur(self.saliency_map, (15, 15), 0)
        
        # Normalize
        max_val = self.saliency_map.max()
        if max_val > 0:
            self.saliency_map = self.saliency_map / max_val
        
        return self.saliency_map.copy()
    
    def threshold(self, 
                  saliency_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Threshold the saliency map to get binary motion regions.
        
        Args:
            saliency_map: Input saliency map (uses current map if None)
            
        Returns:
            binary_map: Binary motion mask
        """
        if saliency_map is None:
            saliency_map = self.saliency_map
        
        if saliency_map is None:
            return np.zeros_like(saliency_map)
        
        if self.threshold_method == 'otsu':
            # Convert to uint8 for OpenCV
            map_uint8 = (saliency_map * 255).astype(np.uint8)
            _, binary = cv2.threshold(map_uint8, 0, 255, 
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return (binary > 0).astype(np.float32)
        else:
            # Fixed threshold
            return (saliency_map > 0.3).astype(np.float32)
    
    def reset(self):
        """Reset saliency map."""
        self.saliency_map = None
