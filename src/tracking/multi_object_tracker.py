"""
Multi-Object Tracker Module
Implements tracking using Kalman Filter and IoU matching.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
from filterpy.kalman import KalmanFilter


class Track:
    """
    Represents a single tracked object.
    """
    
    def __init__(self, 
                 track_id: int,
                 bbox: List[float],
                 detection: Dict):
        self.track_id = track_id
        self.age = 1
        self.time_since_update = 0
        self.state = bbox  # [x1, y1, x2, y2]
        
        # Kalman Filter for smooth prediction
        self.kf = self._init_kalman_filter(bbox)
        
        # History for visualization
        self.history: deque = deque(maxlen=30)
        self.history.append(bbox)
        
        # Store detection info
        self.last_detection = detection
    
    def _init_kalman_filter(self, bbox: List[float]) -> KalmanFilter:
        """Initialize Kalman Filter for bounding box tracking."""
        kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State: [x, y, w, h, vx, vy, vw]
        # Measurement: [x, y, w, h]
        
        # State transition matrix (constant velocity model)
        dt = 1.0  # Time step
        kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise
        kf.R[2:, 2:] *= 10.0
        
        # Measurement noise
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 100.0
        
        # Initial state
        x, y, w, h = self._bbox_to_xywh(bbox)
        kf.x = np.array([x, y, w, h, 0.0, 0.0, 0.0]).reshape(-1, 1)
        
        return kf
    
    def _bbox_to_xywh(self, bbox: List[float]) -> Tuple[float, float, float, float]:
        """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
        x1, y1, x2, y2 = bbox
        return x1, y1, x2 - x1, y2 - y1
    
    def _xywh_to_bbox(self, xywh: Tuple[float, float, float, float]) -> List[float]:
        """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
        x, y, w, h = xywh
        return [x, y, x + w, y + h]
    
    def predict(self) -> List[float]:
        """Predict next state using Kalman Filter."""
        self.kf.predict()
        self.time_since_update += 1
        
        # Get predicted state
        pred_state = self.kf.x[:4].flatten()
        pred_bbox = self._xywh_to_bbox(pred_state)
        
        # Ensure valid coordinates
        pred_bbox = [max(0, c) for c in pred_bbox]
        self.state = pred_bbox
        
        return pred_bbox
    
    def update(self, bbox: List[float], detection: Dict):
        """Update track with new detection."""
        self.time_since_update = 0
        self.age += 1
        self.state = bbox
        self.last_detection = detection
        
        # Update Kalman Filter
        x, y, w, h = self._bbox_to_xywh(bbox)
        self.kf.update([x, y, w, h])
        
        # Add to history
        self.history.append(bbox)
    
    def is_confirmed(self) -> bool:
        """Check if track is confirmed (survived long enough)."""
        return self.age >= 3
    
    def is_lost(self, max_age: int = 30) -> bool:
        """Check if track should be removed."""
        return self.time_since_update > max_age


class MultiObjectTracker:
    """
    Simple multi-object tracker using IoU matching and Kalman Filters.
    Similar to ByteTrack but simplified.
    """
    
    def __init__(self,
                 iou_threshold: float = 0.3,
                 conf_high: float = 0.5,
                 conf_low: float = 0.1,
                 max_age: int = 30,
                 min_hits: int = 3):
        """
        Args:
            iou_threshold: IoU threshold for matching
            conf_high: High confidence threshold
            conf_low: Low confidence threshold
            max_age: Maximum frames without detection before removing track
            min_hits: Minimum hits to confirm a track
        """
        self.iou_threshold = iou_threshold
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracks: List[Track] = []
        self.next_track_id = 0
    
    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries with 'bbox' and 'confidence'
            
        Returns:
            active_tracks: List of active tracks
        """
        # Separate high and low confidence detections
        high_conf_dets = [d for d in detections if d.get('confidence', 0) >= self.conf_high]
        low_conf_dets = [d for d in detections if self.conf_low <= d.get('confidence', 0) < self.conf_high]
        
        # Predict states for all existing tracks
        predicted_tracks = []
        for track in self.tracks:
            pred_bbox = track.predict()
            predicted_tracks.append((track, pred_bbox))
        
        # Match high confidence detections
        matched_indices = self._match_detections(predicted_tracks, high_conf_dets)
        
        # Update matched tracks
        used_detections = set()
        for track_idx, det_idx in matched_indices:
            track = self.tracks[track_idx]
            det = high_conf_dets[det_idx]
            track.update(det['bbox'], det)
            used_detections.add(det_idx)
        
        # Try to match unmatched tracks with low confidence detections
        unmatched_tracks = [i for i in range(len(self.tracks)) 
                          if i not in [m[0] for m in matched_indices]]
        
        if unmatched_tracks and low_conf_dets:
            unmatched_pred = [predicted_tracks[i] for i in unmatched_tracks]
            low_matched = self._match_detections(unmatched_pred, low_conf_dets)
            
            for track_idx_local, det_idx in low_matched:
                track_idx = unmatched_tracks[track_idx_local]
                track = self.tracks[track_idx]
                det = low_conf_dets[det_idx]
                track.update(det['bbox'], det)
                used_detections.add(det_idx)
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(high_conf_dets):
            if i not in used_detections:
                new_track = Track(self.next_track_id, det['bbox'], det)
                self.tracks.append(new_track)
                self.next_track_id += 1
        
        # Remove lost tracks
        self.tracks = [t for t in self.tracks if not t.is_lost(self.max_age)]
        
        # Return only confirmed tracks
        return [t for t in self.tracks if t.is_confirmed()]
    
    def _match_detections(self, 
                         tracks_with_pred: List[Tuple[Track, List[float]]],
                         detections: List[Dict]) -> List[Tuple[int, int]]:
        """
        Match tracks to detections using IoU.
        
        Returns:
            matches: List of (track_idx, det_idx) tuples
        """
        if not tracks_with_pred or not detections:
            return []
        
        # Build IoU cost matrix
        n_tracks = len(tracks_with_pred)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets))
        
        for i, (_, pred_bbox) in enumerate(tracks_with_pred):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(pred_bbox, det['bbox'])
        
        # Greedy matching
        matches = []
        used_tracks = set()
        used_dets = set()
        
        # Sort by IoU (highest first)
        indices = np.argsort(-iou_matrix.flatten())
        
        for idx in indices:
            track_idx = idx // n_dets
            det_idx = idx % n_dets
            
            if track_idx in used_tracks or det_idx in used_dets:
                continue
            
            if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                matches.append((track_idx, det_idx))
                used_tracks.add(track_idx)
                used_dets.add(det_idx)
        
        return matches
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h
        
        # Areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Union
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_track_id = 0
