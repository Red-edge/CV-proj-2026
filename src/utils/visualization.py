# Utils Module
# Utility functions for the RealSense T265 + YOLO Motion Detection System

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


def draw_detections(image: np.ndarray, 
                    detections: List[Dict],
                    color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 2) -> np.ndarray:
    """
    Draw detection boxes on image.
    
    Args:
        image: Input image
        detections: List of detection dicts with 'bbox' and 'confidence'
        color: BGR color for boxes
        thickness: Line thickness
        
    Returns:
        image_with_detections: Image with drawn boxes
    """
    result = image.copy()
    
    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) != 4:
            continue
            
        x1, y1, x2, y2 = map(int, bbox)
        conf = det.get('confidence', 0)
        label = det.get('class_name', 'person')
        
        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label_text = f"{label}: {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(result, 
                     (x1, y1 - text_h - baseline - 5),
                     (x1 + text_w, y1),
                     color, -1)
        cv2.putText(result, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result


def draw_tracks(image: np.ndarray,
                tracks: List,
                show_trajectory: bool = True) -> np.ndarray:
    """
    Draw tracked objects with IDs and trajectories.
    
    Args:
        image: Input image
        tracks: List of Track objects
        show_trajectory: Whether to draw trajectory lines
        
    Returns:
        image_with_tracks: Image with drawn tracks
    """
    result = image.copy()
    
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.state)
        track_id = track.track_id
        
        # Draw box (blue for tracks)
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw ID
        id_text = f"ID:{track_id}"
        (text_w, text_h), baseline = cv2.getTextSize(
            id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(result,
                     (x1, y1 - text_h - baseline - 5),
                     (x1 + text_w, y1),
                     (255, 0, 0), -1)
        cv2.putText(result, id_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw trajectory
        if show_trajectory and len(track.history) > 1:
            for j in range(1, len(track.history)):
                prev = track.history[j-1]
                curr = track.history[j]
                pt1 = (int((prev[0] + prev[2]) / 2), int((prev[1] + prev[3]) / 2))
                pt2 = (int((curr[0] + curr[2]) / 2), int((curr[1] + curr[3]) / 2))
                alpha = j / len(track.history)
                color_traj = (int(255 * alpha), 0, 0)
                cv2.line(result, pt1, pt2, color_traj, 2)
    
    return result


def draw_roi_boxes(image: np.ndarray,
                   roi_boxes: List[Tuple[int, int, int, int]],
                   color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw ROI boxes on image.
    
    Args:
        image: Input image
        roi_boxes: List of (x, y, w, h) tuples
        color: BGR color for boxes
        
    Returns:
        image_with_rois: Image with drawn ROI boxes
    """
    result = image.copy()
    
    for i, (x, y, w, h) in enumerate(roi_boxes):
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result, f'ROI {i}', (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result


def draw_stats(image: np.ndarray,
               fps: float,
               num_rois: int,
               num_tracks: int,
               num_detections: int) -> np.ndarray:
    """
    Draw statistics overlay on image.
    
    Args:
        image: Input image
        fps: Current FPS
        num_rois: Number of ROI boxes
        num_tracks: Number of active tracks
        num_detections: Number of detections
        
    Returns:
        image_with_stats: Image with statistics overlay
    """
    result = image.copy()
    
    stats = [
        f'FPS: {fps:.1f}',
        f'ROIs: {num_rois}',
        f'Tracks: {num_tracks}',
        f'Dets: {num_detections}'
    ]
    
    y_offset = 30
    for stat in stats:
        cv2.putText(result, stat, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        y_offset += 40
    
    return result


def create_composite_view(flow_vis: np.ndarray,
                          saliency_map: Optional[np.ndarray],
                          main_view: np.ndarray) -> np.ndarray:
    """
    Create a composite view with flow visualization and saliency map.
    
    Args:
        flow_vis: Flow visualization image
        saliency_map: Saliency map (H, W) or None
        main_view: Main output view
        
    Returns:
        composite: Combined view
    """
    # Resize all to same height
    target_height = 240
    
    def resize_keep_aspect(img, height):
        h, w = img.shape[:2]
        scale = height / h
        new_w = int(w * scale)
        return cv2.resize(img, (new_w, height))
    
    flow_resized = resize_keep_aspect(flow_vis, target_height)
    main_resized = resize_keep_aspect(main_view, target_height)
    
    panels = [flow_resized, main_resized]
    
    if saliency_map is not None:
        # Convert saliency to colormap
        saliency_uint8 = (saliency_map * 255).astype(np.uint8)
        saliency_color = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
        saliency_resized = resize_keep_aspect(saliency_color, target_height)
        panels.append(saliency_resized)
    
    # Concatenate horizontally
    composite = np.hstack(panels)
    return composite
