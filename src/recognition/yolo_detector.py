"""
YOLO Detector Module
Handles person detection using YOLO models with ROI-based attention.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import time


class YOLODetector:
    """
    YOLO-based person detector with ROI support.
    Supports YOLOv8 and YOLOv10 from ultralytics.
    """
    
    def __init__(self,
                 model_path: str = 'yolov8n.pt',
                 device: str = 'cpu',
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 classes: List[int] = None):
        """
        Args:
            model_path: Path to YOLO model or model name
            device: 'cpu', 'mps' (Mac), or 'cuda'
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            classes: List of class IDs to detect (None = all, [0] = person)
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes if classes is not None else [0]  # Default: person only
        
        self.model = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Load the YOLO model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from ultralytics import YOLO
            
            # Load model
            self.model = YOLO(self.model_path)
            
            # Set device
            if self.device == 'mps' and hasattr(self.model, 'to'):
                try:
                    self.model.to('mps')
                    print(f"✓ YOLO initialized on MPS (Apple Silicon)")
                except Exception:
                    print("⚠ MPS not available, falling back to CPU")
                    self.device = 'cpu'
            else:
                print(f"✓ YOLO initialized on {self.device}")
            
            self.is_initialized = True
            return True
            
        except ImportError:
            print("⚠ ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            print(f"⚠ Failed to load YOLO model: {e}")
            return False
    
    def detect(self, 
               image: np.ndarray,
               roi_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
               use_roi_crop: bool = False) -> List[Dict]:
        """
        Detect persons in the image.
        
        Args:
            image: Input image (BGR format)
            roi_boxes: Optional list of (x, y, w, h) ROI boxes
            use_roi_crop: If True, crop to ROI regions before detection
            
        Returns:
            detections: List of detection dictionaries
        """
        if not self.is_initialized or self.model is None:
            return []
        
        start_time = time.time()
        
        if use_roi_crop and roi_boxes:
            # Detect in each ROI region separately
            all_detections = []
            
            for i, (x, y, w, h) in enumerate(roi_boxes):
                # Crop ROI with padding
                pad = 20
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(image.shape[1], x + w + pad)
                y2 = min(image.shape[0], y + h + pad)
                
                roi_image = image[y1:y2, x1:x2]
                
                if roi_image.size == 0:
                    continue
                
                # Run detection on ROI
                results = self.model(
                    roi_image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=self.classes,
                    verbose=False
                )
                
                # Convert detections to global coordinates
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        cls_ids = result.boxes.cls.cpu().numpy()
                        
                        for box, conf, cls_id in zip(boxes, confs, cls_ids):
                            # Transform coordinates to global
                            x1_local, y1_local, x2_local, y2_local = box
                            all_detections.append({
                                'bbox': [x1_local + x1, y1_local + y1, 
                                        x2_local + x1, y2_local + y1],
                                'confidence': float(conf),
                                'class_id': int(cls_id),
                                'class_name': result.names[int(cls_id)],
                                'roi_id': i
                            })
            
            elapsed = time.time() - start_time
            print(f"ROI detection: {len(all_detections)} detections in {elapsed*1000:.1f}ms")
            return all_detections
            
        else:
            # Full image detection
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                verbose=False
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    cls_ids = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls_id in zip(boxes, confs, cls_ids):
                        detections.append({
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'class_id': int(cls_id),
                            'class_name': result.names[int(cls_id)]
                        })
            
            elapsed = time.time() - start_time
            print(f"Full detection: {len(detections)} detections in {elapsed*1000:.1f}ms")
            return detections
    
    def detect_with_attention(self,
                              image: np.ndarray,
                              attention_mask: np.ndarray,
                              mask_threshold: float = 0.3) -> List[Dict]:
        """
        Detect persons using attention mask to weight predictions.
        
        Args:
            image: Input image
            attention_mask: Soft attention mask (H, W) with values in [0, 1]
            mask_threshold: Minimum attention value to consider
            
        Returns:
            detections: Filtered detections
        """
        # First run normal detection
        detections = self.detect(image)
        
        if len(detections) == 0 or attention_mask is None:
            return detections
        
        # Filter detections based on attention mask
        filtered = []
        h, w = attention_mask.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Get attention score in detection region
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            roi_mask = attention_mask[y1:y2, x1:x2]
            avg_attention = np.mean(roi_mask)
            
            # Keep if attention is above threshold
            if avg_attention >= mask_threshold:
                det['attention_score'] = float(avg_attention)
                # Boost confidence based on attention
                det['confidence'] = det['confidence'] * (0.5 + 0.5 * avg_attention)
                filtered.append(det)
        
        return filtered
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_initialized:
            return {}
        
        return {
            'model_path': self.model_path,
            'device': self.device,
            'classes': self.classes,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold
        }
