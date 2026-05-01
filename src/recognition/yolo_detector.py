#!/usr/bin/env python3
"""
YOLODetector with Pruned/Quantized Model Support + ROI Crop Compatibility
Compatible with: .pt (FP32), .torchscript.pt, .mlpackage (CoreML)
Optimized for Apple Silicon MPS.

Fixed: Properly handles use_roi_crop and rois parameters without passing them to Ultralytics.
"""
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class YOLODetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: str = "mps",
    ):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.is_coreml = False

    def initialize(self) -> bool:
        try:
            path = Path(self.model_path)
            if not path.exists():
                print(f"📥 Downloading default model: yolo11n.pt")
                self.model_path = "yolo11n.pt"
                path = Path("yolo11n.pt")

            ext = path.suffix
            if ext == ".mlpackage":
                self.model = YOLO(str(path), task="detect")
                self.is_coreml = True
                print(f"✅ CoreML model loaded: {path.name}")
            elif ext in (".pt", ".torchscript.pt"):
                self.model = YOLO(str(path))
                if self.device == "mps" and torch.backends.mps.is_available():
                    print(f"✅ Torch model loaded: {path.name} on MPS (Auto precision)")
                else:
                    print(f"✅ Torch model loaded: {path.name} on {self.device}")
            else:
                raise ValueError(f"Unsupported format: {ext}")
            return True
        except Exception as e:
            print(f"❌ YOLO initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def detect(
        self,
        frame: np.ndarray,
        rois: Optional[List[Tuple[int, int, int, int]]] = None,
        use_roi_crop: bool = False,
        **kwargs
    ) -> List[Dict]:
        """
        Run detection on a single BGR frame.
        
        Args:
            frame: BGR image (H, W, 3)
            rois: List of (x, y, w, h) ROI boxes (optional)
            use_roi_crop: If True + rois provided, run YOLO on each ROI crop separately
            **kwargs: Other Ultralytics-compatible args (imgsz, conf, iou, etc.)
        
        Returns:
            List of {"bbox": [x1,y1,x2,y2], "confidence": float, "class_name": str}
        """
        if self.model is None:
            return []

        # 🔥 Handle ROI cropping logic BEFORE calling Ultralytics
        if use_roi_crop and rois:
            all_detections = []
            h, w = frame.shape[:2]
            
            for roi in rois:
                rx, ry, rw, rh = roi
                # Clip ROI to frame bounds
                rx = max(0, min(rx, w-1))
                ry = max(0, min(ry, h-1))
                rx2 = min(w, rx + rw)
                ry2 = min(h, ry + rh)
                
                if rx2 <= rx or ry2 <= ry:
                    continue
                    
                # Crop and run inference
                roi_frame = frame[ry:ry2, rx:rx2]
                if roi_frame.size == 0:
                    continue
                    
                # Run YOLO on cropped ROI
                results = self.model(roi_frame, verbose=False, device=self.device, **kwargs)
                
                # Map detections back to full-frame coordinates
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                        # Add ROI offset
                        x1 += rx
                        y1 += ry
                        x2 += rx
                        y2 += ry
                        
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id]
                        
                        all_detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                            "class_name": cls_name,
                        })
            return all_detections
        else:
            # Full-frame inference (original behavior)
            try:
                # Filter out Ultralytics-incompatible kwargs
                valid_kwargs = {k: v for k, v in kwargs.items() 
                               if k not in ['use_roi_crop', 'rois']}
                
                results = self.model(frame, verbose=False, device=self.device, **valid_kwargs)
                
                detections = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self.model.names[cls_id]

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                            "class_name": cls_name,
                        })
                return detections
            except Exception as e:
                print(f"⚠️ Detection failed: {e}")
                return []