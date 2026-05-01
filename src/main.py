#!/usr/bin/env python3
"""
RealSense/Webcam + YOLO Motion-Guided Detection System
Main entry point with 64-Point Fixed-Grid Optical Flow as ROI Prior, and Gimbal Tracking.
Pipeline:
1. Resize input frame to max 720P (optional).
2. Compute motion intent at 64 FIXED spatial points (8x8 uniform grid):
   - Use LK optical flow magnitude OR frame difference
   - Threshold: magnitude > thresh → "motion intent detected"
3. ROI Generation: Find 1/3-width × full-height vertical strip
   that covers MAXIMUM number of motion-intent points.
4. ATTENTION (Option A): Blur background, keep ROI sharp.
5. YOLO full-frame inference on attention-modified frame.
6. DUAL-THRESHOLD (Option B): Filter detections:
   - Inside ROI: Lower confidence threshold (e.g., 0.25)
   - Outside ROI: Higher confidence threshold (e.g., 0.45)
7. Global NMS + Tracking + Visualization.
8. Gimbal Tracking: Align camera center with largest YOLO target bbox center.
Final output:
- Motion Points: 64 dots (green=static, yellow=micro, red=motion)
- ROI Box: magenta thick box (1/3 width, full height, dynamically positioned)
- YOLO Box: orange/yellow boxes (BG/ROI labeled)
- Track Box: blue boxes
- Tracking Crosshair: green cross on largest target center
- Panels: FPS, motion stats, timing, target coordinates
Coordinate definition:
- Image center is origin for target panel
- x axis: right is positive, y axis: up is positive
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from data_source.realsense_wrapper import RealSenseT265, VideoFallback
from recognition.yolo_detector import YOLODetector
from tracking.multi_object_tracker import MultiObjectTracker
from camera_mapper import CameraAngleMapper
from gimbal_controller import GimbalController


class MotionDetectionPipeline:
    """
    Complete pipeline for motion-guided detection using 64-point fixed-grid optical flow.
    """

    def __init__(
        self,
        use_yolo: bool = False,
        device: str = "cpu",
        reset_warmup_frames: int = 3,
        min_yolo_box_width: int = 30,
        min_yolo_box_height: int = 50,
        min_yolo_box_area: int = 1500,
        yolo_nms_iou: float = 0.45,
        motion_thresh: float = 2.0,
        blur_ksize: int = 21,
        roi_conf_thresh: float = 0.25,
        bg_conf_thresh: float = 0.45,
        motion_method: str = "lk",  # "lk" or "diff"
        num_motion_points: int = 64,
    ):
        self.use_yolo = use_yolo
        self.device = device
        self.reset_warmup_frames = max(0, int(reset_warmup_frames))
        self.warmup_frames_remaining = 0

        self.min_yolo_box_width = int(min_yolo_box_width)
        self.min_yolo_box_height = int(min_yolo_box_height)
        self.min_yolo_box_area = int(min_yolo_box_area)
        self.yolo_nms_iou = float(yolo_nms_iou)
        
        self.motion_thresh = float(motion_thresh)
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.roi_conf_thresh = float(roi_conf_thresh)
        self.bg_conf_thresh = float(bg_conf_thresh)
        self.motion_method = motion_method
        self.num_motion_points = num_motion_points
        
        # ROI: 1/3 width, full height
        self.roi_w_ratio = 1.0 / 3.0
        self.roi_h_ratio = 1.0

        self.yolo_detector = YOLODetector(device=device) if use_yolo else None
        self.tracker = MultiObjectTracker() if use_yolo else None

        self.frame_count = 0
        self.prev_gray = None
        self.fixed_motion_pts = None  # Will be initialized on first frame
        self.start_time = time.time()
        self.display_fps = 0.0

        self.last_tracks: List = []
        self.last_detections: List[Dict] = []
        self.last_motion_count = 0

        self.last_motion_ms = 0.0
        self.last_yolo_ms = 0.0
        self.last_total_ms = 0.0
        self.last_motion_pct = 0.0
        self.last_yolo_pct = 0.0

    def _generate_fixed_motion_points(self, frame_w: int, frame_h: int) -> np.ndarray:
        """Generate num_motion_points fixed coordinates uniformly distributed in image space."""
        cols = int(np.round(np.sqrt(self.num_motion_points * frame_w / frame_h)))
        cols = max(4, min(cols, self.num_motion_points))
        rows = int(np.ceil(self.num_motion_points / cols))
        cell_w = frame_w / cols
        cell_h = frame_h / rows
        
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) >= self.num_motion_points:
                    break
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                pts.append([cx, cy])
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def _compute_motion_intent(self, prev_gray: np.ndarray, curr_gray: np.ndarray, 
                                fixed_pts: np.ndarray) -> np.ndarray:
        """Compute motion magnitude at fixed spatial points."""
        if self.motion_method == "diff":
            diff = cv2.absdiff(curr_gray, prev_gray)
            magnitudes = []
            h, w = curr_gray.shape
            for pt in fixed_pts.reshape(-1, 2):
                x, y = int(np.clip(pt[0], 1, w-2)), int(np.clip(pt[1], 1, h-2))
                patch = diff[y-1:y+2, x-1:x+2]
                magnitudes.append(float(np.max(patch)))
            return np.array(magnitudes, dtype=np.float32)
        else:  # "lk"
            lk_params = dict(winSize=(15, 15), maxLevel=2, 
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, fixed_pts, None, **lk_params)
            if next_pts is None or st is None:
                return np.zeros(len(fixed_pts), dtype=np.float32)
            dx = next_pts[:, 0, 0] - fixed_pts[:, 0, 0]
            dy = next_pts[:, 0, 1] - fixed_pts[:, 0, 1]
            return np.sqrt(dx**2 + dy**2) * st.ravel()

    def _get_roi_from_motion_points(self, motion_mask: np.ndarray, fixed_pts: np.ndarray, 
                                     frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Find 1/3-width × full-height vertical strip that covers MAXIMUM motion-intent points.
        Returns: (rx, ry, rw, rh) or None if no motion detected.
        """
        if not np.any(motion_mask):
            return None
        
        roi_w = int(frame_w * self.roi_w_ratio)
        roi_h = frame_h  # Full height
        
        # Get x-coordinates of motion points
        motion_x = fixed_pts.reshape(-1, 2)[motion_mask, 0]
        if len(motion_x) == 0:
            return None
        
        # Brute-force search: slide ROI window across x-axis
        # Discretize search to 1-pixel steps (64 points → trivial compute)
        best_rx, best_count = 0, 0
        for rx in range(0, frame_w - roi_w + 1, 2):  # Step=2 for speed
            count = np.sum((motion_x >= rx) & (motion_x < rx + roi_w))
            if count > best_count:
                best_count = count
                best_rx = rx
        
        # Fallback: center ROI if no clear winner
        if best_count == 0:
            best_rx = (frame_w - roi_w) // 2
        
        return (best_rx, 0, roi_w, roi_h)

    def process_frame(self, frame: np.ndarray) -> Dict:
        frame_start_time = time.perf_counter()

        results = {
            "frame": frame.copy(),
            "raw_frame": frame.copy(),
            "roi_box": None,
            "motion_pts": None,
            "motion_magnitudes": None,
            "detections": self.last_detections,
            "tracks": self.last_tracks,
            "fps": self.display_fps,
            "person_count": len(self.last_detections) if self.use_yolo else 0,
            "motion_count": self.last_motion_count,
            "warmup": self.warmup_frames_remaining > 0,
            "warmup_remaining": self.warmup_frames_remaining,
            "timing": {
                "motion_ms": self.last_motion_ms,
                "yolo_ms": self.last_yolo_ms,
                "total_ms": self.last_total_ms,
                "motion_pct": self.last_motion_pct,
                "yolo_pct": self.last_yolo_pct,
            },
        }

        # -------------------------
        # 1. Motion Intent Detection (64 Fixed Points)
        # -------------------------
        motion_stage_start = time.perf_counter()
        
        # Initialize fixed points on first frame
        if self.fixed_motion_pts is None:
            h, w = frame.shape[:2]
            self.fixed_motion_pts = self._generate_fixed_motion_points(w, h)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_magnitudes = np.zeros(len(self.fixed_motion_pts), dtype=np.float32)
        motion_mask = np.zeros(len(self.fixed_motion_pts), dtype=bool)
        
        if self.warmup_frames_remaining <= 0 and self.prev_gray is not None:
            motion_magnitudes = self._compute_motion_intent(self.prev_gray, gray, self.fixed_motion_pts)
            motion_mask = motion_magnitudes >= self.motion_thresh
        
        results["motion_pts"] = self.fixed_motion_pts.copy()
        results["motion_magnitudes"] = motion_magnitudes.copy()
        self.last_motion_count = int(np.sum(motion_mask))
        results["motion_count"] = self.last_motion_count

        # -------------------------
        # 2. ROI Generation from Motion Points
        # -------------------------
        roi_box = None
        if self.warmup_frames_remaining <= 0 and np.any(motion_mask):
            h, w = frame.shape[:2]
            roi_box = self._get_roi_from_motion_points(motion_mask, self.fixed_motion_pts, w, h)
            results["roi_box"] = roi_box

        motion_stage_end = time.perf_counter()
        self.prev_gray = gray.copy()

        # -------------------------
        # 3. Attention + YOLO + NMS + Tracking
        # -------------------------
        yolo_stage_start = time.perf_counter()

        if self.warmup_frames_remaining <= 0 and self.use_yolo and self.yolo_detector:
            # Option A: Apply attention blur
            att_frame = self._apply_attention_blur(frame, roi_box)
            
            # Run YOLO
            detections = self.yolo_detector.detect(att_frame)
            
            # Option B: Dual-threshold filtering based on ROI overlap
            detections = self._filter_with_attention(detections, frame.shape, roi_box)
            detections = self._nms_detections(detections, iou_threshold=self.yolo_nms_iou)

            self.last_detections = detections
            results["detections"] = detections

            if self.tracker:
                tracks = self.tracker.update(detections)
                self.last_tracks = tracks
                results["tracks"] = tracks
        else:
            results["detections"] = []
            results["tracks"] = []

        yolo_stage_end = time.perf_counter()

        # Update state
        self.frame_count += 1
        frame_end_time = time.perf_counter()

        # Timing & FPS
        motion_ms = (motion_stage_end - motion_stage_start) * 1000.0
        yolo_ms = (yolo_stage_end - yolo_stage_start) * 1000.0
        total_ms = (frame_end_time - frame_start_time) * 1000.0

        motion_pct = 100.0 * motion_ms / total_ms if total_ms > 0 else 0.0
        yolo_pct = 100.0 * yolo_ms / total_ms if total_ms > 0 else 0.0

        self.last_motion_ms = motion_ms
        self.last_yolo_ms = yolo_ms
        self.last_total_ms = total_ms
        self.last_motion_pct = motion_pct
        self.last_yolo_pct = yolo_pct

        results["timing"] = {
            "motion_ms": self.last_motion_ms,
            "yolo_ms": self.last_yolo_ms,
            "total_ms": self.last_total_ms,
            "motion_pct": self.last_motion_pct,
            "yolo_pct": self.last_yolo_pct,
        }

        if total_ms > 0:
            instant_fps = 1000.0 / total_ms
            self.display_fps = (0.9 * self.display_fps + 0.1 * instant_fps) if self.display_fps > 0 else instant_fps

        elapsed = time.time() - self.start_time
        results["fps"] = self.display_fps if self.display_fps > 0 else (self.frame_count / elapsed if elapsed > 0 else 0.0)
        results["person_count"] = len(results.get("detections", [])) if self.use_yolo else 0

        self._draw_results(results)
        return results

    def _apply_attention_blur(self, frame, roi_box):
        """Option A: Blur background, keep ROI sharp."""
        if roi_box is None:
            return frame.copy()
        rx, ry, rw, rh = roi_box
        h, w = frame.shape[:2]
        blurred = cv2.GaussianBlur(frame, (self.blur_ksize, self.blur_ksize), 0)
        result = blurred.copy()
        result[ry:ry+rh, rx:rx+rw] = frame[ry:ry+rh, rx:rx+rw]
        return result

    def _filter_with_attention(self, detections, frame_shape, roi_box):
        """Option B: Dual-threshold filtering based on ROI overlap."""
        if not detections: return []
        frame_h, frame_w = frame_shape[:2]
        filtered = []
        rx, ry, rw, rh = roi_box if roi_box else (-1, -1, 0, 0)
        
        in_roi = lambda x1, y1, x2, y2: (
            max(x1, rx) < min(x2, rx + rw) and max(y1, ry) < min(y2, ry + rh)
        )
        
        for det in detections:
            if not isinstance(det, dict): continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4: continue
            try:
                x1, y1, x2, y2 = map(float, bbox)
            except: continue
            
            x1 = float(np.clip(x1, 0, frame_w - 1))
            y1 = float(np.clip(y1, 0, frame_h - 1))
            x2 = float(np.clip(x2, 0, frame_w - 1))
            y2 = float(np.clip(y2, 0, frame_h - 1))
            if x2 <= x1 or y2 <= y1: continue
            
            bw, bh = x2 - x1, y2 - y1
            if bw < self.min_yolo_box_width or bh < self.min_yolo_box_height or bw*bh < self.min_yolo_box_area: continue
            
            conf = float(det.get("confidence", 0.0))
            is_in_roi = in_roi(x1, y1, x2, y2)
            
            if is_in_roi and conf < self.roi_conf_thresh: continue
            if not is_in_roi and conf < self.bg_conf_thresh: continue
            
            new_det = det.copy()
            new_det["bbox"] = [x1, y1, x2, y2]
            new_det["confidence"] = conf
            new_det["attention"] = "ROI" if is_in_roi else "BG"
            filtered.append(new_det)
        return filtered

    def _nms_detections(self, detections, iou_threshold=0.45):
        if not detections: return []
        boxes = [det["bbox"] for det in detections if det.get("bbox") and len(det["bbox"]) == 4]
        scores = [float(det.get("confidence", 0.0)) for det in detections if det.get("bbox") and len(det["bbox"]) == 4]
        if not boxes: return []
        
        boxes_np = np.array(boxes, dtype=float)
        order = np.argsort(scores)[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1: break
            rest = order[1:]
            x1 = np.maximum(boxes_np[i, 0], boxes_np[rest, 0])
            y1 = np.maximum(boxes_np[i, 1], boxes_np[rest, 1])
            x2 = np.minimum(boxes_np[i, 2], boxes_np[rest, 2])
            y2 = np.minimum(boxes_np[i, 3], boxes_np[rest, 3])
            w = np.maximum(0.0, x2 - x1)
            h = np.maximum(0.0, y2 - y1)
            inter = w * h
            area_i = (boxes_np[i, 2] - boxes_np[i, 0]) * (boxes_np[i, 3] - boxes_np[i, 1])
            area_r = (boxes_np[rest, 2] - boxes_np[rest, 0]) * (boxes_np[rest, 3] - boxes_np[rest, 1])
            ious = inter / (area_i + area_r - inter + 1e-6)
            order = rest[ious <= iou_threshold]
            
        kept = [detections[i] for i in keep]
        kept.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
        return kept

    def _draw_results(self, results: Dict):
        vis_frame = results["raw_frame"].copy()
        
        # Draw motion points
        if results["motion_pts"] is not None and results["motion_magnitudes"] is not None:
            self._draw_motion_points(vis_frame, results["motion_pts"], 
                                     results["motion_magnitudes"], self.motion_thresh)
        
        # Draw ROI box
        if results["roi_box"]:
            self._draw_roi_boxes(vis_frame, [results["roi_box"]], color=(255, 0, 255), thickness=3)
        
        self._draw_yolo_detections(vis_frame, results.get("detections", []))
        self._draw_tracks(vis_frame, results.get("tracks", []))
        self._draw_stats_panel(vis_frame, results)
        self._draw_target_coordinate_panel(vis_frame, results)
        self._draw_timing_panel(vis_frame, results)
        results["frame"] = vis_frame

    def _draw_motion_points(self, vis_frame, fixed_pts, magnitudes, threshold):
        """Draw 64 fixed points colored by motion intent."""
        for i, pt in enumerate(fixed_pts.reshape(-1, 2)):
            x, y = int(pt[0]), int(pt[1])
            mag = magnitudes[i]
            if mag < threshold * 0.5:
                color = (0, 255, 0)      # Green: static
            elif mag < threshold:
                color = (0, 255, 255)    # Yellow: micro motion
            else:
                color = (0, 0, 255)      # Red: motion
            cv2.circle(vis_frame, (x, y), 3, color, -1)

    def _draw_roi_boxes(self, vis_frame, roi_boxes, color=(0, 255, 0), thickness=2):
        for i, box in enumerate(roi_boxes):
            if box is None or len(box) != 4: continue
            x, y, w, h = map(int, box)
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(vis_frame, f"ROI", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def _draw_yolo_detections(self, vis_frame, detections):
        h, w = vis_frame.shape[:2]
        for det in detections:
            if not isinstance(det, dict): continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4: continue
            try:
                x1, y1, x2, y2 = map(int, bbox)
            except: continue
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
            if x2 <= x1 or y2 <= y1: continue
            
            conf = float(det.get("confidence", 0.0))
            cls = str(det.get("class_name", "person"))
            att = det.get("attention", "BG")
            clr = (0, 165, 255) if att == "BG" else (0, 255, 255)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), clr, 2)
            label = f"{cls} {conf:.2f} [{att}]"
            ly = y2 + 20 if y2 + 20 < h else max(20, y1 - 8)
            cv2.putText(vis_frame, label, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2, cv2.LINE_AA)

    def _draw_tracks(self, vis_frame, tracks):
        h, w = vis_frame.shape[:2]
        for track in tracks:
            if not hasattr(track, "state"): continue
            try:
                x1, y1, x2, y2 = map(int, track.state)
            except: continue
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
            if x2 <= x1 or y2 <= y1: continue
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            tid = getattr(track, "track_id", -1)
            cv2.putText(vis_frame, f"Track ID:{tid}", (x1, max(20, y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            
            history = getattr(track, "history", [])
            if len(history) > 1:
                for j in range(1, len(history)):
                    p1 = (int((history[j-1][0]+history[j-1][2])/2), int((history[j-1][1]+history[j-1][3])/2))
                    p2 = (int((history[j][0]+history[j][2])/2), int((history[j][1]+history[j][3])/2))
                    cv2.line(vis_frame, p1, p2, (255, 0, 0), 2)

    def _bbox_center_relative_to_image_center(self, bbox, fw, fh):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
        return cx, cy, cx - fw/2.0, fh/2.0 - cy

    def _collect_target_centers(self, vis_frame, results):
        fh, fw = vis_frame.shape[:2]
        infos = []
        dets = results.get("detections", [])
        tracks = results.get("tracks", [])
        if dets:
            for i, d in enumerate(dets):
                if not isinstance(d, dict): continue
                bbox = d.get("bbox")
                if not bbox or len(bbox) != 4: continue
                try:
                    cx, cy, rx, ry = self._bbox_center_relative_to_image_center(bbox, fw, fh)
                except: continue
                infos.append({"source":"YOLO","id":i,"cx":cx,"cy":cy,"rel_x":rx,"rel_y":ry,"conf":d.get("confidence",0.0)})
        elif tracks:
            for i, t in enumerate(tracks):
                if not hasattr(t, "state"): continue
                bbox = t.state
                if not bbox or len(bbox) != 4: continue
                try:
                    cx, cy, rx, ry = self._bbox_center_relative_to_image_center(bbox, fw, fh)
                except: continue
                infos.append({"source":"TRACK","id":getattr(t,"track_id",i),"cx":cx,"cy":cy,"rel_x":rx,"rel_y":ry,"conf":None})
        return infos

    def _draw_target_coordinate_panel(self, vis_frame, results):
        infos = self._collect_target_centers(vis_frame, results)
        fh, fw = vis_frame.shape[:2]
        lines = ["Target Centers", "Origin: image center", "x:+right y:+up"]
        if not infos:
            lines.append("No target")
        else:
            for info in infos[:8]:
                conf_str = f" {info['conf']:.2f}" if info["conf"] is not None else ""
                lines.append(f"{info['source']} {info['id']}: ({info['rel_x']:+.0f}, {info['rel_y']:+.0f}){conf_str}")
        
        font, fs, th, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2, 24
        pw = max(260, max(cv2.getTextSize(l, font, fs, th)[0][0] for l in lines) + 24)
        ph = 14 + lh * len(lines)
        px, py = max(8, fw - pw - 8), 8
        
        ov = vis_frame.copy()
        cv2.rectangle(ov, (px, py), (px+pw, py+ph), (0,0,0), -1)
        cv2.addWeighted(ov, 0.58, vis_frame, 0.42, 0, vis_frame)
        
        y = py + 27
        for i, l in enumerate(lines):
            clr = (0,255,255) if i==0 else (180,180,180) if l=="No target" else (0,165,255) if l.startswith("YOLO") else (255,0,0)
            cv2.putText(vis_frame, l, (px+12, y), font, fs, clr, th, cv2.LINE_AA)
            y += lh
        cv2.drawMarker(vis_frame, (fw//2, fh//2), (0,255,255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

    def _draw_stats_panel(self, vis_frame, results):
        n_roi = 1 if results["roi_box"] else 0
        n_dets = len(results.get("detections", []))
        n_tracks = len(results.get("tracks", []))
        motion_cnt = results.get("motion_count", 0)
        warmup_txt = f"WARMUP: {results['warmup_remaining']}" if results["warmup"] else "WARMUP: OFF"
        
        stats = [
            f"FPS: {results['fps']:.1f}",
            f"Motion Pts: {motion_cnt}/64 active",
            f"Attention: ROI={self.roi_conf_thresh:.2f} | BG={self.bg_conf_thresh:.2f}",
            f"Persons: {results['person_count']}",
            f"YOLO Dets: {n_dets}",
            f"Tracks: {n_tracks}",
            f"YOLO: {'ON' if self.use_yolo else 'OFF'}",
            warmup_txt
        ]
        font, fs, th, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2, 25
        px, py, pw, ph = 8, 8, 340, 14 + lh * len(stats)
        ov = vis_frame.copy()
        cv2.rectangle(ov, (px, py), (px+pw, py+ph), (0,0,0), -1)
        cv2.addWeighted(ov, 0.58, vis_frame, 0.42, 0, vis_frame)
        
        y = py + 27
        for s in stats:
            cv2.putText(vis_frame, s, (px+10, y), font, fs, (0,255,255), th, cv2.LINE_AA)
            y += lh
        self._draw_legend(vis_frame, px, py + ph + 18)

    def _draw_legend(self, vis_frame, x, y):
        items = [("Motion: Static", (0, 255, 0)), ("Motion: Micro", (0, 255, 255)), 
                 ("Motion: Active", (0, 0, 255)), ("ROI Box", (255, 0, 255)),
                 ("YOLO ROI", (0, 255, 255)), ("YOLO BG", (0, 165, 255)), ("Track Box", (255, 0, 0))]
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        pw, ph = 180, 24 * len(items) + 10
        ov = vis_frame.copy()
        cv2.rectangle(ov, (x, y-18), (x+pw, y-18+ph), (0,0,0), -1)
        cv2.addWeighted(ov, 0.45, vis_frame, 0.55, 0, vis_frame)
        cy = y
        for t, c in items:
            cv2.rectangle(vis_frame, (x+10, cy-10), (x+24, cy+4), c, -1)
            cv2.putText(vis_frame, t, (x+32, cy+4), font, fs, c, th, cv2.LINE_AA)
            cy += 24

    def _draw_timing_panel(self, vis_frame, results):
        t = results["timing"]
        lines = [
            "Layer Timing",
            f"Motion: {t['motion_ms']:5.1f} ms  {t['motion_pct']:5.1f}%",
            f"YOLO: {t['yolo_ms']:5.1f} ms  {t['yolo_pct']:5.1f}% [FULL+ATT]",
            f"Total: {t['total_ms']:5.1f} ms"
        ]
        fh, fw = vis_frame.shape[:2]
        font, fs, th, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2, 25
        pw = max(300, max(cv2.getTextSize(l, font, fs, th)[0][0] for l in lines) + 24)
        ph = 14 + lh * len(lines)
        px, py = max(8, fw-pw-8), max(8, fh-ph-8)
        ov = vis_frame.copy()
        cv2.rectangle(ov, (px, py), (px+pw, py+ph), (0,0,0), -1)
        cv2.addWeighted(ov, 0.58, vis_frame, 0.42, 0, vis_frame)
        
        y = py + 27
        for i, l in enumerate(lines):
            clr = (0,255,255) if i==0 else (0,255,0) if l.startswith("Motion") else (0,165,255) if l.startswith("YOLO") else (220,220,220)
            cv2.putText(vis_frame, l, (px+12, y), font, fs, clr, th, cv2.LINE_AA)
            y += lh

    def reset(self):
        if self.tracker: self.tracker.reset()
        self.frame_count = 0
        self.prev_gray = None
        # Keep fixed_motion_pts as they are resolution-dependent
        self.start_time = time.time()
        self.display_fps = 0.0
        self.last_tracks, self.last_detections = [], []
        self.last_motion_count = 0
        self.last_motion_ms = self.last_yolo_ms = self.last_total_ms = 0.0
        self.last_motion_pct = self.last_yolo_pct = 0.0
        self.warmup_frames_remaining = self.reset_warmup_frames


# ================= 辅助函数 =================

def resize_to_max_720p(frame: np.ndarray, max_w=1280, max_h=720) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0: return frame
    scale = min(max_w/w, max_h/h, 1.0)
    ow, oh = max(2, int(round(w*scale))), max(2, int(round(h*scale)))
    ow -= ow % 2; oh -= oh % 2
    if ow == w and oh == h: return frame
    return cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)

def resize_input_frame(frame, max_w=1280, max_h=720):
    h, w = frame.shape[:2]
    if h<=0 or w<=0: return frame
    if w<=max_w and h<=max_h:
        ow, oh = w-(w%2), h-(h%2)
        return frame if ow==w and oh==h else cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
    scale = min(max_w/w, max_h/h)
    ow, oh = max(2, int(round(w*scale))), max(2, int(round(h*scale)))
    ow -= ow%2; oh -= oh%2
    return cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)

def add_timestamp(path):
    p = Path(path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = p.suffix or ".mp4"
    return str(p.with_name(f"{p.stem}_{ts}{suf}"))

class VideoRecorder:
    def __init__(self, path, fps=30.0, enabled=True, add_ts=True):
        self.path = add_timestamp(path) if add_ts else path
        self.fps = float(fps) if fps>0 else 30.0
        self.enabled = enabled
        self.writer = None
        self.size = None
        self.n = 0
    def write(self, frame):
        if not self.enabled: return
        rec = resize_to_max_720p(frame)
        h, w = rec.shape[:2]
        if self.writer is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w,h))
            self.size = (w,h)
            print(f"📹 Recording: {self.path} @ {w}x{h}")
        if self.size != (w,h): rec = cv2.resize(rec, self.size, interpolation=cv2.INTER_AREA)
        self.writer.write(rec)
        self.n += 1
    def release(self):
        if self.writer:
            self.writer.release()
            print(f"✅ Saved: {self.path} ({self.n} frames)")
            self.writer = None

def parse_args():
    p = argparse.ArgumentParser(description="64-Point Motion-Guided YOLO Detection with Gimbal Tracking")
    p.add_argument("--source", default="1")
    p.add_argument("--use-yolo", action="store_true")
    p.add_argument("--device", default="cpu", choices=["cpu","mps","cuda"])
    p.add_argument("--reset-warmup-frames", type=int, default=3)
    p.add_argument("--min-yolo-box-width", type=int, default=30)
    p.add_argument("--min-yolo-box-height", type=int, default=50)
    p.add_argument("--min-yolo-box-area", type=int, default=1500)
    p.add_argument("--yolo-nms-iou", type=float, default=0.45)
    p.add_argument("--motion-thresh", type=float, default=2.0, help="Motion magnitude threshold for 64-point detector")
    p.add_argument("--motion-method", type=str, default="lk", choices=["lk","diff"], help="Motion detection method")
    p.add_argument("--blur-ksize", type=int, default=21)
    p.add_argument("--roi-conf", type=float, default=0.25)
    p.add_argument("--bg-conf", type=float, default=0.45)
    p.add_argument("--width", type=int, default=848)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--resize-input", action="store_true")
    p.add_argument("--input-max-width", type=int, default=1280)
    p.add_argument("--input-max-height", type=int, default=720)
    p.add_argument("--record", dest="record", action="store_true", default=True)
    p.add_argument("--no-record", dest="record", action="store_false")
    p.add_argument("--output", default="outputs/motion_guided_yolo.mp4")
    p.add_argument("--no-timestamp", dest="ts", action="store_false", default=True)
    p.add_argument("--record-fps", type=float, default=30.0)
    p.add_argument("--yolo-model", type=str, default="yolo11n.pt",
                   help="YOLO model path: yolo11n.pt / yolo11n.mlpackage / custom.pt")
    # 云台跟踪参数
    p.add_argument("--mock-gimbal", action="store_true", help="Run gimbal in simulation mode")
    p.add_argument("--can-channel", type=str, default="1", help="PCAN channel index/name (macOS default: 1)")
    p.add_argument("--h-fov", type=float, default=45.0, help="Camera horizontal FOV in degrees")
    p.add_argument("--v-fov", type=float, default=30.0, help="Camera vertical FOV in degrees")
    return p.parse_args()

def create_video_source(args):
    if args.source.lower() == "realsense":
        cam = RealSenseT265(args.width, args.height)
        if cam.initialize(): return True, cam, None
        print("⚠ Fallback to webcam")
        fb = VideoFallback("0"); fb.initialize(); return False, None, fb
    fb = VideoFallback(args.source); fb.initialize(); return False, None, fb

def read_frame(use_rs, cam, fb, resize, mw, mh):
    frame, _ = (cam.get_frame() if use_rs else fb.get_frame())
    if frame is None: return None
    if frame.ndim == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return resize_input_frame(frame, mw, mh) if resize else frame

def main():
    args = parse_args()
    print("="*60)
    print("64-Point Motion-Guided YOLO Detection with Gimbal Tracking")
    print("="*60)
    print(f"Source: {args.source} | YOLO: {'ON' if args.use_yolo else 'OFF'} | Device: {args.device}")
    print(f"Input Resize: {'ON' if args.resize_input else 'OFF'}")
    print(f"Motion: Method={args.motion_method} | Thresh={args.motion_thresh} | Points=64")
    print(f"Attention: Blur={args.blur_ksize} | ROI Conf={args.roi_conf} | BG Conf={args.bg_conf}")
    print(f"Model: {args.yolo_model}")
    print(f"Gimbal: {'MOCK' if args.mock_gimbal else 'ACTIVE'} | HFOV={args.h_fov}° | VFOV={args.v_fov}°")
    print("="*60)

    pipe = MotionDetectionPipeline(
        use_yolo=args.use_yolo, device=args.device,
        reset_warmup_frames=args.reset_warmup_frames,
        min_yolo_box_width=args.min_yolo_box_width,
        min_yolo_box_height=args.min_yolo_box_height,
        min_yolo_box_area=args.min_yolo_box_area,
        yolo_nms_iou=args.yolo_nms_iou,
        motion_thresh=args.motion_thresh,
        motion_method=args.motion_method,
        blur_ksize=args.blur_ksize,
        roi_conf_thresh=args.roi_conf,
        bg_conf_thresh=args.bg_conf
    )
    if args.use_yolo:
        from recognition.yolo_detector import YOLODetector
        pipe.yolo_detector = YOLODetector(model_path=args.yolo_model, device=args.device)
        if not pipe.yolo_detector.initialize():
            print("⚠ YOLO init failed. Running without detection.")
            pipe.use_yolo = False
            pipe.yolo_detector = None
            pipe.tracker = None

    use_rs, cam, fb = create_video_source(args)
    if not use_rs and not (fb and getattr(fb, "is_initialized", False)):
        print("❌ No source. Exiting.")
        return

    # 初始化云台跟踪系统
    mapper = CameraAngleMapper(h_fov_deg=args.h_fov, v_fov_deg=args.v_fov, 
                               img_w=args.input_max_width, img_h=args.input_max_height)
    gimbal = GimbalController(channel=args.can_channel, mock=args.mock_gimbal)
    if not args.mock_gimbal:
        time.sleep(1)  # 等待 CAN 初始化
        gimbal.enable_motors()
        
    print("🎯 云台跟踪已启动，按 Q 退出")
    
    # 🔥 修复：初始化 recorder
    rec = VideoRecorder(args.output, args.record_fps, args.record, args.ts)
    win = "Motion-Guided YOLO + Gimbal Tracking - Q to quit"
    print("Controls: q=Quit r=Reset s=Save v=Toggle Record +/-=MotionThresh")

    try:
        while True:
            # 🔥 修复：传入完整参数
            frame = read_frame(use_rs, cam, fb, args.resize_input, 
                              args.input_max_width, args.input_max_height)
            if frame is None:
                time.sleep(0.05)
                continue
            
            res = pipe.process_frame(frame)
            vis_frame = res["frame"]
            detections = res.get("detections", [])
            
            if detections:
                largest = max(detections, key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]))
                x1, y1, x2, y2 = largest["bbox"]
                target_cx, target_cy = (x1+x2)/2.0, (y1+y2)/2.0
                
                yaw_off, pitch_off = mapper.pixel_to_angle(target_cx, target_cy)
                
                # 绝对角度控制：目标角度 = 当前云台角度 + 偏差
                gimbal.set_target_angles(gimbal.current_yaw + yaw_off, 
                                         gimbal.current_pitch + pitch_off)
                
                cmd_y, cmd_p = gimbal.update(dt=0.02)
                
                cv2.drawMarker(vis_frame, (int(target_cx), int(target_cy)), (0, 255, 0), 
                              cv2.MARKER_CROSS, 20, 2)
                cv2.putText(vis_frame, f"Yaw:{cmd_y:.0f}RPM Pitch:{cmd_p:.0f}RPM", 
                           (10, vis_frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
            elif args.mock_gimbal:
                gimbal.update(dt=0.02)
                
            cv2.imshow(win, vis_frame)
            key = cv2.waitKey(10) & 0xFF
            skip = False
            
            if key == ord("q"): 
                gimbal.emergency_stop()
                break
            elif key == ord("r"): 
                pipe.reset()
                skip = True
                print("🔄 Reset")
            elif key == ord("s"):
                fn = f"motion_yolo_{pipe.frame_count}_{datetime.now().strftime('%H%M%S')}.png"
                cv2.imwrite(fn, vis_frame)
                print(f"💾 {fn}")
            elif key == ord("v"):
                rec.enabled = not rec.enabled
                print(f"📹 Record {'ON' if rec.enabled else 'OFF'}")
            elif key == ord("+") or key == ord("="):
                pipe.motion_thresh = min(20.0, pipe.motion_thresh + 0.5)
                print(f"📈 Motion Thresh: {pipe.motion_thresh:.1f}")
            elif key == ord("-") or key == ord("_"):
                pipe.motion_thresh = max(0.5, pipe.motion_thresh - 0.5)
                print(f"📉 Motion Thresh: {pipe.motion_thresh:.1f}")

            if not skip:
                rec.write(vis_frame)

    except KeyboardInterrupt: 
        print("\n⏹ Interrupted")
        gimbal.emergency_stop()
    finally:
        gimbal.close()
        rec.release()
        if use_rs and cam:
            cam.stop()
        if fb:
            fb.stop()
        cv2.destroyAllWindows()
        t = time.time() - pipe.start_time
        print(f"\n📊 {pipe.frame_count} frames | {pipe.frame_count/t:.1f} FPS avg")

if __name__ == "__main__":
    main()