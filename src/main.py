#!/usr/bin/env python3
"""
RealSense/Webcam + YOLO Motion-Guided Detection System
Main entry point with 64/128-Point Fixed-Grid Optical Flow as ROI Prior.
+ IMU Motion Visualization + HTDW Gimbal Tracking
"""
import argparse
import time
import struct
import can
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from data_source.realsense_wrapper import RealSenseT265, VideoFallback
from recognition.yolo_detector import YOLODetector
from tracking.multi_object_tracker import MultiObjectTracker

# ================= HTDW 电机控制封装 (对齐 htdw_motor_ctrl.py) =================
class HTDWMotor:
    """单轴电机控制器，协议完全对齐 htdw_motor_ctrl.py"""
    def __init__(self, bus, ctrl_id: int, max_rpm: float = 3000.0, safe_torque: int = 2000):
        self.bus = bus
        self.ctrl_id = ctrl_id
        self.max_rpm = max_rpm
        self.safe_torque = safe_torque
        self.enabled = False
        self.current_rpm = 0.0

    def _send_frame(self, speed_rpm, torque_raw=None):
        """发送控制帧，协议对齐 htdw_motor_ctrl.py"""
        if torque_raw is None: torque_raw = self.safe_torque
        speed_raw = int(speed_rpm / 0.015)
        speed_raw = max(-32768, min(32767, speed_raw))
        torque_raw = max(-32768, min(32767, torque_raw))
        
        data = bytearray(8)
        data[0], data[1] = 0x07, 0x35
        data[2:4] = struct.pack('<h', speed_raw)
        data[4:6] = struct.pack('<h', torque_raw)
        data[6:8] = (0x8000).to_bytes(2, byteorder='little')
        
        self.bus.send(can.Message(arbitration_id=self.ctrl_id, data=data, is_extended_id=True))

    def enable(self):
        """使能电机"""
        self.enabled = True
        self._send_frame(self.current_rpm, self.safe_torque)

    def set_speed(self, rpm: float):
        """设置速度（仅当使能时生效）"""
        self.current_rpm = max(-self.max_rpm, min(self.max_rpm, rpm))
        if self.enabled:
            self._send_frame(self.current_rpm, self.safe_torque)

    def stop(self):
        """紧急停止"""
        self.enabled = False
        self.bus.send(can.Message(arbitration_id=self.ctrl_id, data=[0x01, 0x00, 0x00], is_extended_id=True))
        self.current_rpm = 0.0


class MotionDetectionPipeline:
    """
    Complete pipeline for motion-guided detection using fixed-grid optical flow.
    (保持你原有逻辑不变)
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
        motion_method: str = "lk",
        num_motion_points: int = 128,
        h_fov: float = 150.0,
        v_fov: float = 120.0
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
        self.h_fov = h_fov
        self.v_fov = v_fov
        self.roi_w_ratio = 1.0 / 3.0
        self.roi_h_ratio = 1.0

        self.yolo_detector = YOLODetector(device=device) if use_yolo else None
        self.tracker = MultiObjectTracker() if use_yolo else None

        self.frame_count = 0
        self.prev_gray = None
        self.fixed_motion_pts = None
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
        cols = int(np.round(np.sqrt(self.num_motion_points * frame_w / frame_h)))
        cols = max(4, min(cols, self.num_motion_points))
        rows = int(np.ceil(self.num_motion_points / cols))
        cell_w = frame_w / cols
        cell_h = frame_h / rows
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) >= self.num_motion_points: break
                pts.append([(c+0.5)*cell_w, (r+0.5)*cell_h])
        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def _compute_motion_intent(self, prev_gray: np.ndarray, curr_gray: np.ndarray, fixed_pts: np.ndarray) -> np.ndarray:
        if self.motion_method == "diff":
            diff = cv2.absdiff(curr_gray, prev_gray)
            mags = []
            h, w = curr_gray.shape
            for pt in fixed_pts.reshape(-1, 2):
                x, y = int(np.clip(pt[0], 1, w-2)), int(np.clip(pt[1], 1, h-2))
                mags.append(float(np.max(diff[y-1:y+2, x-1:x+2])))
            return np.array(mags, dtype=np.float32)
        else:
            lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, fixed_pts, None, **lk_params)
            if next_pts is None or st is None: return np.zeros(len(fixed_pts), dtype=np.float32)
            dx = next_pts[:,0,0] - fixed_pts[:,0,0]
            dy = next_pts[:,0,1] - fixed_pts[:,0,1]
            return np.sqrt(dx**2 + dy**2) * st.ravel()

    def _get_roi_from_motion_points(self, motion_mask: np.ndarray, fixed_pts: np.ndarray, frame_w: int, frame_h: int):
        if not np.any(motion_mask): return None
        roi_w = int(frame_w * self.roi_w_ratio)
        motion_x = fixed_pts.reshape(-1, 2)[motion_mask, 0]
        if len(motion_x) == 0: return None
        best_rx, best_count = 0, 0
        for rx in range(0, frame_w - roi_w + 1, 2):
            count = np.sum((motion_x >= rx) & (motion_x < rx + roi_w))
            if count > best_count: best_count, best_rx = count, rx
        if best_count == 0: best_rx = (frame_w - roi_w) // 2
        return (best_rx, 0, roi_w, frame_h)

    def process_frame(self, frame: np.ndarray, ang_vel=None) -> Dict:
        frame_start_time = time.perf_counter()
        results = {
            "frame": frame.copy(), "raw_frame": frame.copy(), "roi_box": None,
            "motion_pts": None, "motion_magnitudes": None,
            "detections": self.last_detections, "tracks": self.last_tracks,
            "fps": self.display_fps, "person_count": len(self.last_detections) if self.use_yolo else 0,
            "motion_count": self.last_motion_count, "warmup": self.warmup_frames_remaining > 0,
            "warmup_remaining": self.warmup_frames_remaining,
            "timing": {"motion_ms":0,"yolo_ms":0,"total_ms":0,"motion_pct":0,"yolo_pct":0}
        }
        if self.fixed_motion_pts is None:
            h, w = frame.shape[:2]
            self.fixed_motion_pts = self._generate_fixed_motion_points(w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mags = np.zeros(len(self.fixed_motion_pts), dtype=np.float32)
        mask = np.zeros(len(self.fixed_motion_pts), dtype=bool)
        if self.warmup_frames_remaining <= 0 and self.prev_gray is not None:
            mags = self._compute_motion_intent(self.prev_gray, gray, self.fixed_motion_pts)
            mask = mags >= self.motion_thresh
        results["motion_pts"], results["motion_magnitudes"] = self.fixed_motion_pts.copy(), mags.copy()
        self.last_motion_count = int(np.sum(mask))
        results["motion_count"] = self.last_motion_count
        roi_box = None
        if self.warmup_frames_remaining <= 0 and np.any(mask):
            roi_box = self._get_roi_from_motion_points(mask, self.fixed_motion_pts, *gray.shape[::-1])
            results["roi_box"] = roi_box
        self.prev_gray = gray.copy()
        
        t1 = time.perf_counter()
        if self.warmup_frames_remaining <= 0 and self.use_yolo and self.yolo_detector:
            att_frame = cv2.GaussianBlur(frame, (self.blur_ksize, self.blur_ksize), 0)
            if roi_box:
                rx, ry, rw, rh = roi_box
                att_frame[ry:ry+rh, rx:rx+rw] = frame[ry:ry+rh, rx:rx+rw]
            dets = self.yolo_detector.detect(att_frame)
            if roi_box:
                rx, ry, rw, rh = roi_box
                filtered = []
                for d in dets:
                    if not isinstance(d, dict) or not d.get("bbox"): continue
                    x1,y1,x2,y2 = d["bbox"]
                    in_r = max(x1,rx) < min(x2,rx+rw) and max(y1,ry) < min(y2,ry+rh)
                    if (in_r and d.get("confidence",0)>=self.roi_conf_thresh) or \
                       (not in_r and d.get("confidence",0)>=self.bg_conf_thresh):
                        d["attention"] = "ROI" if in_r else "BG"
                        filtered.append(d)
                dets = filtered
            self.last_detections = dets
            results["detections"] = dets
            if self.tracker:
                self.last_tracks = self.tracker.update(dets)
                results["tracks"] = self.last_tracks
        else:
            results["detections"], results["tracks"] = [], []
        t2 = time.perf_counter()
        
        self.frame_count += 1
        total_ms = (time.perf_counter() - frame_start_time) * 1000
        results["timing"] = {"motion_ms": (t1-frame_start_time)*1000, "yolo_ms": (t2-t1)*1000, "total_ms": total_ms}
        if total_ms > 0:
            self.display_fps = 0.9*self.display_fps + 0.1*(1000/total_ms) if self.display_fps else 1000/total_ms
        results["fps"] = self.display_fps
        self._draw_results(results)
        self._draw_imu_overlay(results["frame"], ang_vel)
        return results

    def _draw_imu_overlay(self, frame: np.ndarray, ang_vel):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        shift_x, shift_y = 0.0, 0.0
        if ang_vel is not None:
            try:
                wx, wy, wz = ang_vel
                fx = (w / 2.0) / np.tan(np.radians(self.h_fov / 2.0))
                fy = (h / 2.0) / np.tan(np.radians(self.v_fov / 2.0))
                dt = 0.033
                shift_x = -fx * wy * dt
                shift_y = -fy * wx * dt
                cv2.putText(frame, f"IMU Rot: wx={wx:.2f} wy={wy:.2f}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Est. Shift: ({shift_x:.1f}, {shift_y:.1f})", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except: pass
        else:
            cv2.putText(frame, "IMU: No Data", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        scale = 2.0
        end_x = int(cx + shift_x * scale)
        end_y = int(cy + shift_y * scale)
        mag = np.sqrt(shift_x**2 + shift_y**2)
        color = (0, 255, 0) if mag < 20 else (0, 255, 255) if mag < 50 else (0, 0, 255)
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 3, tipLength=0.3)
        cv2.circle(frame, (cx, cy), 4, color, -1)

    def _apply_attention_blur(self, frame, roi_box):
        if roi_box is None: return frame.copy()
        rx, ry, rw, rh = roi_box
        blurred = cv2.GaussianBlur(frame, (self.blur_ksize, self.blur_ksize), 0)
        result = blurred.copy()
        result[ry:ry+rh, rx:rx+rw] = frame[ry:ry+rh, rx:rx+rw]
        return result

    def _filter_with_attention(self, detections, frame_shape, roi_box):
        if not detections: return []
        frame_h, frame_w = frame_shape[:2]
        filtered = []
        rx, ry, rw, rh = roi_box if roi_box else (-1, -1, 0, 0)
        in_roi = lambda x1, y1, x2, y2: (max(x1, rx) < min(x2, rx + rw) and max(y1, ry) < min(y2, ry + rh))
        for det in detections:
            if not isinstance(det, dict) or not det.get("bbox"): continue
            try: x1, y1, x2, y2 = map(float, det["bbox"])
            except: continue
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(frame_w-1,x2), min(frame_h-1,y2)
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
            i = order[0]; keep.append(i)
            if order.size == 1: break
            rest = order[1:]
            x1 = np.maximum(boxes_np[i, 0], boxes_np[rest, 0])
            y1 = np.maximum(boxes_np[i, 1], boxes_np[rest, 1])
            x2 = np.minimum(boxes_np[i, 2], boxes_np[rest, 2])
            y2 = np.minimum(boxes_np[i, 3], boxes_np[rest, 3])
            w, h = np.maximum(0.0, x2 - x1), np.maximum(0.0, y2 - y1)
            inter = w * h
            area_i = (boxes_np[i, 2] - boxes_np[i, 0]) * (boxes_np[i, 3] - boxes_np[i, 1])
            area_r = (boxes_np[rest, 2] - boxes_np[rest, 0]) * (boxes_np[rest, 3] - boxes_np[rest, 1])
            ious = inter / (area_i + area_r - inter + 1e-6)
            order = rest[ious <= iou_threshold]
        kept = [detections[i] for i in keep]
        kept.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
        return kept

    def _draw_results(self, results: Dict):
        vis = results["raw_frame"].copy()
        if results["motion_pts"] is not None:
            for i, pt in enumerate(results["motion_pts"].reshape(-1, 2)):
                m = results["motion_magnitudes"][i]
                c = (0,255,0) if m < self.motion_thresh*0.5 else (0,255,255) if m < self.motion_thresh else (0,0,255)
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, c, -1)
        if results["roi_box"]:
            x,y,w,h = results["roi_box"]
            cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,255), 3)
        for det in results.get("detections", []):
            if not isinstance(det, dict) or not det.get("bbox"): continue
            x1,y1,x2,y2 = map(int, det["bbox"])
            clr = (0,255,255) if det.get("attention")=="ROI" else (0,165,255)
            cv2.rectangle(vis, (x1,y1), (x2,y2), clr, 2)
        cv2.putText(vis, f"FPS:{results['fps']:.1f} P:{results['person_count']} M:{results['motion_count']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        results["frame"] = vis

    def reset(self):
        if self.tracker: self.tracker.reset()
        self.frame_count, self.prev_gray, self.last_tracks, self.last_detections = 0, None, [], []
        self.last_motion_count, self.display_fps = 0, 0.0
        self.warmup_frames_remaining = self.reset_warmup_frames


# ================= 辅助函数 =================
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

def parse_args():
    p = argparse.ArgumentParser(description="Motion-Guided YOLO + HTDW Gimbal")
    p.add_argument("--source", default="0")
    p.add_argument("--use-yolo", action="store_true")
    p.add_argument("--device", default="cpu", choices=["cpu","mps","cuda"])
    p.add_argument("--resize-input", action="store_true")
    p.add_argument("--input-max-width", type=int, default=1280)
    p.add_argument("--input-max-height", type=int, default=720)
    p.add_argument("--yolo-model", default="yolo11n.pt")
    p.add_argument("--motion-thresh", type=float, default=2.0)
    p.add_argument("--mock-gimbal", action="store_true", help="模拟云台逻辑")
    p.add_argument("--can-channel", default="PCAN_USBBUS1", help="PCAN 通道名")
    p.add_argument("--h-fov", type=float, default=150.0)
    p.add_argument("--v-fov", type=float, default=120.0)
    p.add_argument("--yaw-zero", type=float, default=0.0, help="Yaw 零位补偿 (度)")
    p.add_argument("--pitch-zero", type=float, default=0.0, help="Pitch 零位补偿 (度)")
    return p.parse_args()

def main():
    args = parse_args()
    print("="*60)
    print("Motion-Guided YOLO + HTDW Gimbal Tracking")
    print("="*60)
    
    # 1. 初始化管线
    pipe = MotionDetectionPipeline(use_yolo=args.use_yolo, device=args.device, motion_thresh=args.motion_thresh,
                                   h_fov=args.h_fov, v_fov=args.v_fov)
    if args.use_yolo:
        pipe.yolo_detector = YOLODetector(model_path=args.yolo_model, device=args.device)
        if not pipe.yolo_detector.initialize():
            print("⚠ YOLO init failed. Running without detection.")
            pipe.use_yolo = False; pipe.yolo_detector = None; pipe.tracker = None

    # 2. 初始化视频源
    use_rs = args.source.lower() == "realsense"
    cam = RealSenseT265(848, 800) if use_rs else None
    fb = VideoFallback(args.source) if not use_rs else None
    if use_rs and not cam.initialize():
        print("⚠ Fallback to webcam"); use_rs=False; fb=VideoFallback("0"); fb.initialize()
    elif not use_rs: fb.initialize()
    if not use_rs and not (fb and getattr(fb, "is_initialized", False)):
        print("❌ No source. Exiting."); return

    # 3. 🔥 初始化 CAN 总线与双轴电机 (关键：拉起电机)
    if not args.mock_gimbal:
        try:
            bus = can.interface.Bus(interface="pcan", channel=args.can_channel, bitrate=1000000)
            yaw_motor = HTDWMotor(bus, 0x8001)   # Yaw 轴 ID
            pitch_motor = HTDWMotor(bus, 0x8002) # Pitch 轴 ID
            yaw_motor.enable()
            pitch_motor.enable()
            print("✅ 电机已使能 (Yaw:0x8001, Pitch:0x8002)")
        except Exception as e:
            print(f"⚠ CAN init failed: {e}. Running in MOCK mode.")
            args.mock_gimbal = True
            bus = yaw_motor = pitch_motor = None
    else:
        print("🎮 Gimbal MOCK mode")
        bus = yaw_motor = pitch_motor = None

    # 伺服控制参数
    KP = 2.5  # P 增益：像素偏差 → RPM 转换系数
    win = "Motion-Guided YOLO + Gimbal - Q to quit"
    print("Controls: q=Quit r=Reset s=Save v=Toggle Record +/-=MotionThresh")

    try:
        while True:
            # 获取帧 + IMU 数据
            data = (cam.get_frame() if use_rs else fb.get_frame())
            if not data or data[0] is None: time.sleep(0.05); continue
            frame = data[0]
            if frame.ndim == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if args.resize_input: frame = resize_input_frame(frame, args.input_max_width, args.input_max_height)
            
            # 提取 IMU 角速度
            ang_vel = None
            if len(data) > 1:
                aux = data[1]
                if isinstance(aux, (tuple, list)) and len(aux) >= 3: ang_vel = aux
                elif isinstance(aux, dict) and 'angular_velocity' in aux:
                    av = aux['angular_velocity']
                    if isinstance(av, (tuple, list)) and len(av) >= 3: ang_vel = av

            res = pipe.process_frame(frame, ang_vel=ang_vel)
            vis = res["frame"]
            h, w = vis.shape[:2]
            dets = res.get("detections", [])
            
            if dets:
                # 选最大目标
                largest = max(dets, key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]))
                x1,y1,x2,y2 = largest["bbox"]
                cx, cy = (x1+x2)/2, (y1+y2)/2
                
                # 🔥 计算偏差并转换为 RPM (P 型伺服)
                err_x = cx - w/2
                err_y = cy - h/2
                rpm_yaw = -KP * err_x + args.yaw_zero    # 负号取决于安装方向
                rpm_pitch = KP * err_y + args.pitch_zero
                
                if not args.mock_gimbal:
                    yaw_motor.set_speed(rpm_yaw)
                    pitch_motor.set_speed(rpm_pitch)
                    
                cv2.drawMarker(vis, (int(cx), int(cy)), (0,255,0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(vis, f"Y:{rpm_yaw:.0f} P:{rpm_pitch:.0f}RPM", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            else:
                # 无目标停转
                if not args.mock_gimbal:
                    yaw_motor.set_speed(0.0)
                    pitch_motor.set_speed(0.0)
                cv2.putText(vis, "NO TARGET - STOPPED", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.imshow(win, vis)
            key = cv2.waitKey(10) & 0xFF
            skip = False
            if key == ord("q"): break
            elif key == ord("r"): pipe.reset(); skip = True; print("🔄 Reset")
            elif key == ord("s"):
                fn = f"motion_yolo_{pipe.frame_count}_{datetime.now().strftime('%H%M%S')}.png"
                cv2.imwrite(fn, vis); print(f"💾 {fn}")
            elif key == ord("v"):
                pass
            elif key == ord("+") or key == ord("="):
                pipe.motion_thresh = min(20.0, pipe.motion_thresh + 0.5); print(f"📈 Thresh: {pipe.motion_thresh:.1f}")
            elif key == ord("-") or key == ord("_"):
                pipe.motion_thresh = max(0.5, pipe.motion_thresh - 0.5); print(f"📉 Thresh: {pipe.motion_thresh:.1f}")

    except KeyboardInterrupt: print("\n⏹ Interrupted")
    finally:
        # 🔥 安全释放：先停电机，再关 CAN
        if not args.mock_gimbal and bus:
            yaw_motor.stop()
            pitch_motor.stop()
            time.sleep(0.2)
            bus.shutdown()
        cv2.destroyAllWindows()
        print("✨ Program Exited")

if __name__ == "__main__":
    main()