#!/usr/bin/env python3
"""
YOLO Only Viewer & Tester
Standalone script for debugging/analyzing the YOLO detection pipeline.
Place in the same directory as main.py.

Features:
- Input frame resizing to max 720P (optional)
- Direct YOLO inference on full frame
- Real-time confidence/IoU threshold adjustment
- FPS, latency, and detection count statistics
- Lazy video recording with auto-timestamp & 720P constraint
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from data_source.realsense_wrapper import RealSenseT265, VideoFallback
from recognition.yolo_detector import YOLODetector


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Only Viewer & Tester")
    parser.add_argument("--source", type=str, default="0", help='Video source: 0, path, or "realsense"')
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"], help="Inference device")
    parser.add_argument("--resize-input", action="store_true", default=False, help="Resize input frame to max resolution")
    parser.add_argument("--input-max-width", type=int, default=1280)
    parser.add_argument("--input-max-height", type=int, default=720)
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Initial confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.45, help="Initial NMS IoU threshold")
    parser.add_argument("--min-box-width", type=int, default=30)
    parser.add_argument("--min-box-height", type=int, default=50)
    parser.add_argument("--min-box-area", type=int, default=1500)
    parser.add_argument("--width", type=int, default=848, help="RealSense frame width")
    parser.add_argument("--height", type=int, default=800, help="RealSense frame height")
    parser.add_argument("--record", dest="record", action="store_true", default=True, help="Record output video")
    parser.add_argument("--no-record", dest="record", action="store_false")
    parser.add_argument("--output", type=str, default="outputs/yolo_debug_output.mp4")
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt",
               help="YOLO model path: yolo11n.pt / yolo11n.mlpackage / custom.pt")
    return parser.parse_args()


def resize_input_frame(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0: return frame
    if w <= max_w and h <= max_h:
        out_w, out_h = w - (w % 2), h - (h % 2)
        return frame if out_w == w and out_h == h else cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    scale = min(max_w / float(w), max_h / float(h))
    out_w, out_h = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    out_w -= out_w % 2
    out_h -= out_h % 2
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)


def filter_detections(detections: List[Dict], frame_h: int, frame_w: int, 
                      min_w: int, min_h: int, min_area: int) -> List[Dict]:
    """Filter invalid/tiny detections and clip to frame bounds."""
    if not detections: return []
    filtered = []
    for det in detections:
        if not isinstance(det, dict): continue
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4: continue
        try:
            x1, y1, x2, y2 = map(float, bbox)
        except Exception: continue
        
        x1 = float(np.clip(x1, 0, frame_w - 1))
        y1 = float(np.clip(y1, 0, frame_h - 1))
        x2 = float(np.clip(x2, 0, frame_w - 1))
        y2 = float(np.clip(y2, 0, frame_h - 1))
        
        if x2 <= x1 or y2 <= y1: continue
        bw, bh = x2 - x1, y2 - y1
        if bw < min_w or bh < min_h or bw * bh < min_area: continue
        
        new_det = det.copy()
        new_det["bbox"] = [x1, y1, x2, y2]
        new_det["confidence"] = float(new_det.get("confidence", 0.0))
        filtered.append(new_det)
    return filtered


def nms_detections(detections: List[Dict], iou_thresh: float) -> List[Dict]:
    """Global NMS for full-frame detections."""
    if not detections: return []
    boxes = [det["bbox"] for det in detections if det.get("bbox") and len(det["bbox"]) == 4]
    scores = [float(det.get("confidence", 0.0)) for det in detections if det.get("bbox") and len(det["bbox"]) == 4]
    if not boxes: return []
    
    boxes_np = np.array(boxes, dtype=float)
    scores_np = np.array(scores, dtype=float)
    order = np.argsort(scores_np)[::-1]
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
        union = area_i + area_r - inter
        ious = inter / union
        order = rest[ious <= iou_thresh]
        
    kept = [detections[i] for i in keep]
    kept.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
    return kept


def draw_detections(vis_frame: np.ndarray, detections: List[Dict], color: Tuple[int, int, int] = (0, 165, 255)):
    """Draw bounding boxes, class names, and confidence."""
    h, w = vis_frame.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    for det in detections:
        if not isinstance(det, dict): continue
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4: continue
        try:
            x1, y1, x2, y2 = map(int, bbox)
        except Exception: continue
        
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
        if x2 <= x1 or y2 <= y1: continue
        
        conf = float(det.get("confidence", 0.0))
        cls = str(det.get("class_name", "unknown"))
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{cls} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, font, scale, thick)
        y_label = y1 - 10 if y1 - 10 > lh else y2 + lh + 5
        cv2.rectangle(vis_frame, (x1, y_label - lh - 4), (x1 + lw + 4, y_label + 4), color, -1)
        cv2.putText(vis_frame, label, (x1 + 2, y_label - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_stats_panel(vis_frame: np.ndarray, fps: float, yolo_ms: float, total_ms: float, 
                     det_count: int, conf: float, iou: float):
    h, w = vis_frame.shape[:2]
    font, scale, thick, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, 28
    lines = [
        f"YOLO Only Viewer",
        f"FPS: {fps:.1f}  |  YOLO: {yolo_ms:.1f} ms ({yolo_ms/total_ms*100 if total_ms>0 else 0:.0f}%)",
        f"Detections: {det_count}",
        f"Conf: {conf:.2f}  |  IoU: {iou:.2f}",
        "Controls: q=Quit s=Save v=Record ↑=Conf+ ↓=Conf- ←=IoU- →=IoU+"
    ]
    pw, ph = 460, 14 + lh * len(lines)
    px, py = 10, 10
    
    overlay = vis_frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
    
    y = py + 26
    for i, line in enumerate(lines):
        clr = (0, 255, 255) if i == 0 else (0, 255, 0) if i < 3 else (200, 200, 200)
        cv2.putText(vis_frame, line, (px + 12, y), font, scale, clr, thick, cv2.LINE_AA)
        y += lh


def create_video_source(args):
    use_rs = False
    rs_cam = None
    fallback = None
    if args.source.lower() == "realsense":
        rs_cam = RealSenseT265(width=args.width, height=args.height)
        if rs_cam.initialize():
            use_rs = True
        else:
            print("⚠ Falling back to webcam")
            fallback = VideoFallback("0")
            fallback.initialize()
    else:
        fallback = VideoFallback(args.source)
        fallback.initialize()
    return use_rs, rs_cam, fallback


def main():
    args = parse_args()
    print("=" * 60)
    print("YOLO Only Viewer")
    print("=" * 60)
    print(f"Source: {args.source} | Device: {args.device}")
    print(f"Input Resize: {'ON' if args.resize_input else 'OFF'}")
    print(f"Conf: {args.conf_thresh} | IoU: {args.iou_thresh}")
    print(f"Model: {args.yolo_model} | Device: {args.device}")
    print("-" * 60)

    # Init YOLO
    print(f"🔹 Loading model: {args.yolo_model}")
    from recognition.yolo_detector import YOLODetector
    yolo = YOLODetector(model_path=args.yolo_model, device=args.device)
    if not yolo.initialize():
        print("❌ YOLO initialization failed. Exiting.")
        return

    use_rs, rs_cam, fallback = create_video_source(args)
    if not use_rs and not (fallback and getattr(fallback, "is_initialized", False)):
        print("❌ No valid video source. Exiting.")
        return

    # Recording setup
    recorder = None
    if args.record:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        rec_path = str(out_path.with_stem(f"{out_path.stem}_{ts}"))
        recorder = {"path": rec_path, "writer": None, "fps": 30.0, "size": None, "enabled": True}
        print(f"📹 Recording to: {rec_path}")

    conf_thresh = args.conf_thresh
    iou_thresh = args.iou_thresh
    frame_cnt = 0
    start_t = time.time()
    disp_fps = 0.0

    print("\nRunning... Press 'q' to quit.")
    try:
        while True:
            frame, _ = (rs_cam.get_frame() if use_rs else fallback.get_frame())
            if frame is None:
                time.sleep(0.05)
                continue
            if frame.ndim == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if args.resize_input:
                frame = resize_input_frame(frame, args.input_max_width, args.input_max_height)

            t0 = time.perf_counter()
            raw_dets = yolo.detect(frame, use_roi_crop=False)
            filtered = filter_detections(raw_dets, frame.shape[0], frame.shape[1],
                                         args.min_box_width, args.min_box_height, args.min_box_area)
            final_dets = [d for d in filtered if d.get("confidence", 0.0) >= conf_thresh]
            final_dets = nms_detections(final_dets, iou_thresh)
            yolo_ms = (time.perf_counter() - t0) * 1000.0

            vis = frame.copy()
            draw_detections(vis, final_dets)
            
            total_ms = yolo_ms + 0.5  # approximate overhead
            frame_cnt += 1
            inst_fps = 1000.0 / total_ms if total_ms > 0 else 0.0
            disp_fps = 0.9 * disp_fps + 0.1 * inst_fps if disp_fps > 0 else inst_fps

            draw_stats_panel(vis, disp_fps, yolo_ms, total_ms, len(final_dets), conf_thresh, iou_thresh)
            cv2.imshow("YOLO Debug", vis)

            # Recording
            if recorder and recorder["enabled"]:
                rec_frame = resize_input_frame(vis, 1280, 720)
                h, w = rec_frame.shape[:2]
                if recorder["writer"] is None:
                    recorder["writer"] = cv2.VideoWriter(recorder["path"], cv2.VideoWriter_fourcc(*"mp4v"), recorder["fps"], (w, h))
                    recorder["size"] = (w, h)
                if recorder["size"] != (w, h):
                    rec_frame = cv2.resize(rec_frame, recorder["size"], interpolation=cv2.INTER_AREA)
                recorder["writer"].write(rec_frame)

            # 🔥 Key handling (10ms waitKey for stable cross-platform capture)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"): break
            elif key == ord("s"):
                fn = f"yolo_frame_{frame_cnt}_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(fn, vis)
                print(f"💾 Saved {fn}")
            elif key == ord("v"):
                if recorder:
                    recorder["enabled"] = not recorder["enabled"]
                    print(f"📹 Recording {'ON' if recorder['enabled'] else 'OFF'}")
            elif key == 82 or key == 2490368:  # Up Arrow
                conf_thresh = min(0.95, conf_thresh + 0.05)
            elif key == 84 or key == 2621440:  # Down Arrow
                conf_thresh = max(0.05, conf_thresh - 0.05)
            elif key == 81 or key == 2424832:  # Left Arrow
                iou_thresh = max(0.1, iou_thresh - 0.05)
            elif key == 83 or key == 2555904:  # Right Arrow
                iou_thresh = min(0.95, iou_thresh + 0.05)

    except KeyboardInterrupt:
        print("\n⏹ Interrupted.")
    finally:
        if recorder and recorder["writer"]:
            recorder["writer"].release()
            print(f"✅ Saved recording: {recorder['path']}")
        if use_rs and rs_cam: rs_cam.stop()
        if fallback: fallback.stop()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_t
        avg_fps = frame_cnt / elapsed if elapsed > 0 else 0.0
        print(f"\n📊 Processed {frame_cnt} frames | Avg FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    main()