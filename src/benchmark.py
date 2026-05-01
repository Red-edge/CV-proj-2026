#!/usr/bin/env python3
"""
Quick benchmark: Pure YOLO vs YOLO+ROI (crop mode)
Fixed: Added --model argument support.
"""
import time
import cv2
import numpy as np
import argparse
from pathlib import Path

def load_frames(video_path, max_frames=100, target_h=720):
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        print(f"❌ File not found: {video_path}")
        return []
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ OpenCV cannot open '{video_path.name}'")
        print("\n💡 macOS 解决方案：ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p -y output.mp4\n")
        return []

    frames = []
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        if h != target_h:
            scale = target_h / h
            frame = cv2.resize(frame, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        count += 1
    cap.release()
    
    if not frames:
        print("⚠️ Video contains no valid frames.")
        return []
    print(f"✅ Loaded {len(frames)} frames @ {frames[0].shape[1]}x{frames[0].shape[0]}")
    return frames

def benchmark_yolo(detector, frames, use_roi=False, roi_box=None, n_warmup=5):
    if not frames: return np.nan, np.nan
    times = []
    for i, frame in enumerate(frames):
        if i < n_warmup: continue
        t0 = time.perf_counter()
        infer_frame = frame
        if use_roi and roi_box:
            rx, ry, rw, rh = roi_box
            infer_frame = frame[ry:ry+rh, rx:rx+rw]
            
        dets = detector.detect(infer_frame)
        if use_roi and roi_box and dets:
            rx, ry, _, _ = roi_box
            for d in dets:
                if isinstance(d, dict) and "bbox" in d and len(d["bbox"]) == 4:
                    x1, y1, x2, y2 = d["bbox"]
                    d["bbox"] = [x1+rx, y1+ry, x2+rx, y2+ry]
        times.append((time.perf_counter() - t0) * 1000)
    return (np.mean(times), np.std(times)) if times else (np.nan, np.nan)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Test video path")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path (.pt/.mlpackage)")  # 🔥 新增
    parser.add_argument("--device", default="mps", choices=["cpu","mps","cuda"])
    parser.add_argument("--frames", type=int, default=2000)
    args = parser.parse_args()

    frames = load_frames(args.video, max_frames=args.frames)
    if len(frames) == 0: return

    print(f"🔹 Loading model: {args.model}")
    from recognition.yolo_detector import YOLODetector
    detector = YOLODetector(model_path=args.model, device=args.device)  # 🔥 传入模型路径
    if not detector.initialize():
        print("❌ YOLO init failed")
        return

    h, w = frames[0].shape[:2]
    roi_box = (w//3, 0, w//3, h)

    print(f"\n📊 Benchmark 1: Pure YOLO (Full Frame {w}x{h})")
    mean1, std1 = benchmark_yolo(detector, frames, use_roi=False)
    if not np.isnan(mean1): print(f"   Avg: {mean1:.1f} ± {std1:.1f} ms | FPS: {1000/mean1:.1f}")

    print(f"\n📊 Benchmark 2: YOLO + ROI Crop ({roi_box[2]}x{h})")
    mean2, std2 = benchmark_yolo(detector, frames, use_roi=True, roi_box=roi_box)
    if not np.isnan(mean2): print(f"   Avg: {mean2:.1f} ± {std2:.1f} ms | FPS: {1000/mean2:.1f}")

    if not np.isnan(mean1) and not np.isnan(mean2) and mean2 > 0:
        speedup = mean1 / mean2
        print(f"\n🚀 Speedup: {speedup:.2f}× ({(speedup-1)*100:.0f}% faster)")

if __name__ == "__main__":
    main()