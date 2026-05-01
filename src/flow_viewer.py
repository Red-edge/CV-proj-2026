#!/usr/bin/env python3
"""
Fixed 64-Point Motion Intent Detector (Static Grid Sampling)
Standalone script for debugging motion activity at fixed spatial locations.
Place in the same directory as main.py.

Core Logic:
1. Pre-define 64 fixed coordinates in image space (8x8 uniform grid)
2. For each frame, compute optical flow magnitude OR frame-diff at these 64 points
3. Threshold the magnitude: > thresh → "motion intent detected"
4. Output: binary mask / heatmap / count of active points
5. NO tracking, NO direction, NO point replenishment needed.
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

from data_source.realsense_wrapper import RealSenseT265, VideoFallback


def parse_args():
    parser = argparse.ArgumentParser(description="64-Point Fixed Motion Intent Detector")
    parser.add_argument("--source", type=str, default="0", help='Video source: 0, path, or "realsense"')
    parser.add_argument("--resize-input", action="store_true", default=False, help="Resize input frame to max resolution")
    parser.add_argument("--input-max-width", type=int, default=1280)
    parser.add_argument("--input-max-height", type=int, default=720)
    parser.add_argument("--width", type=int, default=848, help="RealSense frame width")
    parser.add_argument("--height", type=int, default=800, help="RealSense frame height")
    parser.add_argument("--motion-thresh", type=float, default=2.0, help="Motion magnitude threshold (pixels/frame)")
    parser.add_argument("--record", dest="record", action="store_true", default=True, help="Record output video")
    parser.add_argument("--no-record", dest="record", action="store_false")
    parser.add_argument("--output", type=str, default="outputs/motion_intent_64pts.mp4")
    return parser.parse_args()


def resize_input_frame(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0: return frame
    if w <= max_w and h <= max_h:
        out_w, out_h = w - (w % 2), h - (h % 2)
        return frame if out_w == w and out_h == h else cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    scale = min(max_w / float(w), max_h / float(h))
    out_w, out_h = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    out_w -= out_w % 2; out_h -= out_h % 2
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)


def generate_fixed_grid_points(frame_w: int, frame_h: int, num_points: int = 64) -> np.ndarray:
    """
    Generate exactly num_points fixed coordinates uniformly distributed in image space.
    Returns: np.ndarray of shape (N, 1, 2) with dtype=float32, compatible with OpenCV LK.
    """
    # 动态计算行列，使网格单元尽量接近正方形
    cols = int(np.round(np.sqrt(num_points * frame_w / frame_h)))
    cols = max(4, min(cols, num_points))
    rows = int(np.ceil(num_points / cols))
    
    cell_w = frame_w / cols
    cell_h = frame_h / rows
    
    pts = []
    for r in range(rows):
        for c in range(cols):
            if len(pts) >= num_points:
                break
            # 取网格中心作为采样点
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            pts.append([cx, cy])
    
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def compute_motion_intent(prev_gray: np.ndarray, curr_gray: np.ndarray, 
                          fixed_pts: np.ndarray, method: str = "lk") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute motion magnitude at fixed spatial points.
    
    Args:
        prev_gray, curr_gray: consecutive grayscale frames
        fixed_pts: shape (N, 1, 2), fixed coordinates in image space
        method: "lk" for optical flow magnitude, "diff" for frame difference
    
    Returns:
        magnitudes: np.ndarray of shape (N,), motion magnitude at each point
        motions: np.ndarray of shape (N,), binary 0/1 indicating motion intent
    """
    if method == "diff":
        # 帧差法：更简单，对快速运动更鲁棒
        diff = cv2.absdiff(curr_gray, prev_gray)
        # 在每个固定点处取 3x3 邻域的最大差值（抗噪）
        magnitudes = []
        h, w = curr_gray.shape
        for pt in fixed_pts.reshape(-1, 2):
            x, y = int(pt[0]), int(pt[1])
            x = np.clip(x, 1, w-2)
            y = np.clip(y, 1, h-2)
            patch = diff[y-1:y+2, x-1:x+2]
            magnitudes.append(float(np.max(patch)))
        magnitudes = np.array(magnitudes, dtype=np.float32)
        
    else:  # method == "lk"
        # 光流法：对缓慢运动更敏感，可亚像素精度
        lk_params = dict(winSize=(15, 15), maxLevel=2, 
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, fixed_pts, None, **lk_params)
        
        if next_pts is None or st is None:
            return np.zeros(len(fixed_pts), dtype=np.float32), np.zeros(len(fixed_pts), dtype=np.uint8)
        
        # 计算位移幅值: sqrt(dx^2 + dy^2)
        dx = next_pts[:, 0, 0] - fixed_pts[:, 0, 0]
        dy = next_pts[:, 0, 1] - fixed_pts[:, 0, 1]
        magnitudes = np.sqrt(dx**2 + dy**2) * st.ravel()  # 丢失的点幅值为0
        
    return magnitudes


def draw_motion_intent(vis_frame: np.ndarray, fixed_pts: np.ndarray, 
                       magnitudes: np.ndarray, threshold: float):
    """
    Draw fixed points colored by motion intent:
    - Green: no motion (magnitude < threshold)
    - Red: motion detected (magnitude >= threshold)
    - Circle size proportional to magnitude
    """
    for i, pt in enumerate(fixed_pts.reshape(-1, 2)):
        x, y = int(pt[0]), int(pt[1])
        mag = magnitudes[i]
        
        # 颜色：绿(静止) → 黄(微动) → 红(强动)
        if mag < threshold * 0.5:
            color = (0, 255, 0)      # Green
        elif mag < threshold:
            color = (0, 255, 255)    # Yellow
        else:
            color = (0, 0, 255)      # Red
        
        # 半径：基础3px + 幅值缩放
        radius = max(3, min(12, int(3 + mag * 1.5)))
        
        cv2.circle(vis_frame, (x, y), radius, color, -1)
        cv2.circle(vis_frame, (x, y), 1, (255, 255, 255), -1)  # 中心白点


def draw_stats_panel(vis_frame: np.ndarray, fps: float, compute_ms: float, 
                     active_count: int, total_points: int, threshold: float):
    h, w = vis_frame.shape[:2]
    font, scale, thick, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, 28
    lines = [
        f"🎯 Motion Intent Detector (64 Fixed Points)",
        f"FPS: {fps:.1f}  |  Compute: {compute_ms:.2f} ms",
        f"Active Points: {active_count} / {total_points}",
        f"Threshold: {threshold:.1f} px/frame",
        "Controls: q=Quit s=Save v=Record +/-=Thresh"
    ]
    pw, ph = 400, 14 + lh * len(lines)
    px, py = 10, 10
    
    overlay = vis_frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
    
    y = py + 26
    for i, line in enumerate(lines):
        clr = (0, 255, 255) if i == 0 else (0, 255, 0) if i < 3 else (200, 200, 200)
        cv2.putText(vis_frame, line, (px + 12, y), font, scale, clr, thick, cv2.LINE_AA)
        y += lh


def draw_heatmap_overlay(vis_frame: np.ndarray, fixed_pts: np.ndarray, magnitudes: np.ndarray, alpha: float = 0.3):
    """
    Optional: draw a soft heatmap overlay based on motion magnitudes.
    """
    if len(magnitudes) == 0:
        return vis_frame
    
    h, w = vis_frame.shape[:2]
    overlay = vis_frame.copy()
    
    # 归一化幅值到 [0, 1]
    max_mag = np.max(magnitudes)
    if max_mag < 1e-3:
        return vis_frame
    
    norm_mag = magnitudes / max_mag
    
    for i, pt in enumerate(fixed_pts.reshape(-1, 2)):
        x, y = int(pt[0]), int(pt[1])
        intensity = int(255 * norm_mag[i])
        # 红色通道增强表示运动强度
        cv2.circle(overlay, (x, y), 8, (intensity, 0, 0), -1)
    
    cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)
    return vis_frame


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
    print("64-Point Fixed Motion Intent Detector")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Input Resize: {'ON' if args.resize_input else 'OFF'}")
    print(f"Motion Threshold: {args.motion_thresh} px/frame")
    print(f"Method: Optical Flow Magnitude (no direction)")
    print("-" * 60)

    use_rs, rs_cam, fallback = create_video_source(args)
    if not use_rs and not (fallback and getattr(fallback, "is_initialized", False)):
        print("❌ No valid video source. Exiting.")
        return

    # 🔥 初始化：生成64个固定空间坐标点
    # 注意：这些点坐标是相对于当前帧分辨率的，如果resize_input开启，需基于resize后的尺寸生成
    sample_frame, _ = (rs_cam.get_frame() if use_rs else fallback.get_frame())
    if sample_frame is None:
        print("❌ Failed to get sample frame. Exiting.")
        return
    if sample_frame.ndim == 2:
        sample_frame = cv2.cvtColor(sample_frame, cv2.COLOR_GRAY2BGR)
    if args.resize_input:
        sample_frame = resize_input_frame(sample_frame, args.input_max_width, args.input_max_height)
    
    sample_h, sample_w = sample_frame.shape[:2]
    FIXED_PTS = generate_fixed_grid_points(sample_w, sample_h, num_points=64)
    print(f"✅ Generated 64 fixed points on {sample_w}x{sample_h} grid")
    
    prev_gray = None
    motion_method = "lk"  # or "diff"

    # Recording setup
    recorder = None
    if args.record:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        rec_path = str(out_path.with_stem(f"{out_path.stem}_{ts}"))
        recorder = {"path": rec_path, "writer": None, "fps": 30.0, "size": None, "enabled": True}
        print(f"📹 Recording to: {rec_path}")

    frame_cnt = 0
    start_t = time.time()
    disp_fps = 0.0
    motion_thresh = args.motion_thresh

    print("\nRunning... Press 'q' to quit, '+/-' to adjust threshold.")
    try:
        while True:
            frame, _ = (rs_cam.get_frame() if use_rs else fallback.get_frame())
            if frame is None:
                time.sleep(0.05)
                continue
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if args.resize_input:
                frame = resize_input_frame(frame, args.input_max_width, args.input_max_height)

            t0 = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 🔥 核心逻辑：计算64个固定点的运动幅值
            if prev_gray is not None:
                magnitudes = compute_motion_intent(prev_gray, gray, FIXED_PTS, method=motion_method)
                # 二值化：幅值 > 阈值 → 有运动意图
                motions = (magnitudes >= motion_thresh).astype(np.uint8)
                active_count = int(np.sum(motions))
            else:
                # 首帧：无运动
                magnitudes = np.zeros(64, dtype=np.float32)
                motions = np.zeros(64, dtype=np.uint8)
                active_count = 0
            
            prev_gray = gray.copy()
            compute_ms = (time.perf_counter() - t0) * 1000.0

            # 🔥 可视化
            vis = frame.copy()
            draw_motion_intent(vis, FIXED_PTS, magnitudes, motion_thresh)
            # Optional: uncomment below for heatmap overlay
            # vis = draw_heatmap_overlay(vis, FIXED_PTS, magnitudes)
            
            draw_stats_panel(vis, disp_fps, compute_ms, active_count, 64, motion_thresh)
            
            # 右下角显示运动点热力缩略图 (8x8 grid visualization)
            grid_viz = motions.reshape(8, 8) if motions.shape[0] == 64 else np.zeros((8,8), dtype=np.uint8)
            grid_img = np.zeros((40, 40, 3), dtype=np.uint8)
            cell = 5
            for r in range(8):
                for c in range(8):
                    if grid_viz[r, c]:
                        cv2.rectangle(grid_img, (c*cell, r*cell), ((c+1)*cell, (r+1)*cell), (0,0,255), -1)
            cv2.putText(vis, "Motion Grid", (vis.shape[1]-60, vis.shape[0]-45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            vis[vis.shape[0]-40:, vis.shape[1]-40:] = grid_img

            cv2.imshow("64-Point Motion Intent", vis)

            key = cv2.waitKey(10) & 0xFF
            
            if key == ord("q"):
                break
            elif key == ord("s"):
                fn = f"motion_intent_{frame_cnt}_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(fn, vis)
                print(f"💾 Saved {fn}")
            elif key == ord("v"):
                if recorder:
                    recorder["enabled"] = not recorder["enabled"]
                    print(f"📹 Recording {'ON' if recorder['enabled'] else 'OFF'}")
            elif key == ord("+") or key == ord("="):
                motion_thresh = min(20.0, motion_thresh + 0.5)
                print(f"📈 Threshold: {motion_thresh:.1f}")
            elif key == ord("-") or key == ord("_"):
                motion_thresh = max(0.5, motion_thresh - 0.5)
                print(f"📉 Threshold: {motion_thresh:.1f}")

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

            frame_cnt += 1
            inst_fps = 1000.0 / (compute_ms + 0.1) if compute_ms > 0 else 0.0
            disp_fps = 0.9 * disp_fps + 0.1 * inst_fps if disp_fps > 0 else inst_fps

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