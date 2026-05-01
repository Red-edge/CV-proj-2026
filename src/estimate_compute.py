#!/usr/bin/env python3
"""
Compute Requirement Estimator for YOLO-based Pipeline
Fixed: Handles ultralytics v8.2+ model.info() returning None.
Usage: python estimate_compute.py --model yolo11n.pt --input 1280x720 --fps 30
"""
import argparse
import numpy as np
from ultralytics import YOLO

def estimate_yolo_flops(model_path: str) -> float:
    """Estimate YOLO FLOPs @ 640x640 using API or fallback baselines."""
    model = YOLO(model_path)
    
    # Try official API (works in older versions)
    info = model.info(verbose=False)
    if isinstance(info, dict) and 'flops' in info:
        return float(info['flops'])
        
    # Fallback 1: Known official baselines @ 640x640
    known_flops = {
        'yolo11n': 6.5, 'yolov8n': 8.1, 'yolov5n': 4.5,
        'yolov9t': 6.8, 'yolov10n': 6.7, 'yolo11s': 21.5
    }
    model_name = model_path.lower().replace('\\', '/')
    for key, flops in known_flops.items():
        if key in model_name:
            print(f"⚠️  Using official baseline for {key}: {flops} GFLOPs @ 640x640")
            return flops
            
    # Fallback 2: Estimate from parameters (YOLO-n series avg ~2.5 GFLOPs per 1M params)
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    estimated = round(params_m * 2.5, 1)
    print(f"⚠️  Model not in baseline. Estimated ~{estimated} GFLOPs from {params_m:.1f}M params")
    return estimated

def estimate_pipeline_flops(
    model_flops: float,
    num_motion_pts: int = 256,
    fps: float = 30.0,
    overhead_factor: float = 1.5
) -> dict:
    """Estimate full pipeline compute requirements."""
    # Per-frame FLOPs breakdown (GFLOPs)
    preprocess = 0.015          # resize, normalize, color convert
    motion_detect = num_motion_pts * 3e-6  # LK ~3K FLOPs/pt
    roi_search = 0.00003        # sliding window search
    postprocess = 0.000005      # NMS, tracking update
    
    total_per_frame = preprocess + motion_detect + roi_search + model_flops + postprocess
    total_per_second = total_per_frame * fps
    required_with_margin = total_per_second * overhead_factor
    
    return {
        'per_frame_gflops': total_per_frame,
        'per_second_gflops': total_per_second,
        'required_with_margin_tflops': required_with_margin / 1000,
        'breakdown': {
            'preprocess': preprocess,
            'motion_detect': motion_detect,
            'roi_search': roi_search,
            'yolo': model_flops,
            'postprocess': postprocess,
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo11n.pt', help='YOLO model path')
    parser.add_argument('--input', default='1280x720', help='Input resolution (WxH)')
    parser.add_argument('--fps', type=float, default=30.0, help='Target FPS')
    parser.add_argument('--motion-pts', type=int, default=256, help='Fixed motion points')
    args = parser.parse_args()
    
    w, h = map(int, args.input.split('x'))
    
    print(f"🔍 Estimating compute for {args.model} @ {args.input} {args.fps}FPS")
    print("-" * 60)
    
    yolo_flops = estimate_yolo_flops(args.model)
    print(f"📊 YOLO FLOPs: {yolo_flops:.2f} GFLOPs/frame (letterbox-padded to 640x640)")
    
    result = estimate_pipeline_flops(
        model_flops=yolo_flops,
        num_motion_pts=args.motion_pts,
        fps=args.fps
    )
    
    print(f"\n📈 Pipeline Breakdown (per frame):")
    total = result['per_frame_gflops']
    for k, v in result['breakdown'].items():
        pct = (v / total) * 100 if total > 0 else 0
        print(f"  {k:18s}: {v:8.4f} GFLOPs ({pct:5.1f}%)")
    
    print(f"\n🎯 Hardware Requirements:")
    print(f"  Per frame:       {result['per_frame_gflops']:.2f} GFLOPs")
    print(f"  @ {args.fps} FPS:      {result['per_second_gflops']:.1f} GFLOPs/s")
    print(f"  With 1.5× margin: {result['required_with_margin_tflops']:.3f} TFLOPs/s")
    
    print(f"\n💡 Platform Recommendations:")
    req = result['required_with_margin_tflops']
    if req < 0.5:
        print("  ✅ RK3588 NPU (6 TOPS INT8) → 极佳匹配，8× 算力余量")
        print("  ✅ Jetson Orin Nano (20 TOPS) → 性能过剩，适合多模型并行")
    elif req < 2.0:
        print("  ⚠️  RK3588 NPU → 需模型量化/降分辨率")
        print("  ✅ Jetson Orin Nano → 推荐选择")
    else:
        print("  ❌ 边缘 NPU 算力不足，建议:")
        print("     - 模型剪枝 + INT8/INT4 量化")
        print("     - 输入分辨率降至 480p")
        print("     - 考虑云端协同推理")

if __name__ == "__main__":
    main()