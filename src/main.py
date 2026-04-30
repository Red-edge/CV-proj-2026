"""
RealSense T265 + YOLO Motion Detection System
Main entry point for the project

This module:
1. Initializes RealSense T265 camera (or fallback to video file/webcam)
2. Computes optical flow and compensates for ego-motion
3. Generates dynamic ROI based on motion residuals
4. Runs YOLO detection on ROI regions
5. Tracks detected persons across frames

Usage on MacBook M4:
    cd /workspace/src
    python main.py

Or with options:
    python main.py --source 0        # Use webcam
    python main.py --source test.mp4 # Use video file
    python main.py --use-yolo        # Enable YOLO detection
    python main.py --device mps      # Use Apple Silicon GPU
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import time
import argparse

# Import custom modules
from data_source.realsense_wrapper import RealSenseT265, VideoFallback
from motion_detection.optical_flow import OpticalFlowCalculator, EgoMotionCompensator
from motion_detection.roi_generator import ROIGenerator, SaliencyMapGenerator
from recognition.yolo_detector import YOLODetector
from tracking.multi_object_tracker import MultiObjectTracker


class MotionDetectionPipeline:
    """
    Complete pipeline for motion-based person detection.
    """
    
    def __init__(self, 
                 use_yolo: bool = False,
                 device: str = 'cpu',
                 flow_type: str = 'sparse'):
        self.use_yolo = use_yolo
        self.device = device
        
        # Initialize components
        self.motion_detector = OpticalFlowCalculator(flow_type=flow_type)
        self.ego_compensator = EgoMotionCompensator()
        self.roi_generator = ROIGenerator()
        self.saliency_generator = SaliencyMapGenerator()
        
        # Optional YOLO detector
        self.yolo_detector = None
        if use_yolo:
            self.yolo_detector = YOLODetector(device=device)
        
        # Optional tracker
        self.tracker = MultiObjectTracker() if use_yolo else None
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
    
    def process_frame(self, 
                     frame: np.ndarray, 
                     pose_data: Optional[Dict] = None) -> Dict:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input image
            pose_data: Optional pose data from T265
            
        Returns:
            results: Dictionary with all processing results
        """
        results = {
            'frame': frame.copy(),
            'flow_vis': None,
            'roi_boxes': [],
            'detections': [],
            'tracks': [],
            'fps': 0.0
        }
        
        # Step 1: Compute optical flow
        flow_vis, flow_vectors = self.motion_detector.compute(frame)
        results['flow_vis'] = flow_vis
        
        # Step 2: Compensate for ego-motion
        compensated_flows = self.ego_compensator.compensate(
            flow_vectors, pose_data, frame
        )
        
        # Update ego-motion compensator with current frame
        self.ego_compensator.update_frame(frame)
        
        # Step 3: Generate ROI from compensated flow
        roi_boxes = self.roi_generator.generate(compensated_flows, frame.shape)
        results['roi_boxes'] = roi_boxes
        
        # Step 4: Update saliency map
        saliency_map = self.saliency_generator.update(compensated_flows, frame.shape)
        
        # Step 5: Run YOLO detection (if enabled)
        if self.use_yolo and self.yolo_detector and self.frame_count % 2 == 0:
            # Run detection every 2 frames for efficiency
            detections = self.yolo_detector.detect(frame, roi_boxes, use_roi_crop=True)
            
            # Track detections
            if self.tracker:
                tracks = self.tracker.update(detections)
                results['tracks'] = tracks
                results['detections'] = detections
        
        # Step 6: Draw results
        self._draw_results(results, frame)
        
        # Update statistics
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        results['fps'] = self.frame_count / elapsed if elapsed > 0 else 0
        
        return results
    
    def _draw_results(self, results: Dict, frame: np.ndarray):
        """Draw visualization on frame."""
        vis_frame = results['flow_vis'].copy()
        
        # Draw ROI boxes (green)
        for i, (x, y, w, h) in enumerate(results['roi_boxes']):
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_frame, f'ROI {i}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw tracks (blue with ID)
        for track in results.get('tracks', []):
            x1, y1, x2, y2 = map(int, track.state)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_frame, f'ID:{track.track_id}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw trajectory
            if len(track.history) > 1:
                for j in range(1, len(track.history)):
                    prev = track.history[j-1]
                    curr = track.history[j]
                    pt1 = (int((prev[0] + prev[2]) / 2), int((prev[1] + prev[3]) / 2))
                    pt2 = (int((curr[0] + curr[2]) / 2), int((curr[1] + curr[3]) / 2))
                    cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 2)
        
        # Draw FPS and stats
        cv2.putText(vis_frame, f'FPS: {results["fps"]:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(vis_frame, f'ROIs: {len(results["roi_boxes"])}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(vis_frame, f'Tracks: {len(results.get("tracks", []))}', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        results['frame'] = vis_frame
    
    def reset(self):
        """Reset all components."""
        self.motion_detector.reset()
        self.ego_compensator.reset()
        self.roi_generator.reset()
        self.saliency_generator.reset()
        if self.tracker:
            self.tracker.reset()
        self.frame_count = 0
        self.start_time = time.time()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RealSense T265 + YOLO Motion Detection')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source: 0 for webcam, path for video file, or "realsense"')
    parser.add_argument('--use-yolo', action='store_true',
                       help='Enable YOLO person detection')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'mps', 'cuda'],
                       help='Device for YOLO inference')
    parser.add_argument('--flow-type', type=str, default='sparse',
                       choices=['sparse', 'dense'],
                       help='Optical flow type')
    parser.add_argument('--width', type=int, default=848,
                       help='Frame width for RealSense')
    parser.add_argument('--height', type=int, default=800,
                       help='Frame height for RealSense')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 60)
    print("RealSense T265 + YOLO Motion Detection System")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"YOLO: {'Enabled' if args.use_yolo else 'Disabled'}")
    print(f"Device: {args.device}")
    print(f"Flow Type: {args.flow_type}")
    print("-" * 60)
    
    # Initialize pipeline
    pipeline = MotionDetectionPipeline(
        use_yolo=args.use_yolo,
        device=args.device,
        flow_type=args.flow_type
    )
    
    # Initialize YOLO if requested
    if args.use_yolo and pipeline.yolo_detector:
        if not pipeline.yolo_detector.initialize():
            print("⚠ YOLO initialization failed, continuing without detection")
            pipeline.use_yolo = False
    
    # Initialize camera/video source
    use_realsense = False
    realsense_cam = None
    video_fallback = None
    
    if args.source.lower() == 'realsense':
        # Try RealSense T265
        realsense_cam = RealSenseT265(width=args.width, height=args.height)
        if realsense_cam.initialize():
            use_realsense = True
            pipeline.ego_compensator.set_intrinsics(realsense_cam.get_intrinsics())
        else:
            print("⚠ Falling back to webcam")
            video_fallback = VideoFallback('0')
            video_fallback.initialize()
    else:
        # Use specified source (webcam or video file)
        video_fallback = VideoFallback(args.source)
        video_fallback.initialize()
    
    if not use_realsense and not (video_fallback and video_fallback.is_initialized):
        print("❌ No valid video source available. Exiting.")
        return
    
    print("\nControls:")
    print("  q - Quit")
    print("  r - Reset detector")
    print("  s - Save current frame")
    print("-" * 60)
    
    try:
        while True:
            frame = None
            pose_data = None
            
            # Get frame
            if use_realsense:
                frame, pose_data = realsense_cam.get_frame()
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame, pose_data = video_fallback.get_frame()
            
            if frame is None:
                print("⚠ Failed to get frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Process frame
            results = pipeline.process_frame(frame, pose_data)
            
            # Display
            cv2.imshow('Motion Detection - Press Q to quit', results['frame'])
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                pipeline.reset()
                print("Pipeline reset")
            elif key == ord('s'):
                filename = f"frame_{pipeline.frame_count}.png"
                cv2.imwrite(filename, results['frame'])
                print(f"Saved {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        if use_realsense and realsense_cam:
            realsense_cam.stop()
        if video_fallback:
            video_fallback.stop()
        cv2.destroyAllWindows()
    
    print(f"\nProcessed {pipeline.frame_count} frames at {pipeline.frame_count / (time.time() - pipeline.start_time):.1f} FPS average")


if __name__ == '__main__':
    main()
