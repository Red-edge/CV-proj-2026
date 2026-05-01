# t265_pose_basic.py
import pyrealsense2 as rs
import time

def main():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.pose)
    
    profile = pipe.start(cfg)
    
    print("✅ T265 已启动，按 Ctrl+C 退出\n")
    print(f"{'Time':<8} {'X (m)':<10} {'Y (m)':<10} {'Z (m)':<10} {'Yaw (°)':<10}")
    print("-" * 50)
    
    try:
        while True:
            frames = pipe.wait_for_frames()
            pose_frame = frames.first_or_default(rs.stream.pose)
            
            if pose_frame:
                # ✅ 必须强转为 rs.pose_frame 才能调用 get_pose_data()
                pose_frame = rs.pose_frame(pose_frame)
                pose = pose_frame.get_pose_data()
                
                x, y, z = pose.translation.x, pose.translation.y, pose.translation.z
                
                # 简化计算 Yaw（偏航角）
                yaw = 2 * (pose.rotation.x * pose.rotation.y + pose.rotation.w * pose.rotation.z)
                yaw_deg = yaw * 180 / 3.14159
                
                t = time.strftime("%H:%M:%S")
                print(f"{t:<8} {x:<10.3f} {y:<10.3f} {z:<10.3f} {yaw_deg:<10.2f}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 停止采集")
    finally:
        pipe.stop()

if __name__ == "__main__":
    main()