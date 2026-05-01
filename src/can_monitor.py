#!/usr/bin/env python3
"""
HTDW-5047-36-NE CAN 监听器 (PCAN 版)
✅ 适配 Peak PCAN-USB | 自动过滤干扰 | 协议精准映射
"""
import can
import struct
import time

CONFIG = {
    "bitrate": 1000000,      # 协议指定 1Mbps
    "channel": "PCAN_USBBUS1" # Peak 官方通道名（首个设备）
}

K_TQE = 0.004563
D_TQE = -0.493257

def parse_0x27_status(data):
    if len(data) < 8 or data[0] != 0x27 or data[1] != 0x01:
        return None
    pos_raw = struct.unpack_from('<h', data, 2)[0]
    vel_raw = struct.unpack_from('<h', data, 4)[0]
    tqe_raw = struct.unpack_from('<h', data, 6)[0]
    return pos_raw * 0.0001, vel_raw * 0.00025 * 60.0, K_TQE * tqe_raw + D_TQE

def main():
    print(f"🔌 正在初始化 PCAN: {CONFIG['channel']} @ {CONFIG['bitrate']//1000}Mbps...")
    try:
        bus = can.interface.Bus(interface="pcan", channel=CONFIG["channel"], bitrate=CONFIG["bitrate"])
    except Exception as e:
        print(f"❌ 连接失败: {e}\n💡 请检查: 1. Peak驱动是否安装 2. USB是否插紧 3. 是否需 sudo")
        return

    print("✅ PCAN 总线已就绪。开始监听... (Ctrl+C 退出)")
    print("-" * 85)
    print(f"{'时间':<8} | {'CAN ID':<7} | {'DLC':<3} | {'原始数据 (Hex)':<28} | {'协议解析'}")
    print("-" * 85)

    try:
        while True:
            msg = bus.recv(timeout=0.5)
            if msg is None: continue

            data_hex = " ".join(f"{b:02X}" for b in msg.data)
            ts = time.strftime("%H:%M:%S")
            raw_line = f"{ts:<8} | {msg.arbitration_id:#06X} | {msg.dlc:<3} | {data_hex:<28}"
            
            parsed = ""
            if len(msg.data) >= 8 and msg.data[0] == 0x27:
                res = parse_0x27_status(msg.data)
                if res:
                    parsed = f"📊 位置={res[0]:+.4f}转 | 速度={res[1]:+.2f}RPM | 力矩={res[2]:+.3f}Nm"
            elif msg.data and msg.data[0] in (0x17, 0x01, 0x05, 0x07):
                parsed = "📤/[控制/查询] 帧"
            print(f"{raw_line} | {parsed}")
            
    except KeyboardInterrupt:
        print("\n🛑 监听已停止")
    finally:
        bus.shutdown()
        print("🔌 PCAN 接口已安全释放")

if __name__ == "__main__":
    main()