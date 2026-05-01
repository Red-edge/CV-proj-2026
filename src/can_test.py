#!/usr/bin/env python3
import can
import time

# ⚠️ 替换为你的实际端口（第二步 diff 输出的结果）
CHANNEL = "/dev/tty.usbserial-XXXX"  # 例如 /dev/tty.usbserial-A50285BI
BITRATE = 500000

print(f"🔌 尝试打开 slcan 接口: {CHANNEL} @ {BITRATE} bps")
try:
    # slcan 会自动处理串口打开、CAN 波特率设置、协议转换
    bus = can.interface.Bus(bustype="slcan", channel=CHANNEL, bitrate=BITRATE)
    print("✅ 总线打开成功！硬件链路已通。")
    
    # 发送一帧测试数据（ID: 0x601, Data: 0x01 0x00 0x00 0x00 0x00 0x00 0x00 0x00）
    msg = can.Message(arbitration_id=0x601, data=[0x01, 0, 0, 0, 0, 0, 0, 0])
    bus.send(msg)
    print("📤 测试帧已发送。若电机有反应（如继电器吸合/微动），说明协议匹配。")
    
    time.sleep(1)
    bus.shutdown()
except Exception as e:
    print(f"❌ 失败: {e}\n请检查: 1.端口名是否正确 2.是否需 sudo 3.适配器是否支持 slcan")