#!/usr/bin/env python3
"""
HTDW-5047-36-NE 键盘控制 (PCAN 版)
✅ 已适配 macOS PCAN 驱动 | 修复 struct 溢出 | 协议完整对齐
"""
import sys
import tty
import termios
import select
import time
import can
import struct

CONFIG = {
    "channel": "PCAN_USBBUS1",
    "bitrate": 1000000,
    "ctrl_id": 0x8001,           # 0x8000 | id
    "max_rpm": 3000,
    "step_rpm": 50,
    "safe_torque_raw": 2000      # 安全力矩限制
}

class MotorController:
    def __init__(self):
        self.bus = None
        self.current_rpm = 0.0
        self.enabled = False
        self.running = True

    def connect(self):
        print(f"🔌 连接 PCAN: {CONFIG['channel']} @ {CONFIG['bitrate']//1000}Mbps...")
        self.bus = can.interface.Bus(interface="pcan", channel=CONFIG["channel"], bitrate=CONFIG["bitrate"])
        print("✅ PCAN 总线已就绪")

    def send_control_frame(self, speed_rpm, torque_raw=0, pos_raw=0x8000):
        speed_raw = int(speed_rpm / 0.015)
        speed_raw = max(-32768, min(32767, speed_raw))
        torque_raw = max(-32768, min(32767, torque_raw))
        
        data = bytearray(8)
        data[0], data[1] = 0x07, 0x35
        data[2:4] = struct.pack('<h', speed_raw)
        data[4:6] = struct.pack('<h', torque_raw)
        data[6:8] = pos_raw.to_bytes(2, byteorder='little') # 修复 0x8000 溢出

        self.bus.send(can.Message(
            arbitration_id=CONFIG["ctrl_id"],
            data=data,
            is_extended_id=True
        ))

    def send_cmd_raw(self, cmd):
        self.bus.send(can.Message(arbitration_id=CONFIG["ctrl_id"], data=cmd, is_extended_id=True))

    def motor_stop(self): self.send_cmd_raw([0x01, 0x00, 0x00]); self.current_rpm = 0.0
    def motor_brake(self): self.send_cmd_raw([0x01, 0x00, 0x0F]); self.current_rpm = 0.0
    def enable(self): self.enabled = True; self.send_control_frame(self.current_rpm, CONFIG["safe_torque_raw"])
    def set_speed(self, rpm):
        self.current_rpm = max(-CONFIG["max_rpm"], min(CONFIG["max_rpm"], rpm))
        if self.enabled: self.send_control_frame(self.current_rpm, CONFIG["safe_torque_raw"])

    def shutdown(self):
        self.motor_stop()
        if self.bus: self.bus.shutdown()

def init_keyboard():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return old

def restore_keyboard(old): termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

def get_key(timeout=0.05):
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1).strip().lower()
    return None

motor = MotorController()

def main():
    print("🚀 HTDW-5047-36-NE 键盘控制 (PCAN版)")
    motor.connect()
    old_kb = init_keyboard()
    motor.motor_stop()

    print("\n📋 按键: E=使能 | W/S=加/减速 | 空格=归零 | D=停止 | B=刹车 | Q=退出")
    try:
        while motor.running:
            key = get_key()
            if key == 'e': motor.enable()
            elif key == 'w': motor.set_speed(motor.current_rpm + CONFIG["step_rpm"])
            elif key == 's': motor.set_speed(motor.current_rpm - CONFIG["step_rpm"])
            elif key == ' ': motor.set_speed(0)
            elif key == 'd': motor.motor_stop()
            elif key == 'b': motor.motor_brake()
            elif key == 'q': motor.running = False

            msg = motor.bus.recv(timeout=0.05)
            if msg and len(msg.data) >= 4 and msg.data[0] == 0x27:
                vel = struct.unpack_from('<h', msg.data, 4)[0] * 0.00025 * 60
                print(f"\n📥 实际速度: {vel:+.2f} RPM")
    except KeyboardInterrupt: pass
    finally:
        restore_keyboard(old_kb)
        motor.shutdown()

if __name__ == "__main__":
    main()