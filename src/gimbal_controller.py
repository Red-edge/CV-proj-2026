#!/usr/bin/env python3
"""
HTDW-5047-36-NE Gimbal Controller
Based on provided PCAN protocol. Supports dual-axis (Yaw/Pitch) position tracking.
"""
import time
import struct
import numpy as np
try:
    import can
    CAN_AVAILABLE = True
except ImportError:
    CAN_AVAILABLE = False

class HTDWMotor:
    """Single HTDW motor instance with CAN control."""
    def __init__(self, bus, ctrl_id: int, max_rpm: float = 3000.0, safe_torque: int = 2000):
        self.bus = bus
        self.ctrl_id = ctrl_id
        self.max_rpm = max_rpm
        self.safe_torque = safe_torque
        self.current_rpm = 0.0
        self.current_pos = 0.0  # 累计角度 (需通过积分或编码器获取)
        self.enabled = False
        
    def _send_frame(self, speed_rpm, torque_raw=None, pos_raw=None):
        """Send control frame according to HTDW protocol."""
        if torque_raw is None: torque_raw = self.safe_torque
        if pos_raw is None: pos_raw = 0x8000
        
        speed_raw = int(np.clip(speed_rpm / 0.015, -32768, 32767))
        torque_raw = int(np.clip(torque_raw, -32768, 32767))
        
        data = bytearray(8)
        data[0], data[1] = 0x07, 0x35
        struct.pack_into('<h', data, 2, speed_raw)
        struct.pack_into('<h', data, 4, torque_raw)
        struct.pack_into('<H', data, 6, pos_raw)  # 位置占位符
        
        self.bus.send(can.Message(arbitration_id=self.ctrl_id, data=data, is_extended_id=True))

    def enable(self):
        self.enabled = True
        self._send_frame(self.current_rpm)
        
    def set_speed(self, rpm):
        self.current_rpm = float(np.clip(rpm, -self.max_rpm, self.max_rpm))
        if self.enabled: self._send_frame(self.current_rpm)
        
    def stop(self):
        self.enabled = False
        self.bus.send(can.Message(arbitration_id=self.ctrl_id, data=[0x01, 0x00, 0x00], is_extended_id=True))
        self.current_rpm = 0.0

    def brake(self):
        self.enabled = False
        self.bus.send(can.Message(arbitration_id=self.ctrl_id, data=[0x01, 0x00, 0x0F], is_extended_id=True))
        self.current_rpm = 0.0

class GimbalController:
    def __init__(self, 
                 channel: str = "PCAN_USBBUS1", 
                 bitrate: int = 1000000,
                 yaw_id: int = 0x8001, 
                 pitch_id: int = 0x8002,
                 mock: bool = False,
                 yaw_zero_offset: float = 0.0,   # 🔥 新增：Yaw 轴机械/光学零位补偿
                 pitch_zero_offset: float = 0.0): # 🔥 新增：Pitch 轴机械/光学零位补偿
        self.mock = mock
        self.yaw_can_id = yaw_id
        self.pitch_can_id = pitch_id
        
        self.max_yaw = 20.0
        self.max_pitch = 15.0
        self.max_speed_rpm = 500.0
        self.system_latency_ms = 80.0
        
        # 🔥 零位补偿参数
        self.yaw_zero_offset = yaw_zero_offset
        self.pitch_zero_offset = pitch_zero_offset
        
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        
        self.Kp, self.Ki, self.Kd = 3.0, 0.2, 0.1
        self.int_y, self.int_p = 0.0, 0.0
        self.err_y_prev, self.err_p_prev = 0.0, 0.0
        
        if not mock:
            if not CAN_AVAILABLE: raise RuntimeError("python-can not installed.")
            print(f"🔌 初始化 PCAN: {channel} @ {bitrate//1000}Mbps...")
            self.bus = can.interface.Bus(interface="pcan", channel=channel, bitrate=bitrate)
            self.motor_yaw = HTDWMotor(self.bus, yaw_id)
            self.motor_pitch = HTDWMotor(self.bus, pitch_id)
            print("✅ 双轴电机控制器已就绪")
        else:
            print("🎮 Gimbal in MOCK mode")

    def set_target_angles(self, yaw: float, pitch: float):
        # 🔥 叠加零位补偿，确保物理中位对齐
        self.target_yaw = float(np.clip(yaw, -self.max_yaw, self.max_yaw)) + self.yaw_zero_offset
        self.target_pitch = float(np.clip(pitch, -self.max_pitch, self.max_pitch)) + self.pitch_zero_offset

    def stop_motors(self):
        """🔥 无目标时：重置目标为当前位置，发送 0 速指令保持（防漂移）"""
        self.target_yaw = self.current_yaw
        self.target_pitch = self.current_pitch
        if not self.mock:
            self.motor_yaw.set_speed(0.0)
            self.motor_pitch.set_speed(0.0)

    def update(self, dt: float = 0.02):
        # 1. 延迟预测
        lat = self.system_latency_ms / 1000.0
        pred_yaw = self.target_yaw
        pred_pitch = self.target_pitch
        
        # 2. 位置误差
        err_y = pred_yaw - self.current_yaw
        err_p = pred_pitch - self.current_pitch
        
        # 3. PID 计算
        self.int_y = float(np.clip(self.int_y + err_y * dt, -10.0, 10.0))
        self.int_p = float(np.clip(self.int_p + err_p * dt, -10.0, 10.0))
        
        rpm_y = self.Kp * err_y + self.Ki * self.int_y + self.Kd * (err_y - self.err_y_prev) / max(dt, 1e-5)
        rpm_p = self.Kp * err_p + self.Ki * self.int_p + self.Kd * (err_p - self.err_p_prev) / max(dt, 1e-5)
        
        self.err_y_prev = err_y
        self.err_p_prev = err_p
        
        cmd_rpm_y = float(np.clip(rpm_y, -self.max_speed_rpm, self.max_speed_rpm))
        cmd_rpm_p = float(np.clip(rpm_p, -self.max_speed_rpm, self.max_speed_rpm))
        
        if self.mock:
            self.current_yaw += cmd_rpm_y * dt * 0.1
            self.current_pitch += cmd_rpm_p * dt * 0.1
        else:
            if self.motor_yaw.enabled: self.motor_yaw.set_speed(cmd_rpm_y)
            if self.motor_pitch.enabled: self.motor_pitch.set_speed(cmd_rpm_p)
            self._read_feedback()
            
        return cmd_rpm_y, cmd_rpm_p

    def _read_feedback(self):
        """非阻塞读取 CAN 反馈，更新电机状态"""
        while True:
            msg = self.bus.recv(timeout=0)
            if msg is None: break
            if len(msg.data) >= 8 and msg.data[0] == 0x27:
                # 解析速度反馈
                vel_raw = struct.unpack_from('<h', msg.data, 4)[0]
                vel_rpm = vel_raw * 0.00025 * 60
                
                # 更新对应轴状态 (需通过 ID 区分 Yaw/Pitch)
                if msg.arbitration_id == self.motor_yaw.ctrl_id:
                    self.motor_yaw.current_rpm = vel_rpm
                    # 位置积分 (示例：RPM → deg/s → 角度)
                    self.current_yaw += vel_rpm * 0.016 * 0.1  # 0.016s 周期，0.1 为减速比系数
                elif msg.arbitration_id == self.motor_pitch.ctrl_id:
                    self.motor_pitch.current_rpm = vel_rpm
                    self.current_pitch += vel_rpm * 0.016 * 0.1

    def emergency_stop(self):
        print("🛑 云台急停")
        if not self.mock:
            self.motor_yaw.stop()
            self.motor_pitch.stop()
            
    def enable_motors(self):
        if not self.mock:
            self.motor_yaw.enable()
            self.motor_pitch.enable()
            print("✅ 电机已使能")

    def close(self):
        self.emergency_stop()
        if not self.mock and self.bus:
            self.bus.shutdown()