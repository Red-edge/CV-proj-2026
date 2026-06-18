#!/usr/bin/env python3
"""Timed PID tuning motion for the yaw motor.

This script intentionally bypasses limit recovery and boundary braking logic.
It is for controlled PID tuning only. It uses the software zero from the config
and sends speed commands computed from the configured PID gains.

Sequence, each segment lasting --segment-s seconds:
  zero -> +yaw-max-angle -> -yaw-max-angle -> zero
"""

import argparse
import csv
import html
import socket
import struct
import time
from datetime import datetime
from pathlib import Path

CAN_EFF_FLAG = 0x80000000
CAN_EFF_MASK = 0x1FFFFFFF


def normalize_key(key: str) -> str:
    return key.strip().lstrip("-").replace("_", "-")


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_config(path: str) -> dict:
    result = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[normalize_key(key)] = value.strip().strip("\"'")
    return result


def parse_can_id(value: str) -> int:
    return int(value, 0)


def pack_frame(can_id: int, data: bytes, extended: bool = True) -> bytes:
    frame_id = can_id | (CAN_EFF_FLAG if extended else 0)
    return struct.pack("=IB3x8s", frame_id, len(data), data.ljust(8, b"\x00"))


def unpack_frame(raw: bytes):
    can_id, dlc, data = struct.unpack("=IB3x8s", raw)
    return can_id & CAN_EFF_MASK, data[:dlc]


class YawMotor:
    def __init__(
        self,
        iface: str,
        can_id: int,
        zero_deg: float,
        max_rpm: float,
        extended: bool = True,
        motor_invert: bool = False,
    ):
        self.iface = iface
        self.can_id = can_id
        self.zero_deg = zero_deg
        self.max_rpm = abs(max_rpm)
        self.extended = extended
        self.motor_invert = motor_invert
        self.sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.sock.bind((iface,))
        self.sock.settimeout(0.02)

    def close(self):
        self.stop()
        self.sock.close()

    def send(self, data: bytes):
        self.sock.send(pack_frame(self.can_id, data, self.extended))

    def drain(self, seconds: float):
        deadline = time.monotonic() + seconds
        frames = []
        while time.monotonic() < deadline:
            try:
                frames.append(unpack_frame(self.sock.recv(16)))
            except TimeoutError:
                continue
        return frames

    def stop(self):
        try:
            self.send(bytes([0x01, 0x00, 0x00]))
        except OSError:
            pass

    def speed(self, rpm: float):
        safe_rpm = max(-self.max_rpm, min(self.max_rpm, rpm))
        motor_rpm = -safe_rpm if self.motor_invert else safe_rpm
        speed_raw = int(max(-32768, min(32767, motor_rpm / 0.015)))
        data = bytearray(8)
        data[0] = 0x07
        data[1] = 0x35
        struct.pack_into("<h", data, 2, speed_raw)
        struct.pack_into("<h", data, 4, 2000)
        struct.pack_into("<H", data, 6, 0x8000)
        self.send(bytes(data))
        return safe_rpm

    def read_feedback(self):
        self.drain(0.01)
        self.send(bytes([0x17, 0x01]))
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            for frame_id, data in self.drain(0.02):
                if frame_id not in (self.can_id, 0x100) or len(data) < 8:
                    continue
                if data[0] != 0x27 or data[1] != 0x01:
                    continue
                pos_raw, vel_raw, tqe_raw = struct.unpack_from("<hhh", data, 2)
                raw_deg = pos_raw * 0.0001 * 360.0
                rel_deg = raw_deg - self.zero_deg
                vel_rpm = vel_raw * 0.00025 * 60.0
                return {
                    "can_id": frame_id,
                    "data_hex": data.hex(" ").upper(),
                    "pos_raw": pos_raw,
                    "vel_raw": vel_raw,
                    "torque_raw": tqe_raw,
                    "raw_deg": raw_deg,
                    "relative_deg": rel_deg,
                    "feedback_rpm": vel_rpm,
                }
        raise RuntimeError("No 0x27/0x01 feedback received")


class PID:
    def __init__(self, kp: float, ki: float, kd: float, max_rpm: float, deadband: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_rpm = abs(max_rpm)
        self.deadband = abs(deadband)
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float):
        if abs(error) <= self.deadband:
            self.reset()
            return 0.0
        dt = max(0.005, min(0.1, dt))
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        candidate_integral = self.integral + error * dt
        rpm = self.kp * error + self.ki * candidate_integral + self.kd * derivative
        clamped = max(-self.max_rpm, min(self.max_rpm, rpm))
        if rpm == clamped or (clamped >= self.max_rpm and error < 0) or (clamped <= -self.max_rpm and error > 0):
            self.integral = candidate_integral
        self.prev_error = error
        return clamped


def default_output_path():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/pid_tune_{stamp}.csv"


def plot_path_for(csv_path: str) -> str:
    path = Path(csv_path)
    return str(path.with_suffix(".svg"))


def scale_points(rows, x_key, y_key, left, top, width, height, y_min=None, y_max=None):
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    x_min, x_max = min(xs), max(xs)
    if y_min is None:
        y_min = min(ys)
    if y_max is None:
        y_max = max(ys)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    points = []
    for row in rows:
        x = left + (float(row[x_key]) - x_min) / (x_max - x_min) * width
        y = top + height - (float(row[y_key]) - y_min) / (y_max - y_min) * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_svg_plot(csv_path: str, svg_path: str, rows, title: str):
    if not rows:
        return
    width, height = 1100, 620
    left, top, graph_w, graph_h = 80, 56, 960, 430
    angle_values = []
    for row in rows:
        angle_values.extend([float(row["relative_deg"]), float(row["target_deg"])])
    y_min = min(angle_values)
    y_max = max(angle_values)
    pad = max(2.0, (y_max - y_min) * 0.12)
    y_min -= pad
    y_max += pad
    relative_points = scale_points(rows, "time_s", "relative_deg", left, top, graph_w, graph_h, y_min, y_max)
    target_points = scale_points(rows, "time_s", "target_deg", left, top, graph_w, graph_h, y_min, y_max)
    error_points = scale_points(rows, "time_s", "error_deg", left, top, graph_w, graph_h, y_min, y_max)
    x_min = float(rows[0]["time_s"])
    x_max = float(rows[-1]["time_s"])
    kp, ki, kd = rows[0]["kp"], rows[0]["ki"], rows[0]["kd"]
    max_rpm = rows[0]["max_rpm"]

    def x_at(value):
        if x_max <= x_min:
            return left
        return left + (value - x_min) / (x_max - x_min) * graph_w

    def y_at(value):
        return top + graph_h - (value - y_min) / (y_max - y_min) * graph_h

    grid = []
    for i in range(6):
        y_value = y_min + (y_max - y_min) * i / 5.0
        y = y_at(y_value)
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + graph_w}" y2="{y:.2f}" stroke="#E5E7EB"/>')
        grid.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#475569">{y_value:.1f}</text>')
    for i in range(7):
        x_value = x_min + (x_max - x_min) * i / 6.0
        x = x_at(x_value)
        grid.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + graph_h}" stroke="#F1F5F9"/>')
        grid.append(f'<text x="{x:.2f}" y="{top + graph_h + 24}" text-anchor="middle" font-size="12" fill="#475569">{x_value:.2f}s</text>')

    zero_y = y_at(0.0)
    if top <= zero_y <= top + graph_h:
        grid.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + graph_w}" y2="{zero_y:.2f}" stroke="#94A3B8" stroke-dasharray="5 5"/>')

    title_text = html.escape(title)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#FFFFFF"/>
<text x="{left}" y="28" font-size="22" font-family="Arial, sans-serif" fill="#111827">{title_text}</text>
<text x="{left}" y="48" font-size="13" font-family="Arial, sans-serif" fill="#475569">kp={kp} ki={ki} kd={kd} max_rpm={max_rpm}; angle is relative to config software zero</text>
<rect x="{left}" y="{top}" width="{graph_w}" height="{graph_h}" fill="#FFFFFF" stroke="#CBD5E1"/>
{''.join(grid)}
<polyline points="{target_points}" fill="none" stroke="#111827" stroke-width="2.5" stroke-dasharray="8 6"/>
<polyline points="{relative_points}" fill="none" stroke="#2563EB" stroke-width="2.5"/>
<polyline points="{error_points}" fill="none" stroke="#DC2626" stroke-width="1.8" opacity="0.75"/>
<text x="{left}" y="{top + graph_h + 56}" font-size="14" font-family="Arial, sans-serif" fill="#111827">time (s)</text>
<text x="22" y="{top + 20}" transform="rotate(-90 22,{top + 20})" font-size="14" font-family="Arial, sans-serif" fill="#111827">angle / error (deg)</text>
<rect x="{left}" y="{top + graph_h + 78}" width="18" height="4" fill="#2563EB"/>
<text x="{left + 28}" y="{top + graph_h + 84}" font-size="13" font-family="Arial, sans-serif" fill="#111827">feedback relative_deg</text>
<line x1="{left + 210}" y1="{top + graph_h + 80}" x2="{left + 228}" y2="{top + graph_h + 80}" stroke="#111827" stroke-width="2.5" stroke-dasharray="8 6"/>
<text x="{left + 238}" y="{top + graph_h + 84}" font-size="13" font-family="Arial, sans-serif" fill="#111827">target_deg</text>
<rect x="{left + 350}" y="{top + graph_h + 78}" width="18" height="4" fill="#DC2626" opacity="0.75"/>
<text x="{left + 378}" y="{top + graph_h + 84}" font-size="13" font-family="Arial, sans-serif" fill="#111827">error_deg</text>
</svg>
'''
    Path(svg_path).write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Timed yaw PID tuning motion")
    parser.add_argument("config", nargs="?", default="configs/t265_yaw_real.conf")
    parser.add_argument("--segment-s", type=float, default=1.5, help="seconds per target coordinate")
    parser.add_argument("--loop-hz", type=float, default=30.0)
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument("--plot", default=None, help="SVG plot output path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = parse_config(args.config)
    iface = cfg.get("yaw-can-iface", "can0")
    can_id = parse_can_id(cfg.get("yaw-id", "0x8001"))
    zero_deg = float(cfg.get("yaw-feedback-zero-deg", "0"))
    max_angle = abs(float(cfg.get("yaw-max-angle", "30")))
    max_rpm = abs(float(cfg.get("yaw-max-rpm", "30")))
    kp = float(cfg.get("yaw-kp", "0"))
    ki = float(cfg.get("yaw-ki", "0"))
    kd = float(cfg.get("yaw-kd", "0"))
    deadband = abs(float(cfg.get("yaw-deadband", "0.35")))
    extended = not parse_bool(cfg.get("yaw-standard-id", "false"))
    motor_invert = parse_bool(cfg.get("yaw-motor-invert", "false"))
    output = args.output or default_output_path()
    plot_output = args.plot or plot_path_for(output)

    targets = [0.0, +max_angle, -max_angle, 0.0]
    print("Yaw PID tuning")
    print(f"config={args.config}")
    print(f"iface={iface} id=0x{can_id:X} zero={zero_deg:+.3f}deg max_angle=+/-{max_angle:.1f}deg")
    print(
        f"pid: kp={kp} ki={ki} kd={kd} max_rpm={max_rpm} deadband={deadband} "
        f"segment_s={args.segment_s} motor_invert={motor_invert}"
    )
    print(f"targets={targets}")
    print(f"csv={output}")
    print(f"plot={plot_output}")
    if args.dry_run:
        print("DRY RUN: no CAN frames sent")
        return 0

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(plot_output).parent.mkdir(parents=True, exist_ok=True)
    motor = YawMotor(iface, can_id, zero_deg, max_rpm, extended, motor_invert)
    pid = PID(kp, ki, kd, max_rpm, deadband)
    fieldnames = [
        "time_s", "segment", "target_deg", "raw_deg", "relative_deg", "error_deg",
        "cmd_rpm", "feedback_rpm", "feedback_can_id", "feedback_data_hex",
        "pos_raw", "vel_raw", "torque_raw", "kp", "ki", "kd", "max_rpm",
    ]
    rows = []
    try:
        motor.stop()
        time.sleep(0.2)
        t0 = time.monotonic()
        last_t = t0
        with open(output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for segment, target in enumerate(targets):
                pid.reset()
                segment_start = time.monotonic()
                while time.monotonic() - segment_start < args.segment_s:
                    now = time.monotonic()
                    dt = now - last_t
                    last_t = now
                    feedback = motor.read_feedback()
                    rel_deg = feedback["relative_deg"]
                    raw_deg = feedback["raw_deg"]
                    feedback_rpm = feedback["feedback_rpm"]
                    error = target - rel_deg
                    cmd_rpm = pid.update(error, dt)
                    cmd_rpm = motor.speed(cmd_rpm)
                    row = {
                        "time_s": f"{now - t0:.6f}",
                        "segment": segment,
                        "target_deg": f"{target:.6f}",
                        "raw_deg": f"{raw_deg:.6f}",
                        "relative_deg": f"{rel_deg:.6f}",
                        "error_deg": f"{error:.6f}",
                        "cmd_rpm": f"{cmd_rpm:.6f}",
                        "feedback_rpm": f"{feedback_rpm:.6f}",
                        "feedback_can_id": f"0x{feedback['can_id']:X}",
                        "feedback_data_hex": feedback["data_hex"],
                        "pos_raw": feedback["pos_raw"],
                        "vel_raw": feedback["vel_raw"],
                        "torque_raw": feedback["torque_raw"],
                        "kp": kp,
                        "ki": ki,
                        "kd": kd,
                        "max_rpm": max_rpm,
                    }
                    writer.writerow(row)
                    rows.append(row)
                    sleep_s = max(0.0, (1.0 / max(1.0, args.loop_hz)) - (time.monotonic() - now))
                    time.sleep(sleep_s)
        motor.stop()
        write_svg_plot(output, plot_output, rows, f"Yaw PID Tune Feedback - {Path(output).stem}")
        print("OK: PID tune sequence completed")
        print(f"CSV: {output}")
        print(f"Plot: {plot_output}")
        return 0
    finally:
        motor.close()


if __name__ == "__main__":
    raise SystemExit(main())
