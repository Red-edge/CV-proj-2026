#!/usr/bin/env python3
"""Sweep the yaw motor around the configured software zero.

Sequence:
1. Move to software zero.
2. Move left to +yaw-max-angle.
3. Move right to -yaw-max-angle.
4. Move back to software zero.

Positive relative yaw means motor feedback position increased. This matches the
current installation note: a left turn increases feedback position.
"""

import argparse
import socket
import struct
import time
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
        self.sock.settimeout(0.03)

    def close(self):
        self.stop()
        self.sock.close()

    def send(self, data: bytes, label: str = "frame"):
        self.sock.send(pack_frame(self.can_id, data, self.extended))
        print(f"TX {label:<10} id=0x{self.can_id:X} data={data.hex(' ').upper()}")

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
            self.send(bytes([0x01, 0x00, 0x00]), "stop")
        except OSError:
            pass

    def speed(self, rpm: float):
        safe_rpm = max(-self.max_rpm, min(self.max_rpm, rpm))
        motor_rpm = -safe_rpm if self.motor_invert else safe_rpm
        speed_raw = int(max(-32768, min(32767, motor_rpm / 0.015)))
        torque_raw = 2000
        data = bytearray(8)
        data[0] = 0x07
        data[1] = 0x35
        struct.pack_into("<h", data, 2, speed_raw)
        struct.pack_into("<h", data, 4, torque_raw)
        struct.pack_into("<H", data, 6, 0x8000)
        self.send(bytes(data), f"{safe_rpm:+.2f}rpm motor={motor_rpm:+.2f}")

    def read_relative_deg(self):
        self.drain(0.05)
        self.send(bytes([0x17, 0x01]), "status")
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            for frame_id, data in self.drain(0.05):
                if frame_id not in (self.can_id, 0x100) or len(data) < 8:
                    continue
                if data[0] != 0x27 or data[1] != 0x01:
                    continue
                pos_raw, vel_raw, tqe_raw = struct.unpack_from("<hhh", data, 2)
                raw_deg = pos_raw * 0.0001 * 360.0
                rel_deg = raw_deg - self.zero_deg
                vel_rpm = vel_raw * 0.00025 * 60.0
                print(
                    f"RX status raw={raw_deg:+.3f}deg rel={rel_deg:+.3f}deg "
                    f"vel={vel_rpm:+.2f}rpm tqe={tqe_raw:+d}"
                )
                return rel_deg, raw_deg, vel_rpm
        raise RuntimeError("No 0x27/0x01 feedback received")


def move_to(motor: YawMotor, target_deg: float, hard_limit: float, tolerance: float, timeout: float):
    target_deg = max(-hard_limit, min(hard_limit, target_deg))
    print(f"\n== Move to {target_deg:+.2f} deg ==")
    start = time.monotonic()
    last_rel = None
    wrong_direction_count = 0
    while True:
        rel_deg, _raw_deg, _vel_rpm = motor.read_relative_deg()
        err = target_deg - rel_deg
        if abs(err) <= tolerance:
            motor.stop()
            print(f"Reached {target_deg:+.2f} deg, current rel={rel_deg:+.2f} deg")
            return
        if time.monotonic() - start > timeout:
            motor.stop()
            raise RuntimeError(f"Timeout moving to {target_deg:+.2f} deg; current rel={rel_deg:+.2f} deg")

        direction = 1.0 if err > 0 else -1.0
        if rel_deg >= hard_limit and direction > 0:
            motor.stop()
            raise RuntimeError(f"At/above left hard limit {rel_deg:+.2f} deg; refusing outward motion")
        if rel_deg <= -hard_limit and direction < 0:
            motor.stop()
            raise RuntimeError(f"At/below right hard limit {rel_deg:+.2f} deg; refusing outward motion")

        if last_rel is not None:
            delta = rel_deg - last_rel
            if abs(delta) > 0.05 and delta * direction < 0:
                wrong_direction_count += 1
            else:
                wrong_direction_count = 0
            if wrong_direction_count >= 3:
                motor.stop()
                raise RuntimeError("Feedback moved opposite to commanded direction; check yaw direction/invert")
        last_rel = rel_deg

        if rel_deg < -hard_limit and direction > 0:
            remaining_to_limit = -hard_limit - rel_deg
        elif rel_deg > hard_limit and direction < 0:
            remaining_to_limit = rel_deg - hard_limit
        else:
            remaining_to_limit = (hard_limit - rel_deg) if direction > 0 else (rel_deg + hard_limit)
        if remaining_to_limit <= 0.0:
            motor.stop()
            raise RuntimeError(f"No safe travel remains in commanded direction at rel={rel_deg:+.2f} deg")
        slow_rpm = max(1.0, min(motor.max_rpm, abs(err) * 1.5, remaining_to_limit * 1.5))
        rpm = direction * slow_rpm
        motor.speed(rpm)
        time.sleep(0.08)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep yaw motor to software zero and +/- configured limits")
    parser.add_argument("config", nargs="?", default="configs/t265_yaw_real.conf")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tolerance-deg", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    cfg = parse_config(args.config)
    iface = cfg.get("yaw-can-iface", "can0")
    can_id = parse_can_id(cfg.get("yaw-id", "0x8001"))
    zero_deg = float(cfg.get("yaw-feedback-zero-deg", "0"))
    max_angle = abs(float(cfg.get("yaw-max-angle", "30")))
    margin = min(max_angle, abs(float(cfg.get("yaw-limit-margin-deg", "2"))))
    max_rpm = abs(float(cfg.get("yaw-max-rpm", "20")))
    extended = not parse_bool(cfg.get("yaw-standard-id", "false"))
    motor_invert = parse_bool(cfg.get("yaw-motor-invert", "false"))

    print("Yaw limit sweep")
    print(f"config={args.config}")
    print(
        f"iface={iface} id=0x{can_id:X} zero={zero_deg:+.3f}deg "
        f"hard_limit=+/-{max_angle:.1f}deg brake_margin={margin:.1f}deg "
        f"max_rpm={max_rpm:.1f} motor_invert={motor_invert}"
    )
    print("Direction convention: +yaw means left turn and increased feedback position.")
    if args.dry_run:
        print("DRY RUN sequence: recover-to-nearest-limit-if-outside -> 0 -> +left-limit -> -right-limit -> 0")
        return 0

    motor = YawMotor(iface, can_id, zero_deg, max_rpm, extended, motor_invert)
    try:
        motor.stop()
        time.sleep(0.2)
        rel_deg, _raw_deg, _vel_rpm = motor.read_relative_deg()
        if rel_deg > max_angle:
            move_to(motor, +max_angle, max_angle, args.tolerance_deg, args.timeout_s)
        elif rel_deg < -max_angle:
            move_to(motor, -max_angle, max_angle, args.tolerance_deg, args.timeout_s)
        move_to(motor, 0.0, max_angle, args.tolerance_deg, args.timeout_s)
        move_to(motor, +max_angle, max_angle, args.tolerance_deg, args.timeout_s)
        move_to(motor, -max_angle, max_angle, args.tolerance_deg, args.timeout_s * 2.0)
        move_to(motor, 0.0, max_angle, args.tolerance_deg, args.timeout_s)
        motor.stop()
        print("OK: sweep completed")
        return 0
    finally:
        motor.close()


if __name__ == "__main__":
    raise SystemExit(main())
