#!/usr/bin/env python3
"""Benchmark the yaw motor SocketCAN speed/status loop.

Default mode sends 0 rpm speed frames, queries 0x27 status feedback, and reports
actual command/feedback rates plus timing jitter. It does not change motor
configuration or zero position.
"""

import argparse
import math
import socket
import statistics
import struct
import time

CAN_EFF_FLAG = 0x80000000
CAN_EFF_MASK = 0x1FFFFFFF


def parse_can_id(value: str) -> int:
    return int(value, 0)


def pack_frame(can_id: int, data: bytes, extended: bool) -> bytes:
    frame_id = can_id | (CAN_EFF_FLAG if extended else 0)
    return struct.pack("=IB3x8s", frame_id, len(data), data.ljust(8, b"\x00"))


def unpack_frame(raw: bytes):
    can_id, dlc, data = struct.unpack("=IB3x8s", raw)
    return can_id & CAN_EFF_MASK, data[:dlc]


def pct(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def speed_frame(rpm: float, torque_raw: int) -> bytes:
    speed_raw = int(max(-32768, min(32767, rpm / 0.015)))
    torque_raw = int(max(-32768, min(32767, torque_raw)))
    return bytes([0x07, 0x35]) + struct.pack("<hhH", speed_raw, torque_raw, 0x8000)


def parse_status(data: bytes):
    if len(data) < 8 or data[0] != 0x27 or data[1] != 0x01:
        return None
    pos_raw, vel_raw, torque_raw = struct.unpack_from("<hhh", data, 2)
    return {
        "pos_deg": pos_raw * 0.0001 * 360.0,
        "vel_rpm": vel_raw * 0.00025 * 60.0,
        "torque_raw": torque_raw,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark HTDW yaw motor 1kHz SocketCAN inner loop")
    parser.add_argument("--iface", default="can0")
    parser.add_argument("--id", type=parse_can_id, default=0x8001)
    parser.add_argument("--standard-id", action="store_true")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--command-hz", type=float, default=1000.0)
    parser.add_argument("--feedback-hz", type=float, default=1000.0)
    parser.add_argument("--rpm", type=float, default=0.0)
    parser.add_argument("--torque-raw", type=int, default=2000)
    args = parser.parse_args()

    extended = not args.standard_id
    command_period = 1.0 / max(1.0, args.command_hz)
    feedback_period = 1.0 / max(1.0, args.feedback_hz)
    stop_data = bytes([0x01, 0x00, 0x00])
    query_data = bytes([0x17, 0x01])
    cmd_data = speed_frame(args.rpm, args.torque_raw)

    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((args.iface,))
    sock.setblocking(False)

    command_times = []
    feedback_times = []
    jitter_us = []
    missed_feedback = 0
    last_status = None
    next_cmd = time.monotonic()
    next_feedback = next_cmd
    end_time = next_cmd + max(0.1, args.duration_s)
    last_cmd_time = None

    try:
        sock.send(pack_frame(args.id, stop_data, extended))
        while time.monotonic() < end_time:
            now = time.monotonic()
            if now >= next_feedback:
                sock.send(pack_frame(args.id, query_data, extended))
                next_feedback += feedback_period
            if now >= next_cmd:
                sock.send(pack_frame(args.id, cmd_data, extended))
                command_times.append(now)
                if last_cmd_time is not None:
                    actual_us = (now - last_cmd_time) * 1_000_000.0
                    jitter_us.append(actual_us - command_period * 1_000_000.0)
                last_cmd_time = now
                next_cmd += command_period
            got_feedback = False
            while True:
                try:
                    raw = sock.recv(16)
                except BlockingIOError:
                    break
                can_id, data = unpack_frame(raw)
                if can_id not in (args.id, 0x100):
                    continue
                status = parse_status(data)
                if status is not None:
                    got_feedback = True
                    last_status = status
                    feedback_times.append(time.monotonic())
            if not got_feedback and now >= next_feedback:
                missed_feedback += 1
            sleep_s = min(max(0.0, next_cmd - time.monotonic()), 0.0002)
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    finally:
        sock.send(pack_frame(args.id, speed_frame(0.0, args.torque_raw), extended))
        sock.send(pack_frame(args.id, stop_data, extended))
        sock.close()

    elapsed = max(1e-6, command_times[-1] - command_times[0]) if len(command_times) > 1 else args.duration_s
    feedback_elapsed = max(1e-6, feedback_times[-1] - feedback_times[0]) if len(feedback_times) > 1 else args.duration_s
    print(f"command_count={len(command_times)} command_hz_actual={len(command_times) / elapsed:.2f}")
    print(f"feedback_count={len(feedback_times)} feedback_hz_actual={len(feedback_times) / feedback_elapsed:.2f}")
    print(f"missed_feedback={missed_feedback}")
    print(
        "jitter_us "
        f"mean={statistics.fmean(jitter_us) if jitter_us else math.nan:.2f} "
        f"p50={pct(jitter_us, 0.50):.2f} "
        f"p90={pct(jitter_us, 0.90):.2f} "
        f"p95={pct(jitter_us, 0.95):.2f}"
    )
    if last_status:
        print(
            "last_status "
            f"pos_deg={last_status['pos_deg']:+.2f} "
            f"vel_rpm={last_status['vel_rpm']:+.2f} "
            f"torque_raw={last_status['torque_raw']:+d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
