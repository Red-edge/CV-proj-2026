#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct can-utils controller/monitor for HighTorque HTDW-5047-36-NE.

Function:
  - Uses SocketCAN + can-utils directly: cansend / candump
  - Sends speed-control command with target velocity = 0
  - Sends read-status query
  - Prints feedback angle in raw / turns / rad / deg

Hardware/protocol assumptions:
  - Direct CAN connection to motor CANH/CANL through a SocketCAN adapter.
  - HighTorque classic CAN protocol, extended 16-bit ID:
      request ID = 0x8000 | motor_id
      response ID ~= motor_id << 8
  - CAN bitrate: 1 Mbps.
  - Status query command: 17 01 00 00 00 00 00 00
  - Status response: 27 01 pos_i16 vel_i16 tqe_i16, little-endian.

Requires:
  sudo apt install can-utils
  Python 3.8+
"""

import argparse
import math
import os
import re
import signal
import struct
import subprocess
import sys
import time
from typing import Optional, Tuple


STATUS_QUERY = bytes([0x17, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
STOP_CMD = bytes([0x01, 0x00, 0x00])
BRAKE_CMD = bytes([0x01, 0x00, 0x0F])


def i16_to_le_bytes(x: int) -> bytes:
    if x < -32768 or x > 32767:
        raise ValueError(f"int16 out of range: {x}")
    return struct.pack("<h", x)


def make_speed_zero_cmd(torque_raw: int) -> bytes:
    """
    HighTorque normal speed mode:
      [0] 0x07
      [1] 0x07
      [2:4] pos = 0x8000 means no position limit / disabled position target
      [4:6] vel = 0 means 0 speed target
      [6:8] tqe = torque limit / torque command raw, little endian
    """
    return bytes([0x07, 0x07, 0x00, 0x80]) + i16_to_le_bytes(0) + i16_to_le_bytes(torque_raw)


def cansend(iface: str, can_id: int, data: bytes, dry_run: bool = False) -> None:
    # 8 hex digits => extended CAN ID in can-utils.
    frame = f"{can_id:08X}#{data.hex().upper()}"
    if dry_run:
        print(f"[DRY] cansend {iface} {frame}")
        return
    subprocess.run(["cansend", iface, frame], check=False)


def setup_socketcan(iface: str, bitrate: int, restart_ms: int = 100, dry_run: bool = False) -> None:
    cmds = [
        ["sudo", "ip", "link", "set", iface, "down"],
        ["sudo", "ip", "link", "set", iface, "type", "can", "bitrate", str(bitrate), "restart-ms", str(restart_ms)],
        ["sudo", "ip", "link", "set", iface, "up"],
    ]
    for cmd in cmds:
        if dry_run:
            print("[DRY]", " ".join(cmd))
        else:
            subprocess.run(cmd, check=False)


def parse_candump_line(line: str) -> Optional[Tuple[int, bytes]]:
    """
    Supports common candump formats, especially:
      (1710000000.000000) can0 00000100#270138F609000000
      can0  00000100   [8]  27 01 38 F6 09 00 00 00
    """
    line = line.strip()

    # candump -L style: "(ts) can0 00000100#...." or "00000100##0...."
    m = re.search(r"\b([0-9A-Fa-f]{3,8})(?:##[0-9A-Fa-f]?|#)([0-9A-Fa-f]*)\b", line)
    if m:
        can_id = int(m.group(1), 16)
        hx = m.group(2)
        if len(hx) % 2 != 0:
            return None
        return can_id, bytes.fromhex(hx)

    # default candump style: "can0  00000100   [8]  27 01 ..."
    m = re.search(r"\b([0-9A-Fa-f]{3,8})\s+\[\d+\]\s+((?:[0-9A-Fa-f]{2}\s*)+)", line)
    if m:
        can_id = int(m.group(1), 16)
        hx = "".join(m.group(2).split())
        return can_id, bytes.fromhex(hx)

    return None


def parse_status_response(data: bytes):
    """
    Expected response:
      byte0 = 0x27: response, int16, quantity=3
      byte1 = 0x01: starting register POS_FDBK
      byte2..3 = position int16 little-endian, unit 0.0001 turns
      byte4..5 = velocity int16 little-endian, unit 0.00025 turns/s
      byte6..7 = torque int16 little-endian, true torque depends on motor model
    """
    if len(data) < 8:
        return None
    if data[0] != 0x27 or data[1] != 0x01:
        return None

    pos_raw = struct.unpack_from("<h", data, 2)[0]
    vel_raw = struct.unpack_from("<h", data, 4)[0]
    tqe_raw = struct.unpack_from("<h", data, 6)[0]

    pos_turns = pos_raw / 10000.0
    pos_rad = pos_turns * 2.0 * math.pi
    pos_deg = pos_turns * 360.0

    vel_rps = vel_raw * 0.00025
    vel_rad_s = vel_rps * 2.0 * math.pi

    # For 5047/6056 36 ratio int16 feedback, Seeed docs give:
    # true torque = 0.004563 * tqe_raw - 0.493257
    tau_5047_36_nm = 0.004563 * tqe_raw - 0.493257

    return {
        "pos_raw": pos_raw,
        "vel_raw": vel_raw,
        "tqe_raw": tqe_raw,
        "pos_turns": pos_turns,
        "pos_rad": pos_rad,
        "pos_deg": pos_deg,
        "vel_rps": vel_rps,
        "vel_rad_s": vel_rad_s,
        "tau_5047_36_nm": tau_5047_36_nm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Control HTDW-5047-36-NE at 0 speed and print CAN angle feedback using can-utils."
    )
    parser.add_argument("--iface", default="can0", help="SocketCAN interface, default can0")
    parser.add_argument("--id", type=lambda x: int(x, 0), default=1, help="Motor ID, default 1")
    parser.add_argument("--bitrate", type=int, default=1000000, help="CAN bitrate, default 1000000")
    parser.add_argument("--setup-can", action="store_true", help="Bring up SocketCAN interface at --bitrate")
    parser.add_argument("--rate", type=float, default=50.0, help="Command/query rate Hz, default 50")
    parser.add_argument("--torque-raw", type=int, default=150, help="Speed mode torque raw limit, default 150")
    parser.add_argument("--listen-only", action="store_true", help="Only candump and parse feedback; do not send control/query")
    parser.add_argument("--brake-on-exit", action="store_true", help="Send brake command on exit instead of stop")
    parser.add_argument("--no-stop-on-exit", action="store_true", help="Do not send stop on exit")
    parser.add_argument("--dry-run", action="store_true", help="Print cansend/ip commands without executing")
    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    if args.id < 0 or args.id > 0x7F:
        raise ValueError("--id must be in 0..0x7F")

    req_id = 0x8000 | args.id
    resp_id = args.id << 8
    zero_speed_cmd = make_speed_zero_cmd(args.torque_raw)

    print("=== HighTorque direct can-utils control ===")
    print(f"iface          : {args.iface}")
    print(f"motor id       : {args.id} / request CAN ID 0x{req_id:04X} / expected response ID 0x{resp_id:04X}")
    print(f"zero speed cmd : {zero_speed_cmd.hex(' ').upper()}")
    print(f"status query   : {STATUS_QUERY.hex(' ').upper()}")
    print("Press Ctrl+C to exit.")
    print()

    if args.setup_can:
        setup_socketcan(args.iface, args.bitrate, dry_run=args.dry_run)

    if args.dry_run:
        cansend(args.iface, req_id, zero_speed_cmd, dry_run=True)
        cansend(args.iface, req_id, STATUS_QUERY, dry_run=True)
        return 0

    try:
        candump = subprocess.Popen(
            ["candump", "-L", args.iface],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        print("ERROR: candump not found. Install can-utils: sudo apt install can-utils", file=sys.stderr)
        return 2

    running = True

    def handle_sigint(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    next_send = 0.0
    period = 1.0 / args.rate
    last_print = 0.0

    try:
        while running:
            now = time.time()

            if not args.listen_only and now >= next_send:
                # Send target velocity = 0.
                cansend(args.iface, req_id, zero_speed_cmd)

                # Query pos/vel/torque.
                cansend(args.iface, req_id, STATUS_QUERY)

                next_send = now + period

            # Non-blocking-ish read using select.
            import select
            if candump.stdout is None:
                break

            rlist, _, _ = select.select([candump.stdout], [], [], 0.02)
            if rlist:
                line = candump.stdout.readline()
                parsed = parse_candump_line(line)
                if parsed is None:
                    continue
                can_id, data = parsed

                # Response from motor ID N is usually N << 8, e.g. id=1 -> 0x100.
                if can_id != resp_id:
                    continue

                st = parse_status_response(data)
                if st is None:
                    continue

                # Print all valid feedback frames. Throttle only if bus is too noisy.
                print(
                    f"id={args.id:02d} "
                    f"pos_raw={st['pos_raw']:7d} "
                    f"pos_turns={st['pos_turns']: .5f} "
                    f"pos_rad={st['pos_rad']: .6f} "
                    f"pos_deg={st['pos_deg']: .3f} "
                    f"vel_raw={st['vel_raw']:7d} "
                    f"vel_rad_s={st['vel_rad_s']: .6f} "
                    f"tqe_raw={st['tqe_raw']:7d} "
                    f"tau5047≈{st['tau_5047_36_nm']: .3f}Nm"
                )
                last_print = now

            # A small heartbeat if no feedback is seen.
            if now - last_print > 2.0:
                print("waiting for feedback... check ID, bitrate=1M, CANH/CANL, 120Ω termination, common ground")
                last_print = now

    finally:
        try:
            if not args.listen_only and not args.no_stop_on_exit:
                if args.brake_on_exit:
                    print("\nSending brake command...")
                    cansend(args.iface, req_id, BRAKE_CMD)
                else:
                    print("\nSending stop command...")
                    cansend(args.iface, req_id, STOP_CMD)
        finally:
            if candump.poll() is None:
                candump.terminate()
                try:
                    candump.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    candump.kill()

    return 0


if __name__ == "__main__":
    sys.exit(main())
