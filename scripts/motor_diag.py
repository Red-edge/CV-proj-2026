#!/usr/bin/env python3
"""Low-risk HTDW motor diagnostics over Linux SocketCAN.

The script sends only stop/read commands. It does not write configuration,
reset zero, or command non-zero torque/speed.

The script can be used to query the motor's current position/velocity/torque
"""

import argparse
import socket
import struct
import time
from datetime import datetime

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


def send_frame(sock, can_id: int, data: bytes, extended: bool, label: str):
    sock.send(pack_frame(can_id, data, extended))
    print(f"TX {label:<14} id=0x{can_id:X} data={data.hex(' ').upper()}")


def drain(sock, duration_s: float):
    frames = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        try:
            raw = sock.recv(16)
        except TimeoutError:
            continue
        frames.append((time.time(), *unpack_frame(raw)))
    return frames


def parse_read_response(data: bytes):
    if len(data) < 2 or (data[0] >> 4) != 0x2:
        return None
    dtype = (data[0] >> 2) & 0x03
    count = data[0] & 0x03
    addr = data[1]
    payload = data[2:]
    if dtype == 0:
        values = list(struct.unpack_from(f"<{min(count, len(payload))}b", payload, 0)) if payload else []
        type_name = "int8"
    elif dtype == 1:
        n = min(count, len(payload) // 2)
        values = list(struct.unpack_from(f"<{n}h", payload, 0)) if n else []
        type_name = "int16"
    elif dtype == 2:
        n = min(count, len(payload) // 4)
        values = list(struct.unpack_from(f"<{n}i", payload, 0)) if n else []
        type_name = "int32"
    else:
        n = min(count, len(payload) // 4)
        values = list(struct.unpack_from(f"<{n}f", payload, 0)) if n else []
        type_name = "float"
    return type_name, addr, values


def print_frames(frames, motor_id: int):
    parsed = []
    for ts, can_id, data in frames:
        stamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
        print(f"RX {stamp} id=0x{can_id:X} dlc={len(data)} data={data.hex(' ').upper()}")
        parsed_resp = parse_read_response(data)
        if parsed_resp:
            type_name, addr, values = parsed_resp
            print(f"   parsed read: {type_name} addr=0x{addr:02X} values={values}")
            parsed.append((type_name, addr, values, can_id, data))
            if data[0] == 0x27 and addr == 0x01 and len(values) >= 3:
                pos_turns = values[0] * 0.0001
                vel_rpm = values[1] * 0.00025 * 60.0
                print(
                    f"   status: pos={pos_turns:+.5f} turns ({pos_turns * 360.0:+.2f} deg), "
                    f"vel={vel_rpm:+.2f} rpm, torque_raw={values[2]:+d}"
                )
        elif can_id in (motor_id, 0x100) and len(data) == 0:
            print("   empty response: response master switch is active, but this frame carries no register data")
    return parsed


def read_register(sock, can_id: int, extended: bool, cmd: int, addr: int, wait_s: float, label: str):
    send_frame(sock, can_id, bytes([cmd, addr]), extended, label)
    time.sleep(wait_s)
    frames = drain(sock, wait_s)
    return print_frames(frames, can_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read HTDW motor status/protection registers via SocketCAN")
    parser.add_argument("--iface", default="can0")
    parser.add_argument("--id", type=parse_can_id, default=0x8001, help="CAN arbitration id, default 0x8001")
    parser.add_argument("--standard-id", action="store_true", help="send standard CAN frames instead of extended frames")
    parser.add_argument("--wait", type=float, default=0.08, help="wait seconds after each query")
    parser.add_argument("--scan", action="store_true", help="scan a conservative range of read-only registers")
    parser.add_argument("--timed-return-ms", type=int, default=0,
                        help="temporarily enable periodic status return for this many ms; disabled before exit")
    args = parser.parse_args()

    extended = not args.standard_id
    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((args.iface,))
    sock.settimeout(0.02)
    try:
        print("HTDW motor diagnostics: read-only except stop")
        print(f"iface={args.iface} id=0x{args.id:X} frame_type={'extended' if extended else 'standard'}")
        drain(sock, 0.1)

        send_frame(sock, args.id, bytes([0x01, 0x00, 0x00]), extended, "stop")
        time.sleep(0.1)
        print_frames(drain(sock, 0.2), args.id)

        parsed = []
        parsed += read_register(sock, args.id, extended, 0x17, 0x01, args.wait, "read-pos/vel/tqe")

        if args.timed_return_ms > 0:
            period = max(1, min(args.timed_return_ms, 32767))
            data = bytes([0x05, 0xB4, 0x02, period & 0xFF, (period >> 8) & 0xFF])
            send_frame(sock, args.id, data, extended, "timed-return-on")
            time.sleep(0.05)
            print_frames(drain(sock, max(0.5, period / 1000.0 * 5.0)), args.id)
            send_frame(sock, args.id, bytes([0x05, 0xB4, 0x02, 0x00, 0x00]), extended, "timed-return-off")
            time.sleep(0.05)
            print_frames(drain(sock, 0.2), args.id)

        if args.scan:
            print("\nScanning int8 registers 0x00..0x3F in read-only mode")
            for addr in range(0x00, 0x40, 3):
                parsed += read_register(sock, args.id, extended, 0x13, addr, args.wait, f"read-i8@0x{addr:02X}")

            print("\nScanning int16 registers 0x00..0x3F in read-only mode")
            for addr in range(0x00, 0x40, 3):
                parsed += read_register(sock, args.id, extended, 0x17, addr, args.wait, f"read-i16@0x{addr:02X}")

        interesting = []
        for type_name, addr, values, _can_id, _data in parsed:
            if addr != 0x01 and any(abs(float(v)) > 0 for v in values):
                interesting.append((type_name, addr, values))

        if interesting:
            print("\nNon-zero read responses outside pos/vel/torque:")
            for type_name, addr, values in interesting:
                print(f"  {type_name} addr=0x{addr:02X} values={values}")
        else:
            print("\nNo non-zero read responses outside the basic pos/vel/torque status were observed.")
        return 0
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
