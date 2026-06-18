#!/usr/bin/env python3
"""Reset HTDW yaw motor internal zero over Linux SocketCAN.

The script intentionally uses only the Python standard library. It sends the
documented HighTorque/HTDW zero-position command, saves settings, then verifies
with a status query when feedback frames are available.
"""

import argparse
import socket
import struct
import time
from datetime import datetime
from pathlib import Path

CAN_EFF_FLAG = 0x80000000
CAN_EFF_MASK = 0x1FFFFFFF


def parse_can_id(value: str) -> int:
    return int(value, 0)


def parse_hex_bytes(value: str) -> bytes:
    compact = value.replace(" ", "").replace(":", "").replace("-", "")
    if len(compact) % 2 != 0:
        raise argparse.ArgumentTypeError("hex byte string must have an even number of digits")
    data = bytes.fromhex(compact)
    if len(data) > 8:
        raise argparse.ArgumentTypeError("CAN data cannot exceed 8 bytes")
    return data


def pack_frame(can_id: int, data: bytes, extended: bool) -> bytes:
    frame_id = can_id | (CAN_EFF_FLAG if extended else 0)
    return struct.pack("=IB3x8s", frame_id, len(data), data.ljust(8, b"\x00"))


def unpack_frame(raw: bytes):
    can_id, dlc, data = struct.unpack("=IB3x8s", raw)
    return can_id, data[:dlc]


def send_frame(sock, can_id: int, data: bytes, extended: bool, label: str):
    sock.send(pack_frame(can_id, data, extended))
    print(f"TX {label:<12} id=0x{can_id:X} data={data.hex(' ').upper()}")


def drain(sock, duration_s: float):
    deadline = time.monotonic() + duration_s
    frames = []
    while time.monotonic() < deadline:
        try:
            raw = sock.recv(16)
        except TimeoutError:
            continue
        can_id, data = unpack_frame(raw)
        frames.append((time.time(), can_id, data))
    return frames


def print_frames(frames, motor_id: int):
    report = {
        "feedback": [],
        "zero_ack": False,
        "save_error": None,
        "config_errors": [],
    }
    for ts, can_id, data in frames:
        clean_id = can_id & CAN_EFF_MASK
        stamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
        print(f"RX {stamp} id=0x{clean_id:X} data={data.hex(' ').upper()}")
        if clean_id in (motor_id, 0x100) and data == b"\x41\x01\x00":
            report["zero_ack"] = True
            print("   parsed: rezero_pos ACK, code=0")
        if clean_id in (motor_id, 0x100) and len(data) >= 4 and data[0] == 0x30:
            err = (data[1], data[2], data[3])
            report["config_errors"].append(err)
            if data[1] == 0xB3 and data[2] == 0x02:
                report["save_error"] = data[3]
                print(f"   parsed: conf_write failed/unsupported, code={data[3]}")
            else:
                print(f"   parsed: config/write command failed/unsupported, addr=0x{data[1]:02X}, type=0x{data[2]:02X}, code={data[3]}")
        if clean_id in (motor_id, 0x100) and len(data) >= 8 and data[0] == 0x27 and data[1] == 0x01:
            pos_raw, vel_raw, tqe_raw = struct.unpack_from("<hhh", data, 2)
            pos_turns = pos_raw * 0.0001
            vel_rpm = vel_raw * 0.00025 * 60.0
            report["feedback"].append((pos_turns, vel_rpm, tqe_raw))
            print(f"   parsed: pos={pos_turns:+.5f} turns ({pos_turns * 360.0:+.2f} deg), vel={vel_rpm:+.2f} rpm, torque_raw={tqe_raw:+d}")
    return report


def latest_feedback(report):
    feedback = report["feedback"]
    return feedback[-1] if feedback else None


def update_software_zero_config(config_path: str, zero_deg: float) -> None:
    path = Path(config_path).expanduser()
    key = "yaw-feedback-zero-deg"
    line = f"{key}={zero_deg:.6f}\n"
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    else:
        lines = []

    replaced = False
    for idx, existing in enumerate(lines):
        stripped = existing.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        existing_key = stripped.split("=", 1)[0].strip().lstrip("-").replace("_", "-")
        if existing_key == key:
            lines[idx] = line
            replaced = True
            break

    if not replaced:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(line)

    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset HTDW yaw motor internal zero via SocketCAN")
    parser.add_argument("--iface", default="can0")
    parser.add_argument("--id", type=parse_can_id, default=0x8001, help="CAN arbitration id, default 0x8001")
    parser.add_argument("--standard-id", action="store_true", help="send standard CAN frames instead of extended frames")
    parser.add_argument("--zero-frame", type=parse_hex_bytes, default=parse_hex_bytes("40 01 04 64 20 63 0A"))
    parser.add_argument("--save-frame", type=parse_hex_bytes, default=parse_hex_bytes("05 B3 02 00 00"))
    parser.add_argument("--status-frame", type=parse_hex_bytes, default=parse_hex_bytes("17 01"))
    parser.add_argument("--stop-frame", type=parse_hex_bytes, default=parse_hex_bytes("01 00 00"))
    parser.add_argument("--settle-ms", type=int, default=250)
    parser.add_argument("--zero-save-delay-ms", type=int, default=1000,
                        help="delay between rezero_pos and conf_write; official guide recommends about 1000 ms")
    parser.add_argument("--verify-timeout-ms", type=int, default=1200,
                        help="time to wait for status feedback during verification")
    parser.add_argument("--zero-tolerance-deg", type=float, default=2.0,
                        help="maximum absolute position accepted after zero reset")
    parser.add_argument("--no-save", action="store_true", help="do not send save_settings after zero reset")
    parser.add_argument("--dry-run", action="store_true", help="print the frame sequence without opening CAN")
    parser.add_argument("--status-only", action="store_true", help="only request and parse current motor status")
    parser.add_argument("--record-software-zero", metavar="CONFIG",
                        help="read current motor position and write yaw-feedback-zero-deg to the config file")
    args = parser.parse_args()

    extended = not args.standard_id
    if args.dry_run:
        print("DRY RUN: no CAN frames will be sent")
        print(f"iface={args.iface} id=0x{args.id:X} frame_type={'extended' if extended else 'standard'}")
        print(f"stop:      {args.stop_frame.hex(' ').upper()}")
        print(f"set_zero:  {args.zero_frame.hex(' ').upper()}")
        print(f"wait:      {args.zero_save_delay_ms} ms before save")
        if not args.no_save:
            print(f"save:      {args.save_frame.hex(' ').upper()}")
        print(f"status:    {args.status_frame.hex(' ').upper()}")
        print(f"verify:    abs(position_deg) <= {args.zero_tolerance_deg}")
        if args.record_software_zero:
            print(f"record:    write current feedback angle to {args.record_software_zero}")
        print(f"stop:      {args.stop_frame.hex(' ').upper()}")
        return 0

    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((args.iface,))
    sock.settimeout(0.05)

    print("Motor internal zero reset")
    print(f"iface={args.iface} id=0x{args.id:X} frame_type={'extended' if extended else 'standard'}")
    print("Keep the yaw axis mechanically centered and unloaded before running this script.")

    try:
        drain(sock, 0.1)
        if args.status_only:
            send_frame(sock, args.id, args.status_frame, extended, "status")
            time.sleep(args.settle_ms / 1000.0)
            report = print_frames(drain(sock, 0.8), args.id)
            return 0 if report["feedback"] else 2

        if args.record_software_zero:
            send_frame(sock, args.id, args.status_frame, extended, "status")
            time.sleep(args.settle_ms / 1000.0)
            report = print_frames(drain(sock, 0.8), args.id)
            current = latest_feedback(report)
            if not current:
                print("ERROR: no 0x27 status feedback; cannot record software zero.")
                return 2
            pos_turns, vel_rpm, _tqe_raw = current
            if abs(vel_rpm) > 0.5:
                print(f"ERROR: motor velocity is {vel_rpm:+.2f} rpm; hold still before recording software zero.")
                return 5
            zero_deg = pos_turns * 360.0
            update_software_zero_config(args.record_software_zero, zero_deg)
            print(f"OK: wrote yaw-feedback-zero-deg={zero_deg:.6f} to {args.record_software_zero}")
            print("    Main program will treat this feedback angle as yaw 0 deg and limit around +/- yaw-max-angle.")
            return 0

        send_frame(sock, args.id, args.stop_frame, extended, "stop")
        time.sleep(args.settle_ms / 1000.0)
        print_frames(drain(sock, 0.2), args.id)

        send_frame(sock, args.id, args.zero_frame, extended, "set_zero")
        print(f"Waiting {args.zero_save_delay_ms} ms before save; official guide recommends about 1000 ms.")
        time.sleep(args.zero_save_delay_ms / 1000.0)
        zero_report = print_frames(drain(sock, 0.5), args.id)
        if not zero_report["zero_ack"]:
            print("WARN: did not observe rezero_pos ACK 41 01 00.")

        save_report = {"save_error": None}
        if not args.no_save:
            send_frame(sock, args.id, args.save_frame, extended, "save")
            print("Save command sent. Official guide recommends power-cycling the motor after conf_write.")
            time.sleep(args.settle_ms / 1000.0)
            save_report = print_frames(drain(sock, 0.5), args.id)
            if save_report["save_error"] is not None:
                print("ERROR: conf_write returned a failure/unsupported response.")
                print("       Rezero was acknowledged, but flash save did not complete through this direct-CAN command.")
                print("       Power-cycle and verify; if the offset remains, use the official SDK/CANboard conf_write path.")

        send_frame(sock, args.id, args.status_frame, extended, "status")
        time.sleep(args.settle_ms / 1000.0)
        status_report = print_frames(drain(sock, args.verify_timeout_ms / 1000.0), args.id)

        send_frame(sock, args.id, args.stop_frame, extended, "stop")
        time.sleep(0.1)
        print_frames(drain(sock, 0.2), args.id)

        current = latest_feedback(status_report)
        if current:
            pos_turns, _vel_rpm, _tqe_raw = current
            pos_deg = pos_turns * 360.0
            if abs(pos_deg) <= args.zero_tolerance_deg:
                print(f"OK: zero verified, current position {pos_deg:+.2f} deg.")
                return 0
            if save_report["save_error"] is not None:
                print(f"ERROR: current position is still {pos_deg:+.2f} deg and conf_write failed with code={save_report['save_error']}.")
                print("       The direct-CAN rezero command was acknowledged, but the saved/effective zero did not change.")
                print("       Next step: power-cycle the motor and run --status-only; if unchanged, use the official SDK/CANboard reset-zero flow.")
                return 4
            print(f"ERROR: status feedback was received, but current position is still {pos_deg:+.2f} deg.")
            print("       Recenter the yaw axis, power-cycle the motor after save if needed, then retry.")
            return 3
        print("WARN: zero/save frames were sent, but no 0x27 status feedback was observed.")
        print("      Try --status-only first, or power-cycle the motor after conf_write and retry verification.")
        return 2
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
