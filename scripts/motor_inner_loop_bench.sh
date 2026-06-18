#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IFACE="${YAW_CAN_IFACE:-can0}"

if command -v ip >/dev/null 2>&1; then
  sudo ip link set "$IFACE" down >/dev/null 2>&1 || true
  sudo ip link set "$IFACE" type can bitrate 1000000
  sudo ip link set "$IFACE" up
  ip -details link show "$IFACE"
fi

exec python3 "$ROOT_DIR/scripts/motor_inner_loop_bench.py" --iface "$IFACE" "$@"
