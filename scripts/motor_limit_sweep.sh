#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CVPROJ_CAN_IFACE="${CVPROJ_CAN_IFACE:-can0}"
CVPROJ_CAN_BITRATE="${CVPROJ_CAN_BITRATE:-1000000}"

if ip link show "$CVPROJ_CAN_IFACE" >/dev/null 2>&1; then
  echo 123 | sudo -S ip link set "$CVPROJ_CAN_IFACE" down >/dev/null 2>&1 || true
  echo 123 | sudo -S ip link set "$CVPROJ_CAN_IFACE" type can bitrate "$CVPROJ_CAN_BITRATE"
  echo 123 | sudo -S ip link set "$CVPROJ_CAN_IFACE" up
  ip -details -statistics link show "$CVPROJ_CAN_IFACE" | sed -n 1,16p
else
  echo "ERROR: CAN interface $CVPROJ_CAN_IFACE not found" >&2
  exit 1
fi

echo 123 | sudo -S python3 scripts/motor_limit_sweep.py "$@"
