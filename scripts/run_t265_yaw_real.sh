#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -gt 0 ]]; then
  CONFIG_PATH="$1"
  shift
else
  CONFIG_PATH="configs/t265_yaw_real.conf"
fi

CVPROJ_SETUP_CAN="${CVPROJ_SETUP_CAN:-1}"
CVPROJ_CAN_IFACE="${CVPROJ_CAN_IFACE:-can0}"
CVPROJ_CAN_BITRATE="${CVPROJ_CAN_BITRATE:-1000000}"

export LD_LIBRARY_PATH="$HOME/opt/opencv-4.10.0/lib:$HOME/opt/librealsense/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

if [[ "$CVPROJ_SETUP_CAN" == "1" ]]; then
  if ip link show "$CVPROJ_CAN_IFACE" >/dev/null 2>&1; then
    echo 123 | sudo -S ip link set "$CVPROJ_CAN_IFACE" down >/dev/null 2>&1 || true
    echo 123 | sudo -S ip link set "$CVPROJ_CAN_IFACE" type can bitrate "$CVPROJ_CAN_BITRATE"
    echo 123 | sudo -S ip link set "$CVPROJ_CAN_IFACE" up
    ip -details link show "$CVPROJ_CAN_IFACE" | sed -n 1,8p
  else
    echo "WARN: CAN interface $CVPROJ_CAN_IFACE not found; continuing without CAN setup" >&2
  fi
fi

echo 123 | sudo -S env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  ./build-opencv410/cpp/cvproj_capture --config "$CONFIG_PATH" "$@"
