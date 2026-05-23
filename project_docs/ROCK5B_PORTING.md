# ROCK 5B Porting Notes

## Goal

Keep the Windows C++ runtime portable to `Linux ARM64 + RK3588/RKNN` without redesigning the vision/control pipeline.

## Reserved integration points

- `backend=rknn`
- `--model-rknn`
- `configs/rock5b_default.yaml`
- `models/rknn/`
- `tools/convert_onnx_to_rknn/`

## Known migration deltas

- Replace CUDA detector backend with RKNN runtime
- Replace Windows MVS SDK deployment with Linux MVS SDK deployment
- Replace Windows PCAN deployment with Linux PCAN deployment
- Validate video encoding backend on-device
- Prefer `--headless` for long-running deployments

## Current status

The repository now contains the cross-platform C++ interface boundaries for this migration, but the RKNN backend remains a placeholder implementation until the board-specific runtime package is integrated.
