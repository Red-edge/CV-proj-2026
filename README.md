# CV-proj-2026

Motion-guided detection and central-control prototype for:

- Hikrobot MVS industrial camera input
- YOLO-based target detection
- gimbal command output
- future integration of lidar, radar, and fire-control modules

The repository originally started as a "motion prior + YOLO + gimbal linkage" experiment. It now has a clearer central-control entrypoint for Windows and MVS-based deployment.

The repository now also includes a new C++ central-control runtime under `cpp/`, designed for:

- Windows x64 first deployment
- future ROCK 5B / Linux ARM64 migration
- independent `headless` and `record` switches
- asynchronous capture / flow / detect / render pipeline

## Current Status

The following paths are working in the current repository:

- MVS official SDK camera enumeration
- MVS live frame capture from `MV-CS016-10UC`
- processed-video recording
- CPU benchmark with ONNX Runtime
- GPU benchmark with ONNX Runtime CUDA provider
- extension hooks for future actuator and sensor modules

Recommended entrypoints:

- `central_control`
  New C++ runtime built from `cpp/`
- `src/main_control.py`
  Legacy Python central-control program kept as migration reference
- `src/onnx_gpu_benchmark.py`
  Dedicated benchmark and processed-video recorder for CPU/GPU comparison

## Hardware and Software

Validated on:

- Windows
- Python 3.12
- Hikrobot MVS SDK installed on host
- camera model: `MV-CS016-10UC`
- NVIDIA GPU available for ONNX Runtime CUDA benchmarking

Expected host-side prerequisites:

1. Hikrobot `MVS` SDK is installed
2. `MVCAM_COMMON_RUNENV` is available in the environment
3. If you want CUDA benchmarking, install the Python packages used by ONNX Runtime CUDA

## Repository Layout

```text
CV-proj-2026/
├─ README.md
├─ requirements.txt
├─ project_docs/
│  ├─ SYSTEM_BLOCK_DIAGRAM.md
│  └─ WINDOWS_MVS_RUNBOOK.md
└─ src/
   ├─ main.py
   ├─ main_control.py
   ├─ onnx_gpu_benchmark.py
   ├─ control_center.py
   ├─ camera_mapper.py
   ├─ data_source/
   │  ├─ base.py
   │  ├─ mvs_camera.py
   │  └─ realsense_wrapper.py
   ├─ recognition/
   │  └─ yolo_detector.py
   ├─ tracking/
   │  └─ multi_object_tracker.py
   └─ outputs/
```

## C++ Runtime

### Build

See [project_docs/CPP_WINDOWS_BUILD.md](project_docs/CPP_WINDOWS_BUILD.md).

### Example run: show window, no recording

```powershell
.\build\windows-msvc-release\Release\central_control.exe `
  --config .\configs\windows_default.yaml `
  --show-window `
  --no-record `
  --duration-sec 30 `
  --backend cuda
```

### Example run: headless + record

```powershell
.\build\windows-msvc-release\Release\central_control.exe `
  --config .\configs\windows_default.yaml `
  --headless `
  --record-rendered `
  --duration-sec 30 `
  --backend cuda
```

### Required runtime switches

- `--headless`
  run inference without a live GUI window
- `--record-rendered`
  save the processed output video
- `--no-record`
  disable recording even if the config file enables it

### Backend values

- `--backend cuda`
- `--backend cpu`
- `--backend rknn`
- `--backend auto`

## Quick Start

### 1. Create a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Optional packages for ONNX GPU benchmarking

If you want the ONNX Runtime CUDA path used in this repository:

```powershell
.\.venv\Scripts\python.exe -m pip install onnx onnxruntime-gpu
.\.venv\Scripts\python.exe -m pip install nvidia-cublas-cu12
.\.venv\Scripts\python.exe -m pip install nvidia-cuda-runtime-cu12
.\.venv\Scripts\python.exe -m pip install nvidia-cudnn-cu12
.\.venv\Scripts\python.exe -m pip install nvidia-cufft-cu12
```

### 3. Verify MVS camera detection

```powershell
.\.venv\Scripts\python.exe .\src\main_control.py --list-sources
```

Expected output looks like:

```text
Detected MVS cameras:
  [0] model=MV-CS016-10UC serial=DB0178676
```

## Main Control Program

### MVS camera + mock gimbal + YOLO

```powershell
.\.venv\Scripts\python.exe .\src\main_control.py `
  --source mvs `
  --use-yolo `
  --device cpu `
  --mock-gimbal `
  --resize-input `
  --input-max-width 1280 `
  --input-max-height 720
```

### Real gimbal control

Remove `--mock-gimbal` and set the correct PCAN channel:

```powershell
.\.venv\Scripts\python.exe .\src\main_control.py `
  --source mvs `
  --use-yolo `
  --device cpu `
  --can-channel PCAN_USBBUS1 `
  --h-fov 45.0 `
  --v-fov 34.0
```

### Record processed video from the main control pipeline

```powershell
.\.venv\Scripts\python.exe .\src\main_control.py `
  --source mvs `
  --use-yolo `
  --device cpu `
  --mock-gimbal `
  --record-output `
  --output-fps 20 `
  --output-path ".\src\outputs\processed_main.mp4"
```

Important arguments in `main_control.py`:

- `--source`
  `mvs`, `realsense`, webcam index like `0`, or a video path
- `--mvs-index`
  select camera by index
- `--mvs-serial`
  bind to a specific industrial camera by serial number
- `--mvs-exposure-us`
  manual exposure for MVS camera
- `--mvs-gain`
  manual gain for MVS camera
- `--mvs-frame-rate`
  target acquisition frame rate
- `--record-output`
  record processed frames
- `--output-path`
  processed video output path

## GPU Benchmark

This repository includes a dedicated ONNX benchmark path for consistent CPU/GPU comparison.

### Export ONNX model

If `src/yolo11n.onnx` does not exist yet:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO(r'.\src\yolo11n.pt').export(format='onnx', imgsz=640, opset=12, simplify=False)"
```

### Run CUDA benchmark and record processed video

```powershell
.\.venv\Scripts\python.exe .\src\onnx_gpu_benchmark.py `
  --source mvs `
  --provider cuda `
  --model .\src\yolo11n.onnx `
  --max-frames 120 `
  --warmup-frames 5 `
  --output-fps 20 `
  --output-path ".\src\outputs\benchmark_cuda_120.mp4"
```

### Run CPU benchmark and record processed video

```powershell
.\.venv\Scripts\python.exe .\src\onnx_gpu_benchmark.py `
  --source mvs `
  --provider cpu `
  --model .\src\yolo11n.onnx `
  --max-frames 120 `
  --warmup-frames 5 `
  --output-fps 20 `
  --output-path ".\src\outputs\benchmark_cpu_120.mp4"
```

### Example measured results

Measured on the validated machine with the same camera and the same ONNX model:

- CUDA: `avg_fps=94.29`, `avg_infer_ms=6.91`
- CPU: `avg_fps=38.08`, `avg_infer_ms=22.08`

This is roughly:

- `2.48x` higher end-to-end throughput
- `3.19x` lower average inference latency

## Architecture

See [project_docs/SYSTEM_BLOCK_DIAGRAM.md](project_docs/SYSTEM_BLOCK_DIAGRAM.md) for the system block diagram.

High-level flow:

```text
MVS Camera -> Frame Source -> Motion/Detection Pipeline -> Angle Mapping -> ControlCenter -> Gimbal
```

Key modules:

- `src/data_source/mvs_camera.py`
  official MVS SDK wrapper for Hikrobot industrial cameras
- `src/control_center.py`
  central registry for sensors, targets, and actuators
- `src/camera_mapper.py`
  maps image coordinates to yaw/pitch offsets
- `src/main_control.py`
  central-control loop
- `src/onnx_gpu_benchmark.py`
  reproducible benchmark and processed-video recording

## Extension Interfaces

The current code intentionally leaves clean upgrade points for later subsystem integration.

### Future sensors

Add new sensor modules that publish:

- `SensorSnapshot(name="lidar", payload=...)`
- `SensorSnapshot(name="radar", payload=...)`
- `SensorSnapshot(name="imu", payload=...)`

### Future actuators

Add new actuator modules that consume:

- `ControlCommand(name="fire_control", values=...)`
- `ControlCommand(name="turret", values=...)`

### Central dispatch

All future modules should integrate through `ControlCenter`, not by hard-coupling directly into the camera or YOLO classes.

## Troubleshooting

### MVS camera cannot be opened

Check:

- the camera is not occupied by another process
- MVS GUI tools are closed
- you are selecting the correct `--mvs-index` or `--mvs-serial`

### RealSense import breaks startup

This repository now treats RealSense as optional. If `pyrealsense2` is not installed, use:

```powershell
--source mvs
```

or:

```powershell
--source 0
```

### CUDA provider falls back to CPU

Check:

1. `onnxruntime-gpu` is installed
2. NVIDIA driver is available
3. the required CUDA/cuDNN Python packages are installed
4. your benchmark is using `--provider cuda`

## Additional Docs

- [project_docs/WINDOWS_MVS_RUNBOOK.md](project_docs/WINDOWS_MVS_RUNBOOK.md)
- [project_docs/SYSTEM_BLOCK_DIAGRAM.md](project_docs/SYSTEM_BLOCK_DIAGRAM.md)
- [project_docs/CPP_WINDOWS_BUILD.md](project_docs/CPP_WINDOWS_BUILD.md)
- [project_docs/ROCK5B_PORTING.md](project_docs/ROCK5B_PORTING.md)

## Notes

- `src/main.py` is kept for compatibility with the original prototype path
- `src/main_control.py` should be treated as the primary control entrypoint going forward
- `cpp/` is the new long-term runtime direction for deployment and performance work
- benchmark scripts save processed videos into `src/outputs/`
