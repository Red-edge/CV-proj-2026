# CV-proj-2026

面向边缘端实时视觉伺服的计算机视觉项目。仓库当前包含两条互补路径：

- `cpp/`：当前主工程，负责高帧率取流、运动分析、YOLO ONNX/RKNN 推理、目标跟踪、状态滤波、MJPEG 预览、录制与 SocketCAN yaw 控制。
- `src/`：早期 Python 实验工程，保留 RealSense/T265、光流 ROI、YOLO 查看器、云台控制和性能估算脚本，主要用于算法验证和对照。

项目目标不是单独训练一个检测模型，而是把“人体检测”接入一个可运行、可观测、可调参的实时系统：相机输入 -> 运动 ROI/全帧检测 -> 目标关联 -> Kalman 状态估计 -> yaw 控制 -> 视频与 CSV 记录。

## 当前能力

- 支持 OpenCV 视频/摄像头回放、Hikrobot MVS、Aravis USB3 Vision / GenICam、RealSense/T265 等输入后端。
- 支持 YOLO ONNX Runtime 检测和 RKNN Runtime 检测；`models/rknn/` 中保留 WiderPerson YOLOv8n @640 的 FP16/INT8 RKNN 模型。
- C++ 管线已实现固定网格 LK 光流、全局中值流估计、ROI 生成、运动 blob 跟踪和可视化叠加。
- C++ 管线已实现 ByteTrack-style 多目标关联、primary track 选择、Kalman `TargetStateFilter` 和短时目标预测。
- 支持 `--livestream` HTTP MJPEG 实时预览，支持 `--record` 或 `--output` 录制处理后视频与遥测。
- 支持 Linux SocketCAN yaw 电机控制，包含 dry-run、软件零位、相对角限位、速度/加速度约束、30Hz 外环和 1kHz 内环配置。
- 保留 Python 侧脚本，用于离线验证、模型查看、RealSense/T265 实验和电机调试。

## 系统框架

```text
Camera / Video
  ├─ OpenCV file/camera
  ├─ Hikrobot MVS SDK
  ├─ Aravis USB3 Vision
  └─ RealSense / T265
        |
        v
Frame Source Abstraction
        |
        v
Motion Pipeline
  ├─ fixed-grid LK optical flow
  ├─ global motion estimation
  ├─ IMU / pose rotation compensation
  ├─ motion ROI generation
  └─ rendered overlay
        |
        v
Detector
  ├─ ONNX Runtime YOLO
  └─ RKNN Runtime YOLO
        |
        v
Tracking and State
  ├─ ByteTrack-style association
  ├─ primary target selection
  └─ Kalman TargetStateFilter
        |
        v
Output and Control
  ├─ MP4 recording
  ├─ CSV telemetry
  ├─ HTTP MJPEG livestream
  └─ SocketCAN yaw gimbal control
```

## 目录结构

```text
CV-proj-2026/
├── CMakeLists.txt              # 顶层 CMake 配置
├── CMakePresets.json           # Windows / Linux / Rock5B 构建预设
├── configs/                    # 常用运行配置
├── cpp/
│   ├── CMakeLists.txt          # C++ 主程序构建
│   ├── include/cvproj/         # C++ 模块接口
│   └── src/                    # C++ 实现
├── models/
│   └── rknn/                   # RKNN 模型与校准列表
├── project_docs/               # 设计说明、移植记录和运行手册
├── scripts/                    # 电机诊断、调参、限位扫描与启动脚本
├── src/                        # Python 实验版代码与 ONNX/PT 模型
├── third_party/
│   └── rknn/                   # RKNN Runtime 头文件与库
├── tools/
│   └── convert_onnx_to_rknn/   # ONNX 转 RKNN 工具链说明
├── motor-readme.md             # HTDW 电机协议与调试说明
└── requirements.txt            # Python 实验环境依赖
```

## C++ 核心模块

| 模块 | 主要文件 | 作用 |
|---|---|---|
| 程序入口与参数 | `cpp/src/main.cpp` | 解析命令行/配置文件，串联取流、检测、跟踪、录制、直播和 yaw 控制 |
| 视频源抽象 | `cpp/include/cvproj/frame_source.hpp` | 统一不同相机/视频后端的帧读取接口 |
| OpenCV 输入 | `cpp/src/opencv_video_source.cpp` | 用本地视频、USB 摄像头或图像序列做回放验证 |
| Hikrobot MVS | `cpp/src/hikrobot_mvs_source.cpp` | 通过海康 MVS SDK 接工业相机 |
| Aravis 输入 | `cpp/src/aravis_frame_source.cpp` | 通过 USB3 Vision / GenICam 标准协议接工业相机 |
| T265 输入 | `cpp/src/realsense_t265_source.cpp` | 读取 T265 单路图像与姿态/IMU 信息 |
| 运动分析 | `cpp/src/motion_pipeline.cpp` | 固定网格光流、全局运动、ROI、blob 和可视化 |
| ONNX 检测 | `cpp/src/yolo_onnx_detector.cpp` | 使用 ONNX Runtime 执行 YOLO 检测 |
| RKNN 检测 | `cpp/src/yolo_rknn_detector.cpp` | 使用 RKNN Runtime 执行边缘端 YOLO 检测 |
| 目标关联 | `cpp/src/bytetrack_tracker.cpp` | 按 IoU 与置信度进行 ByteTrack-style 关联 |
| 状态滤波 | `cpp/src/target_state_filter.cpp` | 对目标中心做 Kalman 滤波与短时预测 |
| 实时预览 | `cpp/src/mjpeg_server.cpp` | 输出 HTTP MJPEG 实时画面 |
| yaw 控制 | `cpp/src/socketcan_gimbal.cpp` | 通过 SocketCAN 发送 HTDW yaw 电机控制命令 |

## 依赖

### 通用依赖

- CMake 3.22+
- C++17 编译器
- OpenCV，需包含 `core`、`imgproc`、`video`、`videoio`、`highgui`、`dnn`、`imgcodecs`

### 可选依赖

- Aravis 0.8：USB3 Vision / GenICam 工业相机后端。
- Hikrobot MVS SDK：海康工业相机后端。
- librealsense2：RealSense/T265 后端。
- RKNN Runtime：RK3588/Rock5B NPU 推理。
- SocketCAN / `can-utils`：Linux 电机控制与 CAN 调试。

Linux / Rock5B 常用安装：

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev can-utils
sudo apt install -y libaravis-dev
```

macOS 常用安装：

```bash
brew install cmake opencv aravis
```

Python 实验环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 构建

### 默认构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

生成程序：

```text
build/cpp/cvproj_capture
```

### 关闭可选相机后端

没有 Aravis、MVS SDK 或 RealSense 时，可以先只构建 OpenCV 回放和基础处理链：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVPROJ_ENABLE_ARAVIS=OFF \
  -DCVPROJ_ENABLE_HIKROBOT_MVS=OFF \
  -DCVPROJ_ENABLE_REALSENSE=OFF
cmake --build build -j
```

### RKNN Runtime

仓库在 `third_party/rknn/` 下保留 RKNN Runtime 头文件与动态库。CMake 能找到 `rknn_api.h` 和 `librknnrt.so` 时会启用 `--detector rknn`。

## 运行

### 1. 视频回放验证

```bash
./build/cpp/cvproj_capture \
  --backend opencv \
  --source path/to/input.mp4 \
  --detector none \
  --output \
  --max-frames 600 \
  --headless
```

用于验证取流、光流、ROI、blob、叠加和录制。

### 2. ONNX 检测验证

```bash
./build/cpp/cvproj_capture \
  --backend opencv \
  --source path/to/input.mp4 \
  --detector yolo \
  --model src/widerperson_yolov8n_640.onnx \
  --det-input-size 640 \
  --det-conf 0.25 \
  --det-nms 0.45 \
  --output \
  --headless
```

用于在 PC 上验证 YOLO 前后处理、跟踪和滤波逻辑。

### 3. RKNN 检测验证

```bash
./build/cpp/cvproj_capture \
  --backend opencv \
  --source path/to/input.mp4 \
  --detector rknn \
  --model models/rknn/widerperson_yolov8n_640_fp16.rknn \
  --det-input-size 640 \
  --output \
  --headless
```

用于 Rock5B / RK3588 上验证 NPU 推理链路。

### 4. 工业相机实时预览

Aravis 路径：

```bash
./build/cpp/cvproj_capture \
  --backend aravis \
  --source 0 \
  --fps 240 \
  --exposure-us 3000 \
  --livestream \
  --output
```

Hikrobot MVS 路径：

```bash
export HIKROBOT_MVS_ROOT=/opt/MVS
./build/cpp/cvproj_capture \
  --backend hikrobot \
  --source 0 \
  --fps 240 \
  --livestream \
  --output
```

默认 MJPEG 地址为：

```text
http://<device-ip>:8080/
```

### 5. T265 + yaw dry-run

```bash
./build/cpp/cvproj_capture \
  --backend t265 \
  --detector rknn \
  --model models/rknn/widerperson_yolov8n_640_fp16.rknn \
  --yaw-enable \
  --yaw-dry-run \
  --yaw-control-mode visual-servo \
  --yaw-outer-rate-mode fixed \
  --yaw-outer-rate-hz 30 \
  --yaw-inner-rate-hz 1000 \
  --livestream \
  --output
```

dry-run 不向真实电机发命令，适合先检查坐标符号、目标选择、滤波和控制输出。

### 6. SocketCAN yaw 闭环

```bash
sudo ip link set can0 up type can bitrate 1000000

./build/cpp/cvproj_capture \
  --config configs/t265_yaw_rknn_fp16.conf \
  --yaw-enable \
  --yaw-can-iface can0 \
  --yaw-id 0x8001
```

实机闭环前建议先完成：

1. `--yaw-dry-run` 检查目标角误差方向。
2. `scripts/motor_reset_zero.sh` 或对应 Python 脚本设置软件零位。
3. `scripts/motor_limit_sweep.sh` 检查相对限位。
4. 降低 `--yaw-max-rpm` 和 `--yaw-max-accel-rpm-s` 做低速调参。

## 配置文件

`--config` 支持从 `.conf` / OpenCV YAML 文件读取常用参数。典型配置包括：

- `configs/rock5b_default.yaml`：Rock5B 默认部署参数。
- `configs/rock5b_live.conf`：Rock5B 实时预览配置。
- `configs/rock5b_live_fast25.conf`：较快实时配置。
- `configs/t265_yaw_real.conf`：T265 + 真实 yaw 控制。
- `configs/t265_yaw_rknn.conf`：T265 + RKNN 检测。
- `configs/t265_yaw_rknn_fp16.conf`：T265 + RKNN FP16 推荐候选链路。
- `configs/t265_yaw_widerperson.conf`：T265 + WiderPerson 检测模型。
- `configs/windows_default.yaml`：Windows / PC 调试默认配置。

命令行参数会覆盖配置文件中的同名参数。

## 输出文件

推荐使用 `--output` 让程序自动生成同一组输出：

```text
outputs/cvproj_<timestamp>.mp4
outputs/cvproj_<timestamp>.metrics.csv
outputs/cvproj_<timestamp>.targets.csv
```

视频用于查看叠加后的检测、ROI、track、滤波和控制状态；CSV 用于复盘 frame age、检测耗时、track 状态、yaw 命令、反馈频率等指标。需要手动命名时，使用 `--record <output.mp4>` 和 `--telemetry <output.csv>`。

## 模型文件

当前仓库保留的主要模型：

- `src/widerperson_yolov8n_640.onnx`：WiderPerson YOLOv8n @640 ONNX 模型。
- `src/widerperson_yolov8n.onnx`：WiderPerson YOLOv8n ONNX 模型。
- `src/best.onnx`：训练产物中的最佳 ONNX 模型副本。
- `models/rknn/widerperson_yolov8n_640_fp16.rknn`：RK3588 FP16 主候选模型。
- `models/rknn/widerperson_yolov8n_640_int8.rknn`：INT8 量化候选模型。
- `src/yolo11n.onnx`、`src/yolo11n.pt`：通用 YOLO11n 对照模型。
- `src/yolov8n*.onnx`、`src/yolov8n.pt`：YOLOv8n 尺寸对照与实验模型。

大模型文件会显著增加仓库体积，后续如果需要进一步瘦身，建议迁移到 release artifact 或模型下载脚本。

## 已验证结果

截至 2026-05-22 的本机记录：

- C++ 工程可通过 `cmake -S . -B build && cmake --build build -j8` 构建。
- macOS 可枚举海康 `MV-CS016-10UC`，序列号 `DB0178676`。
- Aravis 可发现并接入该相机，已完成真机取流与视频录制验证。
- OpenCV 回放链路输出过 `1280x720 @ 60fps` 的处理视频，离线主处理 FPS 约 `558.8`，端到端循环 FPS 约 `410.0`。
- ONNX 检测样例输出过 `1280x720 @ 60fps` 的处理视频，端到端循环 FPS 约 `345.4`。
- Aravis 真机处理参数记录为 `960x540`、`ExposureTime=3000us`、`target_fps=240`，端到端循环 FPS 约 `355.5`。
- RKNN FP16 + T265 + tracker 链路记录中，平均检测耗时约 `48.0 ms`，检测频率约 `19.96 Hz`，frame age P50/P90 约 `61.6 / 63.1 ms`，内环/反馈频率约 `996.6 Hz`，primary track 覆盖率约 `85.6%`。

## 已知限制

- 长时间 Aravis 在线推理压测仍需在目标硬件上继续补充。
- Hikrobot MVS SDK 后端依赖本机/目标板 SDK 安装与架构匹配。
- RKNN INT8 模型已保留，但仍需要更完整的精度、延迟和稳定性验证。
- Python 目录是实验版与对照实现，不再作为主工程入口。
- 当前仓库仍包含多个实验模型，适合验收复现，但不适合直接作为最小发布包。

## 建议验收顺序

1. `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
2. `cmake --build build -j`
3. 用 `--backend opencv --detector none` 验证基础处理链。
4. 用 `--detector yolo` 验证 PC 检测、跟踪和输出。
5. 在 Rock5B 上用 `--detector rknn` 验证 NPU 推理。
6. 用 `--livestream` 检查实时预览。
7. 用 `--yaw-dry-run` 验证控制符号和限位。
8. 最后接入 SocketCAN 和真实 yaw 电机做低速闭环。
