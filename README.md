# 🎯 Motion-Guided YOLO + Gimbal Tracking System

> 基于 64/128/256 点固定网格光流检测的运动引导目标跟踪系统，集成 YOLO 检测与二轴云台实时联动。

---

## 📋 目录

- [🔧 系统要求](#-系统要求)
- [📦 环境配置](#-环境配置)
- [🗂️ 项目结构](#️-项目结构)
- [🚀 快速启动](#-快速启动)
- [⚙️ 参数说明](#️-参数说明)
- [🎮 云台控制集成](#-云台控制集成)
- [🧪 测试与调试](#-测试与调试)
- [🔍 故障排查](#-故障排查)
- [🛠️ 开发指南](#️-开发指南)

---

## 🔧 系统要求

### 硬件配置

| 组件 | 推荐配置 | 最低配置 | 说明 |
|:---|:---|:---|:---|
| **处理器** | RK3588 / Jetson Orin Nano | MacBook M1/M2/M3 | RK3588 为量产首选 |
| **内存** | ≥4GB | ≥2GB | YOLO 推理需 1.5GB+ |
| **摄像头** | RealSense T265 / USB UVC | 任意 720p 摄像头 | 需支持 MJPEG/YUV422 |
| **云台电机** | HTDW-5047-36-NE ×2 | 任意支持速度控制的舵机 | 需支持位置/速度闭环 |
| **CAN 接口** | PCAN-USB / SocketCAN | USB-TTL (模拟) | 用于电机通信 |
| **存储** | ≥16GB eMMC/SD | ≥8GB | 模型 + 系统 + 日志 |

### 软件环境

| 平台 | 操作系统 | Python | 关键依赖 |
|:---|:---|:---|:---|
| **Mac (开发)** | macOS 12+ | 3.8~3.11 | `python-can`, `ultralytics`, `opencv-python` |
| **RK3588 (部署)** | Ubuntu 22.04 (aarch64) | 3.8~3.10 | 同上 + `rknn-toolkit2` (可选) |
| **Jetson (备选)** | JetPack 5.1+ | 3.8~3.10 | 同上 + `jetson-utils` (可选) |

---

## 📦 环境配置

### 1️⃣ 创建虚拟环境（推荐）

```bash
# Mac / Linux
python3 -m venv cv-env
source cv-env/bin/activate

# Windows (PowerShell)
python -m venv cv-env
.\cv-env\Scripts\Activate.ps1
```

### 2️⃣ 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# macOS 额外：安装 PCAN 驱动支持
brew install libpcan  # 或从 PEAK-System 官网下载 .kext

# RK3588 额外：安装 RKNN 支持（如需量化加速）
pip install rknn-toolkit2-lite --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/simple
```

### 3️⃣ 验证安装

```bash
# 检查 Python 依赖
python -c "import cv2, numpy, ultralytics, can; print('✅ All imports OK')"

# 检查摄像头
python -c "import cv2; cap=cv2.VideoCapture(0); print('✅ Camera:', cap.isOpened())"

# 检查 CAN 接口 (需连接硬件)
python -c "import can; print(can.list_interfaces(bustype='pcan'))"
```

---

## 🗂️ 项目结构

```
CV-proj-2026/
├── src/
│   ├── main.py                 # 主入口：检测 + 跟踪 + 云台联动
│   ├── camera_mapper.py        # 像素坐标 → 云台角度映射
│   ├── gimbal_controller.py    # HTDW 电机控制封装
│   ├── htdw_motor_ctrl.py      # 独立电机测试脚本
│   ├── requirements.txt        # Python 依赖列表
│   │
│   ├── data_source/
│   │   ├── __init__.py
│   │   ├── realsense_wrapper.py  # T265/摄像头封装
│   │   └── video_fallback.py     # OpenCV 摄像头回退
│   │
│   ├── recognition/
│   │   ├── __init__.py
│   │   └── yolo_detector.py      # YOLO 推理封装 (支持.pt/.mlpackage)
│   │
│   └── tracking/
│       ├── __init__.py
│       └── multi_object_tracker.py  # Kalman+IoU 多目标跟踪
│
├── models/                     # 模型存放目录
│   ├── yolo11n.pt             # PyTorch 格式
│   ├── yolo11n.mlpackage      # CoreML 格式 (Mac 加速)
│   └── best_int8.rknn         # RKNN 量化格式 (RK3588)
│
├── outputs/                    # 输出目录 (自动创建)
│   ├── *.mp4                  # 录制视频
│   └── *.png                  # 截图
│
├── README.md                  # 本文档
└── LICENSE                    # 开源协议
```

---

## 🚀 快速启动

### 🔹 模式 1：纯视觉检测（无云台）

```bash
# Mac 测试：摄像头 + YOLO + 光流引导
python src/main.py \
  --use-yolo \
  --source 0 \
  --device mps \
  --resize-input \
  --yolo-model models/yolo11n.mlpackage \
  --motion-thresh 2.0 \
  --num-motion-points 128

# 按键控制：
#   q = 退出 | r = 重置 | s = 截图 | v = 录制 | +/- = 调整运动阈值
```

### 🔹 模式 2：云台跟踪（Mock 模拟）

```bash
# 无硬件测试云台逻辑
python src/main.py \
  --use-yolo \
  --mock-gimbal \
  --source 0 \
  --device mps \
  --resize-input \
  --h-fov 45.0 \
  --v-fov 30.0 \
  --yolo-model models/yolo11n.mlpackage
```

### 🔹 模式 3：真实云台联动（RK3588 + PCAN）

```bash
# 1. 确保硬件连接：
#    - 摄像头 → USB/MIPI
#    - HTDW 电机 → PCAN-USB → RK3588 USB
#    - 激光笔 → 云台出轴

# 2. 加载 CAN 驱动 (RK3588)
sudo modprobe pcan
sudo ip link set can0 up type pcan bitrate 1000000

# 3. 运行主程序
python src/main.py \
  --use-yolo \
  --source realsense \
  --device mps \
  --resize-input \
  --can-channel can0 \
  --h-fov 44.8 \
  --v-fov 29.6 \
  --yolo-model models/yolo11n.mlpackage \
  --motion-thresh 2.5 \
  --num-motion-points 256
```

---

## ⚙️ 参数说明

### 📷 摄像头参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--source` | `"0"` | 摄像头源：`0`=USB, `realsense`=T265, 路径=视频文件 |
| `--width` / `--height` | `848×800` | RealSense 输出分辨率 |
| `--resize-input` | `False` | 是否将输入缩放到 720p 以内 |
| `--input-max-width/height` | `1280×720` | 缩放目标分辨率 |

### 🤖 YOLO 参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--use-yolo` | `False` | 启用 YOLO 检测 |
| `--device` | `"cpu"` | 推理设备：`cpu`/`mps`/`cuda` |
| `--yolo-model` | `"yolo11n.pt"` | 模型路径，支持 `.pt`/`.mlpackage`/`.rknn` |
| `--conf-thresh` | `0.25` | 检测置信度阈值 (ROI 内) |
| `--bg-conf` | `0.45` | 背景区域置信度阈值 |
| `--yolo-nms-iou` | `0.45` | NMS IoU 阈值 |

### 🔍 运动检测参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--motion-method` | `"lk"` | 运动检测方法：`lk`=光流, `diff`=帧差 |
| `--motion-thresh` | `2.0` | 运动幅值阈值 (像素/帧) |
| `--num-motion-points` | `128` | 固定网格点数：`64`/`128`/`256` |
| `--blur-ksize` | `21` | 注意力模糊核大小 (奇数) |

### 🎮 云台参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--mock-gimbal` | `False` | 启用云台模拟模式 (无硬件) |
| `--can-channel` | `"PCAN_USBBUS1"` | CAN 通道名 (Mac) 或 `can0` (Linux) |
| `--h-fov` / `--v-fov` | `45.0` / `30.0` | 摄像头水平/垂直视场角 (°) |

### 📦 输出参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--record` | `True` | 启用视频录制 |
| `--output` | `"outputs/motion_guided_yolo.mp4"` | 输出路径 |
| `--no-timestamp` | `False` | 禁用文件名时间戳 |
| `--record-fps` | `30.0` | 录制帧率 |

---

## 🎮 云台控制集成

### 电机协议说明 (HTDW-5047-36-NE)

```python
# 控制帧格式 (8 bytes, Extended ID)
# ID: 0x8000 | motor_id (e.g., 0x8001=Yaw, 0x8002=Pitch)
# Data:
#   [0:2]  = 0x07, 0x35          # 命令头
#   [2:4]  = speed_raw (int16)   # 速度 = RPM / 0.015
#   [4:6]  = torque_raw (int16)  # 力矩限制 (推荐 2000)
#   [6:8]  = pos_raw (uint16)    # 位置占位 (0x8000)

# 反馈帧解析 (ID: 0x8000|id, Data[0]==0x27)
#   Data[4:6] = vel_raw (int16)  # 实际速度 = raw × 0.00025 × 60 RPM
```

### 独立测试电机

```bash
# 运行独立控制脚本 (键盘控制)
python src/htdw_motor_ctrl.py

# 按键说明：
#   E = 使能电机 | W/S = 加速/减速 | 空格 = 归零
#   D = 停止 | B = 刹车 | Q = 退出
```

### 集成到主程序

`gimbal_controller.py` 已封装双轴控制，主程序自动调用：

```python
# 初始化
mapper = CameraAngleMapper(h_fov=45.0, v_fov=30.0, img_w=1280, img_h=720)
gimbal = GimbalController(channel="PCAN_USBBUS1", mock=False)

# 每帧跟踪逻辑
if detections:
    # 选择最大目标
    largest = max(detections, key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]))
    cx, cy = (largest["bbox"][0]+largest["bbox"][2])/2, (largest["bbox"][1]+largest["bbox"][3])/2
    
    # 像素→角度
    yaw_off, pitch_off = mapper.pixel_to_angle(cx, cy)
    
    # 云台目标 = 当前角度 + 偏差
    gimbal.set_target_angles(
        gimbal.current_yaw + yaw_off,
        gimbal.current_pitch + pitch_off
    )
    
    # 更新控制环 (50Hz)
    gimbal.update(dt=0.02)
```

### 安全保护

- ✅ **软限位**：`±20° Yaw / ±15° Pitch` (代码内)
- ✅ **急停指令**：按 `q` 或 `Ctrl+C` 自动发送 `0x01,0x00,0x00`
- ✅ **力矩限制**：默认 `2000` (约 60% 最大扭矩)，防堵转
- ⚠️ **激光安全**：建议添加头部/眼部区域屏蔽逻辑（见 `yolo_detector.py`）

---

## 🧪 测试与调试

### 1️⃣ 摄像头测试

```bash
# 查看可用摄像头
python -c "import cv2; [print(f'Camera {i}:', cv2.VideoCapture(i).isOpened()) for i in range(4)]"

# 测试分辨率/帧率
python -c "
import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
print(f'Resolution: {cap.get(3)}x{cap.get(4)} @ {cap.get(5)} FPS')
"
```

### 2️⃣ FOV 标定

```bash
# 1. 在 5m 处放置 2.68m 宽标尺 (对应 30°理论覆盖)
# 2. 拍摄图像并运行标定脚本
python tools/calibrate_fov.py --image test.jpg --distance 5.0 --known-width 2.68

# 输出示例：
# 📐 Measured HFOV: 44.8° (target: 45°)
# ✅ FOV within tolerance (±2°)
```

### 3️⃣ 延迟测试

```bash
# 测量端到端延迟 (摄像头→检测→云台指令)
python tools/latency_test.py \
  --video test.mp4 \
  --model models/yolo11n.mlpackage \
  --mock-gimbal

# 输出：
# 📊 End-to-end latency: 82.3 ± 5.1 ms (P95: 91ms)
```

### 4️⃣ 性能基准

```bash
# 对比不同模型/分辨率的推理速度
python tools/benchmark.py \
  --video outputs/test.mp4 \
  --models yolo11n.pt yolo11n.mlpackage \
  --devices cpu mps \
  --frames 100
```

---

## 🔍 故障排查

### ❌ 摄像头无法打开

```bash
# 1. 检查设备权限 (macOS)
#    系统设置 → 隐私与安全性 → 摄像头 → 允许终端/Python

# 2. 检查设备节点 (Linux)
ls -l /dev/video*
sudo chmod 666 /dev/video0  # 临时权限

# 3. 尝试不同源
python main.py --source 1  # 尝试第二个摄像头
python main.py --source realsense  # 尝试 T265
```

### ❌ CAN 通信失败

```bash
# 1. 检查驱动 (macOS)
system_profiler SPUSBDataType | grep -i pcan
# 若无输出：安装 PEAK-System macOS 驱动

# 2. 检查通道名
python -c "import can; print(can.list_interfaces(bustype='pcan'))"
# 输出示例：['PCAN_USBBUS1', 'PCAN_USBBUS2']

# 3. 检查比特率
# 确保与电机固件一致 (默认 1Mbps)
python main.py --can-channel PCAN_USBBUS1 --can-bitrate 1000000
```

### ❌ YOLO 推理慢/崩溃

```bash
# 1. 检查设备
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# 2. 尝试不同模型格式
#    .pt (PyTorch) → .mlpackage (CoreML, Mac 加速) → .rknn (RK3588)

# 3. 降低分辨率
python main.py --resize-input --input-max-width 640 --input-max-height 360

# 4. 启用量化 (RK3588)
yolo export model=yolo11n.pt format=rknn imgsz=640 data=coco.yaml
```

### ❌ 云台不动/抖动

```bash
# 1. 检查电机使能
#    运行 htdw_motor_ctrl.py，按 E 使能，观察是否响应

# 2. 检查控制环频率
#    确保 main.py 中 gimbal.update(dt=0.02) 与帧率匹配

# 3. 调整 PID 参数 (gimbal_controller.py)
#    Kp↑ = 响应快但易振荡 | Ki↑ = 消除静差但易超调 | Kd↑ = 抑制振荡

# 4. 检查机械背隙
#    手动转动云台，感受是否有空程；如有，需软件补偿或机械预紧
```

---

## 🛠️ 开发指南

### 添加新运动检测方法

```python
# 1. 在 MotionDetectionPipeline._compute_motion_intent 中添加分支
if self.motion_method == "your_method":
    # 实现你的算法
    return magnitudes_array

# 2. 在 parse_args() 中添加选项
p.add_argument("--motion-method", choices=["lk", "diff", "your_method"], default="lk")
```

### 支持新电机协议

```python
# 1. 修改 gimbal_controller.py 中的 _pack_and_send()
def _pack_and_send(self, can_id: int, value: float):
    # 根据你的协议重新打包数据
    data = your_protocol_encode(value)
    self._send_can_frame(can_id, data)

# 2. 更新 HTDWMotor 类的 send_control_frame()
```

### 添加新可视化面板

```python
# 1. 在 _draw_results() 中调用新函数
self._draw_your_panel(vis_frame, results)

# 2. 实现绘制逻辑
def _draw_your_panel(self, vis_frame, results):
    # 使用 cv2.putText/rectangle 等绘制
    pass
```

### 模型量化部署 (RK3588)

```bash
# 1. 导出 ONNX
yolo export model=yolo11n.pt format=onnx imgsz=640

# 2. RKNN 量化
rknn-toolkit2 \
  --model yolo11n.onnx \
  --dataset calib_set.txt \
  --quantization RKNN_INT8 \
  --output yolo11n_int8.rknn

# 3. 修改 yolo_detector.py 加载逻辑
if model_path.endswith('.rknn'):
    from rknnlite.api import RKNNLite
    self.rknn = RKNNLite()
    self.rknn.load_rknn(model_path)
    self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
```

---

## 📄 许可证

本项目采用 **MIT License**，允许商业使用、修改和分发，但需保留原作者声明。

```
Copyright © 2026 [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

> 💡 **最后提示**：  
> 1. **先软件后硬件**：先用 `--mock-gimbal` 调通视觉逻辑，再接真实云台  
> 2. **安全第一**：激光指向人体时务必添加头部/眼部屏蔽逻辑  
> 3. **标定先行**：实测摄像头 FOV 和云台零位，避免角度映射误差  
> 4. **渐进优化**：先跑通 64 点 + yolo11n，再逐步增加点数/更换模型  

如有问题，请提交 Issue 或联系维护者。🚀