# CV-proj-2026

面向移动平台的高帧率视觉处理项目。当前仓库已经新增一条以 C++ 为主的实现路径，目标是把原先偏 Python 实验性质的“运动先验 ROI + 视觉处理”流程，迁移成可以直接接海康工业相机、跑高帧率、并录制处理结果的视频管线。

这次迁移的核心取舍是：

- 主目标从 `RealSense T265 + VIO` 转为 `Hikrobot MV-CS016-10UC + 高帧率图像处理`
- 保留项目真正的意图：在运动环境中快速找出值得处理的区域，并对目标做稳定的图像级处理与跟踪
- 当前 C++ 热路径不包含陀螺仪/VIO 相关的光流补偿，优先保证取流、处理、录制、工程结构和高 FPS 可达性
- 当前仓库同时支持 `Hikrobot MVS SDK` 与 `USB3 Vision / GenICam`
- 对“后续要接电机控制”的场景，优先建议走 `Hikrobot MVS SDK`
- `Aravis` 保留为无 SDK 场景下的标准协议回退路径

## 当前状态

- 已新增 `cpp/` CMake 工程
- 已新增 `HikrobotMvsSource`，按海康 MVS SDK 的 C API 方式接入
- 已新增 `OpenCvVideoSource` 回退源，可在没有相机和 SDK 时用摄像头或已有视频调试
- 已新增 `AravisFrameSource`，可直接通过 `USB3 Vision / GenICam` 标准协议接入工业相机
- 已迁移固定网格光流、全局中值流补偿、ROI 生成、运动 blob 跟踪、可视化叠加、视频录制
- 已新增 `YOLO ONNX` 可选检测支路，支持在 C++ 中复用运动 ROI 做低频目标检测
- 已从 `src/yolo11n.pt` 导出 `src/yolo11n.onnx`
- Python 代码仍保留在 `src/`，作为旧实验版本和模型参考

## 本机验证结果

截至 `2026-05-22`，这台 MacBook Air M4 上已经完成的验证有：

- `cmake -S . -B build && cmake --build build -j8` 可成功构建
- 已确认相机被 macOS 枚举为 `MV-CS016-10UC`，序列号 `DB0178676`
- 已确认标准协议栈 `Aravis` 可发现这台相机
- 已通过管理员权限方式成功跑通真机取流
- 使用仓库内现有视频跑通 C++ 处理链并输出新视频
- 输出文件: `outputs/cpp_processed_demo.mp4`
- 输出规格: `1280x720 @ 60fps, 600 frames, 10s`
- 离线回放时主处理链末次显示 FPS 约为 `558.8`
- 离线回放时端到端循环末次显示 FPS 约为 `410.0`
- 已跑通 `YOLO ONNX` 检测支路并输出 `outputs/cpp_yolo_demo.mp4`
- `YOLO ONNX` 样例输出规格: `1280x720 @ 60fps, 300 frames, 5s`
- `YOLO ONNX` 离线样例端到端循环末次显示 FPS 约为 `345.4`
- 已输出真机标准协议处理视频: `outputs/aravis_live_demo.mp4`
- 真机处理视频录制参数: `960x540`, `ExposureTime=3000us`, `target_fps=240`, `480 frames`
- 真机处理主链末次显示 FPS 约为 `516.8`
- 真机端到端循环末次显示 FPS 约为 `355.5`
- 已输出基于真实相机帧的推理视频: `outputs/aravis_live_yolo_demo_v2.mp4`

当前关于 `MVS` 的本机现状：

- 仓库里的 `hikrobot` 后端已经具备取流、增益/曝光设置、latest-frame 读取、livestream 遥测输出这些基础能力
- 但这台 MacBook Air M4 当前还没有检测到本地 `MVS SDK`
- 因此当前已经完成实测的视频与 demo 主要来自 `Aravis` 路径

尚未在本机完成的验证有：

- 基于 `Aravis` 的长时间稳定在线推理压测
- 云台控制迁移到 C++

## 项目结构

```text
CV-proj-2026/
├── CMakeLists.txt
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/cvproj/
│   │   ├── aravis_frame_source.hpp
│   │   ├── frame_source.hpp
│   │   ├── hikrobot_mvs_source.hpp
│   │   ├── motion_pipeline.hpp
│   │   └── opencv_video_source.hpp
│   └── src/
│       ├── aravis_frame_source.cpp
│       ├── hikrobot_mvs_source.cpp
│       ├── main.cpp
│       ├── motion_pipeline.cpp
│       ├── opencv_video_source.cpp
│       └── yolo_onnx_detector.cpp
├── src/
│   └── ... 旧 Python 实现
└── project_docs/
    └── ... 历史设计文档
```

## 重新理解后的项目意图

这个项目不该被理解成“把一个摄像头脚本换成另一个摄像头脚本”，而应该理解成：

1. 从高帧率相机稳定拿到图像
2. 在极低延迟下做运动感知
3. 用运动先验压缩后续处理范围
4. 输出可视化结果与可复盘的视频
5. 未来再把检测器、云台、IMU/VIO 优化逐步挂回主链

所以这次 C++ 迁移优先落地的是“能高帧率跑起来的主链”，而不是先把所有低速附加功能堆进去。

## 依赖

### macOS

- Xcode Command Line Tools
- CMake 3.22+
- OpenCV C++ 开发包
- Aravis

官方资源：

- Hikrobot MVS 下载页: <https://www.hikrobotics.com/en/machinevision/service/download/>
- Hikrobot USB3 Area Scan Camera User Manual: <https://www.hikrobotics.com/en2/source/vision/document/2023/8/18/UD31198B_USB3.0%20Area%20Scan%20Camera%20User%20Manual_V2.4.0_20230307.pdf>
- Aravis 官方文档: <https://aravisproject.github.io/docs/aravis-0.8/>
- Aravis GitHub: <https://github.com/AravisProject/aravis>

### 建议安装

```bash
brew install cmake opencv aravis
```

## 构建

```bash
cd /Users/rededge/Documents/workspace/CV-proj-2026
cmake -S . -B build
cmake --build build -j8
```

生成的主程序：

```text
build/cpp/cvproj_capture
```

如果构建日志里提示 `Aravis enabled`，说明标准协议后端已接入成功。`Hikrobot MVS SDK not found` 在当前路线下不是阻塞项。

## 运行

### 0. 控制场景的后端选择

如果你的目标是后续接入云台/电机控制，推荐优先顺序是：

1. `--backend hikrobot`
2. `--backend aravis`

原因很简单：

- `MVS` 是海康原生 SDK，控制链更适合作为主路径
- `Aravis` 在本机上已经验证能取流，但管理员态的 USB bootstrap 偶发不稳定
- 所以 `Aravis` 更适合作为取流验证或无 SDK 回退方案

### 0.1 安装 MVS SDK

当前仓库查找 MVS SDK 的路径包括：

- `HIKROBOT_MVS_ROOT/include`
- `HIKROBOT_MVS_ROOT/lib`
- `/opt/MVS`
- `/Applications/MVS.app/Contents`

当前这台机器上还没有找到：

- `MvCameraControl.h`
- `MvCameraControl` / `libMvCameraControl`

官方资料里可以确认两点：

- 海康下载中心当前仍提供 `MacOS` 过滤项  
  <https://www.hikrobotics.com/en/machinevision/service/download/>
- 较新的官方手册也写到 `MVS client software` 兼容 `64-bit MacOS`，并说明非 Windows 版本可通过技术支持获取安装包  
  <https://www.hikrobotics.com/en2/Hikrobotics/Machine%20Vision/01%20Product/%E5%B7%A5%E4%B8%9A%E9%9D%A2%E9%98%B5%E7%9B%B8%E6%9C%BA/CH/%E7%94%A8%E6%88%B7%E6%89%8B%E5%86%8C/UD19483B_Camera%20Link%20Area%20Scan%20Camera%20User%20Manual_V2.0.2_20200710.pdf>

安装完成后建议这样构建：

```bash
export HIKROBOT_MVS_ROOT="/Applications/MVS.app/Contents"
cmake -S . -B build
cmake --build build -j8
```

如果构建输出里看到 `Hikrobot MVS SDK enabled`，说明已经切到 MVS 主路径。

### 1. 用已有视频回放验证 C++ 管线

```bash
./build/cpp/cvproj_capture \
  --backend opencv \
  --source src/outputs/motion_guided_yolo_20260501_200409.mp4 \
  --fps 60 \
  --grid-points 128 \
  --motion-thresh 2.0 \
  --record outputs/cpp_processed_demo.mp4 \
  --max-frames 600 \
  --headless
```

这条命令适合在没有海康 SDK、没有真机接入时，先验证：

- 读帧
- 光流处理
- ROI 生成
- 运动目标框
- 处理后视频写盘

仓库当前已经用这条路径生成了：

```text
outputs/cpp_processed_demo.mp4
```

### 1.1 用 YOLO ONNX 路径验证“运动 ROI + 检测”

```bash
./build/cpp/cvproj_capture \
  --backend opencv \
  --source src/outputs/motion_guided_yolo_20260501_200409.mp4 \
  --fps 60 \
  --grid-points 128 \
  --motion-thresh 2.0 \
  --detector yolo \
  --model src/yolo11n.onnx \
  --detect-interval 6 \
  --det-conf 0.25 \
  --record outputs/cpp_yolo_demo.mp4 \
  --max-frames 300 \
  --headless
```

仓库当前已经用这条路径生成了：

```text
outputs/cpp_yolo_demo.mp4
```

### 2. 用普通摄像头调通实时链路

```bash
./build/cpp/cvproj_capture \
  --backend opencv \
  --source 0 \
  --width 1280 \
  --height 720 \
  --fps 60 \
  --grid-points 128 \
  --motion-thresh 2.0
```

### 3. 通过标准协议接入海康 MV-CS016-10UC

macOS `2026-05-22` 这台机器上的实测情况是：

- 普通用户态下，`Aravis/libusb` 能发现并控制相机，但 streaming interface claim 会失败
- 通过 macOS 管理员权限启动采集程序后，真机取流可正常运行

因此推荐直接这样跑：

```bash
osascript -e 'do shell script "\
/Users/rededge/Documents/workspace/CV-proj-2026/build/cpp/cvproj_capture \
  --backend aravis \
  --width 960 \
  --height 540 \
  --fps 240 \
  --exposure-us 3000 \
  --gain-db 12 \
  --target-luma 96 \
  --max-post-gain 6 \
  --gamma 0.8 \
  --pixel-format Mono8 \
  --grid-points 128 \
  --motion-thresh 2.0 \
  --record /Users/rededge/Documents/workspace/CV-proj-2026/outputs/aravis_live_demo.mp4 \
  --max-frames 480 \
  --headless" with administrator privileges'
```

如果你想显式绑定序列号，也可以附加：

```bash
--serial DB0178676
```

这条路径已经在本机生成了：

```text
outputs/aravis_live_demo.mp4
```

文件规格为：

- `960x540`
- `240fps`
- `480 frames`
- 时长 `2s`

### 3.5 通过 MVS 走控制主路径

一旦本机装好了 `MVS SDK`，推荐把实时链路切到：

```bash
./build/cpp/cvproj_capture \
  --backend hikrobot \
  --serial DB0178676 \
  --width 960 \
  --height 540 \
  --fps 240 \
  --exposure-us 3000 \
  --gain-db 12 \
  --target-luma 96 \
  --max-post-gain 6 \
  --gamma 0.8 \
  --pixel-format Mono8 \
  --grid-points 128 \
  --motion-thresh 2.0 \
  --livestream \
  --telemetry outputs/hikrobot_livestream_targets.csv \
  --headless
```

这条命令的定位是：

- 不保存视频
- 只输出结构化目标数据
- 给后续电机控制预留稳定接口

### 3.1 直接录制真机实时推理 demo

如果管理员态下 `Aravis` 的 USB bootstrap 正常，这条命令会在真机采集的同一次运行里，直接输出带推理框的视频：

```bash
cp src/yolo11n.onnx /tmp/cvproj_yolo11n.onnx

osascript -e 'do shell script "\
/Users/rededge/Documents/workspace/CV-proj-2026/build/cpp/cvproj_capture \
  --backend aravis \
  --serial DB0178676 \
  --width 960 \
  --height 540 \
  --fps 240 \
  --exposure-us 3000 \
  --gain-db 12 \
  --target-luma 96 \
  --max-post-gain 6 \
  --gamma 0.8 \
  --pixel-format Mono8 \
  --grid-points 128 \
  --motion-thresh 2.0 \
  --detector yolo \
  --model /tmp/cvproj_yolo11n.onnx \
  --detect-interval 8 \
  --det-conf 0.25 \
  --record /Users/rededge/Documents/workspace/CV-proj-2026/outputs/aravis_live_yolo_direct.mp4 \
  --max-frames 240 \
  --headless" with administrator privileges'
```

说明：

- 这里把模型先复制到 `/tmp`，是为了避免管理员子进程读取 `Documents/` 下 ONNX 文件时遇到权限问题
- `detect-interval=8` 的目的是把 `YOLO` 保持在低频支路，不和 `240fps` 的采集热路径硬绑定

### 3.2 基于真实相机帧生成推理 demo

在本机当前状态下，更稳定的方式是分两步：

1. 先用上一节命令录制真机处理视频
2. 再对这段真实相机视频做 C++ 推理回放

命令如下：

```bash
./build/cpp/cvproj_capture \
  --backend opencv \
  --source outputs/aravis_live_demo.mp4 \
  --fps 120 \
  --grid-points 128 \
  --motion-thresh 2.0 \
  --detector yolo \
  --model src/yolo11n.onnx \
  --detect-interval 6 \
  --det-conf 0.25 \
  --record outputs/aravis_live_yolo_demo_v2.mp4 \
  --max-frames 240 \
  --headless
```

仓库当前已经生成：

```text
outputs/aravis_live_yolo_demo_v2.mp4
```

这个文件基于同机同相机刚录下来的真实帧，输出规格为：

- `960x540`
- `120fps`
- `240 frames`
- 时长 `2s`

## 参数说明

- `--backend`: `opencv`、`aravis` 或 `hikrobot`
- `--source`: OpenCV 相机索引或视频路径
- `--serial`: 相机序列号过滤，当前真机为 `DB0178676`
- `--width`, `--height`: 采集分辨率
- `--fps`: 目标采集帧率
- `--exposure-us`: 曝光时间，单位微秒
- `--gain-db`: 相机侧增益，优先用于改善真机低照度画面
- `--pixel-format`: 相机像素格式，标准协议路径推荐 `Mono8`
- `--target-luma`: 软件侧目标亮度，默认 `96`
- `--max-post-gain`: 软件侧最大亮度放大倍数，默认 `6`
- `--gamma`: 软件侧 gamma，默认 `0.8`，用于提亮暗部
- `--no-auto-brightness`: 关闭软件自动提亮
- `--grid-points`: 固定网格光流点数，推荐 `64 / 128 / 256`
- `--motion-thresh`: 运动判定阈值，单位像素/帧
- `--detector`: `none` 或 `yolo`
- `--model`: ONNX 模型路径，当前已提供 `src/yolo11n.onnx`
- `--det-conf`: 检测置信度阈值
- `--det-nms`: NMS 阈值
- `--detect-interval`: 每隔多少帧跑一次检测，建议高帧率模式下取 `4~8`
- `--record`: 保存处理后视频
- `--record-fps`: 输出视频 fps；如果不指定，默认沿用 `--fps`
- `--telemetry`: 目标遥测输出 csv 路径
- `--livestream`: 直播模式，不保存视频，只输出结构化目标信息
- `--max-frames`: 跑多少帧后退出，便于 benchmark
- `--headless`: 不打开窗口，专门用于录制和压测

## 处理帧率与录制帧率

当前主循环已经把“处理节奏”和“录制节奏”分开：

- 相机侧 `--fps` 只表示采集目标帧率
- 录制侧 `--record-fps` 只表示输出视频的封装帧率
- 实际处理时，程序每轮都会优先取“当前能拿到的最新帧”，而不是强制把积压帧逐张处理完

这意味着：

- 如果处理链比相机慢，程序会自动跳过旧帧，尽量让输出结果跟上最新画面
- 如果需要拿目标框中心、ROI、检测数量等参数做后续控制或分析，应该使用录制视频同名 `.csv` sidecar，而不是假设它一定等间隔对应 `--fps`

当你传入：

```bash
--record outputs/demo.mp4
```

程序会同时生成：

```text
outputs/demo.csv
```

其中每一行都对应一帧真实处理结果，包含：

- `source_timestamp_s`
- `source_delta_ms`
- `effective_fps`
- `pipeline_fps`
- `loop_fps`
- `processing_ms`
- `roi`
- `target_box`
- `target_cx`
- `target_cy`
- `detection_count`

所以后续如果要做控制闭环、时序回放、性能分析，应该以 `.csv` 中的真实时间和真实处理频率为准。

## Livestream 模式

如果你要给后续电机控制预留接口，推荐直接使用 `livestream` 模式：

- 不保存 mp4
- 继续跑实时采集、运动处理、可选检测与目标跟踪
- 每帧输出结构化目标信息到 `csv`

示例：

```bash
osascript -e 'do shell script "\
/Users/rededge/Documents/workspace/CV-proj-2026/build/cpp/cvproj_capture \
  --backend aravis \
  --serial DB0178676 \
  --width 960 \
  --height 540 \
  --fps 240 \
  --exposure-us 3000 \
  --gain-db 12 \
  --target-luma 96 \
  --max-post-gain 6 \
  --gamma 0.8 \
  --pixel-format Mono8 \
  --grid-points 128 \
  --motion-thresh 2.0 \
  --detector yolo \
  --model /tmp/cvproj_yolo11n.onnx \
  --detect-interval 8 \
  --livestream \
  --telemetry /Users/rededge/Documents/workspace/CV-proj-2026/outputs/livestream_targets.csv \
  --headless" with administrator privileges'
```

如果不加 `--telemetry`，默认会写到：

```text
outputs/livestream_targets.csv
```

### 遥测字段

`livestream` 输出按“每帧每目标一行”记录，主要字段包括：

- `track_id`: 跨帧稳定目标编号，适合后续控制环使用
- `target_index`: 当前帧内的目标顺序，`0` 一般表示主目标
- `is_primary`: 是否主目标
- `source`: 目标来源，当前可能是 `detector` 或 `motion`
- `x,y,w,h`: 像素坐标框
- `cx,cy`: 目标中心点像素坐标
- `norm_cx,norm_cy`: 归一化中心坐标，范围约 `0~1`
- `offset_x,offset_y`: 相对图像中心的像素偏差，后续电机控制通常直接用这两个量
- `source_timestamp_s`: 本帧真实时间戳
- `source_delta_ms`: 与上一处理帧之间的真实时间间隔
- `effective_fps`: 实际处理帧率估计

如果某一帧没有可用目标，程序仍然会写一行空目标记录，`track_id=-1`，这样后续控制逻辑可以明确区分“无目标”而不是“丢日志”。

## 低照度画质建议

如果真机画面明显偏黑，不要第一反应只拉长曝光，因为在 `240fps` 附近曝光时间很快会吃掉可用帧率。当前推荐按这个顺序调：

1. 先保持 `--exposure-us 3000`
2. 把 `--gain-db` 提到 `12` 或 `15`
3. 保持软件增强开启，也就是不要加 `--no-auto-brightness`
4. 如果画面还是偏暗，再把 `--target-luma` 从 `96` 提到 `110`
5. 只有在确实需要时，才把曝光继续往 `3500~3800us` 拉

当前主程序默认已经会对输入图像做一层轻量增强：

- 相机侧增益
- CLAHE 局部对比度增强
- 自动亮度拉升
- gamma 提亮暗部

这层增强会同时作用于可视化、录像和后续运动处理，因此比“只在显示端提亮”更适合当前项目。

## 当前 C++ 处理链

当前主程序每帧执行：

1. 采集一帧 BGR 图像
2. 生成或复用固定网格光流点
3. 用 `calcOpticalFlowPyrLK` 计算前后帧光流
4. 用全局中值流做一次图像级背景运动抵消
5. 得到残余运动强度
6. 用滑窗方式生成 ROI
7. 用运动点热区生成 blob，并取最大运动目标框
8. 可选地在 ROI 上跑低频 `YOLO ONNX` 人体检测
9. 叠加 ROI、目标框、检测框、FPS、处理时延
10. 可选写入 mp4

这条链路是为了满足高 FPS 主路径，不包含：

- T265 VIO
- 陀螺仪参与的光流补偿
- 云台控制

当前唯一仍未迁完的是云台控制和陀螺仪/VIO 相关补偿。检测器已经有了 C++ 可选支路，但默认不放在 240fps 热路径中。

## 真机实测

下面这些数据都来自这台 MacBook Air M4 上对 `MV-CS016-10UC` 的实测：

| 模式 | 分辨率 | 曝光 | 目标 FPS | 实测结果 |
|---|---:|---:|---:|---|
| `arv-camera-test` | `1440x1080` | `5000us` | `165` | `164~166 fps`, `0` 失败 |
| `arv-camera-test` | `1280x720` | `5000us` | `240` | `193~194 fps`, `0` 失败 |
| `arv-camera-test` | `960x540` | `3000us` | `240` | `240~241 fps` 可达 |
| `cvproj_capture` 真机处理 | `960x540` | `3000us` | `240` | 端到端录制成功，主链末次 `355.5 fps` |

说明：

- `5000us` 曝光会把可达帧率压到 `200fps` 左右
- 把曝光降到 `3000us` 后，`960x540` 配置可达到 `240fps`
- 真机 `YOLO` 推理支路不建议直接跟 `240fps` 热路径绑定，建议低频触发

## 240 FPS 调优建议

想逼近 `240fps`，建议优先按这个顺序调：

1. 先用 `--headless` 压测，不要显示窗口
2. 先只录制短片段，避免长时间磁盘 IO 影响统计
3. 优先使用 `64` 或 `128` 个光流点
4. 保持单色或低复杂度像素格式，减少 SDK 转色成本
5. 必要时把相机输出裁到业务真正需要的分辨率
6. 把后续检测器留在低频支路，而不是放在每帧热路径

## 已知限制

- 当前仓库里还没有海康 MVS SDK，本机首次构建时如果没装 SDK，会自动退化成 OpenCV 路径
- 当前路线已经不依赖海康 MVS SDK，真机主链走的是 `Aravis` 标准协议
- macOS `2026-05-22` 下普通用户态对 USB3 Vision streaming interface 的访问不稳定，因此当前 README 默认用 `osascript ... with administrator privileges` 启动真机采集
- macOS `2026-05-22` 下管理员态的 `Aravis/libusb` 也偶发 `Failed to bootstrap USB device '---'`，所以“真机采集 + 在线推理”目前不是每次都能一次成功
- 当前 C++ 主链已经迁移了“高帧率运动处理 + 可选 YOLO ONNX 检测”部分，但没有把 Python 里的云台逻辑一比一移植
- 管理员会话下 `Aravis` 的设备枚举名称偶尔不稳定，因此最稳妥的真机录制方式是一次授权后直接起程序并写视频

## 建议的验证顺序

1. `brew install opencv aravis`
2. 构建 `cvproj_capture`
3. 先用已有 mp4 跑回放并生成处理视频
4. 用 `osascript + --backend aravis` 接真机
5. 先验证 `1440x1080 @ 165fps`
6. 再切到 `960x540 + 3000us + 240fps`
7. 最后再决定是否把云台闭环补回 C++ 主链

## Legacy Python

`src/` 目录下的 Python 代码暂时保留，原因是：

- 它仍然是旧实验流程的参考实现
- 里面的 YOLO、RealSense、云台逻辑对后续 C++ 补完功能有参考价值
- 在 C++ 真机链路完全稳定前，保留旧版有助于对比结果

如果你后面要继续推进，我建议下一阶段直接做两件事：

1. 把海康真机的曝光、像素格式、帧率配置做成可配置 profile
2. 把云台控制链也迁到 C++
