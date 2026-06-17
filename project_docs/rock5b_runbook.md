# Rock5B visual auto-aim runbook

This runbook verifies the Rock5B path for realtime preview, annotated recording,
target telemetry, and yaw motor control.

## 1. Build

```bash
cd ~/workspace/CV-proj-2026
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

If camera SDKs are not installed yet, build the OpenCV-only verification binary:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCVPROJ_ENABLE_ARAVIS=OFF -DCVPROJ_ENABLE_HIKROBOT_MVS=OFF
cmake --build build -j$(nproc)
```

## 2. No-motor verification

Use this before connecting the yaw motor. It exercises the same annotated frame
path used by the browser preview and recording output.

```bash
./build/cpp/cvproj_capture \
  --backend opencv \
  --source outputs/aravis_live_demo.mp4 \
  --fps 60 \
  --grid-points 128 \
  --motion-thresh 2.0 \
  --detector yolo \
  --model src/yolo11n.onnx \
  --detect-interval 8 \
  --downsample-width 640 \
  --downsample-height 480 \
  --denoise bilateral \
  --denoise-kernel 5 \
  --output \
  --record-fps 15 \
  --livestream \
  --http-preview 0.0.0.0:8080 --preview-fps 15 \
  --yaw-dry-run \
  --yaw-max-angle 10 \
  --headless
```

Open:

```text
http://<rock5b-ip>:8080/
```

Pass criteria:

- Browser shows the annotated realtime stream.
- `outputs/rock5b_dryrun_overlay.mp4` contains the same overlays.
- `outputs/rock5b_dryrun_overlay.csv` contains `yaw_target_deg`, `yaw_rpm`,
  `yaw_limited`, and `yaw_status`.
- `yaw_target_deg` never exceeds `-10..10`.

## 3. SocketCAN setup

```bash
sudo ip link set can0 down || true
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0
```

Optional bus monitor:

```bash
candump can0
```

## 4. Motor direction check

The image-to-angle convention follows `src/camera_mapper.py`:

- target right of image center -> positive yaw target
- target left of image center -> negative yaw target

Start with low speed:

```bash
./build/cpp/cvproj_capture \
  --backend hikrobot \
  --serial DB0178676 \
  --width 1440 \
  --height 1080 \
  --fps 240 \
  --exposure-us 3000 \
  --gain-db 12 \
  --pixel-format BayerRG8 \
  --grid-points 128 \
  --motion-thresh 2.0 \
  --detector yolo \
  --model src/yolo11n.onnx \
  --detect-interval 8 \
  --downsample-width 640 \
  --downsample-height 480 \
  --denoise bilateral \
  --denoise-kernel 5 \
  --output \
  --record-fps 15 \
  --livestream \
  --http-preview 0.0.0.0:8080 --preview-fps 15 \
  --yaw-enable \
  --yaw-can-iface can0 \
  --yaw-id 0x8001 \
  --yaw-max-angle 10 \
  --yaw-max-rpm 120 \
  --yaw-kp 12 \
  --yaw-hfov 70 \
  --headless
```

If the camera turns away from the target, rerun with:

```bash
--yaw-invert
```

## 5. Safety notes

- The yaw target is clamped in software to `-10..10` degrees.
- The controller sends `0 rpm` when no target is tracked.
- On process exit, the controller sends `0 rpm` and a stop frame.
- Keep `--yaw-max-rpm` low until mechanical direction and CAN wiring are confirmed.

### 前端实时查看

全功能运行时加上 `--http-preview 0.0.0.0:8080 --preview-fps 15` 后，Rock5B 会提供 MJPEG 页面：

```bash
open -na "Google Chrome" --args --new-window "http://192.168.2.55:8080/"
# 或者直接在浏览器打开 http://192.168.2.55:8080/
```

页面显示的是已经叠加 YOLO 框、目标 ID、FPS、录制状态和 yaw 状态的画面；录制文件保存同一份叠加后的画面。

## 5. 录制、降噪与前端帧率

- `--record-fps 15`: 输出叠加后视频为 15fps。
- `--denoise bilateral --denoise-kernel 7`: 默认降噪，降低真机画面噪点并保留检测边缘。
- `--http-preview 0.0.0.0:8080 --preview-fps 15`: 浏览器实时查看叠加画面，前端按 15fps 推流。
- 本次 Rock5B 上 YOLOv8n/YOLO11 ONNX 需要使用本机编译的 OpenCV 4.10：运行前设置 `LD_LIBRARY_PATH=$HOME/opt/opencv-4.10.0/lib`。

录制/前端默认建议：相机保持高帧率采集，输出视频和浏览器预览保持 15fps，避免写盘和网络预览拖慢控制链。

- `--downsample-width 640 --downsample-height 480`: 对采集到的完整相机帧做整帧下采样，例如 `1440x1080 -> 640x480`；这不是 ROI 裁剪，后续光流、YOLO、前端和录制都基于下采样后的完整画面。

- 降噪顺序：整帧下采样后先降噪，再做亮度/CLAHE/gamma 增强，避免增强步骤先放大噪点。

- `--downsample-width` / `--downsample-height`: 对采集到的完整相机帧做整帧 resize。当前 MV-CS016-10UC 读到的原生/最大分辨率为 `1440x1080`，推荐采集原生分辨率后下采样到 `640x480`。

## 自动输出文件命名

日常运行只需要加 `--output`，不需要手写视频或 CSV 文件名。程序会用启动时间生成同一组对应文件：

```text
outputs/cvproj_YYYYMMDD_HHMMSS.mp4
outputs/cvproj_YYYYMMDD_HHMMSS.csv
outputs/cvproj_YYYYMMDD_HHMMSS.targets.csv
```

其中 `.mp4` 是带叠加内容的视频，`.csv` 是处理性能/ROI/yaw 指标，`.targets.csv` 是逐目标框和 yaw 目标遥测。三个文件共享同一个时间戳后缀，便于一一对应。

## MV-CS016-10UC 像素格式

MV-CS016-10UC 是彩色相机，`C` 表示 Color。当前型号规格为 Sony IMX273、`1440x1080`，满分辨率典型输出为 `BayerRG8`。程序在 Aravis 后端中会把 `BayerRG8` 解 Bayer 转成 BGR，再进入下采样、降噪、YOLO、前端和录制。

推荐参数：

```bash
--pixel-format BayerRG8
```

`Mono8` 仍可作为灰度调试格式，但不是这台彩色相机的推荐运行格式。

### PixelFormat 校验

程序启动时会打印 `Aravis PixelFormat: ...`。如果请求 `--pixel-format BayerRG8` 但相机拒绝写入，程序会直接失败并显示实际 PixelFormat，避免误以为彩色运行但实际仍是灰度。`--pixel-format auto` 只用于明确接受相机当前格式的调试场景。

## 安全退出与 MP4 保存

程序已经捕获 `SIGINT` / `SIGTERM`。运行中按一次 `Ctrl-C` 会请求主循环停止，然后正常执行 `VideoWriter.release()`、关闭 CSV、停止预览和 yaw 控制，MP4 会写入完整索引信息。

正确退出方式：

```bash
# 推荐：按一次 Ctrl-C，然后等待出现 finalizing / Processed frames 输出
Ctrl-C
```

不要用 `kill -9`、直接关闭 SSH 窗口、断电或拔设备来结束录制；这些方式会跳过 MP4 收尾，可能导致文件不可读。需要固定时长测试时，也可以用 `--max-frames <N>` 让程序自然退出。

## 红蓝通道校正

如果现场看到红色物体显示成蓝色，说明当前相机输出/解 Bayer 后的 R/B 通道与 OpenCV BGR 约定相反。启动命令中加入：

```bash
--swap-rb
```

该开关会在下采样、降噪、YOLO、前端预览和录制之前统一交换红蓝通道。如果加入后红色恢复正常，就保留这个参数；如果颜色本来正常，则不要加。

## 低噪声推荐参数

画面噪点主要会被自动提亮、CLAHE 和 gamma 放大。低噪声优先时推荐：

```bash
--target-luma 82 \
--max-post-gain 2.0 \
--gamma 1.0 \
--clahe-clip 1.0 \
--denoise bilateral \
--denoise-kernel 7 \
--denoise-sigma-color 75 \
--denoise-sigma-space 11 \
--denoise-passes 2
```

`--max-post-gain` 就是软件提亮的 alpha 上限；数值越高，暗部越亮，但噪点也越明显。如果现场仍然噪，优先把 `--gain-db` 降到 `6~8`，再把 `--max-post-gain` 降到 `1.5`，最后才继续增大滤波核。

## Config 启动方式

日常运行不需要再输入长命令，参数已经写入：

```text
configs/rock5b_live.conf
```

启动：

```bash
cd ~/workspace/CV-proj-2026
./scripts/run_rock5b_live.sh
```

也可以指定另一份配置：

```bash
./scripts/run_rock5b_live.sh configs/your_config.conf
```

配置文件格式是 `key=value`，例如 `gain-db=8`、`denoise=off`、`detect-interval=1`、`output=true`、`http-preview=0.0.0.0:8080`。解析器只会把真实开关项的 `true/false` 转成命令行 flag，数值参数里的 `0/1` 和字符串参数里的 `off/auto` 会保留为参数值。命令行参数仍可覆盖配置，例如：

```bash
./build-opencv410/cpp/cvproj_capture --config configs/rock5b_live.conf --gain-db 6
```


## USB-CAN 自动拉起

`scripts/run_rock5b_live.sh` 会在启动视觉程序前自动配置 CAN：

```bash
sudo ip link set can0 down || true
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

默认参数：`can0 @ 1Mbps`。临时关闭自动 CAN 配置：

```bash
CVPROJ_SETUP_CAN=0 ./scripts/run_rock5b_live.sh
```

如果接口名或波特率不同：

```bash
CVPROJ_CAN_IFACE=can1 CVPROJ_CAN_BITRATE=1000000 ./scripts/run_rock5b_live.sh
```

注意：当前默认配置文件仍是 `yaw-dry-run=true`，只拉起 CAN，不发送真实电机控制。要真实控制电机，需要把配置切到 `yaw-enable=true`，并设置 `yaw-can-iface=can0`、`yaw-id=0x8001`、`yaw-max-rpm` 等参数。


脚本会把配置文件后面的额外参数透传给程序，因此可以这样做短测：

```bash
./scripts/run_rock5b_live.sh configs/rock5b_live.conf --max-frames 30
```


## 识别帧率优化

Rock5B CPU 上要把识别循环推到 25Hz 以上，使用已经实测通过的 fast25 配置：

```bash
./scripts/run_rock5b_live.sh configs/rock5b_live_fast25.conf
```

短测建议：

```bash
./scripts/run_rock5b_live.sh configs/rock5b_live_fast25.conf --max-frames 300
```

这份配置做了几件事：

- 仍然从相机原生 `1440x1080` 采集，然后整帧下采样到 `480x360`，不是裁切。
- 使用固定输入尺寸的 `src/yolov8n_160.onnx`，并设置 `detect-interval=1`，也就是每个处理帧都跑 YOLO。
- 关闭双边降噪和 CLAHE，把 CPU 留给 YOLO；需要低噪声画面时改回 `configs/rock5b_live.conf`。
- 前端和录制降到 `10fps`，避免 MJPEG 推流和 MP4 编码拖慢识别循环。

已在 Rock5B + MV-CS016-10UC + Aravis 上用 `--max-frames 300` 验证：

```text
Last reported end-to-end loop FPS: 25.6355
Last reported detection FPS: 45.5651
```

`Last reported end-to-end loop FPS` 是更严格的实际识别更新节奏；`Last reported detection FPS` 是单次 YOLO 推理耗时换算出的检测器吞吐。metrics CSV 也包含 `detection_ms,detection_fps`。`record-fps` 现在会真正节流写视频，不再每个处理帧都编码成 MP4。
