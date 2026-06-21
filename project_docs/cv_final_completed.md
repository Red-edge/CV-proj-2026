# 基于 YOLOv8n 与 RKNN 的边缘端行人检测和视觉伺服云台跟踪系统

生成日期：2026-06-19  
整理完善日期：2026-06-21  
项目类型：计算机视觉模型训练、边缘部署与 yaw 单轴视觉伺服系统实现  
最终模型：WiderPerson YOLOv8n @640  
最终实机候选链路：T265 单路鱼眼 + RKNN FP16 + ByteTrack + Kalman + IMU 旋转补偿预测 + 30Hz 外环 + 1kHz 内环

---

## 摘要

本项目的初步想法来自室内真人 CS 场景：当敌方目标从墙角或遮挡物后突然出现时，人眼发现、判断和转向瞄准之间存在明显反应时间，容易错过最初的处理窗口。因此项目尝试用“视觉相机感知 + 边缘端识别 + yaw 单轴自动归中”的方式做辅助处理，并搭建了一台样机设备。样机由一台 Rock5B（RK3588）作为边缘计算平台，一台手头已有的 Intel RealSense T265 相机提供单路鱼眼视觉图像和 IMU 信息，一台 HTDW 电机驱动 yaw 轴转动，共同组成从目标出现、视觉识别、状态估计到水平归中的闭环原型。项目并不是单纯训练一个离线检测模型，也不是将 YOLO 检测框直接发送给电机，而是将“墙角突发人体目标辅助归中”扩展为一个由感知、目标关联、状态估计、运动补偿、控制执行和运行可观测性共同组成的实时系统。

在模型层，项目基于 WiderPerson 数据集训练单类别 `person` 检测器。原始 COCO 预训练 YOLOv8n 在 WiderPerson validation set 上的 mAP@0.5:0.95 仅为 `0.2240`，说明直接迁移到密集行人、小目标和局部可见人体场景时存在明显领域差异。经过 WiderPerson fine-tuning 后，W1 模型的 mAP@0.5:0.95 提升到 `0.4477`；进一步将输入尺寸从 `416` 提高到 `640` 后，W2 模型的 mAP@0.5:0.95 达到 `0.4880`，成为本项目最终推荐检测模型。W2 已完成 PyTorch、ONNX FP32、RKNN FP16 和 RKNN INT8 多格式转换，其中 RKNN FP16 是当前 RK3588 实机部署的主候选。

在系统层，项目由“感知输入层、运动 ROI 预测层、YOLO 检测层、目标跟踪层、状态估计层、IMU 旋转补偿层、yaw 控制执行层、可观测记录层”共同构成。检测结果进入 ByteTrack-style tracker 生成稳定 primary track，再进入 Kalman TargetStateFilter 得到滤波中心点和短时预测状态；T265 IMU 的角速度用于 tracker 旋转补偿预测，避免让易受纹理与噪声影响的光流直接决定目标预测；控制侧采用 30Hz 视觉外环与 1kHz 电机内环拆频，并引入软件零位、相对限位、最大速度、最大加速度和反馈保护。

最新 `T265 + RKNN FP16 + IMU 旋转补偿 tracker` 实测中，平均检测耗时约 `48.0 ms`，检测 FPS 约 `19.96 Hz`，frame age P50/P90 为 `61.6 / 63.1 ms`，内环与反馈频率约 `996.6 Hz`，primary track 覆盖率为 `85.6%`。与 CPU ONNX 路径约 `220 ms` 的检测耗时相比，RKNN FP16 对实时闭环更友好，使系统具备继续实机调参所需的延迟条件。

本项目的最终结论是：WiderPerson YOLOv8n @640 在当前数据和模型规模下取得最优检测精度；RKNN FP16 是当前最可靠的边缘端部署形式；ByteTrack + Kalman + IMU 旋转补偿预测 比单帧 YOLO 框直驱更适合视觉伺服；30Hz 外环与 1kHz 内环拆频使控制节奏、安全限位和运行证据更加稳定。

**关键词：** YOLOv8n；WiderPerson；行人检测；RKNN；RK3588；ByteTrack；Kalman Filter；T265；IMU 运动补偿；视觉伺服；云台跟踪；SocketCAN

![系统总体结构](assets/system_architecture.png)

---

## 1. 总体架构与问题定义

### 1.1 项目背景

本项目的出发点不是一般意义上的行人检测论文任务，而是一个具体的室内真人 CS 样机需求：当敌方目标从墙角、门口、掩体后方突然出现时，人需要先注意到画面变化，再判断目标位置，再转动身体或设备进行瞄准。这个链路依赖人的视觉注意力和反应速度，在近距离、高紧张度和遮挡突发场景中很容易出现滞后。因此，最初的问题被提出为：能否用一套轻量视觉设备在目标刚出现时先完成“发现目标并把 yaw 轴大致转向目标”的辅助动作，从而降低人的反应压力。

围绕这一想法，项目搭建了一台边缘端视觉伺服样机。硬件上，样机由三部分组成：第一，Rock5B 作为边缘计算平台，核心 SoC 为 RK3588，用于运行 Linux、相机输入、YOLO/RKNN 推理、目标跟踪和控制逻辑；第二，使用手头已有的 Intel RealSense T265 相机采集视觉图像，当前主链路使用其中一路鱼眼灰度图像，同时利用 T265 的 IMU/pose 信息辅助解释相机自运动；第三，使用 HTDW 电机驱动 yaw 轴，使设备具备水平转向能力。软件上，项目将相机取流、检测、跟踪、滤波、yaw 控制、视频叠加和 CSV 记录组织成一条可复盘的实时链路。

该场景下的目标检测也不是“识别图像中所有行人”这么宽泛。室内真人 CS 中真正重要的是：当墙角出现人体、上半身、肩部、侧身或局部可见目标时，系统能否尽早给出一个可用于 yaw 归中的目标中心。也就是说，检测框不必用于身份识别或精细语义理解，而要服务于控制链路中的“目标大致方向估计”。因此，本项目将检测目标定义为“图像中当前可见、可用于跟踪和 yaw 控制的人体区域”。在训练中，这一目标通过 WiderPerson 数据集中的 pedestrians 与 partially-visible persons 合并为单类 `person` 来实现。

项目的工程目标也不是设计复杂大模型，而是在手头硬件条件下判断这台样机是否满足基本使用需求。具体来说，需要回答三类问题：第一，相机画质和像素数量是否足以在室内真人 CS 的常见距离上看清人体目标；第二，Rock5B/RK3588 的算力是否足以在可接受延迟内运行检测和跟踪；第三，模型识别率、跟踪连续性和控制频率是否足以支撑“辅助归中”的基本闭环。

### 1.2 使用场景与基本需求

本项目关注的典型场景可以抽象为：设备处于室内通道、房间入口或掩体附近，目标从墙角突然进入相机视野；系统需要在较短时间内发现目标，估计目标水平位置，并驱动 yaw 轴向目标方向转动。这个场景与普通离线检测有明显差别：它不只关心一张图是否能框出人，还关心目标出现后的第一时间响应、连续帧目标稳定性、控制符号是否正确以及设备是否能在限位内安全运动。

因此，样机的基本需求可整理为以下几条。

第一，相机输入要覆盖足够宽的视野。墙角突发目标不一定从画面中央出现，使用 T265 鱼眼相机的好处是视场角大，能更早捕获侧向进入画面的目标；缺点是同样的 848×800 像素会被分散到很大的视场里，远距离目标的像素高度较低。

第二，目标在常见室内距离内要有足够像素。真人 CS 室内交战距离通常更接近 3-10 m，而不是几十米远距离。若目标是全身或上半身暴露，T265 的像素数量仍可能支撑检测；若目标只露出很小一部分，或者目标距离超过 10 m，鱼眼图像中目标像素会迅速下降，检测稳定性会明显变差。

第三，识别和控制延迟要低于人的反应链路。样机不需要先于所有人类反应完成精确瞄准，但至少应在目标出现后快速给出方向辅助。当前系统把检测、跟踪和滤波结果输入 30Hz yaw 外环，电机侧以内环接近 1kHz 的频率执行控制，目标是让视觉链路的主要延迟集中在几十毫秒到一百毫秒量级，而不是数百毫秒以上。

第四，识别率要优先保证召回和目标连续性。墙角突发场景中，短时漏检会比单帧误差更影响体验，因为目标一旦丢失，yaw 控制会进入 hold 或切换目标。因此，系统不能直接使用单帧 YOLO 框直驱电机，而要通过 ByteTrack-style 关联和 Kalman 滤波维持 primary track 的连续性。

第五，控制必须受限且可复盘。样机涉及真实 yaw 轴运动，因此必须具备 dry-run、软件零位、相对限位、最大速度、最大加速度、反馈超时和日志记录。只有视频和 CSV 都能复盘时，才能判断问题来自相机、模型、跟踪、控制符号还是电机执行。

### 1.3 相机画质、像素数量与最远可识别距离评估

T265 的优势是体积小、带鱼眼图像和 IMU，适合做快速样机；限制是它并不是为高分辨率彩色识别设计的相机。根据 Intel RealSense T265 产品规格和数据手册，T265 使用双鱼眼跟踪传感器，单路图像分辨率为 848×800，视场角约为 163°，且没有 RGB Sensor；仓库中的 `RealSenseT265Config` 也默认使用 `848×800`。这意味着系统可以获得很宽的观察范围，但单位角度上的像素密度有限。

为了评估它是否满足室内真人 CS 的基本需求，可以做一个近似估算。将鱼眼有效圆形视场近似看作 800 像素直径、163° 视场，则角分辨率约为：

$$
\frac{800}{163}\approx4.9\ \text{px/deg}
$$

若人体可见高度为 $H$，距离为 $D$，其角高度近似为：

$$
\theta=2\arctan\left(\frac{H}{2D}\right)
$$

对应原始鱼眼图像中的像素高度约为：

$$
p \approx \theta_{\text{deg}}\times4.9
$$

由于 YOLO 输入为 640，848×800 图像进入模型时还会经过缩放和 letterbox，按高度从 800 缩放到约 640 估计，网络输入中的目标高度约为原始像素高度的 0.8 倍。按全身可见高度 1.7 m、上半身/躯干可见高度 0.8 m、墙角局部可见高度 0.5 m 估算，可得到下表。

| 距离 | 全身 1.7 m 原图/网络像素 | 上半身 0.8 m 原图/网络像素 | 局部 0.5 m 原图/网络像素 | 评估 |
|---:|---:|---:|---:|---|
| 3 m | 155 / 124 px | 75 / 60 px | 47 / 37 px | 目标像素充足，适合检测和控制 |
| 5 m | 95 / 76 px | 45 / 36 px | 28 / 22 px | 全身、上半身较可用，局部目标开始依赖画质 |
| 8 m | 60 / 48 px | 28 / 22 px | 18 / 14 px | 全身仍可尝试，上半身处于临界区，局部目标不稳定 |
| 10 m | 48 / 38 px | 22 / 18 px | 14 / 11 px | 全身勉强可检，上半身和局部目标风险较高 |
| 12 m | 40 / 32 px | 19 / 15 px | 12 / 9 px | 已接近小目标区域，只适合较完整目标 |
| 15 m | 32 / 25 px | 15 / 12 px | 9 / 7 px | 基本不适合稳定识别局部突发目标 |

由此可见，T265 对本项目的适配结论是：在 3-5 m 近距离室内墙角场景中，样机有较好的像素基础；在 5-8 m 范围内，若目标暴露为全身或较大上半身，仍有机会满足辅助归中；超过约 8-10 m 后，目标在鱼眼图像中的像素高度明显下降，尤其是只露出半身或肩部时，识别稳定性不足。因此，T265 可以支撑“近中距离真人 CS 样机验证”，但不能被理解为适合远距离精确识别的最终相机。

这一结论也解释了项目后续为什么没有只追求更复杂的控制算法。若相机端已经不能提供足够目标像素，控制侧再复杂也无法可靠弥补。因此，基本需求评估必须把相机距离上限放在第一层：当前样机更适合验证 3-8 m 室内突然出现目标的辅助归中，而不是 10 m 以上局部人体的稳定识别。

### 1.4 算力、识别率与实时性评估

Rock5B 使用 RK3588 平台，板端具备 NPU 推理能力，Radxa 对 ROCK 5B 的说明中也标注其内置 NPU 支持 INT4/INT8/INT16/FP16 混合运算，算力最高约 6 TOPS。对本项目而言，关键不是理论 TOPS，而是 YOLOv8n WiderPerson @640 在真实链路中的端到端表现。报告中的实测数据显示，CPU ONNX 路径检测耗时约 220 ms，检测频率约 4 Hz，作为闭环控制明显偏慢；RKNN FP16 路径平均检测耗时约 48.0 ms，检测 FPS 约 19.96 Hz，frame age P50/P90 为 61.6 / 63.1 ms，已经进入可做实机调参的范围。

从识别率看，最终推荐模型 W2（WiderPerson YOLOv8n @640）在验证集上 Precision 为 0.8138，Recall 为 0.6436，mAP@0.5 为 0.7592，mAP@0.5:0.95 为 0.4880。这个结果说明两点：第一，WiderPerson fine-tuning 相比 COCO baseline 明显提升了行人/人体目标识别能力；第二，Recall 仍不是满分，意味着实机中仍会出现漏检，所以必须依赖跟踪和滤波维持控制连续性，而不能把单帧检测结果直接当作可靠控制目标。

结合样机需求，可以给出如下判断。

| 维度 | 当前结果 | 是否满足基本需求 | 依据 |
|---|---|---|---|
| 相机覆盖范围 | T265 约 163° 鱼眼视场 | 满足近距离墙角突发目标捕获 | 宽视场有利于提前看到侧向进入目标 |
| 像素与距离 | 3-5 m 充足，5-8 m 临界可用，10 m 以上风险高 | 部分满足 | 848×800 分辨率分散到鱼眼大视场，远距离小目标像素不足 |
| 边缘算力 | RKNN FP16 平均检测约 48 ms，约 20 Hz | 基本满足 | 明显优于 CPU ONNX 约 220 ms，已可进入闭环调参 |
| 检测精度 | W2 mAP@0.5 为 0.7592，Recall 为 0.6436 | 原型验证满足，稳定产品不足 | 可发现多数目标，但仍需 tracker/Kalman 抗漏检 |
| 控制频率 | 30Hz 外环，约 1kHz 内环/反馈 | 满足基本控制节奏 | 电机侧不再是当前主要瓶颈 |
| 可观测性 | MP4、metrics CSV、targets CSV | 满足调参需求 | 可复盘检测、跟踪、延迟和 yaw 命令 |

因此，本项目在“室内 3-8 m、目标有较明显人体区域暴露、作为辅助归中而非完全自主精确瞄准”的条件下，可以认为满足基本样机需求。它已经证明 Rock5B + T265 + HTDW yaw 电机能够组成一台可运行的视觉伺服原型：T265 提供宽视场和 IMU，RK3588/RKNN FP16 提供约 20Hz 的检测能力，ByteTrack + Kalman 提供跨帧稳定性，HTDW yaw 轴按受限控制命令执行转向。

同时，当前样机还不能被评价为成熟产品。主要原因包括：T265 灰度鱼眼图像远距离细节不足；模型不是针对真实真人 CS 场地重新采集训练；Recall 仍有漏检空间；真实电机闭环在不同负载和速度下还需要更多安全测试。若下一阶段要提升实际可用性，优先方向应是更换更高分辨率/更合适焦距的相机，采集真实场地数据做再训练，并在保持限位保护的前提下进行分级实机闭环测试。

### 1.5 问题定义与项目目标

人体检测模型的离线指标只能说明模型在单张图像上是否能框出目标；本项目真正要解决的是“室内突发目标的快速辅助归中”。如果直接使用单帧 YOLO 框中心点控制 yaw 轴，会出现三个主要问题。

第一，检测框抖动会直接变成电机抖动。YOLO 检测框中心受噪声、遮挡、姿态变化、鱼眼畸变和后处理阈值影响，单帧框本身没有时间连续性。

第二，短时漏检会导致目标突然消失或切换。墙角场景中目标可能只暴露身体一部分，鱼眼边缘、低光照、运动模糊或背景干扰都会造成短时无输出；如果系统直接消费检测列表，就容易进入 hold 或错误切换目标。

第三，控制系统对延迟比对单纯 FPS 更敏感。高 FPS 但控制的是旧画面，仍会造成滞后和抽动；低 FPS 但状态估计和控制节奏稳定，反而可能更适合安全调参。

因此本项目将问题定义为：

$$
\text{室内突发目标辅助归中}=\text{足够像素的视觉输入}+\text{可实时运行的人体检测}+\text{稳定目标状态估计}+\text{安全受限的 yaw 控制}
$$

围绕这一目标，项目需要回答以下问题：

1. T265 的 848×800 鱼眼图像在室内常见距离下是否有足够人体像素？
2. Rock5B / RK3588 是否能以足够低的延迟运行 YOLOv8n @640？
3. 原始 COCO 预训练 YOLOv8n 在行人/人体目标上是否够用，是否必须使用 WiderPerson fine-tuning？
4. ONNX、RKNN FP16 和 RKNN INT8 在精度、速度和边缘部署风险上如何取舍？
5. 光流 ROI、IMU 旋转补偿、ByteTrack 和 Kalman 如何共同提升连续帧中的目标稳定性？
6. yaw 外环、内环、限位和反馈保护如何使检测结果转化为安全可控的电机动作？
7. 这台由 Rock5B、T265 和 HTDW yaw 电机组成的样机，在哪些距离和目标暴露条件下可以认为满足基本需求？

### 1.6 本项目的主要工作

本项目的工作覆盖数据、模型、部署、跟踪、控制和可观测性六个方面。

第一，明确样机场景和硬件边界：室内真人 CS 墙角突发目标，硬件由 Rock5B、T265 和 HTDW yaw 电机构成，系统目标是辅助 yaw 归中而非完整自主决策。

第二，完成相机像素和识别距离评估，说明 T265 适合 3-8 m 近中距离样机验证，但不适合远距离局部目标稳定识别。

第三，完成 WiderPerson 原始标注到 YOLO 单类格式的转换，将 pedestrians 与 partially-visible persons 合并为 `person`，并过滤 riders、ignore regions 和 crowd 等边界不稳定区域。

第四，基于 COCO 预训练 YOLOv8n 进行迁移学习，完成 W0-W4 多组实验，比较原始模型、WiderPerson fine-tuning、输入尺寸、小目标增强和低学习率继续训练的效果。

第五，完成 W2 模型的 PyTorch、ONNX FP32、RKNN FP16 和 RKNN INT8 转换与评估，明确 ONNX 作为精度参考、RKNN FP16 作为当前 RK3588 主部署候选、INT8 作为需谨慎验证的候选格式。

第六，实现检测前运动 ROI 预测、YOLO 检测、ByteTrack-style 目标关联、Kalman TargetStateFilter、T265 IMU 旋转补偿预测和 yaw 控制执行之间的分层数据流。

第七，实现软件零位、相对角限位、最大速度、最大加速度、反馈保护、30Hz 视觉外环和 1kHz 电机内环，使视觉估计与电机执行解耦。

第八，建立 MP4 与 CSV 证据链，记录检测、跟踪、滤波、ROI、IMU、yaw、频率和延时指标，使系统运行结果可复盘、可定位、可调参。

### 1.7 架构层级

最终系统按数据流分为以下层级：

| 层级 | 输入 | 输出 | 核心作用 |
|---|---|---|---|
| 感知输入层 | T265 单路鱼眼图像、IMU 角速度 | 当前帧、相机运动信息 | 在宽视场内捕获墙角突发目标，并提供相机自运动信息 |
| 运动 ROI 预测层 | 当前帧、历史帧、IMU 修正光流 | 网络输入 ROI 或全帧策略 | 在检测前估计可能含目标的图像区域 |
| YOLO 检测层 | 全帧或 ROI letterbox 图像 | person 检测框、置信度 | 发现图像中的人体目标 |
| 目标跟踪层 | 检测框序列 | track ID、primary track | 建立跨帧目标连续性 |
| 状态估计层 | primary track 中心点 | filtered / predicted target center | 抑制检测噪声并处理短时漏检 |
| IMU 旋转补偿层 | T265 角速度、相机内参 | 旋转流与补偿后的运动估计 | 用相机物理运动解释画面整体位移 |
| 控制执行层 | yaw 角误差、目标状态 | 电机速度命令 | 让目标水平归中并保证安全限位 |
| 可观测记录层 | 全链路遥测 | MP4、CSV、图表 | 支撑复盘、调参和问题定位 |

这种组织方式也决定了本文后续结构：先讲输入层和检测前 ROI 预测层，再集中讲 YOLO 层中的数据、训练、评估与部署，然后讲跟踪、滤波、IMU 旋转补偿、控制执行和运行分析。

### 1.8 坐标与 yaw 控制符号

项目中明确两条物理约束：电机向左转时，反馈位置值增加；人物出现在图像左侧时，电机应向左转。令图像宽度为 $W$，目标中心横坐标为 $x_t$，图像中心为 $c_x=W/2$，定义水平像素误差为：

$$
e_x=c_x-x_t
$$

若使用相机水平焦距 $f_x$，则目标相对光轴的 yaw 角误差为：

$$
\alpha=\arctan\left(\frac{c_x-x_t}{f_x}\right)
$$

当目标位于图像左侧时，$x_t<c_x$，因此 $\alpha>0$，对应电机反馈位置增加，即向左转。这个符号约定贯穿检测框中心、tracker 输出、Kalman 滤波目标、控制器目标角和视频箭头方向，是整条链路一致性的基础。

---

## 2. 感知输入层：T265 单路鱼眼与时间约束

### 2.1 输入形态变化

系统输入层覆盖相机取流、图像格式、ROI 裁切和 IMU 对齐等问题。当前实测主链路使用 T265 单路鱼眼图像，重点包括灰度鱼眼图像稳定取流、IMU 与图像时间对齐、鱼眼视角下的目标检测与跟踪稳定性；彩色相机链路中涉及的 BGR/RGB 通道顺序、噪声和 ROI 裁切问题也作为输入层工程约束被纳入考虑。

T265 输入层的价值不仅在于提供图像，还在于同时提供 IMU 角速度。对于 yaw 云台系统，相机自身旋转会造成画面中几乎所有点的全局运动；如果只从图像局部纹理估计运动，光流很容易混入目标运动、背景纹理噪声和鱼眼畸变。因此本项目将 T265 IMU 作为 tracker 旋转补偿预测的主要运动来源。

### 2.2 最新帧优先与延迟意识

视觉伺服不只关注检测 FPS，还关注 frame age 和 detection age。若控制器消费的是过期帧，即使检测模型本身输出稳定，电机仍可能对旧目标位置做响应。因此运行记录中加入 frame age、detection age、queue wait、preprocess、inference、track/filter、render 等字段，用于判断系统到底慢在模型推理、队列堆积还是控制循环。

最新 RKNN FP16 链路中，frame age P50/P90 为 `61.6 / 63.1 ms`，明显优于 CPU ONNX 路径中约 `260 ms` 的旧帧延迟。这说明 NPU 推理不只是提升模型速度，也直接改善了控制系统所消费状态的新鲜度。

---

## 3. 光流 ROI 预测层：检测前的候选区域估计

### 3.1 ROI 层在流程中的位置

运动 ROI 预测层位于相机输入之后、YOLO 网络输入之前。它的作用不是替代检测器，而是在进入神经网络前，根据连续帧运动信息估计画面中更可能包含人体目标的区域，从而为后续检测提供 ROI 候选、诊断信息或全帧策略选择。

在数据流上，该层接收当前帧、历史帧、相机内参和 T265 IMU 角速度；输出可以是用于检测的候选 ROI，也可以是“当前更适合全帧检测”的判断。实际运行中，WiderPerson/RKNN 主链路采用全帧检测以避免 ROI 裁掉人体，但光流 ROI 仍是系统实现的一部分，用于运动分析、可视化和诊断，也为后续恢复 ROI 加速提供基础。

### 3.2 光流 ROI 与 IMU 修正

单纯光流会同时包含目标运动、背景纹理运动和相机自身运动。对于 yaw 云台，相机旋转会造成明显的全局画面运动；如果不加区分，ROI 可能被相机自运动主导，导致候选区域偏离真实人体目标。因此本项目将 LK 光流与 IMU 旋转流结合：先用图像光流估计局部运动，再用 IMU 角速度估计相机旋转导致的全局流，二者相减后得到更接近“场景中相对运动”的补偿运动量。

这个补偿后的运动量可形成 `compensated_global_dx / compensated_global_dy` 等诊断指标，也可用于判断 ROI 是否可信。当 IMU 对齐、纹理条件或运动估计不稳定时，系统可以回退到全帧检测，避免错误 ROI 截断人体。

### 3.3 为什么当前主链路采用全帧检测

实测中，motion ROI 曾出现裁掉人的风险，导致检测器无法看到完整人体。对于 yaw 视觉伺服来说，漏检或框不完整会直接影响 tracker 和控制目标，因此当前 WiderPerson/RKNN 配置选择全帧检测作为更稳妥的默认策略。

这并不意味着 ROI 层没有价值。它在架构上仍承担检测前运动预测与诊断作用：一方面解释图像运动来自目标还是相机自身；另一方面为后续优化提供候选路径。如果未来 RKNN 推理或后处理仍需进一步降耗，可以在确认 ROI 不损害召回的前提下重新启用 ROI 检测。


## 4. YOLO 检测层：数据、训练、评估与部署

YOLO 检测层是整套系统的第一道语义感知模块。本节集中说明目标检测背景、数据集处理、模型结构、训练设计、实验结果、部署格式和定性效果。训练部分全部归入 YOLO 层，因为它服务于同一个目标：为后续 tracker 和控制器提供可靠、低延迟、几何一致的人体框。

目标检测需要同时完成分类与定位。对于输入图像 $I$，检测模型输出若干 $(b_i,c_i,s_i)$，其中 $b_i$ 是边界框，$c_i$ 是类别，$s_i$ 是置信度。Two-stage detector 通常先生成候选区域再分类回归，精度较高但流程较复杂；YOLO 属于 one-stage detector，一次前向传播直接输出目标框和分数，更适合实时检测和边缘部署。本项目选择 YOLOv8n，也是因为它在速度、训练效率、部署成熟度和边缘算力之间具有较好的平衡。

人体检测还涉及完整人体框与可见区域框的区别。完整人体框试图表达人的完整身体范围，遮挡严重时需要对不可见部分进行推断；可见区域框更贴近图像中真实可观察到的人体像素。本项目真实训练采用 WiderPerson 的 pedestrians 与 partially-visible persons 合并为单类 `person`，本质上仍服务于“当前画面中可见、可检测、可跟踪的人体区域”这一目标。

### 4.1 数据集选择与类别定义

项目最终采用 WiderPerson 作为主训练数据集。选择依据包括：数据集中包含大量行人目标，场景覆盖密集人群、遮挡、小目标和局部可见人体；原始标注可以稳定转换为 YOLO 格式；数据集具有明确 train / val 划分，适合客观评估。COCO 在本项目中作为 YOLOv8n 预训练来源和 W0 baseline，用于衡量未进行领域适配时的性能；原 `cv.md` 框架中讨论的 CrowdHuman visible-region 思路为任务定义提供了参考，但实际训练、评估和部署结果以 WiderPerson W0-W4 实验为准。

最终检测类别定义为单类：

| class id | class name | 含义 |
|---:|---|---|
| 0 | person | 可用于系统感知和跟踪的人体目标 |

WiderPerson 原始类别经过如下合并和过滤：

| 原始类别 | 含义 | 处理方式 |
|---:|---|---|
| 1 | pedestrians | 保留，映射为 `person` |
| 2 | riders | 跳过 |
| 3 | partially-visible persons | 保留，映射为 `person` |
| 4 | ignore regions | 跳过 |
| 5 | crowd | 跳过 |

保留 partially-visible persons 的原因是：真实场景中人体经常不是完整出现，局部可见人体仍可能是需要跟踪的目标。过滤 riders、ignore regions 和 crowd 的原因是这些区域边界或语义不稳定，容易污染单类检测器的边界学习。

### 4.2 标注转换与数据质量

WiderPerson 原始框为像素坐标 $(x_1,y_1,x_2,y_2)$。YOLO 训练需要归一化中心点格式，即类别、中心点横坐标、中心点纵坐标、宽度和高度。转换公式为：

$$
w=x_2-x_1,\quad h=y_2-y_1
$$

$$
x_c=\frac{x_1+x_2}{2W},\quad y_c=\frac{y_1+y_2}{2H}
$$

$$
w_n=\frac{w}{W},\quad h_n=\frac{h}{H}
$$

转换后数据规模如下：

| Split | Images | Boxes | Empty Labels | Issues |
|---|---:|---:|---:|---:|
| train | 8000 | 231290 | 6 | 0 |
| val | 1000 | 27163 | 1 | 0 |

原始类别转换统计如下：

| Split | kept class 1 | kept class 3 | kept total | skipped class 2 | skipped class 4 | skipped class 5 |
|---|---:|---:|---:|---:|---:|---:|
| train | 160711 | 70579 | 231290 | 1474 | 3240 | 8979 |
| val | 17831 | 9332 | 27163 | 185 | 409 | 661 |

质检结果显示：图片和标签一一对应，所有类别均被映射为单类 person，归一化坐标均在合法范围内，无越界框、坏图或标签格式错误。少量空标签样本来自 riders、ignore 和 crowd 过滤后没有可训练 person 框的图片，属于可解释情况。

训练集与验证集的框数量分布如下：

![train boxes per image](assets/metrics_training_config_train_boxes_per_image.png)

![val boxes per image](assets/metrics_training_config_val_boxes_per_image.png)

训练集与验证集的目标面积分布如下：

![train box area](assets/metrics_training_config_train_box_area.png)

![val box area](assets/metrics_training_config_val_box_area.png)

这些分布说明 WiderPerson 中存在大量密集、小尺度和局部人体目标，输入尺寸、召回率和边界定位质量会显著影响最终系统效果。

### 4.3 YOLOv8n 模型结构与适配理由

YOLOv8n 由 Backbone、Neck 和 Detect Head 组成。Backbone 提取边缘、纹理、人体局部结构等视觉特征；Neck 融合多尺度特征，使模型同时处理近处大目标和远处小目标；Detect Head 输出边界框与类别分数。

YOLOv8n 使用 C2f 模块进行轻量特征复用，使用 SPPF 扩大感受野，使用 anchor-free decoupled head 降低 anchor 设计负担，并通过 Distribution Focal Loss 提升边界定位质量。对于 WiderPerson 中的小目标和拥挤遮挡人体，多尺度融合与较大输入尺寸尤其重要。

选择 YOLOv8n 的原因并不是追求最高离线精度，而是平衡训练效率、推理速度、部署成熟度和边缘端算力约束。YOLOv8n 足够小，便于快速完成多组消融实验；同时支持 ONNX 与 RKNN 转换，适合后续在 RK3588 上部署。

### 4.4 输入预处理与几何一致性

检测层采用 letterbox 保持输入图像宽高比。直接 resize 会改变图像几何比例，导致检测框投回原图时出现比例误差；对于只做离线检测的模型，这种误差可能只影响 mAP；对于 yaw 视觉伺服，它会直接影响目标中心点和角误差计算。

因此，部署侧需要保证训练输入、推理输入、坐标还原和控制中心点之间几何一致。对于 T265 鱼眼画面，虽然模型并非专门使用鱼眼数据训练，但保持 letterbox 几何关系仍能减少人为缩放误差。

### 4.5 损失函数、优化器与训练策略

YOLOv8 检测训练主要由 box loss、cls loss 和 dfl loss 组成。box loss 优化预测框位置，cls loss 优化类别判断，dfl loss 优化边界分布并提升定位精度。由于本项目只有 person 单类，分类难度低于 COCO 80 类，边界质量和召回率更重要。

主要训练实验使用 AdamW、余弦学习率和 early stopping。AdamW 适合迁移学习和较短实验周期；余弦学习率有利于后期收敛；early stopping 防止长时间无收益训练。

W1/W2 的主要训练设置如下：

| 参数 | 设置 |
|---|---|
| optimizer | AdamW |
| initial learning rate | 0.001 |
| scheduler | cosine learning rate |
| batch | W1: 64；W2-W4: 48 |
| epochs | 100 |
| early stopping patience | 20 |
| mosaic close stage | W1/W2/W4: 15；W3: 20 |

### 4.6 W0-W4 实验设计

项目设计了五组实验，依次验证领域适配、输入尺寸、增强策略和继续训练策略的影响。

| 实验 | 目的 | 主要变量 |
|---|---|---|
| W0 | 原始 COCO baseline | COCO 预训练 YOLOv8n 直接验证 |
| W1 | WiderPerson fine-tune baseline | 输入尺寸 416 |
| W2 | 输入尺寸优化 | 输入尺寸 640 |
| W3 | 小目标增强实验 | 在 W2 基础上加强 scale、mixup、copy-paste 等增强 |
| W4 | 小学习率继续训练 | 在 W2 基础上降低学习率继续训练 |

W3 增强策略的主要差异如下：

| 参数 | W2 | W3 |
|---|---:|---:|
| scale | 0.5 | 0.7 |
| translate | 0.1 | 0.15 |
| degrees | 0.0 | 2.0 |
| shear | 0.0 | 1.0 |
| perspective | 0.0 | 0.0005 |
| mixup | 0.0 | 0.05 |
| copy_paste | 0.0 | 0.05 |
| close_mosaic | 15 | 20 |

实验逻辑是：先验证 COCO 预训练是否足够，再验证 WiderPerson fine-tuning 是否必要；之后只改变输入尺寸检验小目标收益；最后检验强增强和小学习率继续训练是否能超过 W2。

### 4.7 W0-W4 验证结果

| 实验 | 模型 | 输入尺寸 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|---:|---:|
| W0 | COCO YOLOv8n | 416 | 0.6952 | 0.4453 | 0.5279 | 0.2240 |
| W1 | WiderPerson fine-tuned | 416 | 0.7958 | 0.6059 | 0.7129 | 0.4477 |
| W2 | WiderPerson fine-tuned | 640 | 0.8138 | 0.6436 | 0.7592 | 0.4880 |
| W3 | W2 + small-object aug | 640 | 0.8128 | 0.6427 | 0.7586 | 0.4778 |
| W4 | W2 + lower lr | 640 | 0.8116 | 0.6444 | 0.7577 | 0.4851 |

![W1-W4 validation metrics](assets/analysis_W1_W4_validation_metrics.png)

![W1-W4 delta vs W2](assets/analysis_W1_W4_map5095_delta_vs_W2.png)

W0 到 W1 的提升说明 COCO 预训练模型不能直接满足 WiderPerson 行人检测。W0 recall 只有 `0.4453`，大量 WiderPerson 行人未被检出；W1 fine-tune 后 recall 提升到 `0.6059`，mAP@0.5:0.95 从 `0.2240` 提升到 `0.4477`，证明领域迁移非常关键。

W1 到 W2 的提升说明输入尺寸是最有效的调参方向。W2 的 recall、mAP@0.5 和 mAP@0.5:0.95 均明显高于 W1。对于小目标和局部可见人体，`640` 输入能在特征图上保留更多细节。

W3 的小目标增强没有超过 W2。虽然 mAP@0.5 几乎持平，但 mAP@0.5:0.95 降到 `0.4778`，说明模型仍能粗略判断“有人”，但框定位质量变差。对本项目而言，框中心点会进入控制链路，因此边界定位质量不能被牺牲。

W4 的小学习率继续训练也没有超过 W2。它让 recall 微升到 `0.6444`，但 mAP@0.5:0.95 仍低于 W2，说明 W2 已接近当前设置下的较优点。

### 4.8 训练曲线与收敛分析

W2 是最关键实验。训练曲线显示 loss 持续下降，mAP 在约 50 到 70 epoch 后进入平台区，best epoch 为 71。后续 epoch 指标没有明显提升，early stopping 是合理的。

![W2 results](assets/curves_W2_results.png)

![W2 metrics](assets/curves_W2_W2_metrics.png)

![W2 train losses](assets/curves_W2_W2_train_losses.png)

![W2 val losses](assets/curves_W2_W2_val_losses.png)

W3 的曲线显示 best epoch 为 58，之后 mAP@0.5:0.95 未改善，说明增强后的继续训练没有获得更好定位精度。

![W3 results](assets/curves_W3_results.png)

W4 的 best epoch 为 46，说明较小学习率下模型较早达到最佳点，但最终仍低于 W2。

![W4 results](assets/curves_W4_results.png)

训练过程的多指标对比如下。W2 在 mAP@0.5:0.95 上达到最优，W4 在 recall 上略高但综合定位指标不足。

![mAP 曲线对比](assets/image.png)

![Precision 与 Recall 曲线对比](assets/image-1.png)

### 4.9 PR、F1 与混淆矩阵分析

W2 的 PR 曲线、F1 曲线和混淆矩阵用于观察阈值选择和误检漏检趋势。

![W2 PR curve](assets/curves_W2_PR_curve.png)

![W2 F1 curve](assets/curves_W2_F1_curve.png)

![W2 confusion matrix](assets/curves_W2_confusion_matrix.png)

![W2 normalized confusion matrix](assets/curves_W2_confusion_matrix_normalized.png)

在单类检测任务中，混淆矩阵主要反映 background 与 person 的混淆，即误检和漏检。对于 WiderPerson，漏检通常来自极小目标、密集遮挡或只露出局部身体的行人。

### 4.10 W1 与 W2 速度精度权衡

| 实验 | 输入尺寸 | batch | inference ms/img | e2e ms/img | model FPS | e2e FPS | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | 416 | 1 | 4.427 | 5.244 | 225.9 | 190.7 | 0.4477 |
| W2 | 640 | 1 | 4.688 | 5.389 | 213.3 | 185.6 | 0.4880 |
| W1 | 416 | 32 | 0.442 | 0.857 | 2264.7 | 1167.3 | 0.4477 |
| W2 | 640 | 32 | 0.638 | 1.122 | 1567.2 | 891.5 | 0.4880 |

![W1-W2 latency comparison](assets/benchmarks_W1_W2_latency_comparison.png)

![W1-W2 FPS comparison](assets/benchmarks_W1_W2_fps_comparison.png)

![W1-W2 accuracy comparison](assets/benchmarks_W1_W2_accuracy_comparison.png)

W2 在 batch=1 时端到端延迟只比 W1 增加约 `0.145 ms/img`，但 mAP@0.5:0.95 提升 `0.0402`。因此对于单帧相机实时检测，W2 的精度收益明显大于速度损失。

### 4.11 YOLO 定性检测结果

W0 原始 COCO 模型在密集场景中漏检较多：

![W0 qualitative example](assets/qualitative_results_W0_yolov8n_predictions_017568.jpg)

W1 经过 WiderPerson fine-tune 后，行人检测明显更完整：

![W1 qualitative example](assets/qualitative_results_W1_yolov8n_predictions_017568.jpg)

W2 在同类密集行人场景中进一步改善小目标和边界覆盖：

![W2 qualitative example](assets/qualitative_results_W2_yolov8n_predictions_017568.jpg)

这些定性结果与指标一致：W2 能更稳定覆盖密集人群中的多个可见人体，是后续 tracker 的更好输入。

### 4.12 YOLO 部署转换与边缘推理表现

W2 模型完成了 PyTorch、ONNX FP32、RKNN FP16 和 RKNN INT8 多格式转换。ONNX FP32 作为跨平台精度参考；RKNN FP16 作为 RK3588 当前主部署候选；RKNN INT8 作为体积和速度候选，但必须通过板端 score、mAP 和 NMS 行为验证。

导出格式统计如下：

| 格式 | 大小 | 当前定位 |
|---|---:|---|
| PyTorch 权重 | 6.0 MB | 训练源模型与精度基准 |
| ONNX FP32 | 11.7 MB | 精度参考与部署中间表示 |
| RKNN FP16 | 7.7 MB | 当前 RK3588 主候选 |
| RKNN INT8 | 4.7 MB | 候选格式，当前不作为默认控制模型 |

PyTorch 与 ONNX 的精度速度对比如下：

| 后端 | 设备 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | inference | e2e |
|---|---|---:|---:|---:|---:|---:|---:|
| PyTorch W2 | RTX 4090 | 0.8138 | 0.6436 | 0.7592 | 0.4880 | 4.7 ms/img | 5.4 ms/img |
| ONNX FP32 | CPU | 0.8120 | 0.6430 | 0.7570 | 0.4870 | 91.0 ms/img | 93.9 ms/img |

![deploy accuracy comparison](assets/deploy_W2_deploy_accuracy_comparison.png)

![deploy latency comparison](assets/deploy_W2_deploy_latency_comparison.png)

ONNX FP32 与 PyTorch 的 mAP@0.5:0.95 仅差约 `0.001`，说明格式转换本身基本没有造成精度损失。ONNX 慢主要来自 CPU 推理环境，而不是模型结构本身无法加速。

RKNN FP16 不进行 INT8 量化，因此更适合保留 score 精度。RKNN INT8 虽然文件最小、理论速度更快，但当前实机调试中出现 score channel 异常，无法稳定产生有效检测。因此系统默认策略是：ONNX 保留为精度参考，RKNN FP16 用于实机主链路，INT8 暂不作为默认 yaw 控制模型。

### 4.13 YOLO 层在系统中的最终判断

YOLO 层的最终选择是 W2：YOLOv8n WiderPerson @640。它优于 W0，是因为完成了 WiderPerson 领域适配；优于 W1，是因为更大输入尺寸显著改善小目标和遮挡人体；优于 W3，是因为没有被强增强破坏定位质量；优于 W4，是因为继续低学习率训练没有带来综合收益。部署侧优先使用 RKNN FP16，是因为它在速度和 score 保真之间更平衡。

---

## 5. 目标跟踪层：从检测框到 primary track

### 5.1 为什么不能直接使用检测列表

若 yaw 目标直接来自单帧检测列表中的第一个框或最高置信框，这种策略在多人场景和短时漏检时非常脆弱：检测列表排序不一定代表真实主目标；当检测器短时无输出时，系统可能立即 hold；当多个目标靠近或置信度变化时，系统可能频繁切换目标。

目标跟踪层的任务是将单帧检测框序列组织成跨帧 track，并从中选择 primary track 作为控制目标。这样 yaw 控制器消费的是“连续目标”，而不是“当前帧临时框”。

### 5.2 ByteTrack-style 关联逻辑

项目引入 ByteTrack-style tracker，其核心思想是将检测框分为高置信和低置信两类：高分检测先与已有轨迹匹配，未匹配轨迹再尝试用低分检测找回，只有高分未匹配检测可以创建新 track。

这种设计的优点是：低分框不会轻易创建新 ID，避免噪声目标污染 tracker；但低分框仍可帮助已有目标跨过短时置信度下降，提升遮挡或边缘场景下的连续性。

匹配使用 IoU 描述检测框与轨迹预测框的重叠程度：

$$
IoU(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

当 IoU 高于匹配阈值时，可认为检测框与轨迹属于同一目标。

### 5.3 track loss 生存时间修正

系统测试表明，若 track lost 后仍存活过久，yaw 会持续响应一个已经不可信的位置。因此 track loss 策略被收紧：连续 lost 的生存周期限制为一次推理级别，没有新推理结果时不无意义累加 missed 状态，tracker 的目标预测由 IMU 旋转补偿提供，而不是由稀疏光流直接提供。

这使系统在“短时检测波动”和“真实目标消失”之间取得更合理的平衡：允许一次推理级别的恢复，但不让陈旧目标长期控制电机。

![Tracker missed_frames 分布](assets/tracker_missed_frames_distribution.png)

### 5.4 跟踪层实测表现

在最新 `T265 + RKNN FP16 + IMU 旋转补偿 tracker` 运行中，90 帧内检测框总数为 179，primary track 覆盖 77 帧，覆盖率为 `85.6%`。这说明在当前室内测试视频中，检测层和跟踪层已经能够维持较稳定目标。

模型与跟踪综合对比如下：

![模型与跟踪对比](assets/model_tracking_comparison.png)

ONNX 路径中，检测耗时约 `220 ms`，检测 FPS 约 `4 Hz`；RKNN FP16 路径中，检测耗时约 `48 ms`，检测 FPS 接近 `20 Hz`。检测频率提升后，tracker 获得更密集的观测，primary track 覆盖率和目标稳定性也随之改善。

---

## 6. 状态估计层：Kalman TargetStateFilter

### 6.1 从 raw center 到 filtered center

即使 tracker 已经提供了 primary track，检测框中心仍然会随边界框噪声、人体姿态变化和局部遮挡产生抖动。状态估计层的任务是将 primary track 中心点转换为时间连续的目标状态，并在短时漏检时提供可解释预测。

Kalman TargetStateFilter 的状态定义为：

$$
\mathbf{x}_k=[c_x,c_y,v_x,v_y]^T
$$

其中 $c_x,c_y$ 是目标中心点，$v_x,v_y$ 是像素速度。状态转移模型为：

$$
\mathbf{x}_k=\mathbf{F}\mathbf{x}_{k-1}+\mathbf{w}_k
$$

$$
\mathbf{F}=\begin{bmatrix}1&0&\Delta t&0\\0&1&0&\Delta t\\0&0&1&0\\0&0&0&1\end{bmatrix}
$$

观测模型为：

$$
\mathbf{z}_k=\mathbf{H}\mathbf{x}_k+\mathbf{v}_k,\quad \mathbf{H}=\begin{bmatrix}1&0&0&0\\0&1&0&0\end{bmatrix}
$$

有 primary track 时，系统用 bbox center 更新 Kalman；primary 短时丢失时，系统使用 prediction；超过 timeout 或预测帧数上限后，系统认为当前无可靠目标并让 yaw hold。

![目标中心时间序列](assets/target_center_timeseries.png)

### 6.2 状态估计对控制的意义

Kalman 的意义不是简单“让曲线更平滑”，而是将检测噪声、目标运动趋势和短时漏检区分开来。检测器负责提供观测，tracker 负责建立 ID 连续性，Kalman 负责让控制目标成为连续状态。

在视频 UI 中，红点代表实际控制目标，即 filtered center 或 predicted center。这样调试者可以直观看到：电机正在跟随哪个点，并判断控制器消费的是滤波目标、预测目标还是异常目标。

---

## 7. IMU 旋转补偿层：面向目标预测的相机自运动建模

### 7.1 tracker 旋转补偿预测的设计理由

光流适合描述图像局部纹理运动，但在鱼眼画面、低纹理区域、光照变化和镜头旋转时，稀疏 LK 光流容易混合目标运动与相机自身运动。对于 yaw 云台，相机旋转是画面整体移动的重要物理来源；使用 IMU 角速度预测旋转流，比让局部纹理直接决定目标预测更符合系统因果结构。

当前逻辑为：光流仍可作为 ROI、可视化和诊断字段保留；tracker 的短时目标预测使用 T265 IMU 角速度计算出的旋转流；若 IMU 补偿无效，则旋转补偿预测量置零。

### 7.2 IMU 旋转流模型

设归一化像平面坐标为：

$$
x=\frac{u-c_x}{f_x},\quad y=\frac{v-c_y}{f_y}
$$

相机角速度为：

$$
\boldsymbol{\omega}=[\omega_x,\omega_y,\omega_z]^T
$$

纯旋转导致的归一化光流近似为：

$$
\dot{x}=-\omega_xxy+\omega_y(1+x^2)-\omega_zy
$$

$$
\dot{y}=-\omega_x(1+y^2)+\omega_yxy+\omega_zx
$$

换回像素位移：

$$
\Delta u=f_x\dot{x}\Delta t,\quad \Delta v=f_y\dot{y}\Delta t
$$

工程上，该旋转流用于估计画面由相机旋转引起的位移，并作为 tracker 短时目标预测的物理补偿来源。

![IMU 旋转补偿曲线](assets/imu_compensation_dx.png)

### 7.3 IMU 旋转补偿预测的系统收益

IMU 旋转补偿预测避免了稀疏光流受纹理、噪声、局部人体运动影响。它并不能替代检测器，也不能长期追踪消失目标，但在检测间隔内可以为 tracker 和 Kalman 提供更符合云台运动的短时预测。它与光流 ROI 的区别在于：光流 ROI 服务于检测前候选区域估计，而 IMU 旋转补偿预测服务于跟踪阶段的目标状态延续。

---

## 8. 控制执行层：30Hz 外环与 1kHz 内环

### 8.1 软件零位与相对限位

电机内部零点写入曾出现不稳定，因此系统采用软件零位。运行时相对 yaw 角定义为：

$$
\theta_{rel}=\theta_{raw}-\theta_{zero}
$$

所有限位都基于相对 yaw 角，而不是电机内部零点。当前安全测试允许范围为：

$$
-45^\circ \le \theta_{rel} \le 45^\circ
$$

当目标命令越界时，控制器应夹紧到安全边界并主动回到允许范围，而不是简单拒绝响应。这样可以避免内部零点写入不可靠导致实际限位漂移。

### 8.2 视觉外环与电机内环拆频

视觉推理输出是不均匀、低频、受模型耗时影响的；电机速度命令则需要高频、均匀、可限速。项目因此将视觉外环固定为 30Hz，将电机内环目标频率设为 1kHz。

视觉外环根据 filtered / predicted target center 计算 yaw 角误差：

$$
\alpha_k=\arctan\left(\frac{c_x-x_{filtered,k}}{f_x}\right)
$$

控制器再根据角误差、积分项、微分项和前馈项得到轴侧期望速度：

$$
\omega_{axis}=K_p\alpha+K_i\int\alpha dt+K_d\frac{d\alpha}{dt}+K_{ff}\dot{\alpha}
$$

最后根据轴侧角速度与电机侧转速关系生成电机速度命令。

### 8.3 外环固定频率与控制节奏

用户曾观察到 yaw 电机一卡一卡，怀疑外环频率过低。系统因此从“只随推理结果更新”调整为固定 30Hz 外环。即使没有新检测，外环仍可以消费最新 Kalman/IMU 预测状态和安全保护逻辑。这使控制输出节奏更均匀，减少“检测来了才动一下”的离散感。

外环固定 30Hz 的运行时间分布如下：

![外环 30Hz 时间分布](assets/outer_30hz_timing.png)

### 8.4 控制安全约束

控制层的安全约束包括相对角限位、最大速度、最大加速度、反馈必需和反馈超时保护。当前低速测试阶段最大速度设置较保守，目的是优先验证方向、限位、反馈和停止逻辑，而不是追求高速跟踪。

控制层的核心判断是：视觉模块可以有不确定性，但电机执行必须始终受限、可停止、可复盘。只有当目标状态、控制方向、限位状态和反馈频率都能在视频和 CSV 中确认后，才适合分级提高速度。

---

## 9. 可观测记录层：视频 UI、CSV 与系统实测

### 9.1 为什么可观测性是系统能力的一部分

视觉伺服系统调试不能只看“电机有没有转”。如果没有 frame age、detection age、track ID、filtered center、yaw target、current yaw、inner loop frequency 和 feedback state，就无法判断问题来自检测、跟踪、滤波、控制还是通信。

保存视频 UI 采用集中式仪表盘，并扩展 CSV 字段，使每次运行都能复盘目标、tracker、滤波器、IMU 补偿、控制器状态和性能指标。

### 9.2 视频仪表盘内容

视频 UI 集中显示以下信息：

- 当前 frame、source、detection、tracking frame id。
- loop、camera、detection FPS。
- preprocess、motion、inference、track/filter、render 分段延时。
- frame age、detection age、queue wait。
- 检测框数、tracker 输入数、primary track 状态。
- raw、filtered、predicted target center。
- yaw target、current yaw、measured rpm、limit 状态。
- outer loop、inner loop、feedback 频率与 jitter。

最新 RKNN FP16 + IMU 旋转补偿预测 tracker 视频截图如下：

![最新 RKNN FP16 + IMU 旋转补偿 tracker 视频截图](assets/screenshot_175541.jpg)

更多运行截图如下：

| 场景 | 截图 |
|---|---|
| ONNX WiderPerson 离线/全帧 | ![ONNX screenshot](assets/screenshot_170928.jpg) |
| T265 + ONNX WiderPerson | ![T265 ONNX screenshot](assets/screenshot_171151.jpg) |
| T265 外环固定 30Hz | ![outer 30Hz screenshot](assets/screenshot_173846.jpg) |
| RKNN FP16 离线验证 | ![RKNN offline screenshot](assets/screenshot_174007.jpg) |

### 9.3 运行数据汇总

| Run | 场景 | 帧数 | 检测框总数 | Primary track 覆盖 | 检测 FPS | 检测耗时 ms | Frame age P50/P90 ms | 内环 Hz | 最大反馈丢失 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 170928 | ONNX WiderPerson 离线/全帧 | 120 | 301 | 44 (36.7%) | 4.46 | 221.7 | 261.2 / 269.8 | 1005.8 | 0 |
| 171151 | T265 + ONNX WiderPerson | 60 | 180 | 58 (96.7%) | 4.03 | 231.8 | 125.8 / 247.3 | 1005.6 | 0 |
| 173846 | T265 外环固定 30Hz | 120 | 0 | 0 (0.0%) | - | - | 15.1 / 15.7 | 998.2 | 0 |
| 174007 | RKNN FP16 离线验证 | 60 | 252 | 59 (98.3%) | 19.61 | 47.8 | 87.5 / 89.8 | 998.4 | 0 |
| 175541 | T265 + RKNN FP16 + IMU 旋转补偿 tracker | 90 | 179 | 77 (85.6%) | 19.96 | 48.0 | 61.6 / 63.1 | 996.6 | 0 |

从数据可见，CPU ONNX 路径检测耗时约 `220 ms`，检测 FPS 约 `4 Hz`，难以满足高动态闭环；RKNN FP16 检测耗时约 `48 ms`，检测 FPS 约 `20 Hz`，已经进入可继续实机调参的区间。最新链路中最大反馈丢失为 0，内环和反馈频率接近 1kHz，说明当前瓶颈主要仍在检测推理而不是电机反馈或日志写入。

### 9.4 最新链路延时分析

最新 `T265 + RKNN FP16 + IMU 旋转补偿 tracker` 运行中的延时分解如下：

![最新链路延时](assets/latency_breakdown_latest.png)

关键指标为：平均检测耗时 `48.0 ms`，平均检测 FPS `19.96 Hz`，frame age P50/P90 为 `61.6 / 63.1 ms`，内环平均频率 `996.6 Hz`，反馈平均频率 `996.6 Hz`，最大 missed feedback 为 `0`。

这说明 RKNN FP16 已经将系统从“旧帧驱动、明显滞后”的 ONNX CPU 阶段推进到“约 50ms 检测、约 60ms frame age”的实机可调阶段。若后续还要提升到稳定 25Hz 以上，应优先优化 NPU 后处理、阈值策略、线程流水线和图像输入链路。

---

## 10. 仓库实现落地与代码对应

前文从算法和系统层面说明了项目目标，本节补充仓库中的实际工程落地情况。该项目的代码并不是单一训练脚本，而是由 C++ 实时主链路、Python 实验链路、模型文件、运行配置、硬件调试脚本和项目文档共同构成。整理后，仓库的主入口更加明确：实时运行与验收应优先看 `cpp/`，算法实验和历史验证看 `src/`，模型部署看 `models/` 与 `third_party/`，板端运行看 `configs/` 与 `scripts/`。

### 10.1 仓库清理与结构整理

本次整理删除了确定属于中间产物或本地环境痕迹的文件，包括：`build-opencv410/` 构建目录、`srcforgpt.zip` 临时打包文件、根目录和 `src/` 下的 `.DS_Store`、各级 Python `__pycache__/`，以及只包含个人 Python 环境偏好的 `.vscode/` 配置。上述文件不参与项目复现，也会干扰仓库结构阅读和版本管理。

`.gitignore` 同步补充了 `build-*/`、`.vscode/`、`.DS_Store`、`srcforgpt.zip` 和 `*.zip` 等规则，避免后续再次提交构建目录、编辑器设置、系统元数据和临时压缩包。清理后的顶层目录保留为：`configs/`、`cpp/`、`models/`、`project_docs/`、`scripts/`、`src/`、`third_party/`、`tools/`，再加顶层 CMake、README、Python 依赖和电机说明文档。

文档和源码中也移除了明显的 AI 生成痕迹：`project_docs/README_MACOS.md` 与 `project_docs/system_design.md` 中的 `AI Assistant` 作者署名已删除；Python 脚本和说明文档中用于标记“新增、修复、成功、失败”的表情符号已清理，保留真实工程含义的注释和输出文本。这样处理后，仓库从“迭代草稿”更接近可交付项目结构。

### 10.2 C++ 主链路实现

C++ 主链路由 `cpp/src/main.cpp` 串联。程序支持 `--config <file.conf>` 读取配置，也支持命令行覆盖关键参数。主要后端包括 `--backend opencv`、`--backend hikrobot`、`--backend aravis`、`--backend t265` 和 `--backend realsense`。检测器参数为 `--detector none|yolo|rknn`，其中 `yolo` 对应 ONNX Runtime 路径，`rknn` 对应 RKNN Runtime 路径。输出侧支持 `--output` 自动生成视频和 CSV，也支持 `--record`、`--telemetry` 手动指定文件。

取流层采用统一的 frame source 抽象。`cpp/include/cvproj/frame_source.hpp` 定义不同相机或视频源共同遵守的接口；`cpp/src/opencv_video_source.cpp` 用于读取本地视频和普通摄像头，适合无硬件时验证算法链；`cpp/src/hikrobot_mvs_source.cpp` 通过海康 MVS SDK 接入 MV-CS016-10UC 等工业相机；`cpp/src/aravis_frame_source.cpp` 通过 USB3 Vision / GenICam 标准协议作为无 SDK 场景下的回退路径；`cpp/src/realsense_t265_source.cpp` 负责 T265 图像与姿态/IMU 输入。

运动分析层主要由 `cpp/src/motion_pipeline.cpp` 实现。该模块生成固定网格 LK 光流点，估计全局中值运动，结合 T265 pose/IMU 角速度进行相机旋转补偿，进一步生成 motion ROI、运动 blob 和可视化叠加。它的作用不是替代 YOLO，而是在检测前提供运动先验、诊断信息和可视化依据。实际 WiderPerson/RKNN 主链路默认采用全帧检测，是因为实测中 ROI 裁切存在截断人体的风险；但 ROI 层仍然保留，用于解释画面运动来源和后续加速策略。

检测层分为 ONNX 与 RKNN 两条实现。`cpp/src/yolo_onnx_detector.cpp` 用 ONNX Runtime 进行 YOLO 推理，适合 PC 侧验证前后处理、阈值、NMS 和几何还原；`cpp/src/yolo_rknn_detector.cpp` 用 RKNN Runtime 调用 RK3588 NPU，适合 Rock5B 边缘部署。CMake 会在找到 `third_party/rknn/include/rknn_api.h` 和 `third_party/rknn/lib/librknnrt.so` 时启用 RKNN 支持，否则程序仍可构建但不能使用 `--detector rknn`。

跟踪与状态估计层由 `cpp/src/bytetrack_tracker.cpp` 和 `cpp/src/target_state_filter.cpp` 构成。前者实现 ByteTrack-style 目标关联，通过高低置信度检测框和 IoU 匹配维护 track ID，并选出 primary track；后者用 Kalman 滤波器对目标中心点进行平滑和短时预测，避免单帧检测框抖动直接传入控制器。该设计对应报告前文“不能用单帧 YOLO 框直驱电机”的结论。

输出与控制层包括 `cpp/src/mjpeg_server.cpp` 和 `cpp/src/socketcan_gimbal.cpp`。MJPEG server 支持 `--livestream`，可在浏览器实时查看叠加后的图像、检测框、目标状态和运行信息；SocketCAN 模块负责 yaw 电机命令发送、反馈读取、限位保护和控制频率管理。控制参数包括 `--yaw-dry-run`、`--yaw-can-iface`、`--yaw-id`、`--yaw-max-angle`、`--yaw-max-rpm`、`--yaw-max-accel-rpm-s`、`--yaw-outer-rate-hz` 和 `--yaw-inner-rate-hz` 等。

### 10.3 CMake 与跨平台构建

顶层 `CMakeLists.txt` 规定 C++17，默认启用 `CVPROJ_ENABLE_HIKROBOT_MVS`、`CVPROJ_ENABLE_ARAVIS` 和 `CVPROJ_ENABLE_REALSENSE`。OpenCV 是必需依赖，组件包括 `core`、`dnn`、`imgproc`、`highgui`、`videoio`、`video` 和 `imgcodecs`。`cpp/CMakeLists.txt` 将主程序构建为 `cvproj_capture`，并按依赖可用情况选择性链接 RKNN Runtime、Aravis、librealsense2 和 Hikrobot MVS SDK。

如果目标机器没有工业相机 SDK 或 RealSense，可以通过以下方式先构建基础链路：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCVPROJ_ENABLE_ARAVIS=OFF \
  -DCVPROJ_ENABLE_HIKROBOT_MVS=OFF \
  -DCVPROJ_ENABLE_REALSENSE=OFF
cmake --build build -j
```

这种构建仍可使用 OpenCV 视频回放、ONNX 检测、跟踪、滤波、录制和部分可视化逻辑，适合无硬件环境下做软件验收。Rock5B 或 RK3588 部署时，应确认 RKNN Runtime 头文件和动态库位置正确，否则 `--detector rknn` 不会启用。

### 10.4 配置文件与典型运行方式

`configs/` 下保存不同硬件和链路的运行配置。`rock5b_default.yaml` 是 Rock5B 默认配置，包含 `source: "mvs"`、`backend: "rknn"`、`model_rknn`、输出目录、帧率、分辨率、YOLO 输入尺寸、光流点数、曝光、增益、CAN 通道和云台比例系数等字段。`t265_yaw_rknn_fp16.conf` 和相关 T265 配置用于 T265 + RKNN + yaw 控制链路；`windows_default.yaml` 用于 PC 调试。

基础回放验证命令为：

```bash
./build/cpp/cvproj_capture \
  --backend opencv \
  --source path/to/input.mp4 \
  --detector none \
  --output \
  --max-frames 600 \
  --headless
```

ONNX 检测验证命令为：

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

RKNN FP16 验证命令为：

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

T265 + yaw dry-run 命令为：

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

这些命令体现了仓库的验收顺序：先验证回放处理链，再验证 ONNX 检测，再验证 RKNN 推理，最后接入 T265、实时预览和 yaw 控制。

### 10.5 模型文件与部署形态

仓库中保留了多种模型格式，服务于不同验证阶段。`src/widerperson_yolov8n_640.onnx` 是 WiderPerson YOLOv8n @640 的 ONNX 模型，可作为 PC 侧精度和前后处理参考；`models/rknn/widerperson_yolov8n_640_fp16.rknn` 是当前 RK3588/Rock5B 主候选模型；`models/rknn/widerperson_yolov8n_640_int8.rknn` 是 INT8 量化候选，仍需要更谨慎的精度和稳定性验证。

`src/` 下还保留 `yolov8n_128.onnx`、`yolov8n_160.onnx`、`yolov8n_192.onnx`、`yolov8n_224.onnx`、`yolov8n_256.onnx`、`yolov8n_320.onnx`、`yolov8n.onnx`、`yolo11n.onnx`、`yolo11n.pt` 等对照模型。这些文件说明项目经历过输入尺寸、模型版本和部署格式的多轮实验。若未来面向正式发布，应将大型模型迁移到 release artifact、模型下载脚本或单独模型仓库，以减小源码仓库体积。

### 10.6 Python 实验链路的作用

`src/` 目录中的 Python 代码保留为历史实验和快速验证工具。`src/main.py` 与 `src/main-old.py` 是早期视觉伺服主流程，包含取流、检测、目标选择和电机控制尝试；`src/flow_viewer.py` 用于观察固定网格光流和运动点；`src/yolo_viewer.py` 用于快速查看 YOLO 检测效果；`src/benchmark.py` 用于模型速度测试；`src/estimate_compute.py` 用于估算边缘算力；`src/t265_pose_basic.py` 用于 T265 pose/IMU 基础读取验证。

模块化 Python 目录包括 `src/data_source/realsense_wrapper.py`、`src/motion_detection/optical_flow.py`、`src/motion_detection/roi_generator.py`、`src/recognition/yolo_detector.py`、`src/tracking/multi_object_tracker.py` 和 `src/utils/visualization.py`。这些模块对应了 C++ 主链路中的输入、光流、ROI、检测、跟踪和可视化概念，但 Python 版本更适合原型调试，不作为最终高帧率部署入口。

### 10.7 电机调试和安全验证脚本

`scripts/` 目录保存 yaw 电机调试脚本，包括 `motor_reset_zero.py/.sh`、`motor_diag.py/.sh`、`motor_pid_tune.py/.sh`、`motor_inner_loop_bench.py/.sh` 和 `motor_limit_sweep.py/.sh`。这些脚本分别用于软件零位复位、通信诊断、PID 调参、内环频率测试和限位扫描。它们和 C++ 的 `socketcan_gimbal.cpp` 共同构成电机侧验证工具链。

`motor-readme.md` 和顶层 `htdw_5047_canutils_zero_vel_feedback.py` 记录了 HTDW 5047 电机的协议调试背景。报告中的 30Hz 视觉外环和 1kHz 电机内环并不是抽象描述，而是通过 C++ 参数和脚本共同落地：视觉侧按目标状态更新期望角速度，电机侧以高频内环执行速度命令，并通过反馈、限位和超时保护降低实机调参风险。

### 10.8 输出证据链

C++ 程序支持视频、metrics CSV 和 targets CSV 三类输出。`--output` 会自动写入 `outputs/cvproj_<timestamp>.mp4`、`outputs/cvproj_<timestamp>.metrics.csv` 和 `outputs/cvproj_<timestamp>.targets.csv`；`--record` 与 `--telemetry` 可手动指定路径。视频输出用于直接观察检测框、ROI、track、滤波结果、帧率和控制状态，CSV 输出用于离线复盘 frame age、detection age、推理耗时、跟踪状态、目标中心、yaw 命令和反馈频率。

因此，项目的验收不应只看模型 mAP，也要看运行证据链是否完整：同一段实验应同时保存输入来源、模型格式、检测参数、目标状态、控制参数、视频结果和 CSV 指标。只有这样，才能判断问题来自模型精度、前后处理、队列延迟、跟踪漂移、控制符号还是电机执行。

### 10.9 本次整理后的交付状态

整理后的仓库主线更加清晰：README 已重写为项目框架、目录职责、依赖构建、典型命令、配置说明、模型说明、验证结果和限制说明；中间文件已删除；AI 生成相关署名和符号化注释已清理；报告也补充了代码级落地章节。当前仓库仍保留多个实验模型和历史 Python 代码，这是为了保证验收和复现实验上下文。如果后续要面向开源发布，还可以继续做两件事：一是将大模型迁移到 release 或下载脚本，二是将 Python 旧链路移动到 `legacy/` 或在 README 中进一步弱化其入口地位。

## 11. 结论、局限性与下一步

### 11.1 最终结论

本项目完成了一个从公开数据集训练到边缘端部署、再到云台视觉伺服集成的完整计算机视觉系统。最终推荐的模型是 W2：YOLOv8n WiderPerson @640，其 Precision 为 `0.8138`，Recall 为 `0.6436`，mAP@0.5 为 `0.7592`，mAP@0.5:0.95 为 `0.4880`。该模型在当前实验中优于 COCO baseline、416 输入模型、强增强模型和小学习率继续训练模型。

最终推荐的实机候选链路为：T265 单路鱼眼输入，RKNN FP16 YOLOv8n WiderPerson @640 检测，ByteTrack 生成 primary track，Kalman TargetStateFilter 输出 filtered / predicted center，T265 IMU 提供 tracker 旋转补偿预测，30Hz 视觉外环计算 yaw 目标，1kHz 电机内环执行速度控制。

这一链路比原始 YOLO 直驱更稳定，比 CPU ONNX 更快，比当前 INT8 RKNN 更可靠，也比光流驱动 tracker 更符合 yaw 云台的物理运动来源。

### 11.2 当前局限性

当前系统仍有若干限制。第一，真实电机闭环仍需继续低速实测。dry-run、视频和 CSV 已验证控制输出链路，但真实 yaw 电机在负载、惯量、线缆和 CAN 总线环境下仍需谨慎验证。

第二，检测阈值需要按现场重新标定。当前阈值在室内样例可产生稳定 track，但换到不同光照、距离和背景后可能需要重新平衡 precision 与 recall。

第三，INT8 RKNN 暂不可靠。当前 score 输出异常，除非重新导出或修正量化输出，否则不应进入默认控制链路。

第四，异步流水线仍可进一步优化。若要继续降低 frame age，应继续推进采集、预处理、检测、跟踪、控制和日志记录之间的最新帧优先流水线。

第五，YOLOv8n 是轻量模型，在极端小目标、重遮挡和复杂背景下仍有能力上限。T265 鱼眼图像也存在几何畸变，检测模型并非专门针对鱼眼数据训练。

### 11.3 下一步工作

后续建议优先做实机安全验证，而不是盲目继续训练。第一步应在低速条件下确认目标点、yaw 方向、限位和停止逻辑；第二步再分级提高最大速度，并持续观察内环频率、反馈频率和反馈丢失；第三步若仍有卡顿，应优先检查外环周期、目标状态年龄和检测年龄，再调整控制参数。

模型侧若继续优化，建议优先尝试更适合部署的输入尺寸折中，例如在速度不足时训练并验证中等输入尺寸；若追求精度上限，可使用更大 YOLO 模型作为对照，但必须同步评估 RK3588 实机延迟。部署侧若继续探索 INT8，必须先在验证集和板端视频上确认 score 分布、NMS 行为和 mAP 掉点，再考虑进入控制链路。

总体而言，本项目形成了“检测模型、部署格式、目标状态、IMU 补偿、控制输出、延时频率、限位状态和视频证据都可复盘”的边缘端视觉伺服系统。当前最有价值的方向不是继续堆叠复杂模块，而是在已有分层架构上进行安全、可测、渐进的实机闭环调参。

## 12. 参考资料

- Intel RealSense T265 产品规格：<https://www.intel.com/content/www/us/en/products/sku/192742/intel-realsense-tracking-camera-t265/specifications.html>
- Intel RealSense Tracking Camera T265 Datasheet：<https://www.intel.com/content/dam/support/us/en/documents/emerging-technologies/intel-realsense-technology/IntelRealSenseTrackingT265Datasheet.pdf>
- Radxa ROCK 5B 产品页：<https://radxa.com/products/rock5/5b/>
