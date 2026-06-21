# RKNN models

This directory stores RK3588/Rock5B RKNN deployment artifacts.

Current files:

- `widerperson_yolov8n_640_int8.rknn`: INT8 RKNN exported for RK3588.
- `rknn_calib_200.txt`: calibration image list used during conversion.

The C++ application supports RKNN via:

```bash
./scripts/run_t265_yaw_real.sh configs/t265_yaw_rknn.conf --yaw-dry-run
```

Current validation result:

- RKNN runtime loads and runs this model on the NPU.
- Input tensor is `1x640x640x3 INT8 NHWC`.
- Output tensor is `1x5x8400 INT8`.
- Inference throughput on the latest recorded video is about `31-36Hz`.
- The current INT8 output score channel has max value `0`, so YOLOv8 postprocess cannot produce valid detections.

Do not use this exact INT8 model for real yaw control yet. Re-export RKNN with confidence precision preserved, preferably FP16/FP32 output or split/dequantized score output, then replace the `model=` path in `configs/t265_yaw_rknn.conf`.
