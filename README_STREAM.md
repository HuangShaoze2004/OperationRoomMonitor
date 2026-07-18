# RTSP 推流耗材识别 + 医生身份识别

与 `6.27` 推流逻辑一致，已集成 YOLO11n 人体检测 + ReID 医生识别。

## 快速启动

```bash
bash setup.sh
conda activate yolo && pip install -r requirements.txt
bash scripts/run_rtsp_stream.sh
```

## 当前关键配置（`configs/default_config.yaml`）

| 段 | 参数 |
|----|------|
| `phase2` | `det_conf: 0.7`, `imgsz_det: 1920`, `pad_bottom_ratio: 0.5` |
| `classification` | 好坏帧 `0.7`, 耗材最低 `0.8` |
| `basket` | 接触 `det_conf: 0.7`, IoU on `0.03` / off `0.01`, confirm `0.08s`, cooldown `8s` |
| `stream` | 段窗口 contact **+2s ~ +8s**, `ring_buffer_sec: 14`, ByteTrack 已启用 |
| `doctor_identity` | `checkpoint: doctor_identity_package/doctor_reid_best.pth`, `person_yolo_weights: yolo11n.pt`, 每段并行 ReID |

- 真 RTSP：`infer_source: cache`（环缓 raw BGR）
- 本地文件：可在 yaml 将 `stream.infer_source` 改为 `file` 回源 4K
