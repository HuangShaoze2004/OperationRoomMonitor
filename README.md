# 耗材推流识别 + 医生身份识别

基于 RTSP 实时视频流的医疗耗材识别与医生身份辨识系统。通过手部检测触发篮子接触事件，对接触段内的耗材进行分类识别，并并行执行医生 ReID（人员重识别）和人脸识别。

## 功能概览

```
RTSP/本地视频
    │
    ├─ 手部检测 (YOLO + ByteTrack) → 篮子 ROI 接触判定
    │
    ├─ 接触段缓存 (Raw BGR 环形缓冲)
    │
    ├─ 段内耗材识别
    │   ├─ 双手 ROI 裁剪
    │   ├─ 好坏帧门控 (quality gating)
    │   └─ 耗材分类 (41 类，白名单可配)
    │
    └─ 医生身份识别 (并行)
        ├─ ReID 重识别 (YOLO11n 人体检测 + ReID 模型)
        └─ 人脸识别 (InsightFace + 图库 KNN)
```

## 目录结构

```
.
├── main_basket_stream.py          # 推流识别主入口
├── setup.sh                       # 环境与资源完整性检查
├── requirements.txt               # Python 依赖
├── record_rtsp_front_top.sh       # RTSP 录制工具脚本
├── yolo11n.pt                     # YOLO11n 人体检测权重
│
├── configs/
│   └── default_config.yaml        # 全局配置文件
│
├── src/                           # 核心模块
│   ├── stream_orchestrator.py     # 推流编排器（主 Pipeline）
│   ├── stream_ingest.py           # 流读取与帧注入
│   ├── stream_capture.py          # 视频流捕获（RTSP/本地文件）
│   ├── stream_frame_buffer.py     # 帧环形缓冲区（Raw BGR 无损）
│   ├── stream_basket_session.py   # 篮子接触会话管理
│   ├── basket_segmenter.py        # 篮子 ROI 框选与接触分段
│   ├── action_trigger_logic.py    # 接触触发逻辑（IoU + confirm/cooldown）
│   ├── hand_detector.py           # 手部检测器（YOLO + ByteTrack）
│   ├── mediapipe_hand_detector.py # MediaPipe 手部检测（备选）
│   ├── doctor_identity.py         # 医生 ReID 身份识别
│   ├── doctor_face_identity.py    # 医生人脸识别（InsightFace）
│   ├── orchestrator.py            # 离线文件识别编排器
│   ├── segments_offline_orchestrator.py  # 离线分段编排器
│   ├── config.py                  # 配置加载与解析
│   ├── pack_utils.py              # 工具函数
│   ├── paths.py                   # 路径解析
│   ├── excel_segments.py          # Excel 分段读取
│   └── tsv_segments.py            # TSV 分段读写
│
├── code/                          # 段内耗材识别 Pipeline
│   ├── repo_root.py               # 代码根路径常量
│   └── dataset.py                 # 数据集处理
│
├── doctor_identify/               # 医生人脸图库构建与识别
│   ├── run_pipeline.py            # 一键建库 + 评估
│   ├── build_gallery.py           # 构建人脸特征图库
│   ├── extract_faces.py           # 人脸检测与特征提取
│   ├── identify.py                # 单张/批量人脸识别
│   ├── doctor_manager.py          # 医生信息管理
│   ├── parse_excel.py             # Excel 医生名册解析
│   ├── evaluate.py                # 识别准确率评估
│   ├── config.py                  # 图库构建配置
│   └── run.sh                     # 一键运行脚本
│
├── scripts/
│   ├── run_rtsp_stream.sh         # 一键启动推流识别
│   └── run_doctor_identity_batch.py  # 医生身份批量识别
│
├── weights/                       # 模型权重（需自行准备或使用 Git LFS）
│   ├── hand_detect.pt             # 手部检测 YOLO 权重
│   ├── goodbad_frame.pt           # 好坏帧门控 YOLO 权重
│   └── haocai_classify.pt         # 耗材分类 YOLO 权重
│
├── input/                         # 输入数据
│   └── 视频中的商品信息表.xlsx     # 商品编码与名称映射表
│
├── output/                        # 运行时输出（不纳入版本管理）
│   ├── rtsp_stream.txt            # 耗材识别 TSV 结果
│   ├── rtsp_stream.log            # 运行日志
│   └── basket_roi_rtsp.json       # 篮子 ROI 坐标缓存
│
├── people_test/                   # 测试人员视频与截图
├── people_datasets/               # 人员数据集（用于 ReID/人脸注册）
│
├── configs/
│   └── bytetrack_hand.yaml        # ByteTrack 手部跟踪配置
│
└── doctor_identity_package/       # 医生 ReID 模型与标签（需自行准备）
    ├── doctor_reid_best.pth       # ReID 模型权重
    └── labels.csv                 # 医生 ID → 姓名映射
```

## 环境要求

| 项目 | 说明 |
|------|------|
| 操作系统 | Linux (Ubuntu 20.04+ 推荐) |
| Python | 3.10+（推荐使用 conda 环境 `yolo`） |
| GPU | NVIDIA + CUDA 11.8 / 12.x（torch 需与 CUDA 版本匹配） |
| 系统依赖 | `python3-tk`（首帧弹窗框选篮子 ROI 必需） |
| 可选 | `ffmpeg`（RTSP 录制脚本需要） |

## 快速开始

### 1. 环境安装

```bash
# 安装系统依赖
sudo apt install python3-tk ffmpeg

# 创建并激活 conda 环境
conda create -n yolo python=3.10 -y
conda activate yolo

# 安装 PyTorch（根据本机 CUDA 版本选择，示例为 CUDA 12.4）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 检查资源完整性

```bash
bash setup.sh
```

该脚本会检查所有必需的模型权重、数据文件是否就位，并给出缺失项提示。

### 3. 一键启动推流识别

```bash
# 使用默认配置
bash scripts/run_rtsp_stream.sh

# 自定义 RTSP 地址和输出路径
bash scripts/run_rtsp_stream.sh \
  rtsp://192.168.3.140:8554/front_top_test \
  output/rtsp_stream.txt \
  output/basket_roi_rtsp.json \
  output/basket_roi_saved.json
```

**参数说明：**

| 参数位置 | 说明 |
|----------|------|
| 第 1 个 | RTSP 流地址或本地 MP4 文件路径 |
| 第 2 个 | 输出 TSV 文件路径 |
| 第 3 个 | 篮子 ROI 保存路径（JSON） |
| 第 4 个 | 已有 ROI JSON 文件路径（若存在则跳过手动框选） |

**首次运行** 会弹出 Tkinter 窗口，需用鼠标框选篮子区域（ROI）。框选完成后按 Enter 确认，ROI 坐标会自动保存。

### 4. 本地 MP4 测试

将 RTSP 地址替换为本地视频路径即可离线测试：

```bash
bash scripts/run_rtsp_stream.sh \
  /path/to/test_video.mp4 \
  output/test_result.txt
```

本地文件模式下，可配置 `stream.infer_source: file` 回源 4K 分辨率进行段内识别，获得更高质量的分类结果。

## 配置说明

编辑 `configs/default_config.yaml` 可调整所有识别参数。以下为关键配置项：

### 篮子接触分段 (`basket`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `det_conf` | 0.7 | 手部检测置信度阈值 |
| `contact_iou_on` | 0.03 | 判定手进入篮子的 IoU 阈值 |
| `contact_iou_off` | 0.01 | 判定手离开篮子的 IoU 阈值 |
| `confirm_seconds` | 0.08 | 接触确认最短持续时间（秒） |
| `cooldown_seconds` | 8.0 | 两次接触之间的冷却时间（秒） |
| `segment_start_offset_sec` | 2.0 | 段起点相对接触时刻的偏移（秒） |
| `segment_end_offset_sec` | 8.0 | 段终点相对接触时刻的偏移（秒） |
| `min_segment_sec` | 4.0 | 有效段最小时长（秒），不足则丢弃 |
| `contact_tracking_enabled` | true | 是否启用 ByteTrack 手部跟踪 |

### 段内耗材识别 (`phase2` + `classification`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `phase2.det_conf` | 0.7 | 段内手部检测置信度 |
| `phase2.imgsz_det` | 1920 | 检测分辨率 |
| `phase2.pad_bottom_ratio` | 0.5 | 双手 ROI 向下扩展比例 |
| `classification.good_top1_conf_threshold` | 0.7 | 好坏帧门控阈值 |
| `classification.haocai_min_conf` | 0.8 | 耗材分类最低置信度 |
| `classification.imgsz_cls` | 224 | 耗材分类输入尺寸 |

### 医生身份识别 (`doctor_identity`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | true | 是否启用医生 ReID |
| `stream_enabled` | true | 推流模式下是否启用 |
| `checkpoint` | — | ReID 模型权重路径 |
| `labels_csv` | — | 医生 ID 到姓名的映射 CSV |
| `person_yolo_weights` | yolo11n.pt | 人体检测 YOLO 权重 |
| `person_det_conf` | 0.65 | 人体检测置信度阈值 |
| `segment_window_sec` | 3.0 | 每段医生采样窗口（秒） |

### 推流 (`stream`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rtsp` | — | 默认 RTSP 地址 |
| `rtsp_transport` | tcp | RTSP 传输协议 (tcp/udp) |
| `ring_buffer_sec` | 14.0 | 帧环形缓存时长（秒） |
| `infer_workers` | 1 | 段级并行推理线程数 |
| `infer_source` | cache | 段内识别来源 (cache: 环缓; file: 回源文件) |
| `warmup_skip_frames` | 50 | RTSP 预热丢弃帧数 |

### 输出格式 (`output`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `legacy_12_col_only` | true | 是否仅输出 12 列 TSV 格式 |

## 输出说明

### 耗材识别 TSV 文件

输出文件（默认 `output/rtsp_stream.txt`）为 TSV 格式，每行代表一次接触段识别结果：

| 列 | 字段 | 说明 |
|----|------|------|
| 1 | rank | 段序号 |
| 2 | start_sec | 段起始时间（秒） |
| 3 | end_sec | 段结束时间（秒） |
| 4 | product_id_top1 | Top-1 耗材产品编码 |
| 5 | top1_name | Top-1 耗材名称 |
| 6 | top1_conf | Top-1 置信度 |
| 7 | product_id_top2 | Top-2 耗材产品编码 |
| 8 | top2_name | Top-2 耗材名称 |
| 9 | top2_conf | Top-2 置信度 |
| 10 | product_id_top3 | Top-3 耗材产品编码 |
| 11 | top3_name | Top-3 耗材名称 |
| 12 | top3_conf | Top-3 置信度 |

文件末尾会追加一行医生身份识别汇总结果（所有段的投票统计）。

### 运行日志

日志文件（默认 `output/rtsp_stream.log`）包含完整的运行过程记录，包括：
- 模型加载状态
- 每段识别的起止时间、帧数、置信度
- 医生识别结果与置信度
- 异常与错误信息

## 模型权重

本项目依赖以下预训练模型权重：

| 文件 | 用途 | 大小（约） | 获取方式 |
|------|------|-----------|----------|
| `weights/hand_detect.pt` | 手部检测 YOLO | ~6 MB | 需自行训练/提供 |
| `weights/goodbad_frame.pt` | 好坏帧门控 YOLO | ~6 MB | 需自行训练/提供 |
| `weights/haocai_classify.pt` | 耗材分类 YOLO | ~6 MB | 需自行训练/提供 |
| `yolo11n.pt` | 人体检测 YOLO11n | ~5.6 MB | `ultralytics` 自动下载 |
| `doctor_identity_package/doctor_reid_best.pth` | 医生 ReID | — | 需自行训练/提供 |
| `doctor_identity_package/labels.csv` | 医生标签映射 | — | 需自行准备 |

> **注意**：模型权重文件建议通过 Git LFS 管理，或使用内部文件服务器分发。请勿将包含敏感信息的 `doctor_identity_package/` 提交到公开仓库。

## 医生人脸图库

`doctor_identify/` 模块提供独立的医生人脸图库构建与识别功能：

```bash
cd doctor_identify

# 1. 准备医生视频，放到 data/people_datasets/
#    命名格式: 医生姓名(颜色).mp4  例如: 张三(蓝).mp4

# 2. 准备医生名册 Excel

# 3. 一键构建图库并评估
bash run.sh
```

详细流程见 `doctor_identify/` 目录下的代码注释。

## 常用命令

### 直接调用 Python 入口

```bash
python -u main_basket_stream.py \
  --rtsp rtsp://192.168.3.140:8554/front_top_test \
  --excel input/视频中的商品信息表.xlsx \
  --out output/rtsp_stream.txt \
  --config configs/default_config.yaml \
  --save-basket-roi output/basket_roi_rtsp.json \
  --basket-roi-json output/basket_roi_rtsp.json \
  2>&1 | tee output/rtsp_stream.log
```

### 录制 RTSP 流

```bash
# 录制 60 秒
bash record_rtsp_front_top.sh 60 front_top_60s.mp4

# 一直录制直到 Ctrl+C 停止
bash record_rtsp_front_top.sh
```

## 推流识别架构

```
┌─────────────────────────────────────────────────┐
│                  StreamIngestPipeline            │
│                                                  │
│  RTSP/Camera ──► read() ──► RingBuffer           │
│                     │          │                  │
│                     ▼          ▼                  │
│              Session.update()  帧存档             │
│                     │                             │
│                     ▼                             │
│              接触触发? ──Yes──► CachedClip        │
│                     │            │                │
│                     │            ▼                │
│                     │     ThreadPoolExecutor      │
│                     │     ├─ 耗材识别              │
│                     │     └─ 医生ReID (并行)       │
│                     │            │                │
│                     ▼            ▼                │
│                  TSV 实时追加写入                   │
│                  医生投票 → 末行汇总               │
└─────────────────────────────────────────────────┘
```

### 推流段内重试机制

当段内未检测到手部或无有效耗材帧时，系统会自动逐步降低置信度阈值重试：
- 手部检测阈值每次降低 0.1
- 好坏帧/耗材分类阈值每次降低 0.1
- 直至得到有效结果或无法继续降低

### 医生人脸识别

支持基于 InsightFace 的人脸图库 KNN 识别，与 ReID 独立运行：
- 从环缓中按接触时刻提取医生窗口帧（contact → contact+5s）
- 降采样至 ~1fps（最多 5 帧）
- 多帧结果投票得出最终身份

## 常见问题

### Q: 首次运行提示 "未检测到 python3-tk"

```bash
sudo apt install python3-tk
```

### Q: RTSP 流连接失败或花屏

- 确认 RTSP 地址可访问：`ffplay rtsp://192.168.3.140:8554/front_top_test`
- 尝试切换传输协议：修改 `stream.rtsp_transport` 为 `udp`
- 增加预热丢弃帧数：调整 `stream.warmup_skip_frames`

### Q: GPU 内存不足

- 降低 `phase2.imgsz_det`（如 1280 或 960）
- 将 `device.half` 设为 `true` 启用 FP16 推理
- 减少 `stream.infer_workers` 并发数

### Q: 如何禁用医生识别

在 `configs/default_config.yaml` 中设置：
```yaml
doctor_identity:
  enabled: false
  stream_enabled: false
```

## 依赖项

| 包 | 版本 | 用途 |
|----|------|------|
| torch | >=2.0.0 | 深度学习框架 |
| torchvision | >=0.15.0 | 图像变换与预训练模型 |
| ultralytics | >=8.0.0 | YOLO 目标检测 |
| opencv-python | >=4.8.0 | 视频流读取与图像处理 |
| numpy | >=1.23.0 | 数值计算 |
| pandas | >=2.0.0 | 数据处理 |
| openpyxl | >=3.1.0 | Excel 读写 |
| PyYAML | >=6.0 | YAML 配置解析 |
| Pillow | >=10.0.0 | 图像处理 |
| mediapipe | >=0.10.0 | 手部关键点检测（备选方案） |

## License

内部项目，仅供授权人员使用。
