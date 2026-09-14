# AgriVision - 智慧农业病虫害监测与诊断平台 技术文档

> 版本: 1.0 | 开源发布: 2026-08

---

## 一、项目总览

### 1.1 项目定位

AgriVision 是一个面向温室作物病虫害监测的完整平台，包含两个独立但可互联的子系统：

- **System A（智能诊断站）**: 拖拽式网页诊断工具，用户上传叶片图片即可获得五引擎深度分析结果。面向所有用户，可部署到服务器供远程使用。
- **System B（实时监控系统）**: AI双验证实时监控平台，连接ESP32-CAM摄像头，每3秒采集分析，支持多摄像头大屏展示、双引擎融合检测、钉钉告警。面向温室现场部署。

两个系统各自独立运行，但 System B 提供"AI深度诊断"按钮，可将当前画面发送到 System A 进行五引擎分析并生成Qwen2-VL诊断报告。

### 1.2 项目结构

```
AgriVision/
├── docs\                          # 技术文档
│   └── 技术文档.md
├── system_a\                      # System A: 智能诊断站
│   ├── core\                      # 核心代码
│   │   ├── app_fastapi.py         # FastAPI 后端 (五引擎)
│   │   ├── frontend.html          # 前端单页面
│   │   ├── database.py            # SQLite 数据层
│   │   ├── security.py            # Bearer/X-API-Key、回环边界和有界限流
│   │   ├── deploy_ai.py           # 独立桌面推理工具(遗留)
│   │   ├── convert_to_onnx.py     # PyTorch->ONNX 导出
│   │   ├── train_classifier.py    # ResNet-18 训练脚本
│   │   └── data.yaml              # YOLO 数据集配置
│   └── models\                    # 模型文件
│       ├── best_model.onnx        # ResNet-18 ONNX (12类分类)
│       ├── best_model.pth         # ResNet-18 PyTorch (12类)
│       ├── yolov8n.pt             # YOLOv8n (2类检测)
│       └── yolov8n-seg.pt         # YOLOv8n-seg (COCO 分割)
├── system_b\                      # System B: 实时监控系统
│   ├── core\                      # 核心代码
│   │   ├── app.py                 # Flask 主应用
│   │   ├── page_templates.py       # 页面模板加载器
│   │   ├── templates\              # System B 原生 HTML/CSS/JavaScript 模板
│   │   ├── routes\                # 页面、控制、事件、视频、诊断及工程路由
│   │   ├── services\              # 认证/限流、指标等横切服务
│   │   ├── workers\               # 每摄像头有界任务队列与可停止 worker
│   │   ├── repositories\          # SQLite 历史数据访问层
│   │   ├── schemas\               # 请求白名单、类型和范围校验
│   │   ├── detection_enhanced.py  # 颜色阈值引擎
│   │   ├── dual_verifier.py       # 双引擎融合器
│   │   ├── yolo_detector.py       # YOLO引擎封装
│   │   ├── alert_notifier.py      # 钉钉告警
│   │   ├── history.py             # 历史记录管理
│   │   └── config.py              # 参数配置管理
│   ├── models\                    # 模型文件
│   │   └── best.pt                # YOLOv8 检测模型 (2类)
│   └── firmware\                  # ESP32-CAM 固件
└── cameras.json                   # 多摄像头配置 (Phase 9新增)
```

### 1.3 技术栈

| 层级 | System A | System B |
|------|----------|----------|
| Web框架 | FastAPI (异步) | Flask (多线程) |
| 前端 | 纯HTML/JS单文件 | Flask内嵌HTML(4个Tab) |
| 数据库 | SQLite | SQLite（旧 history.json 首次启动迁移） |
| AI引擎1 | ResNet-18 ONNX (12类分类) | OpenCV HSV颜色阈值 |
| AI引擎2 | YOLOv8n (2类检测) | YOLOv8 (2类检测) |
| AI引擎3 | YOLOv8n-seg (COCO分割) | - |
| AI引擎4 | Chinese-CLIP (零样本12类) | - |
| AI引擎5 | Qwen2-VL-2B (LLM报告) | - |
| 融合策略 | 五引擎并行->LLM汇总 | DualVerifier双引擎融合 |
| 告警 | - | 钉钉Webhook |
| 硬件 | - | ESP32-CAM |

---

## 二、System A 详细设计

### 2.1 五引擎架构

```
用户上传 -> validate_image() ->  run_predict()    -> ONNX分类(12类)
                                run_detect()     -> YOLO检测(disease/bug)
                                run_segment()    -> COCO分割(80类)
                                run_clip()       -> CLIP零样本(12类)
                                      |
                              asyncio.gather 并行
                                      |
                              /diagnose -> 返回四引擎结果
                              /report   -> + run_report() -> Qwen2-VL生成中文报告
```

### 2.2 API 端点清单

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | / | 前端页面(由 `system_b/core/templates/main.html` 加载) |
| GET | /health | 健康检查 |
| GET | /health/live | 进程存活检查 |
| GET | /health/ready | 模型就绪检查 |
| POST | /predict | 分类(ONNX) |
| POST | /detect | 检测(YOLO) |
| POST | /segment | 分割(YOLO-seg) |
| POST | /clip | CLIP零样本 |
| POST | /diagnose | 同步四引擎诊断 |
| POST | /diagnose/async | 异步四引擎诊断(返回task_id) |
| GET | /result/{task_id} | 轮询异步结果 |
| POST | /report | 四引擎+LLM完整报告 |
| GET | /history | 查询历史 |
| DELETE | /history | 清空历史 |
| GET | /history/export | 导出JSON |
| GET | /history/export/csv | 导出CSV |

### 2.3 12分类类别

番茄早疫病、番茄晚疫病、番茄细菌性斑病、番茄花叶病毒、马铃薯早疫病、马铃薯晚疫病、玉米灰斑病、玉米叶枯病、玉米锈病、南瓜白粉病、苹果黑星病、健康。

### 2.4 模型文件说明

| 文件 | 大小 | 用途 | 来源 |
|------|------|------|------|
| best_model.onnx | 44MB | ResNet-18 12类分类 | train_classifier.py训练后convert_to_onnx.py导出 |
| best_model.pth | 44MB | ResNet-18 PyTorch权重 | train_classifier.py训练 |
| yolov8n.pt | 6.5MB | YOLOv8n 2类检测 | 自行训练（参见 data.yaml 数据集配置） |
| yolov8n-seg.pt | 7MB | YOLOv8n-seg COCO分割 | ultralytics预训练 |
| models/ (HF缓存) | ~2GB | Chinese-CLIP + Qwen2-VL | HuggingFace镜像下载 |

### 2.5 关键依赖

fastapi, uvicorn, pydantic, onnxruntime (CPU), torch, torchvision (CPU-only), ultralytics (YOLOv8), transformers (CLIP + Qwen2-VL), qwen_vl_utils, opencv-python, Pillow, numpy

### 2.6 启动方式

```bash
cd system_a/core
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

前端: 浏览器打开 frontend.html，默认连接 http://127.0.0.1:8000

---

## 三、System B 详细设计

### 3.1 双引擎架构

```
ESP32-CAM -> fetch_image() -> run_detection_once()
                                  |
                      +-----------+-----------+
                      |                       |
                颜色阈值引擎              YOLO引擎
          (detection_enhanced.py)    (yolo_detector.py)
                      |                       |
                      +---- DualVerifier -----+
                            (dual_verifier.py)
                                  |
                      TemporalFusion(5帧衰减投票 + 3帧候选确认)
                                  |
                      +-----------+-----------+
                 更新全局状态    条件保存     钉钉告警
```

### 3.2 DualVerifier 融合规则

| 情况 | 最终等级 | 置信度 |
|------|----------|--------|
| 两引擎都未检出 | 正常(0) | high |
| 至少一类两引擎一致 | 采信颜色引擎等级 | high |
| 仅YOLO检出(conf>=0.6) | 注意(1) | medium |
| 仅YOLO检出(conf<0.6) | 正常(0) | low |
| 仅颜色引擎检出 | 颜色等级-1 | low |
| 混合情况 | max(颜色,YOLO) | medium |

### 3.3 等级分类(颜色引擎)

| 等级 | 代码 | 条件 |
|------|------|------|
| 正常 | 0 | 病斑=0 且 虫害=0 |
| 注意 | 1 | 病斑1-5个 或 虫害1-3个 |
| 警告 | 2 | 病斑6-15个 或 虫害4-8个 或 面积占比>5% |
| 严重 | 3 | 病斑>15个 或 虫害>8个 或 面积占比>15% |

### 3.4 线程模型

| 组件 | 功能 | 生命周期 |
|------|------|----------|
| Flask 主线程/请求线程 | 页面与 API 请求 | 服务运行期间 |
| `CameraTaskManager` | 每个摄像头分别维护 detection、sd_sync、save_to_sd 有界队列和 worker | 服务运行期间，可停止 |
| 周期调度 worker | 向对应摄像头任务队列投递检测或同步任务 | 服务运行期间，可停止 |
| 离线事件调度器 | 有界批次同步与退出清理 | 默认关闭，显式配置后运行 |

队列满或同一摄像头同一任务正在运行时，新的任务会被拒绝而不会无限堆积；任务支持有界重试、取消和停止后重新创建 worker。`GET /api/tasks` 返回队列深度、运行状态、完成数、失败数、取消数和重试数。任务 worker 使用 daemon 线程，但服务退出时会主动调用 `stop_all()` 清空未执行队列。

### 3.5 System B 工程化模块

System B 以 `app.py` 作为依赖组装入口，业务边界分别位于以下模块：

| 目录 | 职责 |
|------|------|
| `routes/monitoring.py` | 健康探针、摄像头清单、实时/双引擎只读状态 |
| `routes/history.py` | 历史分页、趋势、统计、导出和图片读取 |
| `routes/storage.py` | SD 图片、拍照和同步任务触发/状态 |
| `routes/engineering.py` | `/metrics` 与 `/api/tasks` |
| `routes/control.py` | YOLO、告警、参数、预览和浏览器会话路由 |
| `routes/events.py` | 离线事件查询、确认、HTTP/MQTT 同步和状态 |
| `routes/video.py` | 按摄像头隔离的 MJPEG 视频流 |
| `routes/diagnosis.py` | System A 代理和上游状态检查 |
| `routes/pages.py` | 监控首页和大屏页面入口 |
| `page_templates.py` / `templates/` | 从 UTF-8 模板文件加载监控页面 |
| `services/metrics.py` | 固定标签的 Prometheus 指标注册表 |
| `services/security.py` | System B 的 Bearer/X-API-Key、回环开发边界、有界限流和媒体会话 |
| `services/storage.py` | 摄像头文件列表的单层路径、扩展名和数量校验 |
| `workers/camera_tasks.py` | 每摄像头任务队列、锁、周期调度和停止 |
| `repositories/history_repository.py` | 参数化 SQLite 查询、索引、迁移和聚合 |
| `schemas/api.py` | JSON 白名单、类型、有限值、范围和关联约束 |
| `services/temporal_fusion.py` | 每摄像头衰减加权投票、pending/stable 状态和可审计证据 |

System A 使用 `system_a/core/security.py` 实施同一 token 环境变量约定；标准启动脚本为 `deploy/start_system_a.ps1` 和 `deploy/start_system_b.ps1`。

算法评估和现场闭环的严格清单格式、分层指标、校准、OOD/不确定性边界及真实数据验收要求见 [docs/algorithm-evaluation.md](algorithm-evaluation.md)。拿到真实数据后，先用 `tools/audit_evaluation_dataset.py --check-files` 校验图片路径和 SHA-256，再使用评估 CLI 的 `--strict-provenance` 生成报告；当前仓库的示例仍是占位数据。

System B 路由 Blueprint、参数边界、任务生命周期和远程媒体认证的收口说明见 [docs/architecture-hardening.md](architecture-hardening.md)。

历史服务默认数据库是 `system_b/core/detection_logs/history.db`。旧版 `history.json` 只在数据库为空时迁移一次；日常分页、趋势和统计不会把整个 JSON 文件加载进内存。

### 3.6 API 端点清单

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | / | 监控主页面(4 Tab SPA) |
| GET | /video_feed | MJPEG视频流 |
| POST | /api/auth/session | 将 Bearer token 换成短期浏览器媒体会话 |
| GET | /api/status | 最新检测结果+SD状态 |
| GET | /api/dual_status | 双引擎融合结果 |
| GET | /api/yolo_stats | YOLO运行统计 |
| GET | /api/tasks | 每摄像头任务队列和 worker 状态 |
| GET | /metrics | Prometheus 文本指标 |
| POST | /api/yolo/config | YOLO参数配置 |
| GET | /api/comparison | 双引擎对比历史(100帧) |
| GET/POST | /api/alert/config | 钉钉告警配置 |
| GET | /api/alert/history | 告警历史 |
| POST | /api/alert/test | 测试告警 |
| GET/POST | /api/params | 检测参数读写 |
| GET | /api/params/info | 参数UI元数据 |
| POST | /api/params/reset | 重置参数 |
| POST | /api/preview_mask | Mask预览图 |
| GET | /api/history | 分页历史查询 |
| GET | /api/history/trend | 趋势数据 |
| GET | /api/history/statistics | 统计汇总 |
| GET | /api/history/export | ZIP导出 |
| GET | /history/image/<id> | 历史标注图 |
| GET | /api/sd_images | SD卡图片列表 |
| GET | /api/save_to_sd | 远程拍照 |
| GET | /api/save_status | 拍照状态 |
| GET | /api/sync_now | 手动SD同步 |
| GET | /api/sync_status | 同步状态 |
| POST | /api/offline_events/sync | 手动发送有界离线事件批次；2xx 后确认删除 |
| POST | /api/offline_events/sync_mqtt | 手动通过可选 MQTT 运行时发送有界批次；2xx 后确认删除 |
| GET | /api/offline_events/sync_status | 查看脱敏的自动同步运行状态与计数 |

离线事件同步使用 `system_b/core/process_sync_lock.py` 在同一主机上建立 SQLite 非阻塞互斥，避免多个 Flask worker 同时消费同一批事件。锁数据库位于被 Git 忽略的 `system_b/core/offline_events/` 运行目录；跨主机、多容器部署仍需外部协调机制。

HTTP sink 返回 4xx 时不会继续重试，返回 5xx 或网络异常时才进行有界重试；失败类别以固定枚举写入脱敏状态接口。

MQTT TLS 本机回归由 `tests/test_stage23_mqtt_tls_local.py` 使用临时 CA 和回环 TLS broker 完成，验证范围包括 `mqtts://`、CA 校验、MQTT v5 CONNECT、QoS 1 和队列确认；不包含公网证书链、认证或设备现场链路。

MQTT 认证本机回归由 `tests/test_stage24_mqtt_auth_local.py` 使用合成凭据验证独立 username/password 注入和认证失败 CONNACK；运行时不把凭据放入 URL 或错误响应，公网认证与 ACL 仍需部署验收。

MQTT 连接运行时对所有真实 Paho 客户端统一等待 MQTT v5 CONNACK，并检查失败原因；无认证连接也不会仅凭 TCP 建连成功就返回可用传输器。阶段 25 的失败 CONNACK 回归位于 `tests/test_stage25_mqtt_connack_validation.py`，不连接公网、不写入凭据。

初始连接重试若遇到失败 CONNACK，会先停止当前网络循环并丢弃旧 Paho client，再按原配置创建新的 client 重试；这样不会把旧 socketpair、回调或 `reconnect_on_failure` 临时状态带入下一轮。阶段 26 的生命周期回归位于 `tests/test_stage26_mqtt_retry_lifecycle.py`。

CONNACK 等待由 `connack_timeout` 控制，默认 5 秒，允许范围为 0.1–30 秒；System B 使用 `AGRIVISION_MQTT_CONNACK_TIMEOUT` 注入该值，超出范围或非数值配置回退到安全默认值，底层 API 仍会拒绝非法值。阶段 27 的参数、环境变量和实际等待值回归位于 `tests/test_stage27_mqtt_connack_timeout.py`。

QoS 1 发布确认由 `publish_timeout` 控制，默认 10 秒，允许范围为 0.1–60 秒；System B 使用 `AGRIVISION_MQTT_PUBLISH_TIMEOUT` 注入。等待超时或 `is_published()` 为假时不会确认队列事件，阶段 28 的参数、环境变量和实际等待值回归位于 `tests/test_stage28_mqtt_publish_timeout.py`。

MQTT close 回调是幂等的，并在 `disconnect()` 异常时仍尝试 `loop_stop()`；阶段 29 的生命周期回归位于 `tests/test_stage29_mqtt_close_lifecycle.py`。关闭逻辑不记录 broker 地址、凭据或异常原文。

MQTT 发布器将 Paho `RuntimeError` 与其他发布失败统一纳入有界重试；耗尽后返回失败且不确认事件，不向调用方暴露异常原文。阶段 30 的回归位于 `tests/test_stage30_mqtt_publish_runtime_error.py`。

`/api/offline_events/sync_status` 在 MQTT 启用时额外报告当前生效的 `mqtt_connack_timeout_seconds` 和 `mqtt_publish_timeout_seconds` 数值；未启用 MQTT 时为 `null`。状态接口不返回 broker URL、username、password 或异常文本。阶段 31 的回归位于 `tests/test_stage31_mqtt_status_configuration.py`。

实时监控页的事件同步摘要通过该脱敏接口显示传输类型、队列数量、成功/失败批次数和超时数值。前端使用固定字段与 `textContent` 渲染，避免把运行时配置或异常文本注入页面；阶段 32 的回归位于 `tests/test_stage32_mqtt_status_panel.py`。

阶段 33 将事件状态请求封装为独立的 `loadEventSyncStatus()`，其失败只更新固定的“事件同步状态暂不可用”提示，不传播到实时监控主刷新流程，也不展示异常原文。回归位于 `tests/test_stage33_event_status_isolation.py`；浏览器自动化需在安装 Playwright 后补做。

阶段 34 提供可选浏览器回归依赖 `requirements-ui-test.txt`。安装依赖和 Chromium 后，可在 System B 启动时运行 `python tools/browser_smoke_event_status.py`，验证状态接口失败不会破坏实时监控页面；该工具不读取 cookie、localStorage 或凭据。

阶段 35 将该 smoke 接入 `.github/workflows/public-quality.yml` 的独立 `browser-smoke` job。Job 使用本地服务、有限 curl 探活、15 分钟超时和最终清理；Chromium 以 `--disable-dev-shm-usage`、`--disable-gpu` 和 `--no-sandbox` 启动以适配受限 CI runner。生产服务不安装 UI 测试依赖。

阶段 36 的浏览器 smoke 进一步监听 `pageerror` 和 `console.error`，确保状态接口故障只产生预期的固定提示，不留下未处理前端错误；被测 503 和 favicon 响应均由测试脚本显式控制。

阶段 37 增加 `mqtt_config_status.py` 配置预检。状态接口只返回 `configured`、`scheme`、`tls`、各项是否配置以及固定 `issues` 代码；不连接 broker，也不返回任何原始配置值。远端部署必须使用 TLS，本机回环可使用明文协议进行开发验证。

阶段 38 为 MQTT 增加脱敏运行时状态闭环。状态对象只保留固定生命周期状态和固定失败类别：`not_started`、`not_configured`、`connecting`、`connected`、`connection_failed`、`publish_failed`，以及 `runtime`、`retry_exhausted`、`permanent`。手动同步与后台同步共用同一状态记录；发布重试耗尽仍不确认事件。前端通过白名单映射显示中文状态，不读取或展示异常原文、broker 地址和凭据。

阶段 39 修正手动 MQTT 同步的空队列语义：没有待发送事件时记录 `empty`，返回 `attempts=0`，不调用发布器，也不增加成功批次数。这样面板统计与实际网络行为一致。

阶段 40 将手动同步请求校验前置到 MQTT 连接之前。JSON 必须是对象，`limit` 必须为 1–100 的整数；非法请求直接返回 400，不会建立 broker 连接或触发连接重试。HTTP 与 MQTT 手动入口均不再将非法 JSON/空请求体静默视为空对象。

阶段 41 为 `OfflineEventCache.put()` 增加单条事件字节上限预检：事件先以紧凑 UTF-8 JSON 序列化，再与 `max_bytes` 比较，超限即抛出固定 `ValueError`。检查发生在临时文件创建前，因此不会覆盖旧事件，也不会依赖后置 trim 来处理单条超限数据。

阶段 42 增加 `event_schema.project_event()` 作为统一外发白名单。HTTP `EventTransport`、MQTT `MqttEventTransport`/载荷构造器以及 `/api/offline_events` 查询均先投影事件，只传输协议规定的字段；未知顶层字段和 payload 字段被丢弃，避免旧缓存或篡改缓存把 URL、凭据、图片路径等信息带出系统。

阶段 43 加强 `OfflineEventCache.list_pending()` 的读取保护：解析后必须是字典，且其中的 `event_id` 必须通过既有安全路径校验；无效 JSON、标量、缺失 ID 和不安全 ID 会被忽略但不会被删除。这样坏记录不会进入同步器或查询响应，也不会阻塞同目录中的有效事件。

阶段 44 为 `/api/offline_events/sync_mqtt` 增加固定错误码。配置预检失败返回 `mqtt_not_configured`，配置完整但创建运行时失败返回 `mqtt_connection_failed`；两种响应共用脱敏提示 `mqtt sync unavailable`，不暴露异常文本或任何运行时配置值。

阶段 45 为 `OfflineEventCache` 增加 `_event_paths_by_mtime()`，读取和 trim 均只处理普通 JSON 文件，并对 `stat()` 期间的文件系统竞态进行容错。这样临时文件、目录或并发删除不会破坏有效事件的读取、同步和容量清理。

### 第一阶段：稳定可运行验收

第一阶段统一 System A 使用 `system_a/models/yolov8n.pt`、System B 使用 `system_b/models/best.pt`；当前分支实际追踪五个小型运行时权重，HuggingFace 缓存不随 Git 发布。发布前应确认权重再分发许可；若迁移到 Git LFS 或独立模型包，需同步更新启动脚本和文档。`requirements-a.txt`、`requirements-b.txt`、`requirements-test.txt`、`requirements-mqtt.txt` 与 `requirements-ui-test.txt` 均采用精确版本锁定，Python 统一要求 3.10+，CI 使用 3.11。启动脚本通过锁定文件安装依赖。

启动 System B 后，可运行 `python tools/api_smoke_phase1.py http://127.0.0.1:5000` 验收存活、就绪、参数元信息和非法参数拒绝；System B 到 System A 的深度诊断代理使用 multipart 字段 `file`，与 System A `/report` 的 `UploadFile(file=...)` 一致。

离线事件自动同步默认关闭。设置 `AGRIVISION_EVENTS_SYNC_INTERVAL` 为正数（秒）后启用周期调度，可选 `AGRIVISION_EVENTS_SYNC_LIMIT` 控制每批 1–100 条（默认 50）；调度优先使用 MQTT，否则使用 HTTP sink。间隔为 0 或未配置传输器时不自动外发。

### 3.7 认证、限流与告警配置

System A 和 System B 使用同一组 token 环境变量。System B 的 `/api/*`、`/metrics`、`/video_feed*`、`/dataset/*` 和 `/history/image/*`，以及 System A 的推理、异步结果和历史 API 受安全层保护：

- 本机无 token 时允许回环开发请求；远程请求默认拒绝。
- 设置 `AGRIVISION_API_TOKEN` 后使用 `Authorization: Bearer <token>` 或 `X-API-Key`，token 不写入日志和响应。
- `AGRIVISION_API_RATE_LIMIT_MAX` 与 `AGRIVISION_API_RATE_LIMIT_WINDOW` 控制按来源地址的进程内固定窗口限流。
- System B 浏览器先用 Bearer token 调用 `/api/auth/session`，得到绑定来源地址、短期有效的 HttpOnly/SameSite cookie；MJPEG、数据集和历史图片不会把 token 放入 URL。脚本客户端也可以继续直接发送认证头。
- System A 保留 `/health/live`、`/health/ready` 和 `/docs` 公开，便于探针与接口发现；其他业务路径需要认证。System B 调用 System A `/report` 时使用 `AGRIVISION_SYSTEM_A_API_TOKEN`，未设置时回退到 `AGRIVISION_API_TOKEN`。
- `AGRIVISION_ALERT_WEBHOOK_URL` 仅允许允许列表中的 HTTPS 主机；`AGRIVISION_WEBHOOK_ALLOWED_HOSTS` 显式扩展主机列表。
- `AGRIVISION_ALERT_WEBHOOK_SECRET` 仅从环境变量读取，用于 HMAC-SHA256 签名；配置接口只返回脱敏主机名，不返回 query token 或 secret。

### 3.8 启动方式

```bash
cd system_b/core
python app.py
# 默认运行在 127.0.0.1:5000；System A 默认运行在 127.0.0.1:8000；远程绑定前必须设置 AGRIVISION_API_TOKEN
```

---

## 四、详细文档索引

本项目为 System A 和 System B 分别编写了独立的技术文档，包含完整的架构说明、使用指南和部署教程：

| 文档 | 路径 | 内容 |
|------|------|------|
| System A 详细指南 | [docs/system-a-guide.md](system-a-guide.md) | 五引擎架构、API 接口文档、模型训练导出、服务器部署指南 |
| System B 详细指南 | [docs/system-b-guide.md](system-b-guide.md) | 双引擎融合、多摄像头框架、ESP32-CAM 固件、告警系统 |

---

## 五、快速启动

### 5.1 环境要求

- Python 3.10+ (推荐 3.11)
- pip (阿里云镜像加速)
- 模型文件 (见下方说明)

### 5.2 启动 System A（智能诊断站）

```batch
:: 方式一：一键启动（首次运行自动创建虚拟环境 + 安装依赖）
start_a.bat

:: 方式二：手动启动
cd system_a/core
python -m venv ..\.venv --system-site-packages
call ..\.venv\Scripts\activate
pip install -r ../../requirements-a.txt -i https://mirrors.aliyun.com/pypi/simple/
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

启动后访问 http://127.0.0.1:8000 即可看到诊断界面。

**模型文件清单** (放入 `system_a/models/`):

| 文件 | 用途 | 获取方式 |
|------|------|----------|
| best_model.onnx | ResNet-18 12类分类 | 运行 train_classifier.py 训练后 convert_to_onnx.py 导出 |
| best.pt | YOLOv8 2类检测 | System B 专用训练权重，放入 `system_b/models/` |
| yolov8n-seg.pt | YOLOv8n-seg 分割 | `pip install ultralytics && yolo segment train ...` 或 ultralytics 预训练 |
| hf_cache/ | Chinese-CLIP + Qwen2-VL | 首次启动自动从 HuggingFace 下载 (~2GB) |

### 5.3 启动 System B（实时监控系统）

```batch
:: 一键启动（首次运行自动创建虚拟环境 + 安装依赖）
start_b.bat

:: 手动启动
cd system_b/core
python -m venv ..\.venv
call ..\.venv\Scripts\activate
pip install -r ../../requirements-b.txt -i https://mirrors.aliyun.com/pypi/simple/
python app.py
```

启动后访问 http://127.0.0.1:5000 进入监控界面。

**前置配置**:
1. 修改 `cameras.json` 中的摄像头 IP 地址
2. 将 YOLO 模型放入 `system_b/models/best.pt`
3. 如需 ESP32-CAM，先烧录固件 (参见 System B 文档)

### 5.4 同时启动两个系统

```batch
start_all.bat
```

---

## 六、服务器部署概览

当前开源版本面向本地单人使用。如需部署为公开服务器，以下是关键步骤：

### 6.1 System A 服务器部署

1. **去掉 `--reload`** — 生产环境不需要热重载
2. **Nginx 反向代理** — 处理静态文件、HTTPS、负载均衡
3. **HTTPS** — Let's Encrypt 免费证书
4. **进程管理** — systemd 或 PM2 保证进程不退出
5. **GPU 加速** — 安装 CUDA + onnxruntime-gpu，代码自动检测
6. **模型分发** — 模型文件较大，建议用 Git LFS 或单独下载链接

详细步骤参见 [System A 文档 - 服务器部署指南](system-a-guide.md)。

### 6.2 System B 服务器部署

1. **固定 IP / 域名** — ESP32-CAM 需要能访问到服务器；对外监听前设置 `AGRIVISION_API_TOKEN`
2. **防火墙/反向代理** — 仅开放 Nginx/HTTPS 入口，System B 默认只监听回环地址
3. **多摄像头** — 修改 `cameras.json` 添加多个 ESP32-CAM；每路任务状态相互隔离
4. **历史数据** — 将 `system_b/core/detection_logs/` 挂载到持久化磁盘，数据库为 `history.db`
5. **钉钉告警** — 通过环境变量注入 URL、允许主机和签名 secret，不把 token 写入仓库
6. **容器部署** — 在项目根目录执行 `docker compose -f docker-compose.system-b.yml up --build`；Compose 强制要求 `AGRIVISION_API_TOKEN`

常用环境变量模板见 [`.env.example`](../.env.example)，标准启动脚本为 `deploy/start_system_a.ps1` 和 `deploy/start_system_b.ps1`。部署探针使用 `/health/live`，模型和摄像头就绪状态使用 `/health/ready`；后者返回 503 不代表进程已退出。

详细步骤参见 [System B 文档](system-b-guide.md)。

### 6.3 中国网络环境适配

```bash
# pip 阿里云镜像
pip install xxx -i https://mirrors.aliyun.com/pypi/simple/

# HuggingFace 镜像（System A 的 CLIP 和 Qwen2-VL 模型下载）
set HF_ENDPOINT=https://hf-mirror.com
```

---

## 七、项目亮点

| 维度 | 说明 |
|------|------|
| 多引擎融合 | System A 五引擎并行 + LLM 汇总；System B 双引擎 DualVerifier 融合 |
| 边缘 + 云端 | ESP32-CAM 本地采集 → System B 实时分析 → System A 可部署服务器远程诊断 |
| 模型部署 | ONNX Runtime (GPU 自动检测)、YOLOv8 推理、Chinese-CLIP 零样本、Qwen2-VL 本地 LLM |
| 实时系统 | 多线程检测 + 每摄像头 TemporalFusion 时序融合 + MJPEG 视频流 |
| 算法证据 | 分层混淆矩阵、ECE/Brier、现场 FP/FN、OOD/不确定状态、数据与模型 provenance 审计；真实标注集指标待验收 |
| 工程化 | 配置热更新、SQLite 历史、异步任务、MD5 缓存、钉钉告警、SD 卡同步 |
| 可扩展 | 多摄像头框架、动态添加/删除、配置驱动、A-B 浅连接 |
