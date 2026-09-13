# AgriVision System B -- 双引擎实时监控技术指南

## 目录

1. [系统概述](#1-系统概述)
2. [双引擎检测架构](#2-双引擎检测架构)
3. [多摄像头框架](#3-多摄像头框架)
4. [Web 监控界面](#4-web-监控界面)
5. [告警系统](#5-告警系统)
6. [SD 卡同步](#6-sd-卡同步)
7. [检测参数调节](#7-检测参数调节)
8. [ESP32-CAM 固件](#8-esp32-cam-固件)
9. [API 接口文档](#9-api-接口文档)
10. [本地启动](#10-本地启动)
11. [线程模型](#11-线程模型)

---

## 1. 系统概述

System B 是 AgriVision 的**实时病虫害监控子系统**，基于 Flask 构建，提供从图像采集、双引擎检测、告警通知到数据归档的完整闭环。

### 核心能力

| 能力 | 说明 |
|------|------|
| 双引擎检测 | OpenCV 颜色阈值引擎 + YOLOv8 深度学习引擎，DualVerifier 融合决策 |
| 多摄像头 | 支持同时接入多个 ESP32-CAM，每路独立检测线程 |
| 实时监控 | MJPEG 视频流 + 2 秒轮询状态刷新 |
| 告警通知 | 钉钉 Webhook 推送，分级冷却机制 |
| 数据管理 | 30 天自动归档、分页查询、趋势图表、ZIP 导出 |
| SD 卡同步 | ESP32-CAM 自动拍照存卡，服务端定时拉取 |
| A-B 浅连接 | 一键调用 System A 五引擎深度诊断 |
| 大屏看板 | `/dashboard` 多路网格同时监控 |

### 技术栈

```
后端:  Flask + OpenCV + ultralytics YOLOv8 + threading
前端:  内嵌 SPA (HTML + CSS + JavaScript + Chart.js)
硬件:  ESP32-CAM (OV2640) + SD 卡
通信:  HTTP (MJPEG 流 / REST API) + 钉钉 Webhook
```

### 项目结构

```
system_b/
├── core/
│   ├── app.py                  # Flask 主程序 (~2800 行, 含嵌入前端)
│   ├── detection_enhanced.py   # OpenCV 颜色阈值检测引擎
│   ├── dual_verifier.py        # 双引擎融合决策器
│   ├── yolo_detector.py        # YOLOv8 检测器封装
│   ├── alert_notifier.py       # 钉钉告警通知
│   ├── history.py              # 历史记录管理
│   ├── config.py               # 参数配置管理
│   ├── config/
│   │   └── detection_params.json   # 检测参数持久化
│   └── detection_logs/
│       ├── history.json        # 历史记录数据
│       └── images/             # 标注图片存档
├── firmware/
│   └── CameraWebServer.ino     # ESP32-CAM 固件
└── .venv/                      # Python 虚拟环境
```

---

## 2. 双引擎检测架构

System B 采用**颜色阈值引擎 + YOLOv8 引擎**并行检测，通过 DualVerifier 融合两路结果输出最终决策。

### 2.1 颜色阈值引擎

> 源文件: `detection_enhanced.py`

#### 叶片分割 (8 步流水线)

```
原始图像
  → 中值滤波去噪 (3x3)
  → 绿色优势通道 (G - max(R,B) > GREEN_ADV)
  → 饱和度阈值 (S > SAT_MIN)
  → AND 合并 (绿色优势 ∩ 饱和度)
  → 形态学闭运算 (填充叶片内部空洞)
  → 形态学开运算 (去除细小噪点)
  → 膨胀 (恢复叶片边缘)
  → 轮廓面积过滤 (去除过小/过大区域)
  → 叶片掩码 (leaf_mask)
```

AND 策略的意义：单独用绿色优势通道会将黄色区域误判为叶片，单独用饱和度阈值会将土壤误判为叶片。两者取交集可精确分割绿色叶片区域。

#### 病斑检测

在叶片掩码范围内，使用 HSV 颜色空间分割病斑：

| 病斑类型 | 色相 H | 饱和度 S | 颜色特征 |
|----------|--------|----------|----------|
| 褐色病斑 | 0-15 | >= 50 | 低色相 + 中等饱和度 |
| 黄色病斑 | 18-32 | >= 50 | 黄绿色相 + 中等饱和度 |

每个候选区域需通过**面积过滤**（大于 `min_disease_area` 像素）和**质心验证**（质心必须在叶片掩码内）双重校验。

#### 虫害检测

在叶片掩码范围内检测**白色亮点**（虫害/虫卵）：

- 饱和度: S <= 15 (极低饱和度，接近白色)
- 明度: V >= 220 (高亮度)

同样需要面积过滤和质心验证。

#### 等级分类

采用**双阈值系统**（数量 OR 比例），取更严重的结果：

| 等级 | level_code | 触发条件 |
|------|-----------|----------|
| 正常 | 0 | 所有指标低于 notice 阈值 |
| 注意 | 1 | 病斑/虫害数量或比例超过 notice 阈值 |
| 警告 | 2 | 超过 warning 阈值 |
| 严重 | 3 | 超过 serious 阈值 |

#### 分辨率自适应

当输入图像分辨率与基准分辨率 (320x240) 不同时，系统自动调整参数：

| 参数类型 | 缩放策略 | 说明 |
|----------|----------|------|
| 面积比阈值 | 对数衰减 `1 + k * ln(base/actual)` | 高分辨率下面积比自然减小 |
| 像素面积阈值 | 线性缩放 `(actual/base)^2` | 像素数量与分辨率平方成正比 |
| 形态学核大小 | 平方根缩放 `sqrt(actual/base)` | 核大小与线性尺度成正比 |

所有核大小强制取奇数（OpenCV `morphologyEx` 要求）。

### 2.2 YOLOv8 引擎

> 源文件: `yolo_detector.py`

封装 `ultralytics` 库的 YOLOv8 模型，类别映射：

| 类别 ID | 名称 | 标注颜色 |
|---------|------|----------|
| 0 | disease (病斑) | 红色 (0,0,255) |
| 1 | bug (虫害) | 蓝色 (255,100,0) |

关键特性：
- **线程安全推理**: 使用 `_inference_lock` 保证多线程下模型推理互斥
- **单例模式**: `get_detector()` 双重检查锁，全局共享一个模型实例
- **优雅降级**: 模型加载失败时系统退化为单引擎运行

### 2.3 DualVerifier 融合决策

> 源文件: `dual_verifier.py`

融合颜色引擎和 YOLO 引擎的检测结果，输出统一的置信度等级。

#### 融合规则

| 场景 | 条件 | 融合置信度 | 输出等级 |
|------|------|-----------|----------|
| 双引擎均检出 | 颜色有异常 + YOLO 有检测 | **high** | 取颜色引擎等级 |
| 仅 YOLO 检出 (高置信) | YOLO conf >= 0.6 | **medium** | 设为"注意" |
| 仅 YOLO 检出 (低置信) | YOLO conf < 0.6 | **low** | 保持正常 |
| 仅颜色引擎检出 | 颜色有异常，YOLO 无检测 | **low** | 降一级 |
| 均未检出 | 两引擎均正常 | **high** | 正常 |

#### 可调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `yolo_conf_high` | 0.6 | YOLO 高置信阈值 |
| `yolo_conf_low` | 0.25 | YOLO 低置信阈值 |
| `agreement_boost` | 1.2 | 双引擎一致时的置信度加成 |
| `disagree_penalty` | 0.8 | 不一致时的置信度惩罚 |

### 2.4 检测稳定性

#### 滑动窗口平均

维护最近 5 帧 (`DETECTION_HISTORY_SIZE`) 的检测数值滑动平均，平滑单帧噪声：

```
avg_disease_count = mean(最近5帧的 disease_count)
avg_pest_count    = mean(最近5帧的 pest_count)
```

#### 等级防抖

连续 3 帧 (`LEVEL_CHANGE_THRESHOLD`) 判定为同一等级后才切换，避免等级频繁跳变：

```
if 连续N帧等级 == X:
    切换当前稳定等级为 X
```

---

## 3. 多摄像头框架

### 3.1 配置文件

> 文件: `cameras.json`

```json
[
  {
    "id": "cam1",
    "name": "温室1号",
    "url": "http://192.0.2.10/capture",
    "base": "http://192.0.2.10",
    "enabled": true
  }
]
```

| 字段 | 说明 |
|------|------|
| `id` | 摄像头唯一标识，用于 API 路由和状态字典键 |
| `name` | 显示名称 |
| `url` | 图像捕获端点 (ESP32-CAM 的 `/capture`) |
| `base` | 控制面板基地址 (用于拍照 `/save`、SD 列表 `/list` 等操作) |
| `enabled` | 是否启用，禁用后不启动检测线程 |

额外字段 `system_a_url` 指定 System A 的地址，用于 A-B 浅连接。

### 3.2 摄像头状态

每个摄像头在 `cameras` 字典中维护独立的状态字典：

```python
cameras[cid]["state"] = {
    "latest_frame": bytes,           # 最新标注帧 (JPEG 字节流, 用于 MJPEG 推送)
    "latest_original_image": ndarray, # 最新原始帧 (OpenCV BGR, 用于参数预览)
    "latest_result": dict,           # 最新检测结果 (等级/数量/比例)
    "latest_dual_result": dict,      # 双引擎融合结果
    "detection_history": list,       # 滑动窗口 (最近5帧数值)
    "current_stable_level": int,     # 当前防抖等级
    "comparison_history": list,      # 双引擎对比历史 (最近100帧)
    "last_error": str,               # 最近错误信息
    "frame_lock": Lock,              # 线程安全锁
}
```

### 3.3 检测循环

每个启用的摄像头启动独立的 `detection_loop` 守护线程，每 3 秒执行一次 `run_detection_once()`：

```
run_detection_once(camera_id):
  1. 从摄像头 URL 获取图像 (fetch_image)
  2. 加载当前检测参数配置
  3. 颜色引擎检测 (detect_and_annotate)
  4. YOLO 引擎检测 (yolo_detector.detect)
  5. DualVerifier 融合
  6. 滑动窗口平均
  7. 等级防抖
  8. 更新状态字典
  9. 条件保存 (level_code >= 1 时写入历史记录)
  10. 条件告警 (level_code >= 2 时触发钉钉通知)
```

---

## 4. Web 监控界面

前端以 Python 字符串形式嵌入在 `app.py` 的 `HTML_PAGE` 变量中，Flask 根路由 `/` 直接返回完整 HTML。无需额外前端构建工具。

### 4.1 四个功能标签页

| 标签页 | 功能 |
|--------|------|
| **实时监控** | MJPEG 视频流 + 状态面板 + 双引擎对比图表 |
| **参数调节** | 24 个检测参数滑块 + 4 种掩码实时预览 |
| **历史记录** | 分页查询 + 趋势折线图 + 统计卡片 + ZIP 导出 |
| **模型分析** | YOLO 统计 + 检测分布 + 推理耗时 + 引擎一致性 |

### 4.2 实时监控

- **视频流**: `<img>` 标签加载 `/video_feed/<camera_id>`，MJPEG 格式，约 2 FPS
- **状态面板**: 每 2 秒轮询 `/api/dual_status`，显示等级徽章、病斑/虫害数量、绿色覆盖率
- **双引擎对比**: Chart.js 折线图，每 10 秒刷新，对比颜色引擎和 YOLO 的检测结果
- **摄像头切换**: 下拉菜单选择摄像头，切换视频流和状态数据
- **AI 深度诊断**: 按钮触发，将当前帧发送到 System A 的 `/report` 端点，返回多引擎联合分析报告

### 4.3 参数调节

- 从 `/api/params/info` 获取参数元信息（名称/范围/步长/单位），动态生成滑块
- 用户拖动滑块时，200ms 防抖调用 `/api/preview_mask` 生成预览
- 四种预览: 叶片分割 (`leaf`)、病斑检测 (`disease`)、虫害检测 (`pest`)、标注结果 (`annotated`)
- 参数修改后点击保存，通过 POST `/api/params` 持久化到 JSON 文件

### 4.4 大屏看板

访问 `/dashboard` 进入多路网格视图：

- 自动根据启用摄像头数量选择布局: 1/2/3/4/6/9 宫格
- 每个格子显示: 摄像头名称、等级徽章、MJPEG 视频流、病斑/虫害/绿比指标
- 点击格子可全屏展开，展开后可触发 AI 深度诊断
- 每 3 秒轮询所有摄像头状态
- ESC 键关闭展开或报告弹窗

---

## 5. 告警系统

> 源文件: `alert_notifier.py`

### 5.1 钉钉 Webhook

当检测等级达到**警告** (`level_code >= 2`) 时，系统通过钉钉机器人 Webhook 推送 Markdown 格式告警消息：

```
🌿 AgriVision 告警通知
━━━━━━━━━━━━━━━━
⚠️ 等级: 警告
📊 病斑数量: 15 | 虫害数量: 8
📈 病斑比例: 12.5% | 虫害比例: 6.2%
🌱 绿色覆盖率: 45.3%
🔒 置信度: high
🕐 时间: 2025-07-15 14:30:00
```

### 5.2 冷却机制

每个告警等级独立维护冷却计时器，默认冷却 300 秒 (5 分钟)：

```python
cooldown_seconds = 300  # 同一等级在冷却期内不重复发送
```

避免持续告警轰炸，同时确保等级升级时能立即发送新告警。

### 5.3 告警历史

内存中维护最近 50 条告警记录：

```python
{
    "timestamp": "2025-07-15 14:30:00",
    "level": "警告",
    "level_code": 2,
    "disease_count": 15,
    "pest_count": 8,
    "confidence": "high",
    "sent": True
}
```

### 5.4 配置管理

通过 API 动态配置：

- `GET /api/alert/config` -- 获取当前配置
- `POST /api/alert/config` -- 更新 webhook_url / enabled / cooldown_seconds
- `POST /api/alert/test` -- 发送测试告警验证 Webhook 连通性

---

## 6. SD 卡同步

### 6.1 工作流程

SD 卡同步实现 ESP32-CAM 本地存储与服务端的图片同步，有两种触发方式：

**自动同步** (后台守护线程，每 300 秒):

```
sd_sync_loop(camera_id):
  每 300 秒调用 sync_sd_card(camera_id):
    1. GET {base}/list  → 获取 ESP32 SD 卡文件列表
    2. 对比本地已同步文件，筛选新文件
    3. 逐个 GET {base}/image?name=xxx  → 下载到本地 dataset/ 目录
    4. 更新同步状态 (syncing/new_downloaded/sync_error)
```

**手动触发** (用户点击"立即同步"按钮):

```
POST /api/sync_now → 启动临时守护线程执行 sync_sd_card()
前端轮询 /api/sync_status 直到 syncing=false
```

### 6.2 远程拍照

用户点击"拍照存卡"按钮:

```
GET /api/save_to_sd → 启动临时守护线程执行 save_to_sd_async():
  1. GET {base}/save → 触发 ESP32 立即拍照并写入 SD 卡
  2. 等待 2 秒 (确保写入完成)
  3. 更新拍照状态 (done/file/error)

前端轮询 /api/save_status 直到 done=true
拍照成功后自动触发一次 SD 卡同步
```

### 6.3 图片展示

- `/api/sd_images` -- 返回本地 `dataset/` 目录下的图片文件名列表 (按时间倒序)
- `/dataset/<filename>` -- 提供图片静态访问
- 前端相册页面每 60 秒自动刷新图片列表

---

## 7. 检测参数调节

### 7.1 参数体系

> 配置文件: `config/detection_params.json`
> 管理模块: `config.py`

系统共 24 个可调参数，分为 4 组：

#### 叶片分割参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `green_adv_threshold` | 12 | 0-50 | 绿色优势通道阈值 |
| `sat_min` | 25 | 0-100 | 饱和度最小值 |
| `close_kernel` | 15 | 3-51 | 闭运算核大小 |
| `open_kernel` | 7 | 3-51 | 开运算核大小 |
| `dilate_kernel` | 5 | 3-31 | 膨胀核大小 |
| `min_leaf_area` | 500 | 50-5000 | 最小叶片轮廓面积 |

#### 病斑检测参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `brown_h_max` | 15 | 0-30 | 褐色病斑色相上限 |
| `brown_s_min` | 50 | 0-150 | 褐色病斑饱和度下限 |
| `yellow_h_min` | 18 | 0-30 | 黄色病斑色相下限 |
| `yellow_h_max` | 32 | 15-60 | 黄色病斑色相上限 |
| `yellow_s_min` | 50 | 0-150 | 黄色病斑饱和度下限 |
| `min_disease_area` | 30 | 5-500 | 最小病斑面积 (像素) |

#### 虫害检测参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `pest_s_max` | 15 | 0-50 | 虫害饱和度上限 (白色检测) |
| `pest_v_min` | 220 | 150-255 | 虫害明度下限 |
| `min_pest_area` | 5 | 1-100 | 最小虫害面积 (像素) |

#### 等级阈值参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `disease_notice` / `warning` / `serious` | 5 / 15 / 30 | 病斑数量阈值 |
| `disease_ratio_notice` / `warning` / `serious` | 0.05 / 0.15 / 0.30 | 病斑面积比例阈值 |
| `pest_notice` / `warning` / `serious` | 3 / 10 / 20 | 虫害数量阈值 |
| `pest_ratio_notice` / `warning` / `serious` | 0.02 / 0.08 / 0.20 | 虫害面积比例阈值 |
| `green_ratio_notice` / `warning` / `serious` | 0.50 / 0.30 / 0.15 | 绿色覆盖率阈值 (低于则告警) |

### 7.2 配置持久化

`ConfigManager` 三级容错：

1. 读取 `detection_params.json`，若为合法 JSON 则加载
2. JSON 损坏时使用 `DetectionConfig` 默认值
3. 文件不存在时创建新文件并写入默认值

保存时过滤掉非标准字段（只保留 `DetectionConfig` 已知的参数），防止脏数据。

### 7.3 参数验证

`validate_params()` 对前端提交的参数进行白名单 + 类型检查，拒绝未知参数和类型错误。

---

## 8. ESP32-CAM 固件

> 源文件: `firmware/CameraWebServer.ino`

### 8.1 硬件配置

| 组件 | 规格 |
|------|------|
| 主控 | ESP32-CAM (ESP32 + OV2640) |
| 存储 | MicroSD 卡 (SD_MMC 1-bit 模式) |
| 通信 | WiFi 802.11 b/g/n |

### 8.2 关键配置

```cpp
// WiFi 连接 (需修改为实际网络)
const char* ssid = "【请修改：WiFi名称】";
const char* password = "【请修改：WiFi密码】";

// 自动拍照间隔
#define AUTO_SAVE_INTERVAL 300000  // 5 分钟

// 相机配置
sensor_t* s = esp_camera_sensor_get();
s->set_framesize(s, FRAMESIZE_SVGA);  // 800x600
// JPEG 质量: 10 (0-63, 越小质量越高)
config.jpeg_quality = 10;
```

### 8.3 启动流程

```
setup():
  1. 初始化串口
  2. 配置摄像头 (PSRAM 检测 → UXGA 初始化 → SVGA 运行时)
  3. 初始化 SD_MMC (1-bit 模式, 需跳线 GPIO2 拉高)
  4. 连接 WiFi
  5. 同步 NTP 时间 (pool.ntp.org + ntp.aliyun.com, CST-8)
  6. 打印 IP 地址

loop():
  每 AUTO_SAVE_INTERVAL (5分钟) 检查:
    if SD卡就绪 && NTP已同步:
      拍照 → 写入 /data/AUTO_YYYYMMDD_HHMMSS.jpg
      验证写入完整性 (比较写入字节数与帧长度)
```

### 8.4 HTTP 端点

ESP32-CAM 固件提供以下 HTTP 端点供 System B 调用：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/capture` | GET | 实时拍照返回 JPEG |
| `/save` | GET | 拍照并写入 SD 卡 |
| `/list` | GET | 返回 SD 卡文件列表 (JSON) |
| `/image?name=xxx` | GET | 返回 SD 卡指定图片 |

### 8.5 烧录注意

1. 使用 Arduino IDE，选择板型 "AI Thinker ESP32-CAM"
2. 需要 PSRAM 支持 (OPI PSRAM 或 QSPI PSRAM，根据实际硬件选择)
3. 首次使用需修改 WiFi 名称和密码
4. SD_MMC 1-bit 模式需要将 GPIO2 拉高 (部分板需外接上拉电阻)

---

## 9. API 接口文档

### 9.1 页面路由

| 路径 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主监控页面 (单摄像头视图) |
| `/dashboard` | GET | 大屏网格视图 (多摄像头同时监控) |

### 9.2 摄像头接口

| 路径 | 方法 | 说明 |
|------|------|------|
| `/api/cameras` | GET | 返回所有摄像头配置 |

**响应示例:**

```json
{
  "cameras": {
    "cam1": {"id": "cam1", "name": "温室1号", "enabled": true}
  }
}
```

### 9.3 状态接口

| 路径 | 方法 | 参数 | 说明 |
|------|------|------|------|
| `/api/status` | GET | `camera_id` | 颜色引擎检测结果 |
| `/api/dual_status` | GET | `camera_id` | 双引擎融合检测结果 |
| `/api/yolo_stats` | GET | - | YOLO 模型运行统计 |

`/api/dual_status` 响应字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `level` | string | 等级名称 (正常/注意/警告/严重) |
| `level_code` | int | 等级代码 (0/1/2/3) |
| `disease_count` | int | 病斑数量 |
| `white_count` | int | 虫害数量 |
| `disease_ratio` | float | 病斑面积比例 |
| `white_ratio` | float | 虫害面积比例 |
| `green_ratio` | float | 绿色覆盖率 |
| `dual` | object/null | 双引擎融合结果 (含 confidence/combined_level) |
| `yolo_enabled` | bool | YOLO 引擎开关 |
| `yolo_loaded` | bool | YOLO 模型是否加载成功 |
| `sd_sync` | object | SD 卡同步状态 |
| `error` | string/null | 最近错误信息 |

### 9.4 YOLO 控制接口

| 路径 | 方法 | 说明 |
|------|------|------|
| `/api/yolo/config` | POST | 更新 YOLO 参数 |

**请求体:**

```json
{
  "enabled": true,
  "conf_threshold": 0.5,
  "iou_threshold": 0.45,
  "dual_yolo_conf_high": 0.6,
  "dual_yolo_conf_low": 0.25
}
```

### 9.5 对比分析接口

| 路径 | 方法 | 参数 | 说明 |
|------|------|------|------|
| `/api/comparison` | GET | `camera_id`, `n` | 最近 N 帧双引擎对比数据 |

### 9.6 参数接口

| 路径 | 方法 | 说明 |
|------|------|------|
| `/api/params` | GET | 获取当前检测参数 |
| `/api/params` | POST | 保存检测参数 (JSON body) |
| `/api/params/info` | GET | 获取参数元信息 (名称/范围/步长/单位) |
| `/api/params/reset` | POST | 重置为默认参数 |

### 9.7 预览接口

| 路径 | 方法 | 参数 | 说明 |
|------|------|------|------|
| `/api/preview_mask` | POST | `camera_id` | 生成掩码预览图 |

**请求体:**

```json
{
  "type": "leaf",
  "params": { "green_adv_threshold": 12, "sat_min": 25 }
}
```

`type` 可选值: `leaf` (叶片分割) / `disease` (病斑) / `pest` (虫害) / `annotated` (标注结果)

**响应:** `image/jpeg` 二进制图片数据

### 9.8 历史记录接口

| 路径 | 方法 | 参数 | 说明 |
|------|------|------|------|
| `/api/history` | GET | `page`, `date_from`, `date_to`, `level` | 分页查询历史记录 |
| `/api/history/trend` | GET | `days` (默认7) | 趋势数据 (Chart.js 折线图) |
| `/api/history/statistics` | GET | `date_from`, `date_to` | 统计数据 (各等级记录数) |
| `/api/history/export` | GET | `date_from`, `date_to` | 导出 ZIP (含 JSON + 图片) |
| `/history/image/<record_id>` | GET | - | 查看历史标注图 |

### 9.9 SD 卡接口

| 路径 | 方法 | 参数 | 说明 |
|------|------|------|------|
| `/api/sd_images` | GET | - | 获取已同步图片列表 |
| `/dataset/<filename>` | GET | - | 图片静态访问 |
| `/api/save_to_sd` | GET | `camera_id` | 触发远程拍照 |
| `/api/save_status` | GET | - | 查询拍照进度 |
| `/api/sync_now` | GET | `camera_id` | 手动触发 SD 同步 |
| `/api/sync_status` | GET | - | 查询同步进度 |

### 9.9.1 离线事件队列

System B 会把检测摘要写入本地有界队列 `offline_events/`，用于网络恢复后的传输器消费。队列不保存图像、摄像头 URL、Webhook 或凭据；达到条数/字节上限时会淘汰最旧事件。

| 路径 | 方法 | 说明 |
|------|------|------|
| `/api/offline_events` | GET | 返回待同步事件及数量 |
| `/api/offline_events/ack` | POST | 外部传输成功后按 `event_id` 确认删除 |
| `/api/offline_events/sync` | POST | 手动发送有界批次；仅在传输返回 2xx 后确认删除 |

如需显式启用 HTTP 同步，请设置环境变量 `AGRIVISION_EVENTS_SINK_URL`。远端地址必须使用 HTTPS；仅允许 `localhost`、`127.0.0.1` 或 `::1` 使用 HTTP。服务默认不自动外发，避免部署时因误配置产生数据流出。每次请求携带由有序事件批次生成的 `Idempotency-Key`，接收端应按该键或事件 `event_id` 去重。

当前版本提供本地队列、确认接口和受限的手动 HTTP 同步；MQTT、自动调度和断网恢复联调仍未内置。同步接口在同一 System B 进程内串行执行，并发请求返回 409。

### 9.9.2 本地端到端验收

`tests/test_stage5_e2e.py` 提供不联网的内存接收端：第一次模拟接收后响应丢失，第二次以相同幂等键重试并返回 208。测试验证接收端只保留一份事件，且本地队列只在成功响应后删除。它是协议验收，不代表真实 HTTPS 服务、MQTT broker 或 ESP32 已完成联调。

### 9.10 告警接口

| 路径 | 方法 | 说明 |
|------|------|------|
| `/api/alert/config` | GET | 获取告警配置 |
| `/api/alert/config` | POST | 更新告警配置 |
| `/api/alert/history` | GET | 获取告警历史 |
| `/api/alert/test` | POST | 发送测试告警 |

### 9.11 健康检查接口

| 路径 | 方法 | 说明 |
|------|------|------|
| `/health/live` | GET | 进程存活检查，服务能响应即返回 200 |
| `/health/ready` | GET | 检查启用摄像头是否有帧、YOLO 是否就绪；降级时返回 503 |

### 9.12 A-B 浅连接接口

| 路径 | 方法 | 说明 |
|------|------|------|
| `/api/deep_diagnose` | POST | 代理调用 System A 深度诊断 |
| `/api/system_a/status` | GET | 检查 System A 是否在线 |

**深度诊断请求体:**

```json
{ "camera_id": "cam1" }
```

流程: 获取当前帧 → JPEG 编码 → POST 到 System A `/report` → 返回诊断报告 (超时 180 秒)

代理请求使用 multipart 字段 `file`，与 System A `/report` 的 `UploadFile` 参数保持一致。System A 默认 CORS 仅允许本机 A/B 地址；跨主机部署时应设置 `CORS_ORIGINS` 环境变量。

### 9.13 视频流接口

| 路径 | 方法 | 说明 |
|------|------|------|
| `/video_feed/<camera_id>` | GET | MJPEG 实时视频流 |
| `/video_feed` | GET | 默认摄像头视频流 |

MIME 类型: `multipart/x-mixed-replace; boundary=frame`

每帧间隔 0.5 秒 (约 2 FPS)，使用 HTTP 分块传输编码。

---

## 10. 本地启动

### 10.1 启动脚本

> 文件: `start_b.bat`

双击运行 `start_b.bat` 即可一键启动：

```bat
@echo off
title AgriVision - Monitoring System (System B)

cd /d "%~dp0system_b\core"

:: 首次运行自动创建虚拟环境并安装依赖
if not exist "..\.venv\Scripts\activate" (
    python -m venv ..\.venv
    call ..\.venv\Scripts\activate
    pip install flask opencv-python numpy requests ultralytics -i https://mirrors.aliyun.com/pypi/simple/
)

:: 启动后端
start "AgriVision-Monitor" cmd /k "cd /d "%~dp0system_b\core" && call ..\.venv\Scripts\activate && python app.py"

:: 等待初始化
timeout /t 3 /nobreak >nul

:: 自动打开浏览器
start "" "http://127.0.0.1:5000"
```

### 10.2 启动流程

1. 切换到 `system_b/core` 目录
2. 检查虚拟环境，首次运行自动创建 `.venv` 并安装依赖 (Flask, OpenCV, NumPy, requests, ultralytics)
3. 在新窗口中启动 Flask 后端 (`python app.py`)
4. 等待 3 秒初始化 (首次检测 + 后台线程启动)
5. 自动打开浏览器访问 `http://127.0.0.1:5000`

### 10.3 前置条件

| 依赖 | 说明 |
|------|------|
| Python | 系统 PATH 中可用 |
| YOLO 模型 | `models/best.pt` (相对于 `system_b/core/`) |
| cameras.json | `AgriVision/cameras.json` 配置摄像头地址 |
| ESP32-CAM | 已烧录固件并连接 WiFi (可选，无摄像头时可用测试图) |

### 10.4 启动后

- 主监控页面: `http://127.0.0.1:5000`
- 大屏看板: `http://127.0.0.1:5000/dashboard`
- 后端窗口不可关闭 (关闭即停止服务)

---

## 11. 线程模型

### 11.1 线程架构

```
┌─────────────────────────────────────────────────────────┐
│                      主线程                              │
│              Flask Web 服务器 (port 5000)                │
│     处理所有 HTTP 请求 (页面/API/视频流)                  │
│     threaded=True 支持多客户端并发                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  守护线程 (per camera, daemon=True)                      │
│  ┌─────────────────────┐  ┌─────────────────────┐       │
│  │ detection_loop      │  │ sd_sync_loop        │       │
│  │ 每 3 秒检测一次      │  │ 每 300 秒同步一次    │       │
│  │ run_detection_once() │  │ sync_sd_card()      │       │
│  └─────────────────────┘  └─────────────────────┘       │
│                                                         │
│  临时守护线程 (按需创建, 完成后销毁)                       │
│  ┌─────────────────────┐  ┌─────────────────────┐       │
│  │ save_to_sd_async    │  │ sync_sd_card        │       │
│  │ 用户点击拍照时创建    │  │ 用户点击同步时创建    │       │
│  └─────────────────────┘  └─────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

### 11.2 启动序列

```python
if __name__ == '__main__':
    # 1. 为每个启用的摄像头执行首次检测 (确保打开页面时有数据)
    for cid, cam in cameras.items():
        run_detection_once(cid)

        # 2. 启动检测守护线程 (每 3 秒循环检测)
        det_thread = threading.Thread(target=detection_loop, args=(cid,), daemon=True)
        det_thread.start()

        # 3. 启动 SD 同步守护线程 (每 300 秒循环同步)
        sd_thread = threading.Thread(target=sd_sync_loop, args=(cid,), daemon=True)
        sd_thread.start()

    # 4. 启动 Flask 服务器 (阻塞主线程)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
```

### 11.3 线程安全

| 锁 | 保护对象 | 使用场景 |
|----|----------|----------|
| `frame_lock` (per camera) | 摄像头状态字典 | 检测线程写入 / API 线程读取 |
| `sd_lock` | SD 同步状态 `sd_sync_info` | 同步线程 / API 线程 |
| `save_sd_lock` | 拍照状态 `save_sd_info` | 拍照线程 / API 线程 |
| `config_lock` | 配置管理器 | 参数读写 / 检测线程读取 |
| `_inference_lock` (YOLODetector) | YOLO 模型推理 | 多线程推理互斥 |
| `_lock` (HistoryManager) | 历史记录文件 | 写入/查询/导出互斥 |

### 11.4 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `DETECTION_HISTORY_SIZE` | 5 | 滑动窗口帧数 |
| `LEVEL_CHANGE_THRESHOLD` | 3 | 等级防抖连续帧数 |
| `COMPARISON_HISTORY_SIZE` | 100 | 双引擎对比历史帧数 |
| `SD_SYNC_INTERVAL` | 300 | SD 自动同步间隔 (秒) |
| `SAVE_INTERVAL` | 60 | 检测间隔内的保存节流 (秒) |

### 11.5 前端轮询频率

| 数据 | 间隔 | 接口 |
|------|------|------|
| 检测状态 | 2 秒 | `/api/dual_status` |
| 双引擎对比图表 | 10 秒 | `/api/comparison` |
| SD 卡图片列表 | 60 秒 | `/api/sd_images` |
| 拍照/同步进度 | 2 秒 (触发时) | `/api/save_status`, `/api/sync_status` |
