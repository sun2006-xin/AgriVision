# AgriVision System A -- 智能诊断站 技术指南

> 版本: 2.1 | 最后更新: 2025 年 7 月

---

## 目录

1. [系统概述](#1-系统概述)
2. [五引擎架构详解](#2-五引擎架构详解)
3. [API 接口文档](#3-api-接口文档)
4. [前端页面说明](#4-前端页面说明)
5. [数据层说明](#5-数据层说明)
6. [模型训练与导出](#6-模型训练与导出)
7. [本地启动](#7-本地启动)
8. [服务器部署指南](#8-服务器部署指南)

---

## 1. 系统概述

AgriVision System A 是一个面向大棚种植场景的 **智能病虫害诊断站**，采用 FastAPI 后端 + 单文件 HTML 前端的轻量架构。用户上传一张作物叶片照片，系统即可调用五大 AI 引擎完成从分类、检测到诊断报告的全链路分析。

### 核心能力

| 能力 | 说明 |
|------|------|
| 病害分类 | 12 类植物病害 / 健康，基于 ResNet-18 + ONNX Runtime |
| 目标检测 | 识别病害 / 虫害区域边界框，基于 YOLOv8n |
| 实例分割 | COCO 80 类通用分割，基于 YOLOv8n-seg |
| 零样本分类 | 图文匹配验证分类结果，基于 Chinese-CLIP |
| 诊断报告 | 多模态大模型生成自然语言报告，基于 Qwen2-VL-2B |

### 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    frontend.html (SPA)                   │
│   拖拽上传 → 异步提交 → 轮询结果 → Canvas 可视化         │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP (POST / GET)
┌──────────────────────────▼──────────────────────────────┐
│                 app_fastapi.py (FastAPI)                  │
│                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ ResNet18 │ │ YOLOv8n  │ │YOLOv8n-  │ │Chinese-  │   │
│  │  ONNX    │ │  检测    │ │ seg 分割 │ │  CLIP    │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │
│       └────────────┴────────────┴────────────┘          │
│                           │                              │
│                  ┌────────▼────────┐                     │
│                  │  Qwen2-VL-2B   │                     │
│                  │  诊断报告生成   │                     │
│                  └─────────────────┘                     │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │          database.py (SQLite)                     │   │
│  │   history 表 │ tasks 表 │ cache 表               │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 目录结构

```
system_a/
├── core/
│   ├── app_fastapi.py        # FastAPI 后端主程序（~650 行）
│   ├── database.py           # SQLite 数据层
│   ├── frontend.html         # 单文件前端（HTML/CSS/JS）
│   ├── train_classifier.py   # ResNet-18 训练脚本
│   ├── convert_to_onnx.py    # PyTorch → ONNX 导出
│   ├── data.yaml             # YOLO 数据集配置
│   └── diagnose_history.db   # SQLite 数据库文件（运行后生成）
├── models/
│   ├── best_model.onnx       # ResNet-18 ONNX 推理模型
│   ├── best_model.pth        # ResNet-18 PyTorch 权重（训练产物）
│   ├── yolov8n.pt            # YOLOv8n 目标检测模型
│   ├── yolov8n-seg.pt        # YOLOv8n-seg 实例分割模型
│   └── hf_cache/             # HuggingFace 模型缓存
│       ├── chinese-clip-vit-base-patch16/
│       └── Qwen2-VL-2B-Instruct/
└── start.bat                 # Windows 快捷启动脚本
```

---

## 2. 五引擎架构详解

### 2.1 ResNet-18 ONNX 分类引擎

**用途**: 将叶片图片分类为 12 个类别之一。

**模型详情**:

| 属性 | 值 |
|------|-----|
| 骨干网络 | ResNet-18（ImageNet 预训练） |
| 输入尺寸 | 3 x 224 x 224（RGB，ImageNet 标准化） |
| 输出 | 12 类概率分布（softmax） |
| 推理框架 | ONNX Runtime（自动检测 CUDA，回退 CPU） |
| 模型文件 | `models/best_model.onnx` |

**12 个类别**:

| 编号 | 英文名 | 中文名 |
|------|--------|--------|
| 0 | apple_scab | 苹果黑星病 |
| 1 | corn_gray_leaf_spot | 玉米灰斑病 |
| 2 | corn_leaf_blight | 玉米叶枯病 |
| 3 | corn_rust | 玉米锈病 |
| 4 | healthy | 健康 |
| 5 | potato_early_blight | 马铃薯早疫病 |
| 6 | potato_late_blight | 马铃薯晚疫病 |
| 7 | squash_powdery_mildew | 南瓜白粉病 |
| 8 | tomato_bacterial_spot | 番茄细菌性斑点病 |
| 9 | tomato_early_blight | 番茄早疫病 |
| 10 | tomato_late_blight | 番茄晚疫病 |
| 11 | tomato_mosaic_virus | 番茄花叶病毒病 |

**推理流程**:

1. 图片字节流 → PIL 打开 → RGB 转换
2. Resize(224) → ToTensor → Normalize（ImageNet 均值/标准差）
3. 添加 batch 维度 → numpy 数组
4. ONNX Runtime 推理 → 原始 logits
5. Softmax → 概率分布 → 取 argmax 作为最终类别

**关键代码**（`run_predict` 函数）:

```python
def run_predict(contents: bytes) -> dict:
    tensor = preprocess_image(contents)
    raw = ort_session.run([output_name], {input_name: tensor})[0]
    exp = np.exp(raw[0])
    probs = exp / exp.sum()           # softmax
    idx = int(np.argmax(probs))
    return {
        "class": CLASS_NAMES[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
    }
```

**GPU 加速**: ONNX Runtime 启动时自动检测可用的 ExecutionProvider。若系统安装了 CUDA 和对应的 ONNX Runtime GPU 版本，将自动使用 `CUDAExecutionProvider`，否则回退到 `CPUExecutionProvider`。

---

### 2.2 YOLOv8n 目标检测引擎

**用途**: 在叶片图片中定位病害和虫害区域，返回边界框。

**模型详情**:

| 属性 | 值 |
|------|-----|
| 模型 | YOLOv8n（Nano 版本，轻量高效） |
| 推理框架 | Ultralytics YOLO（`ultralytics` 库） |
| 类别数 | 2 类: `disease`（病害）、`pest`（虫害） |
| 模型文件 | `models/yolov8n.pt` |

**推理流程**:

1. 图片字节流 → OpenCV 解码为 BGR numpy 数组
2. YOLOv8n 推理（自动预处理：resize、归一化等）
3. 遍历检测结果，提取每个目标的信息：
   - 边界框坐标 `(x1, y1, x2, y2)`（原始像素坐标）
   - 置信度分数
   - 类别名称（`disease` 或 `pest`）

**输出结构**:

```json
{
  "total_objects": 3,
  "boxes": [
    {"x1": 120, "y1": 80, "x2": 340, "y2": 290, "confidence": 0.92, "class": "disease"},
    {"x1": 450, "y1": 200, "x2": 580, "y2": 350, "confidence": 0.87, "class": "pest"}
  ]
}
```

---

### 2.3 YOLOv8n-seg 实例分割引擎

**用途**: 对图片中的物体进行像素级分割，返回多边形轮廓和面积占比。

**模型详情**:

| 属性 | 值 |
|------|-----|
| 模型 | YOLOv8n-seg（Nano + Segmentation） |
| 类别数 | COCO 80 类（通用实例分割） |
| 模型文件 | `models/yolov8n-seg.pt` |

**推理流程**:

1. 图片字节流 → OpenCV 解码
2. YOLOv8n-seg 推理，同时输出检测框和分割 mask
3. 对每个检测目标的 mask：
   - 阈值化（> 0.5）生成二值 mask
   - `cv2.findContours` 提取外轮廓
   - `cv2.approxPolyDP` 简化多边形（epsilon = 0.5% 弧长）
   - 计算像素数和面积占比（`pixel_count / total_pixels`）

**输出结构**:

```json
{
  "total_objects": 5,
  "segments": [
    {
      "class": "person",
      "confidence": 0.88,
      "pixel_count": 15234,
      "area_ratio": 0.0312,
      "polygon": [[120, 80], [340, 85], [345, 290], [125, 288]]
    }
  ]
}
```

**说明**: 此引擎使用 COCO 预训练模型，可分割人、车、动物、植物等 80 类常见物体。在农业场景中可用于识别叶片区域、病斑面积占比等。

---

### 2.4 Chinese-CLIP 零样本分类引擎

**用途**: 通过图文匹配进行零样本分类，无需针对特定类别训练，作为 ResNet-18 分类结果的交叉验证。

**模型详情**:

| 属性 | 值 |
|------|-----|
| 模型 | `OFA-Sys/chinese-clip-vit-base-patch16` |
| 架构 | ViT-B/16 图像编码器 + 中文文本编码器 |
| 匹配方式 | 图像特征与文本特征的余弦相似度 |
| 模型缓存 | `models/hf_cache/chinese-clip-vit-base-patch16/` |

**工作原理**:

1. **启动时**：将 12 条中文病害描述文本编码为向量，归一化后缓存为 `CLIP_TEXT_EMBEDS`
2. **推理时**：将输入图片编码为图像向量，与 12 条文本向量计算余弦相似度，经 softmax 转为概率分布

**12 条文本描述**（`CLIP_TEXTS`）:

每条描述对应一个类别，用详细的中文描述该病害的视觉特征。例如：

- `[0] 苹果黑星病`: "苹果叶片上有深褐色圆形病斑，边缘清晰，中心可见黑色小点"
- `[4] 健康`: "一片健康的绿色叶片，表面光滑无斑点，颜色均匀有光泽"
- `[3] 玉米锈病`: "玉米叶片上有橙褐色粉状孢子堆，沿叶脉排列，破裂后散出铁锈色粉末"

**推理代码**:

```python
def run_clip(contents: bytes) -> dict:
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    with torch.no_grad():
        img_inputs = clip_processor(images=img, return_tensors="pt")
        img_out = clip_model.get_image_features(**img_inputs)
        img_embeds = F.normalize(img_out, dim=-1)
        scores = (img_embeds @ CLIP_TEXT_EMBEDS.T).squeeze(0)
        probs = scores.softmax(dim=0)
    top_idx = int(probs.argmax())
    return {
        "top_class": CLASS_NAMES[top_idx],
        "top_score": round(float(probs[top_idx]), 4),
        "scores": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))},
    }
```

**优势**: 无需重新训练即可调整文本描述来适配新类别；与 ResNet-18 形成互补验证。

---

### 2.5 Qwen2-VL-2B 诊断报告引擎

**用途**: 结合图片和四引擎分析结果，生成自然语言诊断报告。

**模型详情**:

| 属性 | 值 |
|------|-----|
| 模型 | `Qwen/Qwen2-VL-2B-Instruct` |
| 类型 | 视觉-语言多模态大模型（2B 参数） |
| 精度 | float32（CPU）/ 自动（CUDA） |
| 模型缓存 | `models/hf_cache/Qwen2-VL-2B-Instruct/` |
| 加载方式 | 懒加载（首次调用 `/report` 时才加载，避免启动时占用内存） |

**工作流程**:

1. 并行执行四引擎（分类 + 检测 + 分割 + CLIP），使用 `asyncio.gather` 并发
2. 将四引擎结果汇总为结构化文本摘要
3. 构造多模态 prompt（图片 base64 + 文本），要求模型以农业专家身份给出诊断
4. 生成报告（max 256 tokens，temperature=0.7，top_p=0.9）

**Prompt 模板**:

```
你是一位农业植物保护专家。以下是对一张作物叶片的 AI 分析结果：
- 分类模型：{类别}（置信度 {X%}）
- 检测模型：检测到 {N} 个目标，类别为 {boxes_info}
- 分割模型：分割出 {M} 个区域，{seg_info}
- CLIP 零样本：Top3 相似度为 {clip_info}

请综合以上结果，用中文给出简洁的诊断报告（200字以内），包含：
1. 最可能的病害/虫害名称及判断依据
2. 严重程度评估
3. 防治建议（1-2条关键措施）
```

**懒加载机制**:

```python
_llm_model = None
_llm_processor = None

def get_llm():
    """首次调用时加载 Qwen2-VL-2B，后续复用（避免启动时占用内存）"""
    global _llm_model, _llm_processor
    if _llm_model is None:
        _llm_processor = AutoProcessor.from_pretrained(...)
        _llm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            ..., torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        _llm_model.eval()
    return _llm_model, _llm_processor
```

---

## 3. API 接口文档

**基础地址**: `http://127.0.0.1:8000`（本地）或 `https://your-domain.com`（服务器部署）

**通用说明**:
- 所有 POST 接口接受 `multipart/form-data`，字段名为 `file`
- 支持格式: JPG / PNG / GIF
- 文件大小限制: 10 MB
- 图片分辨率限制: 最大边不超过 8000px
- 错误响应格式: `{"detail": "错误描述"}`

### 3.1 `GET /` -- 前端页面

返回前端 HTML 页面。浏览器直接访问根 URL 即可使用图形界面。若 `frontend.html` 不存在，返回 JSON 提示消息。

### 3.2 `POST /predict` -- 分类

上传叶片图片，返回 12 类分类结果。

**请求**:

```
POST /predict
Content-Type: multipart/form-data

file: (二进制图片文件)
```

**响应**:

```json
{
  "status": "success",
  "class": "番茄早疫病",
  "confidence": 0.9523,
  "probabilities": {
    "苹果黑星病": 0.0012,
    "玉米灰斑病": 0.0008,
    "玉米叶枯病": 0.0015,
    "玉米锈病": 0.0021,
    "健康": 0.0045,
    "马铃薯早疫病": 0.0034,
    "马铃薯晚疫病": 0.0067,
    "南瓜白粉病": 0.0019,
    "番茄细菌性斑点病": 0.0089,
    "番茄早疫病": 0.9523,
    "番茄晚疫病": 0.0112,
    "番茄花叶病毒病": 0.0055
  }
}
```

**cURL 示例**:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@leaf.jpg"
```

### 3.3 `POST /detect` -- 目标检测

上传叶片图片，返回病害/虫害边界框。

**请求**:

```
POST /detect
Content-Type: multipart/form-data

file: (二进制图片文件)
```

**响应**:

```json
{
  "status": "success",
  "total_objects": 2,
  "boxes": [
    {
      "x1": 120, "y1": 80,
      "x2": 340, "y2": 290,
      "confidence": 0.9234,
      "class": "disease"
    },
    {
      "x1": 450, "y1": 200,
      "x2": 580, "y2": 350,
      "confidence": 0.8712,
      "class": "pest"
    }
  ]
}
```

### 3.4 `POST /segment` -- 实例分割

上传叶片图片，返回多边形轮廓和面积占比。

**请求**:

```
POST /segment
Content-Type: multipart/form-data

file: (二进制图片文件)
```

**响应**:

```json
{
  "status": "success",
  "total_objects": 3,
  "segments": [
    {
      "class": "person",
      "confidence": 0.91,
      "pixel_count": 24500,
      "area_ratio": 0.048,
      "polygon": [[120, 80], [340, 85], [345, 290], [125, 288]]
    }
  ]
}
```

### 3.5 `POST /clip` -- CLIP 零样本分类

上传叶片图片，返回与 12 类中文描述的相似度。

**请求**:

```
POST /clip
Content-Type: multipart/form-data

file: (二进制图片文件)
```

**响应**:

```json
{
  "status": "success",
  "top_class": "番茄早疫病",
  "top_score": 0.3456,
  "scores": {
    "苹果黑星病": 0.0234,
    "玉米灰斑病": 0.0156,
    "玉米叶枯病": 0.0312,
    "玉米锈病": 0.0289,
    "健康": 0.0421,
    "马铃薯早疫病": 0.0876,
    "马铃薯晚疫病": 0.0654,
    "南瓜白粉病": 0.0345,
    "番茄细菌性斑点病": 0.0987,
    "番茄早疫病": 0.3456,
    "番茄晚疫病": 0.1523,
    "番茄花叶病毒病": 0.0747
  }
}
```

### 3.6 `POST /report` -- AI 诊断报告

上传叶片图片，四引擎并行分析后由 Qwen2-VL-2B 生成自然语言诊断报告。

**请求**:

```
POST /report
Content-Type: multipart/form-data

file: (二进制图片文件)
```

**响应**:

```json
{
  "status": "success",
  "report": "根据多模型综合分析，该叶片最可能患有番茄早疫病。分类模型以95.2%的置信度判定为番茄早疫病，检测模型发现2个病害区域，CLIP零样本匹配也将其列为最高相似度类别。病斑呈同心轮纹状，属于中等严重程度。建议立即摘除病叶，并喷施代森锰锌或百菌清等保护性杀菌剂，注意通风降湿。"
}
```

**注意**: 此接口会先并行运行四引擎（`asyncio.gather`），再调用 LLM 生成报告，响应时间较长（CPU 环境可能需要 30-60 秒）。

分类和 CLIP 响应包含 `uncertain`、`undetermined`、`abstain`、`confidence_band`、`decision_status`、`ood_suspected`、`uncertainty_reason` 和 `confidence_semantics` 字段。默认情况下分数低于 `0.55` 会标记为不确定，可通过环境变量 `AGRIVISION_UNCERTAINTY_THRESHOLD` 调整；同时会检查 top-1 margin 和归一化熵。`undetermined` 表示模型没有足够类别优势，`ood_suspected` 是基于分数/熵的拒识启发式；两者都不是经过独立数据集验证的 OOD 检测器。

所有 System A 模型分数目前均为未经校准的辅助分数，不能直接解释为诊断概率。评估清单、`--strict-provenance`、图片 SHA-256 审计、ECE、可靠性分箱、Brier score、按光照/设备/作物切片和现场 FP/FN 统计见 [算法评估与现场闭环](algorithm-evaluation.md)。Qwen2-VL 只负责解释已有分类、检测、分割和 CLIP 证据并生成建议，不得作为诊断真值。

### 3.7 `POST /diagnose` -- 综合诊断（同步）

一次请求同时执行分类 + 检测 + 分割 + CLIP 四引擎，结果写入历史记录。四引擎通过 `asyncio.to_thread` + `asyncio.gather` 并行执行。

**请求**:

```
POST /diagnose
Content-Type: multipart/form-data

file: (二进制图片文件)
```

**响应**:

```json
{
  "status": "success",
  "predict": {
    "class": "番茄早疫病",
    "confidence": 0.9523,
    "probabilities": { "..." : "..." }
  },
  "detect": {
    "total_objects": 2,
    "boxes": [ "..." ]
  },
  "segment": {
    "total_objects": 3,
    "segments": [ "..." ]
  },
  "clip": {
    "top_class": "番茄早疫病",
    "top_score": 0.3456,
    "scores": { "..." : "..." }
  }
}
```

### 3.8 `POST /diagnose/async` -- 异步诊断

提交异步诊断任务，立即返回 `task_id`，前端通过轮询获取结果。这是前端页面实际使用的接口。

**请求**:

```
POST /diagnose/async
Content-Type: multipart/form-data

file: (二进制图片文件)
```

**响应**（提交时）:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "pending",
  "cached": false
}
```

若图片哈希命中缓存:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "done",
  "cached": true
}
```

### 3.9 `GET /result/{task_id}` -- 查询任务结果

轮询异步任务的执行状态和结果。

**请求**: `GET /result/a1b2c3d4e5f6`

**响应**（处理中）:

```json
{ "task_id": "a1b2c3d4e5f6", "status": "processing" }
```

**响应**（完成）:

```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "done",
  "result": {
    "predict": { "class": "番茄早疫病", "confidence": 0.95, "..." : "..." },
    "detect": { "total_objects": 2, "boxes": ["..."] },
    "segment": { "total_objects": 3, "segments": ["..."] },
    "clip": { "top_class": "番茄早疫病", "top_score": 0.35, "..." : "..." }
  }
}
```

**响应**（失败）:

```json
{ "task_id": "a1b2c3d4e5f6", "status": "failed", "error": "图片解码失败" }
```

### 3.10 `GET /history` -- 诊断历史

获取最近 N 条诊断记录。

**请求**: `GET /history?limit=20`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| limit | int | 20 | 返回条数 |

**响应**:

```json
[
  {
    "id": 42,
    "time": "2025-07-15 14:30:22",
    "class": "番茄早疫病",
    "confidence": 0.9523,
    "total_objects": 2,
    "boxes": [
      {"x1": 120, "y1": 80, "x2": 340, "y2": 290, "confidence": 0.92, "class": "disease"}
    ],
    "probabilities": { "番茄早疫病": 0.9523, "..." : "..." }
  }
]
```

### 3.11 `DELETE /history` -- 清空历史

清空所有诊断历史记录。

**响应**: `{ "status": "cleared" }`

### 3.12 `GET /history/export` -- 导出历史（JSON）

导出全部诊断历史为 JSON 文件下载。响应头包含 `Content-Disposition: attachment; filename=diagnose_history.json`。

### 3.13 `GET /history/export/csv` -- 导出历史（CSV）

导出全部诊断历史为 CSV 文件下载（UTF-8 BOM 编码，Excel 兼容）。

CSV 列: `id, time, class, confidence, total_objects, boxes, probabilities`

### 3.14 `GET /health` -- 健康检查

**响应**: `{ "status": "healthy", "backend": "ONNX Runtime + YOLOv8" }`

### 3.15 `GET /docs` -- Swagger 文档

FastAPI 自动生成的交互式 API 文档（Swagger UI），可直接在浏览器中测试各接口。

---

## 4. 前端页面说明

前端为单文件 SPA（`frontend.html`），由 FastAPI 根路由 `/` 直接托管返回，无需任何构建工具。

### 4.1 核心功能

- **图片上传**: 支持点击选择和拖拽上传，上传后在 Canvas 上预览原图
- **异步诊断**: 调用 `/diagnose/async` 提交任务，每 500ms 轮询 `/result/{task_id}`，最长等待 60 秒（120 次 x 500ms）
- **结果可视化**:
  - 分类结果卡片（绿色=健康，红色=病害，橙色=虫害）
  - Canvas 叠加绘制：分割多边形（半透明填充 + 边框 + 类别标签）+ 检测框（彩色边框 + 类别+置信度标签）
  - CLIP 相似度条形图（按分数降序排列，最高分高亮显示）
- **AI 诊断报告**: 诊断完成后可点击按钮调用 `/report`，报告以打字机效果逐字显示（每 30ms 一个字符）
- **诊断历史**: 合并展示本地会话记录（含缩略图）和后端历史记录（含彩色首字占位符），支持 JSON/CSV 导出和清空

### 4.2 API 地址自动检测

前端启动时自动检测 API 地址：

```javascript
let API_BASE = (window.location.origin && window.location.origin !== 'null')
    ? window.location.origin                    // 服务器部署：自动同源
    : 'http://127.0.0.1:8000';                  // 本地 file:// 打开：回退 localhost
```

- **服务器部署**: 前端由 FastAPI 托管，`window.location.origin` 自动指向服务器地址，无需任何配置
- **本地开发**: 直接用浏览器打开 HTML 文件时，`origin` 为 `null`，回退到 `127.0.0.1:8000`

### 4.3 设置面板

点击右上角 "设置" 按钮可手动覆盖 API 地址，适用于前后端分离部署或自定义端口场景。设置仅在当前会话有效，刷新页面后恢复自动检测。

### 4.4 可视化渲染逻辑

Canvas 绘制顺序（从底到顶）:

1. 原始图片（`ctx.drawImage`）
2. 分割多边形（半透明填充 + 实线边框 + 类别+面积标签）
3. 检测边界框（彩色边框 + 类别+置信度标签）

线宽根据图片尺寸自适应：`scale = max(1, round(max(width, height) / 800))`

分割区域使用 8 色循环配色方案，每个区域填充半透明色块并绘制简化后的多边形轮廓。

---

## 5. 数据层说明

数据层由 `database.py` 实现，使用 SQLite 存储，数据库文件 `diagnose_history.db` 与脚本同目录。模块导入时自动执行 `init_db()` 创建表结构。

### 5.1 数据表结构

#### `history` 表 -- 诊断历史

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PRIMARY KEY | 自增主键 |
| created_at | TEXT | 诊断时间（格式 `YYYY-MM-DD HH:MM:SS`） |
| class_name | TEXT | 分类结果类别名 |
| confidence | REAL | 置信度（0~1） |
| total_objects | INTEGER | 检测到的目标数量 |
| boxes_json | TEXT | 检测框 JSON 数组 |
| probabilities_json | TEXT | 12 类概率分布 JSON |

#### `tasks` 表 -- 异步任务

| 字段 | 类型 | 说明 |
|------|------|------|
| task_id | TEXT PRIMARY KEY | 12 位 UUID hex 短码 |
| status | TEXT | 状态流转: `pending` → `processing` → `done` / `failed` |
| result_json | TEXT | 诊断结果 JSON（四引擎完整输出） |
| image_hash | TEXT | 图片 MD5 哈希（用于缓存关联） |
| created_at | TEXT | 创建时间 |
| finished_at | TEXT | 完成时间（仅 done/failed 时有值） |

#### `cache` 表 -- 结果缓存

| 字段 | 类型 | 说明 |
|------|------|------|
| image_hash | TEXT PRIMARY KEY | 图片 MD5 哈希 |
| result_json | TEXT | 缓存的诊断结果（四引擎完整输出） |
| created_at | TEXT | 缓存写入时间 |

### 5.2 缓存机制

- **写入时机**: 异步诊断（`/diagnose/async`）的后台任务完成后，将四引擎结果以图片 MD5 为 key 写入 `cache` 表
- **读取时机**: 新的异步诊断请求先计算图片哈希，查询 `cache` 表。命中则直接创建 `done` 状态的任务并返回 `cached: true`，跳过推理
- **清空策略**: 每次服务启动时调用 `clear_cache()` 清空全部缓存，防止旧模型或旧类别映射的缓存结果污染新服务
- **缓存粒度**: 基于图片 MD5，同一张图片（字节完全相同）才会命中

### 5.3 异步任务系统

```
客户端 POST /diagnose/async
    │
    ├─ 计算图片 MD5 → 检查缓存
    │   ├─ 命中 → 创建 task，状态直接设为 done，返回 cached: true
    │   └─ 未命中 → 创建 task（status: pending）→ 加入 BackgroundTasks
    │
    └─ 返回 { task_id, status: "pending" }

客户端 GET /result/{task_id}  （轮询，每 500ms 一次）
    │
    ├─ status: "pending"     → 后台任务尚未开始
    ├─ status: "processing"  → 四引擎正在执行
    ├─ status: "done"        → 返回完整结果
    └─ status: "failed"      → 返回错误信息
```

后台任务 `background_diagnose` 的执行流程:

1. 更新任务状态为 `processing`
2. 依次执行: `run_predict` → `run_detect` → `run_segment` → `run_clip`
3. 将结果写入缓存（`cache_set`，使用 image_hash 作为 key）
4. 写入诊断历史（`insert_diagnosis`）
5. 更新任务状态为 `done`，附带完整结果
6. 若中途异常，更新任务状态为 `failed`，附带错误信息

---

## 6. 模型训练与导出

### 6.1 ResNet-18 分类器训练

**脚本**: `core/train_classifier.py`

**数据集要求**:

目录结构需符合 `ImageFolder` 格式：

```
dataset/
├── train/
│   ├── apple_scab/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── corn_gray_leaf_spot/
│   ├── ...（共 12 个子目录，目录名对应类别）
│   └── tomato_mosaic_virus/
└── val/
    ├── apple_scab/
    ├── ...
    └── tomato_mosaic_virus/
```

**训练命令**:

```bash
python train_classifier.py --data /path/to/dataset --epochs 10 --batch-size 16
```

**训练配置**:

| 参数 | 值 |
|------|-----|
| 学习率 | 0.001 |
| 优化器 | Adam |
| 损失函数 | CrossEntropyLoss（带逆频率类别权重） |
| 输入尺寸 | 224 x 224 |
| 数据增强（训练集） | 随机水平翻转、随机旋转(15度)、颜色抖动(brightness=0.2, contrast=0.2) |
| 数据增强（验证集） | 仅 Resize + Normalize |
| 设备 | 自动检测 CUDA，回退 CPU |

**类别权重**: 训练脚本自动计算逆频率权重 `weight[c] = total / (num_classes * count[c])`，缓解类别不均衡问题。训练开始时打印各类别样本数和权重。

**断点续训**: 若当前目录已存在 `best_model.pth`，将自动加载并在此基础上继续训练。

**输出**: 训练过程中验证准确率最高的模型保存为 `best_model.pth`（PyTorch state_dict 格式）。

### 6.2 ONNX 模型导出

**脚本**: `core/convert_to_onnx.py`

**前置条件**: 先运行 `train_classifier.py` 生成 `best_model.pth`（放在 `core/` 目录下）。

**导出命令**:

```bash
python convert_to_onnx.py
```

无需额外参数。脚本自动从同目录加载 `best_model.pth`，导出到 `models/best_model.onnx`。

**导出配置**:

| 参数 | 值 |
|------|-----|
| opset_version | 11 |
| 输入名 | `input` |
| 输出名 | `output` |
| 动态轴 | batch_size（支持可变 batch） |
| constant_folding | 启用 |
| dynamo | 禁用（强制使用 legacy tracing 导出器，避免 GBK 编码问题） |

**导出流程**:

1. 构建 ResNet-18 模型结构（`model.fc = nn.Linear(512, 12)`）
2. 加载 `best_model.pth` 权重
3. 创建 dummy 输入 `(1, 3, 224, 224)`
4. `torch.onnx.export` 追踪计算图并导出

**验证导出结果**:

```bash
# 确认文件已生成
ls models/best_model.onnx

# 启动服务测试
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

### 6.3 YOLO 目标检测训练

**配置文件**: `core/data.yaml`

**数据集目录结构**:

```
dataset_for_Yolo/
├── images/
│   ├── train/       # 训练图片
│   └── val/         # 验证图片
└── labels/
    ├── train/       # YOLO 格式标注 (.txt)
    └── val/         # YOLO 格式标注 (.txt)
```

**标注格式**: 每行一个目标，格式为 `class_id cx cy w h`（归一化坐标，值域 0~1）。

- `0` = disease（病害）
- `1` = pest（虫害）

**训练命令**:

```bash
yolo detect train model=yolov8n.pt data=data.yaml epochs=100
```

训练完成后将产物 `runs/detect/train/weights/best.pt` 复制为 `models/yolov8n.pt` 即可替换模型。

---

## 7. 本地启动

### 7.1 环境准备

**Python 版本**: 3.10+

**依赖安装**:

```bash
# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 安装核心依赖
pip install fastapi uvicorn python-multipart
pip install torch torchvision
pip install ultralytics
pip install onnxruntime        # CPU 版本
# pip install onnxruntime-gpu  # GPU 版本（需 CUDA 环境）
pip install transformers
pip install qwen-vl-utils
pip install opencv-python
pip install pillow
```

### 7.2 模型文件准备

确保 `models/` 目录下存在以下文件：

```
models/
├── best_model.onnx       # 必需：运行 convert_to_onnx.py 生成
├── yolov8n.pt            # 必需：YOLOv8n 检测模型
├── yolov8n-seg.pt        # 必需：YOLOv8n-seg 分割模型
└── hf_cache/             # 必需：HuggingFace 模型缓存
    ├── chinese-clip-vit-base-patch16/
    └── Qwen2-VL-2B-Instruct/
```

> **提示**: `best_model.onnx` 需要先训练再导出。YOLO 模型首次运行时 Ultralytics 会自动下载，也可手动放置。HuggingFace 模型需预先下载并放入 `hf_cache/` 目录（代码中使用 `local_files_only=True`，不会联网下载）。

### 7.3 启动服务

**方式一: 快捷启动（Windows）**

```bash
# 双击 start.bat 或在命令行执行
system_a\start.bat
```

服务器或脚本化部署可使用项目根目录的 `deploy/start_system_a.ps1`；它默认绑定 `127.0.0.1`，远程绑定时会强制检查 `AGRIVISION_API_TOKEN`，并以单 worker 启动以避免重复加载模型。

`start.bat` 内容：

```batch
@echo off
cd /d "%~dp0core"
"%~dp0.venv\Scripts\python.exe" -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8000
pause
```

**方式二: 手动启动（开发模式，带热重载）**

```bash
cd system_a/core
uvicorn app_fastapi:app --host 127.0.0.1 --port 8000 --reload
```

**方式三: Python 直接启动**

```bash
cd system_a/core
python -c "import uvicorn; uvicorn.run('app_fastapi:app', host='127.0.0.1', port=8000, reload=True)"
```

### 7.4 访问服务

- **前端页面**: 浏览器打开 `http://127.0.0.1:8000`
- **API 文档**: 浏览器打开 `http://127.0.0.1:8000/docs`（Swagger UI 交互式文档）
- **健康检查**: `curl http://127.0.0.1:8000/health`

---

## 8. 服务器部署指南

本节介绍将 System A 部署为公网可访问服务器的完整流程。

### 8.1 基础部署：移除 --reload

生产环境务必移除 `--reload` 参数，避免文件监控带来的性能开销和不稳定性：

```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --workers 1
```

> **注意**: 由于模型加载占用大量内存（5 个 AI 引擎），建议使用 `--workers 1`（单 worker）。多 worker 会导致每个进程各加载一份模型，内存成倍增长。

### 8.2 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 上传文件大小限制（与后端 10MB 限制对齐，留余量）
    client_max_body_size 15M;

    # 反向代理到 FastAPI
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时配置（LLM 报告生成可能较慢）
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
        proxy_send_timeout 60s;
    }
}
```

### 8.3 HTTPS 配置（Let's Encrypt）

```bash
# 安装 Certbot
sudo apt install certbot python3-certbot-nginx

# 自动获取证书并配置 Nginx
sudo certbot --nginx -d your-domain.com
```

Certbot 会自动修改 Nginx 配置，添加 SSL 相关指令。证书自动续期：

```bash
# 测试续期
sudo certbot renew --dry-run

# Certbot 安装后自动配置了 systemd timer，无需手动设置 crontab
```

更新后的 Nginx 配置（Certbot 自动生成）：

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    client_max_body_size 15M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}
```

### 8.4 进程管理

#### 方式一：systemd（推荐）

创建服务文件 `/etc/systemd/system/agrivision-a.service`：

```ini
[Unit]
Description=AgriVision System A - FastAPI Backend
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/AgriVision/system_a/core
Environment=PATH=/opt/AgriVision/system_a/.venv/bin:/usr/bin
Environment=HF_ENDPOINT=https://hf-mirror.com
EnvironmentFile=/etc/agrivision/system-a.env
ExecStart=/opt/AgriVision/system_a/.venv/bin/python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

# 资源限制
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

启用并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable agrivision-a
sudo systemctl start agrivision-a

# 查看状态
sudo systemctl status agrivision-a

# 查看日志
sudo journalctl -u agrivision-a -f
```

`/etc/agrivision/system-a.env` 由部署管理员创建并限制权限，至少包含 `AGRIVISION_API_TOKEN` 和 `AGRIVISION_API_AUTH_REQUIRED=true`；不要把真实 token 写入仓库或 systemd unit 文件。

#### 方式二：PM2

```bash
# 安装 PM2
npm install -g pm2

# 启动服务
pm2 start \
  --name agrivision-a \
  --interpreter /opt/AgriVision/system_a/.venv/bin/python \
  -- \
  /opt/AgriVision/system_a/.venv/bin/uvicorn app_fastapi:app \
  --host 127.0.0.1 --port 8000 \
  --app-dir /opt/AgriVision/system_a/core

# 设置开机自启
pm2 save
pm2 startup

# 常用命令
pm2 status                # 查看状态
pm2 logs agrivision-a     # 查看日志
pm2 restart agrivision-a  # 重启
```

### 8.5 模型文件分发

当前公开分支实际包含 `system_a/models/` 和 `system_b/models/` 下的五个小型运行时权重；HuggingFace 缓存不进入 Git。发布前必须确认权重的再分发许可。若未来改用 Git LFS 或独立模型包，需同步更新 README、启动脚本、校验哈希和下载说明。以下方案用于后续迁移：

#### 方案一：对象存储 + 下载脚本

将模型上传到阿里云 OSS / 腾讯云 COS / AWS S3，提供下载脚本：

```bash
#!/bin/bash
# download_models.sh
BASE_URL="https://your-bucket.oss-cn-hangzhou.aliyuncs.com/agrivision/models"

mkdir -p models

echo "下载 ResNet-18 ONNX 模型..."
wget -O models/best_model.onnx "$BASE_URL/best_model.onnx"

echo "下载 YOLOv8n 检测模型..."
wget -O models/yolov8n.pt "$BASE_URL/yolov8n.pt"

echo "下载 YOLOv8n-seg 分割模型..."
wget -O models/yolov8n-seg.pt "$BASE_URL/yolov8n-seg.pt"

echo "下载 HuggingFace 模型缓存..."
wget -O models/hf_cache.tar.gz "$BASE_URL/hf_cache.tar.gz"
tar -xzf models/hf_cache.tar.gz -C models/
rm models/hf_cache.tar.gz

echo "模型下载完成！"
```

#### 方案二：HuggingFace 仓库

将模型上传到 HuggingFace 仓库，利用 `huggingface_hub` 下载：

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='your-username/agrivision-models',
    local_dir='models',
    repo_type='model'
)
"
```

#### 方案三：Git LFS（不推荐用于超大文件）

```bash
git lfs install
git lfs track "models/*.onnx"
git lfs track "models/*.pt"
git add .gitattributes
git add models/
git commit -m "Add model files via LFS"
```

> 注意：GitHub 免费账户的 LFS 配额为 1GB，超出后需付费。对于总大小超过 1GB 的模型，建议使用对象存储方案。

#### 方案四：压缩包 + 网盘

将 `models/` 目录打包为 `.tar.gz`，上传到网盘，在 README 中提供下载链接和解压说明：

```bash
tar -czf agrivision-models.tar.gz models/
# 上传到网盘，分享链接
```

### 8.6 GPU 加速（CUDA 配置）

#### 安装 CUDA 版 ONNX Runtime

```bash
# 卸载 CPU 版本
pip uninstall onnxruntime

# 安装 GPU 版本（需先安装 CUDA Toolkit）
pip install onnxruntime-gpu
```

#### CUDA 环境要求

| 组件 | 最低版本 |
|------|----------|
| NVIDIA 驱动 | >= 525.60（Linux）/ >= 528.33（Windows） |
| CUDA Toolkit | 11.8 或 12.x |
| cuDNN | 8.x |

#### 验证 GPU 可用

```python
import onnxruntime as ort
print(ort.get_available_providers())
# 期望输出: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

服务启动时会打印使用的 Provider：

```
[模型] ONNX Runtime 使用: ['CUDAExecutionProvider']
```

#### PyTorch GPU 加速

Chinese-CLIP 和 Qwen2-VL 使用 PyTorch 推理。确保安装了 CUDA 版 PyTorch：

```bash
# 以 CUDA 12.1 为例
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Qwen2-VL-2B 在有 GPU 时自动使用 `device_map="auto"` 将模型加载到 GPU，推理速度显著提升。

### 8.7 环境变量配置

#### HF_ENDPOINT（中国大陆镜像）

在中国大陆服务器上，HuggingFace 可能被墙。设置镜像加速：

```bash
# systemd 方式：在 service 文件中添加
Environment=HF_ENDPOINT=https://hf-mirror.com

# 全局方式
export HF_ENDPOINT=https://hf-mirror.com

# 或在启动脚本中添加
export HF_ENDPOINT=https://hf-mirror.com
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000
```

> **注意**: 如果模型文件已预先下载并缓存在 `models/hf_cache/` 中（代码使用 `local_files_only=True`），则不需要联网下载，无需设置此变量。

#### CORS_ORIGINS

默认只允许本机 System A/B 前端来源。生产环境跨域访问时应显式限制为实际域名：

```bash
# 仅允许指定域名（逗号分隔）
export CORS_ORIGINS="https://your-domain.com,https://www.your-domain.com"
```

#### HF_HOME

模型缓存目录，已在代码中硬编码设置为 `models/hf_cache`：

```python
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "..", "models", "hf_cache")
```

### 8.8 速率限制

System A 已在应用层提供认证和有界限流，和 System B 使用相同的环境变量：

```bash
# 公网或跨主机部署必须设置；本机回环开发可以暂不设置
export AGRIVISION_API_TOKEN="至少 16 个字符的随机值"
export AGRIVISION_API_AUTH_REQUIRED=true
export AGRIVISION_API_RATE_LIMIT_MAX=120
export AGRIVISION_API_RATE_LIMIT_WINDOW=60
```

`/predict`、`/detect`、`/segment`、`/clip`、`/report`、`/diagnose`、`/result/*` 和 `/history*` 需要 `Authorization: Bearer <token>` 或 `X-API-Key`。`/health/live`、`/health/ready`、`/docs` 保持公开，方便探针和接口发现。未配置 token 的远程请求会返回 503；认证失败返回 401；限流返回 429。前端只把 token 保存到当前浏览器会话，不拼接到 URL。

应用层限流适合单进程基础保护；公网仍建议在 Nginx 或网关层增加更细粒度限流、TLS、访问日志和 IP 策略。多进程/多副本部署需要把限流状态移到共享网关或 Redis。

### 8.9 防火墙与端口配置

#### Linux (UFW)

```bash
# 允许 SSH
sudo ufw allow 22/tcp

# 允许 HTTP 和 HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 不要直接暴露 8000 端口（应通过 Nginx 反代）
# 如果必须直接访问：
# sudo ufw allow 8000/tcp

# 启用防火墙
sudo ufw enable
sudo ufw status
```

#### Linux (firewalld)

```bash
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
sudo firewall-cmd --list-all
```

#### 云服务器安全组

在阿里云 / 腾讯云 / AWS 等控制台的安全组规则中：

| 方向 | 端口 | 协议 | 来源 | 说明 |
|------|------|------|------|------|
| 入站 | 22 | TCP | 你的 IP | SSH |
| 入站 | 80 | TCP | 0.0.0.0/0 | HTTP |
| 入站 | 443 | TCP | 0.0.0.0/0 | HTTPS |
| 入站 | 8000 | TCP | 不开放 | 后端直连（通过 Nginx 反代即可） |

### 8.10 完整部署检查清单

```
[ ] Python 3.10+ 已安装，虚拟环境已创建
[ ] 所有依赖已安装（FastAPI, torch, ultralytics, transformers 等）
[ ] models/ 目录下四个模型文件齐全
[ ] hf_cache/ 下 Chinese-CLIP 和 Qwen2-VL 模型已缓存
[ ] 服务可正常启动（uvicorn 无报错）
[ ] /health 接口返回正常
[ ] /health/live 和 /health/ready 探针已接入
[ ] /predict 接口可正常推理
[ ] 公网部署已设置 AGRIVISION_API_TOKEN
[ ] 认证接口和限流行为已完成回归
[ ] Nginx 反代配置完成，可正常访问
[ ] HTTPS 证书已配置（Let's Encrypt 或其他）
[ ] systemd 或 PM2 进程管理已配置，开机自启
[ ] 防火墙 / 安全组规则已配置
[ ] CORS_ORIGINS 已设置为实际域名
[ ] --reload 参数已移除
[ ] 网关层限流已配置（如适用）
[ ] 日志收集已配置（journalctl / PM2 logs）
```

---

> 本文档基于 AgriVision System A v2.1 源码编写。如有问题或建议，欢迎提交 Issue 或 Pull Request。
