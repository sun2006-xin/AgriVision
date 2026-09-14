"""
AgriVision System A — 智能大棚病虫害诊断站（FastAPI 后端）
==========================================================
核心后端服务，集成五大 AI 引擎：
  1. ResNet-18 (ONNX) — 12 类植物病害分类
  2. YOLOv8n          — 病虫害目标检测（边界框）
  3. YOLOv8n-seg      — COCO 实例分割
  4. Chinese-CLIP      — 零样本图文匹配分类
  5. Qwen2-VL-2B      — 视觉语言模型生成诊断报告

依赖: database.py（SQLite 历史/缓存/任务）、frontend.html（前端页面）
本机启动: uvicorn app_fastapi:app --host 127.0.0.1 --port 8000
"""

import os
import logging
import time
import uuid
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "hf_cache")

import cv2
import json
import asyncio
import base64
import hashlib
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import io
from ultralytics import YOLO
from transformers import (
    ChineseCLIPModel, ChineseCLIPProcessor,
    Qwen2VLForConditionalGeneration, AutoProcessor,
)
from qwen_vl_utils import process_vision_info
from database import (
    insert_diagnosis, fetch_history, clear_history,
    create_task, get_task, update_task,
    cache_get, cache_set, clear_cache,
)
from classifier_inference import CLASS_NAMES, predict_classifier
from evaluation import assess_probability_vector
from security import ApiSecurity

# ======================== 路径配置 ========================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR.parent / "models"
FRONTEND_PATH = BASE_DIR / "frontend.html"
ONNX_PATH = str(MODELS_DIR / "best_model.onnx")
YOLO_PATH = str(MODELS_DIR / "yolov8n.pt")


def _confidence_threshold():
    try:
        value = float(os.environ.get("AGRIVISION_UNCERTAINTY_THRESHOLD", "0.55"))
        return value if 0.0 <= value <= 1.0 else 0.55
    except ValueError:
        return 0.55


UNCERTAINTY_THRESHOLD = _confidence_threshold()


def _bounded_runtime_float(name, default, lower, upper):
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if lower <= value <= upper else default


UNCERTAINTY_MARGIN_THRESHOLD = _bounded_runtime_float(
    "AGRIVISION_UNCERTAINTY_MARGIN", 0.15, 0.0, 1.0
)
UNCERTAINTY_ENTROPY_THRESHOLD = _bounded_runtime_float(
    "AGRIVISION_UNCERTAINTY_ENTROPY", 0.75, 0.0, 1.0
)
OOD_MAX_PROBABILITY = _bounded_runtime_float(
    "AGRIVISION_OOD_MAX_PROBABILITY", 0.40, 0.0, 1.0
)

if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError("请先运行 convert_to_onnx.py 生成 best_model.onnx")

# ======================== 模型加载（全局单例） ========================
# ONNX Runtime 自动检测 GPU：有 CUDA 环境用 GPU，否则回退 CPU
_available_providers = ort.get_available_providers()
_onnx_providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in _available_providers else ["CPUExecutionProvider"]
print(f"[模型] ONNX Runtime 使用: {_onnx_providers}")

ort_session = ort.InferenceSession(ONNX_PATH, providers=_onnx_providers)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

detect_model = YOLO(YOLO_PATH)

# YOLOv8-seg 实例分割模型（预训练，COCO 80类）
seg_model = YOLO(str(MODELS_DIR / "yolov8n-seg.pt"))

# Chinese-CLIP 零样本分类模型（首次运行自动从 HuggingFace 下载到本地缓存）
clip_model = ChineseCLIPModel.from_pretrained(
    "OFA-Sys/chinese-clip-vit-base-patch16", cache_dir=str(MODELS_DIR / "hf_cache"),
)
clip_processor = ChineseCLIPProcessor.from_pretrained(
    "OFA-Sys/chinese-clip-vit-base-patch16", cache_dir=str(MODELS_DIR / "hf_cache"),
)
clip_model.eval()

# 12 类 CLIP 文本描述（用于零样本识别）
CLIP_TEXTS = [
    "苹果叶片上有深褐色圆形病斑，边缘清晰，中心可见黑色小点",           # [0] apple_scab
    "玉米叶片上有灰色椭圆形病斑，边缘不规则，中部灰白色",                 # [1] corn_gray_leaf_spot
    "玉米叶片从叶尖开始变褐干枯，病斑长条形，边缘红褐色",                 # [2] corn_leaf_blight
    "玉米叶片上有橙褐色粉状孢子堆，沿叶脉排列，破裂后散出铁锈色粉末",     # [3] corn_rust
    "一片健康的绿色叶片，表面光滑无斑点，颜色均匀有光泽",                 # [4] healthy
    "马铃薯叶片上有褐色同心轮纹病斑，从下部老叶开始蔓延",                 # [5] potato_early_blight
    "马铃薯叶片上有暗绿色水渍状大斑，边缘不明显，潮湿时背面长白霉",       # [6] potato_late_blight
    "南瓜叶片表面覆盖白色粉状霉层，从叶面扩散，后期变灰褐色",             # [7] squash_powdery_mildew
    "番茄叶片上有细小圆形褐色斑点，边缘隆起，中间凹陷，散生分布",         # [8] tomato_bacterial_spot
    "番茄叶片上有不规则褐色病斑，边缘深褐色，中心灰白色，呈同心轮纹状",   # [9] tomato_early_blight
    "番茄叶片上有大面积水渍状暗绿色病斑，迅速扩大变黑褐色，叶片枯死",     # [10] tomato_late_blight
    "番茄叶片出现黄绿相间的花叶斑驳，叶片皱缩畸形，生长缓慢",             # [11] tomato_mosaic_virus
]

# 预计算 CLIP 文本嵌入（启动时一次性计算，避免每次请求重复计算）
with torch.no_grad():
    text_inputs = clip_processor(text=CLIP_TEXTS, return_tensors="pt", padding=True)
    text_out = clip_model.get_text_features(**text_inputs)
    CLIP_TEXT_EMBEDS = F.normalize(
        text_out if isinstance(text_out, torch.Tensor) else text_out.pooler_output
    )

# 启动时清空结果缓存，防止旧模型/旧类别映射的缓存结果污染新服务
clear_cache()
print("[OK] 启动缓存已清空")

# ======================== LLM 懒加载（Qwen2-VL-2B） ========================
_llm_model = None
_llm_processor = None


def get_llm():
    """首次调用时加载 Qwen2-VL-2B，后续复用（避免启动时占用内存）"""
    global _llm_model, _llm_processor
    if _llm_model is None:
        model_path = "Qwen/Qwen2-VL-2B-Instruct"
        _llm_processor = AutoProcessor.from_pretrained(
            model_path, cache_dir=str(MODELS_DIR / "hf_cache"),
        )
        _llm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, cache_dir=str(MODELS_DIR / "hf_cache"),
            torch_dtype=torch.float32,  # CPU 环境用 float32
            device_map="auto" if torch.cuda.is_available() else None,
        )
        _llm_model.eval()
    return _llm_model, _llm_processor


# ======================== 文件安全校验 ========================
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MAGIC = [b"\xff\xd8\xff", b"\x89PNG", b"GIF8"]
MAX_DIMENSION = 8000


def validate_image(contents: bytes):
    """校验文件大小、魔数字节、图片可解码性和分辨率"""
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, f"文件过大，最大允许 {MAX_FILE_SIZE // 1024 // 1024}MB")
    if not any(contents[:len(m)] == m for m in ALLOWED_MAGIC):
        raise HTTPException(400, "不支持的文件格式，仅允许 JPG / PNG / GIF")
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(400, "图片损坏或无法解码")
    # verify() 后图像对象可能不可用，需重新打开检查分辨率
    img = Image.open(io.BytesIO(contents))
    if max(img.width, img.height) > MAX_DIMENSION:
        raise HTTPException(400, f"图片分辨率过高，最大边不超过 {MAX_DIMENSION}px")


# ======================== Pydantic 响应模型 ========================
class BoxItem(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str = Field(alias="class")


class PredictResponse(BaseModel):
    status: str = "success"
    class_name: str = Field(alias="class")
    confidence: float
    uncertain: bool = False
    undetermined: bool = False
    abstain: bool = False
    confidence_band: str = "medium"
    decision_status: str = "known"
    ood_suspected: bool = False
    uncertainty_reason: str = "none"
    confidence_semantics: str = "uncalibrated_softmax_score"
    probabilities: dict[str, float]


class DetectResponse(BaseModel):
    status: str = "success"
    total_objects: int
    boxes: list[BoxItem]


class SegmentItem(BaseModel):
    class_name: str = Field(alias="class")
    confidence: float
    pixel_count: int
    area_ratio: float
    polygon: list[list[int]]


class SegmentResponse(BaseModel):
    status: str = "success"
    total_objects: int
    segments: list[SegmentItem]


class ClipResponse(BaseModel):
    status: str = "success"
    top_class: str
    top_score: float
    uncertain: bool = False
    undetermined: bool = False
    abstain: bool = False
    confidence_band: str = "medium"
    decision_status: str = "known"
    ood_suspected: bool = False
    uncertainty_reason: str = "none"
    confidence_semantics: str = "uncalibrated_similarity_softmax"
    scores: dict[str, float]  # {类别名: 相似度}


class ReportResponse(BaseModel):
    status: str = "success"
    report: str  # LLM 生成的自然语言诊断报告


class DiagnoseResponse(BaseModel):
    status: str = "success"
    predict: PredictResponse
    detect: DetectResponse
    segment: SegmentResponse | None = None
    clip: ClipResponse | None = None


class HistoryItem(BaseModel):
    id: int
    time: str
    class_name: str = Field(alias="class")
    confidence: float
    total_objects: int
    boxes: list[dict]


# ======================== FastAPI 实例 ========================
app = FastAPI(title="大棚病虫害诊断 API", version="2.1")
logger = logging.getLogger("agrivision.system_a")
api_security = ApiSecurity.from_env()


@app.middleware("http")
async def api_security_middleware(request: Request, call_next):
    """Protect inference/history APIs while leaving health and docs public."""
    if request.method != "OPTIONS":
        remote_addr = request.client.host if request.client else ""
        decision = api_security.authorize(request.url.path, remote_addr, request.headers)
        if decision.status == "deny":
            status_code = 503 if decision.reason == "api_auth_not_configured" else 401
            return JSONResponse(
                {"detail": "API token required" if status_code == 503 else "Invalid API token"},
                status_code=status_code,
            )
        if decision.status == "rate_limited":
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(decision.retry_after)},
            )
    return await call_next(request)


@app.middleware("http")
async def request_observation(request, call_next):
    request_id = request.headers.get("X-Request-ID", "")
    if not request_id or len(request_id) > 64 or not all(
        char.isalnum() or char in "._:-" for char in request_id
    ):
        request_id = uuid.uuid4().hex
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed event=request_failed path=%s request_id=%s", request.url.path, request_id)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed event=request_completed method=%s path=%s status=%s duration_ms=%.1f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5000,http://localhost:5000",
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================== 工具函数 ========================
def run_predict(contents: bytes) -> dict:
    """ONNX 分类推理"""
    prediction = predict_classifier(
        ort_session,
        contents,
        class_names=CLASS_NAMES,
        input_name=input_name,
        output_name=output_name,
        threshold=UNCERTAINTY_THRESHOLD,
        margin_threshold=UNCERTAINTY_MARGIN_THRESHOLD,
        entropy_threshold=UNCERTAINTY_ENTROPY_THRESHOLD,
        ood_max_probability=OOD_MAX_PROBABILITY,
    )
    confidence_info = prediction["confidence_info"]
    return {
        "class": prediction["class"],
        "confidence": prediction["confidence"],
        "uncertain": confidence_info["uncertain"],
        "undetermined": confidence_info["undetermined"],
        "abstain": confidence_info["abstain"],
        "confidence_band": confidence_info["band"],
        "decision_status": confidence_info["decision_status"],
        "ood_suspected": confidence_info["ood_suspected"],
        "uncertainty_reason": confidence_info["uncertainty_reason"],
        "confidence_semantics": "uncalibrated_softmax_score",
        "probabilities": {
            label: round(float(probability), 4)
            for label, probability in prediction["probabilities"].items()
        },
    }


def run_detect(contents: bytes) -> dict:
    """YOLO 目标检测"""
    img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "图片解码失败")
    results = detect_model(img_bgr, verbose=False)
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "confidence": round(float(box.conf[0]), 4),
                "class": r.names[int(box.cls[0])] if hasattr(r, "names") else str(int(box.cls[0])),
            })
    return {"total_objects": len(boxes), "boxes": boxes}


def run_segment(contents: bytes) -> dict:
    """YOLOv8-seg 实例分割：返回多边形轮廓 + 面积占比"""
    img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(400, "图片解码失败")
    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    results = seg_model(img_bgr, verbose=False)
    segments = []
    for r in results:
        if r.masks is None:
            continue
        masks = r.masks
        for i in range(len(masks)):
            # 提取多边形轮廓（从 mask 中找轮廓线）
            mask = masks.data[i].cpu().numpy()  # (H, W) float
            mask_uint8 = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            pixel_count = int(mask_uint8.sum())
            area_ratio = round(pixel_count / total_pixels, 4)

            # 取最大轮廓作为多边形
            if contours:
                largest = max(contours, key=cv2.contourArea)
                # 简化多边形（减少点数，前端绘制更高效）
                epsilon = 0.005 * cv2.arcLength(largest, True)
                simplified = cv2.approxPolyDP(largest, epsilon, True)
                polygon = simplified.reshape(-1, 2).tolist()  # [[x,y], ...]
            else:
                polygon = []

            cls_id = int(r.boxes.cls[i])
            cls_name = r.names.get(cls_id, str(cls_id))
            conf = round(float(r.boxes.conf[i]), 4)

            segments.append({
                "class": cls_name,
                "confidence": conf,
                "pixel_count": pixel_count,
                "area_ratio": area_ratio,
                "polygon": polygon,
            })

    return {"total_objects": len(segments), "segments": segments}


def run_clip(contents: bytes) -> dict:
    """Chinese-CLIP 零样本分类：图像与 12 条文本描述计算余弦相似度"""
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    with torch.no_grad():
        img_inputs = clip_processor(images=img, return_tensors="pt")
        img_out = clip_model.get_image_features(**img_inputs)
        # get_image_features 可能返回 Tensor 或 BaseModelOutputWithPooling
        img_embeds = F.normalize(
            img_out if isinstance(img_out, torch.Tensor) else img_out.pooler_output,
            dim=-1,
        )
        # 余弦相似度 = 归一化向量的点积，再 softmax 转为概率
        scores = (img_embeds @ CLIP_TEXT_EMBEDS.T).squeeze(0)
        probs = scores.softmax(dim=0)
    top_idx = int(probs.argmax())
    top_score = round(float(probs[top_idx]), 4)
    probability_map = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    confidence_info = assess_probability_vector(
        probability_map,
        threshold=UNCERTAINTY_THRESHOLD,
        margin_threshold=UNCERTAINTY_MARGIN_THRESHOLD,
        entropy_threshold=UNCERTAINTY_ENTROPY_THRESHOLD,
        ood_max_probability=OOD_MAX_PROBABILITY,
    )
    return {
        "top_class": CLASS_NAMES[top_idx],
        "top_score": top_score,
        "uncertain": confidence_info["uncertain"],
        "undetermined": confidence_info["undetermined"],
        "abstain": confidence_info["abstain"],
        "confidence_band": confidence_info["band"],
        "decision_status": confidence_info["decision_status"],
        "ood_suspected": confidence_info["ood_suspected"],
        "uncertainty_reason": confidence_info["uncertainty_reason"],
        "confidence_semantics": "uncalibrated_similarity_softmax",
        "scores": {label: round(score, 4) for label, score in probability_map.items()},
    }


def run_report(contents: bytes, pred: dict, det: dict, seg: dict, clip: dict) -> dict:
    """Qwen2-VL-2B 多模态诊断报告：结合图像 + 四引擎结果生成自然语言分析"""
    model, processor = get_llm()

    # 构造诊断摘要作为 prompt 上下文
    boxes_info = ", ".join(
        f"{b['class']}({b['confidence']:.0%})" for b in det.get("boxes", [])
    ) or "无"
    seg_info = ", ".join(
        f"{s['class']}({s['area_ratio']:.1%})" for s in seg.get("segments", [])
    ) or "无"
    clip_top3 = sorted(clip.get("scores", {}).items(), key=lambda x: -x[1])[:3]
    clip_info = ", ".join(f"{c}({s:.1%})" for c, s in clip_top3) or "无"

    prompt_text = (
        f"你是一位农业植物保护专家。以下是对一张作物叶片的 AI 分析结果：\n"
        f"- 分类模型：{pred['class']}（置信度 {pred['confidence']:.1%}，"
        f"{'不确定，需人工复核' if pred.get('uncertain') else '可作为辅助证据'}；"
        f"分数未经校准，状态={pred.get('decision_status', 'known')}，"
        f"原因={pred.get('uncertainty_reason', 'none')}）\n"
        f"- 检测模型：检测到 {det['total_objects']} 个目标，类别为 {boxes_info}\n"
        f"- 分割模型：分割出 {seg['total_objects']} 个区域，{seg_info}\n"
        f"- CLIP 零样本：Top3 相似度为 {clip_info}\n\n"
        f"注意：上述模型分数是未经校准的辅助证据，不能作为诊断真值；"
        f"若状态为 uncertain、undetermined 或 ood_suspected，必须明确建议人工复核。"
        f"LLM 只负责解释已有证据和生成建议，不得替代模型评估真值。\n"
        f"请综合以上结果，用中文给出简洁的诊断报告（200字以内），包含：\n"
        f"1. 最可能的病害/虫害名称及判断依据\n"
        f"2. 严重程度评估\n"
        f"3. 防治建议（1-2条关键措施）"
    )

    # 构造多模态消息（图像 + 文本）
    b64 = base64.b64encode(contents).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{b64}"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 生成报告
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    # 只取新生成的 token
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    report = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return {"report": report}


def compute_hash(contents: bytes) -> str:
    """计算图片 MD5 哈希"""
    return hashlib.md5(contents).hexdigest()


def background_diagnose(task_id: str, contents: bytes):
    """后台任务：执行分类+检测+分割，更新任务状态，写入缓存和历史"""
    try:
        update_task(task_id, "processing")
        pred_result = run_predict(contents)
        det_result = run_detect(contents)
        seg_result = run_segment(contents)
        clip_result = run_clip(contents)
        result = {
            "predict": pred_result,
            "detect": det_result,
            "segment": seg_result,
            "clip": clip_result,
        }
        # 写入缓存
        image_hash = compute_hash(contents)
        cache_set(image_hash, result)
        # 写入历史
        insert_diagnosis(
            class_name=pred_result["class"],
            confidence=pred_result["confidence"],
            boxes=det_result["boxes"],
            probabilities=pred_result["probabilities"],
        )
        update_task(task_id, "done", result)
    except Exception as exc:
        logger.error(
            "background_diagnose_failed event=background_diagnose_failed task_id=%s error_type=%s",
            task_id,
            type(exc).__name__,
        )
        update_task(task_id, "failed", {"error": "诊断失败，请重试"})


# ======================== 路由 ========================
@app.get("/")
async def root():
    """根路由：优先返回前端页面，方便服务器部署后直接访问"""
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH, media_type="text/html")
    return {"message": "智能大棚诊断 API 已启动", "docs": "/docs"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """分类：上传叶片图片 → 返回类别 + 置信度"""
    contents = await file.read()
    validate_image(contents)
    result = run_predict(contents)
    return PredictResponse(**result)


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    """检测：上传叶片图片 → 返回边界框 + 类别"""
    contents = await file.read()
    validate_image(contents)
    result = run_detect(contents)
    return DetectResponse(**result)


@app.post("/segment", response_model=SegmentResponse)
async def segment(file: UploadFile = File(...)):
    """分割：上传叶片图片 → 返回多边形轮廓 + 面积占比"""
    contents = await file.read()
    validate_image(contents)
    result = run_segment(contents)
    return SegmentResponse(**result)


@app.post("/clip", response_model=ClipResponse)
async def clip_endpoint(file: UploadFile = File(...)):
    """CLIP 零样本分类：上传叶片图片 → 返回与 12 类文本描述的相似度"""
    contents = await file.read()
    validate_image(contents)
    result = await asyncio.to_thread(run_clip, contents)
    return ClipResponse(**result)


@app.post("/report", response_model=ReportResponse)
async def report_endpoint(file: UploadFile = File(...)):
    """LLM 诊断报告：上传叶片图片 → 四引擎分析 → Qwen2-VL 生成自然语言报告"""
    contents = await file.read()
    validate_image(contents)

    # 先跑四引擎，再交给 LLM 生成报告
    pred_result, det_result, seg_result, clip_result = await asyncio.gather(
        asyncio.to_thread(run_predict, contents),
        asyncio.to_thread(run_detect, contents),
        asyncio.to_thread(run_segment, contents),
        asyncio.to_thread(run_clip, contents),
    )
    result = await asyncio.to_thread(
        run_report, contents, pred_result, det_result, seg_result, clip_result
    )
    return ReportResponse(**result)


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(file: UploadFile = File(...)):
    """综合诊断：一次上传同时完成分类 + 检测 + 分割，并写入历史记录"""
    contents = await file.read()
    validate_image(contents)

    # 四引擎并行执行（线程池隔离，避免阻塞事件循环）
    pred_task = asyncio.to_thread(run_predict, contents)
    det_task = asyncio.to_thread(run_detect, contents)
    seg_task = asyncio.to_thread(run_segment, contents)
    clip_task = asyncio.to_thread(run_clip, contents)
    pred_result, det_result, seg_result, clip_result = await asyncio.gather(
        pred_task, det_task, seg_task, clip_task
    )

    # 写入 SQLite 历史
    insert_diagnosis(
        class_name=pred_result["class"],
        confidence=pred_result["confidence"],
        boxes=det_result["boxes"],
        probabilities=pred_result["probabilities"],
    )

    return DiagnoseResponse(
        predict=PredictResponse(**pred_result),
        detect=DetectResponse(**det_result),
        segment=SegmentResponse(**seg_result),
        clip=ClipResponse(**clip_result),
    )


@app.post("/diagnose/async")
async def diagnose_async(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """异步诊断：提交任务后立即返回 task_id，前端轮询获取结果"""
    contents = await file.read()
    validate_image(contents)

    image_hash = compute_hash(contents)

    # 命中缓存直接返回
    cached = cache_get(image_hash)
    if cached:
        task_id = create_task(image_hash)
        update_task(task_id, "done", cached)
        return {"task_id": task_id, "status": "done", "cached": True}

    # 创建任务并交给后台
    task_id = create_task(image_hash)
    background_tasks.add_task(background_diagnose, task_id, contents)
    return {"task_id": task_id, "status": "pending", "cached": False}


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """轮询任务结果"""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")

    resp = {"task_id": task["task_id"], "status": task["status"]}

    if task["status"] == "done" and task["result"]:
        # 写入缓存（用创建任务时记录的 image_hash）
        # 缓存已在 background_diagnose 中处理
        resp["result"] = task["result"]
    elif task["status"] == "failed":
        resp["error"] = "诊断失败，请重试"

    return resp


@app.get("/history", response_model=list[HistoryItem])
async def get_history(limit: int = 20):
    """获取诊断历史（最近 N 条）"""
    return await asyncio.to_thread(fetch_history, limit)


@app.delete("/history")
async def delete_history():
    """清空诊断历史"""
    await asyncio.to_thread(clear_history)
    return {"status": "cleared"}


@app.get("/history/export")
async def export_history():
    """导出全部诊断历史为 JSON 文件"""
    rows = await asyncio.to_thread(fetch_history, limit=99999)
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content=rows,
        headers={"Content-Disposition": "attachment; filename=diagnose_history.json"},
    )


@app.get("/history/export/csv")
async def export_history_csv():
    """导出全部诊断历史为 CSV 文件"""
    import csv as _csv
    rows = await asyncio.to_thread(fetch_history, limit=99999)
    output = io.StringIO()
    writer = _csv.writer(output)
    writer.writerow(["id", "time", "class", "confidence", "total_objects", "boxes", "probabilities"])
    for r in rows:
        writer.writerow([
            r["id"], r["time"], r["class"], r["confidence"],
            r["total_objects"],
            json.dumps(r.get("boxes", []), ensure_ascii=False),
            json.dumps(r.get("probabilities", {}), ensure_ascii=False),
        ])
    from fastapi.responses import StreamingResponse
    csv_bytes = output.getvalue().encode("utf-8-sig")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=diagnose_history.csv"},
    )


@app.get("/health")
@app.get("/health/ready")
async def health():
    """Readiness-compatible health response without triggering LLM loading."""
    checks = {
        "onnx": ort_session is not None,
        "detector": detect_model is not None,
        "segmenter": seg_model is not None,
        "clip": clip_model is not None,
    }
    ready = all(checks.values())
    payload = {
        "status": "ready" if ready else "degraded",
        "service": "system-a",
        "checks": checks,
        "llm_loaded": _llm_model is not None,
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.get("/health/live")
async def health_live():
    return {"status": "alive", "service": "system-a"}
