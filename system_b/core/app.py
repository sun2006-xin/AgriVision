"""
大棚病虫害实时监控系统 - 主 Web 应用
======================================
本模块是整个系统的核心入口，负责：
  1. 从 ESP32-CAM（或本地测试图片）获取作物叶片图像
  2. 调用检测算法对叶片进行病斑/虫害识别与分级
  3. 通过 Flask 提供 Web 监控界面（实时视频流 + 参数调节 + 历史记录）
  4. 管理 SD 卡图片的远程拍照与同步
"""

import os       # 操作系统相关：路径拼接、目录创建、文件列表等
import sys      # 系统路径操作：用于将虚拟环境加入模块搜索路径
import json     # JSON 解析：读取 cameras.json 多摄像头配置
import logging  # 标准日志：记录请求耗时和关联 ID

# ---------- 项目根目录与虚拟环境路径初始化 ----------
# BASE_DIR: 当前脚本所在目录，即项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# VENV_DIR: 项目内置的 .venv 虚拟环境目录
# 在嵌入式/打包部署场景下，需要确保虚拟环境的 site-packages 优先于系统包被搜索到
VENV_DIR = os.path.join(BASE_DIR, '.venv')
# 将虚拟环境路径插入 sys.path 最前面，保证优先加载虚拟环境中的依赖包
sys.path.insert(0, VENV_DIR)

# ---------- Web 框架与图像处理依赖 ----------
from flask import Flask, Response, g, jsonify, request, send_file, send_from_directory  # Flask: Web 框架; Response: 自定义响应(视频流); jsonify: JSON 响应; request: 请求对象; send_file: 文件下载; send_from_directory: 目录文件发送
import cv2              # OpenCV: 图像编解码、绘制文字、形态学操作等
import time             # 时间相关：计时、延时、时间戳格式化
import threading        # 多线程支持：后台检测循环、SD 同步循环、线程锁
import traceback        # 异常堆栈打印，用于调试后台线程中的错误
import numpy as np      # 数值计算库：生成占位图、图像数组操作
import requests         # HTTP 客户端：从 ESP32 获取图片/文件列表、触发远程拍照

# ---------- 检测模块导入 ----------
# 使用 as detection 起别名，这样代码中可以用简短的 detection 引用
# detection_enhanced 是增强版检测模块，包含改进后的叶片分割和病虫害检测算法
# 起别名后，如果有其他旧版检测模块(detection_original)，切换只需改这一行 import
import detection_enhanced as detection
# 从检测模块导入核心函数和配置类：
#   detect_and_annotate: 主检测函数，返回检测结果、标注图、各 mask
#   fetch_image: 从 ESP32-CAM 或本地测试图片获取图像
#   generate_mask_image: 根据当前参数生成各类 mask 预览图（叶片/病斑/虫害/标注）
#   classify_level: 根据检测指标将结果分为 正常/注意/警告/严重 四个等级
#   DetectionConfig: 检测参数配置类，封装所有可调参数
#   TEST_IMAGE_PATH: 测试图片路径（为空字符串表示使用真实 ESP32-CAM，非空表示离线测试模式）
from detection_enhanced import (
    detect_and_annotate, fetch_image, generate_mask_image, classify_level,
    DetectionConfig, TEST_IMAGE_PATH
)

# ---------- 多摄像头配置（从 cameras.json 加载） ----------
# cameras.json 放在项目根目录
CAMERAS_JSON = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "cameras.json"))

def load_cameras_config():
    """从 cameras.json 加载摄像头配置，返回摄像头字典 {cam_id: cam_info}"""
    if not os.path.exists(CAMERAS_JSON):
        print(f"[配置] cameras.json 不存在，使用默认单摄像头配置")
        return {
            "cam1": {
                "id": "cam1", "name": "默认摄像头",
                "url": "http://【请修改：摄像头IP地址】/capture",
                "base": "http://【请修改：摄像头IP地址】", "enabled": True,
            }
        }
    with open(CAMERAS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cameras_dict = {}
    for cam in data.get("cameras", []):
        cam_id = cam["id"]
        cameras_dict[cam_id] = {
            "id": cam_id,
            "name": cam.get("name", cam_id),
            "url": cam["url"],
            "base": cam.get("base", ""),
            "enabled": cam.get("enabled", True),
        }
    return cameras_dict

# cameras: 所有摄像头的配置和状态字典，key 为 camera_id
cameras = load_cameras_config()

# SYSTEM_A_URL: System A 诊断站地址，用于 A-B 浅连接（Phase 11）
SYSTEM_A_URL = "http://127.0.0.1:8000"
if os.path.exists(CAMERAS_JSON):
    with open(CAMERAS_JSON, 'r', encoding='utf-8') as _f:
        SYSTEM_A_URL = json.load(_f).get("system_a_url", SYSTEM_A_URL)

# 初始化每个摄像头的运行时状态
for _cid, _c in cameras.items():
    _c["state"] = {
        "latest_frame": None,
        "latest_result": {
            "level": "正常", "level_code": 0,
            "disease_count": 0, "white_count": 0,
            "disease_ratio": 0.0, "white_ratio": 0.0, "green_ratio": 0.0,
        },
        "latest_original_image": None,
        "last_error": "",
        "detection_history": [],
        "current_stable_level": "正常",
        "current_stable_level_code": 0,
        "level_change_counter": 0,
        "latest_dual_result": None,
        "comparison_history": [],
        "last_save_time": 0,
        "frame_lock": threading.Lock(),
    }

# 兼容旧代码: URL / ESP32_BASE 取第一个摄像头的值
_first_cam = next(iter(cameras.values()), {})
URL = _first_cam.get("url", "http://【请修改：摄像头IP地址】/capture")
ESP32_BASE = _first_cam.get("base", "http://【请修改：摄像头IP地址】")
detection.URL = URL

# YOLO 模型路径: 绝对路径，不依赖工作目录
_YOLO_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "best.pt"))

def _get_default_camera_id():
    """返回第一个已启用摄像头的 ID"""
    for cid, c in cameras.items():
        if c.get("enabled", True):
            return cid
    return next(iter(cameras), "cam1")

# ---------- 辅助模块导入 ----------
from history import HistoryManager      # 历史记录管理器：负责检测记录的存储、查询、统计、导出
from config import ConfigManager        # 配置管理器：负责检测参数的读取、保存、重置、参数元信息查询

# ---------- YOLO 双引擎模块导入 ----------
from yolo_detector import YOLODetector, get_detector   # YOLO 检测器封装
from dual_verifier import DualVerifier                  # 双引擎验证融合器
from alert_notifier import get_notifier                 # 智能告警通知器
from health import build_service_health, summarize_camera
from observability import resolve_request_id
from event_schema import build_detection_event
from offline_cache import OfflineEventCache
from event_transport import EventTransport, send_events_http

# 创建 Flask 应用实例
app = Flask(__name__)
logger = logging.getLogger("agrivision.system_b")


@app.before_request
def start_request_observation():
    g.request_id = resolve_request_id(request.headers.get("X-Request-ID"))
    g.request_started = time.perf_counter()


@app.after_request
def finish_request_observation(response):
    request_id = getattr(g, "request_id", "")
    response.headers["X-Request-ID"] = request_id
    elapsed_ms = (time.perf_counter() - getattr(g, "request_started", time.perf_counter())) * 1000
    logger.info(
        "request_completed event=request_completed method=%s path=%s status=%s duration_ms=%.1f request_id=%s",
        request.method,
        request.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response

# ============================================================
# 配置常量 - 系统运行时使用的全局常量与状态字典
# ============================================================

# SD 卡自动同步间隔（秒），300 秒 = 5 分钟
# 后台 sd_sync_loop 线程每隔此时间自动从 ESP32 SD 卡下载新图片到本地
SD_SYNC_INTERVAL = 300

# 本地数据集图片存储目录，从 ESP32 SD 卡同步下来的图片保存在此处
# 路径: 项目根目录/dataset/images/
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "images")
OFFLINE_EVENTS_DIR = os.path.join(BASE_DIR, "offline_events")
offline_event_cache = OfflineEventCache(OFFLINE_EVENTS_DIR)
EVENTS_SINK_URL = os.environ.get("AGRIVISION_EVENTS_SINK_URL", "").strip()


event_transport = None
offline_event_sync_lock = threading.Lock()
if EVENTS_SINK_URL:
    try:
        event_transport = EventTransport(EVENTS_SINK_URL, send_events_http)
    except ValueError as error:
        logger.warning("event_sink_config_invalid error=%s", error)

# 异常图片保存间隔（秒）：当检测到"注意"及以上等级时，每隔此时间保存一张标注图到 detection_logs/
SAVE_INTERVAL = 60

# SD 卡同步状态信息字典，供前端轮询显示同步进度
# 所有字段通过 sd_lock 保护，确保多线程安全
sd_sync_info = {
    "total_synced": 0,          # 本地已累计同步的图片总数
    "last_sync_time": "",       # 上次同步完成的时间字符串
    "sd_card_files": 0,         # ESP32 SD 卡上的文件总数
    "syncing": False,           # 是否正在同步（防重复触发）
    "sync_progress": "",        # 当前同步进度描述文本
    "sync_error": "",           # 同步错误信息（空字符串表示无错误）
    "new_downloaded": 0,        # 本次同步新下载的图片数量
}

# 远程拍照存卡状态信息字典，供前端轮询显示拍照进度
save_sd_info = {
    "saving": False,            # 是否正在执行拍照操作
    "done": False,              # 拍照是否已完成（成功或失败都算完成）
    "error": "",                # 拍照错误信息
    "file": "",                 # 拍照成功后 ESP32 返回的文件名
}

# ============================================================
# 全局变量 - 在检测线程和 Web 请求线程之间共享的运行时状态
# 注意: 这些变量会被多个线程同时访问，修改时必须持有对应的锁
# ============================================================

# 滑动窗口大小: 保留最近 5 帧的检测结果用于计算平均值
DETECTION_HISTORY_SIZE = 5

# 等级变化防抖阈值: 需要连续 LEVEL_CHANGE_THRESHOLD 帧的平均等级都不同于当前等级
# 才会触发等级切换，有效防止因偶尔一帧误检导致的等级跳变
LEVEL_CHANGE_THRESHOLD = 3

# 双引擎对比历史大小
COMPARISON_HISTORY_SIZE = 100

# sd_lock: 保护 sd_sync_info 字典的读写
#   写入方: SD 同步线程（sync_sd_card）、手动触发同步
#   读取方: Web 请求线程（/api/status, /api/sync_status）
sd_lock = threading.Lock()

# save_sd_lock: 保护 save_sd_info 字典的读写（远程拍照存卡）
save_sd_lock = threading.Lock()

# config_lock: 保护配置管理器（config_manager）的读写
#   写入方: Web 请求线程（用户通过参数调节页面保存参数）
#   读取方: 后台检测线程（每次检测前读取当前参数）
config_lock = threading.Lock()

# ---------- 管理器实例 ----------
# config_manager: 配置管理器单例，封装检测参数的持久化存储（JSON 文件）
#   支持: 获取当前配置、保存配置、重置为默认值、获取参数元信息（名称/范围/单位）
config_manager = ConfigManager()

# history_manager: 历史记录管理器单例，封装检测记录的持久化存储
#   支持: 添加记录、分页查询、按日期/等级过滤、趋势数据统计、导出为 ZIP
history_manager = HistoryManager()

# ---------- YOLO 双引擎初始化 ----------
# yolo_detector: YOLO 检测器，启动时自动加载模型
# 如果模型加载失败，detector.is_loaded() 返回 False，系统降级为单引擎运行
yolo_detector = get_detector(_YOLO_MODEL_PATH)

# dual_verifier: 双引擎验证融合器，负责将颜色引擎和 YOLO 引擎的结果融合为统一等级
dual_verifier = DualVerifier()
alert_notifier = get_notifier()

# yolo_enabled: YOLO 引擎总开关，可通过 API 动态启停
yolo_enabled = True


# ============================================================
# 占位图生成函数 - 当尚未获取到检测图像时，显示一张带提示文字的黑色图片
# 用于 MJPEG 视频流的首帧或错误状态，避免前端 <img> 标签因无数据而显示破损图标
# ============================================================
def create_wait_image(text="等待中..."):
    """
    生成一张 640x480 的黑色占位图，并在中央绘制提示文字。
    参数:
        text: 主提示文字，默认为"等待中..."，出错时可传入错误信息
    返回:
        JPEG 编码的字节流（bytes），可直接用于 MJPEG 视频流的帧数据
    """
    # 创建 480 行 x 640 列 x 3 通道（BGR）的全黑图像，数据类型为 8 位无符号整数
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # 在图像中央偏上位置绘制主提示文字（白色，字体大小 0.8，线宽 2）
    cv2.putText(img, text, (180, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # 在主文字下方绘制辅助提示（浅灰色，提醒用户查看控制台日志）
    cv2.putText(img, "请查看 PyCharm 控制台", (150, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    # 将 numpy 数组编码为 JPEG 格式的字节流
    _, buf = cv2.imencode('.jpg', img)
    # 返回字节流，供 MJPEG 视频流使用
    return buf.tobytes()


# ============================================================
# 核心检测函数 - 执行一次完整的病虫害检测流程
# ============================================================
# 该函数由后台检测线程（detection_loop）每隔约 3 秒调用一次。
# 整体流程:
#   1. 从 ESP32-CAM 或测试图片获取图像
#   2. 加载当前用户配置（支持热更新，用户调参后下一帧即生效）
#   3. 调用检测算法，获取病斑/虫害/叶片分割结果
#   4. 滑动窗口平均: 对最近 N 帧的数值指标取平均，平滑噪声
#   5. 等级防抖: 需要连续多帧确认等级变化才真正切换，避免频繁跳变
#   6. 更新全局状态（帧数据 + 结果字典），供 Web 前端读取
#   7. 条件保存: 异常等级时保存标注图到磁盘 + 写入历史记录
def run_detection_once(camera_id=None):
    # 获取摄像头配置和状态
    if camera_id is None:
        camera_id = _get_default_camera_id()
    cam = cameras.get(camera_id)
    if not cam:
        print(f"[检测] 摄像头 {camera_id} 不存在")
        return False
    
    state = cam["state"]
    frame_lock = state["frame_lock"]
    
    # 确保异常图片保存目录存在（首次运行时自动创建）
    save_dir = os.path.join(BASE_DIR, "detection_logs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        # ============================================================
        # 第一步: 获取图像
        # fetch_image() 会根据 TEST_IMAGE_PATH 决定数据来源:
        #   - TEST_IMAGE_PATH 非空: 从本地文件加载测试图片（离线开发/调试模式）
        #   - TEST_IMAGE_PATH 为空: 通过 HTTP GET 请求 ESP32-CAM 的 /capture 接口获取实时图片
        # 返回值: img (numpy BGR 数组或 None), source (来源描述字符串)
        # ============================================================
        print(f"[{time.strftime('%H:%M:%S')}] [{cam['name']}] 正在获取图片...")
        img, source = fetch_image(url=cam.get("url"))

        # 图像获取失败处理: 可能是 ESP32 断连、网络超时或测试图片路径无效
        if img is None:
            state["last_error"] = "获取图片失败"
            print(f"  -> {state['last_error']}")
            return False

        # ============================================================
        # 第二步: 加载当前检测参数
        # 使用 config_lock 确保与 Web 请求线程的参数写入互斥
        # 这样用户在参数调节页面修改参数后，下一次检测就能使用新参数（热更新）
        # ============================================================
        with config_lock:
            current_params = config_manager.get_current_config()

        # 将参数字典转换为 DetectionConfig 对象，供检测函数使用
        config = DetectionConfig()
        config.from_dict(current_params)

        # ============================================================
        # 第三步: 执行检测
        # detect_and_annotate() 是检测模块的核心函数，内部流程:
        #   a) 叶片分割: 基于 HSV 颜色空间提取绿色叶片区域
        #   b) 病斑检测: 在叶片区域内检测棕色/黄色异常区域
        #   c) 虫害检测: 检测白色/亮色斑点（白粉病等）
        #   d) 等级分类: 根据数量和面积占比判定严重程度
        #   e) 标注绘制: 在原图上绘制彩色框和文字
        # 返回值:
        #   result: 检测结果字典 (disease_count, white_count, 各 ratio, level_code 等)
        #   annotated: 标注后的图像（numpy 数组，已绘制检测框和文字）
        #   masks: 各分割 mask 字典（叶片/病斑/虫害的单独 mask 图）
        # ============================================================
        result, annotated, masks = detect_and_annotate(img, config)

        # ============================================================
        # 第三步B: YOLO 引擎检测（新增）
        # ============================================================
        yolo_detections = []
        if yolo_enabled and yolo_detector.is_loaded():
            yolo_detections = yolo_detector.detect(img)
            # 在标注图上叠加 YOLO 检测框（虚线框，与颜色引擎的实线框区分）
            annotated = yolo_detector.annotate(annotated, yolo_detections)

        # ============================================================
        # 第三步C: 双引擎验证融合（新增）
        # ============================================================
        dual_result = dual_verifier.verify(result, yolo_detections)

        # 记录双引擎对比数据（局部变量，稍后在锁内写入 state）
        comparison_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'color_disease': result.get('disease_count', 0),
            'color_bug': result.get('white_count', 0),
            'yolo_disease': dual_result['yolo_result']['disease_count'],
            'yolo_bug': dual_result['yolo_result']['bug_count'],
            'yolo_disease_conf': dual_result['yolo_result']['disease_max_conf'],
            'yolo_bug_conf': dual_result['yolo_result']['bug_max_conf'],
            'dual_level': dual_result['level'],
            'confidence': dual_result['confidence'],
            'inference_ms': round(yolo_detector._last_inference_ms, 1),
        }

        # ============================================================
        # 第四步: 滑动窗口平均（第一层平滑）
        # ============================================================
        # 在局部变量上计算，不直接修改 state（稍后在锁内统一写入）
        local_history = list(state["detection_history"])  # 快照
        local_history.append({
            "disease_count": result["disease_count"],
            "white_count": result["white_count"],
            "disease_ratio": result["disease_ratio"],
            "white_ratio": result["white_ratio"],
            "green_ratio": result["green_ratio"],
            "level_code": result["level_code"],
        })
        if len(local_history) > DETECTION_HISTORY_SIZE:
            local_history = local_history[-DETECTION_HISTORY_SIZE:]

        # 计算滑动窗口平均值
        # 为什么至少需要 3 帧才计算平均？
        #   - 1-2 帧时数据太少，平均值没有统计意义，不如直接用当前帧
        #   - 3 帧起开始平均，可以在保证响应速度的同时有效平滑单帧异常
        #   - 系统启动后约 9 秒（3帧 x 3秒/帧）即可进入稳定状态
        if len(local_history) >= 3:
            avg_disease_count = sum(h["disease_count"] for h in local_history) / len(local_history)
            avg_white_count = sum(h["white_count"] for h in local_history) / len(local_history)
            avg_disease_ratio = sum(h["disease_ratio"] for h in local_history) / len(local_history)
            avg_white_ratio = sum(h["white_ratio"] for h in local_history) / len(local_history)
            avg_green_ratio = sum(h["green_ratio"] for h in local_history) / len(local_history)
        else:
            avg_disease_count = result["disease_count"]
            avg_white_count = result["white_count"]
            avg_disease_ratio = result["disease_ratio"]
            avg_white_ratio = result["white_ratio"]
            avg_green_ratio = result["green_ratio"]

        # 使用平均后的指标重新计算等级
        avg_level_code, avg_level_name = classify_level(
            round(avg_disease_count), avg_disease_ratio,
            round(avg_white_count), avg_white_ratio,
            avg_green_ratio, config
        )

        # ============================================================
        # 第五步: 等级变化防抖（第二层平滑）
        # ============================================================
        # 在局部变量上计算防抖，稍后在锁内写入 state
        local_stable_code = state["current_stable_level_code"]
        local_stable_level = state["current_stable_level"]
        local_counter = state["level_change_counter"]

        if avg_level_code != local_stable_code:
            local_counter += 1
            if local_counter >= LEVEL_CHANGE_THRESHOLD:
                local_stable_code = avg_level_code
                local_stable_level = avg_level_name
                local_counter = 0
                print(f"  [等级变化] {local_stable_level}")
        else:
            local_counter = 0

        # 构建最终的稳定结果字典
        stable_result = {
            "level": local_stable_level,
            "level_code": local_stable_code,
            "disease_count": round(avg_disease_count),
            "white_count": round(avg_white_count),
            "disease_ratio": avg_disease_ratio,
            "white_ratio": avg_white_ratio,
            "green_ratio": avg_green_ratio,
        }

        # 将检测摘要写入有界离线队列，供网络恢复后的传输器消费。
        # 队列只保存事件摘要，不保存图像、URL 或通知配置。
        try:
            offline_event_cache.put(build_detection_event(camera_id, stable_result))
        except (TypeError, ValueError, OSError) as cache_error:
            logger.warning("offline_event_enqueue_failed event=offline_event_enqueue_failed error=%s", cache_error)

        # 将标注后的图像编码为 JPEG 字节流
        _, buf = cv2.imencode('.jpg', annotated)

        # ============================================================
        # 加锁: 一次性更新所有共享状态（检测线程 -> Web 请求线程）
        # ============================================================
        # 所有对 state 字典的可变操作都在此锁内完成，
        # 保证 Web 请求线程（api_status/api_comparison 等）看到一致的状态
        with frame_lock:
            state["latest_frame"] = buf.tobytes()
            state["latest_result"] = stable_result
            state["latest_original_image"] = img.copy()
            state["latest_dual_result"] = dual_result
            state["comparison_history"].append(comparison_entry)
            if len(state["comparison_history"]) > COMPARISON_HISTORY_SIZE:
                state["comparison_history"].pop(0)
            state["detection_history"] = local_history
            state["current_stable_level_code"] = local_stable_code
            state["current_stable_level"] = local_stable_level
            state["level_change_counter"] = local_counter

        # ============================================================
        # 第六步: 条件性保存和告警（使用局部变量，无需持锁）
        # ============================================================
        now_time = time.time()
        # 读取 last_save_time 需在锁外安全访问（仅检测线程写，此处用局部快照）
        with frame_lock:
            last_save = state["last_save_time"]

        if local_stable_code >= 1 and now_time - last_save >= SAVE_INTERVAL:
            filename = f"{save_dir}/{time.strftime('%Y%m%d_%H%M%S')}_{local_stable_level}.jpg"
            cv2.imwrite(filename, annotated)
            with frame_lock:
                state["last_save_time"] = now_time
            print(f"  -> {local_stable_level} | 病斑{round(avg_disease_count)} 虫害{round(avg_white_count)} 绿色{avg_green_ratio:.1%} | 已保存 {filename}")
        else:
            print(f"  -> {local_stable_level} | 病斑{round(avg_disease_count)} 虫害{round(avg_white_count)} 绿色{avg_green_ratio:.1%}")

        if local_stable_code >= 1:
            history_manager.add_record(stable_result, annotated, dual_result=dual_result)

        if local_stable_code >= 2:
            alert_result = alert_notifier.notify(
                level=local_stable_level,
                level_code=local_stable_code,
                disease_count=stable_result.get('disease_count', 0),
                pest_count=stable_result.get('white_count', 0),
                disease_ratio=stable_result.get('disease_ratio', 0.0),
                pest_ratio=stable_result.get('white_ratio', 0.0),
                confidence=dual_result.get('confidence', ''),
            )
            if alert_result['sent']:
                print(f"  -> [告警] 已推送钉钉通知: {local_stable_level}")

        state["last_error"] = ""
        return True
    except Exception as e:
        # 异常处理: 捕获所有未预期的错误，避免后台线程崩溃
        state["last_error"] = f"检测出错: {str(e)}"
        print(f"  -> {state['last_error']}")
        traceback.print_exc()    # 打印完整堆栈信息，便于调试
        return False


# ============================================================
# SD 卡同步函数 - 从 ESP32-CAM 的 SD 卡下载新图片到本地
# ============================================================
# 该函数可被以下两种方式触发:
#   1. 后台 sd_sync_loop 线程每隔 SD_SYNC_INTERVAL (300秒) 自动调用
#   2. 用户在前端点击"立即同步"按钮，通过 /api/sync_now 接口手动触发
# 同步流程:
#   1. 请求 ESP32 的 /list 接口获取 SD 卡上的文件列表
#   2. 与本地已有文件做去重比对，找出新文件
#   3. 逐个下载新文件到本地 dataset/images/ 目录
#   4. 更新同步状态信息（供前端轮询显示进度）
def sync_sd_card(camera_id=None):
    global sd_sync_info
    
    # 获取摄像头配置
    if camera_id is None:
        camera_id = _get_default_camera_id()
    cam = cameras.get(camera_id)
    if not cam:
        print(f"[SD同步] 摄像头 {camera_id} 不存在")
        return
    
    esp32_base = cam["base"]

    # 防重复执行: 如果已有同步任务在运行（syncing=True），直接跳过
    # 这在后台自动同步和手动同步同时触发时尤为重要
    with sd_lock:
        syncing = sd_sync_info["syncing"]
    if syncing:
        print("[SD同步] 已有同步任务在运行，跳过")
        return

    # 初始化同步状态: 标记为同步中，清空错误信息，设置进度提示
    with sd_lock:
        sd_sync_info["syncing"] = True
        sd_sync_info["sync_error"] = ""
        sd_sync_info["sync_progress"] = "正在连接 ESP32..."
        sd_sync_info["new_downloaded"] = 0

    # 确保本地数据集目录存在（首次同步时自动创建）
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    try:
        # --- 第一步: 获取 ESP32 SD 卡上的文件列表 ---
        # ESP32 的 /list 接口返回 JSON 数组，包含 SD 卡上所有图片文件名
        with sd_lock:
            sd_sync_info["sync_progress"] = "正在获取 SD 卡文件列表..."
        resp = requests.get(f"{esp32_base}/list", timeout=30)
        if resp.status_code != 200:
            err_msg = f"获取列表失败, 状态码: {resp.status_code}"
            with sd_lock:
                sd_sync_info["sync_error"] = err_msg
            print(f"[SD同步] {err_msg}")
            return
        sd_files = resp.json()
        with sd_lock:
            sd_sync_info["sd_card_files"] = len(sd_files)

        # --- 第二步: 去重比对 ---
        # 将本地已有文件名转为 set（哈希集合），实现 O(1) 查找
        # 筛选出 SD 卡上有但本地没有的文件（差集），避免重复下载
        local_files = set(os.listdir(DATASET_DIR))
        new_files = [f for f in sd_files if f not in local_files]
        new_count = 0

        # 如果没有新文件，直接提示已是最新
        if not new_files:
            with sd_lock:
                sd_sync_info["sync_progress"] = f"已是最新，SD卡 {len(sd_files)} 张"
        else:
            with sd_lock:
                sd_sync_info["sync_progress"] = f"发现 {len(new_files)} 张新图片，开始下载..."

        # --- 第三步: 逐个下载新文件 ---
        # 遍历每个新文件，通过 ESP32 的 /image?file=xxx 接口下载
        for i, filename in enumerate(new_files):
            try:
                # 更新进度文本，格式: "下载中 (当前序号/总数): 文件名"
                with sd_lock:
                    sd_sync_info["sync_progress"] = f"下载中 ({i+1}/{len(new_files)}): {filename}"
                # 从 ESP32 下载指定文件，超时 60 秒（SD 卡读取可能较慢）
                img_resp = requests.get(
                    f"{esp32_base}/image",
                    params={"file": filename},
                    timeout=60
                )
                if img_resp.status_code == 200 and len(img_resp.content) > 0:
                    # 下载成功，写入本地文件
                    save_path = os.path.join(DATASET_DIR, filename)
                    with open(save_path, "wb") as f:
                        f.write(img_resp.content)
                    new_count += 1
                    with sd_lock:
                        sd_sync_info["new_downloaded"] = new_count
                    print(f"  [SD同步] 下载: {filename} ({len(img_resp.content)} 字节)")
                    # 每个文件下载间隔 1 秒，避免 ESP32 过载（嵌入式设备处理能力有限）
                    time.sleep(1)
                else:
                    print(f"  [SD同步] 下载 {filename} 失败，状态码: {img_resp.status_code}")
            except Exception as e:
                # 单个文件下载失败不中断整个同步流程，记录错误后继续下一个
                print(f"  [SD同步] 下载 {filename} 失败: {e}")

        # --- 第四步: 更新最终同步状态 ---
        with sd_lock:
            sd_sync_info["total_synced"] = len(os.listdir(DATASET_DIR))
            sd_sync_info["last_sync_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            sd_sync_info["sync_progress"] = f"完成! 新下载 {new_count} 张, 本地共 {sd_sync_info['total_synced']} 张"
            progress_msg = sd_sync_info['sync_progress']
        print(f"[SD同步] {progress_msg}")

    except Exception as e:
        # 整体异常处理（如 ESP32 连接失败等网络错误）
        with sd_lock:
            sd_sync_info["sync_error"] = f"同步失败: {e}"
            sd_sync_info["sync_progress"] = f"错误: {e}"
            err_msg = sd_sync_info['sync_error']
        print(f"[SD同步] {err_msg}")
    finally:
        # 无论成功或失败，都必须在 finally 中重置 syncing 标志
        # 否则后续同步任务将永远被跳过
        with sd_lock:
            sd_sync_info["syncing"] = False


# ============================================================
# 远程拍照存卡函数 - 触发 ESP32-CAM 拍摄一张照片并保存到 SD 卡
# ============================================================
# 该函数由 /api/save_to_sd 路由在独立守护线程中调用（异步执行）。
# 工作流程: 向 ESP32 发送 GET /save 请求 -> ESP32 拍照并写入 SD 卡 -> 返回文件名
# 使用异步方式是因为 ESP32 拍照+写卡可能耗时数秒，不能阻塞 Web 请求线程。
# 前端通过轮询 /api/save_status 接口获取执行进度。
def save_to_sd_async(camera_id=None):
    global save_sd_info
    
    # 获取摄像头配置
    if camera_id is None:
        camera_id = _get_default_camera_id()
    cam = cameras.get(camera_id)
    if not cam:
        with save_sd_lock:
            save_sd_info["error"] = f"摄像头 {camera_id} 不存在"
            save_sd_info["done"] = True
            save_sd_info["saving"] = False
        return
    
    esp32_base = cam["base"]
    
    # 初始化状态: 标记为执行中，清空之前的结果
    with save_sd_lock:
        save_sd_info["saving"] = True
        save_sd_info["done"] = False
        save_sd_info["error"] = ""
        save_sd_info["file"] = ""
    try:
        # 向 ESP32 发送拍照存卡指令，超时 60 秒（等待 SD 卡写入完成）
        resp = requests.get(f"{esp32_base}/save", timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            with save_sd_lock:
                save_sd_info["file"] = data.get("file", "")
        else:
            with save_sd_lock:
                save_sd_info["error"] = f"ESP32 返回状态码: {resp.status_code}"
    except Exception as e:
        with save_sd_lock:
            save_sd_info["error"] = str(e)
    finally:
        with save_sd_lock:
            save_sd_info["saving"] = False
            save_sd_info["done"] = True


# ============================================================
# 后台线程循环 - 系统启动后以守护线程方式持续运行
# ============================================================
# 守护线程 (daemon=True): 当主线程（Flask 服务器）退出时，这些线程会自动终止
# 系统共有两个后台线程:
#   1. detection_loop: 负责周期性执行病虫害检测
#   2. sd_sync_loop:  负责周期性从 ESP32 SD 卡同步图片

# SD 卡同步循环: 每隔 SD_SYNC_INTERVAL (300秒 = 5分钟) 自动同步一次
# 在系统启动时首次执行会在主线程完成，此循环从第二次开始
def sd_sync_loop(camera_id=None):
    print(f"[SD同步线程] 已启动 (摄像头: {camera_id})")
    while True:
        sync_sd_card(camera_id)
        time.sleep(SD_SYNC_INTERVAL)   # 阻塞等待下一个同步周期


# 检测循环: 每隔 3 秒执行一次检测
# 3 秒间隔的选取考量:
#   - ESP32-CAM 拍摄+传输一张 SVGA (800x600) 图片约需 0.5-1 秒
#   - 检测算法处理一帧约需 0.3-0.5 秒
#   - 剩余时间作为余量，确保不会因网络波动导致帧积压
def detection_loop(camera_id=None):
    print(f"[后台线程] 检测循环已启动 (摄像头: {camera_id})")
    while True:
        run_detection_once(camera_id)
        time.sleep(3)  # 每次检测后等待 3 秒再开始下一帧


# ============================================================
# 网页 HTML - 单页面应用（SPA）前端代码
# ============================================================
# 整个 Web 界面以一个 Python 多行字符串的形式嵌入，由 Flask 的 / 路由直接返回。
# 这种设计避免了额外的静态文件管理，适合嵌入式部署场景。
#
# === 页面结构（三个 Tab 页签） ===
# 1. 实时监控 (tab-monitor):
#    - 左侧: MJPEG 视频流（<img src="/video_feed">，浏览器原生支持 MJPEG）
#    - 右侧: 实时数据面板（病斑数/虫害数/各占比/更新次数），每 2 秒轮询 /api/status
#    - 下方: SD 卡图片相册（gallery），支持拍照存卡、立即同步操作
#
# 2. 参数调节 (tab-params):
#    - 上方: 四个 mask 预览窗口（叶片分割/病斑/虫害/标注图）
#    - 下方: 参数滑块组，分为 叶片分割/病斑检测/虫害检测/分级阈值 四个分组
#    - 操作按钮: 应用参数、重置默认、刷新预览
#    - 调参时自动触发防抖刷新（200ms 延迟），实时查看参数对分割效果的影响
#
# 3. 历史记录 (tab-history):
#    - 统计卡片: 总检测次数、各等级分布
#    - 趋势图表: 近 7 天病斑/虫害数量折线图（Chart.js 渲染）
#    - 记录表格: 支持按日期范围和等级过滤、分页浏览、查看标注图
#    - 数据导出: 导出指定日期范围的记录为 ZIP 文件
#
# === JavaScript 核心函数说明 ===
# - updateStatus(): 轮询 /api/status 获取检测结果和 SD 同步状态，更新页面显示
# - switchTab(tabId): Tab 页签切换，首次切换到 params/history 时自动加载数据
# - loadParams() / applyParams() / resetParams(): 参数的加载、保存、重置
# - refreshMasks(): 向 /api/preview_mask 发送当前参数，获取并显示四种 mask 预览图
# - updateParamValue(): 滑块拖动时的实时值更新 + 防抖触发预览刷新
# - loadHistory(page): 分页加载历史记录表格
# - loadTrend(): 加载并渲染近 7 天趋势折线图
# - loadStatistics(): 加载统计数据填充顶部统计卡片
# - exportData(): 触发历史记录导出（下载 ZIP 文件）
# - loadGallery(): 加载 SD 卡图片列表，渲染网格相册
# - saveToSD() / syncNow(): 触发远程拍照和 SD 卡同步
# - startSavePolling() / startSyncPolling(): 异步操作的进度轮询（每 2 秒）
# - openModal(src) / closeModal(): 图片大图查看（模态框）
# - showToast(msg, type): 轻量级通知提示（成功/错误/信息）
#
# === 轮询定时器 ===
# - setInterval(updateStatus, 2000): 每 2 秒刷新检测状态
# - setInterval(loadGallery, 60000): 每 60 秒刷新 SD 卡相册
# - syncPollTimer / savePollTimer: 同步/拍照操作的临时进度轮询（操作完成后自动清除）
#
# === 外部依赖 ===
# - Chart.js (CDN): 用于绘制趋势折线图
# - 无其他前端框架，纯原生 JavaScript + CSS
# ============================================================
# ========== 网页 HTML ==========
HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>大棚病虫害实时监控系统</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: "Microsoft YaHei", sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
.header { text-align: center; margin-bottom: 20px; }
.header h1 { font-size: 24px; margin-bottom: 10px; }
.status-box { display: inline-block; padding: 8px 24px; border-radius: 20px; font-size: 18px; font-weight: bold; }
.status-normal { background: #2e7d32; }
.status-notice { background: #ef6c00; }
.status-warning { background: #c62828; }
.status-serious { background: #7b0000; }

.camera-selector { display:inline-flex; align-items:center; gap:8px; margin-left:20px; }
.camera-selector select { background:#1e3a5f; color:#fff; border:1px solid #4fc3f7; padding:4px 8px; border-radius:4px; font-size:13px; }
.dashboard-link { display:inline-block; margin-left:16px; padding:6px 14px; background:#16213e; color:#4fc3f7; border:1px solid #4fc3f7; border-radius:6px; text-decoration:none; font-size:13px; transition:all 0.3s; }
.dashboard-link:hover { background:#0f3460; color:#fff; }
.btn-diagnose { display:inline-block; margin-left:12px; padding:6px 16px; background:linear-gradient(135deg,#6a11cb,#2575fc); color:#fff; border:none; border-radius:6px; font-size:13px; cursor:pointer; transition:all 0.3s; }
.btn-diagnose:hover { opacity:0.85; transform:translateY(-1px); }
.btn-diagnose:disabled { opacity:0.5; cursor:not-allowed; transform:none; }
.report-modal { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:9999; justify-content:center; align-items:center; }
.report-modal.show { display:flex; }
.report-modal-content { background:#16213e; border-radius:16px; padding:30px; max-width:700px; width:90%; max-height:85vh; overflow-y:auto; position:relative; border:1px solid #4fc3f7; }
.report-modal-content h2 { color:#4fc3f7; margin-bottom:16px; font-size:20px; }
.report-modal-content .report-text { color:#ddd; line-height:1.8; font-size:14px; white-space:pre-wrap; }
.report-close { position:absolute; top:12px; right:16px; background:none; border:none; color:#aaa; font-size:24px; cursor:pointer; }
.report-close:hover { color:#fff; }
.report-loading { text-align:center; padding:40px 0; color:#4fc3f7; font-size:16px; }
.report-loading .spinner { display:inline-block; width:30px; height:30px; border:3px solid #333; border-top-color:#4fc3f7; border-radius:50%; animation:spin 0.8s linear infinite; margin-bottom:12px; }
@keyframes spin { to { transform:rotate(360deg); } }

.tab-container { max-width: 1400px; margin: 0 auto; }
.tab-buttons { display: flex; gap: 10px; margin-bottom: 20px; }
.tab-btn { padding: 10px 24px; border: none; border-radius: 8px; cursor: pointer; font-size: 15px; color: #fff; background: #16213e; transition: all 0.3s; }
.tab-btn.active { background: #0f3460; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
.tab-btn:hover { background: #0f3460; }
.tab-content { display: none; background: #16213e; border-radius: 12px; padding: 20px; }
.tab-content.active { display: block; }

.container { display: flex; gap: 20px; flex-wrap: wrap; }
.left, .right { background: #0f3460; border-radius: 12px; padding: 20px; }
.left { flex: 1.5; min-width: 320px; }
.right { flex: 1; min-width: 280px; }
.video-box { width: 100%; border-radius: 8px; overflow: hidden; background: #000; }
.video-box img { width: 100%; display: block; }
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px; }
.info-card { background: #1a1a2e; padding: 15px; border-radius: 8px; text-align: center; }
.info-card .label { font-size: 14px; color: #aaa; margin-bottom: 5px; }
.info-card .value { font-size: 22px; font-weight: bold; }
.legend { margin-top: 15px; font-size: 14px; color: #aaa; }
.legend span { margin-right: 15px; }
.red { color: #ff4444; } .blue { color: #4488ff; } .green { color: #44ff44; }

.mask-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px; }
.mask-item { background: #0f3460; border-radius: 8px; padding: 12px; text-align: center; }
.mask-item img { width: 100%; border-radius: 6px; background: #000; }
.mask-item .caption { font-size: 13px; color: #aaa; margin-top: 8px; }

.params-section { margin-top: 20px; }
.params-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
.param-group { background: #0f3460; padding: 15px; border-radius: 8px; }
.param-group h4 { font-size: 14px; color: #88ccff; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 1px solid #222; }
.param-row { margin-bottom: 12px; }
.param-row label { display: block; font-size: 13px; color: #aaa; margin-bottom: 5px; }
.param-row input[type="range"] { width: 100%; height: 6px; border-radius: 3px; background: #333; outline: none; }
.param-row input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #4fc3f7; cursor: pointer; }
.param-row input[type="range"]::-moz-range-thumb { width: 16px; height: 16px; border-radius: 50%; background: #4fc3f7; cursor: pointer; border: none; }
.param-row .value { font-size: 13px; color: #4fc3f7; font-family: monospace; }
.param-actions { display: flex; gap: 10px; margin-top: 20px; }
.btn { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; color: #fff; }
.btn-green { background: #2e7d32; }
.btn-blue { background: #1565c0; }
.btn-gray { background: #546e7a; }
.btn:hover { opacity: 0.85; }

.history-section { margin-top: 20px; }
.filter-bar { display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 15px; align-items: center; }
.filter-bar select, .filter-bar input, .filter-bar button { padding: 8px 12px; border-radius: 6px; border: 1px solid #333; background: #0f3460; color: #eee; font-size: 13px; }
.filter-bar button { cursor: pointer; background: #1565c0; border: none; }
.table-wrapper { overflow-x: auto; }
.history-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.history-table th, .history-table td { padding: 10px; text-align: left; border-bottom: 1px solid #222; }
.history-table th { background: #0f3460; color: #aaa; }
.history-table tr:hover { background: #0f3460; }
.level-badge { padding: 4px 10px; border-radius: 10px; font-size: 12px; font-weight: bold; }
.level-normal { background: #2e7d32; }
.level-notice { background: #ef6c00; }
.level-warning { background: #c62828; }
.level-serious { background: #7b0000; }
.page-bar { display: flex; justify-content: center; align-items: center; gap: 10px; margin-top: 15px; }
.page-bar button { padding: 6px 12px; border-radius: 4px; border: none; background: #0f3460; color: #eee; cursor: pointer; }
.page-bar button.active { background: #1565c0; }
.page-bar button:disabled { opacity: 0.5; cursor: not-allowed; }

.chart-container { background: #0f3460; border-radius: 8px; padding: 20px; margin-top: 20px; }
.chart-container h4 { font-size: 15px; margin-bottom: 15px; color: #88ccff; }
canvas { width: 100% !important; }

.gallery-section { margin-top: 20px; }
.gallery-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; gap: 10px; }
.gallery-header h3 { font-size: 18px; }
.gallery-actions { display: flex; gap: 10px; }
.sd-info { font-size: 13px; color: #888; margin-bottom: 10px; }
.sync-progress { font-size: 13px; color: #4fc3f7; margin-bottom: 8px; min-height: 18px; }
.sync-progress.error { color: #ff5252; }
.sync-progress.done { color: #69f0ae; }

.gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.gallery-item { border-radius: 8px; overflow: hidden; background: #0f3460; cursor: pointer; transition: transform 0.2s; }
.gallery-item:hover { transform: scale(1.03); }
.gallery-item img { width: 100%; height: 130px; object-fit: cover; display: block; }
.gallery-item .name { padding: 6px 8px; font-size: 12px; color: #aaa; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.gallery-empty { text-align: center; color: #666; padding: 40px; font-size: 16px; }

.toast { position: fixed; top: 20px; right: 20px; padding: 12px 24px; border-radius: 8px; color: #fff; font-size: 14px; z-index: 2000; opacity: 0; transition: opacity 0.3s; pointer-events: none; }
.toast.show { opacity: 1; }
.toast-success { background: #2e7d32; }
.toast-error { background: #c62828; }
.toast-info { background: #1565c0; }

.modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); z-index: 1000; justify-content: center; align-items: center; }
.modal.active { display: flex; }
.modal img { max-width: 90%; max-height: 90%; border-radius: 8px; }
.modal-close { position: absolute; top: 20px; right: 30px; font-size: 30px; color: #fff; cursor: pointer; }

.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
.stat-card { background: #0f3460; padding: 15px; border-radius: 8px; text-align: center; }
.stat-card .label { font-size: 13px; color: #aaa; }
.stat-card .value { font-size: 24px; font-weight: bold; }

.engine-section { margin-top: 20px; padding-top: 15px; border-top: 1px solid #333; }
.engine-section h4 { font-size: 14px; color: #88ccff; margin-bottom: 12px; }
.engine-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 13px; }
.engine-row .eng-label { color: #aaa; }
.engine-row .eng-value { font-family: monospace; }
.conf-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }
.conf-high { background: #2e7d32; }
.conf-medium { background: #ef6c00; }
.conf-low { background: #546e7a; }
.agree-tag { display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 11px; }
.agree-both { background: #1b5e20; color: #a5d6a7; }
.agree-yolo { background: #e65100; color: #ffcc80; }
.agree-color { background: #4a148c; color: #ce93d8; }
.agree-none { background: #333; color: #888; }
.yolo-toggle { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.yolo-toggle label { font-size: 13px; color: #aaa; cursor: pointer; }
.yolo-toggle input[type="checkbox"] { width: 16px; height: 16px; cursor: pointer; accent-color: #4fc3f7; }
.engine-status { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
.engine-on { background: #4caf50; }
.engine-off { background: #f44336; }
.legend-yolo { border-bottom: 2px dashed #ff9800; display: inline-block; width: 16px; height: 0; vertical-align: middle; margin-right: 3px; }
.alert-section { margin-top: 12px; padding-top: 10px; border-top: 1px solid #333; }
.alert-section h5 { font-size: 13px; color: #ff9800; margin-bottom: 10px; cursor: pointer; user-select: none; }
.alert-row { margin-bottom: 10px; }
.alert-row label { display: block; font-size: 12px; color: #aaa; margin-bottom: 4px; }
.alert-input { width: 100%; padding: 6px 8px; background: #1a1a2e; border: 1px solid #333; border-radius: 4px; color: #e0e0e0; font-size: 12px; box-sizing: border-box; }
.alert-input:focus { border-color: #ff9800; outline: none; }
.alert-btn { padding: 5px 14px; border: none; border-radius: 4px; font-size: 12px; cursor: pointer; margin-right: 6px; margin-top: 4px; }
.alert-btn-test { background: #455a64; color: #e0e0e0; }
.alert-btn-test:hover { background: #546e7a; }
.alert-btn-save { background: #ef6c00; color: #fff; }
.alert-btn-save:hover { background: #f57c00; }
.alert-log { max-height: 120px; overflow-y: auto; font-size: 11px; color: #888; margin-top: 8px; padding: 6px; background: #111; border-radius: 4px; }
.alert-log-item { padding: 2px 0; border-bottom: 1px solid #222; }
</style>
</head>
<body>
<div class="header">
  <h1>大棚病虫害实时监控系统</h1>
  <div id="statusBadge" class="status-box status-normal">等待中</div>
  <div class="camera-selector">
    <label>摄像头:</label>
    <select id="cameraSelect" onchange="switchCamera()">
    </select>
  </div>
  <a href="/dashboard" class="dashboard-link" title="监控大屏">&#9638; 大屏</a>
  <button class="btn btn-diagnose" id="diagnoseBtn" onclick="deepDiagnose()" title="AI深度诊断">&#129504; AI诊断</button>
</div>

<div class="tab-container">
  <div class="tab-buttons">
    <button class="tab-btn active" data-tab="monitor" onclick="switchTab('monitor')">实时监控</button>
    <button class="tab-btn" data-tab="params" onclick="switchTab('params')">参数调节</button>
    <button class="tab-btn" data-tab="history" onclick="switchTab('history')">历史记录</button>
    <button class="tab-btn" data-tab="analytics" onclick="switchTab('analytics')">模型分析</button>
  </div>

  <!-- 实时监控 Tab -->
  <div id="tab-monitor" class="tab-content active">
    <div class="container">
      <div class="left">
        <div class="video-box"><img id="videoFeed" src="" alt="实时监控画面"></div>
        <div class="legend">
          <span class="red">■ 红框=病斑</span>
          <span class="blue">■ 蓝框=虫害</span>
          <span class="green">■ 绿线=叶片轮廓</span>
          <span><span class="legend-yolo"></span><span style="color:#ff9800">YOLO框</span></span>
        </div>
      </div>
      <div class="right">
        <h3 style="margin-bottom:15px">实时数据</h3>
        <div class="info-grid">
          <div class="info-card"><div class="label">病斑数量</div><div class="value" id="diseaseCount">0</div></div>
          <div class="info-card"><div class="label">虫害数量</div><div class="value" id="pestCount">0</div></div>
          <div class="info-card"><div class="label">病斑占比</div><div class="value" id="diseaseRatio">0.0%</div></div>
          <div class="info-card"><div class="label">虫害占比</div><div class="value" id="pestRatio">0.0%</div></div>
          <div class="info-card"><div class="label">叶片绿色占比</div><div class="value" id="greenRatio">0.0%</div></div>
          <div class="info-card"><div class="label">更新次数</div><div class="value" id="updateCount">0</div></div>
        </div>
        <div class="engine-section">
          <h4>双引擎状态</h4>
          <div class="yolo-toggle">
            <input type="checkbox" id="yoloEnable" checked onchange="toggleYolo(this.checked)">
            <label for="yoloEnable">YOLO 引擎 <span id="yoloStatusDot" class="engine-status engine-off"></span><span id="yoloStatusText">加载中</span></label>
          </div>
          <div class="engine-row"><span class="eng-label">融合等级</span><span class="eng-value" id="dualLevel">-</span></div>
          <div class="engine-row"><span class="eng-label">置信度</span><span class="eng-value" id="dualConf">-</span></div>
          <div class="engine-row"><span class="eng-label">病斑一致性</span><span class="eng-value" id="agreeDisease">-</span></div>
          <div class="engine-row"><span class="eng-label">虫害一致性</span><span class="eng-value" id="agreeBug">-</span></div>
          <div class="engine-row"><span class="eng-label">YOLO 病斑</span><span class="eng-value" id="yoloDisease">0</span></div>
          <div class="engine-row"><span class="eng-label">YOLO 虫害</span><span class="eng-value" id="yoloBug">0</span></div>
          <div class="engine-row"><span class="eng-label">推理耗时</span><span class="eng-value" id="yoloTime">-</span></div>
          <div style="margin-top:10px;padding-top:8px;border-top:1px solid #333">
            <div class="param-row">
              <label>YOLO 置信度阈值</label>
              <input type="range" id="yoloConfSlider" min="0.1" max="0.9" step="0.05" value="0.25" oninput="updateYoloConf(this.value)">
              <div class="value" id="yoloConfVal">0.25</div>
            </div>
          </div>
          <div class="alert-section">
            <h5 onclick="toggleAlertPanel()">&#9889; 智能告警 <span id="alertToggleHint" style="font-size:11px;color:#666">&#9660;</span></h5>
            <div id="alertPanel" style="display:none">
              <div class="alert-row">
                <label><input type="checkbox" id="alertEnable" style="margin-right:6px;accent-color:#ff9800" onchange="saveAlertConfig()"> 启用钉钉告警</label>
              </div>
              <div class="alert-row">
                <label>Webhook URL</label>
                <input type="text" class="alert-input" id="alertWebhook" placeholder="https://oapi.dingtalk.com/robot/send?access_token=..." onblur="saveAlertConfig()">
              </div>
              <div class="alert-row">
                <label>冷却时间 (秒)</label>
                <input type="range" id="alertCooldown" min="60" max="600" step="30" value="300" style="width:100%;accent-color:#ff9800" oninput="document.getElementById('cooldownVal').textContent=this.value+'s'" onblur="saveAlertConfig()">
                <div style="font-size:12px;color:#ff9800" id="cooldownVal">300s</div>
              </div>
              <div>
                <button class="alert-btn alert-btn-test" onclick="testAlert()">发送测试</button>
                <button class="alert-btn alert-btn-save" onclick="saveAlertConfig()">保存配置</button>
              </div>
              <div class="alert-log" id="alertLog">暂无告警记录</div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="chart-container">
      <h4>双引擎对比 (近50帧)</h4>
      <div style="display:flex;gap:20px;flex-wrap:wrap">
        <div style="flex:1;min-width:300px"><canvas id="compareDiseaseChart"></canvas></div>
        <div style="flex:1;min-width:300px"><canvas id="compareBugChart"></canvas></div>
      </div>
    </div>
    <div class="gallery-section">
      <div class="gallery-header">
        <h3>SD 卡图片相册</h3>
        <div class="gallery-actions">
          <button class="btn btn-green" onclick="saveToSD()">拍照存卡</button>
          <button class="btn btn-blue" onclick="syncNow()">立即同步</button>
        </div>
      </div>
      <div class="sd-info" id="sdInfo">等待同步...</div>
      <div class="sync-progress" id="syncProgress"></div>
      <div class="gallery-grid" id="galleryGrid">
        <div class="gallery-empty">正在加载...</div>
      </div>
    </div>
  </div>

  <!-- 参数调节 Tab -->
  <div id="tab-params" class="tab-content">
    <div class="mask-grid">
      <div class="mask-item">
        <img id="maskLeaf" src="" alt="叶片分割">
        <div class="caption">叶片分割 Mask</div>
      </div>
      <div class="mask-item">
        <img id="maskDisease" src="" alt="病斑">
        <div class="caption">病斑 Mask</div>
      </div>
      <div class="mask-item">
        <img id="maskPest" src="" alt="虫害">
        <div class="caption">虫害 Mask</div>
      </div>
      <div class="mask-item">
        <img id="maskAnnotated" src="" alt="标注图">
        <div class="caption">最终标注图</div>
      </div>
    </div>
    <div class="params-section">
      <div class="param-actions">
        <button class="btn btn-blue" onclick="applyParams()">应用参数</button>
        <button class="btn btn-gray" onclick="resetParams()">重置默认</button>
        <button class="btn btn-green" onclick="refreshMasks()">刷新预览</button>
      </div>
      <div class="params-grid" id="paramsGrid">
      </div>
    </div>
  </div>

  <!-- 历史记录 Tab -->
  <div id="tab-history" class="tab-content">
    <div class="stats-grid" id="statsGrid">
      <div class="stat-card"><div class="label">总检测次数</div><div class="value" id="statTotal">0</div></div>
      <div class="stat-card"><div class="label">正常</div><div class="value" id="statNormal">0</div></div>
      <div class="stat-card"><div class="label">注意</div><div class="value" id="statNotice">0</div></div>
      <div class="stat-card"><div class="label">警告/严重</div><div class="value" id="statWarning">0</div></div>
    </div>
    <div class="chart-container">
      <h4>近7天病虫害趋势</h4>
      <canvas id="trendChart"></canvas>
    </div>
    <div class="history-section">
      <div class="filter-bar">
        <input type="date" id="dateFrom" placeholder="起始日期">
        <input type="date" id="dateTo" placeholder="结束日期">
        <select id="levelFilter">
          <option value="">全部等级</option>
          <option value="正常">正常</option>
          <option value="注意">注意</option>
          <option value="警告">警告</option>
          <option value="严重">严重</option>
        </select>
        <button onclick="loadHistory()">查询</button>
        <button onclick="exportData()">导出数据</button>
      </div>
      <div class="table-wrapper">
        <table class="history-table" id="historyTable">
          <thead>
            <tr><th>时间</th><th>等级</th><th>病斑</th><th>虫害</th><th>YOLO</th><th>置信度</th><th>操作</th></tr>
          </thead>
          <tbody id="historyBody"></tbody>
        </table>
      </div>
      <div class="page-bar" id="pageBar"></div>
    </div>
  </div>

  <!-- 模型分析 Tab -->
  <div id="tab-analytics" class="tab-content">
    <div class="stats-grid" style="grid-template-columns:repeat(auto-fit,minmax(140px,1fr))">
      <div class="stat-card"><div class="label">模型</div><div class="value" id="anaModel" style="font-size:13px">-</div></div>
      <div class="stat-card"><div class="label">参数量</div><div class="value" id="anaParams" style="font-size:13px">-</div></div>
      <div class="stat-card"><div class="label">设备</div><div class="value" id="anaDevice" style="font-size:13px">-</div></div>
      <div class="stat-card"><div class="label">总推理次数</div><div class="value" id="anaInferences">0</div></div>
      <div class="stat-card"><div class="label">平均耗时</div><div class="value" id="anaAvgMs">0ms</div></div>
      <div class="stat-card"><div class="label">最近耗时</div><div class="value" id="anaLastMs">0ms</div></div>
    </div>
    <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:16px">
      <div class="chart-container" style="flex:1;min-width:280px">
        <h4>检测分布</h4>
        <canvas id="anaDistChart"></canvas>
      </div>
      <div class="chart-container" style="flex:1;min-width:280px">
        <h4>推理耗时趋势 (近50帧)</h4>
        <canvas id="anaTimeChart"></canvas>
      </div>
    </div>
    <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:16px">
      <div class="chart-container" style="flex:1;min-width:280px">
        <h4>YOLO 置信度趋势</h4>
        <canvas id="anaConfChart"></canvas>
      </div>
      <div class="chart-container" style="flex:1;min-width:280px">
        <h4>引擎一致性统计</h4>
        <canvas id="anaAgreeChart"></canvas>
      </div>
    </div>
  </div>
</div>

<div class="modal" id="imageModal" onclick="closeModal()">
  <span class="modal-close">&times;</span>
  <img id="modalImage" src="" alt="大图">
</div>
<div class="toast" id="toast"></div>

<script>
let count = 0;
let syncPollTimer = null;
let savePollTimer = null;
let trendChart = null;
let currentParams = {};
let paramInfo = {};
let previewDebounce = null;
const statusClasses = {0:'status-normal', 1:'status-notice', 2:'status-warning', 3:'status-serious'};

let currentCameraId = '';
let camerasList = {};

async function loadCameraList() {
    try {
        var resp = await fetch('/api/cameras');
        var data = await resp.json();
        camerasList = data.cameras;
        var select = document.getElementById('cameraSelect');
        select.innerHTML = '';
        for (var cid in data.cameras) {
            var cam = data.cameras[cid];
            if (cam.enabled) {
                var opt = document.createElement('option');
                opt.value = cid;
                opt.textContent = cam.name;
                select.appendChild(opt);
            }
        }
        if (!currentCameraId && select.options.length > 0) {
            currentCameraId = select.options[0].value;
            select.value = currentCameraId;
        }
        if (!currentCameraId) {
            console.warn('No enabled cameras available');
        }
    } catch(e) {
        console.error('loadCameraList error:', e);
        document.getElementById('cameraSelect').innerHTML = '<option value="">加载失败</option>';
    }
}

function switchCamera() {
    var sel = document.getElementById('cameraSelect');
    if (!sel || !sel.value) return;
    currentCameraId = sel.value;
    count = 0;
    updateStatus();
    loadGallery();
    loadComparison();
    // Update video feed
    document.getElementById('videoFeed').src = '/video_feed/' + currentCameraId;
}

function apiURL(path) {
    return path + '?camera_id=' + encodeURIComponent(currentCameraId);
}

function showToast(msg, type) {
  var t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast toast-' + (type || 'info') + ' show';
  setTimeout(function(){ t.className = 'toast'; }, 3000);
}

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelector('.tab-btn[data-tab="' + tabId + '"]').classList.add('active');
  document.getElementById('tab-' + tabId).classList.add('active');
  if (tabId === 'params') {
    loadParams();
    refreshMasks();
  } else if (tabId === 'history') {
    loadHistory();
    loadTrend();
    loadStatistics();
  } else if (tabId === 'analytics') {
    loadAnalytics();
  }
}

async function updateStatus() {
  try {
    const data = await (await fetch(apiURL('/api/dual_status'))).json();
    count++;
    document.getElementById('statusBadge').textContent = data.level;
    document.getElementById('statusBadge').className = 'status-box ' + (statusClasses[data.level_code] || 'status-normal');
    document.getElementById('diseaseCount').textContent = data.disease_count;
    document.getElementById('pestCount').textContent = data.white_count;
    document.getElementById('diseaseRatio').textContent = (data.disease_ratio * 100).toFixed(1) + '%';
    document.getElementById('pestRatio').textContent = (data.white_ratio * 100).toFixed(1) + '%';
    document.getElementById('greenRatio').textContent = (data.green_ratio * 100).toFixed(1) + '%';
    document.getElementById('updateCount').textContent = count;
    // YOLO engine status
    var dot = document.getElementById('yoloStatusDot');
    var txt = document.getElementById('yoloStatusText');
    var cb = document.getElementById('yoloEnable');
    if (data.yolo_loaded) {
      dot.className = 'engine-status engine-on';
      txt.textContent = data.yolo_enabled ? '运行中' : '已禁用';
    } else {
      dot.className = 'engine-status engine-off';
      txt.textContent = '未加载';
    }
    cb.checked = data.yolo_enabled;
    // Dual engine fusion result
    if (data.dual) {
      var d = data.dual;
      document.getElementById('dualLevel').textContent = d.level;
      var confMap = {high:'高',medium:'中',low:'低'};
      var confClass = {high:'conf-high',medium:'conf-medium',low:'conf-low'};
      document.getElementById('dualConf').innerHTML = '<span class="conf-badge '+(confClass[d.confidence]||'conf-low')+'">'+(confMap[d.confidence]||d.confidence)+'</span>';
      // Agreement tags
      document.getElementById('agreeDisease').innerHTML = renderAgree(d.agreement.disease);
      document.getElementById('agreeBug').innerHTML = renderAgree(d.agreement.bug);
      // YOLO detection counts
      var yr = d.yolo_result || {};
      document.getElementById('yoloDisease').textContent = yr.disease_count || 0;
      document.getElementById('yoloBug').textContent = yr.bug_count || 0;
      document.getElementById('yoloTime').textContent = yr.inference_ms ? yr.inference_ms.toFixed(1)+'ms' : '-';
    }
    if (data.sd_sync) {
      var sd = data.sd_sync;
      document.getElementById('sdInfo').textContent =
        'SD卡: ' + sd.sd_card_files + '张 | 已同步: ' + sd.total_synced + '张 | 上次同步: ' + (sd.last_sync_time || '无');
      var prog = document.getElementById('syncProgress');
      if (sd.syncing) {
        prog.textContent = '⏳ ' + sd.sync_progress;
        prog.className = 'sync-progress';
      } else if (sd.sync_error) {
        prog.textContent = '❌ ' + sd.sync_error;
        prog.className = 'sync-progress error';
      } else if (sd.sync_progress) {
        prog.textContent = '✅ ' + sd.sync_progress;
        prog.className = 'sync-progress done';
      }
    }
    if (data.error) console.error(data.error);
  } catch(e) { console.error(e); }
}

function renderAgree(status) {
  var map = {
    'agree': ['<span class="agree-tag agree-both">一致</span>'],
    'yolo_only': ['<span class="agree-tag agree-yolo">仅YOLO</span>'],
    'color_only': ['<span class="agree-tag agree-color">仅颜色</span>'],
    'none': ['<span class="agree-tag agree-none">未检出</span>']
  };
  return (map[status] || map['none'])[0];
}

async function toggleYolo(enabled) {
  try {
    await fetch('/api/yolo/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({enabled: enabled})
    });
    showToast('YOLO 引擎已' + (enabled ? '启用' : '禁用'), 'info');
  } catch(e) { showToast('操作失败: ' + e, 'error'); }
}

let yoloConfDebounce = null;
function updateYoloConf(value) {
  document.getElementById('yoloConfVal').textContent = parseFloat(value).toFixed(2);
  if (yoloConfDebounce) clearTimeout(yoloConfDebounce);
  yoloConfDebounce = setTimeout(async function() {
    try {
      await fetch('/api/yolo/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({conf_threshold: parseFloat(value)})
      });
    } catch(e) { console.error(e); }
  }, 500);
}

function toggleAlertPanel() {
  var panel = document.getElementById('alertPanel');
  var hint = document.getElementById('alertToggleHint');
  if (panel.style.display === 'none') {
    panel.style.display = 'block';
    hint.innerHTML = '&#9650;';
    loadAlertConfig();
    loadAlertHistory();
  } else {
    panel.style.display = 'none';
    hint.innerHTML = '&#9660;';
  }
}

async function loadAlertConfig() {
  try {
    var res = await fetch('/api/alert/config');
    var data = await res.json();
    document.getElementById('alertEnable').checked = data.enabled;
    document.getElementById('alertWebhook').value = data.webhook_url || '';
    document.getElementById('alertCooldown').value = data.cooldown_seconds || 300;
    document.getElementById('cooldownVal').textContent = (data.cooldown_seconds || 300) + 's';
  } catch(e) { console.error('loadAlertConfig error:', e); }
}

async function saveAlertConfig() {
  try {
    await fetch('/api/alert/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        enabled: document.getElementById('alertEnable').checked,
        webhook_url: document.getElementById('alertWebhook').value.trim(),
        cooldown_seconds: parseInt(document.getElementById('alertCooldown').value) || 300
      })
    });
  } catch(e) { console.error('saveAlertConfig error:', e); }
}

async function testAlert() {
  try {
    var btn = event.target;
    btn.textContent = '发送中...';
    btn.disabled = true;
    var res = await fetch('/api/alert/test', { method: 'POST' });
    var data = await res.json();
    btn.textContent = data.success ? '发送成功' : '发送失败';
    setTimeout(function() { btn.textContent = '发送测试'; btn.disabled = false; }, 2000);
  } catch(e) {
    event.target.textContent = '发送失败';
    setTimeout(function() { event.target.textContent = '发送测试'; event.target.disabled = false; }, 2000);
  }
}

async function loadAlertHistory() {
  try {
    var res = await fetch('/api/alert/history');
    var data = await res.json();
    var log = document.getElementById('alertLog');
    if (!data.history || data.history.length === 0) {
      log.innerHTML = '暂无告警记录';
      return;
    }
    var html = '';
    data.history.slice(0, 10).forEach(function(item) {
      html += '<div class="alert-log-item">' + item.timestamp + ' | ' + item.level + ' | 病斑:' + item.disease_count + ' 虫害:' + item.pest_count + '</div>';
    });
    log.innerHTML = html || '暂无告警记录';
  } catch(e) { console.error('loadAlertHistory error:', e); }
}

async function loadParams() {
  try {
    const params = await (await fetch('/api/params')).json();
    const info = await (await fetch('/api/params/info')).json();
    currentParams = params;
    paramInfo = info;
    
    const grid = document.getElementById('paramsGrid');
    const sections = {
      '叶片分割参数': ['GREEN_ADV', 'H_MIN', 'H_MAX', 'S_MIN', 'V_HIGH_THRESH', 'MIN_AREA_RATIO', 'MORPH_KERNEL_SIZE', 'DILATE_ITERATIONS'],
      '病斑检测参数': ['BROWN_H_MIN', 'BROWN_H_MAX', 'BROWN_S_MIN', 'YELLOW_H_MIN', 'YELLOW_H_MAX', 'YELLOW_S_MIN', 'DISEASE_MIN_AREA'],
      '虫害检测参数': ['PEST_S_MAX', 'PEST_V_MIN', 'PEST_MIN_AREA'],
      '分级阈值参数': ['NOTICE_DISEASE_COUNT', 'NOTICE_DISEASE_RATIO', 'WARNING_DISEASE_COUNT', 'WARNING_DISEASE_RATIO', 'SERIOUS_DISEASE_COUNT', 'SERIOUS_DISEASE_RATIO', 'NOTICE_PEST_COUNT', 'WARNING_PEST_COUNT', 'WARNING_PEST_RATIO', 'SERIOUS_PEST_COUNT', 'SERIOUS_PEST_RATIO', 'GREEN_RATIO_NOTICE', 'GREEN_RATIO_WARNING']
    };
    
    grid.innerHTML = '';
    for (const [sectionName, keys] of Object.entries(sections)) {
      var group = document.createElement('div');
      group.className = 'param-group';
      group.innerHTML = '<h4>' + sectionName + '</h4>';
      keys.forEach(key => {
        if (paramInfo[key]) {
          var row = document.createElement('div');
          row.className = 'param-row';
          var p = paramInfo[key];
          row.innerHTML = `
            <label>${p.name}</label>
            <input type="range" id="param-${key}" min="${p.min}" max="${p.max}" step="${p.step}" value="${params[key]}" oninput="updateParamValue('${key}', this.value)">
            <div class="value" id="val-${key}">${params[key]}${p.unit}</div>
          `;
          group.appendChild(row);
        }
      });
      grid.appendChild(group);
    }
  } catch(e) { console.error(e); }
}

function updateParamValue(key, value) {
  document.getElementById('val-' + key).textContent = value + (paramInfo[key]?.unit || '');
  currentParams[key] = parseFloat(value);
  if (previewDebounce) clearTimeout(previewDebounce);
  previewDebounce = setTimeout(refreshMasks, 200);
}

async function refreshMasks() {
  try {
    const types = ['leaf', 'disease', 'pest', 'annotated'];
    types.forEach(async type => {
      const resp = await fetch(apiURL('/api/preview_mask'), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({type: type, params: currentParams})
      });
      if (resp.ok) {
        const blob = await resp.blob();
        document.getElementById('mask' + type.charAt(0).toUpperCase() + type.slice(1)).src = URL.createObjectURL(blob);
      }
    });
  } catch(e) { console.error(e); }
}

async function applyParams() {
  try {
    const resp = await fetch('/api/params', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(currentParams)
    });
    const data = await resp.json();
    if (data.success) {
      showToast('参数已保存', 'success');
    } else {
      showToast(data.message || '保存失败', 'error');
    }
  } catch(e) { showToast('请求失败: ' + e, 'error'); }
}

async function resetParams() {
  try {
    const resp = await fetch('/api/params/reset', {method: 'POST'});
    const data = await resp.json();
    if (data.success) {
      currentParams = data.params;
      loadParams();
      refreshMasks();
      showToast('已重置为默认参数', 'success');
    }
  } catch(e) { showToast('请求失败: ' + e, 'error'); }
}

async function loadHistory(page = 1) {
  try {
    const dateFrom = document.getElementById('dateFrom').value;
    const dateTo = document.getElementById('dateTo').value;
    const level = document.getElementById('levelFilter').value;
    
    const resp = await fetch('/api/history?' + new URLSearchParams({
      page: page,
      date_from: dateFrom,
      date_to: dateTo,
      level: level
    }));
    const data = await resp.json();
    
    const body = document.getElementById('historyBody');
    if (data.records.length === 0) {
      body.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#666">暂无记录</td></tr>';
    } else {
      body.innerHTML = '';
      data.records.forEach(r => {
        const row = document.createElement('tr');
        const levelClass = 'level-' + (r.level === '正常' ? 'normal' : r.level === '注意' ? 'notice' : r.level === '警告' ? 'warning' : 'serious');
        row.innerHTML = `
          <td>${r.date} ${r.time}</td>
          <td><span class="level-badge ${levelClass}">${r.level}</span></td>
          <td>${r.disease_count} (${(r.disease_ratio*100).toFixed(1)}%)</td>
          <td>${r.white_count} (${(r.white_ratio*100).toFixed(1)}%)</td>
          <td>${r.yolo_disease_count || 0}病/${r.yolo_bug_count || 0}虫</td>
          <td>${r.dual_confidence ? '<span class="conf-badge conf-'+(r.dual_confidence==='high'?'high':r.dual_confidence==='medium'?'medium':'low')+'">'+({high:'高',medium:'中',low:'低'}[r.dual_confidence]||r.dual_confidence)+'</span>' : '-'}</td>
          <td>${r.image_path ? '<button class="btn btn-blue" style="padding:4px 8px;font-size:12px" onclick="openRecordImage(\\''+r.id+'\\')">查看图片</button>' : '-'}</td>
        `;
        body.appendChild(row);
      });
    }
    
    const pageBar = document.getElementById('pageBar');
    if (data.pages <= 1) {
      pageBar.innerHTML = '';
    } else {
      let html = '<button onclick="loadHistory(1)"' + (page === 1 ? ' disabled' : '') + '>首页</button>';
      html += '<button onclick="loadHistory(' + (page - 1) + ')"' + (page === 1 ? ' disabled' : '') + '>上一页</button>';
      html += '<span style="padding:0 10px">第 ' + page + ' / ' + data.pages + ' 页</span>';
      html += '<button onclick="loadHistory(' + (page + 1) + ')"' + (page === data.pages ? ' disabled' : '') + '>下一页</button>';
      html += '<button onclick="loadHistory(' + data.pages + ')"' + (page === data.pages ? ' disabled' : '') + '>末页</button>';
      pageBar.innerHTML = html;
    }
  } catch(e) { console.error(e); }
}

async function loadTrend() {
  try {
    const resp = await fetch('/api/history/trend');
    const data = await resp.json();
    
    const ctx = document.getElementById('trendChart').getContext('2d');
    if (trendChart) trendChart.destroy();
    
    trendChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.dates,
        datasets: [
          { label: '病斑数量', data: data.disease_counts, borderColor: '#ff4444', tension: 0.3, fill: false },
          { label: '虫害数量', data: data.pest_counts, borderColor: '#4488ff', tension: 0.3, fill: false }
        ]
      },
      options: {
        responsive: true,
        scales: {
          x: { grid: { color: '#333' }, ticks: { color: '#aaa' } },
          y: { grid: { color: '#333' }, ticks: { color: '#aaa' }, beginAtZero: true }
        },
        plugins: {
          legend: { labels: { color: '#eee' } }
        }
      }
    });
  } catch(e) { console.error(e); }
}

async function loadStatistics() {
  try {
    const resp = await fetch('/api/history/statistics');
    const data = await resp.json();
    document.getElementById('statTotal').textContent = data.total_records;
    document.getElementById('statNormal').textContent = data.level_counts['正常'];
    document.getElementById('statNotice').textContent = data.level_counts['注意'];
    document.getElementById('statWarning').textContent = data.level_counts['警告'] + '/' + data.level_counts['严重'];
  } catch(e) { console.error(e); }
}

async function exportData() {
  try {
    const dateFrom = document.getElementById('dateFrom').value;
    const dateTo = document.getElementById('dateTo').value;
    
    const url = '/api/history/export?' + new URLSearchParams({date_from: dateFrom, date_to: dateTo});
    const a = document.createElement('a');
    a.href = url;
    a.download = 'detection_export.zip';
    a.click();
    showToast('数据导出已开始', 'info');
  } catch(e) { showToast('导出失败: ' + e, 'error'); }
}

function openRecordImage(recordId) {
  openModal('/history/image/' + recordId);
}

async function loadGallery() {
  try {
    const files = await (await fetch('/api/sd_images')).json();
    const grid = document.getElementById('galleryGrid');
    if (files.length === 0) {
      grid.innerHTML = '<div class="gallery-empty">暂无图片, 点击"拍照存卡"或等待自动同步</div>';
      return;
    }
    grid.innerHTML = '';
    files.forEach(function(f) {
      var div = document.createElement('div');
      div.className = 'gallery-item';
      div.onclick = function() { openModal('/dataset/' + encodeURIComponent(f)); };
      div.innerHTML = '<img src="/dataset/' + encodeURIComponent(f) + '" alt="' + f + '" loading="lazy"><div class="name">' + f + '</div>';
      grid.appendChild(div);
    });
  } catch(e) {
    console.error(e);
    document.getElementById('galleryGrid').innerHTML = '<div class="gallery-empty">加载失败，请刷新页面</div>';
  }
}

async function saveToSD() {
  showToast('正在拍照存卡，请稍候...', 'info');
  try {
    var resp = await (await fetch(apiURL('/api/save_to_sd'))).json();
    if (resp.started) {
      startSavePolling();
    } else {
      showToast(resp.message || '拍照任务已在执行', 'info');
    }
  } catch(e) { showToast('请求失败: ' + e, 'error'); }
}

function startSavePolling() {
  if (savePollTimer) clearInterval(savePollTimer);
  savePollTimer = setInterval(async function() {
    try {
      var data = await (await fetch('/api/save_status')).json();
      if (data.done) {
        clearInterval(savePollTimer);
        savePollTimer = null;
        if (data.error) {
          showToast('拍照失败: ' + data.error, 'error');
        } else {
          showToast('拍照成功! 文件: ' + data.file, 'success');
          loadGallery();
          startSyncPolling();
        }
      }
    } catch(e) { console.error(e); }
  }, 2000);
}

function startSyncPolling() {
  if (syncPollTimer) clearInterval(syncPollTimer);
  syncPollTimer = setInterval(async function() {
    try {
      var data = await (await fetch('/api/sync_status')).json();
      if (!data.syncing) {
        clearInterval(syncPollTimer);
        syncPollTimer = null;
        loadGallery();
        if (data.sync_error) {
          showToast('同步出错: ' + data.sync_error, 'error');
        } else {
          showToast('同步完成! 新下载 ' + data.new_downloaded + ' 张', 'success');
        }
      }
    } catch(e) { console.error(e); }
  }, 2000);
}

async function syncNow() {
  try {
    var resp = await (await fetch(apiURL('/api/sync_now'))).json();
    if (resp.started) {
      showToast('同步已启动，请稍候...', 'info');
      startSyncPolling();
    } else {
      showToast(resp.message || '同步已在进行中', 'info');
    }
  } catch(e) { showToast('同步请求失败: ' + e, 'error'); }
}

function openModal(src) {
  var img = document.getElementById('modalImage');
  img.src = src;
  document.getElementById('imageModal').classList.add('active');
}

function closeModal() {
  document.getElementById('imageModal').classList.remove('active');
}

let compareDiseaseChart = null;
let compareBugChart = null;

async function loadComparison() {
  try {
    const data = await (await fetch(apiURL('/api/comparison') + '&n=50')).json();
    const frames = data.frames || [];
    if (frames.length === 0) return;

    const labels = frames.map((f,i) => i+1);
    const colorDisease = frames.map(f => f.color_disease);
    const yoloDisease = frames.map(f => f.yolo_disease);
    const colorBug = frames.map(f => f.color_bug);
    const yoloBug = frames.map(f => f.yolo_bug);

    const chartOpts = {
      responsive: true,
      animation: false,
      scales: {
        x: { grid: { color: '#333' }, ticks: { color: '#aaa', maxTicksLimit: 10 }, title: { display: true, text: '帧序号', color: '#aaa' } },
        y: { grid: { color: '#333' }, ticks: { color: '#aaa' }, beginAtZero: true, title: { display: true, text: '数量', color: '#aaa' } }
      },
      plugins: { legend: { labels: { color: '#eee' } } }
    };

    // Disease comparison chart
    const ctx1 = document.getElementById('compareDiseaseChart').getContext('2d');
    if (compareDiseaseChart) compareDiseaseChart.destroy();
    compareDiseaseChart = new Chart(ctx1, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          { label: '颜色引擎-病斑', data: colorDisease, borderColor: '#ff4444', backgroundColor: 'rgba(255,68,68,0.1)', tension: 0.2, fill: false, pointRadius: 0 },
          { label: 'YOLO-病斑', data: yoloDisease, borderColor: '#ff9800', backgroundColor: 'rgba(255,152,0,0.1)', tension: 0.2, fill: false, pointRadius: 0, borderDash: [5,3] }
        ]
      },
      options: Object.assign({}, chartOpts, { plugins: Object.assign({}, chartOpts.plugins, { title: { display: true, text: '病斑检测对比', color: '#88ccff' } }) })
    });

    // Bug comparison chart
    const ctx2 = document.getElementById('compareBugChart').getContext('2d');
    if (compareBugChart) compareBugChart.destroy();
    compareBugChart = new Chart(ctx2, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          { label: '颜色引擎-虫害', data: colorBug, borderColor: '#4488ff', backgroundColor: 'rgba(68,136,255,0.1)', tension: 0.2, fill: false, pointRadius: 0 },
          { label: 'YOLO-虫害', data: yoloBug, borderColor: '#4caf50', backgroundColor: 'rgba(76,175,80,0.1)', tension: 0.2, fill: false, pointRadius: 0, borderDash: [5,3] }
        ]
      },
      options: Object.assign({}, chartOpts, { plugins: Object.assign({}, chartOpts.plugins, { title: { display: true, text: '虫害检测对比', color: '#88ccff' } }) })
    });
  } catch(e) { console.error('Comparison chart error:', e); }
}

// ============ S7: 模型分析图表 ============
let anaDistChart = null, anaTimeChart = null, anaConfChart = null, anaAgreeChart = null;

async function loadAnalytics() {
  try {
    // 并行获取 YOLO 统计和对比历史
    var statsRes = await fetch('/api/yolo_stats');
    var stats = await statsRes.json();
    var compRes = await fetch(apiURL('/api/comparison') + '&n=50');
    var compData = await compRes.json();
    var frames = compData.frames || [];

    // 1. 更新统计卡片
    document.getElementById('anaModel').textContent = stats.model_name || '-';
    document.getElementById('anaParams').textContent = stats.model_params || '-';
    document.getElementById('anaDevice').textContent = stats.device || '-';
    document.getElementById('anaInferences').textContent = stats.total_inferences || 0;
    document.getElementById('anaAvgMs').textContent = (stats.avg_inference_ms || 0).toFixed(1) + 'ms';
    document.getElementById('anaLastMs').textContent = (stats.last_inference_ms || 0).toFixed(1) + 'ms';

    var darkOpts = {
      responsive: true, animation: false,
      plugins: { legend: { labels: { color: '#eee' } } },
      scales: { x: { grid: { color: '#333' }, ticks: { color: '#aaa', maxTicksLimit: 10 } }, y: { grid: { color: '#333' }, ticks: { color: '#aaa' } } }
    };

    // 2. 检测分布 (环形图)
    var ctx0 = document.getElementById('anaDistChart').getContext('2d');
    if (anaDistChart) anaDistChart.destroy();
    anaDistChart = new Chart(ctx0, {
      type: 'doughnut',
      data: {
        labels: ['YOLO-病斑', 'YOLO-虫害'],
        datasets: [{ data: [stats.disease_detections || 0, stats.bug_detections || 0], backgroundColor: ['#ff9800', '#4caf50'], borderColor: ['#e65100', '#2e7d32'], borderWidth: 2 }]
      },
      options: { responsive: true, animation: false, plugins: { legend: { labels: { color: '#eee' } } } }
    });

    // 3. 推理耗时趋势
    if (frames.length > 0) {
      var labels = frames.map(function(f,i){ return i+1; });
      var msData = frames.map(function(f){ return f.inference_ms || 0; });
      var ctx1 = document.getElementById('anaTimeChart').getContext('2d');
      if (anaTimeChart) anaTimeChart.destroy();
      anaTimeChart = new Chart(ctx1, {
        type: 'line',
        data: { labels: labels, datasets: [{ label: '耗时(ms)', data: msData, borderColor: '#4fc3f7', backgroundColor: 'rgba(79,195,247,0.1)', tension: 0.3, fill: true, pointRadius: 0 }] },
        options: Object.assign({}, darkOpts, { scales: Object.assign({}, darkOpts.scales, { y: Object.assign({}, darkOpts.scales.y, { beginAtZero: true, title: { display: true, text: 'ms', color: '#aaa' } } ) }) })
      });

      // 4. 置信度趋势
      var diseaseConf = frames.map(function(f){ return f.yolo_disease_conf || 0; });
      var bugConf = frames.map(function(f){ return f.yolo_bug_conf || 0; });
      var ctx2 = document.getElementById('anaConfChart').getContext('2d');
      if (anaConfChart) anaConfChart.destroy();
      anaConfChart = new Chart(ctx2, {
        type: 'line',
        data: { labels: labels, datasets: [
          { label: '病斑置信度', data: diseaseConf, borderColor: '#ff9800', tension: 0.3, fill: false, pointRadius: 0 },
          { label: '虫害置信度', data: bugConf, borderColor: '#4caf50', tension: 0.3, fill: false, pointRadius: 0 }
        ]},
        options: Object.assign({}, darkOpts, { scales: Object.assign({}, darkOpts.scales, { y: Object.assign({}, darkOpts.scales.y, { beginAtZero: true, max: 1, title: { display: true, text: '置信度', color: '#aaa' } } ) }) })
      });

      // 5. 引擎一致性统计 (堆叠柱状图)
      var agreeCount = 0, yoloOnlyCount = 0, colorOnlyCount = 0, noneCount = 0;
      frames.forEach(function(f) {
        var yoloHas = (f.yolo_disease > 0 || f.yolo_bug > 0);
        var colorHas = (f.color_disease > 0 || f.color_bug > 0);
        if (yoloHas && colorHas) agreeCount++;
        else if (yoloHas && !colorHas) yoloOnlyCount++;
        else if (!yoloHas && colorHas) colorOnlyCount++;
        else noneCount++;
      });
      var ctx3 = document.getElementById('anaAgreeChart').getContext('2d');
      if (anaAgreeChart) anaAgreeChart.destroy();
      anaAgreeChart = new Chart(ctx3, {
        type: 'bar',
        data: {
          labels: ['一致性统计'],
          datasets: [
            { label: '双引擎一致', data: [agreeCount], backgroundColor: '#2e7d32' },
            { label: '仅YOLO', data: [yoloOnlyCount], backgroundColor: '#ff9800' },
            { label: '仅颜色', data: [colorOnlyCount], backgroundColor: '#ef6c00' },
            { label: '均未检出', data: [noneCount], backgroundColor: '#546e7a' }
          ]
        },
        options: Object.assign({}, darkOpts, { scales: Object.assign({}, darkOpts.scales, { x: Object.assign({}, darkOpts.scales.x, { stacked: true }), y: Object.assign({}, darkOpts.scales.y, { stacked: true, beginAtZero: true }) }), plugins: Object.assign({}, darkOpts.plugins, { title: { display: true, text: '近' + frames.length + '帧引擎一致性', color: '#88ccff' } }) })
      });
    }
  } catch(e) { console.error('loadAnalytics error:', e); }
}

// 启动时加载摄像头列表
loadCameraList().then(function() {
    document.getElementById('videoFeed').src = '/video_feed/' + currentCameraId;
    setInterval(updateStatus, 2000);
    setInterval(loadGallery, 60000);
    setInterval(loadComparison, 10000);
    updateStatus();
    loadGallery();
    loadComparison();
});

// One-time sync of YOLO confidence slider
fetch('/api/yolo_stats').then(r => r.json()).then(stats => {
  if (stats.conf_threshold) {
    document.getElementById('yoloConfSlider').value = stats.conf_threshold;
    document.getElementById('yoloConfVal').textContent = stats.conf_threshold.toFixed(2);
  }
}).catch(()=>{});

// === AI 深度诊断 ===
function deepDiagnose() {
  var btn = document.getElementById('diagnoseBtn');
  var modal = document.getElementById('reportModal');
  var textEl = document.getElementById('reportText');
  var loadingEl = document.getElementById('reportLoading');
  btn.disabled = true;
  btn.textContent = '\u23F3 \u8BCA\u65AD\u4E2D...';
  textEl.textContent = '';
  loadingEl.style.display = 'block';
  textEl.style.display = 'none';
  modal.classList.add('show');

  fetch('/api/deep_diagnose', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({camera_id: currentCameraId})
  }).then(function(r) { return r.json(); }).then(function(data) {
    loadingEl.style.display = 'none';
    textEl.style.display = 'block';
    btn.disabled = false;
    btn.innerHTML = '&#129504; AI\u8BCA\u65AD';
    if (data.error) {
      textEl.textContent = '\u8BCA\u65AD\u5931\u8D25: ' + data.error;
      return;
    }
    // 打字机效果展示报告
    var report = data.report || '\u672A\u8FD4\u56DE\u62A5\u544A\u5185\u5BB9';
    var idx = 0;
    textEl.textContent = '';
    function typeChar() {
      if (idx < report.length) {
        textEl.textContent += report.charAt(idx);
        idx++;
        setTimeout(typeChar, 30);
      }
    }
    typeChar();
  }).catch(function(e) {
    loadingEl.style.display = 'none';
    textEl.style.display = 'block';
    textEl.textContent = '\u8BF7\u6C42\u5931\u8D25: ' + e.message;
    btn.disabled = false;
    btn.innerHTML = '&#129504; AI\u8BCA\u65AD';
  });
}
function closeReport() {
  document.getElementById('reportModal').classList.remove('show');
}
</script>
<!-- AI诊断报告弹窗 -->
<div class="report-modal" id="reportModal">
  <div class="report-modal-content">
    <button class="report-close" onclick="closeReport()">&times;</button>
    <h2>&#129504; AI \u6DF1\u5EA6\u8BCA\u65AD\u62A5\u544A</h2>
    <div class="report-loading" id="reportLoading"><div class="spinner"></div><br>\u6B63\u5728\u5206\u6790\u56FE\u50CF\uFF0C\u8BF7\u7A0D\u5019...</div>
    <div class="report-text" id="reportText" style="display:none"></div>
  </div>
</div>
</body>
</html>"""


# ============================================================
# Flask 路由 - Web API 接口定义
# ============================================================
# 路由分为以下几类:
#   1. 页面路由:     /                          -> 返回嵌入的 HTML 页面
#   2. 摄像头接口:   /api/cameras               -> 获取所有摄像头配置
#   3. 状态接口:     /api/status                -> 获取实时检测结果
#   4. 参数接口:     /api/params (GET/POST)     -> 读取/保存检测参数
#                   /api/params/info            -> 获取参数元信息（名称/范围/单位）
#                   /api/params/reset           -> 重置为默认参数
#   5. 预览接口:     /api/preview_mask           -> 生成 mask 预览图
#   6. 历史接口:     /api/history               -> 分页查询历史记录
#                   /api/history/trend          -> 趋势数据（折线图）
#                   /api/history/statistics     -> 统计数据（统计卡片）
#                   /api/history/export         -> 导出 ZIP 文件
#                   /history/image/<id>         -> 查看历史标注图
#   7. SD 卡接口:    /api/sd_images             -> 获取 SD 卡图片列表
#                   /dataset/<filename>         -> 提供 SD 卡图片静态访问
#                   /api/save_to_sd             -> 触发远程拍照
#                   /api/save_status            -> 查询拍照进度
#                   /api/sync_now               -> 触发 SD 卡同步
#                   /api/sync_status            -> 查询同步进度
#   8. 视频流接口:   /video_feed/<camera_id>    -> MJPEG 实时视频流

# ---------- 1. 页面路由 ----------
# 根路由: 返回完整的单页面 HTML（包含 CSS + JavaScript）
# 前端代码以 Python 字符串形式嵌入在 HTML_PAGE 变量中
@app.route('/')
def index():
    return HTML_PAGE


@app.route('/health/live')
def health_live():
    """Liveness probe: the web process is accepting requests."""
    return jsonify({"status": "alive", "service": "system-b"})


@app.route('/health/ready')
def health_ready():
    """Readiness probe: report model and camera availability."""
    camera_summaries = []
    for camera_id, camera in cameras.items():
        with camera["state"]["frame_lock"]:
            camera_summaries.append(summarize_camera(camera_id, camera["state"]))

    camera_ready = any(camera["has_frame"] for camera in camera_summaries if cameras[camera["id"]].get("enabled", True))
    yolo_ready = (not yolo_enabled) or yolo_detector.is_loaded()
    payload = build_service_health("system-b", {"camera": camera_ready, "yolo": yolo_ready})
    payload["cameras"] = camera_summaries
    return jsonify(payload), (200 if payload["status"] == "ready" else 503)


@app.route('/health')
def health():
    """Compatibility health endpoint; use /health/ready for deployment probes."""
    return health_ready()


# ---------- 2. 摄像头接口 ----------
@app.route('/api/cameras')
def api_cameras():
    """返回所有摄像头配置（不含内部state）"""
    result = {}
    for cid, cam in cameras.items():
        result[cid] = {"id": cam["id"], "name": cam["name"], "enabled": cam.get("enabled", True)}
    return jsonify({"cameras": result})


# ---------- 3. 状态接口 ----------
# 前端每 2 秒轮询此接口获取最新检测结果和 SD 同步状态
# 返回 JSON: { level, level_code, disease_count, white_count, disease_ratio, white_ratio, green_ratio, error, sd_sync }
@app.route('/api/status')
def api_status():
    camera_id = request.args.get('camera_id', _get_default_camera_id())
    cam = cameras.get(camera_id)
    if not cam:
        return jsonify({"error": "摄像头不存在"}), 404
    state = cam["state"]
    # 加锁复制检测结果，避免读取过程中被检测线程覆盖
    with state["frame_lock"]:
        data = state["latest_result"].copy()
    # 附加错误信息
    data['error'] = state["last_error"]
    # 附加 SD 卡同步状态信息
    with sd_lock:
        data['sd_sync'] = sd_sync_info.copy()
    return jsonify(data)


# ---------- 3B. 双引擎接口 ----------
# /api/dual_status: 返回双引擎融合后的检测状态
@app.route('/api/dual_status')
def api_dual_status():
    camera_id = request.args.get('camera_id', _get_default_camera_id())
    cam = cameras.get(camera_id)
    if not cam:
        return jsonify({"error": "摄像头不存在"}), 404
    state = cam["state"]
    with state["frame_lock"]:
        # 基础状态（颜色引擎 + 防抖等级）
        data = state["latest_result"].copy()
        # 双引擎融合结果
        if state["latest_dual_result"] is not None:
            data['dual'] = state["latest_dual_result"]
        else:
            data['dual'] = None
        # YOLO 引擎开关状态
        data['yolo_enabled'] = yolo_enabled
        data['yolo_loaded'] = yolo_detector.is_loaded()
    data['error'] = state["last_error"]
    with sd_lock:
        data['sd_sync'] = sd_sync_info.copy()
    return jsonify(data)


# /api/yolo_stats: 返回 YOLO 模型运行状态和统计信息
@app.route('/api/yolo_stats')
def api_yolo_stats():
    return jsonify(yolo_detector.get_stats())


# /api/yolo/config: 更新 YOLO 检测参数
@app.route('/api/yolo/config', methods=['POST'])
def api_yolo_config():
    global yolo_enabled
    try:
        params = request.get_json()
        if 'enabled' in params:
            yolo_enabled = bool(params['enabled'])
            print(f"[API] YOLO 引擎: {'启用' if yolo_enabled else '禁用'}")
        if 'conf_threshold' in params or 'iou_threshold' in params:
            yolo_detector.update_config(
                conf_threshold=params.get('conf_threshold'),
                iou_threshold=params.get('iou_threshold')
            )
        if 'dual_yolo_conf_high' in params:
            dual_verifier.yolo_conf_high = float(params['dual_yolo_conf_high'])
        if 'dual_yolo_conf_low' in params:
            dual_verifier.yolo_conf_low = float(params['dual_yolo_conf_low'])
        return jsonify({"success": True, "message": "YOLO 参数已更新"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# /api/comparison: 获取最近 N 帧的双引擎对比数据
@app.route('/api/comparison')
def api_comparison():
    camera_id = request.args.get('camera_id', _get_default_camera_id())
    cam = cameras.get(camera_id)
    if not cam:
        return jsonify({"frames": [], "total": 0})
    state = cam["state"]
    n = request.args.get('n', 50, type=int)
    n = min(n, COMPARISON_HISTORY_SIZE)
    with state["frame_lock"]:
        history_copy = list(state["comparison_history"])
    return jsonify({
        'frames': history_copy[-n:],
        'total': len(history_copy)
    })


# ---------- 3C. 告警通知接口 ----------
# GET /api/alert/config:  获取告警通知配置
# POST /api/alert/config: 更新告警通知配置（webhook_url, enabled, cooldown）
# GET /api/alert/history: 获取告警历史记录
@app.route('/api/alert/config', methods=['GET', 'POST'])
def api_alert_config():
    if request.method == 'GET':
        return jsonify(alert_notifier.get_config())
    else:
        try:
            params = request.get_json()
            result = alert_notifier.configure(
                webhook_url=params.get('webhook_url'),
                enabled=params.get('enabled'),
                cooldown=params.get('cooldown_seconds')
            )
            return jsonify({"success": True, "config": result})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/alert/history')
def api_alert_history():
    return jsonify({'history': alert_notifier.get_history()})


@app.route('/api/alert/test', methods=['POST'])
def api_alert_test():
    """发送测试告警"""
    try:
        result = alert_notifier.notify(
            level='测试', level_code=2,
            disease_count=3, pest_count=1,
            disease_ratio=0.15, pest_ratio=0.05,
            confidence='high',
        )
        return jsonify({'success': result.get('sent', False), 'reason': result.get('reason', '')})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ---------- 4. 参数接口 ----------
# GET:  获取当前检测参数配置（参数字典）
# POST: 保存新的检测参数（前端传入完整的参数字典覆盖保存）
# 使用 config_lock 确保参数读写与检测线程的参数读取互斥
@app.route('/api/params', methods=['GET', 'POST'])
def api_params():
    if request.method == 'GET':
        # 读取当前配置并返回 JSON
        with config_lock:
            return jsonify(config_manager.get_current_config())
    else:
        # 保存用户修改后的参数
        try:
            params = request.get_json(silent=True)
            is_valid, errors = config_manager.validate_params(params)
            if not is_valid:
                return jsonify({"success": False, "errors": errors}), 400
            with config_lock:
                config_manager.update_params(params)
            return jsonify({"success": True, "message": "参数已保存"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500


# 获取参数元信息: 每个参数的显示名称、最小值、最大值、步长、单位
# 前端根据这些信息动态生成滑块控件
@app.route('/api/params/info')
def api_params_info():
    return jsonify(config_manager.get_param_info())


# 重置参数为默认值: 将配置恢复为 DetectionConfig 的初始值
# 返回重置后的参数，前端据此刷新滑块显示
@app.route('/api/params/reset', methods=['POST'])
def api_params_reset():
    try:
        with config_lock:
            config_manager.reset_config()
        return jsonify({"success": True, "params": config_manager.get_current_config()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ---------- 5. 预览接口 ----------
# 根据前端传来的参数和 mask 类型，生成对应的分割预览图
# 前端参数调节页面有四个预览窗口: leaf(叶片分割)、disease(病斑)、pest(虫害)、annotated(标注)
# 用户拖动滑块时，前端以 200ms 防抖频率调用此接口，实时查看参数效果
@app.route('/api/preview_mask', methods=['POST'])
def api_preview_mask():
    try:
        camera_id = request.args.get('camera_id', _get_default_camera_id())
        cam = cameras.get(camera_id)
        if not cam:
            return jsonify({"error": "摄像头不存在"}), 404
        state = cam["state"]
        
        data = request.get_json()
        mask_type = data.get('type', 'leaf')    # mask 类型: leaf/disease/pest/annotated
        params = data.get('params', {})          # 当前滑块参数值

        # 获取最新的原始图像（未标注），用于生成预览
        with state["frame_lock"]:
            image = state["latest_original_image"]

        # 如果尚未获取到图像（系统刚启动），返回错误提示
        if image is None:
            return jsonify({"error": "暂无图像"}), 400

        # 调用检测模块的 mask 生成函数，返回 JPEG 编码的字节数据
        mask_data = generate_mask_image(image, params, mask_type)
        return Response(mask_data, mimetype='image/jpeg')
    except Exception as e:
        print(f"预览生成失败: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------- 6. 历史记录接口 ----------
# 分页查询历史检测记录，支持按日期范围和等级过滤
# 查询参数: page(页码), date_from(起始日期), date_to(结束日期), level(等级过滤)
# 返回 JSON: { records: [...], pages: 总页数 }
@app.route('/api/history')
def api_history():
    try:
        page = int(request.args.get('page', 1))
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        level = request.args.get('level', None)

        # 空字符串视为不过滤（前端 select 未选择时传空字符串）
        if level == '':
            level = None

        result = history_manager.query_records(date_from, date_to, level, page)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 趋势数据接口: 返回近 N 天（默认 7 天）每日的平均病斑数和虫害数
# 返回 JSON: { dates: [...], disease_counts: [...], pest_counts: [...] }
# 前端使用 Chart.js 渲染为折线图
@app.route('/api/history/trend')
def api_history_trend():
    try:
        days = int(request.args.get('days', 7))
        result = history_manager.get_trend_data(days)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 统计数据接口: 返回各等级的记录数量分布
# 返回 JSON: { total_records: 总数, level_counts: { 正常: n, 注意: n, 警告: n, 严重: n } }
@app.route('/api/history/statistics')
def api_history_statistics():
    try:
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        result = history_manager.get_statistics(date_from, date_to)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 数据导出接口: 将指定日期范围的历史记录打包为 ZIP 文件下载
# ZIP 内包含: CSV 数据文件 + 对应的标注图片
@app.route('/api/history/export')
def api_history_export():
    try:
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        # 由 history_manager 生成 ZIP 文件并返回路径
        zip_path = history_manager.export_data(date_from, date_to)

        if not zip_path or not os.path.exists(zip_path):
            return jsonify({"error": "导出失败"}), 500

        # 以附件形式发送 ZIP 文件，浏览器会自动触发下载
        return send_file(zip_path, as_attachment=True, download_name=os.path.basename(zip_path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 历史记录图片查看: 根据记录 ID 返回对应的标注图
# 前端历史记录表格中的"查看图片"按钮会打开模态框加载此 URL
@app.route('/history/image/<record_id>')
def serve_history_image(record_id):
    record = history_manager.get_record_by_id(record_id)
    # 验证记录存在、图片路径非空、文件确实存在后，才发送文件
    if record and record.get('image_path') and os.path.exists(record['image_path']):
        return send_file(record['image_path'], mimetype='image/jpeg')
    return "图片不存在", 404


# ---------- 7. SD 卡相关接口 ----------
# 获取本地已同步的 SD 卡图片文件名列表（按时间倒序）
# 前端相册页面使用此列表渲染图片网格
@app.route('/api/sd_images')
def api_sd_images():
    if not os.path.exists(DATASET_DIR):
        return jsonify([])
    # 按文件名倒序排列（最新的在前面），并过滤非图片文件
    files = sorted(os.listdir(DATASET_DIR), reverse=True)
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]
    return jsonify(files)


# SD 卡图片静态访问: 根据文件名提供图片文件
# 前端相册中的每个图片项通过此 URL 加载缩略图
# 注意: path 类型参数支持包含斜杠的路径（如子目录下的文件）
@app.route('/dataset/<path:filename>')
def serve_dataset_image(filename):
    filepath = os.path.join(DATASET_DIR, filename)
    if not os.path.isfile(filepath):
        return "图片不存在，请先同步", 404
    return send_from_directory(DATASET_DIR, filename)


# 触发远程拍照存卡: 在独立守护线程中异步执行，立即返回
# 前端通过轮询 /api/save_status 获取执行结果
@app.route('/api/save_to_sd')
def api_save_to_sd():
    camera_id = request.args.get('camera_id', _get_default_camera_id())
    # 防止重复触发: 如果已有拍照任务在执行，拒绝新请求
    with save_sd_lock:
        is_saving = save_sd_info["saving"]
    if is_saving:
        return jsonify({"started": False, "message": "已有拍照任务在执行"})
    # 启动守护线程执行拍照操作（daemon=True 确保主线程退出时自动清理）
    t = threading.Thread(target=save_to_sd_async, args=(camera_id,), daemon=True)
    t.start()
    return jsonify({"started": True, "message": "拍照已启动"})


# 查询远程拍照进度: 前端轮询此接口直到 done=True
@app.route('/api/save_status')
def api_save_status():
    with save_sd_lock:
        return jsonify(dict(save_sd_info))


# 手动触发 SD 卡同步: 在独立守护线程中异步执行
# 与后台自动同步共享同一个 sync_sd_card() 函数，通过 syncing 标志防重复
@app.route('/api/sync_now')
def api_sync_now():
    camera_id = request.args.get('camera_id', _get_default_camera_id())
    with sd_lock:
        syncing = sd_sync_info["syncing"]
    if syncing:
        return jsonify({"started": False, "message": "同步已在进行中"})
    t = threading.Thread(target=sync_sd_card, args=(camera_id,), daemon=True)
    t.start()
    return jsonify({"started": True, "message": "同步已启动"})


# 查询 SD 卡同步状态: 前端轮询此接口显示同步进度
@app.route('/api/sync_status')
def api_sync_status():
    with sd_lock:
        return jsonify(sd_sync_info.copy())


@app.route('/api/offline_events')
def api_offline_events():
    """Return bounded detection events waiting for a future transport worker."""
    events = offline_event_cache.list_pending()
    return jsonify({"count": len(events), "events": events})


@app.route('/api/offline_events/ack', methods=['POST'])
def api_offline_events_ack():
    """Acknowledge one event after an external transport confirms delivery."""
    params = request.get_json(silent=True) or {}
    event_id = params.get("event_id")
    try:
        offline_event_cache.ack(event_id)
    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 400
    return jsonify({"success": True})


@app.route('/api/offline_events/sync', methods=['POST'])
def api_offline_events_sync():
    """Manually sync a bounded batch; remove events only after 2xx delivery."""
    if event_transport is None:
        return jsonify({"success": False, "error": "event sink is not configured"}), 503
    if not offline_event_sync_lock.acquire(blocking=False):
        return jsonify({"success": False, "error": "event sync is already running"}), 409

    try:
        params = request.get_json(silent=True) or {}
        if not isinstance(params, dict):
            return jsonify({"success": False, "error": "request body must be a JSON object"}), 400
        limit = params.get("limit", 50)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            return jsonify({"success": False, "error": "limit must be an integer from 1 to 100"}), 400

        events = offline_event_cache.list_pending()[:limit]
        result = event_transport.sync(events, offline_event_cache.ack)
        pending = len(offline_event_cache.list_pending())
        return jsonify({
            "success": result["sent"] == len(events),
            "pending": pending,
            **result,
        })
    finally:
        offline_event_sync_lock.release()


# ---------- 8. MJPEG 视频流接口 ----------
# 该接口实现 MJPEG (Motion JPEG) 视频流推送
# 原理: 使用 HTTP 分块传输编码 (chunked transfer) + multipart/x-mixed-replace MIME 类型
#       浏览器会将每个 chunk 中的 JPEG 帧依次替换显示在 <img> 标签中，形成"视频"效果
# 帧率: 每 0.5 秒输出一帧（约 2 FPS），因为后台检测每 3 秒才更新一次最新帧，
#       中间帧是重复的同一张图片，但保持 HTTP 连接不中断
@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    cam = cameras.get(camera_id)
    if not cam:
        return "Camera not found", 404
    state = cam["state"]
    def generate():
        """MJPEG 帧生成器: 无限循环产生 MJPEG 格式的帧数据"""
        while True:
            with state["frame_lock"]:
                frame = state["latest_frame"]
                error = state["last_error"]
            if frame:
                # 有检测帧时，输出 JPEG 编码的标注图
                # 格式: --frame 分隔符 + Content-Type 头 + JPEG 数据
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # 尚无检测帧时（系统刚启动），输出一张黑色占位图
                # 占位图显示"等待中..."或错误信息
                frame = create_wait_image(error or "等待检测中...")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # 帧间隔 0.5 秒，控制输出频率
            time.sleep(0.5)
    # 返回分块传输响应，MIME 类型为 multipart/x-mixed-replace
    # boundary=frame 定义了帧分隔符，浏览器据此识别每一帧的边界
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 向后兼容: /video_feed 重定向到默认摄像头
@app.route('/video_feed')
def video_feed_default():
    default_cam = _get_default_camera_id()
    return video_feed(default_cam)


# ============================================================
# Phase 11: A-B 浅连接 - System B 代理调用 System A 深度诊断
# ============================================================

@app.route('/api/deep_diagnose', methods=['POST'])
def api_deep_diagnose():
    """
    代理端点: 获取当前摄像头帧，发送到 System A /report，返回诊断报告。
    前端"AI深度诊断"按钮调用此端点。
    流程:
      1. 从 cameras[camera_id].state 获取当前帧
      2. JPEG 编码为字节流
      3. POST 到 System A 的 /report 端点
      4. 等待返回（约72秒，LLM 推理较慢）
      5. 将报告返回给前端
    """
    try:
        params = request.get_json() or {}
        camera_id = params.get('camera_id', _get_default_camera_id())
        cam = cameras.get(camera_id)
        if not cam:
            return jsonify({"error": "摄像头不存在"}), 404

        state = cam["state"]
        with state["frame_lock"]:
            frame = state["latest_frame"]
            original_image = state["latest_original_image"]

        if frame is None:
            return jsonify({"error": "暂无画面，请稍后再试"}), 400

        # 优先使用原始图（未标注），如果没有则用标注帧
        if original_image is not None:
            _, buf = cv2.imencode('.jpg', original_image)
            image_bytes = buf.tobytes()
        else:
            # frame 已经是 JPEG 字节流，直接使用
            image_bytes = frame

        # 发送到 System A 的 /report 端点
        report_url = f"{SYSTEM_A_URL}/report"
        print(f"[诊断] 正在将 {cam['name']} 的画面发送到 System A: {report_url}")

        resp = requests.post(
            report_url,
            files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
            timeout=180  # LLM 推理较慢，给足超时
        )

        if resp.status_code == 200:
            report_data = resp.json()
            print(f"[诊断] System A 返回报告成功")
            return jsonify({"success": True, "report": report_data})
        else:
            print(f"[诊断] System A 返回错误: {resp.status_code} {resp.text[:200]}")
            return jsonify({
                "success": False,
                "error": f"System A 返回 {resp.status_code}"
            }), 502

    except requests.exceptions.Timeout:
        return jsonify({"success": False, "error": "System A 响应超时（LLM推理可能需要2分钟）"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"success": False, "error": "无法连接 System A，请确认 System A 已启动"}), 503
    except Exception as e:
        print(f"[诊断] 异常: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/system_a/status')
def api_system_a_status():
    """检查 System A 是否在线"""
    try:
        resp = requests.get(f"{SYSTEM_A_URL}/health", timeout=5)
        if resp.status_code == 200:
            return jsonify({"online": True, "url": SYSTEM_A_URL})
        return jsonify({"online": False, "url": SYSTEM_A_URL})
    except Exception:
        return jsonify({"online": False, "url": SYSTEM_A_URL})


# ============================================================
# Phase 10: 大屏网格视图 - 多摄像头同时监控
# ============================================================

DASHBOARD_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgriVision 监控大屏</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0a1628; color:#e0e0e0; font-family:'Microsoft YaHei',sans-serif; overflow:hidden; height:100vh; }

.dashboard-header {
    background:linear-gradient(135deg,#0d2137,#1a3a5c);
    padding:12px 24px; display:flex; align-items:center; justify-content:space-between;
    border-bottom:2px solid #1e5a8a;
}
.dashboard-header h1 { font-size:20px; color:#4fc3f7; letter-spacing:2px; }
.dashboard-header .clock { font-size:14px; color:#81d4fa; }
.dashboard-header .back-btn {
    background:#1e3a5f; color:#4fc3f7; border:1px solid #4fc3f7;
    padding:6px 16px; border-radius:4px; cursor:pointer; font-size:13px; text-decoration:none;
}
.dashboard-header .back-btn:hover { background:#2a4a6f; }

.grid-container {
    display:grid; gap:8px; padding:8px; height:calc(100vh - 56px);
}
.grid-1 { grid-template-columns:1fr; }
.grid-2 { grid-template-columns:1fr 1fr; }
.grid-3 { grid-template-columns:1fr 1fr 1fr; }
.grid-4 { grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr; }
.grid-6 { grid-template-columns:1fr 1fr 1fr; grid-template-rows:1fr 1fr; }
.grid-9 { grid-template-columns:1fr 1fr 1fr; grid-template-rows:1fr 1fr 1fr; }

.cam-cell {
    background:#111d2e; border:2px solid #1e3a5f; border-radius:8px;
    display:flex; flex-direction:column; overflow:hidden; cursor:pointer;
    transition:border-color 0.3s, transform 0.2s; position:relative;
}
.cam-cell:hover { border-color:#4fc3f7; transform:scale(1.01); }
.cam-cell.expanded {
    position:fixed; top:0; left:0; width:100vw; height:100vh;
    z-index:1000; border-radius:0; border-color:#4fc3f7;
}

.cam-cell-header {
    padding:6px 12px; background:rgba(0,0,0,0.4);
    display:flex; align-items:center; justify-content:space-between;
    font-size:13px; flex-shrink:0;
}
.cam-cell-header .cam-name { color:#81d4fa; font-weight:bold; }
.cam-cell-header .cam-level {
    padding:2px 10px; border-radius:10px; font-size:12px; font-weight:bold;
}
.level-normal { background:#1b5e20; color:#a5d6a7; }
.level-notice { background:#e65100; color:#ffcc80; }
.level-warning { background:#b71c1c; color:#ef9a9a; }
.level-serious { background:#880e4f; color:#f48fb1; }

.cam-cell-video { flex:1; display:flex; align-items:center; justify-content:center; background:#000; overflow:hidden; }
.cam-cell-video img { width:100%; height:100%; object-fit:contain; }

.cam-cell-footer {
    padding:4px 12px; background:rgba(0,0,0,0.4);
    display:flex; gap:16px; font-size:12px; flex-shrink:0;
}
.cam-cell-footer .metric { display:flex; gap:4px; }
.cam-cell-footer .metric .label { color:#607d8b; }
.cam-cell-footer .metric .val { color:#b0bec5; }

.cam-cell .close-expand {
    display:none; position:absolute; top:8px; right:12px;
    background:rgba(0,0,0,0.6); color:#fff; border:none; border-radius:50%;
    width:32px; height:32px; font-size:18px; cursor:pointer; z-index:10;
}
.cam-cell.expanded .close-expand { display:block; }

.cam-cell .diag-btn {
    display:none; position:absolute; bottom:50px; right:12px;
    background:linear-gradient(135deg,#1565c0,#0d47a1); color:#fff;
    border:none; border-radius:6px; padding:8px 16px; font-size:13px;
    cursor:pointer; z-index:10;
}
.cam-cell.expanded .diag-btn { display:block; }
.cam-cell .diag-btn:hover { background:linear-gradient(135deg,#1976d2,#1565c0); }

.no-cameras {
    display:flex; align-items:center; justify-content:center;
    height:100%; color:#607d8b; font-size:18px;
}

/* 诊断报告弹窗 */
.report-overlay {
    display:none; position:fixed; top:0; left:0; width:100vw; height:100vh;
    background:rgba(0,0,0,0.8); z-index:2000;
    align-items:center; justify-content:center;
}
.report-overlay.active { display:flex; }
.report-box {
    background:#1a2a3a; border:1px solid #4fc3f7; border-radius:12px;
    width:700px; max-height:80vh; overflow-y:auto; padding:24px;
}
.report-box h3 { color:#4fc3f7; margin-bottom:16px; }
.report-box .report-content { color:#cfd8dc; line-height:1.8; white-space:pre-wrap; font-size:14px; }
.report-box .close-btn {
    margin-top:16px; background:#1e3a5f; color:#4fc3f7; border:1px solid #4fc3f7;
    padding:8px 24px; border-radius:4px; cursor:pointer; float:right;
}
.report-loading { color:#81d4fa; font-size:14px; }
</style>
</head>
<body>

<div class="dashboard-header">
    <h1>AgriVision 智慧农业监控大屏</h1>
    <span class="clock" id="clock"></span>
    <a href="/" class="back-btn">返回单视图</a>
</div>

<div class="grid-container" id="gridContainer">
    <div class="no-cameras">正在加载摄像头...</div>
</div>

<!-- 诊断报告弹窗 -->
<div class="report-overlay" id="reportOverlay">
    <div class="report-box">
        <h3>AI 深度诊断报告</h3>
        <div class="report-content" id="reportContent">
            <span class="report-loading">正在连接AI诊断引擎...</span>
        </div>
        <button class="close-btn" onclick="closeReport()">关闭</button>
    </div>
</div>

<script>
let dashboardCameras = {};
let expandedCam = null;
let statusTimer = null;

// 时钟
function updateClock() {
    var now = new Date();
    document.getElementById('clock').textContent =
        now.getFullYear() + '-' +
        String(now.getMonth()+1).padStart(2,'0') + '-' +
        String(now.getDate()).padStart(2,'0') + ' ' +
        String(now.getHours()).padStart(2,'0') + ':' +
        String(now.getMinutes()).padStart(2,'0') + ':' +
        String(now.getSeconds()).padStart(2,'0');
}
setInterval(updateClock, 1000);
updateClock();

// 加载摄像头列表并构建网格
async function initDashboard() {
    try {
        var resp = await fetch('/api/cameras');
        var data = await resp.json();
        dashboardCameras = data.cameras;

        var enabledCams = Object.values(dashboardCameras).filter(c => c.enabled);
        var container = document.getElementById('gridContainer');

        if (enabledCams.length === 0) {
            container.innerHTML = '<div class="no-cameras">没有已启用的摄像头</div>';
            return;
        }

        // 根据数量选择网格布局
        var gridClass = 'grid-1';
        if (enabledCams.length === 2) gridClass = 'grid-2';
        else if (enabledCams.length === 3) gridClass = 'grid-3';
        else if (enabledCams.length === 4) gridClass = 'grid-4';
        else if (enabledCams.length <= 6) gridClass = 'grid-6';
        else gridClass = 'grid-9';
        container.className = 'grid-container ' + gridClass;

        container.innerHTML = '';
        enabledCams.forEach(function(cam) {
            var cell = document.createElement('div');
            cell.className = 'cam-cell';
            cell.id = 'cell-' + cam.id;
            cell.onclick = function(e) {
                if (e.target.tagName === 'BUTTON' || e.target.tagName === 'IMG') return;
                toggleExpand(cam.id);
            };
            cell.innerHTML =
                '<div class="cam-cell-header">' +
                    '<span class="cam-name">' + cam.name + '</span>' +
                    '<span class="cam-level level-normal" id="level-' + cam.id + '">等待中</span>' +
                '</div>' +
                '<div class="cam-cell-video">' +
                    '<img id="vid-' + cam.id + '" src="/video_feed/' + cam.id + '" alt="' + cam.name + '">' +
                '</div>' +
                '<div class="cam-cell-footer">' +
                    '<div class="metric"><span class="label">病斑:</span><span class="val" id="dis-' + cam.id + '">0</span></div>' +
                    '<div class="metric"><span class="label">虫害:</span><span class="val" id="bug-' + cam.id + '">0</span></div>' +
                    '<div class="metric"><span class="label">绿比:</span><span class="val" id="grn-' + cam.id + '">0%</span></div>' +
                '</div>' +
                '<button class="close-expand" onclick="event.stopPropagation();collapseAll()">&times;</button>' +
                '<button class="diag-btn" onclick="event.stopPropagation();deepDiagnose(\\'' + cam.id + '\\')">AI 深度诊断</button>';
            container.appendChild(cell);
        });

        // 开始轮询状态
        if (statusTimer) clearInterval(statusTimer);
        statusTimer = setInterval(updateAllStatus, 3000);
        updateAllStatus();
    } catch(e) {
        console.error('initDashboard error:', e);
        document.getElementById('gridContainer').innerHTML = '<div class="no-cameras">加载失败: ' + e.message + '</div>';
    }
}

// 更新所有摄像头状态
async function updateAllStatus() {
    for (var cid in dashboardCameras) {
        if (!dashboardCameras[cid].enabled) continue;
        try {
            var resp = await fetch('/api/dual_status?camera_id=' + encodeURIComponent(cid));
            var data = await resp.json();
            // 更新等级徽章
            var levelEl = document.getElementById('level-' + cid);
            if (levelEl) {
                levelEl.textContent = data.level;
                levelEl.className = 'cam-level ' + getLevelClass(data.level_code);
            }
            // 更新指标
            var disEl = document.getElementById('dis-' + cid);
            var bugEl = document.getElementById('bug-' + cid);
            var grnEl = document.getElementById('grn-' + cid);
            if (disEl) disEl.textContent = data.disease_count;
            if (bugEl) bugEl.textContent = data.white_count;
            if (grnEl) grnEl.textContent = (data.green_ratio * 100).toFixed(1) + '%';
        } catch(e) { /* ignore */ }
    }
}

function getLevelClass(code) {
    var map = {0:'level-normal', 1:'level-notice', 2:'level-warning', 3:'level-serious'};
    return map[code] || 'level-normal';
}

// 展开/折叠摄像头
function toggleExpand(camId) {
    var cell = document.getElementById('cell-' + camId);
    if (cell.classList.contains('expanded')) {
        cell.classList.remove('expanded');
        expandedCam = null;
    } else {
        collapseAll();
        cell.classList.add('expanded');
        expandedCam = camId;
    }
}

function collapseAll() {
    document.querySelectorAll('.cam-cell.expanded').forEach(function(el) {
        el.classList.remove('expanded');
    });
    expandedCam = null;
}

// ESC 键关闭展开
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        collapseAll();
        closeReport();
    }
});

// AI 深度诊断
async function deepDiagnose(camId) {
    var overlay = document.getElementById('reportOverlay');
    var content = document.getElementById('reportContent');
    overlay.classList.add('active');
    content.innerHTML = '<span class="report-loading">正在连接AI诊断引擎，请稍候（约2-3分钟）...</span>';

    try {
        var resp = await fetch('/api/deep_diagnose', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({camera_id: camId})
        });
        var data = await resp.json();

        if (data.success && data.report) {
            // 打字机效果展示报告
            content.innerHTML = '';
            var text = data.report.report || data.report.text || JSON.stringify(data.report, null, 2);
            typeWriter(content, text, 0);
        } else {
            content.innerHTML = '<span style="color:#ef5350">诊断失败: ' + (data.error || '未知错误') + '</span>';
        }
    } catch(e) {
        content.innerHTML = '<span style="color:#ef5350">请求失败: ' + e.message + '</span>';
    }
}

function typeWriter(el, text, idx) {
    if (idx < text.length) {
        el.textContent += text.charAt(idx);
        setTimeout(function() { typeWriter(el, text, idx + 1); }, 20);
    }
}

function closeReport() {
    document.getElementById('reportOverlay').classList.remove('active');
}

// 启动
initDashboard();
</script>
</body>
</html>"""


@app.route('/dashboard')
def dashboard():
    """大屏网格视图 - 同时显示所有摄像头的实时监控"""
    return DASHBOARD_PAGE


# ============================================================
# 程序入口 - 系统启动序列
# ============================================================
# 启动顺序说明:
#   1. 首先在主线程中执行一次检测（run_detection_once），
#      确保用户打开网页时已经有第一帧数据可显示（否则视频流会显示占位图）
#   2. 启动检测后台线程（detection_loop），以 daemon=True 守护线程运行，
#      每 3 秒执行一次检测，持续更新 latest_frame 和 latest_result
#   3. 启动 SD 同步后台线程（sd_sync_loop），以 daemon=True 守护线程运行，
#      每 300 秒自动从 ESP32 SD 卡同步新图片
#   4. 打印启动信息，提示用户在浏览器中访问的地址
#   5. 启动 Flask Web 服务器:
#      - host='0.0.0.0': 监听所有网络接口（允许局域网内其他设备访问）
#      - port=5000: Web 服务端口
#      - debug=False: 生产模式（禁用调试重载器，避免与后台线程冲突）
#      - threaded=True: 启用多线程处理，允许多个浏览器客户端同时访问
#        （这是关键设置: 视频流 /video_feed 会长时间占用一个连接，
#         必须允许其他请求如 /api/status 并行处理）
#
# 线程架构总览:
#   主线程:     Flask Web 服务器（处理所有 HTTP 请求）
#   守护线程 1: detection_loop  -> run_detection_once() [每3秒] (per camera)
#   守护线程 2: sd_sync_loop    -> sync_sd_card()       [每300秒] (per camera)
#   临时守护线程: save_to_sd_async（用户点击拍照时创建，完成后销毁）
#   临时守护线程: sync_sd_card（用户点击同步时创建，完成后销毁）
# ============================================================
if __name__ == '__main__':
    # 第一步: 为每个已启用的摄像头执行首次检测
    for cid, cam in cameras.items():
        if not cam.get("enabled", True):
            continue
        print(f"[启动] 摄像头 {cam['name']} ({cid}) 初始化...")
        run_detection_once(cid)
        
        # 第二步: 启动检测后台守护线程
        # daemon=True 表示守护线程，当主线程（Flask）退出时自动终止
        det_thread = threading.Thread(target=detection_loop, args=(cid,), daemon=True)
        det_thread.start()
        
        # 第三步: 启动 SD 卡同步后台守护线程
        sd_thread = threading.Thread(target=sd_sync_loop, args=(cid,), daemon=True)
        sd_thread.start()

    # 第四步: 打印启动信息
    print("=" * 50)
    print("  Flask 病虫害监控页面已启动")
    for cid, cam in cameras.items():
        if cam.get("enabled", True):
            print(f"  摄像头: {cam['name']} -> {cam['url']}")
    print("  请在浏览器打开: http://127.0.0.1:5000")
    print("=" * 50)

    # 第五步: 启动 Flask Web 服务器（阻塞主线程，持续监听请求）
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
