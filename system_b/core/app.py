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
import atexit   # 进程退出时释放可选 MQTT 客户端
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
from flask import Flask, g, jsonify, request  # Flask: Web 框架; jsonify: JSON 响应; request: 请求对象
import cv2              # OpenCV: 图像编解码、绘制文字、形态学操作等
import time             # 时间相关：计时、延时、时间戳格式化
import threading        # 多线程支持：后台检测循环、SD 同步循环、线程锁
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
from services.temporal_fusion import TemporalFusion

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
SYSTEM_A_API_TOKEN = os.environ.get("AGRIVISION_SYSTEM_A_API_TOKEN", os.environ.get("AGRIVISION_API_TOKEN", "")).strip()
if os.path.exists(CAMERAS_JSON):
    with open(CAMERAS_JSON, 'r', encoding='utf-8') as _f:
        SYSTEM_A_URL = json.load(_f).get("system_a_url", SYSTEM_A_URL)

# 时序融合参数：窗口内按时间衰减加权投票，并用连续候选帧确认等级切换。
TEMPORAL_WINDOW_SIZE = 5
TEMPORAL_MIN_CONSECUTIVE = 3

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
        "temporal_fusion": TemporalFusion(
            window_size=TEMPORAL_WINDOW_SIZE,
            min_consecutive=TEMPORAL_MIN_CONSECUTIVE,
        ),
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
from observability import log_event, resolve_request_id
from event_schema import build_detection_event, project_event
from offline_cache import OfflineEventCache
from event_transport import EventTransport, send_events_http
from mqtt_runtime import create_paho_transport
from mqtt_config_status import build_mqtt_config_status
from event_scheduler import EventSyncScheduler
from event_sync_status import EventSyncStatus
from process_sync_lock import ProcessSyncLock
from schemas.api import validate_alert_patch, validate_sync_request, validate_yolo_patch
from services.metrics import MetricsRegistry
from services.security import ApiSecurity
from services.storage import filter_camera_filenames
from workers.camera_tasks import CameraTaskManager
from routes.engineering import register_engineering_routes
from routes.monitoring import register_monitoring_routes
from routes.history import register_history_routes
from routes.storage import register_storage_routes
from routes.control import register_control_routes
from routes.events import register_event_routes
from routes.video import register_video_routes
from routes.diagnosis import register_diagnosis_routes
from routes.pages import register_page_routes
from page_templates import load_page

# 创建 Flask 应用实例
app = Flask(__name__)
logger = logging.getLogger("agrivision.system_b")
metrics = MetricsRegistry()
api_security = ApiSecurity.from_env()
camera_task_manager = CameraTaskManager(cameras.keys(), metrics=metrics, logger=logger)
register_engineering_routes(app, camera_task_manager, metrics)


@app.before_request
def start_request_observation():
    g.request_id = resolve_request_id(request.headers.get("X-Request-ID"))
    g.request_started = time.perf_counter()


@app.after_request
def finish_request_observation(response):
    request_id = getattr(g, "request_id", "")
    response.headers["X-Request-ID"] = request_id
    elapsed_ms = (time.perf_counter() - getattr(g, "request_started", time.perf_counter())) * 1000
    # 未匹配路由使用固定标签，避免把任意请求路径写成高基数指标标签。
    route = request.url_rule.rule if request.url_rule is not None else "unmatched"
    metrics.observe_http(request.method, route, response.status_code, elapsed_ms / 1000.0)
    log_event(
        logger,
        logging.INFO,
        "request_completed",
        method=request.method,
        route=route,
        status=response.status_code,
        duration_ms=round(elapsed_ms, 1),
        request_id=request_id,
    )
    return response


@app.before_request
def enforce_api_security():
    decision = api_security.authorize(
        request.path,
        request.remote_addr,
        request.headers,
        request.cookies,
    )
    if decision.status == "allow":
        return None
    if decision.status == "rate_limited":
        response = jsonify({"error": "rate limit exceeded"})
        response.headers["Retry-After"] = str(decision.retry_after)
        return response, 429
    if decision.reason == "invalid_api_token":
        return jsonify({"error": "authentication required"}), 401
    return jsonify({"error": "API authentication is not configured"}), 503

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
offline_event_sync_lock = ProcessSyncLock(os.path.join(OFFLINE_EVENTS_DIR, ".sync-lock.db"))
EVENTS_SINK_URL = os.environ.get("AGRIVISION_EVENTS_SINK_URL", "").strip()
MQTT_BROKER_URL = os.environ.get("AGRIVISION_MQTT_BROKER_URL", "").strip()
MQTT_TOPIC = os.environ.get("AGRIVISION_MQTT_TOPIC", "").strip()
MQTT_CLIENT_ID = os.environ.get("AGRIVISION_MQTT_CLIENT_ID", "agrivision-system-b").strip()
MQTT_USERNAME = os.environ.get("AGRIVISION_MQTT_USERNAME")
MQTT_PASSWORD = os.environ.get("AGRIVISION_MQTT_PASSWORD")
MQTT_CA_CERTS = os.environ.get("AGRIVISION_MQTT_CA_CERTS")


def get_mqtt_settings():
    """Return current MQTT settings to route adapters without exposing them."""
    return {
        "broker_url": MQTT_BROKER_URL,
        "topic": MQTT_TOPIC,
        "client_id": MQTT_CLIENT_ID,
        "username": MQTT_USERNAME,
        "password": MQTT_PASSWORD,
        "ca_certs": MQTT_CA_CERTS,
        "connack_timeout": MQTT_CONNACK_TIMEOUT,
        "publish_timeout": MQTT_PUBLISH_TIMEOUT,
    }


def _read_nonnegative_float(name, default):
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("invalid numeric environment variable name=%s", name)
        return default
    return value if value >= 0 else default


def _read_bounded_float(name, default, lower, upper):
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("invalid numeric environment variable name=%s", name)
        return default
    return value if lower <= value <= upper else default


def _read_bounded_int(name, default, lower, upper):
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("invalid integer environment variable name=%s", name)
        return default
    return value if lower <= value <= upper else default


EVENTS_SYNC_INTERVAL = _read_nonnegative_float("AGRIVISION_EVENTS_SYNC_INTERVAL", 0)
EVENTS_SYNC_LIMIT = _read_bounded_int("AGRIVISION_EVENTS_SYNC_LIMIT", 50, 1, 100)
MQTT_CONNACK_TIMEOUT = _read_bounded_float("AGRIVISION_MQTT_CONNACK_TIMEOUT", 5.0, 0.1, 30.0)
MQTT_PUBLISH_TIMEOUT = _read_bounded_float("AGRIVISION_MQTT_PUBLISH_TIMEOUT", 10.0, 0.1, 60.0)


event_transport = None
if EVENTS_SINK_URL:
    try:
        event_transport = EventTransport(EVENTS_SINK_URL, send_events_http)
    except ValueError:
        log_event(logger, logging.WARNING, "event_sink_config_invalid")

mqtt_transport = None
mqtt_close = None
event_sync_scheduler = None
event_sync_status = EventSyncStatus()


def get_mqtt_transport():
    """Create the MQTT client only after an explicit sync request."""
    global mqtt_transport, mqtt_close
    if mqtt_transport is not None:
        event_sync_status.record_mqtt_runtime("connected")
        return mqtt_transport
    if not MQTT_BROKER_URL or not MQTT_TOPIC or not MQTT_CLIENT_ID:
        event_sync_status.record_mqtt_runtime("not_configured")
        raise ValueError("mqtt runtime is not configured")
    event_sync_status.record_mqtt_runtime("connecting")
    try:
        mqtt_transport, mqtt_close = create_paho_transport(
            MQTT_BROKER_URL,
            MQTT_TOPIC,
            MQTT_CLIENT_ID,
            username=MQTT_USERNAME,
            password=MQTT_PASSWORD,
            ca_certs=MQTT_CA_CERTS,
            connack_timeout=MQTT_CONNACK_TIMEOUT,
            publish_timeout=MQTT_PUBLISH_TIMEOUT,
        )
    except Exception:
        event_sync_status.record_mqtt_runtime("connection_failed", "runtime")
        raise
    event_sync_status.record_mqtt_runtime("connected")
    return mqtt_transport


@atexit.register
def close_mqtt_runtime():
    if mqtt_close is not None:
        mqtt_close()


def sync_offline_events_once():
    """Sync one bounded batch for the opt-in background scheduler."""
    if not offline_event_sync_lock.acquire(blocking=False):
        event_sync_status.record("skipped", "none", len(offline_event_cache.list_pending()))
        return {"sent": 0, "skipped": "sync already running"}
    try:
        transport_name = "mqtt" if MQTT_BROKER_URL else "http" if event_transport is not None else "none"
        events = offline_event_cache.list_pending()[:EVENTS_SYNC_LIMIT]
        if not events:
            event_sync_status.record("empty", transport_name, 0)
            return {"sent": 0, "acked": 0, "pending": 0}
        transport = get_mqtt_transport() if MQTT_BROKER_URL else event_transport
        if transport is None:
            event_sync_status.record("skipped", "none", len(events))
            return {"sent": 0, "acked": 0, "pending": len(events)}
        result = transport.sync(events, offline_event_cache.ack)
        pending = len(offline_event_cache.list_pending())
        outcome = "success" if result["sent"] == len(events) else "failure"
        event_sync_status.record(
            outcome,
            transport_name,
            pending,
            sent=result.get("sent", 0),
            acked=result.get("acked", 0),
            failure_type=result.get("failure_type", "") if outcome == "failure" else "",
        )
        if transport_name == "mqtt":
            event_sync_status.record_mqtt_runtime(
                "connected" if outcome == "success" else "publish_failed",
                result.get("failure_type", "retry_exhausted") if outcome == "failure" else "",
            )
        return {**result, "pending": pending}
    except Exception:
        log_event(logger, logging.ERROR, "automatic_offline_event_sync_failed", stage="runtime")
        event_sync_status.record(
            "failure",
            transport_name,
            len(offline_event_cache.list_pending()),
            failure_type="runtime",
        )
        return {"sent": 0, "acked": 0, "error": "sync failed"}
    finally:
        offline_event_sync_lock.release()


def start_event_sync_scheduler():
    """Start automatic event sync only when an explicit interval is configured."""
    global event_sync_scheduler
    if EVENTS_SYNC_INTERVAL <= 0 or (not MQTT_BROKER_URL and event_transport is None):
        return False
    event_sync_scheduler = EventSyncScheduler(
        sync_offline_events_once,
        EVENTS_SYNC_INTERVAL,
        logger=logger,
    )
    return event_sync_scheduler.start()


@atexit.register
def stop_event_sync_scheduler():
    if event_sync_scheduler is not None:
        event_sync_scheduler.stop(timeout=5)


@atexit.register
def stop_camera_task_workers():
    camera_task_manager.stop_all(timeout=5)

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

# 每个摄像头拥有独立的同步/拍照状态。保留上面的两个对象作为旧调用方
# 的兼容默认状态，但运行时状态不再由所有摄像头共享。
camera_sd_sync_info = {
    camera_id: dict(sd_sync_info) for camera_id in cameras
}
camera_save_sd_info = {
    camera_id: dict(save_sd_info) for camera_id in cameras
}


def _get_sd_sync_info(camera_id=None):
    camera_id = camera_id or _get_default_camera_id()
    return camera_sd_sync_info.get(camera_id, sd_sync_info)


def _get_save_sd_info(camera_id=None):
    camera_id = camera_id or _get_default_camera_id()
    return camera_save_sd_info.get(camera_id, save_sd_info)

# ============================================================
# 全局变量 - 在检测线程和 Web 请求线程之间共享的运行时状态
# 注意: 这些变量会被多个线程同时访问，修改时必须持有对应的锁
# ============================================================

# 兼容旧状态字段；实际融合由每摄像头的 TemporalFusion 管理。
DETECTION_HISTORY_SIZE = TEMPORAL_WINDOW_SIZE

# 等级变化防抖阈值: 需要连续 LEVEL_CHANGE_THRESHOLD 帧的平均等级都不同于当前等级
# 才会触发等级切换，有效防止因偶尔一帧误检导致的等级跳变
LEVEL_CHANGE_THRESHOLD = TEMPORAL_MIN_CONSECUTIVE

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


def set_yolo_enabled(enabled):
    """Update the YOLO switch through an explicit dependency callback."""
    global yolo_enabled
    yolo_enabled = bool(enabled)

register_monitoring_routes(
    app,
    cameras=cameras,
    get_default_camera_id=_get_default_camera_id,
    get_sd_sync_info=_get_sd_sync_info,
    sd_lock=sd_lock,
    yolo_detector=yolo_detector,
    get_yolo_enabled=lambda: yolo_enabled,
    summarize_camera=summarize_camera,
    build_service_health=build_service_health,
    comparison_history_size=COMPARISON_HISTORY_SIZE,
)
register_history_routes(app, history_manager)
register_storage_routes(
    app,
    data_set_dir=DATASET_DIR,
    cameras=cameras,
    get_default_camera_id=_get_default_camera_id,
    task_manager=camera_task_manager,
    save_to_sd=lambda camera_id: save_to_sd_async(camera_id),
    sync_sd=lambda camera_id: sync_sd_card(camera_id),
    save_state_lock=save_sd_lock,
    get_save_state=_get_save_sd_info,
    sync_state_lock=sd_lock,
    get_sync_state=_get_sd_sync_info,
    filter_camera_filenames=filter_camera_filenames,
)


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
            log_event(logger, logging.WARNING, "camera_frame_unavailable", camera_id=camera_id)
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
        color_started = time.perf_counter()
        result, annotated, masks = detect_and_annotate(img, config)
        metrics.observe_inference("color", (time.perf_counter() - color_started) * 1000)

        # ============================================================
        # 第三步B: YOLO 引擎检测（新增）
        # ============================================================
        yolo_detections = []
        if yolo_enabled and yolo_detector.is_loaded():
            yolo_detections = yolo_detector.detect(img)
            metrics.observe_inference("yolo", yolo_detector._last_inference_ms)
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
        # 第四步: 可解释时序融合
        # ============================================================
        # 每个摄像头拥有独立的融合器：最近帧按时间衰减加权投票，
        # 连续候选帧达到阈值后才切换稳定等级；同时保留支持度和待切换证据。
        with frame_lock:
            temporal_summary = state["temporal_fusion"].update(result)
        local_history = temporal_summary["recent_frames"]
        averages = temporal_summary["averages"]
        avg_disease_count = averages["disease_count"]
        avg_white_count = averages["white_count"]
        avg_disease_ratio = averages["disease_ratio"]
        avg_white_ratio = averages["white_ratio"]
        avg_green_ratio = averages["green_ratio"]
        local_stable_code = temporal_summary["stable_level_code"]
        local_stable_level = temporal_summary["stable_level"]
        local_counter = temporal_summary["pending_count"]
        if temporal_summary["level_changed"]:
            print(f"  [等级变化] {local_stable_level}")
        comparison_entry.update({
            'temporal_candidate_level': temporal_summary['candidate_level'],
            'temporal_stable_level': temporal_summary['stable_level'],
            'temporal_support': temporal_summary['weighted_support'],
            'temporal_pending_count': temporal_summary['pending_count'],
        })

        # 构建最终的稳定结果字典
        stable_result = {
            "level": local_stable_level,
            "level_code": local_stable_code,
            "disease_count": round(avg_disease_count),
            "white_count": round(avg_white_count),
            "disease_ratio": avg_disease_ratio,
            "white_ratio": avg_white_ratio,
            "green_ratio": avg_green_ratio,
            "temporal": temporal_summary,
        }

        # 将检测摘要写入有界离线队列，供网络恢复后的传输器消费。
        # 队列只保存事件摘要，不保存图像、URL 或通知配置。
        try:
            offline_event_cache.put(build_detection_event(camera_id, stable_result))
        except (TypeError, ValueError, OSError):
            log_event(logger, logging.WARNING, "offline_event_enqueue_failed", camera_id=camera_id)

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
            metrics.inc_alert("success" if alert_result.get("sent") else "failure")
            if alert_result['sent']:
                print(f"  -> [告警] 已推送钉钉通知: {local_stable_level}")

        state["last_error"] = ""
        return True
    except Exception:
        metrics.inc("agrivision_detection_errors_total", {"stage": "runtime"})
        # 异常处理: 捕获所有未预期的错误，避免后台线程崩溃
        state["last_error"] = "检测暂不可用"
        log_event(logger, logging.ERROR, "detection_failed", camera_id=camera_id, stage="runtime")
        print(f"  -> {state['last_error']}")
        return False


# ============================================================
# SD 卡同步函数 - 从 ESP32-CAM 的 SD 卡下载新图片到本地
# ============================================================
# 该函数可被以下两种方式触发:
#   1. CameraTaskManager 的每摄像头周期 worker 每隔 SD_SYNC_INTERVAL (300秒) 投递
#   2. 用户在前端点击"立即同步"按钮，通过 /api/sync_now 投递到对应摄像头队列
# 同步流程:
#   1. 请求 ESP32 的 /list 接口获取 SD 卡上的文件列表
#   2. 与本地已有文件做去重比对，找出新文件
#   3. 逐个下载新文件到本地 dataset/images/ 目录
#   4. 更新同步状态信息（供前端轮询显示进度）
def sync_sd_card(camera_id=None):
    # 获取摄像头配置
    if camera_id is None:
        camera_id = _get_default_camera_id()
    cam = cameras.get(camera_id)
    if not cam:
        print(f"[SD同步] 摄像头 {camera_id} 不存在")
        return

    # 绑定到当前摄像头的状态，避免多摄像头同步互相覆盖进度。
    sd_sync_info = _get_sd_sync_info(camera_id)
    
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
        sd_files = filter_camera_filenames(resp.json())
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
            except Exception:
                # 单个文件下载失败不中断整个同步流程，记录错误后继续下一个
                log_event(logger, logging.WARNING, "sd_file_download_failed", camera_id=camera_id)

        # --- 第四步: 更新最终同步状态 ---
        with sd_lock:
            sd_sync_info["total_synced"] = len(os.listdir(DATASET_DIR))
            sd_sync_info["last_sync_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            sd_sync_info["sync_progress"] = f"完成! 新下载 {new_count} 张, 本地共 {sd_sync_info['total_synced']} 张"
            progress_msg = sd_sync_info['sync_progress']
        print(f"[SD同步] {progress_msg}")

    except Exception:
        # 整体异常处理（如 ESP32 连接失败等网络错误）
        with sd_lock:
            sd_sync_info["sync_error"] = "同步失败，请检查摄像头连接"
            sd_sync_info["sync_progress"] = "同步失败"
            err_msg = sd_sync_info['sync_error']
        log_event(logger, logging.ERROR, "sd_sync_failed", camera_id=camera_id)
        print(f"[SD同步] {err_msg}")
    finally:
        # 无论成功或失败，都必须在 finally 中重置 syncing 标志
        # 否则后续同步任务将永远被跳过
        with sd_lock:
            sd_sync_info["syncing"] = False


# ============================================================
# 远程拍照存卡函数 - 触发 ESP32-CAM 拍摄一张照片并保存到 SD 卡
# ============================================================
# 该函数由 /api/save_to_sd 路由投递到对应摄像头的有界 worker（异步执行）。
# 工作流程: 向 ESP32 发送 GET /save 请求 -> ESP32 拍照并写入 SD 卡 -> 返回文件名
# 使用异步方式是因为 ESP32 拍照+写卡可能耗时数秒，不能阻塞 Web 请求线程。
# 前端通过轮询 /api/save_status 接口获取执行进度。
def save_to_sd_async(camera_id=None):
    # 获取摄像头配置
    if camera_id is None:
        camera_id = _get_default_camera_id()
    save_sd_info = _get_save_sd_info(camera_id)
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
    except Exception:
        with save_sd_lock:
            save_sd_info["error"] = "拍照失败，请检查摄像头连接"
        log_event(logger, logging.ERROR, "save_to_sd_failed", camera_id=camera_id)
    finally:
        with save_sd_lock:
            save_sd_info["saving"] = False
            save_sd_info["done"] = True


# ============================================================
# 周期任务兼容入口 - 委托给 CameraTaskManager
# ============================================================
# 保留函数名供旧调用方使用，但不再在这里创建不可停止的 while True 循环。
def sd_sync_loop(camera_id=None):
    camera_id = camera_id or _get_default_camera_id()
    return camera_task_manager.start_periodic(
        camera_id,
        "sd_sync",
        lambda: sync_sd_card(camera_id),
        SD_SYNC_INTERVAL,
    )


# 检测周期任务: 每隔 3 秒向对应队列投递一次检测
# 3 秒间隔的选取考量:
#   - ESP32-CAM 拍摄+传输一张 SVGA (800x600) 图片约需 0.5-1 秒
#   - 检测算法处理一帧约需 0.3-0.5 秒
#   - 剩余时间作为余量，确保不会因网络波动导致帧积压
def detection_loop(camera_id=None):
    camera_id = camera_id or _get_default_camera_id()
    return camera_task_manager.start_periodic(
        camera_id,
        "detection",
        lambda: run_detection_once(camera_id),
        3,
    )


# ============================================================
# 网页 HTML - 单页面应用（SPA）模板说明
# ============================================================
# Web 界面保留为无需构建工具的原生 HTML/CSS/JavaScript，模板文件位于
# templates/，由 page_templates.py 以 UTF-8 加载；页面路由在 routes/pages.py 注册。
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
# ========== 网页模板 ==========
HTML_PAGE = load_page("main.html")


# ============================================================
# Flask 路由 - Web API 接口定义
# ============================================================
# 路由分为以下几类:
#   1. 页面路由:     /                          -> 返回 templates/main.html
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
# HTML_PAGE 由 page_templates.py 从 templates/main.html 加载
# 页面路由由 routes/pages.py 统一注册。


# /api/yolo/config: 更新 YOLO 检测参数
# YOLO/告警/参数控制路由由 routes/control.py 统一注册。


# ---------- 3C. 告警通知接口 ----------
# GET /api/alert/config:  获取告警通知配置
# POST /api/alert/config: 更新告警通知配置（webhook_url, enabled, cooldown）
# GET /api/alert/history: 获取告警历史记录
# 告警路由由 routes/control.py 统一注册。


# 告警历史和测试发送路由由 routes/control.py 统一注册。


# ---------- 4. 参数接口 ----------
# GET:  获取当前检测参数配置（参数字典）
# POST: 保存新的检测参数（前端传入完整的参数字典覆盖保存）
# 使用 config_lock 确保参数读写与检测线程的参数读取互斥
# 参数路由由 routes/control.py 统一注册。


# 获取参数元信息: 每个参数的显示名称、最小值、最大值、步长、单位
# 前端根据这些信息动态生成滑块控件
# 参数元信息路由由 routes/control.py 统一注册。


# 重置参数为默认值: 将配置恢复为 DetectionConfig 的初始值
# 返回重置后的参数，前端据此刷新滑块显示
# 参数重置路由由 routes/control.py 统一注册。


# ---------- 5. 预览接口 ----------
# 根据前端传来的参数和 mask 类型，生成对应的分割预览图
# 前端参数调节页面有四个预览窗口: leaf(叶片分割)、disease(病斑)、pest(虫害)、annotated(标注)
# 用户拖动滑块时，前端以 200ms 防抖频率调用此接口，实时查看参数效果
# mask 预览路由由 routes/control.py 统一注册。


# 离线事件路由由 routes/events.py 统一注册。


# HTTP 事件同步路由由 routes/events.py 统一注册。


# MQTT 事件同步路由由 routes/events.py 统一注册。


# 事件同步状态路由由 routes/events.py 统一注册。


# ---------- 8. MJPEG 视频流接口 ----------
# 该接口实现 MJPEG (Motion JPEG) 视频流推送
# 原理: 使用 HTTP 分块传输编码 (chunked transfer) + multipart/x-mixed-replace MIME 类型
#       浏览器会将每个 chunk 中的 JPEG 帧依次替换显示在 <img> 标签中，形成"视频"效果
# 帧率: 每 0.5 秒输出一帧（约 2 FPS），因为后台检测每 3 秒才更新一次最新帧，
#       中间帧是重复的同一张图片，但保持 HTTP 连接不中断
# 视频流路由由 routes/video.py 统一注册。


# ============================================================
# Phase 11: A-B 浅连接 - System B 代理调用 System A 深度诊断
# ============================================================

# System A 代理路由由 routes/diagnosis.py 统一注册。


# ============================================================
# Phase 10: 大屏网格视图 - 多摄像头同时监控
# ============================================================

DASHBOARD_PAGE = load_page("dashboard.html")


# 大屏页面路由由 routes/pages.py 统一注册。


# ---------- 模块化路由组合 ----------
# 所有 HTTP 边界现在通过显式依赖注册，避免控制面继续依赖 app.py 的全局变量。
register_control_routes(
    app,
    cameras=cameras,
    get_default_camera_id=_get_default_camera_id,
    config_manager=config_manager,
    config_lock=config_lock,
    yolo_detector=yolo_detector,
    get_yolo_enabled=lambda: yolo_enabled,
    set_yolo_enabled=set_yolo_enabled,
    dual_verifier=dual_verifier,
    alert_notifier=alert_notifier,
    validate_yolo_patch=validate_yolo_patch,
    validate_alert_patch=validate_alert_patch,
    issue_session=lambda remote_addr, headers: api_security.issue_session(remote_addr, headers),
    session_cookie_name=api_security.SESSION_COOKIE_NAME,
    session_ttl_seconds=api_security.SESSION_TTL_SECONDS,
    generate_mask_image=generate_mask_image,
    log_event=log_event,
    logger=logger,
)
register_event_routes(
    app,
    offline_event_cache=offline_event_cache,
    project_event=project_event,
    event_transport=event_transport,
    offline_event_sync_lock=offline_event_sync_lock,
    validate_sync_request=validate_sync_request,
    get_mqtt_transport=get_mqtt_transport,
    build_mqtt_config_status=build_mqtt_config_status,
    get_mqtt_settings=get_mqtt_settings,
    event_sync_status=event_sync_status,
    get_scheduler=lambda: event_sync_scheduler,
    events_sync_interval=EVENTS_SYNC_INTERVAL,
)
register_video_routes(
    app,
    cameras=cameras,
    get_default_camera_id=_get_default_camera_id,
    create_wait_image=create_wait_image,
)
register_diagnosis_routes(
    app,
    cameras=cameras,
    get_default_camera_id=_get_default_camera_id,
    system_a_url=SYSTEM_A_URL,
    system_a_api_token=SYSTEM_A_API_TOKEN,
    cv2_module=cv2,
    requests_module=requests,
    log_event=log_event,
    logger=logger,
)
register_page_routes(app, HTML_PAGE, DASHBOARD_PAGE)


# ============================================================
# 程序入口 - 系统启动序列
# ============================================================
# 启动顺序说明:
#   1. 首先在主线程中执行一次检测（run_detection_once），
#      确保用户打开网页时已经有第一帧数据可显示（否则视频流会显示占位图）
#   2. 为每个摄像头注册 detection 周期任务，投递到可停止的有界 worker，
#      每 3 秒更新 latest_frame 和 latest_result
#   3. 为每个摄像头注册 sd_sync 周期任务，投递到可停止的有界 worker，
#      每 300 秒从 ESP32 SD 卡同步新图片
#   4. 打印启动信息，提示用户在浏览器中访问的地址
#   5. 启动 Flask Web 服务器:
#      - host: 默认监听回环地址；远程绑定必须配置 AGRIVISION_API_TOKEN
#      - port=5000: Web 服务端口
#      - debug=False: 生产模式（禁用调试重载器，避免与后台线程冲突）
#      - threaded=True: 启用多线程处理，允许多个浏览器客户端同时访问
#        （这是关键设置: 视频流 /video_feed 会长时间占用一个连接，
#         必须允许其他请求如 /api/status 并行处理）
#
# 线程架构总览:
#   主线程:     Flask Web 服务器（处理所有 HTTP 请求）
#   CameraTaskManager: per camera / task 有界队列、worker、周期调度和停止信号
#   detection: run_detection_once() [每3秒] (per camera)
#   sd_sync: sync_sd_card() [每300秒] (per camera)
#   save_to_sd: save_to_sd_async()（按需投递，重复任务拒绝）
# ============================================================
if __name__ == '__main__':
    # 在访问摄像头或创建后台 worker 前完成网络暴露边界预检。
    bind_host = os.environ.get("AGRIVISION_BIND_HOST", "127.0.0.1").strip() or "127.0.0.1"
    if bind_host not in {"127.0.0.1", "::1", "localhost"} and not api_security.api_token:
        raise SystemExit("AGRIVISION_API_TOKEN is required for non-loopback binding")

    # 第一步: 为每个已启用的摄像头执行首次检测
    for cid, cam in cameras.items():
        if not cam.get("enabled", True):
            continue
        print(f"[启动] 摄像头 {cam['name']} ({cid}) 初始化...")
        run_detection_once(cid)
        
        # 第二、三步: 通过每摄像头独立队列启动可停止的周期任务
        camera_task_manager.start_periodic(
            cid, "detection", lambda cid=cid: run_detection_once(cid), 3
        )
        camera_task_manager.start_periodic(
            cid, "sd_sync", lambda cid=cid: sync_sd_card(cid), SD_SYNC_INTERVAL
        )

    if start_event_sync_scheduler():
        print(f"  离线事件自动同步已启用: 每 {EVENTS_SYNC_INTERVAL:g} 秒")

    # 第四步: 打印启动信息
    print("=" * 50)
    print("  Flask 病虫害监控页面已启动")
    for cid, cam in cameras.items():
        if cam.get("enabled", True):
            print(f"  摄像头: {cam['name']} -> {cam['url']}")
    print("  请在浏览器打开: http://127.0.0.1:5000")
    print("=" * 50)

    # 第五步: 启动 Flask Web 服务器（阻塞主线程，持续监听请求）
    app.run(host=bind_host, port=5000, debug=False, threaded=True)
