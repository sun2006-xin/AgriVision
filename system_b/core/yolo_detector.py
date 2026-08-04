"""
YOLO 病虫害检测模块
==================
封装 YOLOv8 模型加载、推理、标注的完整流程。

职责：
  1. 加载训练好的 YOLOv8 权重文件
  2. 对输入图像执行目标检测，返回病害/虫害的边界框和置信度
  3. 在图像上绘制检测框和标签（虚线框，与颜色引擎的实线框区分）
  4. 提供模型状态查询和推理统计

类别映射（与 data.yaml 保持一致）：
  0 -> disease（病害）
  1 -> bug（虫害）

使用方式：
    detector = YOLODetector('models/best.pt')
    if detector.is_loaded():
        detections = detector.detect(image)
        annotated = detector.annotate(image, detections)
"""

import os
import time
import threading
import cv2
import numpy as np

# ---------- 类别映射 ----------
# 与 data.yaml 中的 names 保持一致（训练时的数据集配置）
CLASS_NAMES = {
    0: 'disease',   # 病害
    1: 'bug',       # 虫害
}

# ---------- 标注颜色配置 ----------
# YOLO 引擎使用虚线框，与颜色引擎的实线框形成视觉区分
COLORS = {
    'disease': (0, 0, 255),    # BGR: 红色 - 病害
    'bug': (255, 100, 0),      # BGR: 蓝色 - 虫害
}

# 标签背景颜色（半透明）
LABEL_BG_COLORS = {
    'disease': (0, 0, 180),
    'bug': (180, 80, 0),
}


class YOLODetector:
    """
    YOLOv8 病虫害检测器封装

    封装 ultralytics 的 YOLO 模型，提供简洁的 detect/annotate 接口，
    隐藏模型加载、设备选择、后处理等细节。
    """

    def __init__(self, model_path='../models/best.pt', conf_threshold=0.25,
                 iou_threshold=0.45, device='auto'):
        """
        初始化 YOLO 检测器。

        参数:
            model_path: 模型文件路径（相对于项目根目录或绝对路径）
            conf_threshold: 置信度阈值，低于此值的检测框被过滤
            iou_threshold: NMS IoU 阈值，控制重叠框的合并严格程度
            device: 推理设备
                - 'auto': 自动选择（优先 GPU，无 GPU 则 CPU）
                - 'cpu': 强制 CPU
                - 'cuda' 或 '0': 使用第一块 GPU
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # 模型实例（ultralytics YOLO 对象）
        self.model = None

        # 推理统计
        self._total_inferences = 0
        self._total_inference_ms = 0.0
        self._last_inference_ms = 0.0
        self._total_detections = 0
        self._disease_detections = 0
        self._bug_detections = 0

        # 推理锁: 多摄像头并行检测时保护模型推理和统计更新的线程安全
        self._inference_lock = threading.Lock()

        # 尝试加载模型
        self._load_model()

    def _load_model(self):
        """
        加载 YOLO 模型权重文件。

        失败时不抛异常，而是将 self.model 设为 None，
        通过 is_loaded() 检查状态，允许系统降级为单引擎运行。
        """
        try:
            # 解析模型路径（支持相对路径）
            if not os.path.isabs(self.model_path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                self.model_path = os.path.join(base_dir, self.model_path)

            if not os.path.exists(self.model_path):
                print(f"[YOLO] 模型文件不存在: {self.model_path}")
                print("[YOLO] 系统将降级为单引擎（仅颜色检测）运行")
                return

            # 解析设备
            actual_device = self._resolve_device()

            print(f"[YOLO] 正在加载模型: {os.path.basename(self.model_path)}")
            print(f"[YOLO] 设备: {actual_device}")

            # 导入 ultralytics 并加载模型
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)

            # 验证模型加载成功（尝试一次空推理）
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, imgsz=640, device=actual_device)

            # 获取模型信息
            model_info = self._get_model_info()
            print(f"[YOLO] 模型加载成功! {model_info}")
            print(f"[YOLO] 类别: {CLASS_NAMES}")

        except ImportError:
            print("[YOLO] 错误: ultralytics 未安装")
            print("[YOLO] 请运行: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"[YOLO] 模型加载失败: {e}")
            self.model = None

    def _resolve_device(self):
        """
        解析设备配置为 ultralytics 可识别的设备字符串。

        返回:
            'cpu' 或 '0'（GPU 编号）
        """
        if self.device == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    return '0'
            except ImportError:
                pass
            return 'cpu'
        elif self.device == 'cuda':
            return '0'
        else:
            return self.device

    def _get_model_info(self):
        """获取模型的结构信息摘要"""
        if self.model is None:
            return "未加载"
        try:
            total_params = sum(p.numel() for p in self.model.model.parameters())
            param_str = f"{total_params / 1e6:.1f}M 参数"
            file_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            size_str = f"{file_size_mb:.1f}MB"
            return f"{param_str}, 文件 {size_str}"
        except Exception:
            return "信息获取失败"

    def detect(self, image):
        """
        对单帧图像执行 YOLO 检测。

        参数:
            image: BGR 格式的 numpy 数组（OpenCV 标准格式），任意分辨率

        返回:
            检测结果列表，每个元素为字典:
            {
                'class_id': int,           # 类别ID: 0=disease, 1=bug
                'class_name': str,         # 类别名: 'disease' 或 'bug'
                'confidence': float,       # 置信度 [0, 1]
                'bbox': [x1, y1, x2, y2],  # 像素坐标边界框（整数）
            }
            如果模型未加载或推理失败，返回空列表。
        """
        if self.model is None:
            return []

        with self._inference_lock:
          try:
            device = self._resolve_device()

            # 执行推理
            t0 = time.time()
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=device,
                imgsz=640,
            )
            inference_ms = (time.time() - t0) * 1000

            # 更新统计
            self._total_inferences += 1
            self._total_inference_ms += inference_ms
            self._last_inference_ms = inference_ms

            # 解析结果
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy.tolist()
                        cls_name = CLASS_NAMES.get(cls_id, f'class_{cls_id}')

                        detection = {
                            'class_id': cls_id,
                            'class_name': cls_name,
                            'confidence': round(conf, 4),
                            'bbox': [x1, y1, x2, y2],
                        }
                        detections.append(detection)

                        # 更新分类统计
                        self._total_detections += 1
                        if cls_id == 0:
                            self._disease_detections += 1
                        elif cls_id == 1:
                            self._bug_detections += 1

            return detections

          except Exception as e:
            print(f"[YOLO] 推理异常: {e}")
            return []

    def annotate(self, image, detections):
        """
        在图像上绘制 YOLO 检测框和标签。

        使用虚线矩形框，与颜色引擎的实线框形成视觉区分。
        每个框右上角显示类别名和置信度。

        参数:
            image: BGR 格式的 numpy 数组
            detections: detect() 返回的检测结果列表

        返回:
            标注后的图像（新的 numpy 数组，不修改原图）
        """
        if not detections:
            return image.copy()

        annotated = image.copy()
        h, w = annotated.shape[:2]

        # 根据图像分辨率自适应线宽和字体大小
        line_thickness = max(1, min(h, w) // 400)
        font_scale = max(0.4, min(h, w) / 1200)

        for det in detections:
            cls_name = det['class_name']
            conf = det['confidence']
            x1, y1, x2, y2 = det['bbox']

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            color = COLORS.get(cls_name, (200, 200, 200))
            bg_color = LABEL_BG_COLORS.get(cls_name, (150, 150, 150))

            # 绘制虚线矩形框
            self._draw_dashed_rect(annotated, (x1, y1), (x2, y2),
                                   color, line_thickness, dash_length=8)

            # 绘制标签背景和文字
            label = f"{cls_name} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            label_y1 = max(0, y1 - th - baseline - 4)
            label_y2 = y1
            label_x1 = x1
            label_x2 = min(w, x1 + tw + 6)

            # 半透明背景
            overlay = annotated.copy()
            cv2.rectangle(overlay, (label_x1, label_y1),
                          (label_x2, label_y2), bg_color, -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

            # 文字
            cv2.putText(annotated, label,
                        (label_x1 + 3, label_y2 - baseline - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def _draw_dashed_rect(self, img, pt1, pt2, color, thickness, dash_length=8):
        """
        绘制虚线矩形框。
        OpenCV 的 rectangle 不支持虚线，通过分段绘制 line 实现。
        """
        x1, y1 = pt1
        x2, y2 = pt2

        edges = [
            ((x1, y1), (x2, y1)),  # 上边
            ((x2, y1), (x2, y2)),  # 右边
            ((x2, y2), (x1, y2)),  # 下边
            ((x1, y2), (x1, y1)),  # 左边
        ]

        for start, end in edges:
            self._draw_dashed_line(img, start, end, color, thickness, dash_length)

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness, dash_length):
        """在两点之间绘制虚线。"""
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx * dx + dy * dy)

        if length < 1:
            return

        ux = dx / length
        uy = dy / length

        draw = True
        pos = 0.0
        while pos < length:
            seg_end = min(pos + dash_length, length)
            if draw:
                p1 = (int(pt1[0] + ux * pos), int(pt1[1] + uy * pos))
                p2 = (int(pt1[0] + ux * seg_end), int(pt1[1] + uy * seg_end))
                cv2.line(img, p1, p2, color, thickness)
            pos = seg_end
            draw = not draw

    def is_loaded(self):
        """模型是否已成功加载并可进行推理"""
        return self.model is not None

    def get_stats(self):
        """
        返回模型运行统计信息。
        """
        avg_ms = 0.0
        if self._total_inferences > 0:
            avg_ms = round(self._total_inference_ms / self._total_inferences, 1)

        return {
            'model_loaded': self.is_loaded(),
            'model_name': os.path.basename(self.model_path) if self.model_path else 'N/A',
            'model_params': self._get_model_info() if self.is_loaded() else '未加载',
            'device': self._resolve_device(),
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'avg_inference_ms': avg_ms,
            'last_inference_ms': round(self._last_inference_ms, 1),
            'total_inferences': self._total_inferences,
            'total_detections': self._total_detections,
            'disease_detections': self._disease_detections,
            'bug_detections': self._bug_detections,
        }

    def update_config(self, conf_threshold=None, iou_threshold=None):
        """
        动态更新检测参数（无需重新加载模型）。
        """
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)
            print(f"[YOLO] 置信度阈值更新为: {self.conf_threshold}")
        if iou_threshold is not None:
            self.iou_threshold = float(iou_threshold)
            print(f"[YOLO] IoU 阈值更新为: {self.iou_threshold}")


# ============================================================
# 模块级便捷函数（供外部快速调用）
# ============================================================

# 全局检测器单例（延迟初始化）
_detector_instance = None
_detector_lock = threading.Lock()


def get_detector(model_path='../models/best.pt', **kwargs):
    """
    获取全局 YOLO 检测器单例。
    使用双重检查锁定（double-checked locking）确保多线程安全。
    """
    global _detector_instance
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = YOLODetector(model_path=model_path, **kwargs)
    return _detector_instance


def reset_detector():
    """重置全局检测器（用于热重载模型）"""
    global _detector_instance
    _detector_instance = None
