"""
[遗留代码 / Legacy Code]
此文件是早期 3 类分类器（真菌病害/健康/虫害）的独立桌面推理工具，
与当前系统的 12 类模型（app_fastapi.py + best_model.onnx）不兼容。
保留仅供架构参考，实际使用请通过 System A 的 FastAPI 服务。
"""
import os
import sys
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["真菌病害", "健康", "虫害"]
CONFIDENCE_THRESHOLD = 0.75
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备: {device}")

print("正在加载模型...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("模型加载成功！")
print("=" * 50)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def read_image_robust(image_path):
    """
    万能图片读取：彻底解决反斜杠转义和引号问题
    """
    # 1. 去除首尾空格和引号
    image_path = image_path.strip().strip('"').strip("'")
    # 2. 将反斜杠替换为正斜杠（避免 \t 变成制表符）
    image_path = image_path.replace('\\', '/')
    # 3. 转为 Path 对象
    path_obj = Path(image_path)

    # 打印清理后的路径，方便你核对
    print(f"🔍 实际路径: {path_obj}")

    if not path_obj.exists():
        print(f"❌ 文件不存在: {path_obj}")
        print("💡 提示：如果路径包含空格，请确保复制完整路径，不要修改")
        return None
    try:
        with open(path_obj, 'rb') as f:
            img_data = f.read()
        img_array = np.frombuffer(img_data, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print("❌ 图片解码失败，请确认文件是 jpg/png 格式")
        return img_bgr
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return None


def show_image_adaptive(img, title="AI 识别结果"):
    max_height, max_width = 600, 900
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_display = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_display = img
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def select_file_by_gui():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="请选择一张图片",
        filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp"), ("所有文件", "*.*")]
    )
    root.destroy()
    return file_path


def predict_single_image(image_path):
    img_bgr = read_image_robust(image_path)
    if img_bgr is None:
        return

    # AI推理
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    pred_class = CLASS_NAMES[pred_idx.item()]
    conf = confidence.item()

    if conf < CONFIDENCE_THRESHOLD:
        display_text = f"不确定 ({conf:.1%})"
        final_class = "不确定"
    else:
        display_text = f"{pred_class} ({conf:.1%})"
        final_class = pred_class

    print("\n" + "=" * 40)
    print(f"📁 图片: {Path(image_path).name}")
    print(f"🤖 AI判断: {final_class}")
    print(f"🎯 置信度: {conf:.2%}")
    if conf < CONFIDENCE_THRESHOLD:
        print("⚠️ 置信度低于阈值，建议人工复核")
    print("=" * 40)

    # 绿色文字（无背景框）
    cv2.putText(img_bgr, f"AI: {display_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    if conf < CONFIDENCE_THRESHOLD:
        cv2.putText(img_bgr, "请人工复核", (10, img_bgr.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    show_image_adaptive(img_bgr, f"AI: {final_class}")


if __name__ == "__main__":
    print("🌱 病虫害AI分类工具")
    print("=" * 40)
    print("请输入选项: [1] 鼠标点击选择图片  [2] 手动输入路径  [q] 退出")

    while True:
        choice = input("\n请选择 (1/2/q): ").strip()

        if choice.lower() == 'q':
            print("👋 已退出")
            break

        if choice == '1':
            print("正在打开文件选择器...")
            file_path = select_file_by_gui()
            if file_path:
                predict_single_image(file_path)
            else:
                print("⚠️ 未选择任何文件")
            continue

        if choice == '2':
            user_input = input("📎 请粘贴图片路径: ").strip()
            if user_input == "":
                continue
            predict_single_image(user_input)
            continue

        print("❌ 无效输入，请输入 1, 2 或 q")