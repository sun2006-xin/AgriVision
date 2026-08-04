"""
AgriVision System A — PyTorch → ONNX 模型转换工具
===================================================
将训练好的 ResNet-18（12 类）从 best_model.pth 导出为 best_model.onnx，
输出到 models/ 目录供 app_fastapi.py 的 ONNX Runtime 加载推理。

前置: 先运行 train_classifier.py 生成 best_model.pth
用法: python convert_to_onnx.py（无需参数，自动查找同目录 pth 文件）
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models

# 强制使用 legacy tracing 导出器，绕过 dynamo 的 GBK 编码问题
os.environ["PYTORCH_ONNX_DYNAMO"] = "0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PTH_PATH = os.path.join(BASE_DIR, "best_model.pth")
# ONNX 输出到 models/ 目录，与 app_fastapi.py 的读取路径保持一致
ONNX_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.onnx")

# 1. 加载你训练好的模型结构
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 12)

# 2. 加载权重
model.load_state_dict(torch.load(PTH_PATH, map_location="cpu"))
model.eval()  # 切换到推理模式

# 3. 创建 dummy 输入（用于追踪模型计算图）
dummy_input = torch.randn(1, 3, 224, 224)

# 4. 导出为 ONNX（dynamo=False 强制走 tracing 路径）
os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,  # 输出文件名
    export_params=True,
    opset_version=11,   # 稳定版本
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # 支持动态batch
    dynamo=False,
)

print("[OK] 转换成功！已生成 best_model.onnx")