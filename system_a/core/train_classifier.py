"""
AgriVision System A — ResNet-18 分类器训练脚本
================================================
基于迁移学习训练 12 类植物病害分类器：
  -  backbone: ResNet-18（ImageNet 预训练）
  -  输出: best_model.pth（保存到当前工作目录）
  -  训练后需运行 convert_to_onnx.py 转换为 ONNX 格式供推理使用

用法: python train_classifier.py --data /path/to/dataset --epochs 10
数据集结构: data/train/{class_name}/*.jpg, data/val/{class_name}/*.jpg
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import argparse

# ====================== 配置区 ======================
parser = argparse.ArgumentParser(description='训练 ResNet18 植物病害分类器')
parser.add_argument('--data', default=r"【请修改：你的数据集路径】", help='数据集根目录')
parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
args = parser.parse_args()

DATA_ROOT = args.data
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
NUM_CLASSES = 12
LR = 0.001
# ==================================================

# 1. 数据增强
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 加载数据
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_ROOT, "train"), transform=transform_train)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_ROOT, "val"), transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. 加载预训练模型（迁移学习）
model = models.resnet18(weights="IMAGENET1K_V1")

if os.path.exists("best_model.pth"):
    print("发现历史模型 best_model.pth，加载中... 将在此基础上继续训练！")
    checkpoint = torch.load("best_model.pth", map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
else:
    print("未找到历史模型，从头开始训练...")

# 替换最后一层全连接（仅替换一次）
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# 4. 定义损失函数（类别均衡权重）和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 计算逆频率类别权重
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
class_names = train_dataset.classes
train_dir = os.path.join(DATA_ROOT, "train")
class_counts = [
    len([f for f in os.listdir(os.path.join(train_dir, c)) if f.lower().endswith(IMG_EXTS)])
    for c in class_names
]
total = sum(class_counts)
weights = [total / (len(class_names) * count) for count in class_counts]
class_weights = torch.FloatTensor(weights).to(device)
print("类别:", class_names, "样本数:", class_counts, "权重:", [round(w, 3) for w in weights])

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 5. 训练
model.to(device)
print("正在使用设备:", device)

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print("轮次 %d/%d, 损失: %.4f, 验证准确率: %.2f%%" % (epoch + 1, EPOCHS, running_loss / len(train_loader), acc))

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("  *** 新最优模型已保存 (准确率 %.2f%%) ***" % acc)

print("训练完成！模型已保存为 best_model.pth")
