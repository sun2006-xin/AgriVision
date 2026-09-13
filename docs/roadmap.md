# AgriVision 技术路线图

## 项目简介

AgriVision 面向温室和田间场景，提供从 ESP32-CAM 图像采集、实时病虫害筛查、历史记录、告警通知，到多模型辅助诊断报告的开源实验平台。System A 负责图片诊断，System B 负责现场监控；两者通过 REST API 松耦合集成。

## 阶段 1：可运行性与安全基线（已完成）

- 修复 System B → System A `/report` 的 multipart 字段契约。
- 参数写入经过白名单、类型、有限值和范围校验。
- 默认 CORS 收敛到本机 A/B 前端来源。
- 增加接口契约和配置边界回归测试。
- 验收：`python -m unittest tests.test_stage1_contracts -v` 通过。

## 阶段 2：工程化与可观测性（已完成基础项）

- 已增加 `/health/live` 与 `/health/ready`，区分进程存活和模型/摄像头就绪状态。
- 已增加请求 ID、请求耗时和稳定事件名日志，不记录 URL 配置、Webhook 或图像内容。
- 已将配置写入改为同目录临时文件 + 原子替换，降低中断写入造成 JSON 损坏的风险。
- 已增加 6 个跨阶段回归测试；路由/检测/存储的大规模拆分、任务取消重试和固件契约测试列入下一小迭代。
- 验收：`python -m unittest discover -s tests -v` 通过，源码 compileall 通过。

## 阶段 3：算法评估与现场闭环（评估基础已完成）

- 已新增 `system_a/core/evaluation.py`，支持 Accuracy、每类 Precision/Recall/F1、宏平均、混淆矩阵和 ECE。
- 已新增 `system_a/core/evaluate_predictions.py`，可对固定 JSON 预测清单运行评估，不加载模型、不上传数据。
- System A 分类和 CLIP 输出增加 `uncertain`/`confidence_band`；低置信度结果会在 LLM 报告中标注为需要人工复核。
- 尚未报告真实模型指标：仓库当前没有可公开验证的标注集；后续必须按作物、病害、光照和设备来源建立固定验证集。
- 后续将 HSV 定位为快速筛查层，YOLO/分割定位为视觉证据层，LLM 仅负责解释与建议。

评估清单格式示例：

```json
{
  "labels": ["健康", "番茄早疫病"],
  "records": [
    {"true": "健康", "pred": "健康", "confidence": 0.92}
  ]
}
```

运行：`python system_a/core/evaluate_predictions.py path/to/predictions.json`。

## 阶段 4：边缘部署与可扩展集成（离线队列基础已完成）

- 已新增版本化、隐私安全的检测事件 envelope，事件只包含摄像头 ID 和检测摘要。
- 已新增有界离线 JSON 队列，支持原子写入、容量限制、待同步读取和传输确认删除。
- 已在 System B 检测流程接入离线事件队列，并提供 `/api/offline_events` 与 `/api/offline_events/ack`。
- 尚未接入 MQTT/HTTP 传输器、断点续传和真实 ESP32 断网恢复联调；这些必须在下一小迭代中用仿真设备和现场日志验收。
- 量化模型、边缘推理、Docker/服务管理部署列为后续工作。

## 发布规则

每个阶段单独建立分支、完成测试和隐私扫描后提交并推送；模型缓存、数据库、摄像头地址、Webhook 和本地日志不得进入公开提交。
