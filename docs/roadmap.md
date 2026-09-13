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
- 已在 System B 检测流程接入离线事件队列，并提供 `/api/offline_events`、`/api/offline_events/ack`。
- 已增加手动 `/api/offline_events/sync` HTTP 同步入口：通过 `AGRIVISION_EVENTS_SINK_URL` 显式启用，仅允许 HTTPS 远端或本机 HTTP，并在 2xx 成功后确认删除事件。
- MQTT、断点续传和真实 ESP32 断网恢复联调仍未完成；这些必须用仿真设备和现场日志验收。
- 量化模型、边缘推理、Docker/服务管理部署列为后续工作。

## 阶段 5：可靠同步契约（基础项已完成）

- 已为有序事件批次生成稳定 SHA-256 幂等键，并通过 `Idempotency-Key` 发送给接收端，便于远端去重。
- 已增加 System B 进程内同步锁；并发触发同步返回 409，不会同时消费同一批本地事件。
- 已保持失败不确认、2xx 才确认删除、批次上限 100 和远端 HTTPS 约束。
- 接收端幂等去重、跨进程锁、MQTT、自动调度和 ESP32 真实断网恢复仍未验收；下一阶段需用模拟接收端和设备日志补证据。

## 阶段 6：同步端到端验收（本地模拟已完成）

- 已增加本地内存接收端测试，模拟“接收成功但响应丢失”的网络故障。
- 已验证传输器重试不会造成重复事件，接收端按稳定批次幂等键去重；只有重试得到 2xx 后本地队列才确认删除。
- 该测试不连接互联网、不读取摄像头、不上传图片或模型；真实远端服务、跨进程部署和 ESP32 断网恢复仍未验收。

## 阶段 7：本机 HTTP 回环验收（已完成）

- 已将 HTTP 发送函数下沉到独立传输模块，生产代码与测试复用同一请求实现。
- 已用临时 `127.0.0.1` HTTP 服务验证请求路径、版本化 JSON、幂等键和 202 成功后的队列删除。
- 该验收仍不等价于公网 HTTPS、认证、反向代理、MQTT broker 或真实 ESP32 网络恢复；这些保留为部署阶段验收项。

## 阶段 8：开源发布 CI 门禁（已完成）

- 已增加轻量 GitHub Actions 工作流，使用不含模型/设备配置的测试依赖运行全量 unittest、Python AST 校验和跟踪文件隐私扫描。
- 已将公开发布扫描脚本固化为 `tools/validate_public_repo.py`，扫描凭据、RTSP 账号密码和常见私有网段，避免仅依赖人工检查。
- CI 不下载模型、不访问摄像头、不连接外部事件接收端；GitHub Actions 已对提交 `1c68338` 返回 `completed / success`。

## 阶段 10：MQTT 传输契约准备（已完成基础项）

- 已增加注入式 `MqttEventTransport`，校验具体发布主题，使用版本化批次载荷、稳定 `batch_id`、QoS 1 和非 retain 发布。
- 已验证发布失败后的有限重试，以及仅在发布器明确返回成功后确认本地事件。
- 当前不自动连接 broker，也未写入任何 broker 地址、账号或密码；真实 MQTT 客户端、TLS、认证和设备联调仍需单独配置并验收。

## 阶段 11：MQTT 契约云端回归（已完成）

- GitHub Actions 已对阶段 10 提交 `de89384` 返回 `completed / success`，确认 MQTT 契约测试在干净 Runner 上可复现。
- 运行记录：[Public quality gate #3](https://github.com/sun2006-xin/AgriVision/actions/runs/34762437681)。
- 该证据仍只覆盖软件契约，不覆盖真实 broker、TLS/认证或 ESP32 现场链路。

## 阶段 12：MQTT 连接安全边界（已完成基础项）

- 已增加 broker URL 校验：远端只允许 `mqtts://`，本机开发才允许 `mqtt://`。
- 已拒绝 URL 内嵌账号/密码、路径、查询参数、片段和非法端口，凭据必须通过独立安全配置注入。
- 真实 TLS 证书校验、认证、broker 连接和 ESP32 现场联调仍未验收。

## 阶段 13：可选 Paho MQTT 运行时（已完成基础项）

- 已新增 `mqtt_runtime.py`，懒加载可选 `paho-mqtt`，避免项目导入时自动联网。
- 已实现独立凭据参数、TLS broker 连接、Paho 网络循环、QoS 发布完成等待和可关闭生命周期。
- 已增加 `requirements-mqtt.txt`；本地使用假 Paho 客户端验证运行时流程，但真实 broker、TLS 证书链、认证和 ESP32 现场仍未验收。

## 阶段 14：Paho 运行时云端回归（已完成）

- GitHub Actions 已对提交 `4f33688` 返回 `completed / success`，确认可选 Paho 运行时代码和假客户端测试在干净 Runner 上通过。
- 运行记录：[Public quality gate #6](https://github.com/sun2006-xin/AgriVision/actions/runs/34762728744)。
- 真实 broker、TLS 证书链、认证和 ESP32 现场链路仍未验收。

## 阶段 15：本机 MQTT 协议验收（已完成基础项）

- 已用 Python 标准库启动临时本机 MQTT broker，并用 Paho 2.1.0 实际完成 CONNECT、QoS 1 PUBLISH/PUBACK 和正常关闭。
- 已验证事件队列在本机 broker 接收成功后确认删除；测试仅绑定 `127.0.0.1` 随机端口，不产生公网流量。
- 该验收不覆盖公网 broker 的 TLS 证书链、用户名/密码认证、ACL、重连策略或 ESP32 无线链路；这些仍需部署环境验收。

## 阶段 16：System B MQTT 手动同步接入（已完成基础项）

- 已新增 `/api/offline_events/sync_mqtt`，仅在显式请求且配置完整时懒加载 Paho 客户端；未配置时返回 503，不自动联网。
- 支持独立的 `AGRIVISION_MQTT_BROKER_URL`、`AGRIVISION_MQTT_TOPIC`、`AGRIVISION_MQTT_CLIENT_ID`、`AGRIVISION_MQTT_USERNAME`、`AGRIVISION_MQTT_PASSWORD` 和可选 CA 路径环境变量。
- 连接失败统一返回不含地址/凭据的 503；真实部署仍需 TLS 证书、认证、ACL、重连和 ESP32 现场验收。

## 阶段 17：MQTT 初始连接重试（已完成基础项）

- Paho 运行时已增加最多 5 次的有界初始连接重试和线性退避；连接成功后再启动网络循环。
- 初始连接最终失败时转换为通用运行时错误，由 System B 返回不泄露配置的 503。
- Paho 已有的断线重连、真实 broker 认证、TLS 证书链和 ESP32 无线恢复仍需部署环境验收。

## 发布规则

每个阶段单独建立分支、完成测试和隐私扫描后提交并推送；模型缓存、数据库、摄像头地址、Webhook 和本地日志不得进入公开提交。
