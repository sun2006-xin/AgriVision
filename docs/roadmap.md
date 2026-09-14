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

## 阶段 18：MQTT 运行中断线重连验收（本机已完成）

- 已用本地 broker 主动断开第一次连接，验证 Paho 网络循环重新连接，传输器重试后完成 QoS 1 发布并确认队列事件。
- 已验证 broker 最终只收到一条有效事件；测试不使用公网 broker、不包含凭据。
- 真实公网 broker 的 TLS/认证、ACL、长时间网络抖动和 ESP32 无线恢复仍未验收。

## 阶段 19：离线事件自动调度（本机已完成）

- 新增默认关闭的 `EventSyncScheduler`；设置 `AGRIVISION_EVENTS_SYNC_INTERVAL` 为正数后，System B 才会周期性消费有界离线事件队列。
- 自动同步优先使用已配置的 MQTT，未配置 MQTT 时回退到 HTTP sink；每轮复用现有进程锁和 `AGRIVISION_EVENTS_SYNC_LIMIT` 上限，异常不会终止后台线程。
- 退出时会停止调度线程并关闭 MQTT；未配置传输器、间隔为 0 或非法配置时保持不外发。
- 真实公网 TLS/认证、跨进程部署、多实例协调和 ESP32 长时间断网恢复仍未验收。

## 阶段 20：离线同步可观测性（本机已完成）

- 新增 `/api/offline_events/sync_status`，返回自动同步是否启用、调度线程状态、队列待处理数、成功/失败批次数和事件计数。
- 状态只保存固定枚举、计数和 UTC 时间戳，不返回 broker URL、凭据、异常原文或事件正文；失败原因统一为脱敏的 `sync failed`。
- 已用测试验证初始状态、成功计数和异常脱敏；真实监控系统、告警阈值和多实例聚合仍需部署阶段设计。

## 阶段 21：跨进程离线同步互斥（本机已完成）

- 将单纯的进程内线程锁升级为基于 SQLite `BEGIN IMMEDIATE` 的非阻塞运行时锁，System B 的手动和自动同步共用 `offline_events/.sync-lock.db`。
- 已用真实 Windows spawn 子进程验证：一个进程持锁时另一个进程不能消费，释放后可重新获取；异常退出上下文也会释放锁。
- 锁文件位于 `.gitignore` 已覆盖的运行目录，不含事件正文或凭据；该方案只解决同一主机多进程，不覆盖跨主机/容器多实例协调。

## 阶段 22：HTTP 同步错误分类重试（本机已完成）

- HTTP sink 返回 4xx 时立即判定为 `permanent`，不再重复发送认证失败、路径错误或请求格式错误。
- 5xx、非标准非 2xx 和网络异常仍按现有最多 5 次策略有限重试；耗尽后标记为 `retry_exhausted`。
- 自动同步状态接口只暴露固定失败类别，不暴露 HTTP 响应正文、URL、凭据或异常文本；成功后确认删除规则保持不变。

## 阶段 23：本机 MQTT TLS 证书验收（本机已完成）

- 使用 `cryptography` 在测试临时目录生成带 `127.0.0.1` SAN 的短期自签名证书，启动仅监听回环地址的 TLS broker。
- 使用真实 Paho 运行时配置 `mqtts://` 和 `ca_certs` 完成 TLS 握手、MQTT v5 CONNECT、QoS 1 PUBLISH/PUBACK 和事件确认。
- 证书和私钥不进入仓库、不连接公网；公网证书链、认证、ACL、轮换和 ESP32 无线恢复仍需部署环境验收。

## 阶段 24：本机 MQTT TLS 认证验收（本机已完成）

- 在临时 TLS broker 上使用合成 username/password 验证 Paho 独立凭据注入，正确凭据可完成发布，错误凭据收到失败 CONNACK 后立即返回通用连接错误且不发布事件。
- 修复运行时对 MQTT v5 `ReasonCode` 的处理：认证失败不再被误判为连接成功；认证探测与正式网络 loop 共用单一生命周期，避免重复发布和 socket 资源泄漏。
- 凭据只存在测试进程内，不写入 URL、日志或仓库；公网认证、ACL、证书轮换和 ESP32 无线恢复仍需部署环境验收。

## 阶段 25：MQTT 全连接 CONNACK 验收（本机已完成）

- Paho 运行时现在对所有真实 Paho 客户端统一等待并检查 MQTT v5 CONNACK，不再仅在配置 username 时检查。
- 无认证本机连接收到失败 CONNACK 时，会在返回 transport 前转换为通用连接错误；成功连接、TLS、认证、QoS 1 发布和运行中断线重连行为保持不变。
- 新增本机失败 CONNACK 回归，并复跑旧版 fake Paho、普通 MQTT、TLS、认证和重连测试；公网 broker、ACL、证书轮换和 ESP32 无线恢复仍需部署环境验收。

## 阶段 26：MQTT 失败连接重试生命周期（本机已完成）

- 失败 CONNACK 或连接异常进入下一次初始重试前，会停止上一轮网络循环、恢复临时连接选项，并使用全新配置的 Paho client，避免旧 socketpair 和连接状态泄漏。
- 新增“第一次 CONNACK 失败、第二次成功”的生命周期回归，确认重试期间不会同时运行多个网络循环；既有匿名、TLS、认证、QoS 1 和运行中重连行为保持通过。
- 本机验证不等价于公网 broker 长时间故障、跨主机协调或 ESP32 无线恢复；这些仍需部署环境验收。

## 阶段 27：MQTT CONNACK 等待边界（本机已完成）

- 将 CONNACK 等待时间从硬编码改为 `connack_timeout` 参数，默认 5 秒，并限制在 0.1–30 秒；System B 通过 `AGRIVISION_MQTT_CONNACK_TIMEOUT` 注入，兼顾边缘设备启动延迟与失败快速返回。
- 新增参数边界和实际等待值回归；该参数不包含地址、凭据或其他敏感信息，也不会改变发布重试上限。
- 本机时序验证不等价于公网 broker 延迟分布、长时间断网或 ESP32 现场无线恢复，仍需部署环境验收。

## 阶段 28：MQTT 发布确认等待边界（本机已完成）

- 将 Paho QoS 1 发布确认的硬编码 10 秒改为 `publish_timeout` 参数，默认 10 秒，并限制在 0.1–60 秒；System B 通过 `AGRIVISION_MQTT_PUBLISH_TIMEOUT` 注入。
- 新增发布确认实际等待值和配置边界回归；只有 Paho 明确报告发布完成时才确认离线事件，超时不会放宽确认规则。
- 本机验证不等价于公网 broker 的吞吐、QoS 延迟、长时间断网或 ESP32 现场恢复，仍需部署环境验收。

## 阶段 29：MQTT 关闭生命周期（本机已完成）

- MQTT close 回调改为幂等且异常安全：重复调用直接返回，断开异常不会阻止网络 loop 停止。
- 新增 disconnect 异常和重复 close 回归，确保手动同步、后台调度退出和进程 atexit 清理不会因二次关闭而产生未处理异常。
- 本机客户端生命周期验证不等价于公网 broker 长时间故障或 ESP32 现场恢复，仍需部署环境验收。

## 阶段 30：MQTT 发布状态异常重试（本机已完成）

- 将 Paho 发布过程中的 `RuntimeError` 纳入现有有限重试路径，避免客户端状态异常直接越过传输器边界。
- 验证异常期间不会确认离线事件，重试次数仍受 `max_attempts` 上限约束，状态接口不会暴露 Paho 异常原文。
- 本机 fake publisher 验证不等价于公网 broker 的全部错误类型、长时间断网或 ESP32 现场恢复，仍需部署环境验收。

## 阶段 31：MQTT 配置状态可观测性（本机已完成）

- `/api/offline_events/sync_status` 在 MQTT 启用时报告当前生效的 CONNACK 等待和发布确认超时，仅返回数值字段；未启用 MQTT 时返回 `null`。
- 回归测试验证状态字段与实际配置传递一致，并确认状态响应不包含 broker 地址、账号、密码或异常文本。
- 本机软件验证不等价于公网 broker、ACL、证书轮换或 ESP32 现场链路验收。

## 阶段 32：事件同步状态面板（本机已完成）

- 实时监控页新增事件同步状态摘要，展示当前传输类型、待发送数量、成功/失败批次数和 MQTT 超时数值。
- 前端通过现有脱敏状态接口读取数据，采用白名单字段和纯文本渲染，不把 broker 地址、账号、密码或异常文本写入页面。
- 新增静态前端回归；页面行为验证属于本机软件范围，公网 broker、ACL、证书轮换和 ESP32 现场链路仍需验收。

## 阶段 33：事件状态刷新故障隔离（本机已完成）

- 事件同步状态请求从实时监控主刷新流程中独立出来；状态接口暂时不可用时，页面显示固定提示，检测数据刷新不被中断。
- 异常路径不展示异常原文，继续使用脱敏状态接口和固定字段渲染。
- 静态回归和本机服务启动检查已完成；Playwright 浏览器实测因当前环境未安装依赖而未完成，公网 broker 与 ESP32 现场链路仍需验收。

## 阶段 34：前端状态面板浏览器回归（本机已完成）

- 新增独立的 `requirements-ui-test.txt` 和 `tools/browser_smoke_event_status.py`，不改变生产依赖。
- 真实 Chromium smoke 已验证首页加载、事件状态节点存在、状态接口失败时显示固定提示，且实时监控主状态节点仍可用。
- 浏览器验证仅覆盖本机页面故障隔离，不代表公网 broker、ACL、证书轮换或 ESP32 现场链路验收。

## 阶段 35：公开 CI 浏览器质量门禁（本机已完成）

- 在 GitHub Actions 增加独立 `browser-smoke` job：安装可选 UI 依赖和 Chromium，启动本地 System B 后运行状态面板 smoke。
- 通过有限探活、15 分钟 job 超时和 `always()` 清理保证服务生命周期受控；浏览器使用低资源参数兼容 CI runner。
- 本机已复现并修复 Chromium 资源启动失败，随后 smoke 通过；CI 云端最终结果需以推送后的 GitHub Actions 运行记录为准。

## 阶段 36：浏览器控制台错误回归（本机已完成）

- smoke 现在同时监听 `pageerror` 和浏览器 `console.error`，非预期错误会使质量门禁失败。
- 被测的状态接口 503 和缺失 favicon 被显式模拟/处理，不被误报为页面脚本错误；真实 Chromium 回归已通过。
- 本阶段只覆盖本机前端错误边界，不代表公网 broker、ACL、证书轮换或 ESP32 现场链路验收。

## 阶段 37：MQTT 部署配置安全预检（本机已完成）

- `/api/offline_events/sync_status` 新增脱敏 MQTT 配置预检摘要：配置是否完整、协议/TLS 模式、主题/客户端 ID/凭据/CA 是否已配置及固定问题代码。
- 预检不建立 broker 连接，不返回 URL、文件路径、用户名、密码或原始异常；远端明文 MQTT 会明确标记为 `remote_tls_required`。
- 实时监控面板显示“配置就绪/待完善”和安全模式，便于现场联调前发现配置缺口；公网 broker、ACL 和 ESP32 仍需现场验收。

## 阶段 38：MQTT 运行时状态闭环（本机已完成）

- 统一记录 `not_started`、`not_configured`、`connecting`、`connected`、`connection_failed` 和 `publish_failed` 六种固定运行时状态。
- 手动同步和后台同步现在共同更新状态面板；发布重试耗尽返回固定的 `retry_exhausted` 类别，事件仍不会在确认前被删除。
- 前端只将白名单状态映射为中文标签，不展示异常原文或配置值；本机 75/75 测试、公开扫描和编译检查通过。
- 本阶段仍不等同于公网 broker、ACL、证书轮换或 ESP32 现场验收。

## 阶段 39：MQTT 空队列同步语义修复（本机已完成）

- 手动同步在队列为空时直接返回成功且 `attempts=0`，不调用发布器，也不虚增成功批次数。
- 空同步仍记录固定的 `empty` 状态；事件确认、发布重试和运行时状态逻辑不受影响。
- 本阶段只修正本机接口统计语义，不代表公网 broker、ACL、证书轮换或 ESP32 现场验收。

## 阶段 40：同步请求前置校验（本机已完成）

- MQTT 手动同步在建立 broker 连接前先校验 JSON 对象和 `limit`，非法请求直接返回 400，不触发网络连接、重试或凭据使用。
- 同步接口不再把非法 JSON 或空请求体静默转换为空对象；HTTP 与 MQTT 手动入口保持一致的请求边界。
- 本阶段只覆盖接口输入校验顺序，不代表公网 broker、ACL、证书轮换或 ESP32 现场验收。

## 阶段 41：离线事件单条大小保护（本机已完成）

- 离线缓存现在在写入前按 UTF-8 字节数检查单条事件，超过 `max_bytes` 直接拒绝。
- 超限事件不会生成临时文件、覆盖已有事件或被 trim 静默删除，避免边界条件下的数据丢失。
- 本阶段只覆盖本机缓存保护，不代表公网 broker、ACL、证书轮换或 ESP32 现场验收。

## 阶段 42：外发事件隐私白名单（本机已完成）

- HTTP、MQTT 和离线事件查询统一使用事件协议白名单，只保留版本、事件标识、时间、摄像头标识和检测统计字段。
- 缓存中即使存在旧版本或被篡改的 `url`、密码、图片路径及未知 payload 字段，也不会进入外发载荷或查询响应。
- 本阶段只覆盖软件层字段投影，不代表公网 broker、ACL、证书轮换或 ESP32 现场验收。

## 阶段 43：离线缓存记录读取保护（本机已完成）

- 缓存读取只接受合法 JSON 对象和安全 `event_id`，标量、缺失 ID 或路径穿越记录不会进入同步器。
- 无效文件不会被自动删除，便于运维取证；同时不会阻塞有效事件，也不会因异常记录导致手动同步接口报错。
- 本阶段只覆盖本机缓存读取边界，不代表公网 broker、ACL、证书轮换或 ESP32 现场验收。

## 发布规则

每个阶段单独建立分支、完成测试和隐私扫描后提交并推送；模型缓存、数据库、摄像头地址、Webhook 和本地日志不得进入公开提交。
