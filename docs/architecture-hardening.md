# System B 架构与运行时边界收口

本次迭代把阶段 2 的工程化基线继续收口为可维护的模块边界，重点解决“目录已经拆出，但主应用仍直接承载路由和状态”的问题。

## 已完成

- `routes/control.py`：YOLO、告警、检测参数和 mask 预览接口。
- `routes/events.py`：离线事件查询、确认、HTTP/MQTT 同步和同步状态。
- `routes/video.py`：按摄像头隔离的 MJPEG 视频流。
- `routes/diagnosis.py`：System B 到 System A 的深度诊断代理和状态检查。
- `routes/pages.py`：主页面和大屏页面入口。
- `page_templates.py` 与 `templates/`：以 UTF-8 加载无需构建工具的主页面和大屏模板；内联脚本有 Node 语法回归。
- `app.py` 现在负责依赖组装、设备适配、检测循环和应用启动，不再直接声明业务 HTTP 路由。

所有 Blueprint 都通过显式依赖注册，模块可以在不加载模型和摄像头的情况下单独测试。

## 参数边界

`ConfigManager` 是参数的唯一校验入口：

- 拒绝未知字段、bool 冒充数字、整数参数小数、NaN/Infinity 和越界值。
- 校验形态学核的奇数约束。
- 校验病斑、虫害、绿色占比阈值的单调关系。
- 配置文件启动加载、批量更新、单参数更新和 mask 预览均复用同一套后端校验。
- 损坏或不合法的持久化配置回退默认值，不把坏值交给检测引擎。

## 任务生命周期

`CameraTaskManager` 继续按摄像头和任务类型隔离队列，并增加：

- 有界的 `max_retries` 和 `retry_backoff`。
- `cancel()` 取消周期投递、清空尚未执行的任务并记录取消状态。
- `stop_all()` 清理队列，避免停止后残留任务让系统永远处于 busy。
- 受控停止后允许重新创建 worker，便于测试和服务生命周期管理。
- `/api/tasks` 暴露有限的完成、失败、取消、重试和最近任务状态，不暴露异常原文。

## 远程媒体安全

远程部署时，认证范围不仅包含 `/api/*` 和 `/metrics`，还包含：

- `/video_feed*`
- `/dataset/*`
- `/history/image/*`

浏览器先用 Bearer token 调用 `/api/auth/session`，服务端签发绑定来源地址、短期有效的 HttpOnly/SameSite cookie；视频 `<img>` 不需要把 token 放入 URL。未配置 token 的远程请求仍然失败关闭，回环开发保持兼容。

## 验收边界

本阶段应验证：全量 unittest、Python AST/compileall（排除本地 `.venv`）、真实 System B 路由表、媒体认证会话和公开文件扫描。Docker、多实例跨主机协调、真实摄像头长期稳定性以及真实田间标注数据仍属于部署/数据验收，不在本地软件测试中冒充完成。
