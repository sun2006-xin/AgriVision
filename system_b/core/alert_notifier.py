"""
智能告警通知模块
===============
当检测等级达到"警告"或"严重"时，通过钉钉 Webhook 推送告警消息。

功能特性：
  - 钉钉机器人 Webhook 通知
  - 告警冷却机制（同一等级在冷却期内不重复发送）
  - 告警历史记录（最近 50 条）
  - 可通过 API 动态配置 Webhook URL 和开关

使用方式：
    from alert_notifier import AlertNotifier
    notifier = AlertNotifier()
    notifier.configure(webhook_url="https://oapi.dingtalk.com/robot/send?access_token=xxx")
    notifier.notify(level="警告", disease_count=5, pest_count=2)
"""

import json
import time
import requests
from datetime import datetime


class AlertNotifier:
    """告警通知器"""

    def __init__(self):
        self.webhook_url = ""
        self.enabled = False
        # 告警冷却：同一等级在冷却期（秒）内不重复发送
        self.cooldown_seconds = 300  # 5 分钟
        self._last_alert_time = {}  # {level: timestamp}
        # 告警历史（最近 50 条）
        self.alert_history = []
        self.MAX_HISTORY = 50

    def configure(self, webhook_url=None, enabled=None, cooldown=None):
        """配置通知器参数"""
        if webhook_url is not None:
            self.webhook_url = webhook_url
        if enabled is not None:
            self.enabled = enabled
        if cooldown is not None:
            self.cooldown_seconds = int(cooldown)
        return {
            'webhook_url': self.webhook_url[:30] + '...' if len(self.webhook_url) > 30 else self.webhook_url,
            'enabled': self.enabled,
            'cooldown_seconds': self.cooldown_seconds,
        }

    def get_config(self):
        """获取当前配置"""
        return {
            'webhook_url': self.webhook_url,
            'enabled': self.enabled,
            'cooldown_seconds': self.cooldown_seconds,
        }

    def notify(self, level, level_code, disease_count=0, pest_count=0,
               disease_ratio=0.0, pest_ratio=0.0, confidence='', image_path=None):
        """
        发送告警通知。

        仅在以下条件全部满足时才发送：
          1. 通知器已启用 (enabled=True)
          2. Webhook URL 已配置
          3. 等级 >= "警告" (level_code >= 2)
          4. 同一等级不在冷却期内

        返回:
            dict: {'sent': bool, 'reason': str}
        """
        if not self.enabled:
            return {'sent': False, 'reason': '通知未启用'}

        if not self.webhook_url:
            return {'sent': False, 'reason': '未配置 Webhook URL'}

        if level_code < 2:
            return {'sent': False, 'reason': f'等级 {level} 未达到告警阈值(警告+)'}

        # 冷却检查
        now = time.time()
        last_time = self._last_alert_time.get(level, 0)
        if now - last_time < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - (now - last_time))
            return {'sent': False, 'reason': f'冷却中，剩余 {remaining} 秒'}

        # 构建消息
        msg = self._build_message(level, level_code, disease_count, pest_count,
                                  disease_ratio, pest_ratio, confidence)

        # 发送
        success = self._send_dingtalk(msg)

        # 更新冷却时间
        if success:
            self._last_alert_time[level] = now

        # 记录历史
        self.alert_history.insert(0, {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': level,
            'level_code': level_code,
            'disease_count': disease_count,
            'pest_count': pest_count,
            'confidence': confidence,
            'sent': success,
        })
        if len(self.alert_history) > self.MAX_HISTORY:
            self.alert_history = self.alert_history[:self.MAX_HISTORY]

        return {'sent': success, 'reason': '发送成功' if success else '发送失败'}

    def _build_message(self, level, level_code, disease_count, pest_count,
                       disease_ratio, pest_ratio, confidence):
        """构建钉钉 Markdown 告警消息"""
        emoji = {2: '⚠️', 3: '🚨'}.get(level_code, '⚠️')
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        text = f"""## {emoji} 病虫害告警 - {level}

**时间**: {now_str}

**检测详情**:
- 病斑数量: {disease_count} 个 (占比 {disease_ratio*100:.1f}%)
- 虫害数量: {pest_count} 个 (占比 {pest_ratio*100:.1f}%)
- 融合置信度: {confidence}

> 来自大棚病虫害实时监控系统"""
        return text

    def _send_dingtalk(self, text):
        """发送钉钉 Webhook 消息"""
        try:
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": "病虫害告警",
                    "text": text
                }
            }
            headers = {'Content-Type': 'application/json'}
            resp = requests.post(self.webhook_url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                return result.get('errcode', -1) == 0
            return False
        except Exception as e:
            print(f"[AlertNotifier] 发送失败: {e}")
            return False

    def get_history(self):
        """获取告警历史"""
        return self.alert_history


# 全局单例
_notifier = None

def get_notifier():
    global _notifier
    if _notifier is None:
        _notifier = AlertNotifier()
    return _notifier
