"""
检测参数配置管理模块

负责管理虫害检测系统的所有可调参数（如颜色阈值、形态学参数、等级判定标准等）。
通过 ConfigManager 类实现参数的持久化存储（JSON 文件）和运行时读写，
使得用户在 Web 界面上调整的参数能够在重启后保留。
"""

import json
import math
import os
import tempfile
from detection_enhanced import DetectionConfig, DEFAULT_CONFIG

# 配置文件存放目录，位于本模块所在目录下的 config/ 文件夹
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
# 检测参数的 JSON 配置文件完整路径：config/detection_params.json
# 该文件保存了用户调整后的所有检测参数，格式为键值对的 JSON 对象
CONFIG_FILE = os.path.join(CONFIG_DIR, "detection_params.json")


class ConfigManager:
    """
    检测参数配置管理器

    作为 DetectionConfig 默认参数与用户实际使用参数之间的桥梁：
    - DetectionConfig 定义了所有检测参数的默认值（代码内置）
    - ConfigManager 在此基础上支持用户通过 Web 界面调整参数，
      并将调整后的参数持久化到 JSON 文件中
    - 启动时优先从 JSON 文件加载用户配置，若文件不存在或损坏则回退到默认值

    典型用法：
        config_mgr = ConfigManager()
        current = config_mgr.get_current_config()   # 获取当前生效的参数
        config_mgr.update_param("GREEN_ADV", 35)     # 修改单个参数并自动保存
    """

    def __init__(self):
        """
        初始化配置管理器

        初始化顺序：
        1. _ensure_dir()  — 先确保 config/ 目录存在，避免后续读写文件时报错
        2. _load_config() — 再从 JSON 文件加载用户配置，若文件不存在则用默认值初始化并创建文件
        """
        self._ensure_dir()
        self._load_config()
    
    def _ensure_dir(self):
        """
        确保配置目录存在

        使用 exist_ok=True 保证目录已存在时不会报错，
        这是后续文件读写操作的前置条件。
        """
        os.makedirs(CONFIG_DIR, exist_ok=True)

    def _load_config(self):
        """
        从 JSON 文件加载用户配置

        加载策略（三级回退）：
        1. 文件存在且 JSON 格式正确 → 通过 _apply_config() 应用配置
        2. 文件存在但 JSON 损坏或读取失败 → 捕获异常，回退到 DEFAULT_CONFIG 的默认值
        3. 文件不存在（首次运行） → 使用 DEFAULT_CONFIG 默认值，并立即调用 save_config() 创建配置文件

        这样设计保证了无论何种情况，self.current_config 都能被正确初始化。
        """
        if os.path.exists(CONFIG_FILE):
            try:
                # 尝试从 JSON 文件读取用户之前保存的配置
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    is_valid, errors = self.validate_params(data)
                    if not is_valid:
                        raise ValueError("; ".join(errors))
                    self._apply_config(data)
            except (json.JSONDecodeError, IOError, TypeError, ValueError) as e:
                # JSON 解析失败或文件 I/O 错误，回退到代码内置的默认配置
                print(f"加载配置失败，使用默认配置: {e}")
                self.current_config = DEFAULT_CONFIG.to_dict()
        else:
            # 首次运行，配置文件尚不存在，使用默认值并创建初始配置文件
            self.current_config = DEFAULT_CONFIG.to_dict()
            try:
                self.save_config(self.current_config)
            except IOError as e:
                # 写入失败时仍可使用默认配置运行，仅打印警告
                print(f"创建初始配置文件失败（默认配置仍可使用）: {e}")
    
    def _apply_config(self, data):
        """
        将外部数据应用到 DetectionConfig 实例上

        为什么不直接使用传入的 data 字典？
        - 创建一个全新的 DetectionConfig 实例（自带所有默认值）
        - 再通过 from_dict() 将 JSON 中的数据覆盖上去
        - 最后通过 to_dict() 导出

        这样做的好处：
        1. 只保留 DetectionConfig 中已知的参数，自动过滤掉 JSON 中可能残留的废弃字段
        2. 对于 JSON 中缺失的参数，自动使用 DetectionConfig 的默认值填充
        3. 保证了 self.current_config 的字段集合始终与代码定义一致
        """
        is_valid, errors = self.validate_params(data)
        if not is_valid:
            raise ValueError("; ".join(errors))
        config = DetectionConfig()      # 新建实例，内置所有默认值
        config.from_dict(self._normalize_config(data))
        self.current_config = config.to_dict()  # 导出为字典，保证字段完整性

    @staticmethod
    def _normalize_config(params):
        """Normalize JSON-compatible values before they reach the detector."""
        normalized = dict(params)
        if isinstance(normalized.get("base_resolution"), list):
            normalized["base_resolution"] = tuple(normalized["base_resolution"])
        return normalized

    @staticmethod
    def _check_order(candidate, keys, label, errors):
        values = [candidate.get(key) for key in keys]
        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ) and any(left > right for left, right in zip(values, values[1:])):
            errors.append(f"{label}必须按从低到高排列")

    def save_config(self, params):
        """
        将参数持久化到 JSON 配置文件

        使用 ensure_ascii=False 保证中文等非 ASCII 字符正常显示，
        indent=2 使文件具有良好的可读性。
        保存成功后同步更新内存中的 self.current_config。
        写入失败时仅打印错误，不抛出异常，保证 Web 应用继续运行。

        :param params: 要保存的参数字典（通常为 self.current_config）
        :return: True 保存成功，False 保存失败
        """
        if not isinstance(params, dict):
            raise ValueError("参数必须是 JSON 对象")
        candidate = DEFAULT_CONFIG.to_dict()
        candidate.update(params)
        is_valid, errors = self.validate_params(candidate, require_complete=True)
        if not is_valid:
            raise ValueError("; ".join(errors))
        candidate = self._normalize_config(candidate)
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix="detection_params.", suffix=".tmp", dir=CONFIG_DIR
            )
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(candidate, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, CONFIG_FILE)
            self.current_config = candidate
            return True
        except IOError as e:
            print(f"保存配置失败: {e}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    
    def get_current_config(self):
        """
        获取当前生效的配置（用户可能已修改过的版本）

        返回的是运行时内存中的参数字典，包含用户通过 Web 界面调整后的值。
        这是检测引擎实际使用的配置。
        """
        return dict(self.current_config)

    def get_default_config(self):
        """
        获取代码内置的默认配置（始终反映最新的 DetectionConfig 默认值）

        与 get_current_config() 的区别：
        - get_current_config() 返回用户调整后的参数
        - get_default_config() 每次新建一个 DetectionConfig 实例，返回未经修改的原始默认值
        - 常用于 Web 界面上"对比默认值"或"恢复默认"按钮的参照
        """
        fresh_config = DetectionConfig()
        return fresh_config.to_dict()

    def reset_config(self):
        """
        恢复出厂设置：将所有参数重置为 DetectionConfig 的默认值

        执行流程：
        1. 新建 DetectionConfig 实例获取最新默认值
        2. 更新内存中的 self.current_config
        3. 调用 save_config() 将默认值写回 JSON 文件，覆盖用户之前的修改

        此操作不可逆，Web 界面上通常会弹出确认对话框。
        """
        fresh_config = DetectionConfig()
        self.current_config = fresh_config.to_dict()
        self.save_config(self.current_config)
    
    def update_param(self, key, value):
        """
        更新单个参数并立即持久化

        仅当 key 存在于当前配置中时才更新，防止引入非法参数名。
        更新成功后立即调用 save_config() 写入 JSON 文件。

        :param key: 参数名（如 "GREEN_ADV"、"SAT_MIN" 等）
        :param value: 新的参数值
        :return: True 表示更新成功，False 表示参数名不存在
        """
        if key not in self.current_config:
            return False
        self.update_params({key: value})
        return True

    def update_params(self, params):
        """
        批量更新多个参数（最后一次性保存）

        与 update_param() 的区别：
        - update_param() 每更新一个参数就写一次磁盘
        - update_params() 先在内存中批量修改所有参数，最后只调用一次 save_config()
        - 适用于 Web 界面"保存所有设置"的场景，减少磁盘 I/O 次数

        只更新当前配置中已存在的参数，忽略未知的 key。

        :param params: 要更新的参数字典，如 {"GREEN_ADV": 35, "SAT_MIN": 40}
        """
        if not isinstance(params, dict):
            raise ValueError("参数必须是 JSON 对象")
        candidate = dict(self.current_config)
        candidate.update(params)
        is_valid, errors = self.validate_params(candidate, require_complete=True)
        if not is_valid:
            raise ValueError("; ".join(errors))
        self.save_config(candidate)
    
    def validate_params(self, params, require_complete=False):
        """
        验证参数的合法性（参数名 + 类型检查）

        验证规则：
        1. 参数名必须存在于 DEFAULT_CONFIG 中，拒绝未知参数（防止拼写错误或注入）
        2. 类型检查以 DEFAULT_CONFIG 中对应参数的类型为基准；整数参数拒绝小数和 bool
        3. 检查有限值、后端范围、奇数核大小和等级阈值关联约束
        4. 可选地要求参数集合完整，用于持久化前的最终校验

        :param params: 待验证的参数字典
        :return: (is_valid, errors) — is_valid 为 True 表示全部通过，
                 errors 为错误信息列表（为空时表示无错误）
        """
        errors = []
        if not isinstance(params, dict):
            return False, ["参数必须是 JSON 对象"]

        # 以 DEFAULT_CONFIG 作为"参数白名单"和类型参照标准
        defaults = DEFAULT_CONFIG.to_dict()
        param_info = self.get_param_info()
        unknown = sorted(set(params) - set(defaults))
        errors.extend(f"未知参数: {key}" for key in unknown)
        if require_complete:
            missing = sorted(set(defaults) - set(params))
            errors.extend(f"缺少参数: {key}" for key in missing)
        
        for key, value in params.items():
            if key not in defaults:
                continue
            
            # bool 是 int 的子类，必须先排除，否则 True/False 会通过 int 检查
            if key == "resolution_adaptive":
                if not isinstance(value, bool):
                    errors.append(f"参数 {key} 必须为布尔值，当前值: {value}")
                continue

            if key == "base_resolution":
                if (
                    not isinstance(value, (list, tuple))
                    or len(value) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
                    or any(item < 16 or item > 4096 for item in value)
                ):
                    errors.append(f"参数 {key} 必须是两个 16-4096 的整数")
                continue

            if isinstance(value, bool):
                errors.append(f"参数 {key} 必须为数字，当前值: {value}")
                continue

            default_value = defaults[key]
            if isinstance(default_value, int):
                if not isinstance(value, int):
                    errors.append(f"参数 {key} 必须为整数，当前值: {value}")
            elif isinstance(default_value, float):
                if not isinstance(value, (int, float)):
                    errors.append(f"参数 {key} 必须为数字，当前值: {value}")

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    errors.append(f"参数 {key} 必须是有限数值，当前值: {value}")
                bounds = param_info.get(key)
                if bounds and not bounds["min"] <= value <= bounds["max"]:
                    errors.append(
                        f"参数 {key} 超出范围 [{bounds['min']}, {bounds['max']}]，当前值: {value}"
                    )

            if key == "MORPH_KERNEL_SIZE" and isinstance(value, int) and value % 2 == 0:
                errors.append(f"参数 {key} 必须为奇数")

        candidate = DEFAULT_CONFIG.to_dict()
        current = getattr(self, "current_config", None)
        if isinstance(current, dict):
            candidate.update(current)
        candidate.update(params)
        self._check_order(
            candidate,
            ("BROWN_H_MIN", "BROWN_H_MAX"),
            "棕色色相范围",
            errors,
        )
        self._check_order(
            candidate,
            ("YELLOW_H_MIN", "YELLOW_H_MAX"),
            "黄色色相范围",
            errors,
        )
        self._check_order(
            candidate,
            ("NOTICE_DISEASE_COUNT", "WARNING_DISEASE_COUNT", "SERIOUS_DISEASE_COUNT"),
            "病斑数量阈值",
            errors,
        )
        self._check_order(
            candidate,
            ("NOTICE_DISEASE_RATIO", "WARNING_DISEASE_RATIO", "SERIOUS_DISEASE_RATIO"),
            "病斑占比阈值",
            errors,
        )
        self._check_order(
            candidate,
            ("NOTICE_PEST_COUNT", "WARNING_PEST_COUNT", "SERIOUS_PEST_COUNT"),
            "虫害数量阈值",
            errors,
        )
        self._check_order(
            candidate,
            ("WARNING_PEST_RATIO", "SERIOUS_PEST_RATIO"),
            "虫害占比阈值",
            errors,
        )
        self._check_order(
            candidate,
            ("GREEN_RATIO_WARNING", "GREEN_RATIO_NOTICE"),
            "绿色占比阈值",
            errors,
        )
        
        return len(errors) == 0, errors
    
    def get_param_info(self):
        """
        获取所有可调参数的 UI 元信息（用于驱动 Web 前端的滑块控件）

        返回一个字典，key 为参数名，value 为描述该参数如何在前端展示的元信息字典：
        - name  : 参数的中文显示名称（如"绿色通道优势阈值"），用于前端标签
        - min   : 滑块的最小值，限制用户可调范围的下限
        - max   : 滑块的最大值，限制用户可调范围的上限
        - step  : 滑块的步进值（如 1 表示整数调节，0.01 表示精确到百分位）
        - unit  : 单位字符串（如"个"），为空时表示无单位（纯数值）

        前端 Web 页面接收到此信息后，会为每个参数动态生成一个带标签的滑块控件，
        用户拖动滑块时参数值实时回传到后端并调用 update_param() 更新。

        参数分为三大类：
        1. 图像分割参数 — 颜色阈值（GREEN_ADV, SAT_MIN 等）和形态学参数（MORPH_KERNEL_SIZE 等）
        2. 病害/虫害 HSV 颜色范围 — 棕色、黄色、虫害区域的色调/饱和度/明度边界
        3. 等级判定标准 — 各等级（注意/警告/严重）对应的病斑数、病斑占比、虫害数、虫害占比、绿色占比

        :return: 参数元信息字典
        """
        # 每个参数的 UI 元信息：显示名称、滑块范围、步进、单位
        info = {
            "GREEN_ADV": {"name": "绿色通道优势阈值", "min": 0, "max": 50, "step": 1, "unit": ""},
            "SAT_MIN": {"name": "饱和度下限", "min": 0, "max": 255, "step": 1, "unit": ""},
            "MORPH_KERNEL_SIZE": {"name": "形态学核大小", "min": 1, "max": 7, "step": 1, "unit": ""},
            "DILATE_ITERATIONS": {"name": "膨胀迭代次数", "min": 0, "max": 5, "step": 1, "unit": ""},
            "BROWN_H_MIN": {"name": "棕色H最小值", "min": 0, "max": 30, "step": 1, "unit": ""},
            "BROWN_H_MAX": {"name": "棕色H最大值", "min": 0, "max": 30, "step": 1, "unit": ""},
            "BROWN_S_MIN": {"name": "棕色S最小值", "min": 0, "max": 255, "step": 5, "unit": ""},
            "YELLOW_H_MIN": {"name": "黄色H最小值", "min": 10, "max": 40, "step": 1, "unit": ""},
            "YELLOW_H_MAX": {"name": "黄色H最大值", "min": 10, "max": 40, "step": 1, "unit": ""},
            "YELLOW_S_MIN": {"name": "黄色S最小值", "min": 0, "max": 255, "step": 5, "unit": ""},
            "PEST_S_MAX": {"name": "虫害S最大值", "min": 0, "max": 100, "step": 1, "unit": ""},
            "PEST_V_MIN": {"name": "虫害V最小值", "min": 150, "max": 255, "step": 1, "unit": ""},
            "NOTICE_DISEASE_COUNT": {"name": "注意级病斑数", "min": 1, "max": 10, "step": 1, "unit": "个"},
            "NOTICE_DISEASE_RATIO": {"name": "注意级病斑占比", "min": 0.01, "max": 0.1, "step": 0.01, "unit": ""},
            "WARNING_DISEASE_COUNT": {"name": "警告级病斑数", "min": 3, "max": 15, "step": 1, "unit": "个"},
            "WARNING_DISEASE_RATIO": {"name": "警告级病斑占比", "min": 0.05, "max": 0.2, "step": 0.01, "unit": ""},
            "SERIOUS_DISEASE_COUNT": {"name": "严重级病斑数", "min": 5, "max": 20, "step": 1, "unit": "个"},
            "SERIOUS_DISEASE_RATIO": {"name": "严重级病斑占比", "min": 0.1, "max": 0.3, "step": 0.01, "unit": ""},
            "NOTICE_PEST_COUNT": {"name": "注意级虫害数", "min": 2, "max": 10, "step": 1, "unit": "个"},
            "WARNING_PEST_COUNT": {"name": "警告级虫害数", "min": 5, "max": 20, "step": 1, "unit": "个"},
            "WARNING_PEST_RATIO": {"name": "警告级虫害占比", "min": 0.01, "max": 0.1, "step": 0.01, "unit": ""},
            "SERIOUS_PEST_COUNT": {"name": "严重级虫害数", "min": 10, "max": 30, "step": 1, "unit": "个"},
            "SERIOUS_PEST_RATIO": {"name": "严重级虫害占比", "min": 0.03, "max": 0.1, "step": 0.01, "unit": ""},
            "GREEN_RATIO_NOTICE": {"name": "注意级绿色占比", "min": 0.1, "max": 0.4, "step": 0.01, "unit": ""},
            "GREEN_RATIO_WARNING": {"name": "警告级绿色占比", "min": 0.05, "max": 0.25, "step": 0.01, "unit": ""},
        }

        return info
