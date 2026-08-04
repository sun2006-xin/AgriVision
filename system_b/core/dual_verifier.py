"""
双引擎验证融合器
================
将颜色阈值引擎和 YOLO 引擎的检测结果进行融合，输出统一的告警等级。

融合规则：
  1. 两引擎都检出 → 高置信度，直接采信颜色引擎等级
  2. 仅 YOLO 检出（高置信度 >= 0.6）→ 中等置信度，采信 YOLO 结果
  3. 仅 YOLO 检出（低置信度 < 0.6）→ 低置信度，标记待确认
  4. 仅颜色引擎检出 → 可能是 YOLO 漏检，保留但降一级
  5. 两引擎都未检出 → 正常

使用方式：
    verifier = DualVerifier()
    result = verifier.verify(color_result, yolo_detections)
"""


class DualVerifier:
    """双引擎验证融合器"""

    def __init__(self, yolo_conf_high=0.6, yolo_conf_low=0.25,
                 agreement_boost=1.2, disagree_penalty=0.8):
        """
        初始化融合器参数。

        参数:
            yolo_conf_high: YOLO 高置信度阈值
            yolo_conf_low: YOLO 低置信度阈值（低于此值视为不可靠）
            agreement_boost: 两引擎一致时的等级提升系数（用于占比加权）
            disagree_penalty: 两引擎不一致时的等级降低系数
        """
        self.yolo_conf_high = yolo_conf_high
        self.yolo_conf_low = yolo_conf_low
        self.agreement_boost = agreement_boost
        self.disagree_penalty = disagree_penalty

    def verify(self, color_result, yolo_detections):
        """
        融合两个引擎的检测结果。

        参数:
            color_result: 颜色引擎的检测结果字典，包含:
                - level: 等级名称（'正常'/'注意'/'警告'/'严重'）
                - level_code: 等级代码（0/1/2/3）
                - disease_count: 病斑数量
                - white_count: 虫害数量
                - disease_ratio: 病斑占比
                - white_ratio: 虫害占比
                - green_ratio: 绿色占比
            yolo_detections: YOLO 引擎的检测结果列表，每个元素为:
                - class_id: 0=disease, 1=bug
                - class_name: 'disease' 或 'bug'
                - confidence: 置信度
                - bbox: [x1, y1, x2, y2]

        返回:
            融合结果字典:
            {
                'level': str,              # 最终等级
                'level_code': int,         # 最终等级代码
                'confidence': str,         # 'high'/'medium'/'low'
                'color_result': dict,      # 原始颜色引擎结果
                'yolo_result': dict,       # YOLO 引擎汇总结果
                'agreement': {
                    'disease': str,        # 'agree'/'yolo_only'/'color_only'/'none'
                    'bug': str             # 'agree'/'yolo_only'/'color_only'/'none'
                }
            }
        """
        # 汇总 YOLO 检测结果
        yolo_summary = self._summarize_yolo(yolo_detections)

        # 判断各类型的引擎一致性
        disease_agreement = self._check_agreement(
            color_has=color_result.get('disease_count', 0) > 0,
            yolo_has=yolo_summary['disease_count'] > 0,
            yolo_max_conf=yolo_summary['disease_max_conf']
        )
        bug_agreement = self._check_agreement(
            color_has=color_result.get('white_count', 0) > 0,
            yolo_has=yolo_summary['bug_count'] > 0,
            yolo_max_conf=yolo_summary['bug_max_conf']
        )

        # 融合等级判定
        color_level_code = color_result.get('level_code', 0)
        final_level_code, confidence = self._fuse_level(
            color_level_code, yolo_summary, disease_agreement, bug_agreement
        )

        # 等级代码转名称
        level_names = {0: '正常', 1: '注意', 2: '警告', 3: '严重'}
        final_level = level_names.get(final_level_code, '正常')

        return {
            'level': final_level,
            'level_code': final_level_code,
            'confidence': confidence,
            'color_result': {
                'disease_count': color_result.get('disease_count', 0),
                'white_count': color_result.get('white_count', 0),
                'disease_ratio': color_result.get('disease_ratio', 0.0),
                'white_ratio': color_result.get('white_ratio', 0.0),
                'green_ratio': color_result.get('green_ratio', 0.0),
                'level': color_result.get('level', '正常'),
            },
            'yolo_result': yolo_summary,
            'agreement': {
                'disease': disease_agreement,
                'bug': bug_agreement,
            }
        }

    def _summarize_yolo(self, yolo_detections):
        """
        汇总 YOLO 检测结果为统计摘要。

        返回:
            {
                'disease_count': int,
                'bug_count': int,
                'disease_max_conf': float,
                'bug_max_conf': float,
                'total_count': int,
                'inference_ms': float (如果有的话)
            }
        """
        disease_confs = []
        bug_confs = []

        for det in yolo_detections:
            if det['class_id'] == 0:
                disease_confs.append(det['confidence'])
            elif det['class_id'] == 1:
                bug_confs.append(det['confidence'])

        return {
            'disease_count': len(disease_confs),
            'bug_count': len(bug_confs),
            'disease_max_conf': max(disease_confs) if disease_confs else 0.0,
            'bug_max_conf': max(bug_confs) if bug_confs else 0.0,
            'total_count': len(yolo_detections),
        }

    def _check_agreement(self, color_has, yolo_has, yolo_max_conf):
        """
        判断两个引擎对某一类目标的一致性。

        返回:
            'agree': 两引擎都检出
            'yolo_only': 仅 YOLO 检出
            'color_only': 仅颜色引擎检出
            'none': 两引擎都未检出
        """
        if color_has and yolo_has:
            return 'agree'
        elif yolo_has and not color_has:
            return 'yolo_only'
        elif color_has and not yolo_has:
            return 'color_only'
        else:
            return 'none'

    def _fuse_level(self, color_level_code, yolo_summary, disease_agreement, bug_agreement):
        """
        根据两引擎的一致性融合等级。

        返回:
            (final_level_code, confidence)
        """
        # 计算 YOLO 综合置信度
        yolo_max_conf = max(yolo_summary['disease_max_conf'], yolo_summary['bug_max_conf'])
        yolo_total = yolo_summary['disease_count'] + yolo_summary['bug_count']

        # 情况1: 两引擎都未检出 → 正常
        if color_level_code == 0 and yolo_total == 0:
            return 0, 'high'

        # 情况2: 两引擎都检出（至少一个类型 agree）
        has_agree = disease_agreement == 'agree' or bug_agreement == 'agree'
        if has_agree:
            # 两引擎一致，高置信度，直接采信颜色引擎等级
            return color_level_code, 'high'

        # 情况3: 仅 YOLO 检出
        yolo_only = (disease_agreement == 'yolo_only' or bug_agreement == 'yolo_only')
        color_none = color_level_code == 0
        if yolo_only and color_none:
            if yolo_max_conf >= self.yolo_conf_high:
                # YOLO 高置信度，颜色引擎未检出 → 中等置信度，定为"注意"
                return 1, 'medium'
            elif yolo_max_conf >= self.yolo_conf_low:
                # YOLO 低置信度 → 低置信度，记录但不上报
                return 0, 'low'
            else:
                return 0, 'low'

        # 情况4: 仅颜色引擎检出
        color_only = (disease_agreement == 'color_only' or bug_agreement == 'color_only')
        if color_only and yolo_total == 0:
            # 颜色引擎检出但 YOLO 未检出 → 降一级（可能是 YOLO 漏检）
            reduced_level = max(0, color_level_code - 1)
            return reduced_level, 'low'

        # 情况5: 混合情况（部分 agree，部分不一致）
        # 取两引擎等级的较大值，但降低置信度
        yolo_implied_level = 0
        if yolo_summary['disease_count'] > 0 or yolo_summary['bug_count'] > 0:
            yolo_implied_level = 1  # YOLO 检出至少为"注意"

        final_level = max(color_level_code, yolo_implied_level)
        return final_level, 'medium'

    def update_config(self, **kwargs):
        """动态更新融合参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, float(value))
                print(f"[DualVerifier] {key} 更新为: {value}")
