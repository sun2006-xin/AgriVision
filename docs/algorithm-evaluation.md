# AgriVision 算法评估与现场闭环

## 目标和边界

本阶段把“模型分数”变成可复现的证据链，但不把没有标注集的推理结果包装成模型成绩。评估工具不加载模型、不上传图片，只读取公开定义的 JSON 清单和其中的标注、预测、置信度及采集元数据。

当前仓库没有可公开验证的真实标注集，因此示例文件只用于验证格式和计算流程，不能用于宣称 Accuracy、召回率或现场误报率。

## 评估清单格式

严格模式的顶层字段如下：

| 字段 | 要求 | 含义 |
|---|---|---|
| `schema_version` | 必须为 `1` | 清单契约版本 |
| `labels` | 非空且唯一 | 固定类别顺序，决定混淆矩阵行列 |
| `records` | 非空数组 | 每张图或一个独立样本一条记录 |
| `dataset` | 对象 | 数据集名称、版本和标注策略 |

每条记录必须包含：

```json
{
  "id": "sample-001",
  "true": "健康",
  "pred": "健康",
  "confidence": 0.91,
  "probabilities": {"健康": 0.91, "番茄早疫病": 0.09},
  "crop": "番茄",
  "lighting": "daylight",
  "device": "esp32-cam-v1",
  "source": "field",
  "split": "test",
  "group_id": "capture-001"
}
```

`probabilities` 提供完整类别向量时会计算多分类 Brier score；只有 `confidence` 时会明确标记为 `top1` 分数。`field_true` 和 `field_pred` 可以显式描述现场“是否异常”的二值结果；否则通过 `healthy_label` 或 `positive_label` 推导。

`crop`、`lighting`、`device`、`source` 和 `split` 是强制元数据。`disease` 切片直接使用 `true` 标签。相同 `group_id` 不允许跨越 train、validation、test，避免同一采集序列泄漏到多个切分。

## 运行评估

```bash
python system_a/core/evaluate_predictions.py \
  docs/examples/evaluation_manifest.example.json \
  --strict-metadata \
  --healthy-label 健康
```

输出包括：

- Overall Accuracy、宏/加权 Precision、Recall、F1 和固定标签顺序的混淆矩阵。
- `slices.crop`、`slices.disease`、`slices.lighting`、`slices.device`、`slices.source`、`slices.split`，每个切片都带支持数和分类指标。
- `calibration.reliability_bins`、ECE 和 Brier score。ECE 只表示提交的置信度与标签的一致程度，不表示模型已经完成校准。
- `field_error_rates` 中的 FP、FN、误报率、漏报率、现场二值 Precision/Recall；没有现场记录或二值定义时 `available=false`，不会填入伪造的 0。

不带 `--strict-metadata` 仍可读取旧版 `true/pred/confidence` 清单，便于迁移；旧格式的切片会落入 `unknown`，不能作为完整现场评估报告。

## 运行时不确定性和 OOD

System A 的分类和 CLIP 输出现在带有：

- `decision_status`：`known`、`uncertain`、`undetermined` 或 `ood_suspected`。
- `uncertainty_reason`：低置信度、低 top-1 margin、高熵或启发式 OOD 原因。
- `abstain`：是否拒绝把当前结果当作确定诊断；`undetermined` 表示高熵且没有足够类别优势。
- `confidence_semantics`：明确说明这是未经校准的 softmax/相似度分数。

`ood_suspected` 是基于最大分数和归一化熵的拒识启发式，不是经过独立 OOD 数据训练和验证的检测器。上线前应增加已知类验证集、未知作物/背景集，并报告 AUROC、AUPR、FPR95 或业务上约定的拒识指标。

## 视觉引擎职责

| 层 | 职责 | 不应承担的结论 |
|---|---|---|
| HSV 颜色规则 | 快速筛查和低成本预警 | 不能单独证明病害类别 |
| YOLO / 分割 | 目标位置、区域和视觉证据 | 不能代替固定验证集上的类别评估 |
| 分类 / CLIP | 类别候选和辅助分数 | 未校准分数不能当作诊断真值 |
| LLM | 解释已有证据、标明证据来源、生成建议 | 不能改变真值、替代评估或凭空确认病害 |

## System B 多帧融合

System B 为每个摄像头维护独立的 `TemporalFusion`：

1. 最近窗口内的等级按时间衰减加权投票，较新的帧权重更高。
2. 候选等级先进入 pending 状态，连续达到阈值后才成为 stable 等级。
3. 数量和面积比使用同一组权重计算平均值。
4. API 的 `temporal` 字段保留窗口大小、衰减系数、各等级加权票数、候选支持度、待切换计数和最近帧，便于现场回放和误报分析。

这套逻辑用于降低单帧噪声，不等同于真正的时序模型。后续可在固定视频片段上比较单帧、滑动平均和该融合器的延迟、误报、漏报与告警稳定性。

## 建集和发布清单

1. 按作物、病害、光照、设备来源建立固定 test 集，按采集序列而不是随机图片拆分。
2. 记录真实标签来源、标注人员、争议样本和无法判断样本；无法判断样本应进入 abstain/OOD 分析，不要强行归类。
3. 固定模型版本、阈值、类别顺序和推理时间，保留原始预测清单及评估输出。
4. 首先发布基线和切片分布，再设置业务门槛；在真实数据缺失时不填写“现场误报率已达标”。
