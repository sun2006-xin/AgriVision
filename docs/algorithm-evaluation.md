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
| `model` | 严格证据模式必需 | 模型名称、版本和权重 SHA-256 |

每条记录必须包含：

```json
{
  "id": "sample-001",
  "image": "images/sample-001.jpg",
  "image_sha256": "...64 位小写十六进制...",
  "true": "健康",
  "pred": "健康",
  "confidence": 0.91,
  "probabilities": {"健康": 0.91, "番茄早疫病": 0.09},
  "crop": "番茄",
  "lighting": "daylight",
  "device": "esp32-cam-v1",
  "source": "field",
  "split": "test",
  "group_id": "capture-001",
  "annotation_status": "verified",
  "annotation_source": "two-reviewer-adjudication"
}
```

`probabilities` 提供完整类别向量时会计算多分类 Brier score；只有 `confidence` 时会明确标记为 `top1` 分数。`field_true` 和 `field_pred` 可以显式描述现场“是否异常”的二值结果；否则通过 `healthy_label` 或 `positive_label` 推导。

`crop`、`lighting`、`device`、`source` 和 `split` 是强制元数据。`disease` 切片直接使用 `true` 标签。相同 `group_id` 不允许跨越 train、validation、test，避免同一采集序列泄漏到多个切分。

### 可复现证据模式

只使用 `--strict-metadata` 时，清单仍兼容早期格式；要把结果作为可复核评估的候选报告，使用 `--strict-provenance`。该模式额外要求：

- `dataset.name`、`dataset.version`、`dataset.label_policy`，以及顶层 `model.name`、`model.version`、`model.weights_sha256`。
- 每条记录的安全相对路径 `image`、小写 `image_sha256`、`group_id`、`annotation_status` 和 `annotation_source`。
- `split` 只能是 `train`、`validation` 或 `test`；同一内容哈希只能出现一次，同一采集组不能跨切分。
- `annotation_status` 只能是 `verified` 或 `adjudicated`。无法判断/争议样本不能混入主分类指标清单，应另行进入 abstain/OOD 分析。

结构门禁不读取图片；它只证明清单字段完整。要证明图片确实存在且没有被替换，另行执行文件审计：

```bash
python tools/audit_evaluation_dataset.py path/to/manifest.json \
  --image-root path/to/dataset --check-files \
  --output evaluation-audit.json
```

审计器会输出按作物、病害、光照、设备、来源和切分的样本覆盖，以及缺失文件、非普通文件、超过 10 MB 文件和 SHA-256 不匹配列表。生成器在初次审计后还会对实际送入模型的字节再次验哈希，避免文件竞态造成“旧哈希、新内容”。结构错误退出码为 `2`，文件审计失败为 `1`，全部通过为 `0`。

## 运行评估

```bash
python system_a/core/evaluate_predictions.py \
  docs/examples/evaluation_manifest.example.json \
  --strict-provenance \
  --healthy-label 健康
```

拿到真实图片后，在上面的命令中追加 `--check-files --image-root path/to/dataset`，评估会在计算指标前阻止缺图或错哈希输入。

### 从真实图片生成预测清单

标注清单可以先省略 `pred`、`confidence` 和 `probabilities`，保留 `true` 及全部 provenance 字段。当前 System A 分类器的离线 runner 会复用线上相同的 224×224/ImageNet 预处理、12 类顺序和 softmax 逻辑：

```bash
python tools/generate_prediction_manifest.py \
  path/to/labeled-manifest.json \
  --image-root path/to/dataset \
  --model system_a/models/best_model.onnx \
  --output predictions.json

python system_a/core/evaluate_predictions.py predictions.json \
  --strict-metadata --strict-provenance \
  --check-files --image-root path/to/dataset \
  --healthy-label 健康
```

runner 会先比较清单中的 `model.weights_sha256` 与实际模型文件，再逐张图片校验 SHA-256 后推理；类别数量或顺序不匹配会直接失败。当前内置分类器的类别顺序见 `classifier_inference.py`，自定义模型必须显式扩展该契约，不能静默复用错误映射。

输出包括：

- Overall Accuracy、宏/加权 Precision、Recall、F1 和固定标签顺序的混淆矩阵。
- `slices.crop`、`slices.disease`、`slices.lighting`、`slices.device`、`slices.source`、`slices.split`，每个切片都带支持数和分类指标。
- `robustness.lighting` 和 `robustness.device` 汇总每个切片的支持数、Accuracy、宏 F1、最差切片及指标 gap；这是非配对切片比较，不等同于同一对象的跨光照/跨设备实验。
- `calibration.reliability_bins`、ECE 和 Brier score。ECE 只表示提交的置信度与标签的一致程度，不表示模型已经完成校准。
- `field_error_rates` 中的 FP、FN、误报率、漏报率、现场二值 Precision/Recall；没有现场记录或二值定义时 `available=false`，不会填入伪造的 0。
- `field_threshold_curve` 在有完整概率向量时输出多个运营阈值下的 FP/FN/FPR/FNR；可用 `--field-thresholds 0.35,0.55,0.75` 覆盖默认阈值，不能把单一阈值结果当成业务验收。

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

1. 按作物、病害、光照、设备来源建立固定 test 集，按采集序列而不是随机图片拆分；为每个样本记录相对路径和 SHA-256，并先运行数据审计。
2. 记录真实标签来源、标注人员、争议样本和无法判断样本；无法判断样本应进入 abstain/OOD 分析，不要强行归类。
3. 固定模型版本、阈值、类别顺序和推理时间，保留原始预测清单及评估输出。
4. 首先发布基线和切片分布，再设置业务门槛；在真实数据缺失时不填写“现场误报率已达标”。示例清单中的路径、哈希和模型版本是占位值，不代表真实数据或模型性能。
