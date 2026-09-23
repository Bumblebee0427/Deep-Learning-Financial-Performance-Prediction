# Individual prediction：数据审计与验证基线

运行：

```bash
python evaluation_framework.py
```

依赖：Python、NumPy、pandas、scikit-learn。脚本只使用 `train.csv` 的标签拟合与评分，读取 `test_individual.csv` 仅为完成题目要求的结构与缺失值检查，不生成测试集预测。

## 数据结构

| 数据 | 行 | 列 | 说明 |
| --- | ---: | ---: | --- |
| train.csv | 100,000 | 212 | 9 个 Q0 目标、202 个候选预测列、`Id` |
| test_individual.csv | 50,000 | 203 | 与训练集同名同序的 202 个候选预测列、`Id` |

202 个候选预测列中，198 个是 `float64`，4 个是 `object` 类别列：`industry`、`sector`、`financialCurrency`、`recommendationKey`。9 个目标均为 `float64`，`Id` 是 `int64`，并从预测列中明确剔除。逐列名称、类型、缺失数、缺失比例、占位值数和唯一值数见 [`audit_outputs/train_columns.csv`](audit_outputs/train_columns.csv) 与 [`audit_outputs/test_columns.csv`](audit_outputs/test_columns.csv)。

## 缺失与异常

- 两份 CSV 均没有 pandas 识别的空值，也没有数值型的 `inf` 或 `-inf`。9 个训练标签全部非空。
- 类别列却包含字面量占位值。训练集 `industry` 和 `sector` 各 1,344 行，`financialCurrency` 和 `recommendationKey` 各 1,232 行；测试集相应为 682、682、623、623 行。后续模型应把它们作为类别状态处理，不可因 `isna()` 全为零而忽略。
- 数据字典称 `Q0_fiscal_year_end` 至 `Q10_fiscal_year_end` 是标志，但训练集 11 列共 12,861 个值既不接近 0 也不接近 1。逐列数量在 [`audit_outputs/fiscal_flags.csv`](audit_outputs/fiscal_flags.csv)。后续建模前须决定保留连续值还是按业务意义离散化；本基线不用特征，因此不作选择。
- 目标有强烈长尾和不合常规的符号。例如 `Q0_TOTAL_ASSETS` 的最小值约为 -2.18e11、最大值约为 3.85e11；`Q0_OPERATING_INCOME` 有 79,542 个负值。所有目标的最小值、1% 分位数、中位数、均值、99% 分位数、最大值及负值计数见 [`audit_outputs/target_distributions.csv`](audit_outputs/target_distributions.csv)。对这些合成数据，不应擅自裁剪负值或极端值。

## 评分与划分

按 rubric，对每个目标计算 `100 * mean(abs(y - yhat) / (0.5 * (abs(y) + abs(yhat))))`，然后对 9 个目标的 sMAPE 等权求平均。rubric 没有定义 `y=yhat=0` 时的 `0/0`；代码将该项记为 0，其余完全按公式计算。评分要求标签与预测具有相同索引和 9 个目标列，且均为有限值。

训练集按行打乱，`random_state=42`，其中 80,000 行训练、20,000 行验证。固定的行级划分及 `Id` 可见 [`audit_outputs/split_ids.csv`](audit_outputs/split_ids.csv)。各目标的常数预测只取训练部分的目标均值。

| 目标 | 验证 sMAPE (%) |
| --- | ---: |
| Q0_TOTAL_ASSETS | 137.6763 |
| Q0_TOTAL_LIABILITIES | 141.1114 |
| Q0_TOTAL_STOCKHOLDERS_EQUITY | 135.5391 |
| Q0_GROSS_PROFIT | 138.2339 |
| Q0_COST_OF_REVENUES | 144.3980 |
| Q0_REVENUES | 140.8092 |
| Q0_OPERATING_INCOME | 128.2923 |
| Q0_OPERATING_EXPENSES | 127.7104 |
| Q0_EBITDA | 131.5961 |
| **9 目标平均** | **136.1519** |

完整精度及训练部分的均值见 [`audit_outputs/baseline_scores.csv`](audit_outputs/baseline_scores.csv)。

## 验证假设与泄漏检查

数据字典将所有 Q0 定位于同一最新季度（Q3 2023），Q1–Q10 是该季度之前的历史记录；数据没有明确的公司标识或逐行日期。训练集没有完全相同的预测列行，也没有完全相同的 Q1–Q10 历史行。因此，随机划分是目前可执行的初始验证方式，但不能证明它衡量了“全新公司”泛化：若合成记录共享同一基础公司而特征略有扰动，训练与验证仍可能有关联。取得公司分组信息后应改做分组验证；若目标是跨季度部署，应使用时间后推验证。

`Id` 在训练与测试中各自唯一且两边没有重复，但 rubric 明确禁止用作特征。更值得核实的是来自 Q0 的同季度元数据：`totalRevenue` 与 `Q0_REVENUES` 在训练集的 Pearson 相关系数为 0.9655，虽没有完全相等的行，但可能反映目标季度的已知财务信息。`ebitda`、`totalCash`、`totalDebt`、`freeCashflow`、`operatingCashflow`、`revenuePerShare` 等也属于同季度元数据。rubric 允许除 `Id` 外的列作为候选特征，但在建立后续模型前，应按实际预测时点确认这些字段是否可获得。训练集同季度元数据的初步数值检查见 [`audit_outputs/same_quarter_metadata.csv`](audit_outputs/same_quarter_metadata.csv)。

后续模型可以复用 `predictor_columns()`、`split_training_data()`、`score_predictions()`。所有拟合、编码、填补与调参都应仅在训练部分完成，然后对验证部分评分；本脚本没有在测试集上选择任何模型或参数。
