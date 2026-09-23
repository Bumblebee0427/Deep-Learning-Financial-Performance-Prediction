# 5 折 OOF 与逐目标方案（仅训练集）

本轮只读取 `train.csv`。固定 `KFold(n_splits=5, shuffle=True, random_state=42)`，每行恰好预测一次；每折的缺失填补与 one-hot 编码只在该折训练部分拟合。沿用原有 `score_predictions()`，`Id` 仅用于对齐。没有运行 Ridge，也没有读取或预测测试集。

复现顺序：`python oof_models.py --all`、`python hard_target_oof.py --all`、`python analyze_oof.py --all`。本次环境为 NumPy 2.4.2、pandas 2.3.3、scikit-learn 1.8.0、LightGBM 4.6.0、PyArrow 16.1.0。`fold_assignments.parquet` 固定行、`Id` 与折号；三份基础模型及三份难目标 OOF 文件保留原始行序。

## 1. 基础模型与季度滞后基线

下表均为 OOF sMAPE（%）。列顺序为资产、负债、股东权益、毛利、营收成本、营收、营业利润、营业费用、EBITDA；完整目标列名见 [`base_model_scores.csv`](base_model_scores.csv) 和 [`lag_scores.csv`](lag_scores.csv)。

| 方法 | 资产 | 负债 | 权益 | 毛利 | 成本 | 营收 | 营业利润 | 营业费用 | EBITDA | 9 目标平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 14.99 | 16.70 | 21.45 | 35.74 | 22.90 | 19.74 | 59.49 | 12.49 | 60.20 | **29.30** |
| HistGradientBoosting | 18.07 | 19.87 | 24.08 | 36.47 | 24.83 | 21.21 | 60.76 | 14.83 | 61.45 | 31.29 |
| RandomForest | **4.33** | **9.90** | **10.44** | 39.18 | 39.90 | 27.11 | 68.33 | 25.18 | 72.04 | 32.94 |
| Q1 延续 | 37.30 | 10.05 | 75.55 | **30.30** | **18.05** | **18.00** | 105.91 | 31.13 | 100.09 | 47.38 |
| Q1–Q2 均值 | 45.34 | 20.20 | 80.67 | 30.96 | 20.69 | 21.26 | 104.15 | 31.12 | 98.45 | 50.32 |
| 0.60Q1+0.25Q2+0.10Q3+0.05Q4 | 73.85 | 43.63 | 106.26 | 31.29 | 21.26 | 22.76 | 102.96 | **30.48** | 97.68 | 58.91 |
| Q1–Q4 均值 | 141.98 | 117.74 | 159.63 | 33.07 | 26.63 | 28.88 | **101.94** | 31.78 | **96.37** | 82.00 |

单模型的五折训练时间合计：LightGBM 89.0 秒、HistGradientBoosting 181.7 秒、RandomForest 505.3 秒。Q1 对毛利、成本和营收有竞争力，但不能作为全部目标的统一方案。

## 2. 逐目标 RF/LightGBM 混合与直接方法

预设随机森林权重为 0、0.25、0.50、0.75、1；其余权重给 LightGBM。再比较 HistGradientBoosting、四种滞后法、最佳 RF/LightGBM 混合与滞后法的 0.25/0.50/0.75 混合，以及三个难目标的新增 LightGBM 方案。所有选择只用同一批 OOF 预测。逐候选结果在 [`direct_candidates.csv`](direct_candidates.csv)。

| 目标 | 最佳 RF 权重 | 最佳 RF/LGB sMAPE | 最佳直接方法 | 直接 sMAPE |
| --- | ---: | ---: | --- | ---: |
| TOTAL_ASSETS | 1.00 | 4.33 | RandomForest | 4.33 |
| TOTAL_LIABILITIES | 1.00 | 9.90 | 0.75 RF + 0.25 Q1 | 9.82 |
| TOTAL_STOCKHOLDERS_EQUITY | 1.00 | 10.44 | RandomForest | 10.44 |
| GROSS_PROFIT | 0.25 | 35.27 | LightGBM + 滞后特征 + arcsinh 目标 | 23.14 |
| COST_OF_REVENUES | 0.00 | 22.90 | Q1 延续 | 18.05 |
| REVENUES | 0.00 | 19.74 | 0.25 LightGBM + 0.75 Q1 | 17.67 |
| OPERATING_INCOME | 0.25 | 58.95 | LightGBM + 滞后特征 + arcsinh 目标 | 51.84 |
| OPERATING_EXPENSES | 0.00 | 12.49 | LightGBM | 12.49 |
| EBITDA | 0.25 | 59.93 | LightGBM + 滞后特征 + arcsinh 目标 | 50.36 |

这些直接方法的平均 OOF sMAPE 为 **22.02%**。

## 3. 会计恒等式

训练目标本身的检查结果：

| 恒等式 | 两侧之间的 sMAPE |
| --- | ---: |
| 资产 = 负债 + 权益 | 约 0（3.34×10⁻¹⁵%） |
| 营收 ≈ 毛利 + 营收成本 | 0.80% |
| 营业利润 = 毛利 − 营业费用 | 约 0（3.38×10⁻¹⁴%） |

先对各基础模型分别比较直接、纯派生、以及直接权重 0/0.25/0.50/0.75/1 的混合，详见 [`accounting_model_candidates.csv`](accounting_model_candidates.csv)。再以第 2 节选择的**直接 OOF 预测**作为各公式的输入；下面的派生值始终从这组直接预测计算，不递归使用最终混合结果。

| 目标 | 直接 sMAPE | 纯派生 sMAPE | 最终选择 | 最终 sMAPE |
| --- | ---: | ---: | --- | ---: |
| 资产 | 4.33 | 4.87（负债+权益） | 保留直接预测 | 4.33 |
| 负债 | 9.82 | 9.90（资产−权益） | 0.50 直接 + 0.50 派生 | 9.66 |
| 权益 | 10.44 | 10.28（资产−负债） | 纯派生 | 10.28 |
| 毛利 | 23.14 | 31.91（营收−成本）；34.56（营业利润+费用） | 保留直接预测 | 23.14 |
| 营收成本 | 18.05 | 29.51（营收−毛利） | 保留直接预测 | 18.05 |
| 营收 | 17.67 | 13.81（毛利+成本） | 纯派生 | 13.81 |
| 营业利润 | 51.84 | 52.34（毛利−费用） | 0.50 直接 + 0.50 派生 | 50.59 |
| 营业费用 | 12.49 | 11.42（毛利−营业利润） | 0.25 直接 + 0.75 派生 | 11.04 |
| EBITDA | 50.36 | 无给定恒等式 | 保留直接预测 | 50.36 |

逐权重评分在 [`accounting_candidates.csv`](accounting_candidates.csv)。

## 4. 难目标：LightGBM 原始目标与 arcsinh 目标

在三个目标上加入 Q1 值、Q1−Q2 变化、Q1/Q2 各自符号和同号标志、Q1–Q4 均值/中位数/标准差/向近期斜率、Q1–Q10 均值/向近期斜率。两种新模型使用**完全相同**的折、特征和 LightGBM 参数；差别只有是否用训练折的 `median(abs(y))` 缩放后做 `arcsinh(y/scale)`，并在预测后用 `sinh` 逆变换。

| 目标 | 基础原始目标 | 加特征后原始目标 | 加特征后 arcsinh 目标 | 符号准确率：基础 → arcsinh |
| --- | ---: | ---: | ---: | ---: |
| OPERATING_INCOME | 59.49 | 59.00 | **51.84** | 91.55% → 92.66% |
| EBITDA | 60.20 | 59.21 | **50.36** | 91.64% → 92.83% |
| GROSS_PROFIT | 35.74 | 35.88 | **23.14** | — |

三个目标的 arcsinh 方案在**每一折**均优于同特征的原始目标方案，逐折数值见 [`hard_fold_scores.csv`](hard_fold_scores.csv)。因此本轮的主要改善来自目标变换，而单独加入这些滞后特征作用较小。

## 5. 最终逐目标配方与 OOF 分数

定义 `D_目标` 为第 2 节所选的**直接**预测。最终配方中的派生项只引用 `D_目标`，从而避免公式循环。

| 目标 | 精确配方 | OOF sMAPE |
| --- | --- | ---: |
| TOTAL_ASSETS | `D_A = RF`；最终 `D_A` | 4.33 |
| TOTAL_LIABILITIES | `D_L = 0.75 RF + 0.25 Q1`；最终 `0.50 D_L + 0.50 (D_A − D_E)` | 9.66 |
| TOTAL_STOCKHOLDERS_EQUITY | `D_E = RF`；最终 `D_A − D_L` | 10.28 |
| GROSS_PROFIT | `D_GP = LightGBM(arcsinh 目标, 新增滞后特征)`；最终 `D_GP` | 23.14 |
| COST_OF_REVENUES | `D_C = Q1_COST_OF_REVENUES`；最终 `D_C` | 18.05 |
| REVENUES | `D_R = 0.25 LightGBM + 0.75 Q1`；最终 `max(0, D_GP + D_C)` | 13.81 |
| OPERATING_INCOME | `D_OI = LightGBM(arcsinh 目标, 新增滞后特征)`；最终 `0.50 D_OI + 0.50 (D_GP − D_OE)` | 50.59 |
| OPERATING_EXPENSES | `D_OE = LightGBM`；最终 `max(0, 0.25 D_OE + 0.75 (D_GP − D_OI))` | 11.04 |
| EBITDA | `D_EBITDA = LightGBM(arcsinh 目标, 新增滞后特征)`；最终 `D_EBITDA` | 50.36 |

**最终平均 OOF sMAPE：21.2509%**，比最佳统一基础模型 LightGBM 的 29.3010% 低 27.47%（相对降低）。最终方案五个折的平均 sMAPE 为 21.07%–21.44%。详细机器可读配方见 [`target_recipes.csv`](target_recipes.csv)，逐行最终 OOF 预测见 [`selected_final_oof.parquet`](selected_final_oof.parquet)。

训练集中营收与营业费用均非负，因此最终对这两个目标加零下界。纯会计派生营收原本产生 376 个负预测；置零后没有改变这些行的 sMAPE（正真实值对应负预测或零预测都为 200%），并去除了明显不合理的负营收。营业费用本轮没有触发置零。

**解释边界：** 基础模型的每行预测是严格 OOF；但目标方法和混合权重是看过同一批 OOF 标签后选出的，因此 21.2509% 是**选择后的 OOF 分数**，可能偏乐观，不能当作完全独立的最终泛化估计。数据还没有明确公司分组，Q0 同季度元数据的实际可用时间也需核实。下一步可固定上述候选方案，再做独立验证或嵌套选择；本轮没有生成测试集提交。
