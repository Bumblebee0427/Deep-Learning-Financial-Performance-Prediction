# Individual prediction 提交代码

`submission_code.py` 是最终的 individual prediction 流水线。从仓库根目录运行 `python submission_code.py`，读取 `train.csv` 和 `test_individual.csv`，生成 `submission_code_output.csv`。运行需要已安装 NumPy、pandas、scikit-learn 和 LightGBM。`submission_code_for_pdf.txt` 与 Python 文件内容完全相同，可用于排版成 PDF。

模型选择完成后，八个直接模型使用全部 100,000 行训练数据拟合。`Id` 明确排除在预测特征之外。流水线使用 LightGBM、arcsinh 目标变换、毛利润／营业利润／EBITDA 的目标专属滞后特征、营业利润和 EBITDA 的符号加金额建模，以及固定的会计恒等式调和。代码不执行交叉验证、搜索或参数调整。

已将 `submission_code_output.csv` 与既有 `final_submission.csv` 逐项比较：两者均为 **50,000 行、10 列**，列名与顺序相同，`Id` 值与顺序相同；九个目标各自的**最大绝对差和平均绝对差均为 0**。输出没有 NaN／无穷值或重复 `Id`，收入与营业费用均非负。
