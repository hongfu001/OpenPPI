# prediction-inference

面向 prediction-assisted inference 的 Python 算法库。目前实现 **ASI、RASI（稳健主动推断）与独立两样本 PPI 的二元均值估计及终点 Wald 区间**。每种方法继承共同的 `Method`，统一检查输入并返回 `InferenceResult`。

## 安装与运行

```sh
python -m pip install -e '.[simulation,test]'
PYTHONPATH=src:. python examples/asi_mean.py
PYTHONPATH=src:. python examples/ppi_mean.py
PYTHONPATH=src:. python examples/rasi_mean.py
PYTHONPATH=src:. python main.py --reps 100
python -m pytest
```

仿真在源码仓库中运行，不属于核心安装包。核心仅依赖 NumPy/SciPy，实验额外需要 pandas 和 scikit-learn。

## 目录与职责

```text
src/prediction_inference/
├── __init__.py       # 对外接口
├── base.py           # Method、InferenceResult
├── data.py           # ActiveData、PPIData、RobustActiveData
├── targets.py        # Target、Mean；唯一的抽样分数定义
├── _stats.py         # 私有公式：ipw_correct、wald_interval
├── sampling/          # 固定预算、固定模型的抽样：每个方法一个文件
│   ├── __init__.py
│   ├── base.py        # 公共部分：SamplingDesign 接口、LabelRequest、select_for_labeling
│   ├── uniform.py     # Uniform：均匀抽样（PPI 基线）
│   ├── asi.py         # ASI：Uncertainty、tune_tau
│   └── rasi.py        # RASI：RobustDesign、BurnIn、select_burn_in、路径与稳健 ρ、交叉验证选 c、robust_active_label
└── methods/
    ├── __init__.py
    ├── asi.py        # ASI(Method)
    ├── rasi.py       # RASI(ASI)
    └── ppi.py        # PPI(Method)
```

- `dataset/judge_sim.py`：DGP 和返回 DataFrame 的 `make_judge_data`。
- `model/learners.py`：预测模型封装；`model/judge.py`：评委均值模型。
- `common/`：配置、通用工具（种子、输出、汇总、PPI 切分）和核心测试。
- `examples/asi_mean.py`、`examples/ppi_mean.py`、`examples/rasi_mean.py`：单次运行示例。
- `examples/simulations/`：仿真策略与重复实验入口；`online.py` 是序贯版 ASI（仅仿真使用，见下文）；`main.py` 保留兼容命令行入口。

依赖方向：examples 可使用所有层；model、dataset 不依赖算法或示例；算法独立于 model、dataset、common。

## 方法接口

```python
from prediction_inference import ASI, PPI, RASI, Mean

# record 是 ActiveData，data 是 PPIData，robust 是 RobustActiveData。
# result = ASI()(record, target=Mean(), alpha=.05)
# result = RASI()(robust, target=Mean(), alpha=.05)
# result = PPI()(data, target=Mean(), alpha=.05)
```

`Method` 位于 `prediction_inference.base`，负责检查数据类型、目标类型和 alpha，再调用子类的 `_estimate(data, target)`。子类返回 `(point, se, diagnostics)`，可在 diagnostics 中放入 `values`。主类将这些值交给区间函数，并从最终诊断信息中移除 `values`。

`InferenceResult` 保留 `estimate`、`standard_error`、`confidence_interval`、`alpha`、`method`、`diagnostics` 六个字段。

默认使用私有的 Wald 公式。自定义区间函数的签名为 `interval(point, se, alpha, values=None)`，通过 `ASI(interval=...)` 或 `PPI(interval=...)` 传入。ASI 的 values 是校正值数组；PPI 的是包含 `residuals` 与 `unlabeled_predictions` 两个数组的字典。

旧的 `asi`、`asi_mean`、`ppi`、估计器与区间类接口已删除。`_stats.py` 仅供库内部使用。

## 固定预算抽样

核心库只考虑固定模型、固定预算：先用模型预测，再一次性决定抽样概率，选出待标样本。三种抽样设计各占一个文件，都继承 `SamplingDesign`，通过 `calibrate(reference_scores, rho)` 校准，`probability(scores)` 返回标注前就确定的概率：

| 设计 | 文件 | 行为 |
| --- | --- | --- |
| `Uniform()` | `uniform.py` | 常数概率 `rho = budget / n`；配合 ASI 估计器即 PPI 基线 |
| `Uncertainty(tau=.5)` | `asi.py` | `pi = (1-tau) * clip(eta*u, 0, 1) + tau*rho`，`eta = rho / mean(u)`，u 默认为 `2*min(p, 1-p)`（ASI 论文算法 1）；下限 `tau * rho`，截断可能使期望标注量略低于预算 |
| `RobustDesign` | `rasi.py` | RASI 第二批：主动规则 π ∝ 预测误差，沿预算保持路径向均匀插值（见下文 RASI 一节） |

`Mean.sampling_scores` 是 `2*min(p, 1-p)` 的唯一定义位置。

一批选点、标注、组装记录的流程三种设计通用（完整示例见 `examples/asi_mean.py`）：

```python
from prediction_inference import ASI, Mean, Uncertainty, select_for_labeling

predictions = model.predict(features)          # 任意已训练好的固定模型
request = select_for_labeling(predictions, target=Mean(), design=Uncertainty(tau=.5),
                              budget=400, random_state=42)
labels = annotate(request.indices)             # 导出行号交给标注员，按同样顺序返回标签
record = request.with_labels(labels)           # -> ActiveData
result = ASI()(record, target=Mean(), alpha=.05)
```

`select_for_labeling` 不读取任何标签，按本批分数以 `rho = budget / n` 校准设计副本，返回 `LabelRequest`：全部 n 行的 `predictions`、`propensities` 与布尔 `queried`（复制并只读），以及已校准的设计副本 `design`（仅供诊断，不含标签）。可选参数 `scores=` 替换默认的 `target.sampling_scores`，例如 RASI 第二批传入预测误差。`request.indices` 是待标行号；`request.with_labels(labels)` 要求标签与 `indices` 一一对应，否则报错，返回 `ActiveData`。`ActiveData` 的预测、已观察标签、查询标记和查询概率一一对应，未查询标签必须为 NaN；构造时复制并保护数组。

有独立历史标注数据时，可按 ASI 论文式 (12) 选 τ：`tune_tau(historical_predictions, historical_labels, target=Mean(), rho=budget/n)`，在 `np.linspace(.01, .99, 100)` 上最小化 `mean((Y-f)^2 / pi_tau)`。历史预测应来自未用这些标签训练的模型（例如训练集的留出部分）。

## 独立两样本 PPI（四步）

```python
from sklearn.ensemble import RandomForestRegressor
from common.utils import split
from dataset.judge_sim import make_judge_data
from prediction_inference import PPI, Mean, PPIData

# 1. 数据、随机切分和独立训练
frame = make_judge_data(n=2000, seed=0)
train, labeled, unlabeled = split(
    frame.drop(columns='label').to_numpy(), frame['label'].to_numpy(),
    train_size=400, labeled_size=400, random_state=42)
model = RandomForestRegressor(n_estimators=60, min_samples_leaf=8,
                              max_depth=8, random_state=0, n_jobs=1).fit(*train)
data = PPIData(labels=labeled[1],
               labeled_predictions=model.predict(labeled[0]),
               unlabeled_predictions=model.predict(unlabeled))
# 2. 目标
target = Mean()
# 3. 推断
result = PPI()(data, target=target, alpha=.05)
# 4. 输出
print(result.estimate, result.confidence_interval, result.diagnostics)
```

`split` 返回 `(X_train, y_train)`、`(X_labeled, y_labeled)`、`X_unlabeled`，大小参数是行数，剩余行用于未标注样本，各部分至少两行。它不使用标签分层，也不返回未标注标签。

数据行需要来自 iid 总体，三个部分互不重叠。预测模型仅用独立训练集拟合，推断前固定。`PPIData` 要求标注标签和预测对齐，两个推断样本各至少两行，所有数组复制并设为只读。目前真实标签仅支持二元值。

## 稳健主动推断 RASI（两批）

实现 Li, Zrnic & Candès（NeurIPS 2025）的算法 1，参照官方代码 `lphLeo/Robust-Active-Statistical-Inference`。RASI 是两批标注，第二批与 ASI 共用 `select_for_labeling` 接口（完整示例见 `examples/rasi_mean.py`）：

```python
import numpy as np
from prediction_inference import RASI, Mean, RobustDesign, select_burn_in, select_for_labeling

rng = np.random.default_rng(42)                     # 一个生成器贯穿两批，保证两批抽样独立
predictions = model.predict(pool)                   # 独立训练、固定不变的模型
# 批 1：均匀随机 burn-in，全部标注（概率 1，计入预算）
burn = select_burn_in(len(pool), size=100, random_state=rng)
burn_labels = annotate(burn.indices)
# 两批之间：在 burn-in 上拟合 e(x) ≈ |Y - f(x)|；constraint=None 时 K 折交叉验证选 ℓ2 半径 c
rest = burn.remaining
design = RobustDesign.fit(pool[burn.indices], burn_labels, predictions[burn.indices],
                          error_model_factory=make_regressor, rho=300 / len(rest),
                          path='geometric', random_state=rng.integers(2**32))
# 批 2：剩余行稳健抽样
request = select_for_labeling(predictions[rest], target=Mean(), design=design, budget=300,
                              scores=design.errors(pool[rest]), random_state=rng)
labels = annotate(rest[request.indices])            # 池中行号是 rest[request.indices]
# 合并两批 -> RobustActiveData -> RASI
record = burn.combine(predictions, burn_labels, request, labels)
result = RASI()(record, target=Mean(), alpha=.05)
```

- `select_burn_in(n, size=...)` 返回 `BurnIn`：`indices` 是 burn-in 行号，`remaining` 是进入第二批的行号；不读取标签。
- `RobustDesign` 是普通的 `SamplingDesign`。`fit` 只在两批之间使用 burn-in 标签；`calibrate` 只读取无标签的分数 `design.errors(...)`：主动规则 `pi = clip(eta * u, 0, 1)`，`eta = rho / mean(u)`，u 为预测误差；再沿预算保持路径（`geometric` 默认，另有 `linear`、`hellinger`）网格搜索 ρ，最小化 `mean(e^2/pi) + c*sqrt(mean(1/pi^2))`，即 ℓ2 约束下极小极大问题的闭式内层，与官方 `MinMaxOptimizer_l2` 目标一致。
- `BurnIn.combine` 把两批合并为 `RobustActiveData`（位于 `data.py`，继承 `ActiveData`，额外记录 `burn_in`、`rho_robust`、`constraint`、`path`）。burn-in 行的倾向得分为 1，因此 `RASI` 直接复用 ASI 的 IPW 校正，点估计等于官方"burn-in 标签与剩余 AIPW 值拼接后取均值"。
- 便捷函数 `robust_active_label(pool, oracle=..., target=Mean(), model=..., error_model_factory=..., budget=400, burn_in=100)` 用 oracle 一次跑完上述步骤，供仿真使用；同一 `random_state` 下与分步写法结果一致。

与官方实现的差异：交叉验证准则用 AIPW 值标准差的期望解析式，代替官方的 200 次蒙特卡洛；主动分数下限为 `1e-3 * mean(u)`，保证所有行的概率为正（官方在概率为 0 时跳过该行，会破坏无偏性）；默认候选半径为 `(0, .01, ..., .07, .1)`，官方 Pew 实验为 `(.02, ..., .07)`；只实现论文主推的 π ∝ 预测误差，未实现图 2 中 π ∝ `min(p, 1-p)` 的变体。目前只支持二元均值。

## 公式与适用范围

ASI 使用 `Z_i = f_i + R_i / pi_i * (Y_i - f_i)`，点估计为 `mean(Z)`，标准误为 `std(Z, ddof=1) / sqrt(n)`。私有 `ipw_correct` 只计算公式，读取已查询位置的标签；数据校验由 `ActiveData` 完成。

ASI 保留总体均值、渐近终点 Wald 设计，需要 iid 总体、标注前确定的正查询概率和相应的矩条件。该方差公式不提供有限样本或 anytime 保证。

PPI 使用 `mean(f_unlabeled) + mean(y_labeled - f_labeled)`，标准误为 `sqrt(var(f_unlabeled, ddof=1)/N + var(y_labeled - f_labeled, ddof=1)/n)`。两项来自独立推断样本，预测模型需要独立训练。

## 序贯版本（仅仿真）

核心库暂不考虑序贯变体。ASI 论文的序贯设定（算法 2：边标注边更新模型）只保留在 `examples/simulations/online.py`，供 `main.py` 仿真使用：

- `active_label(features, oracle=..., target=Mean(), budget=..., model=..., update=..., update_every=..., update_on=...)`：按块或每收集满 `update_every` 个标签（`update_on='labels'`）更新模型并重新打分剩余行；`update='refit'` 传全部历史查询数据，`'finetune'` 只传新增数据，`update='none'` 固定模型。`oracle(index)` 只在该行被选中时调用一次。
- `Exploring(tau, explore)`：在核心 `Uncertainty` 上加入"前 explore 行使用 rho"。
- `BudgetCapped(tau, explore, greedy_steps)`：论文式 (8)，原始概率不超过剩余虚拟预算 `rho*(t+1) - 已查询数`；`greedy_steps>0` 时每个 `int(greedy_steps/rho)` 行窗口的最后 greedy_steps 行直接用完剩余预算。

## 重复实验与测试

```sh
PYTHONPATH=src:. python main.py --reps 100
python -m pytest
```

默认运行 `asi`、`asi_judge`、`asi_pretrained` 三种策略，各 100 次，使用上述序贯版本。仿真默认与论文序贯实验一致：`--update-on labels --update-every 100`、`--asi-budget-cap 1 --greedy-steps 100`、τ=0.5。`--tune-tau 1` 按式 (12) 在独立历史数据上调 τ：`asi_judge` 用评委模型在 `pretrain_size` 条历史样本上的预测，`asi_pretrained` 用预训练数据 80% 训练、20% 留出；冷启动的 `asi` 没有历史数据，仍用 `--asi-tau`。`asi_pretrained` 使用独立种子流生成 `pretrain_size` 条初始标签，后续用历史查询标签在线 refit；预训练标签在 `pretrain_labels` 列单独记录，不计入在线期望预算。

`results_asi/config.json` 记录参数；`per_rep_final.csv` 保存 300 条逐次结果；`fixed_budget_comparison.csv` 保存三种策略的汇总。

测试分三个文件：

- `common/test_core.py`：ASI 校正公式与区间函数收到的 values、`Uncertainty` 的 η 与预算下限、`tune_tau`、`LabelRequest` 只含选点信息且 `with_labels` 校验标签数量、PPI 公式与切分，以及固定预测下 200 次 DGP 覆盖率。
- `common/test_rasi.py`：三条路径的端点与预算、`RobustDesign` 逐行概率复现 `budget_path`、`scores=` 不影响默认选点、分步两批写法与便捷函数一致、与官方极小极大目标逐点一致、c=0 时 ρ=0 且 c 增大时 ρ 单调趋于 1、查询只发生在选中行、RASI 与 ASI 点估计一致，以及 150 次 DGP 覆盖率。
- `common/test_online.py`：序贯版本的查询时机、历史保留、预训练更新、按标签数触发更新、`Exploring` 与周期性用完预算，以及固定模型下与 `select_for_labeling` 一致。

若当前 Anaconda 环境在加载 readline 调试依赖时崩溃，可用 `python -m pytest -p no:debugging`。

后续再做配置拆分、Quantile/OLS 与连续标签 Mean、PPI++、序贯版本并入核心库、ConfidenceSequence 和更多组合有效性检查。
