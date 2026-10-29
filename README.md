[![Stars](https://img.shields.io/github/stars/hongfu001/OpenPPI)](https://github.com/hongfu001/OpenPPI/stargazers)
[![Forks](https://img.shields.io/github/forks/hongfu001/OpenPPI)](https://github.com/hongfu001/OpenPPI/network/members)
[![Issues](https://img.shields.io/github/issues/hongfu001/OpenPPI)](https://github.com/hongfu001/OpenPPI/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/hongfu001/OpenPPI)](https://github.com/hongfu001/OpenPPI/pulls)

<div align="center">

**OpenPPI**: An Open-Source Framework for Prediction-Powered Inference.
[Method Reference](docs/methods.md) · [Examples](examples/simulation)

</div>

OpenPPI is a Python library for prediction-powered inference: valid confidence intervals that combine a small labeled sample with predictions from a fixed machine-learning model. It implements representative methods for passive PPI (labeled and unlabeled samples given) and for active inference (the method chooses which units to label under a budget), each validated against the paper and the official code. Each method is a **sampling component** (which rows to label) combined with an **inference component** (estimate and interval); `openppi.methods` records the published combinations, so `get_method('ASI')` returns its parts. Every inference component inherits from `InferenceMethod`, takes a validated data snapshot and returns an `InferenceResult` with the estimate, standard error, confidence interval and diagnostics. The core depends only on NumPy and SciPy. The codebase is under active construction; comments, issues, contributions and collaborations are all welcome!

# Overview

## Prediction-Powered Inference

| Year | Title | Venue | Code Link | Registry name |
| ---- | ----- | ----- | --------- | -------------- |
| 2026 | **[Calibeating Prediction-Powered Inference](https://arxiv.org/abs/2604.21260)** | Arxiv | [Link](https://github.com/Larsvanderlaan/ppi-aipw) | `CalPPI`, `LinearCalPPI` |
| 2026 | **[Efficient Inference for Noisy LLM-as-a-Judge Evaluation](https://arxiv.org/abs/2601.05420)** | ICML'26 | [Link](https://github.com/yiqunchen/debias-llm-as-a-judge) | `RG`, `MLE` |
| 2026 | **[Multiple-Prediction-Powered Inference](https://arxiv.org/abs/2603.27414)** | ICLR'26 | | `MultiPPI` |
| 2026 | **[Cluster-Robust Prediction-Powered Inference](https://arxiv.org/abs/2610.09601)** | Arxiv | [Link](https://github.com/Michael-Howes/ppi_py) | `Cluster-PPI` |
| 2026 | **[Predictions as Surrogates: Revisiting Surrogate Outcomes in the Age of AI](https://arxiv.org/abs/2501.09731)** | Biometrika | [Link](https://github.com/Wenlong2000/RePPI) | `RePPI` |
| 2025 | **[FAB-PPI: Frequentist, Assisted by Bayes, Prediction-Powered Inference](https://arxiv.org/abs/2502.02363)** | ICML'25 | [Link](https://github.com/stefanocortinovis/fab-ppi) | `FAB-PPI` |
| 2025 | **[Assumption-Lean and Data-Adaptive Post-Prediction Inference](https://jmlr.org/papers/v26/24-0056.html)** | JMLR | [Link](https://github.com/qlu-lab/pspa) | `PSPA` |
| 2024 | **[Stratified Prediction-Powered Inference for Effective Hybrid Evaluation of Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract.html)** | NeurIPS'24 | | `StratPPI` |
| 2023 | **[PPI++: Efficient Prediction-Powered Inference](https://arxiv.org/abs/2311.01453)** | Ann. Appl. Stat. | [Link](https://github.com/aangelopoulos/ppi_py) | `PPI++` |
| 2023 | **[Prediction-Powered Inference](https://arxiv.org/abs/2301.09633)** | Science | [Link](https://github.com/aangelopoulos/ppi_py) | `PPI` |

## Prediction-Powered Inference for Time Series

| Year | Title | Venue | Code Link | Registry name |
| ---- | ----- | ----- | --------- | -------------- |
| 2026 | **[Prediction-powered inference for time series across space](https://arxiv.org/abs/2610.08715)** | NeurIPS'26 TS-LIMITS Workshop | [Link](https://github.com/shahzarrizvi/ppitsas) | `PPITSAS` |

## Active Inference

| Year | Title | Venue | Code Link | Registry name |
| ---- | ----- | ----- | --------- | -------------- |
| 2026 | **[Revisiting Active Sequential Prediction-Powered Mean Estimation](https://arxiv.org/abs/2604.18569)** | ICLR'26 | [Link](https://openreview.net/forum?id=Iw0tMeLed8) | `FTRL` |
| 2025 | **[Robust Sampling for Active Statistical Inference](https://arxiv.org/abs/2511.08991)** | NeurIPS'25 | [Link](https://github.com/lphLeo/Robust-Active-Statistical-Inference) | `RASI` |
| 2025 | **[Balanced Active Inference](https://papers.neurips.cc/paper_files/paper/2025/hash/4a3dadd6fd404a113e190e07380f0761-Abstract-Conference.html)** | NeurIPS'25 | [Link](https://github.com/Uninfty/Balanced_Active_Inference) | `BAI` |
| 2024 | **[Active Statistical Inference](https://arxiv.org/abs/2403.03208)** | ICML'24 | [Link](https://github.com/tijana-zrnic/active-inference) | `ASI` |


## Supported targets

| Target | Outcomes | Methods |
| ------ | -------- | ------- |
| `Mean()` | binary or continuous | all registered methods except OLS; RG and MLE require binary labels and judgments, and the cluster and multi-source containers currently require binary labels |
| `LinearRegression(fit_intercept, focus)` | continuous | OLS, PPI, PPI++, PSPA (passive); ASI, RASI (active) |

The tables below list the implemented methods. The last column is the name in the method registry, `openppi.get_method(name)`.

## Methods as components

| Method | Sampling component | Inference component | Data |
| ------ | ------------------ | ------------------- | ---- |
| OLS | pre-split samples | `OLS()` | `RegressionData` |
| PPI | pre-split samples | `TwoSample(lam=1.0)` | `PPIData` / `RegressionData` |
| PPITSAS | chronological blocks | `PPITSAS()` (mean only, Newey-West HAC) | `PPIData` |
| PPI++ | pre-split samples | `TwoSample(lam='tuned')` | `PPIData` / `RegressionData` |
| PSPA | pre-split samples | `PSPA()` | `PPIData` / `RegressionData` |
| RG / MLE | pre-split samples | `RoganGladen()` / `MisclassificationMLE()` | `PPIData` |
| CalPPI / LinearCalPPI | pre-split samples | `CalibratedPPI(calibration='isotonic' / 'linear')` | `PPIData` |
| RePPI | pre-split samples | `RePPI()` | `PPIData` |
| FAB-PPI | pre-split samples | `FABPPI()` | `PPIData` |
| Cluster-PPI | pre-split samples | `ClusterRobustPPI()` | `ClusteredPPIData` |
| StratPPI | `stratified_allocate` (optional budget planner) | `StratPPI()` | `StratifiedPPIData` |
| MultiPPI | `multi_allocate` | `MultiPPI()` | `MultiPPIData` |
| ASI | `Uncertainty(tau=.5)` | `ActiveRegressionIPW()` | `ActiveData` / `ActiveRegressionData` |
| RASI | burn-in, then `RobustDesign.fit(...)` | `ActiveRegressionIPW()` | `ActiveData` / `ActiveRegressionData` |
| FTRL | `FTRLDesign()` | `IPW()` | `ActiveData` |
| BAI | `BalancedDesign()` | `BAI()` | `BalancedActiveData` |

PPI and PPI++ share the two-sample estimator and differ only in λ; ASI and RASI use the mean/regression IPW dispatcher, while FTRL retains the mean IPW estimator. Components can be combined when the target, data and sampling-design covariance agree. For the supported target/data mapping of a method, inspect `get_method(name).target_data_types`.

# Getting Started

## Installation

OpenPPI requires Python 3.10 or later. Clone the repository and install it in editable mode; the core needs only NumPy and SciPy, and the `simulation` and `test` extras add pandas, scikit-learn and matplotlib for the examples and tests:

```sh
git clone https://github.com/hongfu001/OpenPPI.git
cd OpenPPI
python -m pip install -e '.[simulation,test]'
```

The package is imported as `openppi`.

## Mean estimation

Here is a simple example of power-tuned PPI++ for a binary mean, with an independently trained model:

```python
from sklearn.ensemble import RandomForestRegressor
from dataset.judge_sim import make_judge_data          # examples/simulation/dataset
from openppi import Mean, PPIData, TwoSample, split

# Split iid rows into a training, a labeled and an unlabeled sample.
frame = make_judge_data(n=2000, seed=0)
train, labeled, unlabeled = split(
    frame.drop(columns='label').to_numpy(), frame['label'].to_numpy(),
    train_size=400, labeled_size=400, random_state=42)
# Fit the model on the training sample only, then fix it.
model = RandomForestRegressor(n_estimators=60, min_samples_leaf=8, max_depth=8, random_state=0).fit(*train)
data = PPIData(labels=labeled[1], labeled_predictions=model.predict(labeled[0]),
               unlabeled_predictions=model.predict(unlabeled))

result = TwoSample(lam='tuned')(data, target=Mean(), alpha=.05)     # PPI++
print(f"Estimate: {result.estimate:.4f}")
print(f"Confidence interval: {result.confidence_interval}")
print(f"Lambda: {result.diagnostics['prediction_weight']:.3f}")
```

Active inference selects the rows to label first and then runs inference on the labeled record:

```python
from openppi import Mean, get_method, select_for_labeling

recipe = get_method('ASI')                                # Uncertainty(tau=.5) + ActiveRegressionIPW()
request = select_for_labeling(predictions, target=Mean(), design=recipe.sampling(),
                              budget=400, random_state=42)
record = request.with_labels(annotate(request.indices))   # labels only for the selected rows
result = recipe.inference(record, target=Mean(), alpha=.05)
```

`PPIData` accepts finite real outcomes and predictions. For continuous outcomes in active tasks, use `Mean(outcome_type='continuous')` and pass explicit nonnegative `scores=`; the binary probability-uncertainty rule is never applied implicitly in continuous mode.

## Time-series mean inference

`PPITSAS` uses the same corrected point estimate as PPI, with Newey-West HAC
variances for the prediction and residual time series. It supports **Mean only**.
Each call handles one location; array rows must retain chronological order
within consecutive, regularly spaced blocks. Pool locations for training if
useful, but do not concatenate them into a single inference series.

```python
from openppi import Mean, PPIData, PPITSAS, get_method

# Old unlabeled history -> recent rectification -> latest fitting block.
# f is a predictor trained on the separate fitting block, then fixed.
data = PPIData(labels=y_rectification,
               labeled_predictions=f_rectification,
               unlabeled_predictions=f_history)
result = get_method('PPITSAS').inference(data, target=Mean(outcome_type='continuous'))
# PPITSAS(max_lag=10)(data, target=Mean()) selects a fixed Bartlett lag.
```

The default selects a separate Newey-West plug-in lag for each sequence.
Stationarity, suitable mixing/moment conditions and negligible cross-block
and training dependence are required; the paper gives a proof sketch rather
than a completed general validity theorem. These are pointwise intervals
for long-run means, not forecast intervals or simultaneous spatial bands.
Do not use the iid random `split()` helper for this case.

The executed [PPITSAS notebook](examples/simulation/ppitsas.ipynb) follows the
same four steps as the PPI notebook and compares iid PPI with HAC on stationary,
spatially correlated AR(1) data. A shared linear predictor deliberately omits a
quadratic term. This is a lightweight simulation case, not a reproduction of
the paper's weather experiment. To repeat it and save per-site results:

```sh
cd examples/simulation
python time_series.py --reps 200 --out results_time_series
```

Coverage Monte Carlo SE uses independent replicate averages, accounting for
correlation among sites within a replicate. See the [method notes](docs/methods.md#ppitsas-time-series-mean)
for formulas, reference checks and chronological splitting details.

## Linear regression

The squared-loss M-estimator supports **OLS, PPI, PPI++ and PSPA** for passive coefficient inference, and **ASI and RASI** for active coefficient inference. Target covariates V define the population least-squares projection; the independent predictor may use richer features Z. The interfaces keep those roles separate:

```python
from openppi import LinearRegression, RegressionData, get_method

data = RegressionData(
    labels=y_labeled, labeled_predictions=f_labeled,
    unlabeled_predictions=f_unlabeled,
    labeled_covariates=V_labeled, unlabeled_covariates=V_unlabeled)
result = get_method('PPI++').inference(
    data, target=LinearRegression(fit_intercept=True, focus=1), alpha=.05)
print(result.estimate, result.covariance, result.confidence_interval)
```

Covariates exclude the intercept. `focus` is a zero-based coordinate **after** adding the intercept, so `focus=1` targets the first slope. Intervals are pointwise, not simultaneous.

For ASI, train an absolute-error predictor on independent pilot labels and select with a coefficient-specific score:

```python
from openppi import LinearRegression, Uncertainty, linear_regression_scores, select_for_labeling, get_method

target = LinearRegression(fit_intercept=True, focus=1)
scores = linear_regression_scores(V_pool, predicted_absolute_errors, target=target)
request = select_for_labeling(f_pool, target=target, covariates=V_pool,
                             scores=scores, design=Uncertainty(tau=.5), budget=400)
record = request.with_labels(annotate(request.indices))
result = get_method('ASI').inference(record, target=target)
```

RASI's `RobustDesign.fit(..., target=target, covariates=V_burn, reference_covariates=V_pool)` fits on burn-in labels, robustifies the focused errors and optionally selects its radius by coefficient-score cross-validation. Active inference solves the regression score equation with a full matrix sandwich rather than a mean standard error. See [derivations and reference differences](docs/methods.md#linear-regression-m-estimator-first-stage).

# Experiments

Every method has an executed notebook in [examples/simulation](examples/simulation) (`ppi.ipynb`, `ppitsas.ipynb`, `ppi_plus_plus.ipynb`, `asi.ipynb`, ...). They run from that folder, so the simulated `dataset` and `model` packages import directly. The [method reference](docs/methods.md) documents each method's interface, formulas, differences from the official implementations, and validation results.


# Repository layout

```text
openppi/              installable core library (NumPy and SciPy only)
├── sampling/         which rows to label: uniform, uncertainty, robust, regression, balanced,
│                     ftrl, stratified, allocation
├── inference/        estimators: ipw (IPW, ActiveRegressionIPW), two_sample (TwoSample),
│                     linear (OLS), adaptive (PSPA), misclassification (RG, MLE),
│                     calibrated (CalibratedPPI), recalibrated (RePPI), fab (FABPPI),
│                     cluster (ClusterRobustPPI), time_series (PPITSAS), stratified (StratPPI), multi (MultiPPI),
│                     balanced (BAI)
├── common/           data structures and split(), targets, regression helpers,
│                     InferenceResult, shared formulas
└── methods.py        registry: method = sampling component + inference component
examples/
├── simulation/       notebooks per method, coverage.py, regression.py, online.py,
│                     simulated dataset/ and model/
└── real_data/        (to come)
tests/                unit and coverage tests (examples/simulation/test_online.py covers online.py)
docs/methods.md       method reference
```

# TODO

OpenPPI is still under active development. We will add the following features down the road:

- Sequential (online) active inference in the core library; it currently lives in `examples/simulation/online.py`.

# Citation

If you find our repository useful for your research, please consider citing:

```bibtex
@software{openppi,
  title = {OpenPPI: An Open-Source Framework for Prediction-Powered Inference},
  author = {Gao, Hongfu and Liu, Pengfei and Hu, Jiang and Zhou, Wang},
  url = {https://github.com/hongfu001/OpenPPI},
  year = {2026}
}
```

# Contributors

- Hongfu Gao
- Pengfei Liu
- Jiang Hu
- Wang Zhou
