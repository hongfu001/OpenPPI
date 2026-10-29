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

OpenPPI supports population means with binary or continuous outcomes, plus an initial linear-regression coefficient target for OLS, PPI, PPI++ and PSPA. RG/MLE require binary labels and judgments; the current cluster and multi-source data containers also retain their binary-label validation. It implements the following methods. The last column is the name in the method registry, `openppi.get_method(name)`.

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

## Active Inference

| Year | Title | Venue | Code Link | Registry name |
| ---- | ----- | ----- | --------- | -------------- |
| 2026 | **[Revisiting Active Sequential Prediction-Powered Mean Estimation](https://arxiv.org/abs/2604.18569)** | ICLR'26 | [Link](https://openreview.net/forum?id=Iw0tMeLed8) | `FTRL` |
| 2025 | **[Robust Sampling for Active Statistical Inference](https://arxiv.org/abs/2511.08991)** | NeurIPS'25 | [Link](https://github.com/lphLeo/Robust-Active-Statistical-Inference) | `RASI` |
| 2025 | **[Balanced Active Inference](https://papers.neurips.cc/paper_files/paper/2025/hash/4a3dadd6fd404a113e190e07380f0761-Abstract-Conference.html)** | NeurIPS'25 | [Link](https://github.com/Uninfty/Balanced_Active_Inference) | `BAI` |
| 2024 | **[Active Statistical Inference](https://arxiv.org/abs/2403.03208)** | ICML'24 | [Link](https://github.com/tijana-zrnic/active-inference) | `ASI` |

## Methods as components

| Method | Sampling component | Inference component | Data |
| ------ | ------------------ | ------------------- | ---- |
| CalPPI / LinearCalPPI | pre-split samples | `CalibratedPPI(calibration='isotonic' / 'linear')` | `PPIData` |
| StratPPI | `stratified_allocate` (optional budget planner) | `StratPPI()` | `StratifiedPPIData` |
| PPI | pre-split samples | `TwoSample(lam=1.0)` | `PPIData` |
| PPI++ | pre-split samples | `TwoSample(lam='tuned')` | `PPIData` |
| RePPI | pre-split samples | `RePPI()` | `PPIData` |
| FAB-PPI | pre-split samples | `FABPPI()` | `PPIData` |
| Cluster-PPI | pre-split samples | `ClusterRobustPPI()` | `ClusteredPPIData` |
| MultiPPI | `multi_allocate` | `MultiPPI()` | `MultiPPIData` |
| ASI | `Uncertainty(tau=.5)` | `IPW()` | `ActiveData` |
| RASI | burn-in, then `RobustDesign.fit(...)` | `IPW()` | `ActiveData` |
| FTRL | `FTRLDesign()` | `IPW()` | `ActiveData` |
| BAI | `BalancedDesign()` | `BAI()` | `BalancedActiveData` |

ASI, RASI and FTRL share the IPW estimator and differ only in how rows are chosen; PPI and PPI++ share the two-sample estimator and differ only in λ. Components can be combined freely, e.g. a `RobustDesign` with any estimator that accepts `ActiveData`.

## TODO

OpenPPI is still under active development. We will add the following features down the road:

- Quantile, OLS and GLM targets, and the M-estimation versions of PPI++, FAB-PPI and Cluster-Robust PPI++.
- Continuous labels for mean estimation.
- Sequential (online) active inference in the core library; it currently lives in `examples/simulation/online.py`.
- Labeling-design planning formulas for clustered data (Appendix C of Cluster-Robust PPI).
- Confidence sequences with anytime-valid guarantees.

## Installation

OpenPPI requires Python 3.10 or later. Clone the repository and install it in editable mode; the core needs only NumPy and SciPy, and the `simulation` and `test` extras add pandas, scikit-learn and matplotlib for the examples and tests:

```sh
git clone https://github.com/hongfu001/OpenPPI.git
cd OpenPPI
python -m pip install -e '.[simulation,test]'
```

The package is imported as `openppi`.

## Unit Test

Each method is tested against hand calculations, the formulas of the official code and Monte Carlo coverage. The repeated-sampling coverage checks are marked `slow`; skip them for a quick run:

```sh
python -m pytest -m "not slow"     # about 2 seconds
python -m pytest                   # everything, about 45 seconds
```

If an Anaconda environment crashes while loading the readline debugging dependency, use `python -m pytest -p no:debugging`.

## Examples

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

recipe = get_method('ASI')                                # Uncertainty(tau=.5) + IPW()
request = select_for_labeling(predictions, target=Mean(), design=recipe.sampling(),
                              budget=400, random_state=42)
record = request.with_labels(annotate(request.indices))   # labels only for the selected rows
result = recipe.inference(record, target=Mean(), alpha=.05)
```

Every method has an executed notebook in [examples/simulation](examples/simulation) (`ppi.ipynb`, `ppi_plus_plus.ipynb`, `asi.ipynb`, ...). They run from that folder, so the simulated `dataset` and `model` packages import directly. The repeated ASI experiment and its plots:

```sh
cd examples/simulation
python coverage.py --reps 100          # writes results_asi/
jupyter notebook comparison.ipynb      # coverage, width and estimates by policy
```

The [method reference](docs/methods.md) documents each method's interface, formulas, differences from the official implementations, and validation results.

## Repository layout

```text
openppi/              installable core library (NumPy and SciPy only)
├── sampling/         which rows to label, one file per mechanism: uniform, uncertainty, robust,
│                     balanced, ftrl, allocation
├── inference/        estimators, one file per estimator: ipw (IPW), two_sample (TwoSample),
│                     recalibrated (RePPI), fab (FABPPI), cluster (ClusterRobustPPI), multi (MultiPPI),
│                     balanced (BAI)
├── common/           data structures and split(), targets, InferenceResult, shared formulas
└── methods.py        registry: method = sampling component + inference component
examples/
├── simulation/       notebooks per method, coverage.py, online.py, simulated dataset/ and model/
└── real_data/        (to come)
tests/                unit and coverage tests (examples/simulation/test_online.py covers online.py)
docs/methods.md       method reference
```

## Citation

If you find our repository useful for your research, please consider citing:

```bibtex
@software{openppi,
  title = {OpenPPI: An Open-Source Framework for Prediction-Powered Inference},
  author = {Gao, Hongfu and Zhou, Wang and Liu, Pengfei and Hu, Jiang},
  url = {https://github.com/hongfu001/OpenPPI},
  year = {2026}
}
```

## Contributors

- Hongfu Gao
- Wang Zhou
- Pengfei Liu
- Jiang Hu

## Continuous-response regression simulation

The original binary simulations remain available. The new [regression notebook](examples/simulation/regression.ipynb) compares fixed-predictor mean inference for homoskedastic and heteroskedastic continuous labels, both with exact population mean **1**:

```sh
python examples/simulation/regression.py --reps 100 --out examples/simulation/results_regression
```

Given-label methods share independently trained predictions, labeled data and unlabeled data: Label-only, PPI, PPI++, PSPA, RePPI, CalPPI and LinearCalPPI. The selection group compares Uniform+IPW, ASI, RASI, BAI, FTRL and StratPPI with their respective sampling rules. StratPPI collects fresh conditional samples under a pilot-planned stratified allocation. Summaries report coverage, width, bias, RMSE and actual labels, with predictor-training, error-pilot and burn-in label costs shown separately. Default selection budgets are expected counts for Bernoulli designs; no equality of total labeling cost is implied.

`PPIData` accepts finite real outcomes and predictions. Use `Mean(outcome_type='continuous')` for active regression tasks, and pass explicit nonnegative `scores=`; the binary probability-uncertainty rule is never applied implicitly in continuous mode. `SimulationLearner` and `SklearnModel` return raw predictions when `outcome_type='continuous'`. See [method notes](docs/methods.md#continuous-response-mean-simulation) for the data-generating model, score choices and online interface.

## Linear-regression M-estimator

The first coefficient-inference stage supports **OLS, PPI, PPI++ and PSPA**, while preserving the existing mean interface. Target covariates V define the population least-squares projection; the independent predictor may use richer features Z. The interfaces keep those roles separate:

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

Covariates exclude the intercept. `focus` is a zero-based coordinate **after** adding the intercept, so `focus=1` targets the first slope. Intervals are pointwise, not simultaneous. For the supported-target/data mapping, inspect `get_method('PPI++').target_data_types`. Other registered methods do not yet advertise regression support.

```sh
python examples/simulation/regression.py --target linear --focus 1 --reps 100
```

The [coefficient notebook](examples/simulation/linear_regression.ipynb) shows coefficients, covariance, the focus-coefficient comparison and all-coefficient supplement. This stage implements the squared-loss M-estimator rather than a general loss-optimization framework. See [derivations and reference differences](docs/methods.md#linear-regression-m-estimator-first-stage).
