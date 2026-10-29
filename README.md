[![Stars](https://img.shields.io/github/stars/hongfu001/OpenPPI)](https://github.com/hongfu001/OpenPPI/stargazers)
[![Forks](https://img.shields.io/github/forks/hongfu001/OpenPPI)](https://github.com/hongfu001/OpenPPI/network/members)
[![Issues](https://img.shields.io/github/issues/hongfu001/OpenPPI)](https://github.com/hongfu001/OpenPPI/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/hongfu001/OpenPPI)](https://github.com/hongfu001/OpenPPI/pulls)

<div align="center">

**OpenPPI**: An Open-Source Framework for Prediction-Powered Inference.
[Method Reference](docs/methods.md) · [Examples](examples)

</div>

OpenPPI is a Python library for prediction-powered inference: valid confidence intervals that combine a small labeled sample with predictions from a fixed machine-learning model. It implements representative methods for passive PPI (labeled and unlabeled samples given) and for active inference (the method chooses which units to label under a budget), each validated against the paper and the official code. Every method inherits from a common `Method`, takes a validated data snapshot and returns an `InferenceResult` with the estimate, standard error, confidence interval and diagnostics. The core depends only on NumPy and SciPy. The codebase is under active construction; comments, issues, contributions and collaborations are all welcome!

# Overview

OpenPPI currently targets the mean of binary labels and implements the following methods.

## Prediction-Powered Inference

| Year | Title | Venue | Code Link | Implementation |
| ---- | ----- | ----- | --------- | -------------- |
| 2026 | **[Multiple-Prediction-Powered Inference](https://arxiv.org/abs/2603.27414)** | ICLR'26 | | sampling.multippi.multi_allocate methods.multippi.MultiPPI |
| 2026 | **[Cluster-Robust Prediction-Powered Inference](https://arxiv.org/abs/2610.09601)** | Arxiv | [Link](https://github.com/Michael-Howes/ppi_py) | methods.cluster_ppi.ClusterRobustPPI |
| 2026 | **[Predictions as Surrogates: Revisiting Surrogate Outcomes in the Age of AI](https://arxiv.org/abs/2501.09731)** | Arxiv | [Link](https://github.com/Wenlong2000/RePPI) | methods.reppi.RePPI |
| 2025 | **[FAB-PPI: Frequentist, Assisted by Bayes, Prediction-Powered Inference](https://arxiv.org/abs/2502.02363)** | ICML'25 | [Link](https://github.com/stefanocortinovis/fab-ppi) | methods.fab_ppi.FABPPI |
| 2023 | **[PPI++: Efficient Prediction-Powered Inference](https://arxiv.org/abs/2311.01453)** | Arxiv | [Link](https://github.com/aangelopoulos/ppi_py) | methods.ppi_plus.PPIPlusPlus |
| 2023 | **[Prediction-Powered Inference](https://arxiv.org/abs/2301.09633)** | Science | [Link](https://github.com/aangelopoulos/ppi_py) | methods.ppi.PPI |

## Active Inference

| Year | Title | Venue | Code Link | Implementation |
| ---- | ----- | ----- | --------- | -------------- |
| 2026 | **[Revisiting Active Sequential Prediction-Powered Mean Estimation](https://arxiv.org/abs/2604.18569)** | ICLR'26 | | sampling.ftrl.FTRLDesign |
| 2025 | **[Robust Sampling for Active Statistical Inference](https://arxiv.org/abs/2511.08991)** | NeurIPS'25 | [Link](https://github.com/lphLeo/Robust-Active-Statistical-Inference) | sampling.rasi.RobustDesign methods.rasi.RASI |
| 2025 | **[Balanced Active Inference](https://papers.neurips.cc/paper_files/paper/2025/hash/4a3dadd6fd404a113e190e07380f0761-Abstract-Conference.html)** | NeurIPS'25 | [Link](https://github.com/Uninfty/Balanced_Active_Inference) | sampling.bai.BalancedDesign methods.bai.BAI |
| 2024 | **[Active Statistical Inference](https://arxiv.org/abs/2403.03208)** | ICML'24 | | sampling.asi.Uncertainty methods.asi.ASI |

## TODO

OpenPPI is still under active development. We will add the following features down the road:

- Quantile, OLS and GLM targets, and the M-estimation versions of PPI++, FAB-PPI and Cluster-Robust PPI++.
- Continuous labels for mean estimation.
- Sequential (online) active inference in the core library; it currently lives in `examples/simulations/online.py`.
- Labeling-design planning formulas for clustered data (Appendix C of Cluster-Robust PPI).
- Confidence sequences with anytime-valid guarantees.

## Installation

OpenPPI requires Python 3.10 or later. Clone the repository and install it in editable mode; the `simulation` and `test` extras add pandas and scikit-learn for the examples and tests:

```sh
git clone https://github.com/hongfu001/OpenPPI.git
cd OpenPPI
python -m pip install -e '.[simulation,test]'
```

The package is imported as `prediction_inference`.

## Unit Test

Each method is tested against hand calculations, the formulas of the official code and Monte Carlo coverage:

```sh
python -m pytest
```

If an Anaconda environment crashes while loading the readline debugging dependency, use `python -m pytest -p no:debugging`.

## Examples

Here is a simple example of power-tuned PPI++ for a binary mean, with an independently trained model:

```python
from sklearn.ensemble import RandomForestRegressor
from common.utils import split
from dataset.judge_sim import make_judge_data
from prediction_inference import PPIPlusPlus, Mean, PPIData

# Split iid rows into a training, a labeled and an unlabeled sample.
frame = make_judge_data(n=2000, seed=0)
train, labeled, unlabeled = split(
    frame.drop(columns='label').to_numpy(), frame['label'].to_numpy(),
    train_size=400, labeled_size=400, random_state=42)
# Fit the model on the training sample only, then fix it.
model = RandomForestRegressor(n_estimators=60, min_samples_leaf=8, max_depth=8, random_state=0).fit(*train)
data = PPIData(labels=labeled[1], labeled_predictions=model.predict(labeled[0]),
               unlabeled_predictions=model.predict(unlabeled))

result = PPIPlusPlus()(data, target=Mean(), alpha=.05)
print(f"Estimate: {result.estimate:.4f}")
print(f"Confidence interval: {result.confidence_interval}")
print(f"Lambda: {result.diagnostics['prediction_weight']:.3f}")
```

Active inference selects the rows to label first and then runs inference on the labeled record:

```python
from prediction_inference import ASI, Mean, Uncertainty, select_for_labeling

request = select_for_labeling(predictions, target=Mean(), design=Uncertainty(tau=.5),
                              budget=400, random_state=42)
record = request.with_labels(annotate(request.indices))   # labels only for the selected rows
result = ASI()(record, target=Mean(), alpha=.05)
```

Every method has a runnable script in the [examples](examples) folder:

```sh
PYTHONPATH=src:. python examples/ppi_plus_mean.py
PYTHONPATH=src:. python examples/cluster_ppi_mean.py
```

The [method reference](docs/methods.md) documents each method's interface, formulas, differences from the official implementations, and validation results.

## Citation

If you find our repository useful for your research, please consider citing:

```bibtex
@software{openppi,
  title = {OpenPPI: An Open-Source Framework for Prediction-Powered Inference},
  author = {Gao, Hongfu and Zhou, Wang},
  url = {https://github.com/hongfu001/OpenPPI},
  year = {2026}
}
```

## Contributors

- Hongfu Gao
- Wang Zhou
