# OpenPPI

A Python library for prediction-powered inference (package name `prediction_inference`). It currently implements **binary mean estimation with endpoint Wald intervals for ASI, RASI (robust active inference), BAI (balanced active inference), the FTRL query policy, independent two-sample PPI, PPI++, FAB-PPI and Cluster-Robust PPI++**. Every method inherits from a common `Method`, which validates the inputs and returns an `InferenceResult`.

## Installation and usage

```sh
python -m pip install -e '.[simulation,test]'
PYTHONPATH=src:. python examples/asi_mean.py
PYTHONPATH=src:. python examples/bai_mean.py
PYTHONPATH=src:. python examples/cluster_ppi_mean.py
PYTHONPATH=src:. python examples/fab_ppi_mean.py
PYTHONPATH=src:. python examples/ftrl_mean.py
PYTHONPATH=src:. python examples/ppi_mean.py
PYTHONPATH=src:. python examples/ppi_plus_mean.py
PYTHONPATH=src:. python examples/rasi_mean.py
PYTHONPATH=src:. python main.py --reps 100
python -m pytest
```

Simulations run from the source repository and are not part of the core package. The core depends only on NumPy/SciPy; experiments additionally need pandas and scikit-learn.

## Layout

```text
src/prediction_inference/
├── __init__.py       # public interface
├── base.py           # Method, InferenceResult
├── data.py           # ActiveData, PPIData, ClusteredPPIData, RobustActiveData, BalancedActiveData
├── targets.py        # Target, Mean; the single definition of sampling scores
├── _stats.py         # private formulas: ipw_correct, wald_interval
├── _fab.py           # private FAB formulas: horseshoe/Gaussian posterior means, spending function, FAB region
├── sampling/          # fixed-budget, fixed-model sampling: one file per method
│   ├── __init__.py
│   ├── base.py        # shared parts: SamplingDesign interface, LabelRequest, select_for_labeling
│   ├── uniform.py     # Uniform: uniform sampling (PPI baseline)
│   ├── asi.py         # ASI: Uncertainty, tune_tau
│   ├── bai.py         # BAI: BalancedDesign, cube_sample (pure-NumPy cube method)
│   ├── ftrl.py        # FTRL: FTRLDesign (no-regret query probabilities oblivious to the current row; inference via ASI)
│   └── rasi.py        # RASI: RobustDesign, BurnIn, select_burn_in, paths and robust ρ, cross-validated c, robust_active_label
└── methods/
    ├── __init__.py
    ├── asi.py        # ASI(Method)
    ├── bai.py        # BAI(Method), cube_variance
    ├── cluster_ppi.py # ClusterRobustPPI(Method), cluster_moments
    ├── fab_ppi.py    # FABPPI(Method)
    ├── rasi.py       # RASI(ASI)
    ├── ppi.py        # PPI(Method)
    └── ppi_plus.py   # PPIPlusPlus(Method), tuned_lambda
```

- `dataset/judge_sim.py`: the DGP, `make_judge_data`, which returns a DataFrame, and `make_clustered_data`, units nested in iid clusters with a biased predictor (unit-level mean 0.5).
- `model/learners.py`: prediction model wrappers; `model/judge.py`: the judge mean model.
- `common/`: configuration, shared utilities (seeding, output, summaries, PPI split) and the core tests.
- `examples/asi_mean.py`, `examples/bai_mean.py`, `examples/cluster_ppi_mean.py`, `examples/fab_ppi_mean.py`, `examples/ftrl_mean.py`, `examples/ppi_mean.py`, `examples/ppi_plus_mean.py`, `examples/rasi_mean.py`: single-run examples.
- `examples/simulations/`: simulation strategies and the repeated-experiment entry point; `online.py` is sequential ASI (simulation only, see below); `main.py` keeps the backwards-compatible command line.

Dependency direction: examples may use every layer; model and dataset do not depend on the algorithms or the examples; the algorithms are independent of model, dataset and common.

## Method interface

```python
from prediction_inference import ASI, BAI, ClusterRobustPPI, FABPPI, PPI, PPIPlusPlus, RASI, Mean

# record is ActiveData, balanced is BalancedActiveData, data is PPIData, robust is RobustActiveData,
# clustered is ClusteredPPIData.
# result = ASI()(record, target=Mean(), alpha=.05)
# result = BAI()(balanced, target=Mean(), alpha=.05)
# result = RASI()(robust, target=Mean(), alpha=.05)
# result = PPI()(data, target=Mean(), alpha=.05)
# result = PPIPlusPlus()(data, target=Mean(), alpha=.05)
# result = FABPPI()(data, target=Mean(), alpha=.05)
# result = ClusterRobustPPI()(clustered, target=Mean(), alpha=.05)
```

`Method` lives in `prediction_inference.base`. It checks the data type, target type and alpha, then calls the subclass's `_estimate(data, target)`. The subclass returns `(point, se, diagnostics)` and may put `values` into diagnostics; the base class passes these to the interval function and removes `values` from the final diagnostics.

`InferenceResult` has six fields: `estimate`, `standard_error`, `confidence_interval`, `alpha`, `method` and `diagnostics`.

The private Wald formula is the default. A custom interval function has the signature `interval(point, se, alpha, values=None)` and is passed as `ASI(interval=...)`, `PPI(interval=...)` or `PPIPlusPlus(interval=...)`. For ASI, values is the array of corrected values; for PPI it is a dict with the arrays `residuals` and `unlabeled_predictions`; for PPI++ the dict has the same keys, holding `Y - λf` and `λf̃`.

The old `asi`, `asi_mean`, `ppi`, estimator and interval-class interfaces have been removed. `_stats.py` is internal to the library.

## Fixed-budget sampling

The core library only considers a fixed model and a fixed budget: predict with the model, decide the sampling probabilities once, then select the rows to label. Each sampling design lives in its own file and inherits from `SamplingDesign`; `calibrate(reference_scores, rho)` calibrates it and `probability(scores)` returns probabilities fixed before labeling. `sample(scores, propensities, rng)` and `record(...)` are two overridable hooks, defaulting to independent Bernoulli sampling and returning `ActiveData`; BAI overrides them to sample jointly with the cube method and return `BalancedActiveData`:

| Design | File | Behavior |
| --- | --- | --- |
| `Uniform()` | `uniform.py` | Constant probability `rho = budget / n`; with the ASI estimator this is the PPI baseline |
| `Uncertainty(tau=.5)` | `asi.py` | `pi = (1-tau) * clip(eta*u, 0, 1) + tau*rho`, `eta = rho / mean(u)`, u defaults to `2*min(p, 1-p)` (Algorithm 1 of the ASI paper); floor `tau * rho`; clipping can leave the expected number of labels slightly below the budget |
| `BalancedDesign(tau=.5)` | `bai.py` | BAI: the same probabilities as `Uncertainty`, sampled jointly with the cube method so that `sum(x * xi / pi) ≈ sum(x)` (see the BAI section) |
| `FTRLDesign()` | `ftrl.py` | FTRL: `p_t` depends only on Φ of earlier rows and rises from `tau/8` to `tau = budget/n` (see the FTRL section) |
| `RobustDesign` | `rasi.py` | RASI second batch: active rule π ∝ prediction error, interpolated towards uniform along a budget-preserving path (see the RASI section) |

`Mean.sampling_scores` is the only place where `2*min(p, 1-p)` is defined.

The flow of selecting, labeling and assembling a record for one batch is shared by all designs (full example in `examples/asi_mean.py`):

```python
from prediction_inference import ASI, Mean, Uncertainty, select_for_labeling

predictions = model.predict(features)          # any already trained, fixed model
request = select_for_labeling(predictions, target=Mean(), design=Uncertainty(tau=.5),
                              budget=400, random_state=42)
labels = annotate(request.indices)             # export row ids to annotators; labels come back in the same order
record = request.with_labels(labels)           # -> ActiveData
result = ASI()(record, target=Mean(), alpha=.05)
```

`select_for_labeling` reads no labels. It calibrates a copy of the design on this batch's scores with `rho = budget / n` and returns a `LabelRequest`: `predictions`, `propensities` and boolean `queried` for all n rows (copied and read-only), plus the calibrated design copy `design` (diagnostics only, no labels). The optional `scores=` replaces the default `target.sampling_scores`, e.g. prediction errors for RASI's second batch. `request.indices` are the rows to label; `request.with_labels(labels)` requires labels in one-to-one correspondence with `indices`, otherwise it raises, and returns `ActiveData`. In `ActiveData` the predictions, observed labels, query flags and query probabilities are aligned and unqueried labels must be NaN; the arrays are copied and protected on construction.

With independent historical labeled data, τ can be chosen by eq. (12) of the ASI paper: `tune_tau(historical_predictions, historical_labels, target=Mean(), rho=budget/n)` minimizes `mean((Y-f)^2 / pi_tau)` over `np.linspace(.01, .99, 100)`. The historical predictions should come from a model not trained on those labels (e.g. a held-out part of the training set).

## Balanced active inference, BAI (one batch)

Implements Algorithm 1 of Chen, Zhou, Peng & Wang (NeurIPS 2025), following the official code `Uninfty/Balanced_Active_Inference`. The sampling probabilities are the same as ASI's (the τ mixture rule; the paper's τ is the weight of the active part, i.e. `1 - tau` here, and the two agree at 0.5), but rows are no longer sampled independently: the cube method (Deville & Tillé 2004) selects them jointly, keeping each row's inclusion probability at π while satisfying the balancing constraint `sum(x * xi / pi) ≈ sum(x)`. Full example in `examples/bai_mean.py`:

```python
from prediction_inference import BAI, BalancedDesign, Mean, select_for_labeling

request = select_for_labeling(predictions, target=Mean(), design=BalancedDesign(tau=.5),
                              budget=400, random_state=42)
record = request.with_labels(annotate(request.indices))   # -> BalancedActiveData
result = BAI()(record, target=Mean(), alpha=.05)

# As in the official code: an error model fits the signed error e(x) ≈ f(x) - y; probabilities follow |e|, and the signed e is balanced
request = select_for_labeling(predictions, target=Mean(), design=BalancedDesign(balancing=e),
                              budget=400, scores=np.abs(e), random_state=42)
```

- **Balancing variables**: `scores` (û in the paper's main text) are balanced by default. The official code balances the signed error prediction, which is exactly what the paper's Assumption 5 (`sgn(f - f̂)·û ≈ f - f̂`) needs; pass it via `balancing=`. The gain comes from balancing away the systematic bias of f̂: when f̂ is unbiased or label noise dominates (e.g. this repository's binary simulated data), BAI and ASI are nearly identical. `fixed_size=True` additionally balances π itself, fixing the number selected at about `sum(pi)`.
- **Cube method**: `cube_sample` is a pure-NumPy implementation (the official code calls the R package `BalancedSampling` through rpy2). Rows are first shuffled; in the flight phase each step moves π on q+1 non-integer rows along the null space of the balancing constraints, choosing the direction by the martingale property so inclusion probabilities are preserved; the landing phase repeats while dropping the last balancing variable one at a time, and the remaining rows are drawn independently. Complexity is O(n·p³), about 70 ms per draw for n=2000, p=1.
- **Variance**: the design variance uses the Deville–Tillé approximation for balanced sampling (`cube_variance`, matching the official `estimate_cube_variance` value by value), i.e. the residual sum of squares after a weighted regression of the residuals `Y - f` on the balancing variables. By default the Hájek-weighted label variance divided by n is added, so that, like ASI and PPI, the interval targets the population mean E[Y] (V0 in Theorem 1 of the paper); `BAI(finite_population=True)` keeps only the design variance and targets the mean of the pool labels, as in the official code. Diagnostics report `design_variance`, `population_variance` and the `relative_imbalance` of each balancing variable.
- **Validation**: reproducing the paper's Friedman setting (budget 0.1, finite-pool target, strongly shrunk gradient boosting as in the official code), the classical active inference interval width is 0.369 and BAI's is 0.245 (34% narrower, both with coverage about 0.90; the paper's Table 1 reports 0.3305 → 0.1052, the gap coming from XGBoost versus sklearn and different data sizes).

## FTRL query policy (one batch)

Implements the query policy of Sfyraki & Wang (ICLR 2026), following the code in their supplementary material. The paper finds that ASI's mixture rule gives the narrowest intervals when the uniform weight is close to 1, and proves that if the query probability does not depend on the current row's covariates, minimizing `Σ Φ_t / p` with Follow-the-Regularized-Leader drives `p_t` to the budget cap `τ = T_b/T`. The closed-form update is

`p_t = max(β, min(τ, -γ θ_{t-1}))`, `θ_t = θ_{t-1} - Φ_t / p_t²`, `θ_0 = 0`,

with defaults `β = τ/8` and `γ = 1/√n`, as in the official code. Φ_t ≈ `E[(Y - f)^2 | X]` is passed as `scores` (for binary labels with calibrated f, `f(1 - f)` works; the official code fits squared residuals with linear regression). `p_t` depends only on Φ of the rows before t, so with a fixed model the whole sequence can be computed before labeling, making it an ordinary `SamplingDesign`. The paper's estimator and interval are the same as ZC24's, so inference uses `ASI()` directly:

```python
from prediction_inference import ASI, FTRLDesign, Mean, select_for_labeling

request = select_for_labeling(predictions, target=Mean(), design=FTRLDesign(),
                              scores=predictions * (1 - predictions), budget=400, random_state=42)
result = ASI()(request.with_labels(annotate(request.indices)), target=Mean(), alpha=.05)
```

Row order is arrival order (the official code shuffles on every trial). Since `p_t ≤ τ`, the expected number of labels never exceeds the budget; in practice `p_t` reaches τ within a row or two and the design then equals uniform sampling, which is exactly the paper's conclusion. The paper's full Algorithm 1 also updates the model and Φ during the sequential process; that is a sequential variant and is not yet in the core library.

## Independent two-sample PPI (four steps)

```python
from sklearn.ensemble import RandomForestRegressor
from common.utils import split
from dataset.judge_sim import make_judge_data
from prediction_inference import PPI, Mean, PPIData

# 1. Data, random split and independent training
frame = make_judge_data(n=2000, seed=0)
train, labeled, unlabeled = split(
    frame.drop(columns='label').to_numpy(), frame['label'].to_numpy(),
    train_size=400, labeled_size=400, random_state=42)
model = RandomForestRegressor(n_estimators=60, min_samples_leaf=8,
                              max_depth=8, random_state=0, n_jobs=1).fit(*train)
data = PPIData(labels=labeled[1],
               labeled_predictions=model.predict(labeled[0]),
               unlabeled_predictions=model.predict(unlabeled))
# 2. Target
target = Mean()
# 3. Inference
result = PPI()(data, target=target, alpha=.05)
# 4. Output
print(result.estimate, result.confidence_interval, result.diagnostics)
```

`split` returns `(X_train, y_train)`, `(X_labeled, y_labeled)` and `X_unlabeled`. The size arguments are row counts; the remaining rows form the unlabeled sample, and each part has at least two rows. It does not stratify on labels and does not return unlabeled labels.

Rows must come from an iid population and the three parts must not overlap. The prediction model is fit only on the independent training set and fixed before inference. `PPIData` requires the labels and predictions to be aligned and each inference sample to have at least two rows; all arrays are copied and made read-only. True labels are currently binary only.

## PPI++ (power-tuned two-sample PPI)

Implements the mean version of Angelopoulos, Duchi & Zrnic (2023) (Example 6.1 and Algorithm 2 of the paper), following `ppi_mean_test` in `yiqunchen/pppower`. It uses the same `PPIData` as PPI (full example in `examples/ppi_plus_mean.py`):

```python
from prediction_inference import PPIPlusPlus, Mean

result = PPIPlusPlus()(data, target=Mean(), alpha=.05)        # plug-in λ
result.diagnostics['prediction_weight']                        # the λ actually used
PPIPlusPlus(lam=0)(data, target=Mean())                        # classical estimate: labels only
PPIPlusPlus(lam=1)(data, target=Mean())                        # same point estimate as PPI
```

- **Point estimate**: `mean(Y) + λ * (mean(f_unlabeled) - mean(f_labeled))`, unbiased for any λ.
- **λ**: by default `λ̂ = Cov_n(Y, f) / ((1 + n/N) * Var_{N+n}(f))` (`tuned_lambda`), which minimizes the asymptotic variance. The more accurate the predictions, the closer λ̂ is to 1; useless predictions give λ̂ near 0 (the classical estimate); predictions anticorrelated with labels give a negative λ̂. Mean estimation is convex for every real λ, so λ̂ is not clipped to [0, 1]; constant predictions give λ = 0.
- **Standard error**: `sqrt(λ² Var_{N+n}(f) / N + Var_n(Y - λf) / n)`, i.e. Algorithm 2's `Σ̂ = (n/N) V̂_f + V̂_Δ` divided by n.
- **Difference from pppower**: pppower uses only the labeled sample for Var(f) in λ̂ and the variance; following the paper, both samples are pooled here, which is nearly identical when N ≫ n. With `lam=1` the standard error differs slightly from `PPI`, whose Var(f) uses only the unlabeled sample.
- **Validation**: in `examples/ppi_plus_mean.py`, λ̂ = 0.536 and the 95% interval widths are 0.0981 (classical), 0.0965 (PPI) and 0.0933 (PPI++).

## FAB-PPI (a prior on the rectifier)

Implements the mean version of Cortinovis & Caron (ICML 2025) (Algorithm 2 in the supplementary material), following `ppi_fab_mean_ci` in the official code `stefanocortinovis/fab-ppi`. It uses the same `PPIData` as PPI (full example in `examples/fab_ppi_mean.py`):

```python
from prediction_inference import FABPPI, Mean

result = FABPPI()(data, target=Mean(), alpha=.05)                  # FAB-PPI, horseshoe prior
result = FABPPI(lam=None)(data, target=Mean(), alpha=.05)          # FAB-PPI++, λ as in PPIPlusPlus
result = FABPPI(prior='gaussian')(data, target=Mean())             # Gaussian prior (not robust, for comparison only)
result = FABPPI(imputed_uncertainty=False)(data, target=Mean())    # the paper's experiments: ignore the uncertainty in mean(f̃) when N ≫ n
```

- **Rectifier**: `Δ̂ = mean(Y - λf) + (λ - 1) mean(f̃)`, `σ̂² = Var_n(Y - λf)/n + (λ-1)² Var(f̃)/N` (sign as in the official code, "labels minus predictions"; the prior is symmetric, so this is equivalent to the paper's `f - Y`).
- **Prior**: the default is the horseshoe prior `Δ ~ N(0, ν²σ̂²)`, `ν ~ C⁺(0, 1)`, with scale σ̂ (the paper's parameter-free choice). In standardized units `y = Δ̂/σ̂` the posterior mean is `y(1 - κ(y))` with `κ = (2/3) ₁F₁(2, 5/2, -y²/2) / ₁F₁(1, 3/2, -y²/2)`; for `|y| ≥ 4` the equivalent Dawson form `1/(√2 D(y/√2)) - 1/y` is used for numerical stability. The Gaussian prior `N(0, σ̂²)` has posterior mean y/2 and is inconsistent (Proposition 4.3 of the paper); it is included only for comparison.
- **Point estimate**: `mean(f̃) + σ̂ · E[Δ/σ̂ | y]`. `standard_error` reports the frequentist standard error of the unshrunk PPI estimate, `sqrt(σ̂² + Var(f̃)/N)`, for reference only; the interval is not of Wald form.
- **Interval**: `[mean(f̃) + σ̂·L - pad, mean(f̃) + σ̂·U + pad]`, where `[L, U]` is the 1-δ FAB region and `pad = sqrt(Var(f̃)/N) · z_{1-(α-δ)/2}` (the Minkowski sum in eq. 21). The FAB region comes from the spending function w(β), which equalizes `π(y)/φ(y-β)` at the two ends of the acceptance region (Neyman–Pearson); the region is `{β : β + z_{α(1-w)} ≤ y ≤ β + z_{1-αw}}`, solved with nested brentq in about 2 ms. For large |y| the horseshoe region reverts to the z-interval (Proposition 4.2), while the Gaussian region's width grows like 2|y|/3.
- **δ**: with `delta=None`, as in the official code, δ minimizes the interval width over (0, α); a fixed value can also be passed. `imputed_uncertainty=False` sets δ = α and drops the pad, matching Section 5 of the paper ("N large enough to ignore the uncertainty in m_θ"); it is only appropriate when N ≫ n.
- **Differences from the official implementation**: the official code clips λ̂ to [0, 1]; here, consistently with `PPIPlusPlus`, it is not clipped (eq. 14 of the paper has no clipping). The official pad uses `Σ(λf̃ - mean f̃)²/(N-1)/N`, which contains a bias term when λ ≠ 1; here it is `Var(f̃)/N` as in the paper's Algorithm 2. Only the horseshoe prior the paper recommends and the Gaussian comparison prior are implemented; scaled horseshoe, Laplace and other priors needing numerical integration are not.
- **Validation**: the horseshoe posterior mean agrees with numerical integration to within 1e-6; the Gaussian-prior region matches the official closed-form roots point by point; under W ~ N(β, 1) the FAB region has about nominal coverage; coverage over 200 DGP repetitions passes. In `examples/fab_ppi_mean.py`, N = 1200 and n = 400, so the uncertainty in mean(f̃) is not negligible and the full FAB-PPI interval (width 0.121) is wider than PPI's (0.097); the paper's gains come from the `imputed_uncertainty=False` setting with N ≫ n (0.079 in this example, but N is not large enough, so it is shown for comparison only).

## Cluster-Robust PPI++ (clustered and partially labeled data)

Implements the mean version of Broska & Howes (2026) (Theorem 1 and eq. 12), following `ppi_mean_ci_cluster` in the official code `Michael-Howes/ppi_py`. Units are nested in iid clusters and may be arbitrarily dependent within a cluster; clusters can be fully labeled, partially labeled or fully unlabeled, so labeled and unlabeled units are not independent and `PPIData` does not apply. `ClusteredPPIData` holds one row per unit (full example in `examples/cluster_ppi_mean.py`):

```python
import numpy as np
from prediction_inference import ClusterRobustPPI, ClusteredPPIData, Mean

data = ClusteredPPIData(labels=np.where(labeled, y, np.nan),   # NaN where the label was not revealed
                        predictions=f, labeled=labeled, clusters=cluster_ids)
result = ClusterRobustPPI()(data, target=Mean(), alpha=.05)    # tuned λ
ClusterRobustPPI(lam=0)(data, target=Mean())                   # labels only, cluster-robust SE
```

- **Assumptions**: clusters are iid; within a cluster every unit is labeled independently with a cluster-level probability p_g chosen independently of the data (unit-level Bernoulli labeling and whole-cluster labeling are the two extremes). Labels chosen from the predictions or from earlier labels are not covered.
- **Estimand**: the unit-level mean `E[Σ_i Y_gi] / E[m_g]` (eq. 3), not the mean of cluster means.
- **Point estimate**: `λ mean(f_unlabeled) + mean(Y - λf)` over labeled units, as in PPI++.
- **Standard error**: per cluster, `ε̂_g = Σ ξ(Y - θ̂)` and `Ẑ_g = Σ (ξ - (n/N)(1 - ξ))(f - mean f)` with mean f over all units; `Σ̂ = Σ_g [ε̂_g, Ẑ_g]ᵀ[ε̂_g, Ẑ_g] / G` and `SE² = G (Σ̂₁₁ - 2λΣ̂₁₂ + λ²Σ̂₂₂) / n²` (eq. 9). There is no small-sample correction, as in the paper.
- **λ**: by default `λ̂ = Σ̂₁₂ / Σ̂₂₂` with Σ̂ evaluated at the λ = 1 estimate (eq. 12), not clipped. Diagnostics report λ, the PPI correlation `ρ̂ = Σ̂₁₂ / sqrt(Σ̂₁₁ Σ̂₂₂)` (Section 2.5), the numbers of clusters and partially labeled clusters; a custom interval function receives the per-cluster sums as `cluster_rectifier` and `cluster_control`.
- **Singleton clusters** recover PPI++: the estimate, λ̂ and SE match `PPIPlusPlus` up to the ddof convention of the sample moments.
- **Not implemented**: the M-estimator version (Appendix A), coordinate-wise λ, sample weights, and the design-planning variance formulas of Appendix C.
- **Validation**: on `make_clustered_data` with 300 clusters and 200 repetitions, the cluster-robust interval covers at the nominal rate under both unit-level and whole-cluster labeling, while i.i.d. PPI++ undercovers. In `examples/cluster_ppi_mean.py` (labeling probability 0.3), the 95% widths are: partially labeled, 0.075 (labels only), 0.044 (i.i.d. PPI++, too narrow) and 0.072 (Cluster-Robust PPI++); whole clusters, 0.125, 0.044 and 0.075. As in the paper, PPI helps most when labels are concentrated in few clusters.

## Robust active inference, RASI (two batches)

Implements Algorithm 1 of Li, Zrnic & Candès (NeurIPS 2025), following the official code `lphLeo/Robust-Active-Statistical-Inference`. RASI labels in two batches; the second batch shares the `select_for_labeling` interface with ASI (full example in `examples/rasi_mean.py`):

```python
import numpy as np
from prediction_inference import RASI, Mean, RobustDesign, select_burn_in, select_for_labeling

rng = np.random.default_rng(42)                     # one generator across both batches keeps their sampling independent
predictions = model.predict(pool)                   # independently trained, fixed model
# Batch 1: uniformly random burn-in, all labeled (probability 1, counted in the budget)
burn = select_burn_in(len(pool), size=100, random_state=rng)
burn_labels = annotate(burn.indices)
# Between batches: fit e(x) ≈ |Y - f(x)| on the burn-in; with constraint=None, K-fold CV picks the ℓ2 radius c
rest = burn.remaining
design = RobustDesign.fit(pool[burn.indices], burn_labels, predictions[burn.indices],
                          error_model_factory=make_regressor, rho=300 / len(rest),
                          path='geometric', random_state=rng.integers(2**32))
# Batch 2: robust sampling of the remaining rows
request = select_for_labeling(predictions[rest], target=Mean(), design=design, budget=300,
                              scores=design.errors(pool[rest]), random_state=rng)
labels = annotate(rest[request.indices])            # pool row ids are rest[request.indices]
# Combine both batches -> RobustActiveData -> RASI
record = burn.combine(predictions, burn_labels, request, labels)
result = RASI()(record, target=Mean(), alpha=.05)
```

- `select_burn_in(n, size=...)` returns a `BurnIn`: `indices` are the burn-in rows and `remaining` the rows entering the second batch; it reads no labels.
- `RobustDesign` is an ordinary `SamplingDesign`. `fit` uses the burn-in labels only between the batches; `calibrate` reads only the label-free scores `design.errors(...)`: the active rule `pi = clip(eta * u, 0, 1)`, `eta = rho / mean(u)`, with u the prediction error; it then grid-searches ρ along a budget-preserving path (`geometric` by default, also `linear` and `hellinger`) to minimize `mean(e^2/pi) + c*sqrt(mean(1/pi^2))`, the closed-form inner problem of the ℓ2-constrained minimax, matching the objective of the official `MinMaxOptimizer_l2`.
- `BurnIn.combine` merges both batches into `RobustActiveData` (in `data.py`, inheriting from `ActiveData` and additionally recording `burn_in`, `rho_robust`, `constraint` and `path`). Burn-in rows have propensity 1, so `RASI` reuses ASI's IPW correction directly, and the point estimate equals the official "mean of burn-in labels concatenated with the remaining AIPW values".
- The convenience function `robust_active_label(pool, oracle=..., target=Mean(), model=..., error_model_factory=..., budget=400, burn_in=100)` runs all the steps above with an oracle for simulations; with the same `random_state` it matches the step-by-step version.

Differences from the official implementation: the cross-validation criterion uses a closed form for the expected standard deviation of the AIPW values instead of the official 200 Monte Carlo draws; active scores are floored at `1e-3 * mean(u)` so every row has positive probability (the official code skips rows with probability 0, which breaks unbiasedness); the default candidate radii are `(0, .01, ..., .07, .1)`, versus `(.02, ..., .07)` in the official Pew experiment; only the paper's main rule π ∝ prediction error is implemented, not the π ∝ `min(p, 1-p)` variant of Figure 2. Only binary means are currently supported.

## Formulas and scope

ASI uses `Z_i = f_i + R_i / pi_i * (Y_i - f_i)`, with point estimate `mean(Z)` and standard error `std(Z, ddof=1) / sqrt(n)`. The private `ipw_correct` only evaluates the formula and reads labels at queried positions; data validation is done by `ActiveData`.

ASI keeps the population-mean, asymptotic endpoint Wald design, requiring an iid population, positive query probabilities fixed before labeling and the corresponding moment conditions. The variance formula gives no finite-sample or anytime guarantee.

PPI uses `mean(f_unlabeled) + mean(y_labeled - f_labeled)` with standard error `sqrt(var(f_unlabeled, ddof=1)/N + var(y_labeled - f_labeled, ddof=1)/n)`. The two terms come from independent inference samples, and the prediction model must be trained independently. PPI++ generalizes this by multiplying the prediction terms by λ; see the PPI++ section.

## Sequential version (simulation only)

The core library does not yet cover sequential variants. The sequential setting of the ASI paper (Algorithm 2: update the model while labeling) is kept only in `examples/simulations/online.py`, for the `main.py` simulations:

- `active_label(features, oracle=..., target=Mean(), budget=..., model=..., update=..., update_every=..., update_on=...)`: updates the model and rescores the remaining rows per block, or every time `update_every` labels have been collected (`update_on='labels'`); `update='refit'` passes all queried data so far, `'finetune'` only the new data, and `update='none'` keeps the model fixed. `oracle(index)` is called once, only when that row is selected.
- `Exploring(tau, explore)`: adds "use rho for the first explore rows" on top of the core `Uncertainty`.
- `BudgetCapped(tau, explore, greedy_steps)`: eq. (8) of the paper; raw probabilities never exceed the remaining virtual budget `rho*(t+1) - queried so far`; with `greedy_steps>0`, the last greedy_steps rows of every `int(greedy_steps/rho)`-row window spend the remaining budget outright.

## Repeated experiments and tests

```sh
PYTHONPATH=src:. python main.py --reps 100
python -m pytest
```

By default the three strategies `asi`, `asi_judge` and `asi_pretrained` run 100 times each, using the sequential version above. Simulation defaults match the paper's sequential experiments: `--update-on labels --update-every 100`, `--asi-budget-cap 1 --greedy-steps 100`, τ=0.5. `--tune-tau 1` tunes τ by eq. (12) on independent historical data: `asi_judge` uses the judge model's predictions on `pretrain_size` historical samples, and `asi_pretrained` trains on 80% of the pretraining data and holds out 20%; the cold-start `asi` has no historical data and keeps `--asi-tau`. `asi_pretrained` draws `pretrain_size` initial labels from an independent seed stream and then refits online on the queried labels; pretraining labels are recorded separately in the `pretrain_labels` column and do not count towards the online expected budget.

`results_asi/config.json` records the parameters; `per_rep_final.csv` holds the 300 per-repetition results; `fixed_budget_comparison.csv` holds the summary of the three strategies.

The tests are split into eight files:

- `common/test_core.py`: the ASI correction formula and the values received by the interval function, η and the budget floor of `Uncertainty`, `tune_tau`, `LabelRequest` containing only selection information with `with_labels` checking the label count, the PPI formula and split, and coverage over 200 DGP repetitions with fixed predictions.
- `common/test_bai.py`: cube sampling satisfies the balancing constraints with unchanged inclusion probabilities, the sample size under `fixed_size`, records with default and signed balancing variables, the variance function matching the official `estimate_cube_variance` value by value, BAI and ASI point estimates agreeing along with the variance decomposition, and coverage over 150 DGP repetitions.
- `common/test_cluster_ppi.py`: the estimate, SE and tuned λ against hand-computed cluster sums, singleton clusters recovering PPI++, λ=0 giving the labels-only mean with the tuned SE no larger than λ=0 or λ=1, and coverage over 200 repetitions under unit-level and whole-cluster labeling, with i.i.d. PPI++ undercovering.
- `common/test_fab_ppi.py`: the horseshoe posterior mean against numerical integration, the Gaussian region against the official closed-form roots, symmetry of the spending function, the horseshoe region being shorter at 0 and reverting to the z-interval for large y while the Gaussian one does not, the posterior mean lying inside the region, region coverage under W ~ N(β,1), the method's point estimate and the interval formula with `imputed_uncertainty=False`, λ for FAB-PPI++, and coverage over 200 DGP repetitions.
- `common/test_ftrl.py`: query probabilities matching the official `compute_pt` loop value by value, rising monotonically from β to τ without exceeding the budget, `p_t` unaffected by the current row, and coverage over 200 DGP repetitions.
- `common/test_ppi_plus.py`: λ̂, the point estimate and the standard error against hand calculations, λ=0 giving the classical mean and λ=1 PPI's point estimate, λ=0 for constant predictions, the tuned standard error no larger than either baseline, λ̂ < 0 for anticorrelated predictions, and coverage over 200 DGP repetitions.
- `common/test_rasi.py`: endpoints and budgets of the three paths, `RobustDesign` per-row probabilities reproducing `budget_path`, `scores=` not changing the default selection, the step-by-step two-batch version matching the convenience function, pointwise agreement with the official minimax objective, ρ=0 at c=0 and ρ increasing monotonically towards 1 as c grows, queries happening only at selected rows, RASI and ASI point estimates agreeing, and coverage over 150 DGP repetitions.
- `common/test_online.py`: query timing of the sequential version, history retention, pretraining updates, updates triggered by label counts, `Exploring` and periodic budget spending, and agreement with `select_for_labeling` under a fixed model.

If the current Anaconda environment crashes while loading the readline debugging dependency, use `python -m pytest -p no:debugging`.

Planned next: configuration splitting, Quantile/OLS and continuous-label Mean, GLM/M-estimation versions of PPI++, moving the sequential version into the core library, ConfidenceSequence and more checks of valid combinations.

## Citation

If you use OpenPPI in your research, please cite:

```bibtex
@software{openppi,
  title = {OpenPPI: An Open-Source Framework for Prediction-Powered Inference},
  author = {Gao, Hongfu and Zhou, Wang},
  url = {https://github.com/hongfu001/OpenPPI},
  year = {2026}
}
```
