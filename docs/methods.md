# OpenPPI method reference

Detailed notes on every method in OpenPPI: layout, interfaces, formulas, differences from the official implementations, and validation. See the [README](../README.md) for an overview.

## Layout

```text
openppi/                     # installable core (NumPy and SciPy only)
├── __init__.py              # public interface
├── sampling/                # fixed-budget, fixed-model sampling: one file per mechanism
│   ├── base.py              # SamplingDesign interface, LabelRequest, select_for_labeling
│   ├── uniform.py           # Uniform: uniform sampling (PPI baseline)
│   ├── uncertainty.py       # Uncertainty (ASI's rule), tune_tau
│   ├── robust.py            # RobustDesign, BurnIn, select_burn_in, paths and robust ρ, cross-validated c (RASI)
│   ├── balanced.py          # BalancedDesign, cube_sample (pure-NumPy cube method; BAI)
│   ├── ftrl.py              # FTRLDesign (no-regret query probabilities oblivious to the current row)
│   └── allocation.py        # multi_allocate, Allocation (how often to query each subset of sources; MultiPPI)
├── inference/               # estimators and intervals: one file per estimator
│   ├── base.py              # InferenceMethod
│   ├── ipw.py               # IPW: f + R/pi (Y - f) (ASI, RASI, FTRL)
│   ├── two_sample.py        # TwoSample(lam=1 | 'tuned' | number), tuned_lambda (PPI, PPI++)
│   ├── misclassification.py # RoganGladen, MisclassificationMLE (hard binary judges)
│   ├── adaptive.py          # PSPA (adaptive mean inference)
│   ├── recalibrated.py      # RePPI, BinnedMean, three_folds
│   ├── fab.py               # FABPPI and the FAB formulas: posterior means, spending function, FAB region
│   ├── cluster.py           # ClusterRobustPPI, cluster_moments
│   ├── multi.py             # MultiPPI
│   └── balanced.py          # BAI, cube_variance
├── common/
│   ├── data.py              # ActiveData, PPIData, ClusteredPPIData, MultiPPIData, BalancedActiveData; split()
│   ├── targets.py           # Target, Mean; the single definition of sampling scores
│   ├── results.py           # InferenceResult
│   └── stats.py             # shared formulas: ipw_correct, wald_interval, ledoit_wolf, multi_weights
└── methods.py               # registry: Recipe, METHODS, get_method (composition only)

examples/simulation/         # run everything from this folder
├── dataset/judge_sim.py     # DGP, make_judge_data, make_multi_rater_data, make_clustered_data
├── model/                   # learners.py: prediction model wrappers; judge.py: the judge mean model
├── *.ipynb                  # one executed notebook per method (ppi, ppi_plus_plus, asi, ...)
├── coverage.py              # repeated ASI experiment (command line, policies, summary)
├── comparison.ipynb         # plots of coverage.py output
├── online.py                # sequential ASI (simulation only, see below); tested by test_online.py
├── config.py, utils.py      # experiment configuration; seeding and output helpers
└── results_asi/             # output of coverage.py with the default configuration
examples/real_data/          # (to come)
tests/                       # unit and coverage tests
```

- `dataset/judge_sim.py`: `make_judge_data` returns a DataFrame; `make_multi_rater_data` gives a binary label with autoraters of different accuracy; `make_clustered_data` gives units nested in iid clusters with a biased predictor (unit-level mean 0.5).
- The notebooks and scripts in `examples/simulation/` run from that folder, so `import dataset`, `import model`, `import config` and `import utils` work directly; the tests add that folder to the path through `pyproject.toml`.

Sampling files are named by mechanism and inference files by estimator; method names appear only in `methods.py`, so a component shared by several methods (IPW by ASI, RASI and FTRL) is not named after one of them.

Dependency direction: examples may use every layer; `openppi` never imports from `examples/`; within `openppi`, `sampling/` and `inference/` both build on `common/` and do not import each other, and `methods.py` imports both.

## Methods, components and interface

A method is a sampling component plus an inference component. `openppi/methods.py` records each published combination as a `Recipe(name, sampling, inference, data_type, reference)`; `get_method(name)` returns it and `METHODS` lists them all. The file only records composition. `sampling` is `None` for two-sample methods, whose samples are split in advance, and otherwise a class, factory or function that builds a fresh design on every use, so no state is shared between runs (`RobustDesign` is registered as a class because it is fitted on burn-in labels). `inference` is a configured, stateless estimator.

| Method | `sampling` | `inference` | `data_type` |
| --- | --- | --- | --- |
| PPI | `None` | `TwoSample(lam=1.0)` | `PPIData` |
| PPI++ | `None` | `TwoSample(lam='tuned')` | `PPIData` |
| RG | `None` | `RoganGladen()` | `PPIData` |
| MLE | `None` | `MisclassificationMLE()` | `PPIData` |
| PSPA | `None` | `PSPA()` | `PPIData` |
| RePPI | `None` | `RePPI()` | `PPIData` |
| FAB-PPI | `None` | `FABPPI()` | `PPIData` |
| Cluster-PPI | `None` | `ClusterRobustPPI()` | `ClusteredPPIData` |
| MultiPPI | `multi_allocate` | `MultiPPI()` | `MultiPPIData` |
| ASI | `partial(Uncertainty, tau=.5)` | `IPW()` | `ActiveData` |
| RASI | `RobustDesign` (fitted after a burn-in) | `IPW()` | `ActiveData` |
| FTRL | `FTRLDesign` | `IPW()` | `ActiveData` |
| BAI | `BalancedDesign` | `BAI()` | `BalancedActiveData` |

```python
from openppi import BAI, IPW, ClusterRobustPPI, FABPPI, MultiPPI, RePPI, TwoSample, Mean, get_method

# record is ActiveData (ASI, RASI, FTRL), balanced is BalancedActiveData, data is PPIData,
# clustered is ClusteredPPIData, multi is MultiPPIData.
# result = IPW()(record, target=Mean(), alpha=.05)
# result = BAI()(balanced, target=Mean(), alpha=.05)
# result = TwoSample()(data, target=Mean(), alpha=.05)               # PPI
# result = TwoSample(lam='tuned')(data, target=Mean(), alpha=.05)    # PPI++
# result = FABPPI()(data, target=Mean(), alpha=.05)
# result = ClusterRobustPPI()(clustered, target=Mean(), alpha=.05)
# result = MultiPPI()(multi, target=Mean(), alpha=.05)
# result = RePPI()(data, target=Mean(), alpha=.05)
# result = get_method('ASI').inference(record, target=Mean(), alpha=.05)   # the same through the registry
```

Internal steps (`active_rule`, `budget_path`, `robust_rho`, `select_constraint`, `select_burn_in`, `cube_sample`, `ledoit_wolf`, ...) are not exported at the top level; import them from their submodule, e.g. `from openppi.sampling import select_burn_in`.

`InferenceMethod` lives in `openppi.inference.base` (exported as `openppi.InferenceMethod`). It checks the data type, target type and alpha, then calls the subclass's `_estimate(data, target)`. The subclass returns `(point, se, diagnostics)` and may put `values` into diagnostics; the base class passes these to the interval function and removes `values` from the final diagnostics.

`InferenceResult` (`openppi.common.results`) has six fields: `estimate`, `standard_error`, `confidence_interval`, `alpha`, `method` and `diagnostics`.

The Wald formula in `openppi.common.stats` is the default. A custom interval function has the signature `interval(point, se, alpha, values=None)` and is passed as e.g. `IPW(interval=...)` or `TwoSample(interval=...)`. For IPW, values is the array of corrected values; for `TwoSample` it is a dict with the arrays `residuals` (`Y - λf`) and `unlabeled_predictions` (`λf̃`).

The old `ASI`, `RASI`, `PPI` and `PPIPlusPlus` classes are replaced by `IPW` and `TwoSample`; RASI's design parameters now travel in `ActiveData.design_info` (written by `BurnIn.combine`) and `IPW` copies them into the diagnostics, so `RobustActiveData` and the `robust_active_label` wrapper are gone. `openppi.common.stats` holds shared formulas and is not part of the top-level interface.

## Fixed-budget sampling

The core library only considers a fixed model and a fixed budget: predict with the model, decide the sampling probabilities once, then select the rows to label. Each sampling design lives in its own file and inherits from `SamplingDesign`; `calibrate(reference_scores, rho)` calibrates it and `probability(scores)` returns probabilities fixed before labeling. `sample(scores, propensities, rng)` and `record(...)` are two overridable hooks, defaulting to independent Bernoulli sampling and returning `ActiveData`; BAI overrides them to sample jointly with the cube method and return `BalancedActiveData`:

| Design | File | Behavior |
| --- | --- | --- |
| `Uniform()` | `uniform.py` | Constant probability `rho = budget / n`; with `IPW` this is the PPI baseline |
| `Uncertainty(tau=.5)` | `uncertainty.py` | `pi = (1-tau) * clip(eta*u, 0, 1) + tau*rho`, `eta = rho / mean(u)`, u defaults to `2*min(p, 1-p)` (Algorithm 1 of the ASI paper); floor `tau * rho`; clipping can leave the expected number of labels slightly below the budget |
| `BalancedDesign(tau=.5)` | `balanced.py` | BAI: the same probabilities as `Uncertainty`, sampled jointly with the cube method so that `sum(x * xi / pi) ≈ sum(x)` (see the BAI section) |
| `FTRLDesign()` | `ftrl.py` | FTRL: `p_t` depends only on Φ of earlier rows and rises from `tau/8` to `tau = budget/n` (see the FTRL section) |
| `RobustDesign` | `robust.py` | RASI second batch: active rule π ∝ prediction error, interpolated towards uniform along a budget-preserving path (see the RASI section) |

`Mean.sampling_scores` is the only place where `2*min(p, 1-p)` is defined.

The flow of selecting, labeling and assembling a record for one batch is shared by all designs (full example in `examples/simulation/asi.ipynb`):

```python
from openppi import IPW, Mean, Uncertainty, select_for_labeling

predictions = model.predict(features)          # any already trained, fixed model
request = select_for_labeling(predictions, target=Mean(), design=Uncertainty(tau=.5),
                              budget=400, random_state=42)
labels = annotate(request.indices)             # export row ids to annotators; labels come back in the same order
record = request.with_labels(labels)           # -> ActiveData
result = IPW()(record, target=Mean(), alpha=.05)
```

`select_for_labeling` reads no labels. It calibrates a copy of the design on this batch's scores with `rho = budget / n` and returns a `LabelRequest`: `predictions`, `propensities` and boolean `queried` for all n rows (copied and read-only), plus the calibrated design copy `design` (diagnostics only, no labels). The optional `scores=` replaces the default `target.sampling_scores`, e.g. prediction errors for RASI's second batch. `request.indices` are the rows to label; `request.with_labels(labels)` requires labels in one-to-one correspondence with `indices`, otherwise it raises, and returns `ActiveData`. In `ActiveData` the predictions, observed labels, query flags and query probabilities are aligned and unqueried labels must be NaN; the arrays are copied and protected on construction.

With independent historical labeled data, τ can be chosen by eq. (12) of the ASI paper: `tune_tau(historical_predictions, historical_labels, target=Mean(), rho=budget/n)` minimizes `mean((Y-f)^2 / pi_tau)` over `np.linspace(.01, .99, 100)`. The historical predictions should come from a model not trained on those labels (e.g. a held-out part of the training set).

## Balanced active inference, BAI (one batch)

Implements Algorithm 1 of Chen, Zhou, Peng & Wang (NeurIPS 2025), following the official code `Uninfty/Balanced_Active_Inference`. The sampling probabilities are the same as ASI's (the τ mixture rule; the paper's τ is the weight of the active part, i.e. `1 - tau` here, and the two agree at 0.5), but rows are no longer sampled independently: the cube method (Deville & Tillé 2004) selects them jointly, keeping each row's inclusion probability at π while satisfying the balancing constraint `sum(x * xi / pi) ≈ sum(x)`. Full example in `examples/simulation/bai.ipynb`:

```python
from openppi import BAI, BalancedDesign, Mean, select_for_labeling

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

with defaults `β = τ/8` and `γ = 1/√n`, as in the official code. Φ_t ≈ `E[(Y - f)^2 | X]` is passed as `scores` (for binary labels with calibrated f, `f(1 - f)` works; the official code fits squared residuals with linear regression). `p_t` depends only on Φ of the rows before t, so with a fixed model the whole sequence can be computed before labeling, making it an ordinary `SamplingDesign`. The paper's estimator and interval are the same as ZC24's, so inference uses `IPW()` directly:

```python
from openppi import IPW, FTRLDesign, Mean, select_for_labeling

request = select_for_labeling(predictions, target=Mean(), design=FTRLDesign(),
                              scores=predictions * (1 - predictions), budget=400, random_state=42)
result = IPW()(request.with_labels(annotate(request.indices)), target=Mean(), alpha=.05)
```

Row order is arrival order (the official code shuffles on every trial). Since `p_t ≤ τ`, the expected number of labels never exceeds the budget; in practice `p_t` reaches τ within a row or two and the design then equals uniform sampling, which is exactly the paper's conclusion. The paper's full Algorithm 1 also updates the model and Φ during the sequential process; that is a sequential variant and is not yet in the core library.

## Independent two-sample PPI (four steps)

```python
from sklearn.ensemble import RandomForestRegressor
from dataset.judge_sim import make_judge_data
from openppi import Mean, PPIData, TwoSample, split

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
result = TwoSample()(data, target=target, alpha=.05)   # lam = 1: PPI
# 4. Output
print(result.estimate, result.confidence_interval, result.diagnostics)
```

`split` returns `(X_train, y_train)`, `(X_labeled, y_labeled)` and `X_unlabeled`. The size arguments are row counts; the remaining rows form the unlabeled sample, and each part has at least two rows. It does not stratify on labels and does not return unlabeled labels.

Rows must come from an iid population and the three parts must not overlap. The prediction model is fit only on the independent training set and fixed before inference. `PPIData` requires the labels and predictions to be aligned and each inference sample to have at least two rows; all arrays are copied and made read-only. True labels are currently binary only.

## PPI++ (power-tuned two-sample PPI)

Implements the mean version of Angelopoulos, Duchi & Zrnic (2023) (Example 6.1 and Algorithm 2 of the paper), following `ppi_mean_test` in `yiqunchen/pppower`. It uses the same `PPIData` as PPI (full example in `examples/simulation/ppi_plus_plus.ipynb`):

```python
from openppi import Mean, TwoSample

result = TwoSample(lam='tuned')(data, target=Mean(), alpha=.05)   # PPI++: plug-in λ
result.diagnostics['prediction_weight']                           # the λ actually used
TwoSample(lam=0)(data, target=Mean())                             # classical estimate: labels only
TwoSample(lam=1)(data, target=Mean())                             # PPI
TwoSample(lam=1, variance='pooled')(data, target=Mean())          # PPI's estimate, PPI++'s SE
```

- **Point estimate**: `mean(Y) + λ * (mean(f_unlabeled) - mean(f_labeled))`, unbiased for any λ.
- **λ**: by default `λ̂ = Cov_n(Y, f) / ((1 + n/N) * Var_{N+n}(f))` (`tuned_lambda`), which minimizes the asymptotic variance. The more accurate the predictions, the closer λ̂ is to 1; useless predictions give λ̂ near 0 (the classical estimate); predictions anticorrelated with labels give a negative λ̂. Mean estimation is convex for every real λ, so λ̂ is not clipped to [0, 1]; constant predictions give λ = 0.
- **Standard error**: `sqrt(λ² Var(f) / N + Var_n(Y - λf) / n)`. With `variance='pooled'`, Var(f) is taken on both samples, Algorithm 2's `Σ̂ = (n/N) V̂_f + V̂_Δ` divided by n; with `variance='unlabeled'`, on the unlabeled sample as in PPI. The default is `'unlabeled'` for λ = 1 (so `TwoSample(lam=1)` is exactly PPI) and `'pooled'` otherwise (so `TwoSample(lam='tuned')` is exactly PPI++); the result's `method` is `'PPI'` or `'PPI++'` accordingly.
- **Difference from pppower**: pppower uses only the labeled sample for Var(f) in λ̂ and the variance; following the paper, both samples are pooled here, which is nearly identical when N ≫ n. The two variance forms differ only slightly; e.g. at λ = 1 on the judge data the SEs are 0.0240764 (PPI) and 0.0240808 (pooled).
- **Validation**: in `examples/simulation/ppi_plus_plus.ipynb`, λ̂ = 0.536 and the 95% interval widths are 0.0981 (classical), 0.0965 (PPI) and 0.0933 (PPI++).

## RePPI (recalibrated predictions)

Implements recalibrated PPI of Ji, Lei & Zrnic (*Predictions as Surrogates*, 2026), Algorithm 1 with the squared loss, following `reppi.py` in the official code `Wenlong2000/RePPI`. It uses the same `PPIData` as PPI (full example in `examples/simulation/reppi.ipynb`):

```python
from sklearn.ensemble import RandomForestRegressor
from openppi import RePPI, Mean

result = RePPI(random_state=0)(data, target=Mean(), alpha=.05)              # binned recalibration
result = RePPI(learner_factory=lambda: RandomForestRegressor(min_samples_leaf=20),
               random_state=0)(data, target=Mean())                         # any fit/predict learner
```

- **Idea**: the most efficient PPI estimator imputes the loss gradient with `E[∇ℓ | X, Ŷ]` (eq. 8, from the surrogate-outcome literature), not with the raw prediction. For the mean this is the recalibrated prediction `m(Ŷ) = E[Y | Ŷ]`; PPI++ only applies a linear correction to Ŷ, so RePPI gains whenever E[Y | Ŷ] is non-linear in Ŷ, e.g. for coarse or skewed categorical ratings (Section 4.3).
- **Algorithm**: the labeled rows are split at random into three folds. In each rotation one fold fits `m` with `learner_factory()` (the paper's θ₀ fold only shifts the score by a constant for the mean) and another does inference: `θ_k = mean(Y) + a_k (mean_U m − mean_k m)` with `a_k = Cov_k(Y, m) / Var_k(m) / (1 + n/N)`, the power tuning of eq. (13) applied to the recalibrated prediction. The estimate averages the three folds. Cross-fitting keeps every labeled row's recalibrated value out of sample, so flexible learners do not bias the interval (Appendix B).
- **Standard error**: as in the official code, `sqrt(var(Y − a m̃) / n + var(m̄_U) / N)`, where m̃ are the out-of-fold `a_k m` on the labeled rows and m̄_U averages `a_k m_k` over the three learners on the unlabeled rows. Diagnostics give `fold_estimates` and `fold_weights`.
- **Learner**: the default `BinnedMean` (NumPy only) averages labels within prediction bins, one bin per value for predictions with at most 10 distinct values, otherwise about n^(1/3) quantile bins. Any object with `fit(features, labels)` and `predict(features)`, such as a scikit-learn regressor, can be passed; features are the predictions as an n × 1 array.
- **Differences from the official implementation**: the folds are combined with equal weights as in the official code (the paper weights by fold size; identical when n is divisible by 3). `PPIData` carries no covariates, so recalibration uses the prediction only; this is the full optimal score for the mean (Example 1) but not for regression targets. The GLM/M-estimation version, extra covariates and the distribution-shift reweighting of Appendix G are not implemented.
- **Validation**: on the paper's discrete-prediction model with levels 0, .5, 1 and P(Y = 1) = .1, .2, .9, the standard errors of RePPI and PPI++ match their theoretical values `sqrt((Var Y − Var(E[Y|Ŷ])/(1+r))/n)` and `sqrt((Var Y − Cov(Y,Ŷ)²/Var(Ŷ)/(1+r))/n)` within 2%. In `examples/simulation/reppi.ipynb`, this model (n = 500, N = 4500, 200 repetitions) gives mean 95% widths of 0.0696 (PPI), 0.0663 (PPI++) and 0.0625 (RePPI), about 12% fewer labels than PPI++ for the same width, with coverage near 0.95. For a trained model's 5-point rating, E[label | rating] is close to linear and RePPI matches PPI++ (width 0.0747 vs 0.0742): recalibration costs nothing when there is nothing to gain.

## FAB-PPI (a prior on the rectifier)

Implements the mean version of Cortinovis & Caron (ICML 2025) (Algorithm 2 in the supplementary material), following `ppi_fab_mean_ci` in the official code `stefanocortinovis/fab-ppi`. It uses the same `PPIData` as PPI (full example in `examples/simulation/fab_ppi.ipynb`):

```python
from openppi import FABPPI, Mean

result = FABPPI()(data, target=Mean(), alpha=.05)                  # FAB-PPI, horseshoe prior
result = FABPPI(lam=None)(data, target=Mean(), alpha=.05)          # FAB-PPI++, λ as in PPI++
result = FABPPI(prior='gaussian')(data, target=Mean())             # Gaussian prior (not robust, for comparison only)
result = FABPPI(imputed_uncertainty=False)(data, target=Mean())    # the paper's experiments: ignore the uncertainty in mean(f̃) when N ≫ n
```

- **Rectifier**: `Δ̂ = mean(Y - λf) + (λ - 1) mean(f̃)`, `σ̂² = Var_n(Y - λf)/n + (λ-1)² Var(f̃)/N` (sign as in the official code, "labels minus predictions"; the prior is symmetric, so this is equivalent to the paper's `f - Y`).
- **Prior**: the default is the horseshoe prior `Δ ~ N(0, ν²σ̂²)`, `ν ~ C⁺(0, 1)`, with scale σ̂ (the paper's parameter-free choice). In standardized units `y = Δ̂/σ̂` the posterior mean is `y(1 - κ(y))` with `κ = (2/3) ₁F₁(2, 5/2, -y²/2) / ₁F₁(1, 3/2, -y²/2)`; for `|y| ≥ 4` the equivalent Dawson form `1/(√2 D(y/√2)) - 1/y` is used for numerical stability. The Gaussian prior `N(0, σ̂²)` has posterior mean y/2 and is inconsistent (Proposition 4.3 of the paper); it is included only for comparison.
- **Point estimate**: `mean(f̃) + σ̂ · E[Δ/σ̂ | y]`. `standard_error` reports the frequentist standard error of the unshrunk PPI estimate, `sqrt(σ̂² + Var(f̃)/N)`, for reference only; the interval is not of Wald form.
- **Interval**: `[mean(f̃) + σ̂·L - pad, mean(f̃) + σ̂·U + pad]`, where `[L, U]` is the 1-δ FAB region and `pad = sqrt(Var(f̃)/N) · z_{1-(α-δ)/2}` (the Minkowski sum in eq. 21). The FAB region comes from the spending function w(β), which equalizes `π(y)/φ(y-β)` at the two ends of the acceptance region (Neyman–Pearson); the region is `{β : β + z_{α(1-w)} ≤ y ≤ β + z_{1-αw}}`, solved with nested brentq in about 2 ms. For large |y| the horseshoe region reverts to the z-interval (Proposition 4.2), while the Gaussian region's width grows like 2|y|/3.
- **δ**: with `delta=None`, as in the official code, δ minimizes the interval width over (0, α); a fixed value can also be passed. `imputed_uncertainty=False` sets δ = α and drops the pad, matching Section 5 of the paper ("N large enough to ignore the uncertainty in m_θ"); it is only appropriate when N ≫ n.
- **Differences from the official implementation**: the official code clips λ̂ to [0, 1]; here, consistently with `TwoSample(lam='tuned')`, it is not clipped (eq. 14 of the paper has no clipping). The official pad uses `Σ(λf̃ - mean f̃)²/(N-1)/N`, which contains a bias term when λ ≠ 1; here it is `Var(f̃)/N` as in the paper's Algorithm 2. Only the horseshoe prior the paper recommends and the Gaussian comparison prior are implemented; scaled horseshoe, Laplace and other priors needing numerical integration are not.
- **Validation**: the horseshoe posterior mean agrees with numerical integration to within 1e-6; the Gaussian-prior region matches the official closed-form roots point by point; under W ~ N(β, 1) the FAB region has about nominal coverage; coverage over 200 DGP repetitions passes. In `examples/simulation/fab_ppi.ipynb`, N = 1200 and n = 400, so the uncertainty in mean(f̃) is not negligible and the full FAB-PPI interval (width 0.121) is wider than PPI's (0.097); the paper's gains come from the `imputed_uncertainty=False` setting with N ≫ n (0.079 in this example, but N is not large enough, so it is shown for comparison only).

## Cluster-Robust PPI++ (clustered and partially labeled data)

Implements the mean version of Broska & Howes (2026) (Theorem 1 and eq. 12), following `ppi_mean_ci_cluster` in the official code `Michael-Howes/ppi_py`. Units are nested in iid clusters and may be arbitrarily dependent within a cluster; clusters can be fully labeled, partially labeled or fully unlabeled, so labeled and unlabeled units are not independent and `PPIData` does not apply. `ClusteredPPIData` holds one row per unit (full example in `examples/simulation/cluster_ppi.ipynb`):

```python
import numpy as np
from openppi import ClusterRobustPPI, ClusteredPPIData, Mean

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
- **Singleton clusters** recover PPI++: the estimate, λ̂ and SE match `TwoSample(lam='tuned')` up to the ddof convention of the sample moments.
- **Not implemented**: the M-estimator version (Appendix A), coordinate-wise λ, sample weights, and the design-planning variance formulas of Appendix C.
- **Validation**: on `make_clustered_data` with 300 clusters and 200 repetitions, the cluster-robust interval covers at the nominal rate under both unit-level and whole-cluster labeling, while i.i.d. PPI++ undercovers. In `examples/simulation/cluster_ppi.ipynb` (labeling probability 0.3), the 95% widths are: partially labeled, 0.075 (labels only), 0.044 (i.i.d. PPI++, too narrow) and 0.072 (Cluster-Robust PPI++); whole clusters, 0.125, 0.044 and 0.075. As in the paper, PPI helps most when labels are concentrated in few clusters.

## MultiPPI (several predictors with different costs)

Implements Multiple-Prediction-Powered Inference of Cowen-Breen, Agarwal, Bates, Cohen, Eisenstein, Globerson & Fisch (ICLR 2026). The paper has no public code; this is a reproduction from Section 4 and Appendices C, F.1, G and H. Let X = (X_1, ..., X_k) stack the label (index 0 here) and k - 1 predictions, e.g. autoraters of different size or thinking budget. Any subset I of sources can be queried at cost c_I (costs may be non-additive), and the goal is a^T E[X], by default the mean of the label. Full example in `examples/simulation/multippi.ipynb`:

```python
import itertools
from openppi import MultiPPI, MultiPPIData, Mean, multi_allocate
from openppi.common.stats import ledoit_wolf

full = labeled_rows                          # N x k: label and every prediction (Appendix C.3)
sigma = ledoit_wolf(full)                    # covariance estimate, reused for estimation
subsets = [(0, 1, 2, 3)] + [s for r in (1, 2, 3) for s in itertools.combinations((1, 2, 3), r)]
costs = {s: 0. if 0 in s else sum(price[i] for i in s) for s in subsets}
allocation = multi_allocate(sigma, costs, budget=300., caps={(0, 1, 2, 3): len(full)})
samples = {(0, 1, 2, 3): full, **{s: query(s, n) for s, n in allocation.counts.items() if 0 not in s}}
result = MultiPPI(covariance=sigma)(MultiPPIData(samples), target=Mean(), alpha=.05)
```

- **Estimator**: with n_I iid draws of X_I for each subset, `θ̂ = Σ_I mean(X_I) · λ_I` (eq. 5). For fixed counts the minimum-variance unbiased weights are `λ_I = n_I Σ_I⁻¹ P_I S a` with `S = (Σ_I n_I P_Iᵀ Σ_I⁻¹ P_I)⁺`, and the variance is `aᵀ S a` (`multi_weights`; Appendix F.1). They satisfy `Σ_I P_Iᵀ λ_I = a`, so θ̂ is unbiased for any fixed covariance; a wrong covariance only costs efficiency (Theorems 3 and 4).
- **Allocation**: `multi_allocate(covariance, costs, budget, a=None, caps=None)` minimizes `V(n) = aᵀ (Σ_I n_I P_Iᵀ Σ_I⁻¹ P_I)⁺ a` over continuous n ≥ 0 under linear budgets `Σ_I c_I n_I ≤ B` (a vector cost and budget give several constraints, Appendix B) and per-subset caps such as `n_full ≤ N`, then floors n (Appendix C.2) and drops subsets with fewer than two draws. V is convex in n (Appendix G), so SciPy's SLSQP on log V finds the optimum; the paper's single-budget SOCP and multi-budget SDP are both this problem, solved without cvxpy. `Allocation` holds `counts`, `weights`, `variance` (for the floored counts) and `fractional`.
- **Data**: `MultiPPIData(samples)` maps each index tuple to an n_I × |I| array, columns in sorted index order, independent across subsets. The subset with every index holds the fully labeled rows. Labels must be binary.
- **Covariance and interval**: `MultiPPI()` estimates Σ on the fully labeled rows with Ledoit–Wolf (the paper's choice; `ledoit_wolf` matches sklearn) or `covariance='empirical'`, or uses a given matrix. As in Section 4.2.1 the labeled rows are reused for estimation; this adds finite-sample bias but keeps asymptotic normality (Theorem 3). The interval is `θ̂ ± z · sqrt(Σ_I var(X_I · λ_I) / n_I)` (Appendix C.2). Diagnostics give `counts`, `weights` and `planned_variance`.
- **Special cases**: subsets `{(0, j), (j,)}` give scalar PPI++ with rater j, `{full, models}` vector PPI++ and `{(0,)}` the classical mean, so the baselines are MultiPPI with a restricted collection of subsets. With the sample covariance of the labeled rows and the pooled prediction variance, `{(0, 1), (1,)}` reproduces `TwoSample(lam='tuned')` (PPI++) exactly.
- **Note on Appendix H**: the dual SOCP should read `sup aᵀy s.t. y_Iᵀ Σ_I⁻¹ y_I ≤ c_I`; the version printed with `Σ_I` does not match the primal. The tests check `V = U² / B` against this dual.
- **Not implemented**: other linear targets are supported through `a`, but only the label mean is wired to `Mean`; the restricted subset collection of Appendix D.4 is left to the caller.
- **Validation**: `examples/simulation/multippi.ipynb` (200 labeled rows, budget 300, raters costing 0.05, 0.3 and 1, 300 repetitions) gives RMSE relative to classical of 95% (PPI++ with the cheap rater), 89% (PPI++ with the best rater), 87% (vector PPI++) and 83% (MultiPPI), all with coverage about 0.95, matching the paper's finding that MultiPPI beats every baseline by combining raters.

## Robust active inference, RASI (two batches)

Implements Algorithm 1 of Li, Zrnic & Candès (NeurIPS 2025), following the official code `lphLeo/Robust-Active-Statistical-Inference`. RASI labels in two batches; the second batch shares the `select_for_labeling` interface with ASI (full example in `examples/simulation/rasi.ipynb`):

```python
import numpy as np
from openppi import IPW, Mean, RobustDesign, select_for_labeling
from openppi.sampling import select_burn_in

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
# Combine both batches -> ActiveData with design_info -> IPW
record = burn.combine(predictions, burn_labels, request, labels)
result = IPW()(record, target=Mean(), alpha=.05)
```

- `select_burn_in(n, size=...)` returns a `BurnIn`: `indices` are the burn-in rows and `remaining` the rows entering the second batch; it reads no labels.
- `RobustDesign` is an ordinary `SamplingDesign`. `fit` uses the burn-in labels only between the batches; `calibrate` reads only the label-free scores `design.errors(...)`: the active rule `pi = clip(eta * u, 0, 1)`, `eta = rho / mean(u)`, with u the prediction error; it then grid-searches ρ along a budget-preserving path (`geometric` by default, also `linear` and `hellinger`) to minimize `mean(e^2/pi) + c*sqrt(mean(1/pi^2))`, the closed-form inner problem of the ℓ2-constrained minimax, matching the objective of the official `MinMaxOptimizer_l2`.
- `BurnIn.combine` merges both batches into `ActiveData` and records the design in `design_info` (`burn_in`, `rho_robust`, `constraint`, `path`), which `IPW` copies into the diagnostics. Burn-in rows have propensity 1, so the IPW correction applies directly, and the point estimate equals the official "mean of burn-in labels concatenated with the remaining AIPW values".

Differences from the official implementation: the cross-validation criterion uses a closed form for the expected standard deviation of the AIPW values instead of the official 200 Monte Carlo draws; active scores are floored at `1e-3 * mean(u)` so every row has positive probability (the official code skips rows with probability 0, which breaks unbiasedness); the default candidate radii are `(0, .01, ..., .07, .1)`, versus `(.02, ..., .07)` in the official Pew experiment; only the paper's main rule π ∝ prediction error is implemented, not the π ∝ `min(p, 1-p)` variant of Figure 2. Only binary means are currently supported.

## Formulas and scope

`IPW` (used by ASI, RASI and FTRL) computes `Z_i = f_i + R_i / pi_i * (Y_i - f_i)`, with point estimate `mean(Z)` and standard error `std(Z, ddof=1) / sqrt(n)`. The private `ipw_correct` only evaluates the formula and reads labels at queried positions; data validation is done by `ActiveData`.

It keeps the population-mean, asymptotic endpoint Wald design, requiring an iid population, positive query probabilities fixed before labeling and the corresponding moment conditions. The variance formula gives no finite-sample or anytime guarantee.

PPI (`TwoSample(lam=1)`) uses `mean(f_unlabeled) + mean(y_labeled - f_labeled)` with standard error `sqrt(var(f_unlabeled, ddof=1)/N + var(y_labeled - f_labeled, ddof=1)/n)`. The two terms come from independent inference samples, and the prediction model must be trained independently. PPI++ generalizes this by multiplying the prediction terms by λ; see the PPI++ section.

## Sequential version (simulation only)

The core library does not yet cover sequential variants. The sequential setting of the ASI paper (Algorithm 2: update the model while labeling) is kept only in `examples/simulation/online.py`, for `coverage.py`:

- `active_label(features, oracle=..., target=Mean(), budget=..., model=..., update=..., update_every=..., update_on=...)`: updates the model and rescores the remaining rows per block, or every time `update_every` labels have been collected (`update_on='labels'`); `update='refit'` passes all queried data so far, `'finetune'` only the new data, and `update='none'` keeps the model fixed. `oracle(index)` is called once, only when that row is selected.
- `Exploring(tau, explore)`: adds "use rho for the first explore rows" on top of the core `Uncertainty`.
- `BudgetCapped(tau, explore, greedy_steps)`: eq. (8) of the paper; raw probabilities never exceed the remaining virtual budget `rho*(t+1) - queried so far`; with `greedy_steps>0`, the last greedy_steps rows of every `int(greedy_steps/rho)`-row window spend the remaining budget outright.

## Repeated experiments and tests

```sh
cd examples/simulation
python coverage.py --reps 100
cd ../..
python -m pytest -m "not slow"   # quick
python -m pytest                  # including the slow coverage checks
```

By default the three strategies `asi`, `asi_judge` and `asi_pretrained` run 100 times each, using the sequential version above. Simulation defaults match the paper's sequential experiments: `--update-on labels --update-every 100`, `--asi-budget-cap 1 --greedy-steps 100`, τ=0.5. `--tune-tau 1` tunes τ by eq. (12) on independent historical data: `asi_judge` uses the judge model's predictions on `pretrain_size` historical samples, and `asi_pretrained` trains on 80% of the pretraining data and holds out 20%; the cold-start `asi` has no historical data and keeps `--asi-tau`. `asi_pretrained` draws `pretrain_size` initial labels from an independent seed stream and then refits online on the queried labels; pretraining labels are recorded separately in the `pretrain_labels` column and do not count towards the online expected budget.

`examples/simulation/results_asi/config.json` records the parameters; `per_rep_final.csv` holds the 300 per-repetition results; `fixed_budget_comparison.csv` holds the summary of the three strategies.

The tests in `tests/` are split by component; the repeated-sampling coverage checks are marked `slow`:

- `tests/test_core.py`: the IPW correction formula and the values received by the interval function, η and the budget floor of `Uncertainty`, `tune_tau`, `LabelRequest` containing only selection information, the PPI formula and `split`, and coverage over 200 DGP repetitions with fixed predictions.
- `tests/test_two_sample.py`: λ̂, the point estimate and the standard error against hand calculations; `TwoSample(lam=1)` equal to PPI's formula, `variance='pooled'` changing only the SE, and `lam=0` equal to the mean of labels; λ=0 for constant predictions; the tuned standard error no larger than λ=0 or λ=1; λ̂ < 0 for anticorrelated predictions; coverage over 200 DGP repetitions.
- `tests/test_robust.py`: endpoints and budgets of the three paths, pointwise agreement with the official minimax objective, ρ=0 at c=0 and ρ increasing monotonically towards 1 as c grows, the two-batch flow querying only selected rows and recording the design in the IPW diagnostics, `RobustDesign` per-row probabilities reproducing `budget_path`, `scores=` not changing the default selection, and coverage over 150 DGP repetitions.
- `tests/test_bai.py`: cube sampling satisfies the balancing constraints with unchanged inclusion probabilities, the sample size under `fixed_size`, the variance function matching the official `estimate_cube_variance` value by value, BAI and IPW point estimates agreeing along with the variance decomposition, and coverage over 150 DGP repetitions.
- `tests/test_ftrl.py`: query probabilities matching the official `compute_pt` loop value by value, rising monotonically from β to τ without exceeding the budget, `p_t` unaffected by the current row, and coverage over 200 DGP repetitions.
- `tests/test_cluster_ppi.py`: the estimate, SE and tuned λ against hand-computed cluster sums, singleton clusters recovering PPI++, λ=0 giving the labels-only mean with the tuned SE no larger than λ=0 or λ=1, and coverage over 200 repetitions under unit-level and whole-cluster labeling, with i.i.d. PPI++ undercovering.
- `tests/test_fab_ppi.py`: the horseshoe posterior mean against numerical integration, the Gaussian region against the official closed-form roots, symmetry of the spending function, the horseshoe region being shorter at 0 and reverting to the z-interval for large y while the Gaussian one does not, the posterior mean lying inside the region, region coverage under W ~ N(β,1), the method's point estimate and the interval formula with `imputed_uncertainty=False`, λ for FAB-PPI++, and coverage over 200 DGP repetitions.
- `tests/test_multippi.py`: `ledoit_wolf` against sklearn, the allocation against a brute-force search for two sources and against the dual SOCP for four, unbiased and variance-minimizing weights, the label-plus-one-rater case reproducing PPI++, and coverage over 200 repetitions with a smaller SE than labels only.
- `tests/test_reppi.py`: the fold estimates against a hand calculation with a fixed learner, `BinnedMean` recovering E[Y | Ŷ] for discrete predictions, the SEs of RePPI and PPI++ against their theoretical values on the discrete-prediction model, a scikit-learn learner, and coverage over 200 repetitions.
- `tests/test_methods.py`: every registry recipe's data type agrees with its inference component, and sampling components are built fresh on each use.

`examples/simulation/test_online.py` tests the simulation-only `online.py` (query timing of the sequential version, history retention, pretraining updates, updates triggered by label counts, `Exploring` and periodic budget spending, agreement with `select_for_labeling` under a fixed model); run it with `python -m pytest examples/simulation/test_online.py`.

If the current Anaconda environment crashes while loading the readline debugging dependency, use `python -m pytest -p no:debugging`.

Planned next: configuration splitting, Quantile/OLS and continuous-label Mean, GLM/M-estimation versions of PPI++, moving the sequential version into the core library, ConfidenceSequence and more checks of valid combinations.


## Post-prediction adaptive inference (PSPA)

[Assumption-Lean and Data-Adaptive Post-Prediction Inference](https://jmlr.org/papers/v26/24-0056.html), Miao, Miao, Wu, Zhao & Lu, Journal of Machine Learning Research 26(179):1–31 (2025). The implementation specializes Section 2.1 and Algorithm 1 to the project's binary mean target, checked against the official [pspa_y(method="mean")](https://github.com/qlu-lab/pspa/blob/main/R/main.R) and [optim_weights, optim_est, Sigma_cal](https://github.com/qlu-lab/pspa/blob/main/R/utils.R), commit `8e77b124d58b884730484d1923d909a61ed70025`. The Python implementation evaluates the scalar formulas using NumPy; no R runtime dependency or vendored R code is added.

```python
from openppi import Mean, get_method

# data is PPIData from independent labeled and unlabeled inference samples.
result = get_method('PSPA').inference(data, target=Mean(), alpha=.05)
print(result.estimate, result.confidence_interval)
print(result.diagnostics['prediction_weight'])
```

- **Weight**: with n labeled and N unlabeled rows, `w = clip(Cov(Y, f_L) / (Var(f_L) + (n/N) Var(f_U)), 0, 1)`. All variances and covariances use `ddof=1`, matching R's `cov`. The official implementation clips at both 0 and 1; the supplied paper's Algorithm 1 prints only the upper cap. We follow the official code, so anticorrelated predictions receive zero weight.
- **Estimate and interval**: `theta = mean(Y) + w * (mean(f_U) - mean(f_L))`, with `SE = sqrt(Var(Y - w*f_L)/n + w**2 * Var(f_U)/N)` and the shared Wald interval. This is algebraically the scalar `Sigma_cal / n` from the official package. In the mean case the one-step update solves the estimating equation exactly. Constant predictions in both samples give weight zero (the official code otherwise encounters 0/0).
- **Difference from PPI++**: PSPA uses separate labeled/unlabeled prediction variances and clips the weight to [0, 1]. Our PPI++ uses the pooled prediction variance and an unconstrained weight. PSPA minimizes this empirical variance over [0, 1], including the classical and PPI endpoints; that is not a guarantee of improved finite-sample coverage or true variance for every dataset.
- **Scope**: `PSPA()` inherits `InferenceMethod`, consumes `PPIData`, and returns `InferenceResult` with the weight and sample counts. The same iid sampling and independent predictor assumptions as PPI apply; confidence intervals are asymptotic. Continuous outcomes, quantiles, OLS/GLM and predicted-covariate inference from the R package are not implemented here.
- **Validation**: `tests/test_pspa.py` checks the official R numerical fixture, the scalar sandwich formula, the shared interval callback, both weight caps, constant predictions, constrained variance minimization and repeated-sampling coverage with accurate, anticorrelated and constant predictions. `examples/simulation/pspa.ipynb` follows the PPI notebook's split -> target -> inference -> output format.

## Binary misclassification correction: RG and MLE

Implements the binary methods benchmarked in [Efficient Inference for Noisy LLM-as-a-Judge Evaluation](https://arxiv.org/abs/2601.05420), Chen, Lu, Li, Guo & Li (2026), following [debiasLLMReporting](https://github.com/yiqunchen/debias-llm-as-a-judge), commit `632a89565883be25d94d38b7cd3ca78d62a491f4`. RG originates with Rogan & Gladen (1978). No new sampling component or dependency is needed.

```python
from openppi import Mean, PPIData, get_method

# All predictions must be hard 0/1 judgments. Threshold probabilities explicitly
# before constructing this record if that is the judge definition you intend.
data = PPIData(labels=y_labeled, labeled_predictions=judge_labeled,
               unlabeled_predictions=judge_unlabeled)
rg = get_method('RG').inference(data, target=Mean(), alpha=.05)
mle = get_method('MLE').inference(data, target=Mean(), alpha=.05)
```

**Model and assumptions.** Let `theta=P(Y=1)`, `q1=P(judge=1|Y=1)` (sensitivity), `q0=P(judge=0|Y=0)` (specificity), and `J=q0+q1-1`. Then `p=P(judge=1)=theta*q1+(1-theta)*(1-q0)`. Both inference samples must be independent iid draws from the same population, with a fixed judge independent of the inference data. The error rates must transfer across samples; an outcome-stratified calibration sample with a different class balance is not supported by this interface. Soft probabilities are rejected rather than silently rounded.

**RG.** The raw estimator is `(mean(judge_U)+q0_hat-1)/J_hat`. The default follows the R function `rg_point_and_ci`, including its finite-sample adjustment: with `N` unlabeled rows, calibration class counts `m0,m1`, and `z=normal_quantile(1-alpha/2)`, set `p_tilde=(N*p_hat+z**2/2)/(N+z**2)`, `q0_tilde=(correct_negatives+1)/(m0+2)`, `q1_tilde=(correct_positives+1)/(m1+2)`. The adjusted estimator is clipped to [0,1]. Its variance is

```text
[p_tilde*(1-p_tilde)/(N+z²)
 + (1-theta)²*q0_tilde*(1-q0_tilde)/(m0+2)
 + theta²*q1_tilde*(1-q1_tilde)/(m1+2)] / J_tilde².
```

The point estimate therefore depends on `alpha`, as in R. Both calibration classes must be present. `abs(J_tilde)<eps_J` (default `1e-4`) raises `ValueError` instead of returning R's NA. Negative J is allowed by RG. Diagnostics report adjusted sensitivity/specificity, both raw and adjusted Youden indices, the unclipped estimate and whether clipping occurred. A small nonzero index can still produce a very large standard error.

**MLE.** Fits the joint likelihood of the four calibration cells `(Y,judge)` and the unlabeled judge counts. The binary saturated model has an exact interior solution: `p_hat` is the judge-positive proportion pooled across both samples, `s_hat=P_hat(Y=1|judge=1)` and `t_hat=P_hat(Y=1|judge=0)` use the calibration cells, and `theta_hat=p_hat*s_hat+(1-p_hat)*t_hat`. Recover `q1=p_hat*s_hat/theta_hat` and `q0=(1-p_hat)*(1-t_hat)/(1-theta_hat)`. This exactly maximizes the same likelihood as the R optimizer. No iterative optimizer is necessary for this binary interior case; tests also check agreement with SciPy optimization of the original three-parameter likelihood.

The observed information is diagonal in `(p,s,t)`, with inverse diagonal `p*(1-p)/(n+N)`, `s*(1-s)/n_judge_positive`, `t*(1-t)/n_judge_negative`. Applying the gradient `(s-t,p,1-p)` gives the prevalence variance, equivalent to the observed Fisher inverse in R's `mle_point_and_ci`. The official positive orientation `J>0` is enforced. All four calibration cells must be positive: boundary solutions have nonregular Fisher intervals and raise a clear error here, instead of inheriting the R optimizer's boundary or convergence artifacts. Diagnostics include sensitivity, specificity, J, log likelihood, calibration counts and the closed-form solver designation.

**Intervals and PPI++.** Both methods default to the official logit Wald interval: clamp theta to `[1e-6,1-1e-6]`, transform by logit, use `SE/(theta*(1-theta))`, and transform the endpoints back. The common `interval=` callback remains configurable. This differs from OpenPPI's default untransformed PPI++ Wald interval. Under regular binary models MLE and optimally calibrated PPI++ are asymptotically efficient, but they need not agree numerically in finite samples; our PPI++ also uses a different pooled prediction-variance tuning convention. RG can be much less efficient near a random judge. No finite-sample coverage guarantee is claimed.

**Validation.** `tests/test_misclassification.py` contains three fixtures from each official R function (point, variance and interval, plus MLE error rates), including RG estimates clipped to each boundary. `tests/fixtures/generate_misclassification.R` regenerates them from the pinned official sources. Tests also cover hard-label validation, random and reversed judges, missing classes/cells, alpha-dependent adjustment, interval callbacks, the joint likelihood optimum, and repeated-sampling coverage. The executed `examples/simulation/misclassification.ipynb` compares RG, MLE and PPI++ with a fixed hard binary judge.
