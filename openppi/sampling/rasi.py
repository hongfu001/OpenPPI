"""RASI sampling (Li, Zrnic and Candes, NeurIPS 2025): burn-in batch, then a robust batch.

Batch 1 labels a uniform burn-in subset with probability 1. Between the
batches, RobustDesign.fit learns an error model e(x) ~ |Y - f(x)| and the
l2 radius c from those labels only. Batch 2 is an ordinary
select_for_labeling call on the remaining rows with scores e(x): the
design moves along a budget-preserving path between the active rule and
uniform sampling to the point minimizing

    max_{||eps||_2 <= c sqrt(n)} mean((e^2 + eps) / pi)
        = mean(e^2 / pi) + c * sqrt(mean(1 / pi^2)),

the objective of the reference MinMaxOptimizer_l2. BurnIn.combine merges
both batches into RobustActiveData; burn-in rows carry propensity 1, so
the ASI correction reproduces the reference concatenated estimator.
"""
from dataclasses import dataclass
import numpy as np

from ..common.data import ROBUST_PATHS, RobustActiveData
from ..common.targets import Mean
from .base import LabelRequest, SamplingDesign, select_for_labeling, validate_calibration

PATHS = ROBUST_PATHS
DEFAULT_CONSTRAINTS = (0., .01, .02, .03, .04, .05, .06, .07, .1)


def _validate_path(path):
    if path not in PATHS:
        raise ValueError(f'path must be one of {PATHS}')


def active_rule(scores, rho, floor=1e-3):
    """Budget-scaled active rule clip(eta * u, 0, 1), eta = rho / mean(u).

    Scores are floored at floor * mean(u) so every row keeps a positive
    probability, which unbiased inverse weighting requires.
    """
    u = validate_calibration(scores, rho)
    if u.mean() == 0:
        return np.full(len(u), rho)
    u = np.maximum(u, floor * u.mean())
    return np.clip(rho / u.mean() * u, 0, 1)


def _path_raw(base, rho, t, path):
    if path == 'linear':
        return (1 - t) * base + t * rho
    if path == 'geometric':
        return base ** (1 - t) * rho ** t
    return ((1 - t) * np.sqrt(base) + t * np.sqrt(rho)) ** 2


def budget_path(base, rho, t, path='geometric'):
    """Point t in [0, 1] on a path from base (t=0) to uniform rho (t=1).

    The unnormalized path is renormalized to mean rho and clipped to
    [0, 1]; clipping can only lower the expected label count.
    """
    _validate_path(path)
    raw = _path_raw(np.asarray(base, dtype=float), rho, t, path)
    return np.clip(raw / raw.mean() * rho, 0, 1)


def robust_objective(errors, pi, constraint):
    """l2-robust variance proxy mean(e^2/pi) + c * sqrt(mean(1/pi^2))."""
    return float(np.mean(errors ** 2 / pi) + constraint * np.sqrt(np.mean(1 / pi ** 2)))


def robust_rho(errors, base, rho, *, constraint=0., path='geometric', grid=101):
    """Grid-search the path point minimizing the robust objective.

    errors estimate |Y - f(X)| per row; base is the active rule. With
    constraint=0 and base proportional to errors the minimizer is t=0
    (paper, Appendix A.4); larger constraints push t toward uniform.
    """
    _validate_path(path)
    e = np.asarray(errors, dtype=float)
    if e.shape != np.shape(base) or not np.isfinite(e).all() or (e < 0).any():
        raise ValueError('errors must be finite, nonnegative and aligned with base')
    if constraint < 0 or not isinstance(grid, int) or grid < 2:
        raise ValueError('Require constraint >= 0 and an integer grid >= 2')
    ts = np.linspace(0, 1, grid)
    risks = [robust_objective(e, budget_path(base, rho, t, path), constraint) for t in ts]
    return float(ts[int(np.argmin(risks))])


def _fit_error_model(error_model_factory, x, y, p):
    error_model = error_model_factory()
    error_model.fit(x, np.abs(y - p))
    return error_model


def _predict_errors(error_model, x):
    errors = np.asarray(error_model.predict(x), dtype=float)
    if errors.shape != (len(x),) or not np.isfinite(errors).all():
        raise ValueError('Error model must predict one finite value per row')
    return np.maximum(errors, 0)


def select_constraint(features, labels, predictions, *, error_model_factory, rho,
                      candidates=DEFAULT_CONSTRAINTS, folds=5, path='geometric',
                      grid=101, random_state=None):
    """Choose the l2 radius c by K-fold cross-validation on labeled burn-in rows.

    Each fold fits the error model on the other folds, solves for the path
    point on the held-out fold, and scores the resulting design by the
    expected standard deviation of the AIPW values,
    sqrt(mean(Y^2 + (Y - f)^2 (1/pi - 1)) - mean(Y)^2). The reference code
    estimates the same quantity with 200 Monte Carlo draws.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    p = np.asarray(predictions, dtype=float)
    if not isinstance(folds, int) or folds < 2 or len(y) < 2 * folds:
        raise ValueError('Require folds >= 2 and at least two rows per fold')
    if x.ndim != 2 or len(x) != len(y) or p.shape != y.shape:
        raise ValueError('Require aligned features, labels and predictions')
    candidates = np.asarray(candidates, dtype=float)
    if candidates.ndim != 1 or not len(candidates) or (candidates < 0).any():
        raise ValueError('candidates must be a nonempty vector of nonnegative radii')
    parts = np.array_split(np.random.default_rng(random_state).permutation(len(y)), folds)
    designs = []
    for k, test in enumerate(parts):
        train = np.concatenate([part for j, part in enumerate(parts) if j != k])
        model = _fit_error_model(error_model_factory, x[train], y[train], p[train])
        errors = _predict_errors(model, x[test])
        designs.append((test, errors, active_rule(errors, rho)))
    risks = []
    for c in candidates:
        risk = 0.
        for test, errors, rule in designs:
            pi = budget_path(rule, rho, robust_rho(errors, rule, rho, constraint=c,
                                                    path=path, grid=grid), path)
            second = np.mean(y[test] ** 2 + (y[test] - p[test]) ** 2 * (1 / pi - 1))
            risk += np.sqrt(max(second - y[test].mean() ** 2, 0.))
        risks.append(risk)
    return float(candidates[int(np.argmin(risks))])


class RobustDesign(SamplingDesign):
    """Second-batch design: active rule pi ~ e(x) moved along a budget path.

    Build it with RobustDesign.fit on burn-in labels, then pass
    scores=design.errors(remaining_features) to select_for_labeling.
    calibrate reads only these outcome-free scores: it sets the active
    rule's scale, solves the robust path point rho_robust on the
    reference scores, and fixes the path normalizer, so probability is a
    per-row map that reproduces budget_path on the reference pool.
    """
    def __init__(self, error_model, *, constraint, path='geometric', grid=101, floor=1e-3):
        _validate_path(path)
        if not callable(getattr(error_model, 'predict', None)):
            raise TypeError('error_model must expose predict(X)')
        if constraint < 0 or not isinstance(grid, int) or grid < 2 or not 0 < floor < 1:
            raise ValueError('Require constraint >= 0, integer grid >= 2 and floor in (0, 1)')
        self.error_model = error_model
        self.constraint = float(constraint)
        self.path = path
        self.grid = grid
        self.floor = floor

    @classmethod
    def fit(cls, features, labels, predictions, *, error_model_factory, rho, constraint=None,
            candidates=DEFAULT_CONSTRAINTS, folds=5, path='geometric', grid=101, random_state=None):
        """Fit e(x) to |Y - f(x)| on burn-in rows; choose c by CV when constraint is None.

        rho is the second batch's sampling rate (its budget over the number
        of remaining rows), used only by cross-validation.
        """
        if not callable(error_model_factory):
            raise TypeError('error_model_factory must be callable')
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float)
        p = np.asarray(predictions, dtype=float)
        if x.ndim != 2 or len(x) < 2 or len(x) != len(y) or p.shape != y.shape:
            raise ValueError('Require aligned burn-in features, labels and predictions')
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError('Burn-in data must be finite')
        if constraint is None:
            constraint = select_constraint(x, y, p, error_model_factory=error_model_factory, rho=rho,
                                           candidates=candidates, folds=folds, path=path,
                                           grid=grid, random_state=random_state)
        return cls(_fit_error_model(error_model_factory, x, y, p),
                   constraint=constraint, path=path, grid=grid)

    def errors(self, features):
        """Predicted |Y - f(x)|, clipped at zero; use as select_for_labeling scores."""
        return _predict_errors(self.error_model, np.asarray(features, dtype=float))

    def _base(self, scores):
        u = np.asarray(scores, dtype=float)
        if self.eta is None:
            return np.full(u.shape, self.rho)
        return np.clip(self.eta * np.maximum(u, self.score_floor), 0, 1)

    def calibrate(self, reference_scores, rho):
        scores = validate_calibration(reference_scores, rho)
        self.rho = rho
        if scores.mean() == 0:
            self.eta = self.score_floor = None
        else:
            self.score_floor = self.floor * scores.mean()
            self.eta = rho / np.maximum(scores, self.score_floor).mean()
        base = self._base(scores)
        self.rho_robust = robust_rho(scores, base, rho, constraint=self.constraint,
                                     path=self.path, grid=self.grid)
        self.scale = rho / _path_raw(base, rho, self.rho_robust, self.path).mean()
        return self

    def probability(self, scores):
        raw = _path_raw(self._base(scores), self.rho, self.rho_robust, self.path)
        return np.clip(raw * self.scale, 0, 1)


@dataclass(frozen=True)
class BurnIn:
    """First batch: a uniform random subset labeled with probability 1.

    indices go to the annotators now; remaining rows enter the second
    batch. Selection reads no labels. Arrays are copied and read-only.
    """
    n: int
    indices: np.ndarray
    remaining: np.ndarray

    def __post_init__(self):
        for name in ('indices', 'remaining'):
            array = np.array(getattr(self, name), dtype=int, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        rows = np.concatenate([self.indices, self.remaining])
        if len(rows) != self.n or not np.array_equal(np.sort(rows), np.arange(self.n)):
            raise ValueError('indices and remaining must partition range(n)')

    def combine(self, predictions, labels, request, request_labels):
        """Merge both batches into RobustActiveData over all n rows.

        labels align with indices; request is the second batch's
        LabelRequest over remaining rows and request_labels align with
        request.indices. The request must come from a RobustDesign.
        """
        p = np.asarray(predictions, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if p.shape != (self.n,) or labels.shape != (len(self.indices),):
            raise ValueError('Require n predictions and one label per burn-in index')
        if not isinstance(request, LabelRequest) or not isinstance(request.design, RobustDesign):
            raise TypeError('request must be a LabelRequest selected with a RobustDesign')
        if not np.array_equal(request.predictions, p[self.remaining]):
            raise ValueError('request must cover the remaining rows in order')
        second = request.with_labels(request_labels)
        observed = np.full(self.n, np.nan)
        queried = np.zeros(self.n, dtype=bool)
        propensities = np.ones(self.n)
        observed[self.indices] = labels
        queried[self.indices] = True
        observed[self.remaining] = second.observed_labels
        queried[self.remaining] = second.queried
        propensities[self.remaining] = second.propensities
        burn_in = np.zeros(self.n, dtype=bool)
        burn_in[self.indices] = True
        design = request.design
        return RobustActiveData(p, observed, queried, propensities, burn_in=burn_in,
                                rho_robust=design.rho_robust, constraint=design.constraint,
                                path=design.path)


def select_burn_in(n, *, size, random_state=None):
    """Choose a uniform random burn-in subset of size rows out of n."""
    if not isinstance(n, (int, np.integer)) or not isinstance(size, (int, np.integer)) or not 2 <= size < n:
        raise ValueError('Require integers with 2 <= size < n')
    order = np.random.default_rng(random_state).permutation(n)
    return BurnIn(int(n), np.sort(order[:size]), np.sort(order[size:]))


def robust_active_label(features, *, oracle, target, budget, model, error_model_factory,
                        burn_in, constraint=None, candidates=DEFAULT_CONSTRAINTS, folds=5,
                        path='geometric', grid=101, random_state=None):
    """Run both batches with an oracle; returns RobustActiveData.

    Convenience wrapper for simulations: select_burn_in -> label ->
    RobustDesign.fit -> select_for_labeling on remaining rows -> label ->
    BurnIn.combine. model is a fixed predictor exposing predict(X) ->
    probabilities; burn_in labels count toward budget, an expected total.
    """
    if not isinstance(target, Mean):
        raise TypeError('Only the binary Mean target is currently supported')
    x = np.array(features, dtype=float, copy=True)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Require a finite feature matrix with n >= 2')
    if not callable(oracle):
        raise TypeError('oracle must be callable')
    if not callable(getattr(model, 'predict', None)):
        raise TypeError('model must expose predict(X)')
    if not isinstance(burn_in, int) or not 2 <= burn_in < budget < len(x):
        raise ValueError('Require integers with 2 <= burn_in < budget < n')
    predictions = np.asarray(model.predict(x), dtype=float)
    if predictions.shape != (len(x),) or not ((predictions >= 0) & (predictions <= 1)).all():
        raise ValueError('Model must predict one probability in [0, 1] per row')

    def ask(rows):
        labels = np.array([float(oracle(int(i))) for i in rows])
        if not np.isfinite(labels).all() or not np.isin(labels, [0., 1.]).all():
            raise ValueError('Mean active labeling currently requires binary oracle labels')
        return labels

    rng = np.random.default_rng(random_state)
    burn = select_burn_in(len(x), size=burn_in, random_state=rng)
    labels = ask(burn.indices)
    rest = burn.remaining
    second_budget = budget - burn_in
    if not 0 < second_budget < len(rest):
        raise ValueError('Budget after burn-in must give a sampling rate in (0, 1)')
    design = RobustDesign.fit(
        x[burn.indices], labels, predictions[burn.indices],
        error_model_factory=error_model_factory, rho=second_budget / len(rest),
        constraint=constraint, candidates=candidates, folds=folds, path=path, grid=grid,
        random_state=None if constraint is not None else rng.integers(2**32))
    request = select_for_labeling(predictions[rest], target=target, design=design,
                                  budget=second_budget, scores=design.errors(x[rest]),
                                  random_state=rng)
    return burn.combine(predictions, labels, request, ask(rest[request.indices]))
