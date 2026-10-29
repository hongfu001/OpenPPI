"""BAI sampling (Chen, Zhou, Peng and Wang, NeurIPS 2025): ASI probabilities, cube selection.

The probabilities are the tau-mixed ASI rule. Instead of independent
Bernoulli draws, the cube method (Deville and Tille, 2004) selects the
batch so that the Horvitz-Thompson total of the uncertainty matches the
pool total, sum(u * xi / pi) = sum(u), while every row keeps inclusion
probability pi. The reference code calls the R package BalancedSampling;
this module is a pure NumPy cube method with flight and landing phases.
"""
import numpy as np

from ..common.data import BalancedActiveData
from .uncertainty import Uncertainty

_EPS = 1e-10


def _flight(pi, a, order, rng):
    """Move pi on the hyperplane sum(a * pi) = const until <= q rows stay fractional.

    a holds the q balancing columns divided by the original probabilities.
    Each step perturbs q + 1 fractional rows along a null-space direction of
    their a-rows, choosing the step sign so that E[pi_new] = pi_old; at
    least one row reaches 0 or 1 per step.
    """
    q = a.shape[1]
    pending = iter(k for k in order if _EPS < pi[k] < 1 - _EPS)
    group = []
    while True:
        while len(group) < q + 1:
            k = next(pending, None)
            if k is None:
                return
            if _EPS < pi[k] < 1 - _EPS:
                group.append(k)
        rows = np.array(group)
        u = np.ones(1) if q == 0 else np.linalg.svd(a[rows].T)[2][-1]
        up, down = u > _EPS, u < -_EPS
        current = pi[rows]
        step_up = min(np.min((1 - current[up]) / u[up], initial=np.inf),
                      np.min(current[down] / -u[down], initial=np.inf))
        step_down = min(np.min(current[up] / u[up], initial=np.inf),
                        np.min((1 - current[down]) / -u[down], initial=np.inf))
        if rng.random() < step_down / (step_up + step_down):
            current = current + step_up * u
        else:
            current = current - step_down * u
        current[current < _EPS] = 0.
        current[current > 1 - _EPS] = 1.
        pi[rows] = np.clip(current, 0, 1)
        group = [k for k in group if _EPS < pi[k] < 1 - _EPS]


def cube_sample(propensities, balancing, rng):
    """Cube method: inclusion probabilities propensities, balanced on balancing.

    Rows are processed in random order. The flight phase uses all p
    balancing columns; the landing phase drops the last column and
    repeats until the remaining rows are settled by independent draws.
    Returns boolean selection indicators.
    """
    pi = np.array(propensities, dtype=float, copy=True)
    x = np.asarray(balancing, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    if pi.ndim != 1 or x.shape[0] != len(pi) or not ((pi > 0) & (pi <= 1)).all():
        raise ValueError('Require probabilities in (0, 1] aligned with the balancing matrix')
    a = x / pi[:, None]
    order = rng.permutation(len(pi))
    for q in range(x.shape[1], -1, -1):
        _flight(pi, a[:, :q], order, rng)
    return pi > .5


class BalancedDesign(Uncertainty):
    """ASI probabilities with cube selection balanced on outcome-free variables.

    tau is the uniform share, as in Uncertainty; the paper's tau is the
    active share, so tau=.5 coincides. By default the cube method balances
    on the scores (the paper's text). The reference code instead fits the
    error model to signed errors f(x) - y, samples with pi ~ |e(x)| and
    balances on the signed e(x); pass those as balancing (one value or row
    per batch row) and |e(x)| as select_for_labeling scores. The gain over
    ASI comes from balancing a proxy of the signed error (Assumption 5).
    fixed_size=True also balances on pi, fixing the selected count at
    about sum(pi).
    """
    def __init__(self, tau=.5, balancing=None, fixed_size=False):
        super().__init__(tau=tau)
        if balancing is not None:
            balancing = np.array(balancing, dtype=float, copy=True)
            balancing = balancing[:, None] if balancing.ndim == 1 else balancing
            if balancing.ndim != 2 or not np.isfinite(balancing).all():
                raise ValueError('balancing must be a finite vector or n x p matrix')
        self.extra = balancing
        self.fixed_size = bool(fixed_size)

    def sample(self, scores, propensities, rng):
        x = np.asarray(scores, dtype=float)[:, None] if self.extra is None else self.extra
        if len(x) != len(propensities):
            raise ValueError('balancing must have one row per batch row')
        self.balancing = np.column_stack([propensities, x]) if self.fixed_size else x
        return cube_sample(propensities, self.balancing, rng)

    def record(self, predictions, observed_labels, queried, propensities):
        return BalancedActiveData(predictions, observed_labels, queried, propensities,
                                  balancing=self.balancing)
