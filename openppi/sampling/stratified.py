"""StratPPI allocation: Proposition 3 and confidence-based Example 2."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StratifiedAllocation:
    """Integer sample counts and continuous rates, keyed by stratum.

    Rates solve the allocation with the requested per-stratum minimum; counts
    floor the continuous allocation and distribute remaining rows by marginal
    variance reduction. Plan from independent pilot data or label-free scores.
    """
    labeled_counts: dict
    unlabeled_counts: dict
    labeled_rates: dict
    unlabeled_rates: dict


def _allocate(weights, variances, budget, min_count):
    k = len(weights)
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or budget < k * min_count:
        raise ValueError('Each budget must be an integer >= number of strata * min_count')
    scale = weights * np.sqrt(variances)
    continuous = np.full(k, float(min_count))
    active = np.ones(k, dtype=bool)
    while active.any():
        remaining = budget - continuous[~active].sum()
        scores = scale[active]
        # When all remaining variances are zero, any allocation is optimal.
        if scores.sum() == 0:
            scores = weights[active]
        proposal = remaining * scores / scores.sum()
        below = proposal < min_count
        if not below.any():
            continuous[active] = proposal
            break
        active[np.flatnonzero(active)[below]] = False
    counts = np.floor(continuous).astype(int)
    # Round without losing budget; tie-break by fractional remainder.
    for _ in range(budget - int(counts.sum())):
        gain = scale**2 / (counts * (counts + 1))
        chosen = np.argmax(gain) if gain.max() > 0 else np.argmax(continuous - counts)
        counts[chosen] += 1
    return counts, continuous / budget


def stratified_allocate(weights, residual_variances, prediction_variances, *,
                        labeled_budget, unlabeled_budget, min_count=2):
    """Allocate both samples using eq. 18: rate_k proportional to w_k * SD_k.

    All three mappings must have the same keys. residual_variances estimates
    Var(Y-lambda_k*f | k), prediction_variances estimates Var(lambda_k*f | k).
    Weights are population probabilities, never observed sampling fractions.
    min_count enforces estimable sample variances; without active minima this
    is exactly Proposition 3. These are plug-in plans with lambda held fixed,
    not a joint optimum over lambda and counts. Collect fresh inference data
    after planning, using a fixed partition and iid sampling within each layer.
    """
    keys = list(weights)
    if not keys or set(keys) != set(residual_variances) or set(keys) != set(prediction_variances):
        raise ValueError('weights and variance mappings must have identical nonempty keys')
    w = np.array([weights[key] for key in keys], dtype=float)
    v = np.array([[values[key] for key in keys] for values in (residual_variances, prediction_variances)], dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any() or not np.isclose(w.sum(), 1., rtol=0, atol=1e-10):
        raise ValueError('Population weights must be positive, finite and sum to one')
    if not np.isfinite(v).all() or (v < 0).any():
        raise ValueError('Variances must be finite and nonnegative')
    if isinstance(min_count, (bool, np.bool_)) or not isinstance(min_count, (int, np.integer)) or min_count < 2:
        raise ValueError('min_count must be an integer >= 2')
    labeled, lr = _allocate(w, v[0], labeled_budget, min_count)
    unlabeled, ur = _allocate(w, v[1], unlabeled_budget, min_count)
    return StratifiedAllocation(dict(zip(keys, map(int, labeled))), dict(zip(keys, map(int, unlabeled))),
                               dict(zip(keys, map(float, lr))), dict(zip(keys, map(float, ur))))


def confidence_residual_variance(predictions, confidences, lam=1.):
    """Eq. 20--21 for binary Y, where confidences estimates P(Y=1 | X).

    Returns the label-free mixture variance E[c(1-c)] + Var(c-lam*f).
    This uses ddof=0 because the empirical confidence mixture is the modeled
    distribution. Confidence misspecification affects allocation efficiency,
    not the validity of inference on subsequently collected independent data.
    """
    f, c = np.asarray(predictions, dtype=float), np.asarray(confidences, dtype=float)
    if f.ndim != 1 or c.shape != f.shape or len(f) < 2 or not np.isfinite(f).all() or not np.isfinite(c).all():
        raise ValueError('Require aligned finite prediction and confidence vectors of length >= 2')
    if ((f < 0) | (f > 1) | (c < 0) | (c > 1)).any():
        raise ValueError('Binary predictions and confidences must be in [0, 1]')
    if not np.isscalar(lam) or not np.isfinite(lam):
        raise ValueError('lam must be finite and scalar')
    return float(np.mean(c * (1 - c)) + np.var(c - lam * f))
