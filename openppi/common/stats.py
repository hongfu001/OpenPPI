"""Private formulas used by methods and sampling designs."""
import numpy as np
from scipy.stats import norm


def ipw_correct(predictions, labels, queried, propensities):
    """Return f + R/pi*(Y-f) from already validated query-time arrays.

    Read labels only at queried rows. Data validation belongs to ActiveData.
    """
    values = np.array(predictions, dtype=float, copy=True)
    q = np.asarray(queried, dtype=bool)
    values[q] += (np.asarray(labels)[q] - values[q]) / np.asarray(propensities)[q]
    return values


def wald_interval(estimate, standard_error, alpha=0.05, values=None):
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0, 1)')
    point, se = np.asarray(estimate), np.asarray(standard_error)
    if point.shape != se.shape or not np.isfinite(point).all() or not np.isfinite(se).all() or (se < 0).any():
        raise ValueError('Require aligned finite estimates and nonnegative finite SEs')
    half = norm.ppf(1 - alpha / 2) * se
    if point.ndim == 0:
        return float(point - half), float(point + half)
    return point - half, point + half



def ledoit_wolf(rows):
    """Ledoit-Wolf shrinkage covariance of the rows (same formula as sklearn's ledoit_wolf)."""
    x = np.asarray(rows, dtype=float)
    x = x - x.mean(axis=0)
    n, k = x.shape
    sample = x.T @ x / n
    mu = np.trace(sample) / k
    delta = ((sample - mu * np.eye(k)) ** 2).sum() / k
    beta = (((x ** 2).T @ x ** 2) / n - sample ** 2).sum() / (k * n)
    shrinkage = 0. if delta == 0 else min(beta, delta) / delta
    return (1 - shrinkage) * sample + shrinkage * mu * np.eye(k)


def multi_weights(covariance, counts, a):
    """Optimal unbiased weights lambda_I = n_I Sigma_I^-1 P_I S a and their variance a^T S a.

    counts maps index tuples to positive sample sizes; S = (sum_I n_I P_I^T Sigma_I^-1 P_I)^+
    (MultiPPI, eq. 7 and Appendix F.1). Raises if a is not covered by the subsets.
    """
    k = len(a)
    precision = np.zeros((k, k))
    blocks = {}
    for subset, n in counts.items():
        idx = list(subset)
        blocks[subset] = np.linalg.inv(covariance[np.ix_(idx, idx)])
        precision[np.ix_(idx, idx)] += n * blocks[subset]
    covered = sorted({i for subset in counts for i in subset})
    if np.any(np.delete(a, covered)):
        raise ValueError('The target a must be covered by the sampled subsets')
    s_a = np.zeros(k)
    s_a[covered] = np.linalg.solve(precision[np.ix_(covered, covered)], a[covered])
    weights = {subset: n * blocks[subset] @ s_a[list(subset)] for subset, n in counts.items()}
    return weights, float(a @ s_a)
