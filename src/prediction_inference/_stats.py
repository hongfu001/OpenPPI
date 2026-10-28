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
    if not np.isfinite(estimate) or not np.isfinite(standard_error) or standard_error < 0:
        raise ValueError('Require a finite estimate and nonnegative finite SE')
    half = float(norm.ppf(1 - alpha / 2) * standard_error)
    return float(estimate - half), float(estimate + half)

