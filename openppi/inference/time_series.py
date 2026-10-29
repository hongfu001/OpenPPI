"""Mean PPI for ordered time series with Newey-West uncertainty (PPITSAS)."""
import numpy as np

from .base import InferenceMethod
from ..common.data import PPIData


def _series(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('HAC requires a finite time-ordered vector with at least two rows')
    return x


def _lag(max_lag, n):
    if isinstance(max_lag, (bool, np.bool_)) or not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError('max_lag must be None or a nonnegative integer')
    return min(int(max_lag), n - 1)


def default_newey_west_lag(series):
    """NW(1994) Bartlett plug-in lag, without prewhitening, as in official code.

    Autocovariances use denominator T. The reference constant is 1.1447
    (rounded from (3/2)^(1/3) in the paper). Constant/degenerate pilot sums
    use lag zero. The resulting lag is capped at T-1 when evaluating HAC.
    """
    x = _series(series)
    n = len(x)
    centered = x - x.mean()
    m = min(int(np.floor(4 * (n / 100) ** (2 / 9))), n - 1)
    if m == 0:
        return 0
    gamma = np.array([np.dot(centered[:n-j], centered[j:]) / n for j in range(m + 1)])
    s0 = float(gamma[0] + 2 * gamma[1:].sum())
    s1 = float(2 * np.dot(np.arange(1, m + 1), gamma[1:]))
    if not np.isfinite(s0) or not np.isfinite(s1) or abs(s0) <= 1e-14:
        return 0
    bandwidth = 1.1447 * ((s1 / s0) ** 2) ** (1 / 3) * n ** (1 / 3)
    return int(np.floor(bandwidth)) if np.isfinite(bandwidth) else 0


def newey_west_var_of_mean(series, max_lag=None):
    """Bartlett HAC variance of the mean; lag zero uses Var(ddof=0)/T.

    The sum estimates a long-run variance and is divided by T once more
    to obtain the variance of the sample mean. No iid ddof correction or
    prewhitening is applied. Lags larger than T-1 are capped.
    """
    x = _series(series)
    n = len(x)
    lag = _lag(default_newey_west_lag(x) if max_lag is None else max_lag, n)
    centered = x - x.mean()
    omega = float(np.dot(centered, centered) / n)
    for j in range(1, lag + 1):
        gamma = float(np.dot(centered[j:], centered[:-j]) / n)
        omega += 2 * (1 - j / (lag + 1)) * gamma
    return max(float(omega / n), 0.)


class PPITSAS(InferenceMethod):
    """Pointwise long-run mean inference at one location, not regression.

    PPIData arrays must retain chronological order within consecutive,
    regularly spaced blocks. Unlabeled history precedes the rectification
    block; train the fixed predictor on a later, separate fitting block
    (possibly pooling fitting data across locations). Do not randomly split
    or flatten locations into a single time series.

    Requires stationarity and suitable mixing/moment conditions. As in the
    paper, cross-block covariance and predictor-training dependence are
    neglected asymptotically; no finite-sample or simultaneous guarantee is
    claimed. max_lag=None selects a separate plug-in lag for each series.
    """
    name = 'PPITSAS'
    data_type = PPIData

    def __init__(self, max_lag=None, **kwargs):
        super().__init__(**kwargs)
        if max_lag is not None:
            _lag(max_lag, 2)
        self.max_lag = max_lag

    def _estimate(self, data, target):
        residuals = data.labels - data.labeled_predictions
        predictions = data.unlabeled_predictions
        lag_f = _lag(default_newey_west_lag(predictions) if self.max_lag is None else self.max_lag,
                     len(predictions))
        lag_r = _lag(default_newey_west_lag(residuals) if self.max_lag is None else self.max_lag,
                     len(residuals))
        var_f = newey_west_var_of_mean(predictions, lag_f)
        var_r = newey_west_var_of_mean(residuals, lag_r)
        return float(predictions.mean() + residuals.mean()), float(np.sqrt(var_f + var_r)), {
            'values': {'residuals': residuals, 'unlabeled_predictions': predictions},
            'labeled_count': len(residuals),
            'unlabeled_count': len(predictions),
            'prediction_weight': 1.,
            'weight_strategy': 'fixed',
            'variance_design': 'time_series_newey_west',
            'unlabeled_lag': lag_f,
            'residual_lag': lag_r,
            'unlabeled_mean_variance': var_f,
            'residual_mean_variance': var_r,
        }
