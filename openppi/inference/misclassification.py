"""Binary judge correction: Rogan--Gladen and joint MLE (Chen et al., ICML 2026)."""
import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

from .base import InferenceMethod
from ..common.data import PPIData


def logit_interval(point, se, alpha, values=None):
    """Official package's logit Wald interval, with its 1e-6 endpoint clamp."""
    th = np.clip(point, 1e-6, 1 - 1e-6)
    radius = norm.ppf(1 - alpha / 2) * se / (th * (1 - th))
    return tuple(float(x) for x in expit([logit(th) - radius, logit(th) + radius]))


def _counts(data):
    y, f, g = data.labels, data.labeled_predictions, data.unlabeled_predictions
    if not np.isin(y, [0, 1]).all():
        raise ValueError('RG and MLE require binary labels in {0, 1}')
    if not (np.isin(f, [0, 1]).all() and np.isin(g, [0, 1]).all()):
        raise ValueError('RG and MLE require binary predictions in {0, 1}; threshold probabilities explicitly')
    return np.array([np.sum((y == a) & (f == b))
                     for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))], dtype=float)


class RoganGladen(InferenceMethod):
    """Adjusted RG as in debiasLLMReporting::rg_point_and_ci.

    Uses alpha-dependent Agresti--Coull smoothing for test prevalence and
    add-one smoothing for sensitivity/specificity, then clips theta to [0,1].
    A near-zero adjusted Youden index raises instead of returning R's NA.
    """
    name = 'RG'
    data_type = PPIData

    def __init__(self, eps_J=1e-4, interval=logit_interval):
        super().__init__(interval=interval)
        if not np.isfinite(eps_J) or not 0 < eps_J < 1:
            raise ValueError('eps_J must be in (0, 1)')
        self.eps_J = eps_J

    def _estimate(self, data, target):
        return self._estimate_with_alpha(data, target, .05)

    def _estimate_with_alpha(self, data, target, alpha):
        n00, n01, n10, n11 = _counts(data)
        m0, m1, n = n00 + n01, n10 + n11, len(data.unlabeled_predictions)
        if min(m0, m1) == 0:
            raise ValueError('RG needs both label classes in the calibration sample')
        z2 = norm.ppf(1 - alpha / 2)**2
        p = (data.unlabeled_predictions.sum() + z2 / 2) / (n + z2)
        q0, q1 = (n00 + 1) / (m0 + 2), (n11 + 1) / (m1 + 2)
        j = q0 + q1 - 1
        if abs(j) < self.eps_J:
            raise ValueError(f'RG adjusted Youden index q0 + q1 - 1 = {j:g} is near zero')
        raw = (p + q0 - 1) / j
        theta = float(np.clip(raw, 0, 1))
        variance = (p * (1 - p) / (n + z2)
                    + (1 - theta)**2 * q0 * (1 - q0) / (m0 + 2)
                    + theta**2 * q1 * (1 - q1) / (m1 + 2)) / j**2
        return theta, float(np.sqrt(variance)), {
            'specificity': float(q0), 'sensitivity': float(q1),
            'youden_index': float(j), 'raw_youden_index': float(n00 / m0 + n11 / m1 - 1),
            'unclipped_estimate': float(raw), 'estimate_clipped': bool(raw != theta),
            'labeled_count': len(data.labels), 'unlabeled_count': n,
            'variance_design': 'adjusted_delta', 'finite_sample_adjustment': True,
        }


class MisclassificationMLE(InferenceMethod):
    """Joint binary MLE with observed Fisher SE and logit CI.

    The saturated binary model has a closed-form interior maximizer in the
    coordinates p=P(f=1), s=P(Y=1|f=1), t=P(Y=1|f=0). These coordinates
    factor the likelihood into three binomials. This gives the same optimum
    as numerical optimization, without convergence tolerance artifacts.
    Matches the official positive-Youden orientation; boundary fits and
    nonpositive orientation raise rather than report unreliable Wald results.
    """
    name = 'MLE'
    data_type = PPIData

    def __init__(self, interval=logit_interval):
        super().__init__(interval=interval)

    def _estimate(self, data, target):
        n00, n01, n10, n11 = counts = _counts(data)
        if np.any(counts == 0):
            raise ValueError('MLE requires all four calibration cells positive for an interior Fisher interval')
        m, n = len(data.labels), len(data.unlabeled_predictions)
        p = (n01 + n11 + data.unlabeled_predictions.sum()) / (m + n)
        s, t = n11 / (n01 + n11), n10 / (n00 + n10)
        theta = p * s + (1 - p) * t
        q1, q0 = p * s / theta, (1 - p) * (1 - t) / (1 - theta)
        j = q0 + q1 - 1
        if j <= 0:
            raise ValueError(f'MLE requires positive judge orientation q0 + q1 - 1 > 0; fitted index = {j:g}')
        # Inverse observed information in (p,s,t) at the exact stationary point.
        inv_info = np.diag([p * (1 - p) / (m + n),
                            s * (1 - s) / (n01 + n11),
                            t * (1 - t) / (n00 + n10)])
        gradient = np.array([s - t, p, 1 - p])
        variance = gradient @ inv_info @ gradient
        cell_probs = np.array([(1 - p) * (1 - t), p * (1 - s), (1 - p) * t, p * s])
        u1 = data.unlabeled_predictions.sum()
        ll = counts @ np.log(cell_probs) + u1 * np.log(p) + (n - u1) * np.log1p(-p)
        return float(theta), float(np.sqrt(variance)), {
            'specificity': float(q0), 'sensitivity': float(q1), 'youden_index': float(j),
            'labeled_count': m, 'unlabeled_count': n, 'log_likelihood': float(ll),
            'variance_design': 'observed_fisher', 'solver': 'closed_form_interior',
            'calibration_counts': counts.astype(int).tolist(),
        }
