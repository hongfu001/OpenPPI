"""Calibrated PPI (van der Laan & van der Laan 2026, Algorithms 1--2)."""
import numpy as np

from .base import InferenceMethod
from ..common.data import PPIData


def _isotonic_fit(scores, labels):
    """Weighted pool-adjacent-violators on distinct scores; ties share a fit.

    Linear interpolation between fitted scores, with endpoint clipping, matches
    the official package's sklearn isotonic backend. Uses NumPy only.
    """
    knots, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=labels)
    blocks = []
    for index, (total, count) in enumerate(zip(sums, counts)):
        blocks.append([index, index + 1, float(total), int(count)])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    fitted = np.empty(len(knots))
    for start, stop, total, count in blocks:
        fitted[start:stop] = total / count
    return knots, fitted


class CalibratedPPI(InferenceMethod):
    """Post-hoc isotonic or affine calibration followed by pooled-mean inference.

    Calibration uses the labeled inference sample, as in Algorithms 1--2; the
    original predictor must be independently trained. 'paper' variance uses
    the sum of squared influence values divided by (n+N)^2. 'two_sample'
    uses within-sample centered population variances (ddof=0), matching the
    current official package's Wald convention. Linear predictions are not
    clipped, preserving the paper's affine normal equations.
    """
    data_type = PPIData

    def __init__(self, calibration='isotonic', variance='paper', **kwargs):
        super().__init__(**kwargs)
        if calibration not in ('isotonic', 'linear'):
            raise ValueError("calibration must be 'isotonic' or 'linear'")
        if variance not in ('paper', 'two_sample'):
            raise ValueError("variance must be 'paper' or 'two_sample'")
        self.calibration, self.variance = calibration, variance
        self.name = 'CalPPI' if calibration == 'isotonic' else 'LinearCalPPI'

    def _estimate(self, data, target):
        y, f, g = data.labels, data.labeled_predictions, data.unlabeled_predictions
        n, big_n = len(y), len(g)
        total = n + big_n
        rho = n / total
        # Fit and evaluate calibration using only the labeled outcomes.
        if self.calibration == 'isotonic':
            knots, fitted = _isotonic_fit(f, y)
            m_l, m_u = np.interp(f, knots, fitted), np.interp(g, knots, fitted)
            calibration_info = {'knots': knots.tolist(), 'fitted_values': fitted.tolist(),
                                'block_count': int(len(np.unique(fitted)))}
        else:
            centered = f - f.mean()
            spread = centered @ centered
            slope = float(centered @ (y - y.mean()) / spread) if spread > 0 else 0.
            intercept = float(y.mean() - slope * f.mean())
            m_l, m_u = intercept + slope * f, intercept + slope * g
            calibration_info = {'intercept': intercept, 'slope': slope}
        point = float((m_l.sum() + m_u.sum()) / total)
        residuals = y - m_l
        influence_l = m_l - point + residuals / rho
        influence_u = m_u - point
        if self.variance == 'paper':
            variance = (influence_l @ influence_l + influence_u @ influence_u) / total**2
        else:
            variance = (n * influence_l.var() + big_n * influence_u.var()) / total**2
        return point, float(np.sqrt(variance)), {
            'values': {'residuals': residuals, 'labeled_predictions': m_l,
                       'unlabeled_predictions': m_u, 'labeled_influence': influence_l,
                       'unlabeled_influence': influence_u},
            'labeled_count': n, 'unlabeled_count': big_n,
            'labeled_fraction': rho, 'calibration': self.calibration,
            'calibration_parameters': calibration_info,
            'calibration_residual_mean': float(residuals.mean()),
            'variance_design': 'pooled_calibrated_influence',
            'variance_convention': self.variance,
        }
