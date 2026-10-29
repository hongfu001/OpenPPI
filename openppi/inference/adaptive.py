"""PSPA mean inference (Miao et al., arXiv:2311.14220, Algorithm 1)."""
import numpy as np

from .base import InferenceMethod
from ..common.data import PPIData


class PSPA(InferenceMethod):
    """Binary mean specialization of the official pspa_y(method='mean').

    Weight = clip(Cov(Y, f_L) / (Var(f_L) + n/N Var(f_U)), 0, 1),
    using sample covariances (ddof=1). The [0, 1] constraint follows the
    official optim_weights; the paper's Algorithm 1 only prints the upper cap.
    Constant predictions in both samples use weight zero instead of 0/0.
    The predictor must be trained independently of both iid inference samples.
    """
    name = 'PSPA'
    data_type = PPIData

    def _estimate(self, data, target):
        y, f, g = data.labels, data.labeled_predictions, data.unlabeled_predictions
        n, big_n = len(y), len(g)
        denominator = f.var(ddof=1) + n / big_n * g.var(ddof=1)
        weight = (float(np.clip(np.cov(y, f, ddof=1)[0, 1] / denominator, 0., 1.))
                  if denominator > 0 else 0.)
        residuals, predictions = y - weight * f, weight * g
        point = float(y.mean() + weight * (g.mean() - f.mean()))
        # Algebraically Sigma_cal / n; this form avoids cancellation.
        se = float(np.sqrt(residuals.var(ddof=1) / n + predictions.var(ddof=1) / big_n))
        return point, se, {
            'values': {'residuals': residuals, 'unlabeled_predictions': predictions},
            'labeled_count': n,
            'unlabeled_count': big_n,
            'prediction_weight': weight,
            'weight_strategy': 'tuned_clipped',
            'variance_design': 'independent_two_sample',
        }
