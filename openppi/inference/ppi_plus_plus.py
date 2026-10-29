"""PPI++ (Angelopoulos, Duchi & Zrnic 2023): power-tuned two-sample PPI."""
import numpy as np
from .base import InferenceMethod
from ..common.data import PPIData


def tuned_lambda(labels, labeled_predictions, unlabeled_predictions):
    """Plug-in lambda = Cov_n(Y, f) / ((1 + n/N) Var_{N+n}(f)), Example 6.1.

    Var(f) pools both samples as in the paper; pppower uses labeled f only.
    Mean estimation stays convex for every real lambda, so it is not clipped.
    """
    n, big_n = len(labels), len(unlabeled_predictions)
    var_f = np.concatenate([labeled_predictions, unlabeled_predictions]).var(ddof=1)
    if var_f == 0:
        return 0.
    cov = np.cov(labels, labeled_predictions, ddof=1)[0, 1]
    return float(cov / ((1 + n / big_n) * var_f))


class PPIPlusPlus(InferenceMethod):
    """Binary mean PPI++: mean(Y) + lam * (mean(f_unlabeled) - mean(f_labeled)).

    lam=None uses the plug-in tuned_lambda; lam=0 is the classical mean of
    labels and lam=1 is PPI's point estimate. The SE follows Algorithm 2:
    sqrt(lam^2 Var_{N+n}(f) / N + Var_n(Y - lam f) / n).
    """
    name = 'PPI++'
    data_type = PPIData

    def __init__(self, lam=None, **kwargs):
        super().__init__(**kwargs)
        if lam is not None and not np.isfinite(lam):
            raise ValueError('lam must be finite or None')
        self.lam = lam

    def _estimate(self, data, target):
        labels, f_labeled, f_unlabeled = data.labels, data.labeled_predictions, data.unlabeled_predictions
        lam = tuned_lambda(labels, f_labeled, f_unlabeled) if self.lam is None else float(self.lam)
        residuals = labels - lam * f_labeled
        predictions = lam * f_unlabeled
        var_f = np.concatenate([f_labeled, f_unlabeled]).var(ddof=1)
        point = float(predictions.mean() + residuals.mean())
        se = float(np.sqrt(lam ** 2 * var_f / len(f_unlabeled) + residuals.var(ddof=1) / len(residuals)))
        return point, se, {
            'values': {'residuals': residuals, 'unlabeled_predictions': predictions},
            'labeled_count': len(residuals),
            'unlabeled_count': len(predictions),
            'prediction_weight': lam,
            'weight_strategy': 'tuned' if self.lam is None else 'fixed',
            'variance_design': 'independent_two_sample',
        }
