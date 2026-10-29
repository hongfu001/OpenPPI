"""Two-sample prediction-powered estimators: PPI (lam = 1) and PPI++ (tuned lam)."""
import numpy as np
from .base import InferenceMethod
from ..common.data import PPIData


def tuned_lambda(labels, labeled_predictions, unlabeled_predictions):
    """Plug-in lambda = Cov_n(Y, f) / ((1 + n/N) Var_{N+n}(f)), PPI++ Example 6.1.

    Var(f) pools both samples as in the paper; pppower uses labeled f only.
    Mean estimation stays convex for every real lambda, so it is not clipped.
    """
    n, big_n = len(labels), len(unlabeled_predictions)
    var_f = np.concatenate([labeled_predictions, unlabeled_predictions]).var(ddof=1)
    if var_f == 0:
        return 0.
    cov = np.cov(labels, labeled_predictions, ddof=1)[0, 1]
    return float(cov / ((1 + n / big_n) * var_f))


class TwoSample(InferenceMethod):
    """Real-valued mean mean(lam f_unlabeled) + mean(Y - lam f)_labeled from independent samples.

    lam=1 is PPI (Angelopoulos et al. 2023), lam='tuned' is PPI++ with the
    plug-in tuned_lambda (Angelopoulos, Duchi & Zrnic 2023), lam=0 is the
    classical mean of labels; any finite number is allowed. The SE is
    sqrt(lam^2 Var(f) / N + Var_n(Y - lam f) / n), with Var(f) taken on the
    unlabeled sample (variance='unlabeled', as in PPI) or on both samples
    (variance='pooled', PPI++ Algorithm 2). The default is 'unlabeled' for
    lam=1 and 'pooled' otherwise.
    """
    data_type = PPIData

    def __init__(self, lam=1.0, variance=None, **kwargs):
        super().__init__(**kwargs)
        if lam != 'tuned' and not (isinstance(lam, (int, float, np.number)) and np.isfinite(lam)):
            raise ValueError("lam must be 'tuned' or a finite number")
        if variance is None:
            variance = 'unlabeled' if lam == 1 else 'pooled'
        if variance not in ('unlabeled', 'pooled'):
            raise ValueError("variance must be 'unlabeled' or 'pooled'")
        self.lam, self.variance = lam, variance
        self.name = 'PPI' if lam == 1 and variance == 'unlabeled' else 'PPI++'

    def _estimate(self, data, target):
        labels, f_labeled, f_unlabeled = data.labels, data.labeled_predictions, data.unlabeled_predictions
        lam = tuned_lambda(labels, f_labeled, f_unlabeled) if self.lam == 'tuned' else float(self.lam)
        residuals = labels - lam * f_labeled
        predictions = lam * f_unlabeled
        var_f = (f_unlabeled.var(ddof=1) if self.variance == 'unlabeled'
                 else np.concatenate([f_labeled, f_unlabeled]).var(ddof=1))
        point = float(predictions.mean() + residuals.mean())
        se = float(np.sqrt(lam ** 2 * var_f / len(f_unlabeled) + residuals.var(ddof=1) / len(residuals)))
        return point, se, {
            'values': {'residuals': residuals, 'unlabeled_predictions': predictions},
            'labeled_count': len(residuals),
            'unlabeled_count': len(predictions),
            'prediction_weight': lam,
            'weight_strategy': 'tuned' if self.lam == 'tuned' else 'fixed',
            'variance_design': 'independent_two_sample',
        }
