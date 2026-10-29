"""FAB-PPI (Cortinovis & Caron, ICML 2025): PPI with a prior on the rectifier."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtri
from .._fab import PRIORS, fab_region, posterior_mean
from ..base import Method
from ..data import PPIData
from .ppi_plus import tuned_lambda


class FABPPI(Method):
    """Binary mean FAB-PPI (Algorithm 2 of the paper) and, with lam=None, FAB-PPI++.

    The rectifier Delta = mean(Y - lam f) + (lam - 1) mean(f_unlabeled) gets a
    horseshoe (default) or Gaussian prior scaled by its own standard error.
    The estimate is mean(f_unlabeled) + E[Delta | Delta_hat]; the interval is
    the 1 - delta FAB region for Delta plus a 1 - (alpha - delta) Wald interval
    for mean(f_unlabeled). delta=None minimizes the width over (0, alpha) as the
    official code does; imputed_uncertainty=False drops the second interval and
    uses delta = alpha, as in the paper's experiments with N >> n.
    standard_error is the frequentist SE of the unshrunk PPI estimate.
    """
    data_type = PPIData

    def __init__(self, lam=1., prior='horseshoe', delta=None, imputed_uncertainty=True, interval=None):
        super().__init__(interval or self._fab_interval)
        if lam is not None and not np.isfinite(lam):
            raise ValueError('lam must be finite or None')
        if prior not in PRIORS:
            raise ValueError(f'prior must be one of {PRIORS}')
        if delta is not None and not 0 < delta:
            raise ValueError('delta must be positive')
        self.lam, self.prior, self.delta, self.imputed_uncertainty = lam, prior, delta, imputed_uncertainty
        self.name = 'FAB-PPI++' if lam is None else 'FAB-PPI'

    def _estimate(self, data, target):
        labels, f_labeled, f_unlabeled = data.labels, data.labeled_predictions, data.unlabeled_predictions
        n, big_n = len(labels), len(f_unlabeled)
        lam = tuned_lambda(labels, f_labeled, f_unlabeled) if self.lam is None else float(self.lam)
        imputed = float(f_unlabeled.mean())
        imputed_se = float(np.sqrt(f_unlabeled.var(ddof=1) / big_n))
        rectifier = float((labels - lam * f_labeled).mean() + (lam - 1) * imputed)
        rectifier_se = float(np.sqrt((labels - lam * f_labeled).var(ddof=1) / n
                                     + (lam - 1) ** 2 * imputed_se ** 2))
        if rectifier_se == 0:
            raise ValueError('FAB-PPI requires a rectifier with positive variance')
        shrunk = rectifier_se * posterior_mean(rectifier / rectifier_se, self.prior)
        return imputed + shrunk, float(np.hypot(rectifier_se, imputed_se)), {
            'values': {'imputed_mean': imputed, 'imputed_se': imputed_se,
                       'rectifier': rectifier, 'rectifier_se': rectifier_se},
            'labeled_count': n,
            'unlabeled_count': big_n,
            'prediction_weight': lam,
            'weight_strategy': 'tuned' if self.lam is None else 'fixed',
            'prior': self.prior,
            'rectifier': rectifier,
            'rectifier_posterior_mean': shrunk,
            'variance_design': 'independent_two_sample',
        }

    def _fab_interval(self, point, se, alpha, values):
        m, m_se, r, r_se = (values[k] for k in ('imputed_mean', 'imputed_se', 'rectifier', 'rectifier_se'))

        def interval(delta):
            low, high = fab_region(r / r_se, delta, self.prior)
            pad = m_se * ndtri(1 - (alpha - delta) / 2) if self.imputed_uncertainty else 0.
            return float(m + r_se * low - pad), float(m + r_se * high + pad)

        if not self.imputed_uncertainty:
            return interval(alpha)
        if self.delta is not None:
            if not self.delta < alpha:
                raise ValueError('delta must lie in (0, alpha)')
            return interval(self.delta)
        best = minimize_scalar(lambda d: np.subtract(*interval(d)[::-1]), bounds=(0, alpha), method='bounded')
        return interval(best.x)
