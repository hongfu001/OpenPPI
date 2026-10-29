"""FAB-PPI (Cortinovis & Caron, ICML 2025): PPI with a prior on the rectifier."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import dawsn, hyp1f1, ndtri
from ..common.data import PPIData
from .base import InferenceMethod
from .two_sample import tuned_lambda

# FAB (frequentist, assisted by Bayes) formulas for W ~ N(beta, 1).
#
# Priors are scaled by the noise sd, so everything is in standardized units
# (Cortinovis & Caron 2025, eqs. 22-23 with tau = sigma). The FAB region at
# level 1 - alpha is {beta : beta + z_{alpha (1 - w)} <= w_obs <= beta + z_{1 - alpha w}},
# where the spending function w(beta) equalizes pi(y) / phi(y - beta) at the
# two acceptance endpoints (Neyman-Pearson; supplementary S1.2).

PRIORS = ('horseshoe', 'gaussian')
_BIG = 1e300


def log_marginal(y, prior):
    """log pi(y) up to a constant: horseshoe via Dawson's function, Gaussian N(0, 2)."""
    if prior == 'gaussian':
        return -y * y / 4
    u = abs(y) / np.sqrt(2)
    return np.log(np.sqrt(2) * dawsn(u) / abs(y)) if u > 1e-8 else 0.


def posterior_mean(y, prior):
    """E[beta | W = y] (Tweedie). Horseshoe: (1 - kappa(y)) y, eq. S27."""
    if prior == 'gaussian':
        return y / 2
    if abs(y) < 4:    # Kummer ratio is stable here; the Dawson form is stable beyond.
        return y * (1 - 2 / 3 * hyp1f1(2, 2.5, -y * y / 2) / hyp1f1(1, 1.5, -y * y / 2))
    return 1 / (np.sqrt(2) * dawsn(y / np.sqrt(2))) - 1 / y


def spending(beta, alpha, prior):
    """Share w(beta) of alpha spent in the left tail; w(-beta) = 1 - w(beta)."""
    if beta > 0:
        return 1 - spending(-beta, alpha, prior)

    def tail(q, sign):
        if q <= 0:
            return _BIG
        z = sign * ndtri(q)
        return log_marginal(beta - z, prior) + z * z / 2

    return brentq(lambda w: tail(alpha * (1 - w), 1) - tail(alpha * w, -1), 0., 1., xtol=1e-12)


def fab_region(y, alpha, prior):
    """Endpoints of the 1 - alpha FAB interval for beta given W = y."""
    upper = lambda b: y + ndtri(min(1 - alpha * spending(b, alpha, prior), 1 - 1e-16)) - b
    lower = lambda b: y + ndtri(max(alpha * (1 - spending(b, alpha, prior)), 1e-300)) - b
    a = b = y + ndtri(1 - alpha)    # upper >= 0 here and decreases in b
    while upper(b) > 0:
        b += 1
    high = brentq(upper, a, b, xtol=1e-10) if b > a else a
    a = b = y + ndtri(alpha)        # lower <= 0 here and decreases in b
    while lower(a) < 0:
        a -= 1
    low = brentq(lower, a, b, xtol=1e-10) if b > a else b
    return low, high



class FABPPI(InferenceMethod):
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
