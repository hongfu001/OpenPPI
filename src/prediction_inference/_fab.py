"""Private FAB (frequentist, assisted by Bayes) formulas for W ~ N(beta, 1).

Priors are scaled by the noise sd, so everything is in standardized units
(Cortinovis & Caron 2025, eqs. 22-23 with tau = sigma). The FAB region at
level 1 - alpha is {beta : beta + z_{alpha (1 - w)} <= w_obs <= beta + z_{1 - alpha w}},
where the spending function w(beta) equalizes pi(y) / phi(y - beta) at the
two acceptance endpoints (Neyman-Pearson; supplementary S1.2).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import dawsn, hyp1f1, ndtri

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
