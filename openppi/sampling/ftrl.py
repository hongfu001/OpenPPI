"""FTRL sampling (Sfyraki and Wang, ICLR 2026): covariate-oblivious query probabilities.

Rows are visited in order. Row t is queried with probability

    p_t = max(beta, min(tau, -gamma * theta_{t-1})),  theta_t = theta_{t-1} - Phi_t / p_t^2,

the closed form of Follow-the-Regularized-Leader on the losses Phi_t / p
(Lemma 3), with theta_0 = 0, tau = budget / n, beta = tau / 8 and
gamma = 1 / sqrt(n) as in the reference code. Phi_t >= 0 approximates
E[(Y - f(X))^2 | X] and is passed as the scores. p_t depends only on rows
before t, so with a fixed model the whole sequence is set before any label
is seen; it starts at beta and rises to tau. The estimator and interval are
those of ASI, so inference uses IPW().
"""
import numpy as np

from .base import SamplingDesign, validate_calibration


class FTRLDesign(SamplingDesign):
    """No-regret query probabilities that ignore the current row's covariates.

    Rows should be in arrival (or random) order. beta_fraction sets
    beta = beta_fraction * tau; gamma defaults to 1 / sqrt(n). Since
    p_t <= tau, the expected label count never exceeds the budget.
    """
    def __init__(self, beta_fraction=1 / 8, gamma=None):
        if not 0 < beta_fraction <= 1:
            raise ValueError('beta_fraction must be in (0, 1]')
        if gamma is not None and not gamma > 0:
            raise ValueError('gamma must be positive')
        self.beta_fraction = beta_fraction
        self.gamma = gamma

    def calibrate(self, reference_scores, rho):
        scores = validate_calibration(reference_scores, rho)
        self.tau = rho
        self.beta = self.beta_fraction * rho
        self.step = 1 / np.sqrt(len(scores)) if self.gamma is None else self.gamma
        return self

    def probability(self, scores):
        phi = np.asarray(scores, dtype=float)
        p = np.empty(phi.shape)
        theta = 0.
        for t, loss in enumerate(phi):
            p[t] = max(self.beta, min(self.tau, -self.step * theta))
            theta -= loss / p[t] ** 2
        return p
