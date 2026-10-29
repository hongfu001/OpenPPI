"""ASI sampling (Zrnic and Candes, 2024): uncertainty-proportional rule mixed with uniform."""
import numpy as np

from ..common.targets import Mean
from .base import SamplingDesign, validate_calibration


class Uncertainty(SamplingDesign):
    """pi = (1 - tau) * clip(eta * u, 0, 1) + tau * rho, eta = rho / mean(u).

    Algorithm 1 of the paper. Clipping can leave the expected count
    slightly below budget; the uniform share tau keeps pi >= tau * rho.
    """
    def __init__(self, tau=.5):
        if not 0 < tau <= 1:
            raise ValueError('tau must be in (0, 1]')
        self.tau = tau

    def calibrate(self, reference_scores, rho):
        scores = validate_calibration(reference_scores, rho)
        self.rho = rho
        # All-zero reference scores carry no ranking; fall back to uniform.
        self.eta = rho / scores.mean() if scores.mean() > 0 else None
        return self

    def probability(self, scores):
        scores = np.asarray(scores, dtype=float)
        raw = np.full(scores.shape, self.rho) if self.eta is None else np.clip(self.eta * scores, 0, 1)
        return (1 - self.tau) * raw + self.tau * self.rho


def tune_tau(predictions, labels, *, target, rho, taus=None):
    """Choose tau by Eq. (12) on historical labeled data.

    Minimizes mean((Y - f)^2 / pi_tau(X)) over the grid taus, where
    pi_tau is Uncertainty(tau) calibrated on the historical scores. Use
    predictions from a model not trained on these labels, e.g. a
    held-out split of the training data.
    """
    if not isinstance(target, Mean):
        raise TypeError('Only the binary Mean target is currently supported')
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.ndim != 1 or p.shape != y.shape or len(p) < 2 or not np.isfinite(y).all():
        raise ValueError('Require aligned prediction and label vectors with n >= 2')
    taus = np.linspace(.01, .99, 100) if taus is None else np.asarray(taus, dtype=float)
    if taus.ndim != 1 or not len(taus):
        raise ValueError('taus must be a nonempty vector')
    scores = target.sampling_scores(np.empty((len(p), 0)), p)
    squared = (y - p) ** 2
    risks = [np.mean(squared / Uncertainty(tau=float(tau)).calibrate(scores, rho).probability(scores))
             for tau in taus]
    return float(taus[int(np.argmin(risks))])
