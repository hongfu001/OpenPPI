"""Uniform sampling: every row has probability rho = budget / n (the PPI baseline)."""
import numpy as np

from .base import SamplingDesign, validate_calibration


class Uniform(SamplingDesign):
    """Independent Bernoulli queries with constant probability rho."""
    def calibrate(self, reference_scores, rho):
        validate_calibration(reference_scores, rho)
        self.rho = rho
        return self

    def probability(self, scores):
        return np.full(np.shape(scores), self.rho)
