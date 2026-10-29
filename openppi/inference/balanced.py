"""Balanced active inference: GD estimator with a cube-sampling variance estimator."""
import numpy as np

from .base import InferenceMethod
from ..common.data import BalancedActiveData
from ..common.stats import ipw_correct, wald_interval


def cube_variance(residuals, balancing, propensities, n, max_iter=100, tol=1e-6):
    """Deville-Tille variance approximation for a balanced (cube) sample.

    residuals, balancing and propensities are restricted to selected rows;
    n is the pool size. Calibration constants c solve the fixed point of the
    reference estimate_cube_variance; residuals are regressed on the
    balancing variables with weights c / pi^2, and the variance of the
    mean is sum(c * e^2 / pi^2) / n^2.
    """
    y = np.asarray(residuals, dtype=float)
    x = np.asarray(balancing, dtype=float)
    pi = np.asarray(propensities, dtype=float)
    m, p = x.shape
    if m <= p:
        raise ValueError('Cube variance needs more selected rows than balancing variables')
    c = m / (m - p) * (1 - pi)
    for _ in range(max_iter):
        inverse = np.linalg.pinv((x * (c / pi ** 2)[:, None]).T @ x)
        quad = np.einsum('ij,jk,ik->i', x, inverse, x) / pi
        disc = 1 + 4 * quad * (1 - pi)
        safe = np.where(quad > 0, quad, 1.)
        update = np.where(quad > 0, (np.sqrt(disc) - 1) / (2 * safe), 1 - pi)
        done = np.max(np.abs(update - c)) < tol
        c = update
        if done:
            break
    weights = c / pi ** 2
    coef = np.linalg.lstsq(x * np.sqrt(weights)[:, None], y * np.sqrt(weights), rcond=None)[0]
    return float(np.sum(c * (y - x @ coef) ** 2 / pi ** 2) / n ** 2)


class BAI(InferenceMethod):
    """Real-valued mean inference from cube-balanced queries.

    The point estimate is the GD (AIPW) mean, as in ASI. The standard error
    is the cube design variance plus, by default, the Hajek-weighted label
    variance over n, so the interval targets the population mean E[Y] like
    ASI and PPI. finite_population=True keeps only the design variance and
    targets the mean of the pool's labels, as in the reference code.
    """
    name = 'BAI'
    data_type = BalancedActiveData

    def __init__(self, interval=wald_interval, finite_population=False):
        super().__init__(interval)
        self.finite_population = finite_population

    def _estimate(self, data, target):
        values = ipw_correct(data.predictions, data.observed_labels,
                             data.queried, data.propensities)
        q = data.queried
        n = len(values)
        y, f, pi = data.observed_labels[q], data.predictions[q], data.propensities[q]
        design = cube_variance(y - f, data.balancing[q], pi, n)
        w = 1 / pi
        label_var = float(np.sum(w * (y - np.sum(w * y) / w.sum()) ** 2) / w.sum())
        population = 0. if self.finite_population else label_var / n
        x = data.balancing
        imbalance = (x[q] / pi[:, None]).sum(0) - x.sum(0)
        return float(values.mean()), float(np.sqrt(design + population)), {
            'values': values,
            'audits': int(q.sum()),
            'sum_sampling_probabilities': float(data.propensities.sum()),
            'min_pi': float(data.propensities.min()),
            'max_pi': float(data.propensities.max()),
            'design_variance': design,
            'population_variance': population,
            'relative_imbalance': (np.abs(imbalance) / np.maximum(np.abs(x.sum(0)), 1e-12)).tolist(),
            'variance_design': 'cube_finite_population' if self.finite_population
                               else 'cube_plus_population',
        }
