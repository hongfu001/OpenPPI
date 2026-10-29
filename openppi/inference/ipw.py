"""Inverse-propensity-weighted correction for actively labeled records (ASI, RASI, FTRL)."""
import numpy as np
from .base import InferenceMethod
from ..common.data import ActiveData
from ..common.stats import ipw_correct


class IPW(InferenceMethod):
    """Binary mean inference from predictable Bernoulli queries.

    The corrected values are f + R/pi (Y - f); the estimate is their mean
    and the SE their standard deviation over sqrt(n). Rows labeled with
    probability 1 (a RASI burn-in) contribute their label. The record's
    design_info is copied into the diagnostics.
    """
    name = 'IPW'
    data_type = ActiveData

    def _estimate(self, data, target):
        values = ipw_correct(data.predictions, data.observed_labels,
                             data.queried, data.propensities)
        point = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        diagnostics = {
            'values': values,
            'audits': int(data.queried.sum()),
            'sum_sampling_probabilities': float(data.propensities.sum()),
            'min_pi': float(data.propensities.min()),
            'max_pi': float(data.propensities.max()),
            'prediction_weight': 1.0,
            'weight_strategy': 'fixed',
            'variance_design': 'active_terminal_plugin',
        }
        diagnostics.update(data.design_info)
        return point, se, diagnostics
