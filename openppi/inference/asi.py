"""Active IPW correction with a configurable interval function."""
import numpy as np
from .base import InferenceMethod
from ..common.data import ActiveData
from ..common.stats import ipw_correct


class ASI(InferenceMethod):
    """Binary mean inference from predictable Bernoulli queries."""
    name = 'ASI'
    data_type = ActiveData

    def _estimate(self, data, target):
        values = ipw_correct(data.predictions, data.observed_labels,
                             data.queried, data.propensities)
        point = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        return point, se, {
            'values': values,
            'audits': int(data.queried.sum()),
            'sum_sampling_probabilities': float(data.propensities.sum()),
            'min_pi': float(data.propensities.min()),
            'max_pi': float(data.propensities.max()),
            'prediction_weight': 1.0,
            'weight_strategy': 'fixed',
            'variance_design': 'active_terminal_plugin',
        }
