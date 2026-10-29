"""StratPPI mean inference (Fisch et al., NeurIPS 2024, Algorithm 1)."""
from collections.abc import Mapping

import numpy as np

from .base import InferenceMethod
from .two_sample import TwoSample
from ..common.data import StratifiedPPIData


class StratPPI(InferenceMethod):
    """Population-weighted within-stratum correction and independent variance.

    lam is 'tuned' (one weight per stratum, eq. 17), a finite scalar, or a
    mapping from stratum keys to finite weights. Mean weights are unconstrained.
    Variance uses within-stratum unlabeled predictions (eq. 13), or optionally
    pooled predictions as in the existing TwoSample implementation. Tuning
    uses pooled prediction variance, the same plug-in convention as PPI++.
    """
    name = 'StratPPI'
    data_type = StratifiedPPIData

    def __init__(self, lam='tuned', variance='unlabeled', **kwargs):
        super().__init__(**kwargs)
        if isinstance(lam, Mapping):
            lam = dict(lam)
            for value in lam.values():
                if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
                    raise ValueError('Each stratum weight must be a finite number')
        else:
            TwoSample(lam=lam, variance=variance)  # shared parameter validation
        if variance not in ('unlabeled', 'pooled'):
            raise ValueError("variance must be 'unlabeled' or 'pooled'")
        self.lam, self.variance = lam, variance

    def _estimate(self, data, target):
        if isinstance(self.lam, dict) and self.lam.keys() != data.strata.keys():
            raise ValueError('lam must contain exactly the stratum keys')
        point, variance, details = 0., 0., {}
        for key, sample in data.strata.items():
            lam = self.lam[key] if isinstance(self.lam, dict) else self.lam
            estimate, se, diagnostics = TwoSample(lam=lam, variance=self.variance)._estimate(sample, target)
            w = data.weights[key]
            point += w * estimate
            variance += w**2 * se**2
            diagnostics.pop('values')
            details[key] = {'population_weight': w, 'estimate': estimate,
                            'variance_contribution': w**2 * se**2, **diagnostics}
        return float(point), float(np.sqrt(variance)), {
            'strata': details,
            'labeled_count': sum(len(s.labels) for s in data.strata.values()),
            'unlabeled_count': sum(len(s.unlabeled_predictions) for s in data.strata.values()),
            'variance_design': 'independent_stratified_two_sample',
            'prediction_variance': self.variance,
        }
