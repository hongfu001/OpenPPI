"""MultiPPI (Cowen-Breen et al., ICLR 2026): unbiased linear combination of subset sample means."""
import numpy as np
from ..common.stats import ledoit_wolf, multi_weights
from .base import InferenceMethod
from ..common.data import MultiPPIData


class MultiPPI(InferenceMethod):
    """Estimate a^T E[X] as sum_I mean(X_I) @ lambda_I with lambda_I = n_I Sigma_I^-1 P_I S a.

    The weights use the realized counts and a covariance estimate: Ledoit-Wolf
    (default) or the sample covariance of the fully labeled rows, which are
    reused for estimation (Section 4.2.1), or a given covariance. Any fixed
    covariance keeps the estimate unbiased; the SE is
    sqrt(sum_I var(X_I @ lambda_I) / n_I) (Appendix C.2). a defaults to e_1,
    the mean of the label.
    """
    name = 'MultiPPI'
    data_type = MultiPPIData

    def __init__(self, covariance='ledoit_wolf', a=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(covariance, str) and covariance not in ('ledoit_wolf', 'empirical'):
            raise ValueError("covariance must be 'ledoit_wolf', 'empirical' or a matrix")
        self.covariance, self.a = covariance, a

    def _estimate(self, data, target):
        if isinstance(self.covariance, str):
            if data.full is None or len(data.full) <= data.dimension:
                raise ValueError('Estimating the covariance needs more fully labeled rows than sources')
            sigma = ledoit_wolf(data.full) if self.covariance == 'ledoit_wolf' else np.cov(data.full.T)
        else:
            sigma = np.asarray(self.covariance, dtype=float)
            if sigma.ndim != 2 or len(sigma) < data.dimension:
                raise ValueError('The covariance must cover every index in the samples')
        a = np.eye(len(sigma))[0] if self.a is None else np.asarray(self.a, dtype=float)
        counts = {subset: len(rows) for subset, rows in data.samples.items()}
        weights, planned = multi_weights(sigma, counts, a)
        values = {subset: data.samples[subset] @ w for subset, w in weights.items()}
        point = float(sum(v.mean() for v in values.values()))
        se = float(np.sqrt(sum(v.var(ddof=1) / len(v) for v in values.values())))
        return point, se, {
            'values': values,
            'counts': counts,
            'weights': {subset: w.tolist() for subset, w in weights.items()},
            'planned_variance': planned,
            'covariance': self.covariance if isinstance(self.covariance, str) else 'given',
            'variance_design': 'independent_subsets',
        }
