"""Validated snapshots of the information available at query time."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ActiveData:
    """Aligned query-time arrays. Unqueried labels must be NaN.

    Predictions and propensities must be chosen before seeing each label,
    using only available covariates and past observations. Arrays are copied
    and made read-only so later model updates cannot overwrite this record.
    """
    predictions: np.ndarray
    observed_labels: np.ndarray
    queried: np.ndarray
    propensities: np.ndarray

    def __post_init__(self):
        for name in ('predictions', 'observed_labels', 'queried', 'propensities'):
            raw = np.asarray(getattr(self, name))
            if name == 'queried' and raw.dtype.kind != 'b':
                raise ValueError('queried must contain booleans')
            array = np.array(raw, dtype=bool if name == 'queried' else float, copy=True)
            if array.ndim != 1:
                raise ValueError('All arrays must be one-dimensional')
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if len(self.predictions) < 2 or len({len(self.predictions), len(self.observed_labels), len(self.queried), len(self.propensities)}) != 1:
            raise ValueError('Aligned arrays of length >= 2 are required')
        if not np.isfinite(self.predictions).all():
            raise ValueError('predictions must be finite')
        if not (np.isfinite(self.propensities).all() and ((self.propensities > 0) & (self.propensities <= 1)).all()):
            raise ValueError('propensities must be finite and in (0, 1]')
        if not np.isfinite(self.observed_labels[self.queried]).all():
            raise ValueError('Queried labels must be finite')
        if not np.isnan(self.observed_labels[~self.queried]).all():
            raise ValueError('Unqueried labels must be NaN')


@dataclass(frozen=True)
class PPIData:
    """Disjoint iid labeled and unlabeled samples for binary mean PPI.

    The predictor must be trained independently of both inference samples.
    Only the labeled sample needs observed outcomes. Arrays are snapshots.
    """
    labels: np.ndarray
    labeled_predictions: np.ndarray
    unlabeled_predictions: np.ndarray

    def __post_init__(self):
        for name in ('labels', 'labeled_predictions', 'unlabeled_predictions'):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
                raise ValueError('PPI requires finite vectors with at least two rows per sample')
            if not ((array >= 0) & (array <= 1)).all():
                raise ValueError('Binary Mean PPI requires probabilities in [0, 1]')
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if len(self.labels) != len(self.labeled_predictions):
            raise ValueError('Labels and labeled predictions must align')
        if not np.isin(self.labels, [0., 1.]).all():
            raise ValueError('Mean PPI currently requires binary labels')


ROBUST_PATHS = ('linear', 'geometric', 'hellinger')


@dataclass(frozen=True)
class RobustActiveData(ActiveData):
    """ActiveData from a burn-in batch plus a robust second batch.

    burn_in marks uniformly chosen rows labeled with probability 1 before
    the error model was fit; they count toward the labeling budget.
    rho_robust, constraint and path describe the second-batch design.
    """
    burn_in: np.ndarray = None
    rho_robust: float = np.nan
    constraint: float = np.nan
    path: str = 'geometric'

    def __post_init__(self):
        super().__post_init__()
        burn_in = np.array(self.burn_in, dtype=bool, copy=True)
        if burn_in.shape != self.queried.shape:
            raise ValueError('burn_in must align with the other arrays')
        if not (self.queried[burn_in].all() and (self.propensities[burn_in] == 1).all()):
            raise ValueError('Burn-in rows must be queried with propensity 1')
        if self.path not in ROBUST_PATHS:
            raise ValueError(f'path must be one of {ROBUST_PATHS}')
        burn_in.setflags(write=False)
        object.__setattr__(self, 'burn_in', burn_in)


@dataclass(frozen=True)
class BalancedActiveData(ActiveData):
    """ActiveData from cube (balanced) sampling.

    balancing is the n x p matrix of balancing variables x used by the cube
    method, which enforced sum(x * queried / propensities) ~ sum(x). The
    variance estimator needs it because the queries are not independent.
    """
    balancing: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        balancing = np.array(self.balancing, dtype=float, copy=True)
        if balancing.ndim == 1:
            balancing = balancing[:, None]
        if balancing.ndim != 2 or len(balancing) != len(self.queried) or not balancing.shape[1] \
                or not np.isfinite(balancing).all():
            raise ValueError('balancing must be a finite n x p matrix aligned with the other arrays')
        balancing.setflags(write=False)
        object.__setattr__(self, 'balancing', balancing)
