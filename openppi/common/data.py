"""Validated snapshots of the information available at query time."""
from dataclasses import dataclass, field
from types import MappingProxyType
import numpy as np


@dataclass(frozen=True)
class ActiveData:
    """Aligned query-time arrays. Unqueried labels must be NaN.

    Predictions and propensities must be chosen before seeing each label,
    using only available covariates and past observations. Arrays are copied
    and made read-only so later model updates cannot overwrite this record.
    design_info records parameters of the sampling design (e.g. the burn-in
    size and robust rho of RASI); inference copies it into the diagnostics.
    """
    predictions: np.ndarray
    observed_labels: np.ndarray
    queried: np.ndarray
    propensities: np.ndarray
    design_info: dict = field(default_factory=dict)

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
        if not isinstance(self.design_info, dict):
            raise TypeError('design_info must be a dict')
        object.__setattr__(self, 'design_info', dict(self.design_info))


@dataclass(frozen=True)
class PPIData:
    """Disjoint iid labeled and unlabeled samples for real-valued mean PPI.

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
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if len(self.labels) != len(self.labeled_predictions):
            raise ValueError('Labels and labeled predictions must align')



@dataclass(frozen=True)
class ClusteredPPIData:
    """One row per unit for binary mean PPI with clustered data.

    clusters holds a cluster id per unit; clusters are iid draws, units within
    a cluster may be arbitrarily dependent. labeled marks the units whose label
    was revealed, each with a cluster-level probability chosen independently of
    the data; clusters may be fully, partially or not labeled. Unlabeled labels
    must be NaN. Arrays are copied and made read-only.
    """
    labels: np.ndarray
    predictions: np.ndarray
    labeled: np.ndarray
    clusters: np.ndarray

    def __post_init__(self):
        raw = np.asarray(self.labeled)
        if raw.dtype.kind != 'b':
            raise ValueError('labeled must contain booleans')
        for name, dtype in (('labels', float), ('predictions', float), ('labeled', bool), ('clusters', None)):
            array = np.array(getattr(self, name), dtype=dtype, copy=True)
            if array.ndim != 1:
                raise ValueError('All arrays must be one-dimensional')
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if len({len(self.labels), len(self.predictions), len(self.labeled), len(self.clusters)}) != 1:
            raise ValueError('labels, predictions, labeled and clusters must align')
        if not (np.isfinite(self.predictions).all() and ((self.predictions >= 0) & (self.predictions <= 1)).all()):
            raise ValueError('Binary Mean PPI requires probabilities in [0, 1]')
        if not np.isin(self.labels[self.labeled], [0., 1.]).all():
            raise ValueError('Mean PPI currently requires binary labels')
        if not np.isnan(self.labels[~self.labeled]).all():
            raise ValueError('Unlabeled labels must be NaN')
        if self.labeled.sum() < 2 or (~self.labeled).sum() < 1 or len(np.unique(self.clusters)) < 2:
            raise ValueError('Need two labeled units, one unlabeled unit and two clusters')


@dataclass(frozen=True)
class MultiPPIData:
    """Samples of subsets of X = (label, prediction_2, ..., prediction_k) for MultiPPI.

    samples maps an index tuple I (0 is the label) to an n_I x |I| array of
    iid draws of X_I, independent across subsets; columns follow the sorted
    indices. The subset containing every index holds the fully labeled rows,
    which also estimate the covariance unless one is given. Labels are binary.
    Arrays are copied and made read-only.
    """
    samples: dict

    def __post_init__(self):
        clean = {}
        for subset, rows in self.samples.items():
            key = tuple(sorted(subset))
            array = np.array(rows, dtype=float, copy=True)
            if array.ndim == 1:
                array = array[:, None]
            if key in clean or not key or key[0] < 0 or array.ndim != 2 or array.shape[1] != len(key):
                raise ValueError('samples must map distinct index tuples to n x |I| arrays')
            if len(array) < 2 or not np.isfinite(array).all():
                raise ValueError('Each subset needs at least two finite rows')
            if key[0] == 0 and not np.isin(array[:, 0], [0., 1.]).all():
                raise ValueError('MultiPPI Mean currently requires binary labels')
            array.setflags(write=False)
            clean[key] = array
        k = max(key[-1] for key in clean) + 1
        if not any(key[0] == 0 for key in clean):
            raise ValueError('At least one subset must contain the label (index 0)')
        object.__setattr__(self, 'samples', clean)
        object.__setattr__(self, 'dimension', k)

    @property
    def full(self):
        """The fully labeled rows (all k columns), or None."""
        return self.samples.get(tuple(range(self.dimension)))


def split(features, labels, *, train_size, labeled_size, random_state=None):
    """Randomly partition rows for independent training and two-sample PPI.

    Returns (X_train, y_train), (X_labeled, y_labeled), X_unlabeled.
    Sizes are integer row counts. Shuffling ignores labels; use iid source
    rows and train only on the returned training set. Unlabeled outcomes
    are deliberately omitted from the return value.
    """
    x = np.asarray(features)
    y = np.asarray(labels)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError('Require aligned feature matrix and label vector')
    for size in (train_size, labeled_size):
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 2:
            raise ValueError('train_size and labeled_size must be integer counts >= 2')
    if len(x) - train_size - labeled_size < 2:
        raise ValueError('Leave at least two rows for the unlabeled sample')
    indices = np.random.default_rng(random_state).permutation(len(x))
    train = indices[:train_size]
    labeled = indices[train_size:train_size + labeled_size]
    unlabeled = indices[train_size + labeled_size:]
    return (x[train], y[train]), (x[labeled], y[labeled]), x[unlabeled]

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


@dataclass(frozen=True)
class StratifiedPPIData:
    """Independent within-stratum PPI samples with known population weights.

    strata maps each stratum key to PPIData; weights maps the same keys to
    strictly positive population probabilities summing to one. The fixed
    partition and weights must be chosen independently of inference labels.
    Sample fractions need not equal population probabilities. Each stratum
    needs two labeled and two unlabeled rows, as validated by PPIData.
    Uncertainty in estimated population weights is not included.
    """
    strata: dict
    weights: dict

    def __post_init__(self):
        strata, weights = dict(self.strata), dict(self.weights)
        if not strata or strata.keys() != weights.keys():
            raise ValueError('strata and weights must have the same nonempty keys')
        if any(not isinstance(data, PPIData) for data in strata.values()):
            raise TypeError('Each stratum must be PPIData')
        weights = {key: float(value) for key, value in weights.items()}
        values = np.array(list(weights.values()))
        if not np.isfinite(values).all() or (values <= 0).any() or not np.isclose(values.sum(), 1., rtol=0, atol=1e-10):
            raise ValueError('Population weights must be positive, finite and sum to one')
        object.__setattr__(self, 'strata', MappingProxyType(strata))
        object.__setattr__(self, 'weights', MappingProxyType(weights))
