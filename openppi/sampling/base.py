"""Shared pieces for fixed-budget sampling: design interface and one-batch selection."""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import numpy as np

from ..common.data import ActiveData
from ..common.targets import Mean


class SamplingDesign(ABC):
    """Maps outcome-free scores to sampling probabilities for one batch.

    sample and record are hooks with independent-Bernoulli defaults;
    joint designs such as cube sampling override both.
    """
    @abstractmethod
    def calibrate(self, reference_scores, rho):
        """Calibrate from outcome-free scores and return this design."""

    @abstractmethod
    def probability(self, scores):
        """Return sampling probabilities for scores, fixed before any label is seen."""

    def sample(self, scores, propensities, rng):
        """Draw selection indicators; default is independent (Poisson) sampling."""
        return rng.random(len(propensities)) < propensities

    def record(self, predictions, observed_labels, queried, propensities):
        """Build the inference record once labels are attached."""
        return ActiveData(predictions, observed_labels, queried, propensities)


def validate_calibration(reference_scores, rho):
    scores = np.asarray(reference_scores, dtype=float)
    if not 0 < rho < 1:
        raise ValueError('rho must be in (0, 1)')
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Require a nonempty vector of finite nonnegative scores')
    return scores


@dataclass(frozen=True)
class LabelRequest:
    """Rows selected for human labeling. Holds selection only, no labels.

    predictions and propensities cover all n rows and are fixed before any
    label is seen. design is the calibrated copy that produced them, kept
    for diagnostics. Arrays are copied and made read-only, as in ActiveData.
    """
    predictions: np.ndarray
    propensities: np.ndarray
    queried: np.ndarray
    design: SamplingDesign = None

    def __post_init__(self):
        for name in ('predictions', 'propensities', 'queried'):
            raw = np.asarray(getattr(self, name))
            if name == 'queried' and raw.dtype.kind != 'b':
                raise ValueError('queried must contain booleans')
            array = np.array(raw, dtype=bool if name == 'queried' else float, copy=True)
            if array.ndim != 1:
                raise ValueError('All arrays must be one-dimensional')
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if len({len(self.predictions), len(self.propensities), len(self.queried)}) != 1:
            raise ValueError('predictions, propensities and queried must align')

    @property
    def indices(self):
        """Row numbers to hand to the annotators, in increasing order."""
        return np.flatnonzero(self.queried)

    def with_labels(self, labels):
        """Return ActiveData given labels aligned one-to-one with indices."""
        labels = np.asarray(labels, dtype=float)
        if labels.shape != (len(self.indices),):
            raise ValueError('Require one label per entry of indices, in the same order')
        observed = np.full(len(self.queried), np.nan)
        observed[self.indices] = labels
        build = ActiveData if self.design is None else self.design.record
        return build(self.predictions, observed, self.queried, self.propensities)


def select_for_labeling(predictions, *, target, design, budget, scores=None, random_state=None):
    """Select one batch of rows to label from fixed-model predictions.

    Reads no labels. scores defaults to target.sampling_scores; pass
    outcome-free scores from elsewhere, such as RobustDesign.errors, to
    override them. The design is copied, calibrated on the batch's own
    scores with rho = budget / n, draws the selection (independently
    unless the design overrides sample), and is recorded in the request.
    Label rows request.indices, then call with_labels.
    """
    if not isinstance(target, Mean):
        raise TypeError('Only the binary Mean target is currently supported')
    p = np.asarray(predictions, dtype=float)
    if p.ndim != 1 or len(p) < 2 or not 0 < budget < len(p):
        raise ValueError('Require a prediction vector with n >= 2 and 0 < budget < n')
    if not isinstance(design, SamplingDesign):
        raise TypeError('design must implement SamplingDesign')
    if scores is None:
        scores = target.sampling_scores(np.empty((len(p), 0)), p)
    else:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != p.shape or not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError('scores must be finite, nonnegative and aligned with predictions')
    design = deepcopy(design).calibrate(scores, budget / len(p))
    propensities = np.asarray(design.probability(scores), dtype=float)
    if propensities.shape != p.shape or not np.isfinite(propensities).all() \
            or not ((propensities > 0) & (propensities <= 1)).all():
        raise ValueError('Sampling design must return probabilities in (0, 1]')
    queried = np.asarray(design.sample(scores, propensities, np.random.default_rng(random_state)))
    if queried.shape != p.shape or queried.dtype.kind != 'b':
        raise ValueError('Sampling design must return one boolean per row')
    return LabelRequest(p, propensities, queried, design)
