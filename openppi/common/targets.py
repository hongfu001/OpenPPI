"""Statistical targets shared by sampling and inference."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class Target(ABC):
    @abstractmethod
    def sampling_scores(self, features, predictions):
        """Return query scores without inspecting outcomes."""


@dataclass(frozen=True)
class Mean(Target):
    """Population mean; outcome_type controls only the default query score.

    Binary mode preserves the ASI proxy 2*min(f,1-f). Continuous mode
    requires explicit outcome-free scores via select_for_labeling(scores=...).
    Inference formulas support real outcomes in either mode where the method
    permits them; no classification loss is imposed on the mean estimand.
    """
    outcome_type: str = 'binary'

    def __post_init__(self):
        if self.outcome_type not in ('binary', 'continuous'):
            raise ValueError("outcome_type must be 'binary' or 'continuous'")

    def sampling_scores(self, features, predictions):
        if self.outcome_type == 'continuous':
            raise ValueError('Continuous Mean requires explicit regression uncertainty scores via scores=')
        p = np.asarray(predictions, dtype=float)
        if p.ndim != 1 or len(p) != len(features) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError('Mean sampling requires aligned binary probabilities')
        return 2 * np.minimum(p, 1 - p)
