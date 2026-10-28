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
    """Population mean of binary labels, with ASI uncertainty sampling.

    Scores are the original binary ASI proxy 2*min(f, 1-f). They do not
    observe |Y-f| before querying and are not a universal optimal rule.
    Other targets require their own sampling scores and variance estimator.
    """
    def sampling_scores(self, features, predictions):
        p = np.asarray(predictions, dtype=float)
        if p.ndim != 1 or len(p) != len(features) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError('Mean sampling requires aligned binary probabilities')
        return 2 * np.minimum(p, 1 - p)
