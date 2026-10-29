"""Common inference-method lifecycle."""
from abc import ABC, abstractmethod

from ..common.results import InferenceResult
from ..common.stats import wald_interval
from ..common.targets import Mean


class InferenceMethod(ABC):
    name: str
    data_type: type
    supported_targets: tuple = (Mean,)

    def __init__(self, interval=wald_interval):
        self.interval = interval

    def __call__(self, data, *, target, alpha=.05):
        if not isinstance(data, self.data_type):
            raise TypeError(f'{self.name} requires {self.data_type.__name__}')
        if not isinstance(target, self.supported_targets):
            raise TypeError(f'{self.name} does not support target {type(target).__name__}')
        if not 0 < alpha < 1:
            raise ValueError('alpha must be in (0, 1)')
        point, se, diagnostics = self._estimate(data, target)
        ci = self.interval(point, se, alpha, values=diagnostics.pop('values', None))
        return InferenceResult(point, se, ci, alpha, self.name, diagnostics)

    @abstractmethod
    def _estimate(self, data, target):
        """Return (point, se, diagnostics), optionally including values."""
