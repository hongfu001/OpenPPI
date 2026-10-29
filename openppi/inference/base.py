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
        if not isinstance(target, self.supported_targets):
            raise TypeError(f'{self.name} does not support target {type(target).__name__}')
        required = next((dtype for kind, dtype in getattr(self, 'target_data_types', {}).items()
                         if isinstance(target, kind)), self.data_type)
        if not isinstance(data, required):
            raise TypeError(f'{self.name} requires {required.__name__} for {type(target).__name__}')
        if not 0 < alpha < 1:
            raise ValueError('alpha must be in (0, 1)')
        point, se, diagnostics = self._estimate_with_alpha(data, target, alpha)
        ci = self.interval(point, se, alpha, values=diagnostics.pop('values', None))
        covariance = diagnostics.pop('covariance', None)
        return InferenceResult(point, se, ci, alpha, self.name, diagnostics, covariance=covariance)

    def _estimate_with_alpha(self, data, target, alpha):
        """Allow finite-sample estimators to depend on the requested error rate."""
        return self._estimate(data, target)

    @abstractmethod
    def _estimate(self, data, target):
        """Return (point, se, diagnostics), optionally including values."""
