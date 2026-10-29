"""The result every inference method returns."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class InferenceResult:
    estimate: float | np.ndarray
    standard_error: float | np.ndarray
    confidence_interval: tuple[float | np.ndarray, float | np.ndarray]
    alpha: float
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    covariance: np.ndarray | None = None

    def __post_init__(self):
        if np.ndim(self.estimate) == 0:
            return
        point = np.array(self.estimate,dtype=float,copy=True)
        se = np.array(self.standard_error,dtype=float,copy=True)
        cov = np.array(self.covariance,dtype=float,copy=True)
        lower,upper = (np.array(a,dtype=float,copy=True) for a in self.confidence_interval)
        p = len(point)
        if point.ndim != 1 or se.shape != point.shape or lower.shape != point.shape or upper.shape != point.shape or cov.shape != (p,p):
            raise ValueError('Vector inference needs aligned coefficient arrays and a p x p covariance')
        if not all(np.isfinite(a).all() for a in (point,se,cov,lower,upper)) or (se < 0).any() or (lower > upper).any():
            raise ValueError('Invalid vector inference result')
        if not np.allclose(cov,cov.T) or np.linalg.eigvalsh(cov).min() < -1e-10*max(1.,np.linalg.norm(cov)):
            raise ValueError('Coefficient covariance must be symmetric positive semidefinite')
        if not np.allclose(se**2,np.diag(cov)):
            raise ValueError('Standard errors must agree with the covariance diagonal')
        for a in (point,se,cov,lower,upper):
            a.setflags(write=False)
        object.__setattr__(self,'estimate',point)
        object.__setattr__(self,'standard_error',se)
        object.__setattr__(self,'covariance',cov)
        object.__setattr__(self,'confidence_interval',(lower,upper))
