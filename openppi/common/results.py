"""The result every inference method returns."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InferenceResult:
    estimate: float
    standard_error: float
    confidence_interval: tuple[float, float]
    alpha: float
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
