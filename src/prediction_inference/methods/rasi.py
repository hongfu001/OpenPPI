"""Robust active inference: ASI correction on a robustly sampled record."""
from .asi import ASI
from ..data import RobustActiveData


class RASI(ASI):
    """Binary mean inference from burn-in plus robust Bernoulli queries.

    Burn-in rows carry propensity 1, so their corrected value is the label
    itself and the point estimate matches the reference implementation.
    """
    name = 'RASI'
    data_type = RobustActiveData

    def _estimate(self, data, target):
        point, se, diagnostics = super()._estimate(data, target)
        diagnostics.update(
            burn_in=int(data.burn_in.sum()),
            rho_robust=data.rho_robust,
            constraint=data.constraint,
            path=data.path,
            variance_design='robust_active_terminal_plugin',
        )
        return point, se, diagnostics
