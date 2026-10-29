"""Stationary spatial AR(1) covariates and errors, with exact site means."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter


@dataclass(frozen=True)
class TimeSeriesDGP:
    sites: int = 9
    phi: float = .7
    error_phi: float = .5
    quadratic: float = .4
    noise_sd: float = .5
    spatial_length: float = .3

    def __post_init__(self):
        if isinstance(self.sites, bool) or not isinstance(self.sites, (int, np.integer)) or self.sites < 1:
            raise ValueError('sites must be a positive integer')
        if not np.isfinite([self.phi, self.error_phi, self.quadratic,
                            self.noise_sd, self.spatial_length]).all():
            raise ValueError('DGP parameters must be finite')
        if abs(self.phi) >= 1 or abs(self.error_phi) >= 1:
            raise ValueError('AR coefficients must have absolute value < 1')
        if self.noise_sd < 0 or self.spatial_length <= 0:
            raise ValueError('Require noise_sd >= 0 and spatial_length > 0')

    @property
    def coordinates(self):
        return np.linspace(0, 1, self.sites)

    def truth(self):
        return 1 + .5 * self.coordinates

    def draw(self, n, rng):
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError('n must be an integer >= 2')
        s = self.coordinates
        covariance = np.exp(-np.abs(s[:, None] - s[None, :]) / self.spatial_length)
        chol = np.linalg.cholesky(covariance)

        def ar(phi):
            # Stationary initial state; all sites evolve jointly, without a burn-in.
            initial = rng.normal(size=self.sites) @ chol.T
            innovations = rng.normal(size=(n, self.sites)) @ chol.T
            return lfilter([np.sqrt(1 - phi**2)], [1, -phi], innovations,
                           axis=0, zi=phi * initial[None, :])[0]

        x, error = ar(self.phi), ar(self.error_phi)
        y = (self.truth() + (.6 + .2*s) * x + self.quadratic * (x**2 - 1)
             + self.noise_sd * error)
        # The fitted linear predictor omits x^2: rectification is necessary.
        locations = np.broadcast_to(s, x.shape)
        features = np.stack([locations, x, locations*x], axis=-1)
        return dict(x=x, y=y, features=features, truth=self.truth())
