"""Continuous outcomes and correlated, miscalibrated judges; exact mean E[Y]=1."""
from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class RegressionDGP:
    noise: str = 'homoskedastic'
    judges: int = 3
    offsets: tuple | None = None
    slopes: tuple | None = None
    common_loadings: tuple | None = None
    judge_noise: tuple | None = None

    def __post_init__(self):
        if self.noise not in ('homoskedastic', 'heteroskedastic'):
            raise ValueError('noise must be homoskedastic or heteroskedastic')
        if isinstance(self.judges, bool) or not isinstance(self.judges, int) or self.judges < 1:
            raise ValueError('judges must be a positive integer')
        defaults = dict(offsets=np.linspace(-.3,.5,self.judges),
                        slopes=np.linspace(.7,1.3,self.judges),
                        common_loadings=np.linspace(.2,.6,self.judges),
                        judge_noise=np.linspace(.3,.7,self.judges))
        for name, default in defaults.items():
            values = default if getattr(self,name) is None else np.asarray(getattr(self,name),dtype=float)
            if values.shape != (self.judges,) or not np.isfinite(values).all():
                raise ValueError(f'{name} must have one finite value per judge')
            if name == 'judge_noise' and (values < 0).any():
                raise ValueError('judge_noise must be nonnegative')
            object.__setattr__(self,name,tuple(values))

    @staticmethod
    def conditional_mean(x):
        return 1 + .8*x[:,0] - .6*x[:,1] + .4*(x[:,2]**2-1)

    def draw(self, n, rng):
        if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n < 2:
            raise ValueError('n must be an integer >= 2')
        x = rng.normal(size=(n,5))
        mean = self.conditional_mean(x)
        sigma = np.full(n,.5) if self.noise=='homoskedastic' else .3+.7*expit(x[:,0])
        # All judge noise is independent of the human-label noise.
        y = mean + sigma*rng.normal(size=n)
        common = rng.normal(size=(n,1))
        judges = (np.asarray(self.offsets) + mean[:,None]*np.asarray(self.slopes)
                  + common*np.asarray(self.common_loadings)
                  + rng.normal(size=(n,self.judges))*np.asarray(self.judge_noise))
        return dict(x=x, y=y, j=judges, mean=mean, sigma=sigma,
                    features=np.column_stack([x,judges]))

    @staticmethod
    def truth():
        """Exact E[Y]; centered Gaussian covariates and noise contribute zero."""
        return 1.

    @staticmethod
    def projection_coefficients(fit_intercept=True):
        """Exact OLS projection on X: the centered quadratic is orthogonal to X.

        This is the population projection, despite a misspecified linear
        conditional-mean model. It is unrelated to the judge-feature slopes.
        """
        slopes = np.array([.8, -.6, 0., 0., 0.])
        return np.r_[1., slopes] if fit_intercept else slopes
