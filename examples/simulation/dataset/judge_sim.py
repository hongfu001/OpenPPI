"""Synthetic data generators."""
import pandas as pd

import numpy as np

from scipy.stats import qmc
from scipy.stats import norm
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss

class DGP:
    def __init__(self, c):
        self.c = c
        k = c.judges
        self.d = np.array([[.30, -.20, .10, .05, 0], [.20, -.10, .25, 0, .1],
                           [.35, -.15, 0, .10, 0], [.15, -.30, .15, 0, .05],
                           [.25, -.20, .20, .05, -.05]])[:k]
        self.load = c.signal * np.linspace(.6, 1.4, 5)[:k]
        if c.regime == 'weak':
            self.load *= .05
        if c.regime == 'inverted':       # judges reward what humans penalise (still informative, wrong sign)
            self.load *= -1
        self.temp = c.judge_temp if c.regime in ('overconfident', 'confident_failure') else 1.
        self.offset = np.full(k, c.bias if c.regime == 'biased' else 0.)
        common = c.common_sd if c.regime != 'correlated' else max(c.common_sd, 1.5)
        self.Q = common**2 * np.ones((k, k)) + c.noise_sd**2 * np.eye(k)
        self.Qinv = np.linalg.inv(self.Q)
        self.post_var = 1 / (1 + self.load @ self.Qinv @ self.load)
        self.gh, self.gw = hermgauss(40)

    def hard(self, x):
        return x[:, 4] <= norm.ppf(self.c.polarized)

    def eta(self, x, hval):
        core = self.base(x) + self.c.beta_h * hval
        if self.c.polarized > 0:
            core = np.where(self.hard(x), self.c.kappa_hard, self.c.kappa_easy) * core
        return self.c.beta0 + core

    def base(self, x):
        if self.c.model == 'linear':
            return .8 * x[:, 0] - .6 * x[:, 1] + .4 * x[:, 2]
        return .8 * np.sin(x[:, 0]) - .5 * (x[:, 1]**2 - 1) + .4 * x[:, 2] * x[:, 3]

    def draw(self, n, rng):
        x = rng.normal(size=(n, 5)); h = rng.normal(size=n)
        area = (x[:, 0] > 1) & (x[:, 1] < 0)
        flip = self.c.regime in ('shared_failure', 'confident_failure')
        sign = np.where(area, -1., 1.) if flip else np.ones(n)
        l = (self.offset + x @ self.d.T + sign[:, None] * h[:, None] * self.load
             + rng.multivariate_normal(np.zeros(self.c.judges), self.Q, size=n))
        j = expit(self.temp * l)     # temp != 1: miscalibrated (overconfident) but invertible -> same m(Z)
        y = rng.binomial(1, expit(self.eta(x, h)))
        # Oracle m(Z)=E[Y|X,J]: Gaussian posterior H|X,L (U integrated out via Q),
        # then 40-point Gauss-Hermite integration.
        residual = l - self.offset - x @ self.d.T
        post_mean = self.post_var * sign * (residual @ self.Qinv @ self.load)
        m = np.zeros(n)
        for a, w in zip(self.gh, self.gw):
            m += w * expit(self.eta(x, post_mean + np.sqrt(2 * self.post_var) * a))
        m /= np.sqrt(np.pi)
        return {'x': x, 'j': j, 'y': y.astype(float), 'area': area, 'hard': self.hard(x), 'oracle': m,
                # learner features: J, or logit(J) = temp * L computed exactly (no saturation at 0/1)
                'features': np.column_stack([x, j if self.c.judge_features == 'prob' else self.temp * l])}

    def truth(self):
        if self.c.model == 'linear' and self.c.beta0 == 0:
            return .5, 0.   # exact by symmetry (also with polarized: kappa depends only on X5)
        power = int(np.ceil(np.log2(self.c.truth_size)))
        vals = []
        for seed in range(4):
            z = norm.ppf(np.clip(qmc.Sobol(6, scramble=True, seed=900 + seed).random_base2(power),
                                 1e-12, 1 - 1e-12))
            vals.append(expit(self.eta(z[:, :5], z[:, 5])).mean())
        return float(np.mean(vals)), float(np.std(vals, ddof=1) / 2)

def make_judge_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    latent = rng.normal(size=n)
    judges = expit(latent[:, None] + rng.normal(size=(n, 5)))
    labels = rng.binomial(1, expit(.8*x[:, 0] - .6*x[:, 1] + latent))

    data = pd.DataFrame(
        np.column_stack([x, judges]),
        columns=[f'x{i}' for i in range(5)] + [f'judge{i}' for i in range(5)],
    )
    data['label'] = labels
    return data


def make_clustered_data(clusters=300, seed=0, cluster_sd=1.5, noise_sd=.1, bias=.05):
    """Units nested in iid clusters sharing a random effect; the unit-level mean of label is 0.5.

    prediction = clip(P(label = 1) + noise + bias, 0, 1) is a biased, fixed predictor.
    """
    rng = np.random.default_rng(seed)
    cluster = np.repeat(np.arange(clusters), rng.integers(2, 30, clusters))
    p = expit(rng.normal(0, cluster_sd, clusters)[cluster] + rng.normal(size=len(cluster)))
    return pd.DataFrame({
        'cluster': cluster,
        'label': rng.binomial(1, p).astype(float),
        'prediction': np.clip(p + rng.normal(0, noise_sd, len(cluster)) + bias, 0, 1),
    })


def make_multi_rater_data(n, seed=0, noise_sd=(2., 1., .4), signal=3.):
    """Columns: binary label, then one autorater score per noise level (less noise = better rater).

    label ~ Bernoulli(expit(signal * z)) and rater j scores expit(z + N(0, noise_sd[j]^2)) with
    z ~ N(0, 1), so the mean of the label is 0.5 by symmetry.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    label = rng.binomial(1, expit(signal * z)).astype(float)
    raters = [expit(z + rng.normal(0, sd, n)) for sd in noise_sd]
    return np.column_stack([label] + raters)
