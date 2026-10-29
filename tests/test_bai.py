"""Balanced active inference: cube sampling, variance, coverage."""
import numpy as np
import pytest

from openppi import BAI, IPW, BalancedDesign, Mean, select_for_labeling
from openppi.sampling.balanced import cube_sample
from openppi.inference.balanced import cube_variance
from config import Config
from utils import rng_for
from dataset.judge_sim import DGP


def reference_cube_variance(y, x, xi, pi, n, max_iter=100, tol=1e-6):
    """estimate_cube_variance from the reference code, restricted to selected rows."""
    s = np.where(xi == 1)[0]
    y, x, pi = y[s], x[s, :], pi[s]
    m, p = len(y), x.shape[1]
    c = (m / (m - p)) * (1.0 - pi)
    for _ in range(max_iter):
        M = sum(c[i] / pi[i] ** 2 * np.outer(x[i], x[i]) for i in range(m))
        Minv = np.linalg.inv(M)
        c_new = np.empty(m)
        for k in range(m):
            A = (x[k] @ Minv @ x[k]) / pi[k]
            disc = 1.0 + 4.0 * A * (1.0 - pi[k])
            c_new[k] = 1.0 - pi[k] if disc < 0 or A == 0 else (-1.0 + np.sqrt(disc)) / (2.0 * A)
        if np.max(np.abs(c_new - c)) < tol:
            break
        c = c_new
    w = c / pi ** 2
    b = np.linalg.solve((x.T * w) @ x, (x.T * w) @ y)
    return (c * (y - x @ b) ** 2 / pi ** 2).sum() / n ** 2


@pytest.mark.slow
def test_cube_sample_balances_and_keeps_inclusion_probabilities():
    rng = np.random.default_rng(0)
    u = rng.uniform(.05, 1, 60)
    pi = np.clip(.5 * .25 * u / u.mean() + .125, 0, 1)
    draws = np.array([cube_sample(pi, u, np.random.default_rng(s)) for s in range(3000)])
    # Landing settles at most one row, so the imbalance is below one weighted value.
    imbalance = np.abs((draws * (u / pi)).sum(1) - u.sum())
    assert imbalance.max() <= (u / pi).max() + 1e-9
    z = (draws.mean(0) - pi) / np.sqrt(pi * (1 - pi) / 3000)
    assert abs(z.mean()) < .5 and np.abs(z).max() < 4.5


def test_fixed_size_selects_about_sum_pi():
    rng = np.random.default_rng(1)
    u = rng.uniform(.05, 1, 200)
    pi = np.clip(.5 * .2 * u / u.mean() + .1, 0, 1)
    for s in range(50):
        xi = cube_sample(pi, np.column_stack([pi, u]), np.random.default_rng(s))
        assert abs(xi.sum() - pi.sum()) <= 1 + 1e-9


def test_cube_variance_matches_reference():
    rng = np.random.default_rng(2)
    n = 400
    x = np.column_stack([rng.uniform(.05, 1, n), rng.normal(size=n)])
    pi = np.clip(.3 * x[:, 0] / x[:, 0].mean(), .05, 1)
    y = rng.normal(size=n) + x[:, 1]
    xi = cube_sample(pi, x, rng)
    ours = cube_variance(y[xi], x[xi], pi[xi], n)
    assert ours == pytest.approx(reference_cube_variance(y, x, xi.astype(int), pi, n), rel=1e-4)


def test_bai_point_matches_asi_and_variance_parts():
    rng = np.random.default_rng(4)
    p = rng.uniform(.05, .95, 500)
    y = rng.binomial(1, p).astype(float)
    request = select_for_labeling(p, target=Mean(), design=BalancedDesign(), budget=100, random_state=4)
    record = request.with_labels(y[request.indices])
    result = BAI()(record, target=Mean())
    finite = BAI(finite_population=True)(record, target=Mean())
    assert result.estimate == IPW()(record, target=Mean()).estimate == finite.estimate
    d = result.diagnostics
    assert result.standard_error ** 2 == pytest.approx(d['design_variance'] + d['population_variance'])
    assert finite.standard_error ** 2 == pytest.approx(d['design_variance'])
    assert d['relative_imbalance'][0] < .05 and 'values' not in d


@pytest.mark.slow
def test_bai_interval_coverage():
    config = Config(n=800, reps=150)
    dgp = DGP(config)
    truth, _ = dgp.truth()
    covered = 0
    for rep in range(config.reps):
        data = dgp.draw(config.n, rng_for(config.seed, rep, 1))
        request = select_for_labeling(data['oracle'], target=Mean(), design=BalancedDesign(),
                                      budget=config.rho * config.n, random_state=rng_for(config.seed, rep, 2))
        record = request.with_labels(data['y'][request.indices])
        low, high = BAI()(record, target=Mean(), alpha=config.alpha).confidence_interval
        covered += low <= truth <= high
    coverage = covered / config.reps
    assert .88 < coverage <= 1.0, f'coverage={coverage:.3f}'
