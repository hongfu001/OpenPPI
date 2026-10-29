"""FAB-PPI: horseshoe/Gaussian formulas, FAB region coverage, robustness, method coverage."""
import numpy as np
import pytest
from scipy import integrate, stats
from scipy.optimize import brentq
from scipy.special import ndtri

from prediction_inference import FABPPI, PPIPlusPlus, Mean, PPIData
from prediction_inference._fab import fab_region, posterior_mean, spending
from common.config import Config
from common.utils import rng_for
from dataset.judge_sim import DGP


def horseshoe_density(beta):
    return integrate.quad(lambda v: stats.norm.pdf(beta, scale=v) * 2 / np.pi / (1 + v * v), 0, np.inf)[0]


@pytest.mark.parametrize('y', [.3, -1.2, 2.5, 6.])
def test_horseshoe_posterior_mean_matches_integration(y):
    weight = lambda b: horseshoe_density(b) * stats.norm.pdf(y - b)
    num = integrate.quad(lambda b: b * weight(b), -40, 40, points=[0, y], limit=400)[0]
    den = integrate.quad(weight, -40, 40, points=[0, y], limit=400)[0]
    assert posterior_mean(y, 'horseshoe') == pytest.approx(num / den, rel=1e-6)


@pytest.mark.parametrize('y', [0., .7, -2.])
def test_gaussian_region_matches_closed_form(y):
    # Closed-form roots of the official fabzCI for a N(0, sigma^2) prior: (y + q) / 3 = theta.
    up = lambda t: (y + ndtri(min(.9 + stats.norm.cdf(y - t), 1))) / 3 - t
    lo = lambda t: (y + ndtri(max(.1 - stats.norm.cdf(t - y), 0))) / 3 - t
    np.testing.assert_allclose(fab_region(y, .1, 'gaussian'), (brentq(lo, -50, 50), brentq(up, -50, 50)), atol=1e-8)


def test_spending_symmetry_and_horseshoe_robustness():
    assert spending(0., .1, 'horseshoe') == pytest.approx(.5)
    assert spending(1.3, .1, 'horseshoe') == pytest.approx(1 - spending(-1.3, .1, 'horseshoe'))
    low, high = fab_region(0., .1, 'horseshoe')
    assert high - low < 2 * ndtri(.95)                      # shorter at zero
    low, high = fab_region(40., .1, 'horseshoe')
    assert high - low == pytest.approx(2 * ndtri(.95), rel=.01)   # reverts to the z-interval
    low, high = fab_region(40., .1, 'gaussian')
    assert high - low > 2 * 2 * ndtri(.95)                   # Gaussian prior does not (width ~ 2|y|/3)
    for y in (0., 1.5, -4.):
        low, high = fab_region(y, .1, 'horseshoe')
        assert low <= posterior_mean(y, 'horseshoe') <= high


def test_fab_region_has_exact_coverage():
    rng = np.random.default_rng(0)
    for beta in (0., 2.):
        draws = beta + rng.standard_normal(300)
        covered = np.mean([low <= beta <= high for low, high in (fab_region(w, .1, 'horseshoe') for w in draws)])
        assert .85 < covered < .95


def test_method_estimate_and_limits():
    d = DGP(Config()).draw(3000, np.random.default_rng(1))
    data = PPIData(d['y'][:300], d['oracle'][:300], d['oracle'][300:])
    result = FABPPI()(data, target=Mean(), alpha=.1)
    diag = result.diagnostics
    imputed = data.unlabeled_predictions.mean()
    assert diag['rectifier'] == pytest.approx((data.labels - data.labeled_predictions).mean())
    assert result.estimate == pytest.approx(imputed + diag['rectifier_posterior_mean'])
    assert abs(diag['rectifier_posterior_mean']) < abs(diag['rectifier'])
    low, high = result.confidence_interval
    assert low <= result.estimate <= high and 'values' not in diag
    # With delta = alpha and no imputed term, the interval is mean(f~) + sigma * FAB region.
    narrow = FABPPI(imputed_uncertainty=False)(data, target=Mean(), alpha=.1)
    sigma = result.standard_error ** 2 - data.unlabeled_predictions.var(ddof=1) / len(data.unlabeled_predictions)
    region = np.array(fab_region(diag['rectifier'] / np.sqrt(sigma), .1, 'horseshoe'))
    np.testing.assert_allclose(narrow.confidence_interval, imputed + np.sqrt(sigma) * region)
    # Fixed delta, tuned lambda, and argument checks.
    assert np.diff(FABPPI(delta=.05)(data, target=Mean(), alpha=.1).confidence_interval) >= high - low - 1e-9
    tuned = FABPPI(lam=None)(data, target=Mean(), alpha=.1)
    assert tuned.method == 'FAB-PPI++'
    assert tuned.diagnostics['prediction_weight'] == pytest.approx(
        PPIPlusPlus()(data, target=Mean()).diagnostics['prediction_weight'])
    with pytest.raises(ValueError):
        FABPPI(prior='laplace')
    with pytest.raises(ValueError):
        FABPPI(delta=.2)(data, target=Mean(), alpha=.1)


def test_fab_ppi_interval_coverage():
    config = Config(n=800, reps=200)
    dgp = DGP(config)
    truth, _ = dgp.truth()
    covered = 0
    for rep in range(config.reps):
        data = dgp.draw(config.n, rng_for(config.seed, rep, 1))
        noise = rng_for(config.seed, rep, 2).normal(0, .1, config.n)
        f = np.clip(data['oracle'] + noise, 0, 1)
        record = PPIData(data['y'][:200], f[:200], f[200:])
        low, high = FABPPI(lam=None, delta=config.alpha / 2)(record, target=Mean(), alpha=config.alpha).confidence_interval
        covered += low <= truth <= high
    assert .90 < covered / config.reps <= 1.0, f'coverage={covered / config.reps:.3f}'
