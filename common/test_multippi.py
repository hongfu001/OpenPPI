"""MultiPPI: Ledoit-Wolf, allocation optimality, PPI++ special case, unbiasedness, coverage."""
import itertools

import numpy as np
import pytest
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf as sklearn_ledoit_wolf

from prediction_inference import MultiPPI, MultiPPIData, PPIPlusPlus, Mean, PPIData, multi_allocate
from prediction_inference._stats import ledoit_wolf, multi_weights
from dataset.judge_sim import make_multi_rater_data

FULL = (0, 1, 2, 3)
MODELS = [s for r in (1, 2, 3) for s in itertools.combinations((1, 2, 3), r)]
PRICE = {1: .05, 2: .3, 3: 1.}     # rater 3 is the best and the most expensive
COSTS = {FULL: sum(PRICE.values()), **{s: sum(PRICE[i] for i in s) for s in MODELS}}
GIVEN = {**COSTS, FULL: 0.}     # Appendix C.3: N labeled rows are given, the budget buys rater queries


def test_ledoit_wolf_matches_sklearn():
    x = np.random.default_rng(0).normal(size=(60, 4)) @ np.random.default_rng(1).normal(size=(4, 4))
    np.testing.assert_allclose(ledoit_wolf(x), sklearn_ledoit_wolf(x)[0], atol=1e-12)


def test_two_sources_match_brute_force():
    sigma = np.array([[1, .8], [.8, 1]])
    allocation = multi_allocate(sigma, {(0, 1): 1., (1,): .1}, 100.)

    def variance(n12):           # Var(X1 - lam X2)/n12 + lam^2 Var(X2)/n2 at the optimal lam
        n2 = (100 - n12) / .1
        lam = .8 * n2 / (n12 + n2)
        return (1 - 2 * lam * .8 + lam ** 2) / n12 + lam ** 2 / n2

    grid = np.linspace(1, 99.9, 50001)
    assert allocation.fractional[(0, 1)] == pytest.approx(grid[np.argmin([variance(g) for g in grid])], abs=.01)
    assert allocation.counts == {(0, 1): 76, (1,): 230}


def test_allocation_matches_dual_socp():
    # Appendix H: V = U^2 / B with U = sup a^T y s.t. y_I^T Sigma_I^-1 y_I <= c_I for each I.
    rng = np.random.default_rng(3)
    root = rng.normal(size=(4, 4))
    sigma = root @ root.T + np.eye(4)
    allocation = multi_allocate(sigma, COSTS, 500.)
    a = np.eye(4)[0]
    continuous = multi_weights(sigma, {s: v for s, v in allocation.fractional.items() if v > 1e-6}, a)[1]
    constraints = [{'type': 'ineq', 'fun': lambda y, s=s: COSTS[s] - y[list(s)] @ np.linalg.inv(
        sigma[np.ix_(s, s)]) @ y[list(s)]} for s in COSTS]
    dual = minimize(lambda y: -y[0], np.zeros(4), constraints=constraints, method='SLSQP', options={'ftol': 1e-14})
    assert continuous == pytest.approx(dual.x[0] ** 2 / 500., rel=1e-6)


def test_weights_are_unbiased_and_optimal():
    sigma = np.cov(make_multi_rater_data(5000, seed=1).T)
    counts = {FULL: 100, (3,): 50, (1, 2): 400}
    weights, variance = multi_weights(sigma, counts, np.eye(4)[0])
    total = np.zeros(4)
    for s, w in weights.items():
        total[list(s)] += w
    np.testing.assert_allclose(total, np.eye(4)[0], atol=1e-12)       # sum_I P_I^T lambda_I = a
    assert variance == pytest.approx(sum(w @ sigma[np.ix_(s, s)] @ w / counts[s] for s, w in weights.items()))
    shifted = dict(weights)
    shifted[FULL] = weights[FULL] + np.array([0, 0, 0, .05])       # stay unbiased by moving weight between
    shifted[(3,)] = weights[(3,)] - .05                               # subsets: the variance can only grow
    assert sum(w @ sigma[np.ix_(s, s)] @ w / counts[s] for s, w in shifted.items()) > variance


def test_label_and_one_rater_is_ppi_plus_plus():
    x = make_multi_rater_data(1500, seed=2)
    labeled, unlabeled = x[:300, [0, 3]], x[300:, 3]
    pooled = np.r_[labeled[:, 1], unlabeled]
    sigma = np.array([[1., np.cov(labeled.T)[0, 1]], [np.cov(labeled.T)[0, 1], pooled.var(ddof=1)]])
    data = MultiPPIData({(0, 1): labeled, (1,): unlabeled})
    multi = MultiPPI(covariance=sigma)(data, target=Mean())
    ppi = PPIPlusPlus()(PPIData(labeled[:, 0], labeled[:, 1], unlabeled), target=Mean())
    assert multi.estimate == pytest.approx(ppi.estimate)
    assert multi.diagnostics['weights'][(1,)][0] == pytest.approx(ppi.diagnostics['prediction_weight'])


def run(rep, budget=300., labeled=200):
    rng = np.random.default_rng(rep)
    full = make_multi_rater_data(labeled, seed=rng.integers(2 ** 32))
    allocation = multi_allocate(ledoit_wolf(full), GIVEN, budget, caps={FULL: labeled})
    samples = {FULL: full}
    for s, n in allocation.counts.items():
        if s != FULL:
            samples[s] = make_multi_rater_data(n, seed=rng.integers(2 ** 32))[:, list(s)]
    return MultiPPI()(MultiPPIData(samples), target=Mean()), full


def test_multippi_coverage_and_gain_over_labels():
    covered, ratio = 0, []
    for rep in range(200):
        result, full = run(rep)
        low, high = result.confidence_interval
        covered += low <= .5 <= high
        ratio.append(result.standard_error / (full[:, 0].std(ddof=1) / np.sqrt(len(full))))
    assert .90 < covered / 200 <= 1.0, f'coverage={covered / 200:.3f}'
    assert np.mean(ratio) < .9
