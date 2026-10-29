"""Robust active sampling: paths, minimax objective, labeling flow, coverage."""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from openppi import (
    ASI, RASI, BurnIn, Mean, RobustActiveData, RobustDesign, Uncertainty, active_rule,
    budget_path, robust_active_label, robust_rho, select_burn_in, select_constraint,
    select_for_labeling,
)
from openppi.sampling.rasi import robust_objective
from config import Config
from utils import rng_for
from dataset.judge_sim import DGP


def reference_objective(e_hat, pi, nb, n, constraint_sum, r):
    """MinMaxOptimizer_l2.objective_function from the reference code."""
    ratio = nb / n
    denominator = pi ** (1 - r) * ratio ** r / np.sum(pi ** (1 - r) * ratio ** r) * nb
    denominator = np.clip(denominator, 0, 1)
    return (np.sum(e_hat ** 2 / denominator)
            + constraint_sum * np.sqrt(np.sum(1 / denominator ** 2))) / n


@pytest.mark.parametrize('path', ['linear', 'geometric', 'hellinger'])
def test_paths_connect_active_and_uniform_within_budget(path):
    base = active_rule(np.linspace(.05, 1, 50), .2)
    np.testing.assert_allclose(budget_path(base, .2, 0., path), base)
    np.testing.assert_allclose(budget_path(base, .2, 1., path), .2)
    for t in (.25, .5, .75):
        pi = budget_path(base, .2, t, path)
        assert pi.mean() <= .2 + 1e-12 and (pi > 0).all() and (pi <= 1).all()


def test_objective_matches_reference_geometric_minimax():
    rng = np.random.default_rng(0)
    errors = rng.uniform(.01, 1, 300)
    base = active_rule(errors, .2)
    n, c = len(errors), .05
    for r in (0., .3, .7, 1.):
        ours = robust_objective(errors, budget_path(base, .2, r, 'geometric'), c)
        assert ours == pytest.approx(reference_objective(errors, base, .2 * n, n, c * np.sqrt(n), r))


def test_rho_is_active_without_constraint_and_uniform_with_large_one():
    errors = np.random.default_rng(1).uniform(.05, 1, 400)
    base = active_rule(errors, .2)
    # Appendix A.4: pi proportional to the error estimate is optimal when c = 0.
    assert robust_rho(errors, base, .2, constraint=0.) == 0.
    path = [robust_rho(errors, base, .2, constraint=c) for c in (0., .05, .5, 50.)]
    assert path == sorted(path) and path[-1] == 1.


def test_select_constraint_returns_a_candidate():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 2))
    p = np.clip(.5 + .2 * x[:, 0], .01, .99)
    y = rng.binomial(1, p).astype(float)
    c = select_constraint(x, y, p, error_model_factory=lambda: DecisionTreeRegressor(max_depth=2),
                          rho=.2, candidates=(0., .05, .5), random_state=0)
    assert c in (0., .05, .5)


class Fixed:
    def predict(self, x):
        return x[:, 0]


def test_flow_queries_only_selected_rows_and_reuses_asi_formula():
    rng = np.random.default_rng(3)
    x = np.column_stack([rng.uniform(.05, .95, 500), rng.normal(size=500)])
    y = rng.binomial(1, x[:, 0]).astype(float)
    calls = []

    def oracle(i):
        calls.append(i)
        return y[i]

    record = robust_active_label(
        x, oracle=oracle, target=Mean(), budget=120, model=Fixed(),
        error_model_factory=lambda: DecisionTreeRegressor(max_depth=2),
        burn_in=40, constraint=.05, random_state=4)
    assert isinstance(record, RobustActiveData)
    assert sorted(calls) == np.flatnonzero(record.queried).tolist() and len(set(calls)) == len(calls)
    assert record.burn_in.sum() == 40 and record.queried[record.burn_in].all()
    np.testing.assert_array_equal(record.propensities[record.burn_in], 1.)
    assert record.propensities.sum() <= 120 + 1e-9
    result = RASI()(record, target=Mean())
    assert result.estimate == ASI()(record, target=Mean()).estimate
    assert result.diagnostics['burn_in'] == 40 and result.diagnostics['constraint'] == .05
    # Burn-in values are the labels themselves, as in the reference concatenation.
    with pytest.raises(TypeError):
        RASI()(ASI.data_type(record.predictions, record.observed_labels,
                             record.queried, record.propensities), target=Mean())


class Errors:
    """Error model stub predicting |x0 - .5|, so design.errors is known exactly."""
    def predict(self, x):
        return np.abs(x[:, 0] - .5)


@pytest.mark.parametrize('path', ['linear', 'geometric', 'hellinger'])
def test_robust_design_reproduces_budget_path(path):
    x = np.column_stack([np.linspace(.01, .99, 300), np.zeros(300)])
    design = RobustDesign(Errors(), constraint=.05, path=path)
    scores = design.errors(x)
    request = select_for_labeling(np.full(300, .5), target=Mean(), design=design, budget=60,
                                  scores=scores, random_state=0)
    expected_t = robust_rho(scores, active_rule(scores, .2), .2, constraint=.05, path=path)
    assert request.design.rho_robust == expected_t
    np.testing.assert_allclose(request.propensities,
                               budget_path(active_rule(scores, .2), .2, expected_t, path))
    assert not hasattr(design, 'rho')  # the caller's design is copied, not calibrated


def test_scores_override_leaves_default_selection_unchanged():
    p = np.linspace(.05, .95, 80)
    default = select_for_labeling(p, target=Mean(), design=Uncertainty(), budget=20, random_state=5)
    explicit = select_for_labeling(p, target=Mean(), design=Uncertainty(), budget=20, random_state=5,
                                   scores=Mean().sampling_scores(np.empty((80, 0)), p))
    np.testing.assert_array_equal(default.queried, explicit.queried)
    np.testing.assert_array_equal(default.propensities, explicit.propensities)
    with pytest.raises(ValueError):
        select_for_labeling(p, target=Mean(), design=Uncertainty(), budget=20, scores=-np.ones(80))


def test_two_batch_flow_matches_convenience_wrapper():
    rng = np.random.default_rng(6)
    x = np.column_stack([rng.uniform(.05, .95, 400), rng.normal(size=400)])
    y = rng.binomial(1, x[:, 0]).astype(float)
    factory = lambda: DecisionTreeRegressor(max_depth=2, random_state=0)
    wrapped = robust_active_label(x, oracle=lambda i: y[i], target=Mean(), budget=100,
                                  model=Fixed(), error_model_factory=factory, burn_in=30,
                                  constraint=.02, random_state=7)
    # Same steps by hand, sharing one generator as the wrapper does.
    stream = np.random.default_rng(7)
    p = Fixed().predict(x)
    burn = select_burn_in(400, size=30, random_state=stream)
    assert isinstance(burn, BurnIn) and len(burn.remaining) == 370
    design = RobustDesign.fit(x[burn.indices], y[burn.indices], p[burn.indices],
                              error_model_factory=factory, rho=70 / 370, constraint=.02)
    rest = burn.remaining
    request = select_for_labeling(p[rest], target=Mean(), design=design, budget=70,
                                  scores=design.errors(x[rest]), random_state=stream)
    record = burn.combine(p, y[burn.indices], request, y[rest[request.indices]])
    for name in ('queried', 'propensities', 'burn_in'):
        np.testing.assert_array_equal(getattr(record, name), getattr(wrapped, name))
    np.testing.assert_array_equal(record.observed_labels[record.queried], y[record.queried])
    assert record.rho_robust == wrapped.rho_robust and record.constraint == .02
    with pytest.raises(TypeError):  # a non-robust second batch cannot be combined
        burn.combine(p, y[burn.indices], select_for_labeling(
            p[rest], target=Mean(), design=Uncertainty(), budget=70, random_state=0), [])
    with pytest.raises(ValueError):
        burn.combine(p, y[burn.indices][:-1], request, y[rest[request.indices]])


@pytest.mark.slow
def test_rasi_interval_coverage():
    config = Config(n=800, reps=150)
    dgp = DGP(config)
    truth, _ = dgp.truth()
    covered = 0
    for rep in range(config.reps):
        data = dgp.draw(config.n, rng_for(config.seed, rep, 1))
        model = type('Oracle', (), {'predict': lambda self, x: x[:, -1]})()
        features = np.column_stack([data['features'], data['oracle']])
        record = robust_active_label(
            features, oracle=lambda i: data['y'][i], target=Mean(), budget=160, model=model,
            error_model_factory=lambda: DecisionTreeRegressor(max_depth=3, min_samples_leaf=5),
            burn_in=60, constraint=.05, random_state=rng_for(config.seed, rep, 2))
        low, high = RASI()(record, target=Mean(), alpha=config.alpha).confidence_interval
        covered += low <= truth <= high
    coverage = covered / config.reps
    assert .88 < coverage <= 1.0, f'coverage={coverage:.3f}'
