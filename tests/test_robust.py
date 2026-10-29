"""Robust (RASI) sampling: paths, minimax objective, two-batch flow, coverage."""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from openppi import IPW, Mean, RobustDesign, Uncertainty, select_for_labeling
from openppi.sampling.robust import active_rule, budget_path, robust_objective, robust_rho, select_burn_in
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


class Fixed:
    def predict(self, x):
        return x[:, 0]


def two_batches(x, oracle, predictions, *, budget, burn_in, error_model_factory, constraint, random_state):
    """Burn-in batch, RobustDesign fit, robust batch, combine: the RASI flow of rasi.ipynb."""
    rng = np.random.default_rng(random_state)
    burn = select_burn_in(len(x), size=burn_in, random_state=rng)
    labels = np.array([oracle(int(i)) for i in burn.indices], dtype=float)
    rest = burn.remaining
    design = RobustDesign.fit(x[burn.indices], labels, predictions[burn.indices],
                              error_model_factory=error_model_factory, rho=(budget - burn_in) / len(rest),
                              constraint=constraint)
    request = select_for_labeling(predictions[rest], target=Mean(), design=design, budget=budget - burn_in,
                                  scores=design.errors(x[rest]), random_state=rng)
    second = np.array([oracle(int(i)) for i in rest[request.indices]], dtype=float)
    return burn, burn.combine(predictions, labels, request, second)


def test_flow_queries_only_selected_rows_and_records_the_design():
    rng = np.random.default_rng(3)
    x = np.column_stack([rng.uniform(.05, .95, 500), rng.normal(size=500)])
    y = rng.binomial(1, x[:, 0]).astype(float)
    p = Fixed().predict(x)
    calls = []

    def oracle(i):
        calls.append(i)
        return y[i]

    burn, record = two_batches(x, oracle, p, budget=120, burn_in=40,
                               error_model_factory=lambda: DecisionTreeRegressor(max_depth=2),
                               constraint=.05, random_state=4)
    assert sorted(calls) == np.flatnonzero(record.queried).tolist() and len(set(calls)) == len(calls)
    assert record.queried[burn.indices].all()
    np.testing.assert_array_equal(record.propensities[burn.indices], 1.)
    assert record.propensities.sum() <= 120 + 1e-9
    result = IPW()(record, target=Mean())
    assert result.diagnostics['burn_in'] == 40 and result.diagnostics['constraint'] == .05
    assert result.diagnostics['variance_design'] == 'robust_active_terminal_plugin'
    # Burn-in rows have propensity 1, so their IPW value is the label: the reference concatenation.
    corrected = p + record.queried / record.propensities * (np.nan_to_num(record.observed_labels) - p)
    assert result.estimate == pytest.approx(corrected.mean())
    np.testing.assert_array_equal(corrected[burn.indices], y[burn.indices])
    with pytest.raises(TypeError):  # a non-robust second batch cannot be combined
        burn.combine(p, y[burn.indices], select_for_labeling(
            p[burn.remaining], target=Mean(), design=Uncertainty(), budget=70, random_state=0), [])


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
        _, record = two_batches(
            features, lambda i: data['y'][i], model.predict(features), budget=160, burn_in=60,
            error_model_factory=lambda: DecisionTreeRegressor(max_depth=3, min_samples_leaf=5),
            constraint=.05, random_state=rng_for(config.seed, rep, 2))
        low, high = IPW()(record, target=Mean(), alpha=config.alpha).confidence_interval
        covered += low <= truth <= high
    coverage = covered / config.reps
    assert .88 < coverage <= 1.0, f'coverage={coverage:.3f}'
