"""EPI: primal optimization, exact binary calibration, finite-m LR and cross-fitting."""
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import chi2

from openppi import EPI, Mean, LinearRegression, PPIData, RegressionData, get_method
from openppi.inference.empirical_likelihood import empirical_weights


def sample(seed=12, n=120, m=480):
    rng = np.random.default_rng(seed)
    f, fu = rng.normal(size=n), rng.normal(size=m)
    return PPIData(2 * f + rng.normal(size=n), f, fu)


def test_weights_agree_with_independent_primal_optimizer():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(30, 2)) + .1
    weights, loss, rank = empirical_weights(a)
    primal = minimize(lambda w: -np.log(w).sum(), np.full(30, 1 / 30),
                      jac=lambda w: -1 / w, bounds=[(1e-10, 1)] * 30,
                      constraints={'type': 'eq', 'fun': lambda w: np.r_[w.sum() - 1, w @ a],
                                   'jac': lambda w: np.vstack([np.ones(30), a.T])},
                      method='SLSQP', options={'ftol': 1e-10, 'maxiter': 1000})
    assert primal.success
    assert -np.log(weights).sum() <= primal.fun + 1e-8
    np.testing.assert_allclose(weights, primal.x, atol=3e-6)
    np.testing.assert_allclose(weights @ a, 0, atol=1e-9)
    assert loss == pytest.approx(-np.log(30 * weights).sum())
    assert rank == 2 and (weights > 0).all()
    redundant, _, r = empirical_weights(np.column_stack([a, a[:, 0], np.zeros(30)]))
    np.testing.assert_allclose(redundant, weights, atol=1e-9)
    assert r == 2


def test_binary_auxiliary_has_exact_poststratified_mean():
    f = np.array([0.] * 30 + [1.] * 30)
    fu = np.array([0.] * 30 + [1.] * 90)
    y = np.r_[np.linspace(-1, 1, 30), np.linspace(2, 4, 30)]
    data = PPIData(y, f, fu)
    result = EPI(interval_method='wald')(data, target=Mean())
    pooled = np.r_[f, fu].mean()
    assert result.estimate == pytest.approx((1 - pooled) * y[:30].mean() + pooled * y[30:].mean())
    np.testing.assert_allclose(result.diagnostics['weights'][:30], (1 - pooled) / 30)
    np.testing.assert_allclose(result.diagnostics['weights'][30:], pooled / 30)


def test_theorem_one_variance_and_corrected_lr_endpoints():
    data = sample()
    r = EPI()(data, target=Mean())
    n, m = len(data.labels), len(data.unlabeled_predictions)
    var = data.labels.var(ddof=1)
    reduction = np.cov(data.labels, data.labeled_predictions)[0, 1] ** 2 / data.labeled_predictions.var(ddof=1)
    expected = var - (1 - n / (n + m)) * reduction
    assert r.standard_error ** 2 == pytest.approx(expected / n)
    scale = expected / (var - reduction)
    assert r.diagnostics['el_ratio_scale'] == pytest.approx(scale)
    assert scale > 1
    h = r.diagnostics['auxiliary_moments']
    _, base, _ = empirical_weights(h)
    for endpoint in r.confidence_interval:
        _, loss, _ = empirical_weights(np.column_stack([data.labels - endpoint, h]))
        assert 2 * (loss - base) == pytest.approx(scale * chi2.ppf(.95, 1), rel=1e-6)
    assert r.confidence_interval[0] < r.estimate < r.confidence_interval[1]


def test_constant_predictions_and_affine_invariance():
    data = sample()
    plain = EPI()(PPIData(data.labels, np.ones(120), np.ones(480)), target=Mean())
    assert plain.estimate == pytest.approx(data.labels.mean())
    assert plain.standard_error == pytest.approx(data.labels.std(ddof=1) / np.sqrt(120))
    assert plain.diagnostics['auxiliary_rank'] == 0
    r = EPI()(data, target=Mean())
    transformed = EPI()(PPIData(7 + 3 * data.labels, -4 + 5 * data.labeled_predictions,
                              -4 + 5 * data.unlabeled_predictions), target=Mean())
    assert transformed.estimate == pytest.approx(7 + 3 * r.estimate)
    np.testing.assert_allclose(transformed.confidence_interval, 7 + 3 * np.array(r.confidence_interval), atol=1e-7)


class RecordingLearner:
    training_sets = []

    def fit(self, features, labels):
        self.training_sets.append(features.copy())
        return self

    def predict(self, features):
        return features[:, 0]


def test_crossfit_excludes_heldout_rows_and_centers_each_fold():
    data = sample()
    RecordingLearner.training_sets = []
    r = EPI(auxiliary='crossfit', folds=3, learner_factory=RecordingLearner)(data, target=Mean())
    from openppi.inference.empirical_likelihood import polynomial_basis
    basis = polynomial_basis(np.r_[data.labeled_predictions, data.unlabeled_predictions][:, None], 1)
    rng = np.random.default_rng(0)
    labeled = rng.permutation(120).reshape(3, -1)
    unlabeled = rng.permutation(480).reshape(3, -1)
    h = np.empty((120, 1))
    for k, (il, iu) in enumerate(zip(labeled, unlabeled)):
        np.testing.assert_allclose(RecordingLearner.training_sets[k], basis[np.setdiff1d(np.arange(120), il)])
        h[il] = basis[il] - np.vstack([basis[il], basis[120 + iu]]).mean(axis=0)
    np.testing.assert_allclose(r.diagnostics['auxiliary_moments'], h)
    w, _, _ = empirical_weights(h)
    assert r.estimate == pytest.approx(w @ data.labels)
    assert get_method('EPI-CF').inference.name == 'EPI-CF'


@pytest.mark.parametrize('auxiliary', ['basis', 'crossfit'])
def test_regression_weighted_normal_equations(auxiliary):
    rng = np.random.default_rng(23)
    x, z = rng.normal(size=(240, 2)), rng.normal(size=(960, 2))
    y = 1 + x @ [2., -1.] + rng.normal(size=240)
    data = RegressionData(y, y + rng.normal(size=240), 1 + z @ [2., -1.], x, z)
    r = EPI(auxiliary=auxiliary, degree=2)(data, target=LinearRegression())
    design = np.column_stack([np.ones(240), x])
    w = r.diagnostics['weights']
    np.testing.assert_allclose(design.T @ (w * (design @ r.estimate - y)), 0, atol=1e-9)
    assert np.linalg.eigvalsh(r.covariance).min() >= 0
    np.testing.assert_allclose(r.standard_error ** 2, r.covariance.diagonal())
    assert r.diagnostics['interval_method'] == 'wald'


@pytest.mark.parametrize('kwargs', [{'degree': 0}, {'degree': True}, {'folds': 1},
                                    {'auxiliary': 'unknown'}, {'interval_method': 'unknown'},
                                    {'learner_factory': RecordingLearner}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        EPI(**kwargs)


def test_infeasibility_and_fold_sizes_are_explicit():
    with pytest.raises(ValueError, match='infeasible'):
        empirical_weights(np.ones((10, 1)))
    with pytest.raises(ValueError, match='infeasible'):
        EPI()(PPIData(np.arange(10.), np.arange(10.), np.arange(10.) + 100), target=Mean())
    with pytest.raises(ValueError, match='divisible'):
        EPI(auxiliary='crossfit', folds=7)(sample(), target=Mean())
    with pytest.raises(ValueError, match='positive residual'):
        EPI()(PPIData(np.arange(10.), np.arange(10.), np.arange(20.)), target=Mean())


@pytest.mark.slow
def test_mean_repeated_sampling_coverage():
    covered = 0
    for seed in range(150):
        result = EPI()(sample(seed, n=100, m=400), target=Mean())
        lo, hi = result.confidence_interval
        covered += lo <= 0 <= hi
    assert .88 <= covered / 150 <= 1
