"""Label-shift Algorithm 2: independent matrix formulas, EIF SE and shifted coverage."""
import numpy as np
import pytest

from openppi import EfficientLabelShift, LabelShiftData, Mean, PPIData, TwoSample, get_method
from dataset.label_shift_sim import make_label_shift_case


def manual_discrete(data, refine=False):
    """Literal Section 5 class-indicator solves, independent of implementation."""
    y, p = data.labels, data.conditional_probabilities
    classes = np.unique(y)
    n, m = len(y), len(p) - len(y)
    frequency = np.array([(y == k).mean() for k in classes])
    index = np.searchsorted(classes, y)

    def apply(rho, function):
        w = 1 / (p @ (rho ** 2 + n / m * rho))
        weighted = w[:, None] * p * rho
        a = np.array([weighted[:n][y == k].mean(axis=0) for k in classes])
        b = weighted @ np.linalg.solve(a, function)
        values = function[index]
        return b[n:].mean(axis=0) + (rho[index] * (values - b[:n])).mean(axis=0), b

    rho = np.ones(len(classes))
    q = np.array([apply(rho, np.eye(len(classes))[:, j])[0] for j in range(len(classes))])
    rho = q / frequency
    initial = rho.copy()
    if refine:
        q = np.array([apply(rho, np.eye(len(classes))[:, j])[0] for j in range(len(classes))])
        rho = q / frequency
    point, b = apply(rho, classes)
    residual = rho[index] * (y - b[:n])
    se = np.sqrt(residual.var(ddof=1) / n + b[n:].var(ddof=1) / m)
    return point, se, initial, rho


@pytest.mark.parametrize('refine', [False, True])
def test_discrete_matches_class_indicator_algorithm_and_eif_variance(refine):
    data, _, _ = make_label_shift_case(seed=5)
    expected = manual_discrete(data, refine)
    result = EfficientLabelShift(refine_ratio=refine)(data, target=Mean())
    assert result.estimate == pytest.approx(expected[0], abs=1e-10)
    assert result.standard_error == pytest.approx(expected[1], abs=1e-10)
    np.testing.assert_allclose(result.diagnostics['initial_density_ratio'], expected[2])
    np.testing.assert_allclose(result.diagnostics['density_ratio'], expected[3])
    assert result.diagnostics['fredholm_residual'] < 1e-10
    assert result.diagnostics['estimated_target_probabilities'].sum() == pytest.approx(1)


def test_multiclass_nonconsecutive_labels_and_permutation():
    rng = np.random.default_rng(6)
    support = np.array([-2., 1., 5.])
    y = np.repeat(support, [90, 100, 110])
    # Three informative feature categories with overlapping class distributions.
    confusion = np.array([[.8, .15, .05], [.1, .8, .1], [.1, .15, .75]])
    source_class = np.searchsorted(support, y)
    source_feature = np.array([rng.choice(3, p=confusion[k]) for k in source_class])
    target_class = rng.choice(3, size=1200, p=[.15, .25, .6])
    target_feature = np.array([rng.choice(3, p=confusion[k]) for k in target_class])
    prior = np.array([.3, 1 / 3, 11 / 30])
    posterior = confusion.T * prior
    posterior /= posterior.sum(axis=1, keepdims=True)
    probs = posterior[np.r_[source_feature, target_feature]]
    data = LabelShiftData(y, source_feature[:, None], target_feature[:, None], probs)
    result = EfficientLabelShift()(data, target=Mean(outcome_type='continuous'))
    expected = manual_discrete(data)
    assert result.estimate == pytest.approx(expected[0])
    assert abs(result.estimate - support @ [.15, .25, .6]) < 4 * result.standard_error
    order = rng.permutation(len(y))
    shuffled = LabelShiftData(y[order], data.labeled_features[order], data.unlabeled_features,
                             np.vstack([probs[:len(y)][order], probs[len(y):]]))
    assert EfficientLabelShift()(shuffled, target=Mean()).estimate == pytest.approx(result.estimate)


def test_default_conditional_kernel_and_frozen_snapshots():
    data, _, truth = make_label_shift_case(n=500, seed=10)
    fitted = LabelShiftData(data.labels, data.labeled_features, data.unlabeled_features)
    r = EfficientLabelShift()(fitted, target=Mean())
    assert r.diagnostics['conditional_model'] == 'nadaraya_watson'
    assert abs(r.estimate - truth) < 4 * r.standard_error
    assert not fitted.labels.flags.writeable
    assert not fitted.labeled_features.flags.writeable
    assert not data.conditional_probabilities.flags.writeable
    assert get_method('LabelShift').data_type is LabelShiftData


def test_continuous_full_kernel_algorithm_against_literal_matrix_calculation():
    y = np.linspace(-1, 1, 12)
    target_x = np.linspace(-.8, 1, 24)
    x = np.r_[y, target_x]
    weights = np.exp(-((x[:, None] - y[None, :]) / .5) ** 2 / 2)
    weights /= weights.sum(axis=1, keepdims=True)
    data = LabelShiftData(y, y[:, None], target_x[:, None], weights)
    h, l, cutoff = .3, .2, 1e-3
    s = np.exp(-((y[:, None] - y[None, :]) / l) ** 2 / 2)
    s /= s.sum(axis=1, keepdims=True)
    kernel = np.exp(-((y[:, None] - y[None, :]) / h) ** 2 / 2)
    n, m = len(y), len(target_x)

    def operators(rho):
        b = weights * rho[None, :] / (weights @ (rho ** 2 + n / m * rho))[:, None]
        return b, np.linalg.pinv(s @ b[:n], rcond=cutoff)

    b, inverse = operators(np.ones(n))
    c = np.ones(n) / n + (b[n:].mean(axis=0) - b[:n].mean(axis=0)) @ inverse
    rho = np.clip((c @ kernel) / kernel.mean(axis=0), .01, 100)
    rho /= rho.mean()
    b, inverse = operators(rho)
    predictions = b @ inverse @ y
    expected = predictions[n:].mean() + np.mean(rho * (y - predictions[:n]))
    r = EfficientLabelShift(outcome='continuous', density_bandwidth=h, inverse_bandwidth=l,
                           rcond=cutoff, ratio_bounds=(.01, 100))(data, target=Mean())
    assert r.estimate == pytest.approx(expected)
    np.testing.assert_allclose(r.diagnostics['density_ratio'], rho, rtol=1e-9)
    assert r.diagnostics['operator_solves'][-1]['rank'] < n


def test_continuous_known_conditional_distribution_exact_recovery():
    y = np.array([-1., 0., 1., 2.])
    target = np.repeat(y, [2, 4, 6, 8])
    probabilities = np.vstack([np.eye(4), np.eye(4)[np.searchsorted(y, target)]])
    data = LabelShiftData(y, y[:, None], target[:, None], probabilities)
    r = EfficientLabelShift(outcome='continuous', density_bandwidth=.01, inverse_bandwidth=.01,
                           rcond=1e-10)(data, target=Mean())
    assert r.estimate == pytest.approx(target.mean())
    assert r.standard_error == pytest.approx(target.std(ddof=1) / np.sqrt(len(target)))
    assert r.diagnostics['fredholm_residual'] < 1e-10


@pytest.mark.parametrize('kwargs', [{'outcome': 'auto'}, {'rcond': 0}, {'rcond': 1},
                                    {'conditional_bandwidth': -1}, {'density_bandwidth': float('nan')},
                                    {'inverse_bandwidth': 0}, {'ratio_bounds': (0, 10)},
                                    {'initial_ratio': [1, 1]}, {'refine_ratio': 'yes'}])
def test_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        EfficientLabelShift(**kwargs)


def test_identification_and_density_failures_are_explicit():
    data, _, _ = make_label_shift_case(seed=8)
    uninformative = np.full_like(data.conditional_probabilities, .5)
    with pytest.raises(ValueError, match='not identifiable'):
        EfficientLabelShift()(LabelShiftData(data.labels, data.labeled_features,
                                             data.unlabeled_features, uninformative), target=Mean())
    with pytest.raises(ValueError, match='initial_ratio'):
        EfficientLabelShift(initial_ratio=lambda y: -np.ones(len(y)))(data, target=Mean())
    wrong_columns = np.full((1800, 3), 1 / 3)
    with pytest.raises(ValueError, match='columns'):
        EfficientLabelShift()(LabelShiftData(data.labels, data.labeled_features,
                                             data.unlabeled_features, wrong_columns), target=Mean())
    with pytest.raises(TypeError, match='LabelShiftData'):
        EfficientLabelShift()(PPIData([0, 1], [0, 1], [0, 1]), target=Mean())


def test_container_rejects_invalid_probabilities_and_feature_shapes():
    with pytest.raises(ValueError, match='align'):
        LabelShiftData([0, 1], np.ones((3, 1)), np.ones((2, 1)))
    with pytest.raises(ValueError, match='sum to one'):
        LabelShiftData([0, 1], np.ones((2, 1)), np.ones((2, 1)), np.ones((4, 2)))
    with pytest.raises(ValueError, match='columns'):
        LabelShiftData([0, 1], np.ones((2, 1)), np.ones((2, 2)))


@pytest.mark.slow
def test_shifted_vs_unshifted_coverage_and_bias():
    for target in (.3, .7):
        estimates, coverage, ppi_estimates, ppi_coverage = [], [], [], []
        for seed in range(250):
            data, standard, truth = make_label_shift_case(seed=100 + seed, target_prevalence=target)
            result = EfficientLabelShift()(data, target=Mean())
            ppi = TwoSample()(standard, target=Mean())
            estimates.append(result.estimate)
            ppi_estimates.append(ppi.estimate)
            coverage.append(result.confidence_interval[0] <= truth <= result.confidence_interval[1])
            ppi_coverage.append(ppi.confidence_interval[0] <= truth <= ppi.confidence_interval[1])
        assert abs(np.mean(estimates) - target) < .01
        assert .9 < np.mean(coverage) <= 1
        if target == .7:
            assert abs(np.mean(ppi_estimates) - target) > .1
            assert np.mean(ppi_coverage) < .1
        else:
            assert abs(np.mean(ppi_estimates) - target) < .01
            assert np.mean(ppi_coverage) > .9


def test_simulation_retains_failures_and_uses_target_truth():
    from label_shift import run_simulation
    results, summary = run_simulation(reps=3, n=2, m=20, case='label_shift', seed=9)
    assert len(results) == 15
    assert results['truth'].eq(.7).all()
    shifted = summary[summary['method'].eq('LabelShift')].iloc[0]
    assert shifted['successful_reps'] == 0 and shifted['failure_rate'] == 1
    assert shifted['coverage_all'] == 0 and np.isnan(shifted['coverage'])
    failed = results[results['method'].eq('LabelShift')]
    assert failed['status'].eq('failed').all() and failed['message'].str.len().gt(0).all()
