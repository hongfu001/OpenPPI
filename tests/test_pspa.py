"""PSPA: official R fixture, clipping, variance minimization and coverage."""
import numpy as np
import pytest

from openppi import Mean, PPIData, PSPA, TwoSample, get_method
from openppi.common.stats import wald_interval


def test_official_mean_formula_and_interval_values():
    data = PPIData([1, 0, 1, 1], [.8, .4, .6, .2], [.1, .2, .3, .4, .5])
    captured = {}

    def interval(point, se, alpha, *, values):
        captured.update(values)
        return wald_interval(point, se, alpha)

    result = PSPA(interval=interval)(data, target=Mean(), alpha=.1)
    y, f, g = data.labels, data.labeled_predictions, data.unlabeled_predictions
    weight = np.cov(y, f)[0, 1] / (f.var(ddof=1) + len(y) / len(g) * g.var(ddof=1))
    sigma = y.var(ddof=1) + weight**2 * (f.var(ddof=1) + 4 / 5 * g.var(ddof=1)) - 2 * weight * np.cov(y, f)[0, 1]
    assert result.estimate == pytest.approx(y.mean() + weight * (g.mean() - f.mean()))
    assert result.standard_error == pytest.approx(np.sqrt(sigma / 4))
    assert result.diagnostics['prediction_weight'] == pytest.approx(weight)
    assert result.confidence_interval == pytest.approx(wald_interval(result.estimate, result.standard_error, .1))
    np.testing.assert_allclose(captured['residuals'], y - weight * f)
    np.testing.assert_allclose(captured['unlabeled_predictions'], weight * g)
    assert result.method == 'PSPA' and 'values' not in result.diagnostics
    assert get_method('PSPA').inference(data, target=Mean()).estimate == result.estimate
    # Fixture from official pspa_y(method='mean'), commit 8e77b124d58b884730484d1923d909a61ed70025.
    assert result.estimate == pytest.approx(0.67307692307692313)
    assert result.standard_error == pytest.approx(0.24350538350285358)


@pytest.mark.parametrize('predictions, expected', [([0, 1, 0, 0], 0.), ([.6, .4, .6, .6], 1.), ([.5]*4, 0.)])
def test_weight_clipping_and_constant_predictions(predictions, expected):
    data = PPIData([1, 0, 1, 1], predictions, [.5]*6)
    result = PSPA()(data, target=Mean())
    assert result.diagnostics['prediction_weight'] == expected
    if expected == 0:
        assert result.estimate == pytest.approx(data.labels.mean())
        assert result.standard_error == pytest.approx(data.labels.std(ddof=1) / 2)


def test_minimizes_sample_variance_on_unit_interval_and_differs_from_ppi_plus():
    data = PPIData([1, 0, 1, 1], [.8, .4, .6, .2], [.1, .2, .3, .4, .5])
    result = PSPA()(data, target=Mean())
    for weight in np.linspace(0, 1, 21):
        assert result.standard_error <= TwoSample(lam=float(weight), variance='unlabeled')(data, target=Mean()).standard_error + 1e-14
    assert result.diagnostics['prediction_weight'] != pytest.approx(TwoSample(lam='tuned')(data, target=Mean()).diagnostics['prediction_weight'])


@pytest.mark.slow
def test_coverage_with_accurate_and_misspecified_predictions():
    rng = np.random.default_rng(421)
    covered = np.zeros(3)
    for _ in range(250):
        p = rng.uniform(.05, .95, 1800)
        y = rng.binomial(1, p[:300])
        for j, f in enumerate((p, 1 - p, np.full(len(p), .8))):
            result = PSPA()(PPIData(y, f[:300], f[300:]), target=Mean())
            low, high = result.confidence_interval
            covered[j] += low <= .5 <= high
    assert np.all((covered / 250 > .90) & (covered / 250 < .99)), covered
