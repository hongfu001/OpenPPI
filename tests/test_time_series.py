"""PPITSAS mean formulas, official agreement and stationary time-series simulation."""
import json
from pathlib import Path
import numpy as np
import pytest

from openppi import PPITSAS, PPIData, Mean, LinearRegression, TwoSample, get_method
from openppi.inference.time_series import newey_west_var_of_mean
from dataset.time_series_sim import TimeSeriesDGP
from time_series import prepare_records, run_simulation


def test_bartlett_formula_and_mean_correction_by_hand():
    # gamma0=1.25, gamma1=.3125; lag 1 gives (1.25 + .3125)/4.
    assert newey_west_var_of_mean([1.,2.,3.,4.], 1) == pytest.approx(25/64)
    data = PPIData([1.,3.,6.,8.], [1.,2.,3.,4.], [1.,2.,3.,4.])
    result = PPITSAS(max_lag=1)(data, target=Mean(outcome_type='continuous'))
    assert result.estimate == 4.5
    assert result.standard_error == pytest.approx(np.sqrt(77/64))
    assert result.estimate == TwoSample()(data, target=Mean()).estimate
    assert result.diagnostics['variance_design'] == 'time_series_newey_west'
    assert result == get_method('PPITSAS').inference.__class__(max_lag=1)(data, target=Mean())
    zero = PPITSAS(max_lag=0)(data, target=Mean())
    assert zero.standard_error**2 == pytest.approx(.9375)  # ddof=0, not iid PPI's ddof=1


REFERENCE = json.loads(Path(__file__).with_name('fixtures').joinpath('time_series.json').read_text())


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda case: case['name'])
def test_official_reference(case):
    data = PPIData(case['labels'], case['labeled_predictions'], case['unlabeled_predictions'])
    result = PPITSAS(max_lag=case['max_lag'])(data, target=Mean())
    expected = case['expected']
    assert result.estimate == pytest.approx(expected['estimate'], abs=1e-14)
    np.testing.assert_allclose(result.confidence_interval, expected['ci'], rtol=1e-13, atol=1e-14)
    for key in ('unlabeled_mean_variance', 'residual_mean_variance', 'unlabeled_lag', 'residual_lag'):
        assert result.diagnostics[key] == pytest.approx(expected[key], abs=1e-14)


def test_mean_only_and_interval_values():
    data = PPIData([1.,2.,3.], [0.,1.,1.], [1.,2.,4.,5.])
    seen = {}
    def interval(point, se, alpha, *, values):
        seen.update(values)
        return point-se, point+se
    result = PPITSAS(interval=interval)(data, target=Mean())
    np.testing.assert_array_equal(seen['residuals'], [1.,1.,2.])
    assert 'values' not in result.diagnostics
    assert result.confidence_interval == (result.estimate-result.standard_error,
                                          result.estimate+result.standard_error)
    with pytest.raises(TypeError, match='does not support'):
        PPITSAS()(data, target=LinearRegression())
    for lag in (-1, .5, True, np.nan):
        with pytest.raises(ValueError, match='max_lag'):
            PPITSAS(max_lag=lag)
    for bad in ([1.], [1.,np.nan], [[1.,2.]]):
        with pytest.raises(ValueError, match='HAC requires'):
            newey_west_var_of_mean(bad)


def test_simulation_blocks_do_not_leak_labels():
    dgp = TimeSeriesDGP(sites=3)
    data = dgp.draw(100, np.random.default_rng(2))
    records = prepare_records(data, unlabeled_size=50, rectification_size=30, train_size=20)
    modified = dict(data, y=data['y'].copy())
    modified['y'][:50] = 1e6  # Hidden historical labels cannot affect training or inference.
    again = prepare_records(modified, unlabeled_size=50, rectification_size=30, train_size=20)
    for a, b in zip(records, again):
        assert len(a.labels) == 30 and len(a.unlabeled_predictions) == 50
        for name in ('labels', 'labeled_predictions', 'unlabeled_predictions'):
            np.testing.assert_array_equal(getattr(a,name), getattr(b,name))
    # Rectification labels affect only the correction, never the fitted predictor.
    modified['y'][50:80] += 10
    shifted = prepare_records(modified, unlabeled_size=50, rectification_size=30, train_size=20)
    for a,b in zip(records,shifted):
        np.testing.assert_array_equal(a.labeled_predictions,b.labeled_predictions)
        np.testing.assert_array_equal(a.unlabeled_predictions,b.unlabeled_predictions)
        assert PPITSAS()(b,target=Mean()).estimate == pytest.approx(PPITSAS()(a,target=Mean()).estimate + 10)


@pytest.mark.slow
def test_stationary_time_series_coverage():
    results, summary = run_simulation(reps=200, sites=4)
    summary = summary.set_index('method')
    assert .88 < summary.loc['PPITSAS','coverage'] <= 1.
    assert summary.loc['PPITSAS','coverage'] > summary.loc['PPI','coverage'] + .1
    points = results.pivot(index=['rep','site'],columns='method',values='estimate')
    np.testing.assert_array_equal(points['PPI'],points['PPITSAS'])
    per_rep = results[results.method=='PPITSAS'].groupby('rep').covered.mean()
    assert summary.loc['PPITSAS','coverage_mc_se'] == pytest.approx(per_rep.std(ddof=1)/np.sqrt(200))
