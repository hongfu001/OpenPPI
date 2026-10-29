"""Paper Algorithms 1--2, official-package fixtures, PAVA and coverage."""
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from openppi import CalibratedPPI, Mean, PPIData, get_method
from openppi.common.stats import wald_interval
from openppi.inference.calibrated import _isotonic_fit

FIXTURES = json.loads((Path(__file__).parent / 'fixtures/calibrated.json').read_text())


@pytest.mark.parametrize('case', FIXTURES)
@pytest.mark.parametrize('calibration', ['isotonic', 'linear'])
def test_official_package_fixtures(case, calibration):
    data=PPIData(**{k:v for k,v in case.items() if k!='expected'})
    result=CalibratedPPI(calibration=calibration,variance='two_sample')(data,target=Mean())
    expected=case['expected'][calibration]
    assert result.estimate==pytest.approx(expected['estimate'],abs=1e-12)
    assert result.standard_error==pytest.approx(expected['se'],abs=1e-12)
    assert result.confidence_interval==pytest.approx(expected['ci'],abs=1e-12)


@pytest.mark.parametrize('calibration', ['isotonic','linear'])
def test_paper_influence_formula_and_interval_callback(calibration):
    case=FIXTURES[0]
    data=PPIData(**{k:v for k,v in case.items() if k!='expected'})
    captured={}
    def interval(point,se,alpha,values=None):
        captured.update(values)
        return wald_interval(point,se,alpha)
    result=CalibratedPPI(calibration=calibration,interval=interval)(data,target=Mean(),alpha=.1)
    y,f,g=data.labels,data.labeled_predictions,data.unlabeled_predictions
    if calibration=='isotonic':
        fit=IsotonicRegression(out_of_bounds='clip').fit(f,y)
        m_l,m_u=fit.predict(f),fit.predict(g)
    else:
        beta=np.linalg.lstsq(np.c_[np.ones(len(f)),f],y,rcond=None)[0]
        m_l,m_u=beta[0]+beta[1]*f,beta[0]+beta[1]*g
    n,big_n=len(y),len(g); total=n+big_n
    point=np.r_[m_l,m_u].mean()
    d_l=m_l-point+(y-m_l)/(n/total); d_u=m_u-point
    se=np.sqrt(np.sum(d_l**2)+np.sum(d_u**2))/total
    assert result.estimate==pytest.approx(point)
    assert result.standard_error==pytest.approx(se)
    assert result.confidence_interval==pytest.approx(wald_interval(point,se,.1))
    for key,expected in [('labeled_predictions',m_l),('unlabeled_predictions',m_u),('labeled_influence',d_l),('unlabeled_influence',d_u)]:
        np.testing.assert_allclose(captured[key],expected,atol=1e-14)
    assert abs(result.diagnostics['calibration_residual_mean'])<1e-14
    # Pooled plug-in equals pooled AIPW because the fitted residuals average zero.
    assert point==pytest.approx(y.mean()+big_n/total*(m_u.mean()-m_l.mean()))


def test_pava_matches_sklearn_with_ties_and_extrapolation():
    rng=np.random.default_rng(8)
    for _ in range(30):
        f=rng.integers(0,12,100)/12; y=rng.binomial(1,f)
        knots,values=_isotonic_fit(f,y)
        query=np.linspace(-.1,1.1,151)
        expected=IsotonicRegression(out_of_bounds='clip').fit(f,y).predict(query)
        np.testing.assert_allclose(np.interp(query,knots,values),expected,atol=1e-14)
        assert np.diff(values).min()>=0


@pytest.mark.parametrize('calibration',['isotonic','linear'])
def test_constant_scores_and_labels(calibration):
    data=PPIData([0,1,1,0],[.3]*4,[.2,.5,.8])
    result=CalibratedPPI(calibration=calibration)(data,target=Mean())
    assert result.estimate==pytest.approx(.5)
    assert result.standard_error==pytest.approx(.25)
    same=PPIData([1,1,1],[.1,.4,.8],[.2,.6])
    result=CalibratedPPI(calibration=calibration)(same,target=Mean())
    assert result.estimate==pytest.approx(1.)
    assert result.standard_error==pytest.approx(0.,abs=1e-14)


def test_unclipped_linear_predictions_and_negative_slope():
    data=PPIData([1,1,0,0],[.4,.45,.55,.6],[0.,1.])
    r=get_method('LinearCalPPI').inference(data,target=Mean())
    assert r.diagnostics['calibration_parameters']['slope']<0
    assert r.estimate==pytest.approx(.5)
    assert abs(r.diagnostics['calibration_residual_mean'])<1e-14
    assert get_method('CalPPI').sampling is None
    with pytest.raises(ValueError): CalibratedPPI(calibration='auto')
    with pytest.raises(ValueError): CalibratedPPI(variance='pooled')


@pytest.mark.slow
def test_coverage_with_nonlinear_miscalibration():
    rng=np.random.default_rng(731)
    covered=np.zeros(2)
    for _ in range(200):
        f=rng.uniform(0,1,1800)
        y=rng.binomial(1,.1+.8*f[:400]**2)
        data=PPIData(y,f[:400],f[400:])
        for j,calibration in enumerate(['isotonic','linear']):
            r=CalibratedPPI(calibration=calibration)(data,target=Mean())
            low,high=r.confidence_interval
            covered[j]+=low<=.1+.8/3<=high
    assert np.all((covered/200>.90)&(covered/200<.995)),covered


def test_linear_paper_differs_from_official_clipped_extrapolation():
    # The official API clips affine predictions; Algorithms 2 and Appendix M.3 do not.
    data=PPIData([0,1,0,1,1,0,1,1],[.1,.2,.2,.4,.5,.6,.8,.9],[.05,.15,.3,.5,.7,.95])
    r=CalibratedPPI(calibration='linear',variance='two_sample')(data,target=Mean())
    a=r.diagnostics['calibration_parameters']['intercept']
    b=r.diagnostics['calibration_parameters']['slope']
    m_l=np.clip(a+b*data.labeled_predictions,0,1)
    m_u=np.clip(a+b*data.unlabeled_predictions,0,1)
    c=len(m_u)/(len(m_l)+len(m_u))
    official_point=(data.labels-c*m_l).mean()+c*m_u.mean()
    assert official_point==pytest.approx(.6161646286907247)
    assert r.estimate!=pytest.approx(official_point)
