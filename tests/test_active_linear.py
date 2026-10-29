"""ASI/RASI coefficient scores, official fixtures, honest selection and coverage."""
import json
from pathlib import Path
import numpy as np
import pytest

from openppi import (ActiveRegressionData,ActiveRegressionIPW,LinearRegression,
    Mean,Uniform,Uncertainty,BalancedDesign,RobustDesign,RegressionRobustDesign,
    linear_regression_scores,select_for_labeling,get_method)
from openppi.sampling import select_burn_in

FIXTURES=json.loads((Path(__file__).parent/'fixtures/active_linear.json').read_text())


@pytest.mark.parametrize('fixture',FIXTURES)
def test_official_census_coefficient_and_covariance(fixture):
    data=ActiveRegressionData(**{k:v for k,v in fixture.items() if k not in ('method','expected')})
    r=get_method(fixture['method']).inference(data,target=LinearRegression())
    np.testing.assert_allclose(r.estimate,fixture['expected']['estimate'],atol=1e-12)
    np.testing.assert_allclose(r.covariance,fixture['expected']['covariance'],atol=1e-12)
    assert r.diagnostics['variance_design']=='active_regression_score_sandwich'
    assert not data.covariates.flags.writeable


def test_focus_score_is_exact_glm_gradient_error_direction():
    rng=np.random.default_rng(10)
    v=rng.normal(size=(100,2)); ref=rng.normal(size=(300,2)); errors=rng.uniform(.1,2,100)
    x=np.c_[np.ones(100),v]; z=np.c_[np.ones(300),ref]
    inv=np.linalg.inv(z.T@z/len(z))
    for focus in (0,1,2):
        scores=linear_regression_scores(v,errors,target=LinearRegression(focus=focus),reference_covariates=ref)
        np.testing.assert_allclose(scores,np.abs(x@inv[:,focus])*errors)
    with pytest.raises(ValueError,match='focus coordinate'):
        linear_regression_scores(v,errors,target=LinearRegression(focus=None))
    with pytest.raises(ValueError): linear_regression_scores(v,-errors,target=LinearRegression())


def test_request_and_full_score_sandwich_with_translation():
    rng=np.random.default_rng(12)
    v=rng.normal(size=(150,2)); f=1+v[:,0]; y=f+rng.normal(size=150)
    target=LinearRegression()
    scores=linear_regression_scores(v,np.ones(150),target=target)
    request=select_for_labeling(f,target=target,design=Uncertainty(),budget=60,scores=scores,covariates=v,random_state=0)
    record=request.with_labels(y[request.indices])
    assert np.isnan(record.observed_labels[~record.queried]).all()
    r=get_method('ASI').inference(record,target=target)
    pseudo=f.copy(); q=request.indices; pseudo[q]+=(y[q]-f[q])/request.propensities[q]
    x=np.c_[np.ones(150),v]
    np.testing.assert_allclose(x.T@(x@r.estimate-pseudo),0,atol=1e-10)
    shift=select_for_labeling(f+7,target=target,design=Uncertainty(),budget=60,scores=scores,covariates=v,random_state=0).with_labels(y[q]+7)
    shifted=get_method('ASI').inference(shift,target=target)
    np.testing.assert_allclose(shifted.estimate,r.estimate+[7,0,0],atol=1e-12)
    np.testing.assert_allclose(shifted.covariance,r.covariance,atol=1e-12)
    captured={}
    from openppi.common.stats import wald_interval
    def interval(point,se,alpha,values=None):
        captured.update(values);return wald_interval(point,se,alpha)
    ActiveRegressionIPW(interval=interval)(record,target=target)
    np.testing.assert_allclose(captured['pseudo_response'],pseudo)
    assert captured['regression_scores'].shape==(150,3)


class ConstantError:
    def fit(self,x,y):
        self.value=float(np.mean(y));return self
    def predict(self,x):return np.full(len(x),self.value)


def test_rasi_fits_absolute_errors_and_weights_focus_before_robust_path():
    rng=np.random.default_rng(44)
    v=rng.normal(size=(200,2)); features=rng.normal(size=(200,4)); f=v[:,0]+1
    y=f+rng.normal(size=200)
    target=LinearRegression()
    burn=select_burn_in(200,size=40,random_state=3)
    design=RobustDesign.fit(features[burn.indices],y[burn.indices],f[burn.indices],
        covariates=v[burn.indices],reference_covariates=v,target=target,
        error_model_factory=ConstantError,rho=40/160,constraint=.1)
    assert isinstance(design,RegressionRobustDesign)
    assert design.error_model.value==pytest.approx(np.abs(y[burn.indices]-f[burn.indices]).mean())
    rest=burn.remaining
    errors=design.errors(features[rest],v[rest])
    expected=linear_regression_scores(v[rest],np.full(len(rest),design.error_model.value),target=target,reference_covariates=v)
    np.testing.assert_allclose(errors,expected)
    request=select_for_labeling(f[rest],target=target,design=design,budget=40,scores=errors,covariates=v[rest],random_state=4)
    full=burn.combine(f,y[burn.indices],request,y[rest[request.indices]],covariates=v)
    np.testing.assert_equal(full.propensities[burn.indices],1.)
    np.testing.assert_allclose(full.observed_labels[burn.indices],y[burn.indices])
    r=get_method('RASI').inference(full,target=target)
    assert r.diagnostics['burn_in']==40
    assert r.diagnostics['sampling_focus']==1
    with pytest.raises(ValueError,match='full target-covariate'):
        burn.combine(f,y[burn.indices],request,y[rest[request.indices]])


def test_focused_radius_cv_fits_only_burn_in_and_training_folds():
    rng=np.random.default_rng(14)
    v=rng.normal(size=(80,2)); f=v[:,0]+1; y=f+rng.normal(size=80)
    seen=[]
    class TrackedError(ConstantError):
        def fit(self,x,outcomes):
            seen.append((set(x[:,0].astype(int)),outcomes.copy()))
            return super().fit(x,outcomes)
    features=np.arange(80)[:,None]
    design=RobustDesign.fit(features,y,f,covariates=v,reference_covariates=v,target=LinearRegression(),
        error_model_factory=TrackedError,rho=.3,candidates=(0.,.03,.1),folds=4,grid=21,random_state=3)
    assert len(seen)==5
    assert all(len(ids)==60 for ids,values in seen[:-1]) and len(seen[-1][0])==80
    for ids,values in seen:
        assert set(np.round(values,12))<=set(np.round(np.abs(y-f),12))
    assert design.constraint in (0.,.03,.1)
    pi=design.calibrate(design.errors(features,v),.3).probability(design.errors(features,v))
    assert np.all((pi>0)&(pi<=1)) and pi.sum()<=24+1e-10


def test_input_validation_and_unsupported_dependent_designs():
    x=np.arange(20)[:,None].astype(float); f=np.linspace(0,2,20)
    target=LinearRegression()
    with pytest.raises(ValueError,match='target covariates'):
        select_for_labeling(f,target=target,design=Uniform(),budget=10,scores=np.ones(20))
    with pytest.raises(TypeError,match='dependent designs'):
        select_for_labeling(f,target=target,design=BalancedDesign(),budget=10,scores=np.ones(20),covariates=x)
    with pytest.raises(ValueError):
        ActiveRegressionData(f,np.full(20,np.nan),np.zeros(20,dtype=bool),np.ones(20),covariates=x[:10])
    assert LinearRegression in get_method('ASI').supported_targets
    assert LinearRegression in get_method('RASI').supported_targets
    assert LinearRegression not in get_method('BAI').supported_targets
    assert LinearRegression not in get_method('FTRL').supported_targets


@pytest.mark.slow
def test_active_coefficient_coverage_in_both_noise_settings():
    rng=np.random.default_rng(861)
    truth=np.array([1.,.8,-.6]);covered=np.zeros((2,2,3))
    target=LinearRegression()
    for case in range(2):
        for _ in range(200):
            v=rng.normal(size=(2000,2)); f=1+.8*v[:,0]-.6*v[:,1]+.25*(v[:,0]**2-1)
            sigma=np.full(2000,.5) if case==0 else .3+.5/(1+np.exp(-v[:,0]))
            y=f+sigma*rng.normal(size=2000)
            scores=linear_regression_scores(v,np.ones(2000),target=target)
            req=select_for_labeling(f,target=target,design=Uncertainty(),budget=600,scores=scores,covariates=v,random_state=rng)
            r=get_method('ASI').inference(req.with_labels(y[req.indices]),target=target)
            lo,hi=r.confidence_interval;covered[case,0]+=(lo<=truth)&(truth<=hi)
            burn=select_burn_in(2000,size=100,random_state=rng)
            b,rest=burn.indices,burn.remaining
            design=RobustDesign.fit(v[b],y[b],f[b],covariates=v[b],reference_covariates=v,target=target,
                error_model_factory=ConstantError,rho=500/1900,constraint=.1,grid=21)
            req=select_for_labeling(f[rest],target=target,design=design,budget=500,
                scores=design.errors(v[rest],v[rest]),covariates=v[rest],random_state=rng)
            data=burn.combine(f,y[b],req,y[rest[req.indices]],covariates=v)
            r=get_method('RASI').inference(data,target=target)
            lo,hi=r.confidence_interval;covered[case,1]+=(lo<=truth)&(truth<=hi)
    assert np.all((covered/200>.88)&(covered/200<.995)),covered
