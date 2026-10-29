"""Regression M-estimator equations, official fixtures and vector compatibility."""
import json
from pathlib import Path
import numpy as np
import pytest

from openppi import (LinearRegression, RegressionData, Mean, PPIData, OLS,
                     TwoSample, PSPA, get_method)
from openppi.common.regression import score
from openppi.common.stats import wald_interval

FIXTURE=json.loads((Path(__file__).parent/'fixtures/linear.json').read_text())


def sample(seed=0,n=200,big_n=600):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,2)); z=rng.normal(size=(big_n,2))
    y=1+x@np.array([.8,-.6])+rng.normal(size=n)*(.4+.2*np.abs(x[:,0]))
    f=1+x@np.array([.6,-.3]); g=1+z@np.array([.6,-.3])
    return RegressionData(y,f,g,x,z)


def test_ols_score_and_robust_covariance():
    data=sample()
    x=np.c_[np.ones(len(data.labels)),data.labeled_covariates]
    beta=np.linalg.lstsq(x,data.labels,rcond=None)[0]
    inverse=np.linalg.inv(x.T@x/len(x))
    scores=x*(x@beta-data.labels)[:,None]
    expected=inverse@np.cov(scores.T)@inverse.T/len(x)
    result=OLS()(data,target=LinearRegression())
    np.testing.assert_allclose(result.estimate,beta,atol=1e-12)
    np.testing.assert_allclose(result.covariance,expected,atol=1e-12)
    np.testing.assert_allclose(result.standard_error**2,np.diag(expected))
    lo,hi=wald_interval(beta,np.sqrt(np.diag(expected)))
    np.testing.assert_allclose(result.confidence_interval,(lo,hi))
    for a in (result.estimate,result.standard_error,result.covariance,*result.confidence_interval):
        assert not a.flags.writeable
    np.testing.assert_allclose(get_method('PPI++').inference.__class__(lam=0)(data,target=LinearRegression()).estimate,beta)


@pytest.mark.parametrize('lam',[0.,.4,1.])
def test_official_ppi_matched_gram_fixture(lam):
    f=FIXTURE['ppi']
    data=RegressionData(**{k:v for k,v in f.items() if k!='ppi'})
    result=TwoSample(lam=lam)(data,target=LinearRegression())
    expected=f['ppi'][str(lam)]
    np.testing.assert_allclose(result.estimate,expected['estimate'],atol=1e-12)
    np.testing.assert_allclose(result.confidence_interval,(expected['lower'],expected['upper']),atol=1e-12)


def test_ppi_solves_squared_loss_for_unmatched_designs():
    data=sample()
    x=np.c_[np.ones(len(data.labels)),data.labeled_covariates]
    z=np.c_[np.ones(len(data.unlabeled_predictions)),data.unlabeled_covariates]
    lam=.4
    result=TwoSample(lam=lam)(data,target=LinearRegression())
    equation=(score(x,data.labels,result.estimate)-lam*score(x,data.labeled_predictions,result.estimate)).mean(0)+lam*score(z,data.unlabeled_predictions,result.estimate).mean(0)
    np.testing.assert_allclose(equation,0,atol=1e-12)


def test_pspa_official_point_and_weights_with_symmetric_covariance():
    f=FIXTURE['pspa']
    data=RegressionData(**{k:v for k,v in f.items() if k!='expected'})
    result=PSPA()(data,target=LinearRegression())
    np.testing.assert_allclose(result.estimate,f['expected']['estimate'],atol=1e-12)
    w=np.array(result.diagnostics['prediction_weight'])
    np.testing.assert_allclose(w,f['expected']['weights'],atol=1e-12)
    x=np.c_[np.ones(len(data.labels)),data.labeled_covariates]
    z=np.c_[np.ones(len(data.unlabeled_predictions)),data.unlabeled_covariates]
    inverse=np.linalg.inv((x.T@x+z.T@z)/(len(x)+len(z)))
    sy=score(x,data.labels,result.estimate)
    sf=score(x,data.labeled_predictions,result.estimate)
    su=score(z,data.unlabeled_predictions,result.estimate)
    cross=np.cov(np.c_[sy,sf].T)[:3,3:]
    W=np.diag(w)
    meat=(np.cov(sy.T)+W@np.cov(sf.T)@W-cross@W-W@cross.T)/len(x)+W@np.cov(su.T)@W/len(z)
    np.testing.assert_allclose(result.covariance,inverse@meat@inverse.T,atol=1e-12)
    assert np.linalg.eigvalsh(result.covariance).min()>0


def test_focused_tuning_formula_and_intercept_coordinate():
    data=sample(32)
    x=np.c_[np.ones(len(data.labels)),data.labeled_covariates]
    z=np.c_[np.ones(len(data.unlabeled_predictions)),data.unlabeled_covariates]
    beta=np.linalg.lstsq(x,data.labels,rcond=None)[0]
    inv=np.linalg.inv((x.T@x+z.T@z)/(len(x)+len(z)))
    gy,gf,gu=(a@inv.T for a in (score(x,data.labels,beta),score(x,data.labeled_predictions,beta),score(z,data.unlabeled_predictions,beta)))
    for focus in (0,1,2,None):
        r=TwoSample(lam='tuned')(data,target=LinearRegression(focus=focus))
        a,b,c=(gy,gf,gu) if focus is None else (gy[:,[focus]],gf[:,[focus]],gu[:,[focus]])
        numerator=np.sum((a-a.mean(0))*(b-b.mean(0)))/(len(x)-1)
        denominator=(1+len(x)/len(z))*np.trace(np.atleast_2d(np.cov(np.vstack([b,c]).T)))
        assert r.diagnostics['prediction_weight']==pytest.approx(np.clip(numerator/denominator,0,1))
        assert r.diagnostics['focus']==focus
    assert r.diagnostics['coefficient_names']==['intercept','V1','V2']


@pytest.mark.parametrize('method',[OLS,lambda:TwoSample(lam=1),lambda:TwoSample(lam='tuned'),PSPA])
def test_intercept_translation_and_no_intercept(method):
    data=sample(13)
    shifted=RegressionData(data.labels+8,data.labeled_predictions+8,data.unlabeled_predictions+8,data.labeled_covariates,data.unlabeled_covariates)
    a,b=method()(data,target=LinearRegression()),method()(shifted,target=LinearRegression())
    np.testing.assert_allclose(b.estimate,a.estimate+[8,0,0],atol=1e-12)
    np.testing.assert_allclose(b.covariance,a.covariance,atol=1e-12)
    r=method()(data,target=LinearRegression(fit_intercept=False,focus=0))
    assert r.estimate.shape==(2,)


def test_intercept_only_reduces_to_mean():
    rng=np.random.default_rng(9)
    y=rng.normal(size=40); f=y+rng.normal(size=40); g=rng.normal(size=120)
    data=RegressionData(y,f,g,np.empty((40,0)),np.empty((120,0)))
    for estimator in (TwoSample(lam=0),TwoSample(lam=1),PSPA()):
        vector=estimator(data,target=LinearRegression(focus=0))
        scalar=estimator(PPIData(y,f,g),target=Mean())
        assert vector.estimate[0]==pytest.approx(scalar.estimate)
        assert vector.standard_error[0]==pytest.approx(scalar.standard_error)
        assert scalar.covariance is None and isinstance(scalar.estimate,float)


def test_validation_metadata_and_unsupported_methods():
    data=sample()
    for kwargs in (dict(focus=-1),dict(focus=True),dict(fit_intercept=1),dict(max_condition=1)):
        with pytest.raises(ValueError): LinearRegression(**kwargs)
    with pytest.raises(ValueError): OLS()(data,target=LinearRegression(focus=3))
    duplicate=RegressionData(data.labels,data.labeled_predictions,data.unlabeled_predictions,
                             np.ones((200,1)),np.ones((600,1)))
    with pytest.raises(ValueError,match='rank deficient'): OLS()(duplicate,target=LinearRegression())
    with pytest.raises(ValueError): RegressionData([1,2],[1,2],[1,2],[[1],[2]],[[1,2],[3,4]])
    with pytest.raises(TypeError,match='RegressionData'):
        TwoSample()(PPIData([1,2],[1,2],[1,2]),target=LinearRegression())
    for name in ('RePPI','CalPPI','StratPPI','BAI','FAB-PPI','Cluster-PPI','RG','MLE'):
        assert LinearRegression not in get_method(name).supported_targets
        with pytest.raises(TypeError,match='does not support target LinearRegression'):
            get_method(name).inference(data,target=LinearRegression())
    assert get_method('PPI++').target_data_types[LinearRegression] is RegressionData
    with pytest.raises(ValueError,match='coefficient-specific'):
        LinearRegression().sampling_scores(np.ones((2,1)),[1,2])


def test_shared_interval_callback_receives_vectors():
    calls=[]
    def interval(beta,se,alpha,values=None):
        calls.append(beta.shape); return wald_interval(beta,se,alpha)
    result=OLS(interval=interval)(sample(),target=LinearRegression())
    assert calls==[(3,)] and result.covariance.shape==(3,3)


@pytest.mark.slow
def test_regression_coefficients_coverage_under_misspecified_linear_mean():
    rng=np.random.default_rng(876)
    methods=[OLS(),TwoSample(lam=1),TwoSample(lam='tuned'),PSPA()]
    truth=np.array([1.,.8,-.6])
    covered=np.zeros((4,3))
    for _ in range(250):
        x=rng.normal(size=(1800,2))
        mean=1+.8*x[:,0]-.6*x[:,1]+.3*(x[:,0]**2-1)
        y=mean[:500]+(.3+.3*np.abs(x[:500,0]))*rng.normal(size=500)
        f=mean+.2+.2*x[:,1]
        data=RegressionData(y,f[:500],f[500:],x[:500],x[500:])
        for j,method in enumerate(methods):
            r=method(data,target=LinearRegression())
            low,high=r.confidence_interval
            covered[j]+=(low<=truth)&(truth<=high)
    assert np.all((covered/250>.89)&(covered/250<.995)),covered


def test_coefficient_experiment_keeps_target_features_separate_from_predictor_features():
    from regression import RegressionConfig
    from linear_regression import run_linear_experiment
    from dataset.regression_sim import RegressionDGP
    cfg=RegressionConfig(reps=2,n=400,budget=100,train_size=80,pilot_size=50,burn_in=20)
    frame,summary=run_linear_experiment(cfg,('homoskedastic',))
    assert len(frame)==2*7*6
    np.testing.assert_allclose(RegressionDGP.projection_coefficients(),[1,.8,-.6,0,0,0])
    focus=summary[(summary.coordinate==1)&(summary.group=='given_labels')]
    assert set(focus.method)=={'OLS','PPI','PPI++','PSPA'}
    assert (focus.truth==.8).all()
    assert (frame.training_labels==80).all()
    assert (frame[frame.group=='given_labels'].total_labels==180).all()
    assert set(frame[frame.group=='select_labels'].method)=={'Uniform+IPW','ASI','RASI'}
    np.testing.assert_array_equal(frame.total_labels.to_numpy(),(frame.actual_labels+frame.training_labels+frame.pilot_labels).to_numpy())
    matrix=frame[(frame.rep==0)&(frame.method=='PPI++')].sort_values('coordinate')
    cov=np.array([json.loads(row) for row in matrix.covariance_row])
    np.testing.assert_allclose(np.diag(cov),matrix.standard_error.to_numpy()**2)
