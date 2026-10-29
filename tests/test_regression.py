"""Continuous mean inference, simulation honesty, backwards compatibility and coverage."""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from openppi import (Mean, PPIData, StratifiedPPIData, StratPPI, TwoSample, PSPA, RePPI,
                     CalibratedPPI, IPW, Uniform, Uncertainty, RoganGladen,
                     MisclassificationMLE, select_for_labeling)
from config import Config
from dataset.regression_sim import RegressionDGP
from model.learners import SimulationLearner, SklearnModel, RegressionUncertainty
from online import active_label
from regression import RegressionConfig, run_experiment, PASSIVE, ACTIVE

TARGET=Mean(outcome_type='continuous')


def test_continuous_data_and_binary_only_methods():
    data=PPIData([-3.,4.,2.],[8.,-1.,2.],[-4.,7.])
    assert not data.labels.flags.writeable
    for labels,f,g in [([1,np.nan],[1,2],[3,4]),([1,2],[1,np.inf],[3,4]),([1,2],[1],[3,4])]:
        with pytest.raises(ValueError): PPIData(labels,f,g)
    binary_predictions=PPIData([-2.,3.],[0,1],[0,1])
    for method in (RoganGladen,MisclassificationMLE):
        with pytest.raises(ValueError,match='binary labels'):
            method()(binary_predictions,target=TARGET)
    assert TwoSample(lam=0)(data,target=TARGET).estimate==pytest.approx(1.)


@pytest.mark.parametrize('method',[
    lambda:TwoSample(lam=0),lambda:TwoSample(lam=1),lambda:TwoSample(lam='tuned'),
    PSPA,lambda:RePPI(random_state=123),CalibratedPPI,
    lambda:CalibratedPPI(calibration='linear')])
def test_translation_equivariance(method):
    rng=np.random.default_rng(901)
    f=rng.normal(1,2,80); g=rng.normal(1,2,150)
    y=2+.7*f+rng.normal(size=len(f))
    data=PPIData(y,f,g)
    offset=7.3
    translated=PPIData(y+offset,f+offset,g+offset)
    a,b=method()(data,target=TARGET),method()(translated,target=TARGET)
    if a.method == 'RePPI':
        default = method()(data,target=Mean())
        assert default.standard_error == pytest.approx(a.standard_error)
    assert b.estimate==pytest.approx(a.estimate+offset,abs=1e-12)
    assert b.standard_error==pytest.approx(a.standard_error,abs=1e-12)
    np.testing.assert_allclose(b.confidence_interval,np.array(a.confidence_interval)+offset,atol=1e-12)
    sa=StratifiedPPIData({'a':data,'b':data},{'a':.3,'b':.7})
    sb=StratifiedPPIData({'a':translated,'b':translated},{'a':.3,'b':.7})
    ar,br=StratPPI()(sa,target=TARGET),StratPPI()(sb,target=TARGET)
    assert br.estimate==pytest.approx(ar.estimate+offset)
    assert br.standard_error==pytest.approx(ar.standard_error)


def test_dgp_noise_and_independence():
    for noise in ('homoskedastic','heteroskedastic'):
        dgp=RegressionDGP(noise=noise)
        data=dgp.draw(50000,np.random.default_rng(15))
        assert dgp.truth()==1.
        np.testing.assert_allclose(data['mean'],1+.8*data['x'][:,0]-.6*data['x'][:,1]+.4*(data['x'][:,2]**2-1))
        assert data['y'].mean()==pytest.approx(1.,abs=.025)
        eps=(data['y']-data['mean'])/data['sigma']
        residual=data['j']-np.array(dgp.offsets)-data['mean'][:,None]*np.array(dgp.slopes)
        assert abs(np.corrcoef(eps,residual[:,0])[0,1])<.02
        assert np.corrcoef(residual.T)[0,1]>.1
        assert data['j'].min()<0 and data['j'].max()>1
        if noise=='homoskedastic': np.testing.assert_equal(data['sigma'],.5)
        else: assert np.all((data['sigma']>.3)&(data['sigma']<1.))
    with pytest.raises(ValueError): RegressionDGP(noise='bad')
    with pytest.raises(ValueError): RegressionDGP(judges=2,slopes=(1.,))


def test_learner_predictions_are_not_clipped_in_continuous_mode():
    x=np.arange(20)[:,None]; y=np.linspace(3.,8.,20)
    binary=SimulationLearner(Config(),0).update(x,y,np.ones(20))
    continuous=SimulationLearner(Config(outcome_type='continuous',initial_prediction=2.),0)
    np.testing.assert_equal(continuous.predict(x),2.)
    continuous.update(x,y,np.ones(20))
    assert continuous.predict(x).min()>1
    assert binary.predict(x).max()<=.99
    for outcome in ('binary','continuous'):
        adapter=SklearnModel(RandomForestRegressor(n_estimators=5,random_state=0),outcome_type=outcome)
        adapter.update(x,y,np.ones(20))
        if outcome=='continuous': assert adapter.predict(x).max()>1
        else: assert adapter.predict(x).max()<=.99


def test_continuous_sampling_requires_explicit_scores_even_inside_unit_interval():
    for f in (np.linspace(-2,3,80),np.full(80,.5)):
        with pytest.raises(ValueError,match='explicit regression uncertainty scores'):
            select_for_labeling(f,target=TARGET,design=Uncertainty(),budget=30)
    p=np.linspace(-2,3,80); y=2*p+1
    request=select_for_labeling(p,target=TARGET,design=Uniform(),scores=np.ones(80),budget=30,random_state=3)
    sample=request.with_labels(y[request.indices])
    assert np.isnan(sample.observed_labels[~sample.queried]).all()
    r=IPW()(sample,target=TARGET)
    shifted=select_for_labeling(p+7,target=TARGET,design=Uniform(),scores=np.ones(80),budget=30,random_state=3).with_labels(y[request.indices]+7)
    sr=IPW()(shifted,target=TARGET)
    assert sr.estimate==pytest.approx(r.estimate+7)
    assert sr.standard_error==pytest.approx(r.standard_error)
    np.testing.assert_allclose(Mean().sampling_scores(np.empty((2,0)),[.2,.8]),[.4,.4])


def test_continuous_online_updates_preserve_history_and_allow_single_value_labels():
    x=np.arange(80)[:,None].astype(float)
    calls=[]; fitted=[]
    class Model:
        def predict(self,x): return np.full(len(x),fitted[-1] if fitted else -5.)
        def update(self,x,y,sample_weight):
            assert set(x[:,0].astype(int))<=set(calls)
            np.testing.assert_equal(y,3.)
            assert np.all(sample_weight>=1)
            fitted.append(3.)
    def oracle(i): calls.append(i); return 3.
    record=active_label(x,oracle=oracle,target=TARGET,model_factory=Model,update='refit',
        update_every=10,min_train=3,budget=40,initial_prediction=-5.,score_fn=lambda x,p:np.ones(len(x)),random_state=2)
    assert fitted and calls==record.queried.nonzero()[0].tolist()
    np.testing.assert_equal(record.predictions[:10],-5.)
    assert record.predictions[-1]==3.
    with pytest.raises(ValueError,match='score_fn'):
        active_label(x,oracle=oracle,target=TARGET,model=Model(),budget=40)
    with pytest.raises(ValueError,match='hard predictions'):
        active_label(x,oracle=oracle,target=TARGET,model=Model(),budget=40,prediction_mode='hard')


def test_error_learner_uses_provided_honest_residuals():
    x=np.arange(50)[:,None].astype(float)
    f=np.full(50,-2.); y=np.full(50,2.)
    model=RegressionUncertainty().fit(x,y,f)
    np.testing.assert_allclose(model.predict(x),16.)
    np.testing.assert_allclose(model.scores(x),4.)


def test_regression_experiment_groups_costs_reproducibility_and_summary():
    c=RegressionConfig(reps=1,n=400,budget=80,train_size=60,pilot_size=50,burn_in=20)
    frame,summary=run_experiment(c)
    assert len(frame)==2*(len(PASSIVE)+len(ACTIVE))
    assert set(frame.group)=={'given_labels','select_labels'}
    assert np.isfinite(frame[['estimate','width','standard_error']]).all().all()
    np.testing.assert_array_equal(frame.total_labels.to_numpy(),(frame.training_labels+frame.pilot_labels+frame.actual_labels).to_numpy())
    rasi=frame[frame.method=='RASI']
    assert (rasi.burn_in_labels==20).all() and (rasi.pilot_labels==0).all()
    assert (frame[frame.method=='BAI'].actual_labels==80).all()
    assert (frame[frame.method=='StratPPI'].actual_labels==80).all()
    assert (frame.training_labels==60).all()
    assert (summary.rmse>=0).all()
    repeat,_=run_experiment(c,('homoskedastic',))
    np.testing.assert_allclose(frame[frame.case=='homoskedastic'].estimate,repeat.estimate)


@pytest.mark.slow
def test_continuous_interval_coverage_with_fixed_oracle_predictor():
    covered=np.zeros((2,5))
    methods=[TwoSample(lam=1),TwoSample(lam='tuned'),PSPA(),CalibratedPPI(),CalibratedPPI(calibration='linear')]
    rng=np.random.default_rng(902)
    for case,noise in enumerate(('homoskedastic','heteroskedastic')):
        dgp=RegressionDGP(noise=noise)
        for _ in range(200):
            sample=dgp.draw(1600,rng)
            f=sample['mean']
            data=PPIData(sample['y'][:400],f[:400],f[400:])
            for j,method in enumerate(methods):
                r=method(data,target=TARGET)
                covered[case,j]+=r.confidence_interval[0]<=1.<=r.confidence_interval[1]
    assert np.all((covered/200>.89)&(covered/200<.995)),covered
