"""StratPPI mean formulas, allocation propositions, data design and coverage."""
import numpy as np
import pytest
from scipy.optimize import minimize

from openppi import (Mean, PPIData, StratifiedPPIData, StratPPI, TwoSample,
                     confidence_residual_variance, stratified_allocate, get_method)

A = PPIData([1,0,1,1], [.8,.4,.6,.2], [.1,.2,.3,.4,.5])
B = PPIData([0,1,0,0,1,0], [.1,.8,.2,.1,.7,.3], [.1,.2,.8,.4])


def test_weighted_mean_and_variance_by_hand():
    data = StratifiedPPIData({'a': A,'b': B}, {'a': .7,'b': .3})
    weights = {'a': -.5,'b': 1.2}
    point, variance = 0., 0.
    for key, sample in data.strata.items():
        y,f,g = sample.labels,sample.labeled_predictions,sample.unlabeled_predictions
        lam,w = weights[key],data.weights[key]
        point += w * (y.mean() + lam * (g.mean()-f.mean()))
        variance += w**2 * ((y-lam*f).var(ddof=1)/len(y)+lam**2*g.var(ddof=1)/len(g))
    captured=[]
    def interval(point,se,alpha,values=None):
        captured.append((point,se,alpha)); return (point-se,point+se)
    r = StratPPI(lam=weights,interval=interval)(data,target=Mean(),alpha=.1)
    assert r.estimate == pytest.approx(point)
    assert r.standard_error**2 == pytest.approx(variance)
    assert captured == [(r.estimate,r.standard_error,.1)]
    assert r.diagnostics['labeled_count']==10 and r.diagnostics['unlabeled_count']==9
    assert sum(s['variance_contribution'] for s in r.diagnostics['strata'].values())==pytest.approx(variance)
    assert r.diagnostics['strata']['a']['prediction_weight']==-.5


@pytest.mark.parametrize('lam', [0.,1.,-.5,'tuned'])
@pytest.mark.parametrize('variance',['unlabeled','pooled'])
def test_single_stratum_reduces_to_two_sample(lam,variance):
    data=StratifiedPPIData({'only':A},{'only':1.})
    r=StratPPI(lam=lam,variance=variance)(data,target=Mean())
    baseline=TwoSample(lam=lam,variance=variance)(A,target=Mean())
    assert (r.estimate,r.standard_error,*r.confidence_interval)==pytest.approx((baseline.estimate,baseline.standard_error,*baseline.confidence_interval))


def test_tuning_is_within_each_layer_and_not_clipped():
    reverse=PPIData(A.labels,1-A.labeled_predictions,1-A.unlabeled_predictions)
    constant=PPIData(B.labels,[.3]*6,[.3]*4)
    data=StratifiedPPIData({0:reverse,1:constant},{0:.2,1:.8})
    r=get_method('StratPPI').inference(data,target=Mean())
    for k,sample in data.strata.items():
        y,f,g=sample.labels,sample.labeled_predictions,sample.unlabeled_predictions
        expected=0. if k==1 else np.cov(y,f)[0,1]/((1+len(y)/len(g))*np.r_[f,g].var(ddof=1))
        assert r.diagnostics['strata'][k]['prediction_weight']==pytest.approx(expected)
    assert r.diagnostics['strata'][0]['prediction_weight']<0
    assert get_method('StratPPI').sampling is stratified_allocate


def test_data_validation_and_snapshot():
    strata={'a':A}; weights={'a':1.}
    data=StratifiedPPIData(strata,weights)
    strata.clear(); weights['a']=.2
    assert data.strata['a'] is A and data.weights['a']==1.
    with pytest.raises(TypeError): data.weights['a']=.5
    for strata,weights in [({},{}),({'a':A},{'b':1}),({'a':A},{'a':.8}),({'a':A},{'a':np.nan}),({'a':A,'b':B},{'a':1.,'b':0.})]:
        with pytest.raises(ValueError): StratifiedPPIData(strata,weights)
    with pytest.raises(TypeError): StratifiedPPIData({'a':object()},{'a':1.})
    for kwargs in [dict(lam='auto'),dict(lam=np.nan),dict(lam={'a':np.inf}),dict(variance='bad')]:
        with pytest.raises(ValueError): StratPPI(**kwargs)
    with pytest.raises(ValueError,match='stratum keys'):
        StratPPI(lam={'b':1})(data,target=Mean())
    with pytest.raises(TypeError): StratPPI()(A,target=Mean())


def test_neyman_allocation_and_numeric_optimum():
    weights={'a':.2,'b':.3,'c':.5}
    residual={'a':.04,'b':.25,'c':.09}
    prediction={'a':.16,'b':.01,'c':.04}
    plan=stratified_allocate(weights,residual,prediction,labeled_budget=1000,unlabeled_budget=2000)
    for v,rates,counts,budget in [(residual,plan.labeled_rates,plan.labeled_counts,1000),(prediction,plan.unlabeled_rates,plan.unlabeled_counts,2000)]:
        w=np.array(list(weights.values())); v=np.array(list(v.values()))
        expected=w*np.sqrt(v); expected/=expected.sum()
        np.testing.assert_allclose(list(rates.values()),expected)
        assert sum(counts.values())==budget and min(counts.values())>=2
        result=minimize(lambda r: np.sum(w**2*v/r),[1/3]*3,method='SLSQP',bounds=[(.001,1)]*3,constraints={'type':'eq','fun':lambda r:r.sum()-1},options={'ftol':1e-12})
        assert np.sum(w**2*v/expected)==pytest.approx(result.fun,rel=1e-6)
        assert np.sum(w**2*v/expected)<=np.sum(w**2*v/w)


def test_allocation_minima_zero_variance_and_validation():
    w={'a':.1,'b':.9}
    plan=stratified_allocate(w,{'a':0,'b':1},{'a':0,'b':0},labeled_budget=20,unlabeled_budget=10)
    assert plan.labeled_counts=={'a':2,'b':18}
    assert sum(plan.unlabeled_counts.values())==10
    for budget in [3,4.5,True]:
        with pytest.raises(ValueError): stratified_allocate(w,w,w,labeled_budget=budget,unlabeled_budget=10)
    for kwargs in [dict(min_count=1),dict(min_count=True),dict(residual_variances={'a':-1,'b':1}),dict(prediction_variances={'a':np.nan,'b':1}),dict(weights={'a':.5,'b':.9}),dict(weights={'a':1})]:
        args=dict(weights=w,residual_variances=w,prediction_variances=w,labeled_budget=20,unlabeled_budget=20)
        args.update(kwargs)
        with pytest.raises(ValueError): stratified_allocate(**args)


def test_confidence_mixture_matches_equations_20_21():
    f=np.array([.1,.5,.9]); c=np.array([.2,.4,.8]); lam=.6
    mu=np.mean(c-lam*f)
    expected=np.mean((1-c)*(0-lam*f-mu)**2+c*(1-lam*f-mu)**2)
    assert confidence_residual_variance(f,c,lam)==pytest.approx(expected)
    assert confidence_residual_variance(c,c,1)==pytest.approx(np.mean(c*(1-c)))
    with pytest.raises(ValueError): confidence_residual_variance(f,[.1,1.1,.2])


@pytest.mark.slow
def test_coverage_with_disproportionate_stratified_sampling():
    rng=np.random.default_rng(204)
    covered=np.zeros(2)
    estimates=[]
    for _ in range(250):
        strata={}
        for key,offset in [('low',.05),('high',.65)]:
            # Layer population means .2 and .8; sampling fractions .5/.5.
            p=rng.uniform(offset,offset+.3,1200)
            y=rng.binomial(1,p[:200])
            f=np.clip(p+(.15 if key=='low' else -.2),0,1)
            strata[key]=PPIData(y,f[:200],f[200:])
        data=StratifiedPPIData(strata,{'low':.8,'high':.2})
        for j,lam in enumerate([1.,'tuned']):
            r=StratPPI(lam=lam)(data,target=Mean())
            covered[j]+=r.confidence_interval[0]<=.32<=r.confidence_interval[1]
            if j==0: estimates.append(r.estimate)
    assert np.all((covered/250>.90)&(covered/250<.995)),covered
    assert np.mean(estimates)==pytest.approx(.32,abs=.005)
