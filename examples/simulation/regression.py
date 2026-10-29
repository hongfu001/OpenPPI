#!/usr/bin/env python3
"""Fixed-predictor continuous mean experiments; binary coverage.py is unchanged.

Run: python examples/simulation/regression.py --reps 200 --out /tmp/regression_results
The estimand is the superpopulation mean 1, never the realized pool mean.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from openppi import (BAI, IPW, Mean, PPIData, StratPPI, StratifiedPPIData, TwoSample,
                     Uniform, Uncertainty, BalancedDesign, FTRLDesign, RobustDesign,
                     get_method, select_for_labeling, stratified_allocate)
from openppi.sampling import select_burn_in
from config import Config
from dataset.regression_sim import RegressionDGP
from model.learners import RegressionUncertainty, SimulationLearner


PASSIVE = ('Label-only','PPI','PPI++','PSPA','RePPI','CalPPI','LinearCalPPI')
ACTIVE = ('Uniform+IPW','ASI','RASI','BAI','FTRL','StratPPI')


@dataclass(frozen=True)
class RegressionConfig:
    reps: int = 200
    n: int = 2000
    train_size: int = 400
    pilot_size: int = 200
    budget: int = 400
    burn_in: int = 100
    predictor: str = 'rf'
    seed: int = 20261029
    alpha: float = .05
    robust_constraint: float = .03

    def __post_init__(self):
        for name in ('reps','n','train_size','pilot_size','budget','burn_in','seed'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value < (0 if name=='seed' else 1):
                raise ValueError(f'{name} must be a valid integer')
        if not 12 <= self.budget <= self.n-12 or self.pilot_size < 20 or self.train_size < 20:
            raise ValueError('Require 12 <= budget <= n-12 and training/pilot sizes >= 20')
        if not 10 <= self.burn_in < self.budget:
            raise ValueError('Require 10 <= burn_in < budget')
        if self.predictor not in ('rf','svm','mlp','judge','oracle'):
            raise ValueError('Unknown predictor')
        if not 0 < self.alpha < 1 or not np.isfinite(self.robust_constraint) or self.robust_constraint < 0:
            raise ValueError('Invalid alpha or robust_constraint')


def rng_for(c, case, rep, stream):
    return np.random.default_rng(np.random.SeedSequence([c.seed,case,rep,stream]))


def run_rep(c, noise, rep):
    case=('homoskedastic','heteroskedastic').index(noise)
    dgp=RegressionDGP(noise=noise)
    train=dgp.draw(c.train_size,rng_for(c,case,rep,0))
    pilot=dgp.draw(c.pilot_size,rng_for(c,case,rep,1))
    pool=dgp.draw(c.n,rng_for(c,case,rep,2))
    seed=int(rng_for(c,case,rep,3).integers(2**31))
    if c.predictor=='oracle':
        predict=lambda x: dgp.conditional_mean(x[:,:5])
    elif c.predictor=='judge':
        predict=lambda x: x[:,5:].mean(axis=1)
    else:
        model=SimulationLearner(Config(learner=c.predictor,outcome_type='continuous',initial_prediction=1.),seed)
        model.update(train['features'],train['y'],np.ones(c.train_size))
        predict=model.predict
    x,y=pool['features'],pool['y']
    f,pilot_f=predict(x),predict(pilot['features'])
    target=Mean(outcome_type='continuous')
    record=PPIData(y[:c.budget],f[:c.budget],f[c.budget:])
    rows=[]

    def save(method, group, result, inference_labels, *, pilot_labels=0, burn_labels=0):
        low,high=result.confidence_interval
        training=c.train_size if c.predictor in ('rf','svm','mlp') else 0
        rows.append(dict(case=noise,rep=rep,group=group,method=method,truth=1.,
                         estimate=result.estimate,standard_error=result.standard_error,
                         low=low,high=high,width=high-low,covered=low<=1.<=high,
                         error=result.estimate-1.,sq_error=(result.estimate-1.)**2,
                         inference_labels=inference_labels,training_labels=training,
                         pilot_labels=pilot_labels,burn_in_labels=burn_labels,
                         actual_labels=inference_labels+burn_labels,
                         total_labels=training+pilot_labels+inference_labels+burn_labels))

    for name in PASSIVE:
        estimator=(TwoSample(lam=0) if name=='Label-only' else
                   get_method(name).inference if name!='RePPI' else
                   type(get_method(name).inference)(random_state=seed))
        save(name,'given_labels',estimator(record,target=target,alpha=c.alpha),c.budget)

    # Error learner never sees predictor-training or inference-pool outcomes.
    error=RegressionUncertainty(random_state=seed).fit(pilot['features'],pilot['y'],pilot_f)
    squared=error.predict(x)
    for index,(name,design,scores) in enumerate([
        ('Uniform+IPW',Uniform(),np.ones(c.n)),
        ('ASI',Uncertainty(tau=.5),np.sqrt(squared)),
        ('BAI',BalancedDesign(tau=.5,fixed_size=True),np.sqrt(squared)),
        ('FTRL',FTRLDesign(),squared),
    ]):
        request=select_for_labeling(f,target=target,design=design,budget=c.budget,
                                   scores=scores,random_state=rng_for(c,case,rep,10+index))
        labeled=request.with_labels(y[request.indices])
        result=(BAI() if name=='BAI' else IPW())(labeled,target=target,alpha=c.alpha)
        save(name,'select_labels',result,len(request.indices),
             pilot_labels=0 if name=='Uniform+IPW' else c.pilot_size)

    # RASI retains its own absolute-error burn-in and robust design.
    burn=select_burn_in(c.n,size=c.burn_in,random_state=rng_for(c,case,rep,20))
    rest=burn.remaining
    design=RobustDesign.fit(x[burn.indices],y[burn.indices],f[burn.indices],
        error_model_factory=lambda: RandomForestRegressor(n_estimators=40,min_samples_leaf=10,
                                                           max_depth=5,random_state=seed,n_jobs=1),
        rho=(c.budget-c.burn_in)/len(rest),constraint=c.robust_constraint)
    request=select_for_labeling(f[rest],target=target,design=design,budget=c.budget-c.burn_in,
                               scores=design.errors(x[rest]),random_state=rng_for(c,case,rep,21))
    labeled=burn.combine(f,y[burn.indices],request,y[rest[request.indices]])
    save('RASI','select_labels',IPW()(labeled,target=target,alpha=c.alpha),len(request.indices),burn_labels=c.burn_in)

    # Independent stratified samples, planned from the independent pilot.
    # sign(X1) gives exactly known population probabilities .5/.5.
    weights={False:.5,True:.5}
    mask_p=pilot['x'][:,0]>=0
    residual_variances={}
    prediction_variances={}
    for key in weights:
        mask=mask_p==key
        if mask.sum()<2:
            raise ValueError('Pilot needs at least two rows per stratum; increase pilot_size')
        residual_variances[key]=float(np.var(pilot['y'][mask]-pilot_f[mask],ddof=1))
        prediction_variances[key]=float(np.var(pilot_f[mask],ddof=1))
    plan=stratified_allocate(weights,residual_variances,prediction_variances,
                            labeled_budget=c.budget,unlabeled_budget=c.n-c.budget)
    strata={}
    for key in weights:
        nl,nu=plan.labeled_counts[key],plan.unlabeled_counts[key]
        # Fresh conditional draws avoid treating the candidate pool's random
        # stratum proportions as known, or capping counts after observing Y.
        chunks=[]; available=0
        rng=rng_for(c,case,rep,30+int(key))
        while available<nl+nu:
            batch=dgp.draw(max(100,3*(nl+nu-available)),rng)
            mask=(batch['x'][:,0]>=0)==key
            chunks.append((batch['features'][mask],batch['y'][mask]))
            available+=int(mask.sum())
        sx=np.concatenate([a for a,b in chunks])[:nl+nu]
        sy=np.concatenate([b for a,b in chunks])[:nl]
        sf=predict(sx)
        strata[key]=PPIData(sy,sf[:nl],sf[nl:])
    # Lambda=1 was fixed before pilot planning, per Proposition 3.
    result=StratPPI(lam=1.)(StratifiedPPIData(strata,weights),target=target,alpha=c.alpha)
    save('StratPPI','select_labels',result,c.budget,pilot_labels=c.pilot_size)
    return rows


def summarize(frame):
    result=frame.groupby(['case','group','method'],sort=False).agg(
        repetitions=('estimate','size'),coverage=('covered','mean'),
        mean_width=('width','mean'),bias=('error','mean'),mse=('sq_error','mean'),
        mean_actual_labels=('actual_labels','mean'),training_labels=('training_labels','mean'),
        pilot_labels=('pilot_labels','mean'),burn_in_labels=('burn_in_labels','mean'),
        total_labels=('total_labels','mean')).reset_index()
    result['rmse']=np.sqrt(result.pop('mse'))
    result['coverage_mc_se']=np.sqrt(result.coverage*(1-result.coverage)/result.repetitions)
    return result


def run_experiment(c, noises=('homoskedastic','heteroskedastic')):
    if not noises or any(n not in ('homoskedastic','heteroskedastic') for n in noises):
        raise ValueError('Select at least one supported noise case')
    rows=[]
    for noise in noises:
        for rep in range(c.reps):
            rows.extend(run_rep(c,noise,rep))
    frame=pd.DataFrame(rows)
    return frame,summarize(frame)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name,field in RegressionConfig.__dataclass_fields__.items():
        parser.add_argument('--'+name.replace('_','-'),type=type(field.default),default=field.default)
    parser.add_argument('--noise',choices=('both','homoskedastic','heteroskedastic'),default='both')
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results_regression')
    args=parser.parse_args()
    c=RegressionConfig(**{key:getattr(args,key) for key in RegressionConfig.__dataclass_fields__})
    noises=('homoskedastic','heteroskedastic') if args.noise=='both' else (args.noise,)
    frame,summary=run_experiment(c,noises)
    args.out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.out/'per_rep.csv',index=False)
    summary.to_csv(args.out/'summary.csv',index=False)
    (args.out/'config.json').write_text(json.dumps({**asdict(c),'noises':noises,'truth':1.,'outcome_type':'continuous'},indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':
    main()
