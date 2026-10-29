"""Passive and active coefficient simulation, dispatched by regression.py --target linear.

Target covariates V=X (five explanatory variables). The auxiliary predictor
uses Z=(X, judges). It is independently trained and fixed before inference.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from openppi import (LinearRegression, RegressionData, get_method, Uniform, Uncertainty,
                     RobustDesign, linear_regression_scores, select_for_labeling)
from openppi.sampling import select_burn_in
from dataset.regression_sim import RegressionDGP
from regression import fit_predictor, rng_for


METHODS=('OLS','PPI','PPI++','PSPA')
ACTIVE_METHODS=('Uniform+IPW','ASI','RASI')


def run_linear_experiment(config,noises=('homoskedastic','heteroskedastic'),*,fit_intercept=True,focus=1):
    c=config
    target=LinearRegression(fit_intercept=fit_intercept,focus=focus)
    if focus is None:
        raise ValueError('The active coefficient experiment needs a single focus coordinate')
    truth=RegressionDGP.projection_coefficients(fit_intercept)
    if focus is not None and focus >= len(truth):
        raise ValueError('focus is outside the regression design')
    rows=[]
    for noise in noises:
        case=('homoskedastic','heteroskedastic').index(noise)
        dgp=RegressionDGP(noise=noise)
        for rep in range(c.reps):
            train=dgp.draw(c.train_size,rng_for(c,case,rep,0))
            pool=dgp.draw(c.n,rng_for(c,case,rep,2))
            seed=int(rng_for(c,case,rep,3).integers(2**31))
            predict=fit_predictor(c,train,dgp,seed)
            f=predict(pool['features'])  # f uses Z, not just the target V.
            data=RegressionData(labels=pool['y'][:c.budget],labeled_predictions=f[:c.budget],
                unlabeled_predictions=f[c.budget:],labeled_covariates=pool['x'][:c.budget],
                unlabeled_covariates=pool['x'][c.budget:])
            def save(name,group,result,actual_labels,pilot_labels=0,burn_in_labels=0):
                low,high=result.confidence_interval
                training=c.train_size if c.predictor in ('rf','svm','mlp') else 0
                for j,param in enumerate(result.diagnostics['coefficient_names']):
                    error=result.estimate[j]-truth[j]
                    rows.append(dict(case=noise,group=group,rep=rep,method=name,
                        coordinate=j,parameter=param,is_focus=j==focus,truth=truth[j],
                        estimate=result.estimate[j],standard_error=result.standard_error[j],
                        low=low[j],high=high[j],width=high[j]-low[j],
                        covered=low[j]<=truth[j]<=high[j],error=error,sq_error=error**2,
                        actual_labels=actual_labels,training_labels=training,pilot_labels=pilot_labels,
                        burn_in_labels=burn_in_labels,total_labels=actual_labels+training+pilot_labels,
                        prediction_weight=json.dumps(result.diagnostics['prediction_weight']),
                        covariance_row=json.dumps(result.covariance[j].tolist())))

            for name in METHODS:
                result=get_method(name).inference(data,target=target,alpha=c.alpha)
                save(name,'given_labels',result,c.budget)

            # The absolute-error predictor sees independent pilot labels only.
            pilot=dgp.draw(c.pilot_size,rng_for(c,case,rep,1))
            def error_factory():
                return RandomForestRegressor(n_estimators=40,min_samples_leaf=12,max_depth=5,
                                             random_state=seed,n_jobs=1)
            errors=error_factory().fit(pilot['features'],np.abs(pilot['y']-predict(pilot['features'])))
            error_prediction=np.maximum(errors.predict(pool['features']),0.)
            scores=linear_regression_scores(pool['x'],error_prediction,target=target)
            for index,(name,design,u) in enumerate([
                ('Uniform+IPW',Uniform(),np.ones(c.n)),('ASI',Uncertainty(tau=.5),scores)]):
                req=select_for_labeling(f,target=target,design=design,budget=c.budget,
                    scores=u,covariates=pool['x'],random_state=rng_for(c,case,rep,40+index))
                result=get_method('ASI').inference(req.with_labels(pool['y'][req.indices]),target=target,alpha=c.alpha)
                save(name,'select_labels',result,len(req.indices),pilot_labels=c.pilot_size if name=='ASI' else 0)

            # RASI learns only from its own burn-in, then robustifies focused errors.
            burn=select_burn_in(c.n,size=c.burn_in,random_state=rng_for(c,case,rep,42))
            ids,rest=burn.indices,burn.remaining
            design=RobustDesign.fit(pool['features'][ids],pool['y'][ids],f[ids],
                covariates=pool['x'][ids],reference_covariates=pool['x'],target=target,
                error_model_factory=error_factory,rho=(c.budget-c.burn_in)/len(rest),
                constraint=c.robust_constraint)
            req=select_for_labeling(f[rest],target=target,design=design,budget=c.budget-c.burn_in,
                scores=design.errors(pool['features'][rest],pool['x'][rest]),covariates=pool['x'][rest],
                random_state=rng_for(c,case,rep,43))
            record=burn.combine(f,pool['y'][ids],req,pool['y'][rest[req.indices]],covariates=pool['x'])
            result=get_method('RASI').inference(record,target=target,alpha=c.alpha)
            save('RASI','select_labels',result,len(req.indices)+c.burn_in,burn_in_labels=c.burn_in)
    frame=pd.DataFrame(rows)
    summary=frame.groupby(['case','group','method','coordinate','parameter'],sort=False).agg(
        repetitions=('estimate','size'),truth=('truth','first'),coverage=('covered','mean'),
        mean_width=('width','mean'),bias=('error','mean'),mse=('sq_error','mean'),
        actual_labels=('actual_labels','mean'),training_labels=('training_labels','mean'),
        pilot_labels=('pilot_labels','mean'),burn_in_labels=('burn_in_labels','mean'),
        total_labels=('total_labels','mean')).reset_index()
    summary['rmse']=np.sqrt(summary.pop('mse'))
    summary['coverage_mc_se']=np.sqrt(summary.coverage*(1-summary.coverage)/summary.repetitions)
    return frame,summary
