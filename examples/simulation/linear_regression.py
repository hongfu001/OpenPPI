"""Passive coefficient simulation, dispatched by regression.py --target linear.

Target covariates V=X (five explanatory variables). The auxiliary predictor
uses Z=(X, judges). It is independently trained and fixed before inference.
"""
import json
import numpy as np
import pandas as pd

from openppi import LinearRegression, RegressionData, get_method
from dataset.regression_sim import RegressionDGP
from regression import fit_predictor, rng_for


METHODS=('OLS','PPI','PPI++','PSPA')


def run_linear_experiment(config,noises=('homoskedastic','heteroskedastic'),*,fit_intercept=True,focus=1):
    c=config
    target=LinearRegression(fit_intercept=fit_intercept,focus=focus)
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
            for name in METHODS:
                result=get_method(name).inference(data,target=target,alpha=c.alpha)
                low,high=result.confidence_interval
                training=c.train_size if c.predictor in ('rf','svm','mlp') else 0
                for j,param in enumerate(result.diagnostics['coefficient_names']):
                    error=result.estimate[j]-truth[j]
                    rows.append(dict(case=noise,group='given_labels',rep=rep,method=name,
                        coordinate=j,parameter=param,is_focus=j==focus,truth=truth[j],
                        estimate=result.estimate[j],standard_error=result.standard_error[j],
                        low=low[j],high=high[j],width=high[j]-low[j],
                        covered=low[j]<=truth[j]<=high[j],error=error,sq_error=error**2,
                        actual_labels=c.budget,training_labels=training,pilot_labels=0,
                        burn_in_labels=0,total_labels=c.budget+training,
                        prediction_weight=json.dumps(result.diagnostics['prediction_weight']),
                        covariance_row=json.dumps(result.covariance[j].tolist())))
    frame=pd.DataFrame(rows)
    summary=frame.groupby(['case','method','coordinate','parameter'],sort=False).agg(
        repetitions=('estimate','size'),truth=('truth','first'),coverage=('covered','mean'),
        mean_width=('width','mean'),bias=('error','mean'),mse=('sq_error','mean'),
        actual_labels=('actual_labels','mean'),training_labels=('training_labels','mean'),
        total_labels=('total_labels','mean')).reset_index()
    summary['rmse']=np.sqrt(summary.pop('mse'))
    summary['coverage_mc_se']=np.sqrt(summary.coverage*(1-summary.coverage)/summary.repetitions)
    return frame,summary
