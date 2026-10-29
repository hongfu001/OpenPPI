"""Linear-regression M-estimators; target-specific implementations only."""
import numpy as np

from .base import InferenceMethod
from ..common.data import RegressionData
from ..common.targets import LinearRegression
from ..common.regression import designs, checked_inverse, score, covariance, sandwich, diagnostics


class OLS(InferenceMethod):
    """Labeled-only least squares with heteroskedasticity-robust sandwich SE."""
    name='OLS'
    data_type=RegressionData
    supported_targets=(LinearRegression,)

    def _estimate(self,data,target):
        x,z,q,inverse,condition=designs(data,target)
        beta=inverse@(x.T@data.labels/len(x))
        cov=sandwich(inverse,score(x,data.labels,beta))
        return beta,np.sqrt(np.maximum(np.diag(cov),0)),diagnostics(x,z,target,condition,cov,prediction_weight=0.)


def two_sample_regression(data,target,lam):
    """Solve the prediction-powered squared-loss equation, with focused tuning.

    Unlike the reference package's sum-of-two-OLS implementation, this solves
    the rectified loss exactly. They share a population target but can differ
    in finite samples. Tuning uses a consistent labeled OLS pilot, pooled score
    variance and pooled Gram matrix; [0,1] clipping preserves convexity.
    """
    x,z,q,inv_l,condition_l=designs(data,target)
    n,big_n=len(x),len(z)
    qu=z.T@z/big_n
    pooled=(n*q+big_n*qu)/(n+big_n)
    beta0=inv_l@(x.T@data.labels/n)
    if lam == 'tuned':
        inv_pool,_=checked_inverse(pooled,target.max_condition,name='Pooled Gram matrix',positive=True)
        gy=score(x,data.labels,beta0)@inv_pool.T
        gf=score(x,data.labeled_predictions,beta0)@inv_pool.T
        gu=score(z,data.unlabeled_predictions,beta0)@inv_pool.T
        if target.focus is not None:
            gy,gf,gu=(a[:,[target.focus]] for a in (gy,gf,gu))
        numerator=np.sum((gy-gy.mean(0))*(gf-gf.mean(0)))/(n-1)
        denominator=(1+n/big_n)*np.trace(covariance(np.vstack([gf,gu])))
        weight=float(np.clip(numerator/denominator,0,1)) if denominator>0 else 0.
    else:
        weight=float(lam)
    a=(1-weight)*q+weight*qu
    inverse,condition=checked_inverse(a,target.max_condition,name='Rectified loss Hessian',positive=True)
    rhs=x.T@data.labels/n+weight*(z.T@data.unlabeled_predictions/big_n-x.T@data.labeled_predictions/n)
    beta=inverse@rhs
    sl=score(x,data.labels,beta)-weight*score(x,data.labeled_predictions,beta)
    su=weight*score(z,data.unlabeled_predictions,beta)
    cov=sandwich(inverse,sl,su)
    return beta,np.sqrt(np.maximum(np.diag(cov),0)),diagnostics(
        x,z,target,condition,cov,prediction_weight=weight,
        weight_strategy='focused_tuned_clipped' if lam=='tuned' else 'fixed',
        labeled_condition_number=condition_l,estimation_equation='rectified_squared_loss')


def pspa_regression(data,target):
    """Official PSPA OLS one-step point and weights; symmetric score covariance.

    The reference R Sigma_cal cross-term is not a symmetric matrix in general.
    Cov(actual weighted scores) supplies both transpose cross-terms and a PSD
    sandwich instead of copying that expression or silently symmetrizing it.
    """
    x,z,q,inv_l,condition_l=designs(data,target)
    n,big_n=len(x),len(z)
    pooled=(x.T@x+z.T@z)/(n+big_n)
    inverse,condition=checked_inverse(pooled,target.max_condition,name='Pooled Gram matrix',positive=True)
    beta0=inv_l@(x.T@data.labels/n)
    gy,gf,gu=score(x,data.labels,beta0),score(x,data.labeled_predictions,beta0),score(z,data.unlabeled_predictions,beta0)
    cross=(gy-gy.mean(0)).T@(gf-gf.mean(0))/(n-1)
    numerator=np.diag(inv_l@cross@inv_l.T)
    denominator=np.diag(inv_l@(covariance(gf)+n/big_n*covariance(gu))@inv_l.T)
    weights=np.zeros(len(beta0))
    np.divide(numerator,denominator,out=weights,where=denominator>0)
    weights=np.clip(weights,0,1)
    beta=beta0-inverse@(gy.mean(0)+weights*(gu.mean(0)-gf.mean(0)))
    sl=score(x,data.labels,beta)-weights*score(x,data.labeled_predictions,beta)
    su=weights*score(z,data.unlabeled_predictions,beta)
    cov=sandwich(inverse,sl,su)
    return beta,np.sqrt(np.maximum(np.diag(cov),0)),diagnostics(
        x,z,target,condition,cov,prediction_weight=weights.tolist(),
        weight_strategy='pspa_coordinate_weights',estimation_equation='pspa_one_step',
        labeled_condition_number=condition_l,covariance_correction='actual_score_cross_terms')
