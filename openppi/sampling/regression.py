"""Focused ASI scores and coefficient-specific robust RASI sampling."""
import numpy as np

from ..common.targets import LinearRegression
from ..common.regression import design_matrix,focused_direction,checked_inverse
from .robust import (RobustDesign,DEFAULT_CONSTRAINTS,_fit_error_model,_predict_errors,
                     active_rule,budget_path,robust_rho)


def linear_regression_scores(covariates,error_estimates,*,target,reference_covariates=None):
    """ASI eq. 11/Section 7.2: |V' Q^-1 e_j| times predicted |Y-f|.

    Covariates V exclude an intercept if the target inserts it. Errors must
    be label-free predictions of absolute error (or a declared RMS proxy),
    never true residuals from the rows about to be queried. The optional
    reference pool is covariate-only; by default use the current full pool.
    """
    if not isinstance(target,LinearRegression):
        raise TypeError('Require a LinearRegression target')
    reference=covariates if reference_covariates is None else reference_covariates
    h=focused_direction(reference,target)
    v=design_matrix(covariates,target)
    errors=np.asarray(error_estimates,dtype=float)
    if errors.shape!=(len(v),) or not np.isfinite(errors).all() or (errors<0).any():
        raise ValueError('Error estimates must be finite, nonnegative and aligned with covariates')
    if v.shape[1]!=len(h):
        raise ValueError('Reference and candidate covariates must have the same columns')
    return np.abs(v@h)*errors


class RegressionRobustDesign(RobustDesign):
    """RASI Section 3: robust path on coefficient-specific gradient errors.

    Fit the response absolute-error model on burn-in rows only; multiply its
    predictions by the covariate-only focused influence multiplier exactly.
    Optional radius CV uses held-out coefficient scores and independent
    error-model fits in each fold. The original mean design is unchanged.
    """
    def __init__(self,error_model,*,target,direction,constraint,path='geometric',grid=101,floor=1e-3):
        super().__init__(error_model,constraint=constraint,path=path,grid=grid,floor=floor)
        self.target=target
        self.direction=np.array(direction,dtype=float,copy=True)
        self.direction.setflags(write=False)

    @classmethod
    def fit(cls,features,labels,predictions,*,covariates,reference_covariates,target,
            error_model_factory,rho,constraint=None,candidates=DEFAULT_CONSTRAINTS,
            folds=5,path='geometric',grid=101,random_state=None):
        if not isinstance(target,LinearRegression):
            raise TypeError('Require a LinearRegression target')
        x=np.asarray(features,dtype=float)
        y,p=np.asarray(labels,dtype=float),np.asarray(predictions,dtype=float)
        v=design_matrix(covariates,target)
        if x.ndim!=2 or len(x)!=len(v) or y.shape!=(len(v),) or p.shape!=y.shape or not all(np.isfinite(a).all() for a in (x,y,p)):
            raise ValueError('Require aligned finite burn-in features, target covariates, labels and predictions')
        if not callable(error_model_factory) or not 0<rho<1:
            raise ValueError('Require error_model_factory and a sampling rate in (0,1)')
        direction=focused_direction(reference_covariates,target)
        if v.shape[1]!=len(direction):
            raise ValueError('Reference and burn-in covariates must have the same columns')
        if constraint is None:
            values=np.asarray(candidates,dtype=float)
            if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or (values<0).any():
                raise ValueError('candidates must be finite nonnegative radii')
            if isinstance(folds,bool) or not isinstance(folds,int) or folds<2 or len(y)<2*folds:
                raise ValueError('Require folds >= 2 and at least two burn-in rows per fold')
            parts=np.array_split(np.random.default_rng(random_state).permutation(len(y)),folds)
            validation=[]
            for k,test in enumerate(parts):
                train=np.concatenate([part for j,part in enumerate(parts) if j!=k])
                if len(train)<=v.shape[1]:
                    raise ValueError('Each CV training fold needs more rows than regression coefficients')
                inv,_=checked_inverse(v[train].T@v[train]/len(train),target.max_condition,
                                      name='Burn-in CV Gram matrix',positive=True)
                beta=inv@(v[train].T@y[train]/len(train))
                model=_fit_error_model(error_model_factory,x[train],y[train],p[train])
                multipliers=v[test]@direction
                errors=np.abs(multipliers)*_predict_errors(model,x[test])
                # Coefficient influence scores at a cross-fitted pilot beta.
                human=multipliers*(y[test]-v[test]@beta)
                imputed=multipliers*(p[test]-v[test]@beta)
                validation.append((errors,active_rule(errors,rho),human,imputed))
            risks=[]
            for c in values:
                risk=0.
                for errors,base,human,imputed in validation:
                    t=robust_rho(errors,base,rho,constraint=float(c),path=path,grid=grid)
                    pi=budget_path(base,rho,t,path)
                    second=np.mean(human**2+(human-imputed)**2*(1/pi-1))
                    risk+=np.sqrt(max(second-human.mean()**2,0.))
                risks.append(risk)
            constraint=float(values[int(np.argmin(risks))])
        if not np.isfinite(constraint) or constraint<0:
            raise ValueError('constraint must be finite and nonnegative')
        model=_fit_error_model(error_model_factory,x,y,p)
        return cls(model,target=target,direction=direction,constraint=constraint,path=path,grid=grid)

    def errors(self,features,covariates=None):
        if covariates is None:
            raise ValueError('Regression robust scores require target covariates')
        v=design_matrix(covariates,self.target)
        if v.shape[1]!=len(self.direction):
            raise ValueError('Covariate dimension must agree with the fitted reference')
        errors=_predict_errors(self.error_model,np.asarray(features,dtype=float))
        if errors.shape!=(len(v),):
            raise ValueError('Features and target covariates must align')
        return np.abs(v@self.direction)*errors
