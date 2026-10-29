"""Shared least-squares scores, validated design matrices and sandwich algebra."""
import numpy as np


def checked_inverse(matrix, max_condition, *, name='Jacobian', positive=False):
    rank=int(np.linalg.matrix_rank(matrix))
    condition=float(np.linalg.cond(matrix))
    if rank != len(matrix) or not np.isfinite(condition) or condition > max_condition:
        raise ValueError(f'{name} is rank deficient or ill-conditioned (rank={rank}, condition={condition:g})')
    if positive and np.linalg.eigvalsh((matrix+matrix.T)/2).min() <= 0:
        raise ValueError(f'{name} must be positive definite for the squared-loss minimum')
    return np.linalg.solve(matrix,np.eye(len(matrix))),condition


def designs(data,target):
    x,z=data.labeled_covariates,data.unlabeled_covariates
    if target.fit_intercept:
        x=np.column_stack([np.ones(len(x)),x])
        z=np.column_stack([np.ones(len(z)),z])
    p=x.shape[1]
    if p == 0 or target.focus is not None and target.focus >= p:
        raise ValueError('focus must index the design after intercept insertion, and the design must have columns')
    if len(x) <= p:
        raise ValueError('Need more labeled observations than regression coefficients')
    q=x.T@x/len(x)
    inverse,condition=checked_inverse(q,target.max_condition,name='Labeled design Gram matrix',positive=True)
    return x,z,q,inverse,condition


def score(x,y,beta):
    return x*(x@beta-y)[:,None]


def covariance(rows):
    centered=rows-rows.mean(axis=0)
    return centered.T@centered/(len(rows)-1)


def sandwich(inverse,labeled_scores,unlabeled_scores=None):
    meat=covariance(labeled_scores)/len(labeled_scores)
    if unlabeled_scores is not None:
        meat+=covariance(unlabeled_scores)/len(unlabeled_scores)
    cov=inverse@meat@inverse.T
    return (cov+cov.T)/2


def diagnostics(x,z,target,condition,cov,**extra):
    return dict(covariance=cov,labeled_count=len(x),unlabeled_count=len(z),
                focus=target.focus,fit_intercept=target.fit_intercept,
                coefficient_names=(['intercept'] if target.fit_intercept else [])+
                    [f'V{i+1}' for i in range(x.shape[1]-int(target.fit_intercept))],
                condition_number=condition,variance_design='independent_regression_sandwich',**extra)
