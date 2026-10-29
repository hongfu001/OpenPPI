"""Inverse-propensity-weighted correction for actively labeled records (ASI, RASI, FTRL)."""
import numpy as np
from .base import InferenceMethod
from ..common.data import ActiveData, ActiveRegressionData
from ..common.targets import Mean, LinearRegression
from ..common.stats import ipw_correct


class IPW(InferenceMethod):
    """Real-valued mean inference from predictable Bernoulli queries.

    The corrected values are f + R/pi (Y - f); the estimate is their mean
    and the SE their standard deviation over sqrt(n). Rows labeled with
    probability 1 (a RASI burn-in) contribute their label. The record's
    design_info is copied into the diagnostics.
    """
    name = 'IPW'
    data_type = ActiveData

    def _estimate(self, data, target):
        values = ipw_correct(data.predictions, data.observed_labels,
                             data.queried, data.propensities)
        point = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        diagnostics = {
            'values': values,
            'audits': int(data.queried.sum()),
            'sum_sampling_probabilities': float(data.propensities.sum()),
            'min_pi': float(data.propensities.min()),
            'max_pi': float(data.propensities.max()),
            'prediction_weight': 1.0,
            'weight_strategy': 'fixed',
            'variance_design': 'active_terminal_plugin',
        }
        diagnostics.update(data.design_info)
        return point, se, diagnostics


class ActiveRegressionIPW(IPW):
    """ASI/RASI squared-loss M-estimator; retains IPW's existing mean path.

    For pseudo-response T=f+R/pi*(Y-f), solve mean[V(V beta-T)]=0.
    The sandwich uses full-pool regression scores, not pseudo-label variance.
    This terminal plugin requires iid pool rows, positive predictable queries
    and the papers' limiting-design regularity; not an anytime interval.
    """
    supported_targets=(Mean,LinearRegression)
    target_data_types={LinearRegression:ActiveRegressionData}

    def _estimate(self,data,target):
        if not isinstance(target,LinearRegression):
            return super()._estimate(data,target)
        from ..common.regression import design_matrix,checked_inverse,score,sandwich
        v=design_matrix(data.covariates,target)
        n,p=v.shape
        if n<=p or data.queried.sum()<2:
            raise ValueError('Active regression needs n > number of coefficients and at least two queries')
        inverse,condition=checked_inverse(v.T@v/n,target.max_condition,name='Active pool Gram matrix',positive=True)
        pseudo=ipw_correct(data.predictions,data.observed_labels,data.queried,data.propensities)
        beta=inverse@(v.T@pseudo/n)
        scores=score(v,pseudo,beta)
        cov=sandwich(inverse,scores)
        diagnostics=dict(data.design_info)
        diagnostics.update(covariance=cov,audits=int(data.queried.sum()),
            sum_sampling_probabilities=float(data.propensities.sum()),
            min_pi=float(data.propensities.min()),max_pi=float(data.propensities.max()),
            condition_number=condition,focus=target.focus,fit_intercept=target.fit_intercept,
            coefficient_names=(['intercept'] if target.fit_intercept else [])+
                [f'V{i+1}' for i in range(p-int(target.fit_intercept))],
            variance_design='active_regression_score_sandwich',prediction_weight=1.,weight_strategy='fixed',
            values={'pseudo_response':pseudo,'regression_scores':scores,'jacobian':v.T@v/n})
        return beta,np.sqrt(np.maximum(np.diag(cov),0)),diagnostics
