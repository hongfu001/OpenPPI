"""OpenPPI: an open-source framework for prediction-powered inference.

A method is a sampling component plus an inference component; openppi.methods
records the published combinations (get_method('ASI'), METHODS).
"""
from .common import (ActiveData, ActiveRegressionData, BalancedActiveData, ClusteredPPIData, InferenceResult, Mean, MultiPPIData,
                     PPIData, RegressionData, LinearRegression, StratifiedPPIData, Target, split)
from .inference import (BAI, IPW, ActiveRegressionIPW, BinnedMean, CalibratedPPI, ClusterRobustPPI, FABPPI, InferenceMethod,
                        MisclassificationMLE, MultiPPI, OLS, PSPA, RePPI, RoganGladen, StratPPI, TwoSample)
from .methods import METHODS, Recipe, get_method
from .sampling import (Allocation, BalancedDesign, BurnIn, FTRLDesign, LabelRequest, RobustDesign,
                       SamplingDesign, StratifiedAllocation, Uncertainty, Uniform,
                       RegressionLabelRequest, RegressionRobustDesign, linear_regression_scores,
                       confidence_residual_variance, multi_allocate, select_for_labeling,
                       stratified_allocate, tune_tau)

__all__ = [
    # data, targets and results
    'ActiveData', 'ActiveRegressionData', 'BalancedActiveData', 'ClusteredPPIData', 'MultiPPIData', 'PPIData', 'split', 'Mean', 'Target',
    'InferenceResult', 'RegressionData', 'LinearRegression', 'StratifiedPPIData',
    # sampling components
    'SamplingDesign', 'LabelRequest', 'select_for_labeling', 'Uniform', 'Uncertainty', 'tune_tau',
    'BalancedDesign', 'FTRLDesign', 'RobustDesign', 'BurnIn', 'Allocation', 'multi_allocate',
    'StratifiedAllocation', 'stratified_allocate', 'confidence_residual_variance',
    # inference components
    'ActiveRegressionIPW', 'InferenceMethod', 'IPW', 'TwoSample', 'RePPI', 'BinnedMean', 'FABPPI', 'ClusterRobustPPI',
    'OLS', 'CalibratedPPI', 'MultiPPI', 'BAI', 'StratPPI', 'RoganGladen', 'MisclassificationMLE', 'PSPA',
    # method registry
    'RegressionLabelRequest', 'RegressionRobustDesign', 'linear_regression_scores', 'METHODS', 'Recipe', 'get_method',
]
