"""OpenPPI: an open-source framework for prediction-powered inference.

A method is a sampling component plus an inference component; openppi.methods
records the published combinations (get_method('ASI'), METHODS).
"""
from .common import (ActiveData, BalancedActiveData, ClusteredPPIData, InferenceResult, Mean, MultiPPIData,
                     PPIData, Target, split)
from .inference import (BAI, IPW, BinnedMean, ClusterRobustPPI, FABPPI, InferenceMethod, MultiPPI, RePPI,
                        TwoSample)
from .methods import METHODS, Recipe, get_method
from .sampling import (Allocation, BalancedDesign, BurnIn, FTRLDesign, LabelRequest, RobustDesign,
                       SamplingDesign, Uncertainty, Uniform, multi_allocate, select_for_labeling, tune_tau)

__all__ = [
    # data, targets and results
    'ActiveData', 'BalancedActiveData', 'ClusteredPPIData', 'MultiPPIData', 'PPIData', 'split', 'Mean', 'Target',
    'InferenceResult',
    # sampling components
    'SamplingDesign', 'LabelRequest', 'select_for_labeling', 'Uniform', 'Uncertainty', 'tune_tau',
    'BalancedDesign', 'FTRLDesign', 'RobustDesign', 'BurnIn', 'Allocation', 'multi_allocate',
    # inference components
    'InferenceMethod', 'IPW', 'TwoSample', 'RePPI', 'BinnedMean', 'FABPPI', 'ClusterRobustPPI', 'MultiPPI', 'BAI',
    # method registry
    'METHODS', 'Recipe', 'get_method',
]
