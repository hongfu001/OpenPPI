"""Registry of methods: each published method = a sampling component + an inference component.

This file only records the composition; all computation lives in sampling/ and inference/.
`sampling` is None for two-sample methods (the labeled and unlabeled samples are split in
advance), otherwise a class, factory or function that builds a fresh design each time,
so no state is shared between uses. `inference` is a configured, stateless estimator.
"""
from dataclasses import dataclass
from functools import partial

from .common.data import (ActiveData, BalancedActiveData, ClusteredPPIData, MultiPPIData,
                          PPIData, StratifiedPPIData)
from .inference import (BAI, IPW, ActiveRegressionIPW, CalibratedPPI, ClusterRobustPPI, FABPPI, MisclassificationMLE, MultiPPI,
                        OLS, PSPA, RePPI, RoganGladen, StratPPI, TwoSample)
from .sampling import (BalancedDesign, FTRLDesign, RobustDesign, Uncertainty, multi_allocate,
                       stratified_allocate)


@dataclass(frozen=True)
class Recipe:
    name: str
    sampling: object
    inference: object
    data_type: type
    reference: str

    @property
    def supported_targets(self):
        return self.inference.supported_targets

    @property
    def target_data_types(self):
        overrides = getattr(self.inference, 'target_data_types', {})
        return {target: overrides.get(target,self.data_type) for target in self.supported_targets}


METHODS = {recipe.name: recipe for recipe in (
    Recipe('OLS', None, OLS(), OLS.data_type, 'Ordinary least squares'),
    Recipe('PPI', None, TwoSample(lam=1.0), PPIData,
           'Prediction-Powered Inference'),
    Recipe('PPI++', None, TwoSample(lam='tuned'), PPIData,
           'PPI++: Efficient Prediction-Powered Inference'),
    Recipe('RG', None, RoganGladen(), PPIData,
           'Efficient Inference for Noisy LLM-as-a-Judge Evaluation'),
    Recipe('MLE', None, MisclassificationMLE(), PPIData,
           'Efficient Inference for Noisy LLM-as-a-Judge Evaluation'),
    Recipe('CalPPI', None, CalibratedPPI(), PPIData,
           'Calibeating Prediction-Powered Inference'),
    Recipe('LinearCalPPI', None, CalibratedPPI(calibration='linear'), PPIData,
           'Calibeating Prediction-Powered Inference'),
    Recipe('StratPPI', stratified_allocate, StratPPI(), StratifiedPPIData,
           'Stratified Prediction-Powered Inference for Effective Hybrid Evaluation of Language Models'),
    Recipe('PSPA', None, PSPA(), PPIData,
           'Assumption-Lean and Data-Adaptive Post-Prediction Inference'),
    Recipe('RePPI', None, RePPI(), PPIData,
           'Predictions as Surrogates: Revisiting Surrogate Outcomes in the Age of AI'),
    Recipe('FAB-PPI', None, FABPPI(), PPIData,
           'FAB-PPI: Frequentist, Assisted by Bayes, Prediction-Powered Inference'),
    Recipe('Cluster-PPI', None, ClusterRobustPPI(), ClusteredPPIData,
           'Cluster-Robust Prediction-Powered Inference'),
    Recipe('MultiPPI', multi_allocate, MultiPPI(), MultiPPIData,
           'Multiple-Prediction-Powered Inference'),
    Recipe('ASI', partial(Uncertainty, tau=.5), ActiveRegressionIPW(), ActiveData,
           'Active Statistical Inference'),
    Recipe('RASI', RobustDesign, ActiveRegressionIPW(), ActiveData,
           'Robust Sampling for Active Statistical Inference'),
    Recipe('FTRL', FTRLDesign, IPW(), ActiveData,
           'Revisiting Active Sequential Prediction-Powered Mean Estimation'),
    Recipe('BAI', BalancedDesign, BAI(), BalancedActiveData,
           'Balanced Active Inference'),
)}


def get_method(name):
    """The Recipe registered under name, e.g. get_method('ASI')."""
    if name not in METHODS:
        raise KeyError(f'Unknown method {name!r}; choose from {sorted(METHODS)}')
    return METHODS[name]
