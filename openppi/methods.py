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
from .inference import (BAI, IPW, ClusterRobustPPI, FABPPI, MisclassificationMLE, MultiPPI,
                        PSPA, RePPI, RoganGladen, StratPPI, TwoSample)
from .sampling import (BalancedDesign, FTRLDesign, RobustDesign, Uncertainty, multi_allocate,
                       stratified_allocate)


@dataclass(frozen=True)
class Recipe:
    name: str
    sampling: object
    inference: object
    data_type: type
    reference: str


METHODS = {recipe.name: recipe for recipe in (
    Recipe('PPI', None, TwoSample(lam=1.0), PPIData,
           'Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, Science 2023'),
    Recipe('PPI++', None, TwoSample(lam='tuned'), PPIData,
           'Angelopoulos, Duchi & Zrnic, Ann. Appl. Stat.'),
    Recipe('RG', None, RoganGladen(), PPIData,
           'Rogan & Gladen 1978; Chen, Lu, Li, Guo & Li, ICML 2026'),
    Recipe('MLE', None, MisclassificationMLE(), PPIData,
           'Chen, Lu, Li, Guo & Li, Efficient Inference for Noisy LLM-as-a-Judge Evaluation, ICML 2026'),
    Recipe('StratPPI', stratified_allocate, StratPPI(), StratifiedPPIData,
           'Fisch, Maynez, Hofer, Dhingra, Globerson & Cohen, NeurIPS 2024'),
    Recipe('PSPA', None, PSPA(), PPIData,
           'Miao, Miao, Wu, Zhao & Lu, Assumption-Lean and Data-Adaptive Post-Prediction Inference, JMLR 26(179):1–31, 2025'),
    Recipe('RePPI', None, RePPI(), PPIData,
           'Ji, Lei & Zrnic, Predictions as Surrogates, Biometrika'),
    Recipe('FAB-PPI', None, FABPPI(), PPIData,
           'Cortinovis & Caron, ICML 2025'),
    Recipe('Cluster-PPI', None, ClusterRobustPPI(), ClusteredPPIData,
           'Broska & Howes, arXiv 2026'),
    Recipe('MultiPPI', multi_allocate, MultiPPI(), MultiPPIData,
           'Cowen-Breen, Agarwal, Bates, Cohen, Eisenstein, Globerson & Fisch, ICLR 2026'),
    Recipe('ASI', partial(Uncertainty, tau=.5), IPW(), ActiveData,
           'Zrnic & Candès, ICML 2024'),
    Recipe('RASI', RobustDesign, IPW(), ActiveData,
           'Li, Zrnic & Candès, NeurIPS 2025'),
    Recipe('FTRL', FTRLDesign, IPW(), ActiveData,
           'Sfyraki & Wang, ICLR 2026'),
    Recipe('BAI', BalancedDesign, BAI(), BalancedActiveData,
           'Chen, Zhou, Peng & Wang, NeurIPS 2025'),
)}


def get_method(name):
    """The Recipe registered under name, e.g. get_method('ASI')."""
    if name not in METHODS:
        raise KeyError(f'Unknown method {name!r}; choose from {sorted(METHODS)}')
    return METHODS[name]
