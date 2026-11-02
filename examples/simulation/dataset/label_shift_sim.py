"""Separate source/target DGPs satisfying P(X|Y)=Q(X|Y) by construction."""
import numpy as np
from scipy.special import expit, logit

from openppi import LabelShiftData, PPIData


def make_label_shift_case(n=300, m=1500, *, source_prevalence=.3,
                          target_prevalence=.7, separation=1., seed=0):
    """Binary judge case with known source posterior and shifted target prevalence.

    Y_source ~ Bernoulli(p), Y_target ~ Bernoulli(q),
    X|Y ~ Normal(separation*(2Y-1), 1) in BOTH domains.
    f(X)=P_source(Y=1|X) is a fixed Bayes predictor from the specified DGP,
    not an estimated target oracle. This isolates label-shift correction from
    predictor-training errors. Returns (shift_data, standard_ppi_data, truth).
    Target Y is used only to generate X and is never returned to inference.
    """
    for count in (n, m):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 2:
            raise ValueError('n and m must be integer counts >= 2')
    if not 0 < source_prevalence < 1 or not 0 < target_prevalence < 1:
        raise ValueError('Source and target prevalences must lie in (0, 1)')
    if not np.isfinite(separation) or separation <= 0:
        raise ValueError('separation must be finite and positive')
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, source_prevalence, n).astype(float)
    target_y = rng.binomial(1, target_prevalence, m)
    x = separation * (2 * y - 1) + rng.normal(size=n)
    z = separation * (2 * target_y - 1) + rng.normal(size=m)
    f, fu = expit(2 * separation * x + logit(source_prevalence)), expit(2 * separation * z + logit(source_prevalence))
    probabilities = np.column_stack([1 - np.r_[f, fu], np.r_[f, fu]])
    return (LabelShiftData(y, x[:, None], z[:, None], probabilities),
            PPIData(y, f, fu), float(target_prevalence))


def make_continuous_label_shift_case(n=250, m=250, *, seed=0):
    """Section 6's continuous normal-shift DGP, with a source Bayes mean for PPI.

    Source Y~N(0,2), target Y~N(1,1), X|Y~N_3(alpha Y,I),
    alpha=(-.5,.5,1). Conditional distributions are fitted by the estimator's
    NW default, rather than supplied as oracle probability weights.
    """
    rng = np.random.default_rng(seed)
    y, target_y = rng.normal(0, np.sqrt(2), n), rng.normal(1, 1, m)
    alpha = np.array([-.5, .5, 1.])
    x = y[:, None] * alpha + rng.normal(size=(n, 3))
    z = target_y[:, None] * alpha + rng.normal(size=(m, 3))
    f, fu = (x @ alpha) / 2, (z @ alpha) / 2
    return LabelShiftData(y, x, z), PPIData(y, f, fu), 1.
