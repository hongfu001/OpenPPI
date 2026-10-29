"""Sequential (online) active labeling, kept for the simulations only.

The core library covers fixed-budget, fixed-model batches. This module
holds the sequential variant of ASI (Algorithm 2): the model is updated
as labels arrive, and Exploring / BudgetCapped extend the core
Uncertainty design with stream-dependent rules (initial exploration,
Eq. (8) running budget and periodic budget use-up).
"""
from copy import deepcopy
import numpy as np

from openppi import ActiveData, Mean, Uncertainty
from openppi.sampling import SamplingDesign


class Exploring(Uncertainty):
    """Uncertainty design whose first explore stream rows use probability rho."""
    def __init__(self, tau=.5, explore=0):
        super().__init__(tau=tau)
        if not isinstance(explore, int) or explore < 0:
            raise ValueError('explore must be a nonnegative integer')
        self.explore = explore

    def _raw_probability(self, scores, index, queried):
        scores = np.asarray(scores, dtype=float)
        if self.eta is None:
            return np.full(scores.shape, self.rho)
        return np.clip(self.eta * scores, 0, 1)

    def probability(self, scores, *, index=None, queried=0):
        raw = self._raw_probability(scores, index, queried)
        pi = (1 - self.tau) * raw + self.tau * self.rho
        if index is not None:
            pi = np.where(np.asarray(index) < self.explore, self.rho, pi)
        return pi


class BudgetCapped(Exploring):
    """Running-budget rule of Eq. (8), applied before uniform mixing.

    Raw probabilities are capped by the remaining imaginary budget
    rho*(t+1) - queried. With greedy_steps > 0, the last greedy_steps rows
    of every int(greedy_steps / rho)-row window use the full remaining
    budget regardless of scores, so a low-uncertainty model does not
    underuse the budget. Neither rule enforces an exact or hard count.
    Query sequentially and supply the number of labels observed before
    the current row.
    """
    def __init__(self, tau=.5, explore=0, greedy_steps=0):
        super().__init__(tau=tau, explore=explore)
        if not isinstance(greedy_steps, int) or greedy_steps < 0:
            raise ValueError('greedy_steps must be a nonnegative integer')
        self.greedy_steps = greedy_steps

    def _raw_probability(self, scores, index, queried):
        if index is None:
            raise ValueError('BudgetCapped requires the current stream index')
        raw = super()._raw_probability(scores, index, queried)
        index = np.asarray(index)
        cap = np.clip(self.rho * (index + 1) - queried, 0, 1)
        raw = np.minimum(raw, cap)
        if self.greedy_steps:
            window = max(int(self.greedy_steps / self.rho), 1)
            raw = np.where(index % window >= window - self.greedy_steps, cap, raw)
        return raw


def _probability(design, score, index, queried):
    """Stream-aware designs receive the row index and running label count."""
    if isinstance(design, Exploring):
        return design.probability(score, index=index, queried=queried)
    return design.probability(score)


def active_label(features, *, oracle, target, budget, model=None, update='none',
                 model_factory=None, update_every=100, random_state=None,
                 reference_features=None, min_train=30, prediction_mode='soft', design=None,
                 update_on='rows', score_fn=None, initial_prediction=None):
    """Query selected outcomes and retain query-time predictions.

    Models expose predict(X) and, for online learning, update(X, y,
    sample_weight). update='none' freezes a supplied model. 'refit' sends
    all past queried rows; 'finetune' sends only rows added since the last
    update. The model adapter implements the corresponding training rule.
    update_on='rows' updates at every update_every stream rows;
    update_on='labels' updates as soon as update_every new labels have
    been queried (Algorithm 2), then rescores the remaining rows.
    With model=None, supply model_factory to construct an unfitted adapter;
    Binary predictions remain .5 until min_train observations include both
    classes. Continuous Mean uses initial_prediction (default 0), permits
    real labels, and needs only min_train observations. Supply an outcome-free
    score_fn(features, predictions), e.g. an independently trained residual
    learner. Refresh such a learner with honest residuals if the predictor
    changes; score_fn is never given current or future outcomes.
    A supplied pretrained model predicts from the first block. Models are
    deep-copied so caller state is preserved. No fit method is called here.

    design defaults to Uncertainty(tau=.5) and owns mixing; Exploring and
    BudgetCapped add initial exploration and running-budget adjustments. The design is copied before calibration to
    preserve caller state. budget is an expected count, not a hard cap.
    """
    if not isinstance(target, Mean):
        raise TypeError('Only the Mean target is currently supported')
    continuous = target.outcome_type == 'continuous'
    if continuous and prediction_mode == 'hard':
        raise ValueError('Continuous outcomes do not support hard predictions')
    if continuous and score_fn is None:
        raise ValueError('Continuous online sampling requires score_fn(features, predictions)')
    initial_prediction = (0. if continuous else .5) if initial_prediction is None else float(initial_prediction)
    if not np.isfinite(initial_prediction):
        raise ValueError('initial_prediction must be finite')
    x = np.array(features, dtype=float, copy=True)
    ref = x if reference_features is None else np.array(reference_features, dtype=float, copy=True)
    if x.ndim != 2 or len(x) < 2 or ref.ndim != 2 or not len(ref) or x.shape[1] != ref.shape[1] or not np.isfinite(x).all() or not np.isfinite(ref).all():
        raise ValueError('Require finite aligned feature matrices and n >= 2')
    if update not in ('none', 'refit', 'finetune'):
        raise ValueError('update must be none, refit or finetune')
    if update_on not in ('rows', 'labels'):
        raise ValueError('update_on must be rows or labels')
    cold = model is None
    if cold:
        if update == 'none' or not callable(model_factory):
            raise ValueError('model=None requires online update and model_factory')
        model = model_factory()
    elif model_factory is not None:
        raise ValueError('Supply model or model_factory, not both')
    if not callable(getattr(model, 'predict', None)):
        raise TypeError('model must expose predict(X)')
    if update != 'none' and not callable(getattr(model, 'update', None)):
        raise TypeError('Online models must expose update(X, y, sample_weight)')
    if update != 'none' and getattr(model, 'mode', update) != update:
        raise ValueError('Model adapter mode must match update')
    if not callable(oracle):
        raise TypeError('oracle must be callable as oracle(index)')
    if not isinstance(update_every, int) or update_every < 1 or not isinstance(min_train, int) or min_train < 2:
        raise ValueError('update_every >= 1 and min_train >= 2 must be integers')
    if not 0 < budget < len(x) or prediction_mode not in ('soft', 'hard'):
        raise ValueError('Require 0 < budget < n and soft/hard predictions')
    if design is None:
        design = Uncertainty(tau=.5)
    if not isinstance(design, SamplingDesign):
        raise TypeError('design must implement SamplingDesign')
    design = deepcopy(design)
    rho = budget / len(x)
    uniforms = np.random.default_rng(random_state).random(len(x))
    predictions = np.full(len(x), initial_prediction)
    probabilities = np.zeros(len(x))
    queried = np.zeros(len(x), dtype=bool)
    observed = np.full(len(x), np.nan)
    model = deepcopy(model)
    ready = not cold
    last_update = 0
    nlab = 0
    def predict(features):
        if not ready:
            return np.full(len(features), initial_prediction)
        values = np.asarray(model.predict(features), dtype=float)
        if values.shape != (len(features),) or not np.isfinite(values).all():
            raise ValueError('Model must predict one finite prediction per row')
        if not continuous and ((values < 0) | (values > 1)).any():
            raise ValueError('Model predictions must be probabilities in [0, 1]')
        return values

    def try_update(start):
        nonlocal ready, last_update
        idx = np.flatnonzero(queried[:start])
        if update == 'none' or len(idx) < min_train or (not continuous and len(np.unique(observed[idx])) < 2):
            return False
        train = idx if update == 'refit' else idx[idx >= last_update]
        if not len(train):
            return False
        model.update(x[train], observed[train], sample_weight=1 / probabilities[train])
        ready = True
        last_update = start
        return True

    pending = 0  # labels queried since the last update
    start = 0
    while start < len(x):
        if update_on == 'rows':
            try_update(start)
            end = min(start + update_every, len(x))
        else:
            end = len(x)
        p = predict(x[start:end])
        pref = predict(ref)
        scorer = target.sampling_scores if score_fn is None else score_fn
        scores = np.asarray(scorer(x[start:end], p), dtype=float)
        ref_scores = np.asarray(scorer(ref, pref), dtype=float)
        if scores.shape != p.shape or ref_scores.shape != pref.shape or not np.isfinite(scores).all() or not np.isfinite(ref_scores).all() or (scores < 0).any() or (ref_scores < 0).any():
            raise ValueError('score_fn must return aligned finite nonnegative scores')
        design.calibrate(ref_scores, rho)
        predictions[start:end] = p
        stop = end
        for offset, t in enumerate(range(start, end)):
            pi = float(_probability(design, scores[offset], t, nlab))
            if not np.isfinite(pi) or not 0 < pi <= 1:
                raise ValueError('Sampling design must return probabilities in (0, 1]')
            probabilities[t] = pi
            queried[t] = uniforms[t] < pi
            if queried[t]:
                label = float(oracle(t))
                if not np.isfinite(label) or (not continuous and label not in (0., 1.)):
                    raise ValueError('Oracle must return finite labels (0/1 in binary mode)')
                observed[t] = label
                nlab += 1
                pending += 1
                if update_on == 'labels' and pending >= update_every and try_update(t + 1):
                    pending = 0
                    stop = t + 1
                    break
        start = stop
    if prediction_mode == 'hard':
        predictions = (predictions > .5).astype(float)
    return ActiveData(predictions, observed, queried, probabilities)
