
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def make_learner(c, seed):
    """IPW-weighted regression of Y on Z = (X, J); predictions are clipped to [.01, .99] by the caller."""
    if c.learner == 'rf':
        return RandomForestRegressor(n_estimators=60, min_samples_leaf=8, max_depth=8,
                                     random_state=seed, n_jobs=1)
    if c.learner == 'svm':
        return make_pipeline(StandardScaler(), SVR(kernel='rbf', C=c.svm_c, epsilon=c.svm_epsilon,
                                                     gamma=c.svm_gamma if c.svm_gamma == 'scale' else float(c.svm_gamma)))
    hidden = tuple(int(v) for v in c.mlp_hidden.split(',') if v.strip())
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=hidden, alpha=c.mlp_alpha,
                                                        learning_rate_init=1e-3, max_iter=300,
                                                        random_state=seed))

import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning

def fit_learner(c, x, y, weight, seed):
    """Weighted least squares sum_s (R_s/pi_s)(Y_s - m(Z_s))^2 over audited items s < t."""
    model = make_learner(c, seed)
    last = model.steps[-1][0] if hasattr(model, 'steps') else None
    kw = {f'{last}__sample_weight': weight} if last else {'sample_weight': weight}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        try:
            model.fit(x, y, **kw)
        except TypeError:   # scikit-learn < 1.7: MLPRegressor has no sample_weight -> weighted resampling
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(y), size=len(y), replace=True, p=weight / weight.sum())
            model.fit(x[idx], y[idx])
    return model



class SimulationLearner:
    """Refit the configured learner on cumulative queried observations."""
    def __init__(self, config, seed):
        self.config, self.seed = config, seed
        self.model = None
        self.initial = None

    def update(self, x, y, sample_weight):
        # Preserve independent pretraining observations during subsequent refits.
        if self.initial is not None:
            px, py, pw = self.initial
            x = np.concatenate([px, x])
            y = np.concatenate([py, y])
            sample_weight = np.concatenate([pw, sample_weight])
        self.model = fit_learner(self.config, x, y, sample_weight, self.seed)
        return self

    def retain_pretraining(self, x, y):
        self.update(x, y, np.ones(len(y)))
        self.initial = (np.array(x, copy=True), np.array(y, copy=True), np.ones(len(y)))
        return self

    def predict(self, x):
        if self.model is None:
            return np.full(len(x), .5)
        return np.clip(self.model.predict(x), .01, .99)


class SklearnModel:
    """Probability regressor adapter; finetune requires partial_fit support.

    Refit clones the estimator on each cumulative training set. Finetune
    retains the estimator and consumes only newly queried observations.
    """
    def __init__(self, estimator, mode='refit', pretrained=False):
        from copy import deepcopy
        if mode not in ('refit', 'finetune'):
            raise ValueError('mode must be refit or finetune')
        if mode == 'finetune' and not callable(getattr(estimator, 'partial_fit', None)):
            raise TypeError('finetune requires an estimator with partial_fit')
        self.mode = mode
        self.estimator = deepcopy(estimator)
        self.fitted = pretrained

    def predict(self, x):
        return (np.clip(self.estimator.predict(x), .01, .99)
                if self.fitted else np.full(len(x), .5))

    def update(self, x, y, sample_weight):
        if self.mode == 'refit':
            from sklearn.base import clone
            self.estimator = clone(self.estimator)
            self.estimator.fit(x, y, sample_weight=sample_weight)
        else:
            self.estimator.partial_fit(x, y, sample_weight=sample_weight)
        self.fitted = True
        return self
