"""RePPI (Ji, Lei & Zrnic 2026): PPI with a cross-fitted recalibration of the predictions."""
import numpy as np
from .base import InferenceMethod
from ..common.data import PPIData


class BinnedMean:
    """E[Y | f] by averaging labels within prediction bins (the default, NumPy-only learner).

    Predictions with at most max_bins distinct values (discrete autoraters) get one bin
    per value; otherwise about n^(1/3) quantile bins. Empty bins predict the overall mean.
    """
    def __init__(self, max_bins=10):
        self.max_bins = max_bins

    def fit(self, predictions, labels):
        f, y = np.ravel(predictions), np.asarray(labels, dtype=float)
        values = np.unique(f)
        if len(values) <= self.max_bins:
            self.edges = (values[1:] + values[:-1]) / 2
        else:
            bins = max(2, min(self.max_bins, round(len(f) ** (1 / 3))))
            self.edges = np.unique(np.quantile(f, np.linspace(0, 1, bins + 1)[1:-1]))
        index = np.searchsorted(self.edges, f)
        counts = np.bincount(index, minlength=len(self.edges) + 1)
        sums = np.bincount(index, y, minlength=len(self.edges) + 1)
        self.means = np.where(counts > 0, sums / np.maximum(counts, 1), y.mean())
        return self

    def predict(self, predictions):
        return self.means[np.searchsorted(self.edges, np.ravel(predictions))]


def three_folds(n, rng):
    """Random split of range(n) into three near-equal folds (Algorithm 1, step 1)."""
    order = rng.permutation(n)
    return order[:n // 3], order[n // 3:2 * n // 3], order[2 * n // 3:]


class RePPI(InferenceMethod):
    """Binary mean RePPI, Algorithm 1 with the squared loss, as in the official reppi.py.

    In each of three rotations one labeled fold gives theta_0 (for the mean the
    score is theta_0 - m(f), so only m matters), one fits m(f) ~ E[Y | f] with
    learner_factory(), and one does inference:
    theta_k = mean(Y) + a_k * (mean_U m - mean_k m), a_k = Cov(Y, m) / Var(m) / (1 + n/N),
    i.e. eq. (13) power tuning of the recalibrated prediction. The estimate is the
    fold average; the SE is sqrt(var(Y - a m~) / n + var(mean_k a_k m_k(f_U)) / N) with
    m~ the out-of-fold recalibrated labeled predictions. learner_factory returns an
    object with fit(features, labels) and predict(features), e.g. a scikit-learn
    regressor; features are the predictions as an n x 1 array.
    """
    name = 'RePPI'
    data_type = PPIData

    def __init__(self, learner_factory=BinnedMean, random_state=None, **kwargs):
        super().__init__(**kwargs)
        self.learner_factory, self.random_state = learner_factory, random_state

    def _estimate(self, data, target):
        y, f, f_u = data.labels, data.labeled_predictions, data.unlabeled_predictions
        n, big_n = len(y), len(f_u)
        if n < 6:
            raise ValueError('RePPI needs at least two labeled rows per fold')
        folds = three_folds(n, np.random.default_rng(self.random_state))
        estimates, weights, labeled_fit, unlabeled_fit = [], [], np.empty(n), np.zeros(big_n)
        for train, infer in ((folds[1], folds[2]), (folds[2], folds[0]), (folds[0], folds[1])):
            model = self.learner_factory().fit(f[train, None], y[train])
            m_k, m_u = np.ravel(model.predict(f[infer, None])), np.ravel(model.predict(f_u[:, None]))
            spread = m_k.var(ddof=1)
            a = np.cov(y[infer], m_k)[0, 1] / spread / (1 + n / big_n) if spread > 0 else 0.
            estimates.append(y[infer].mean() + a * (m_u.mean() - m_k.mean()))
            weights.append(float(a))
            labeled_fit[infer] = a * m_k
            unlabeled_fit += a * m_u / 3
        point = float(np.mean(estimates))
        residuals = y - labeled_fit
        se = float(np.sqrt(residuals.var(ddof=1) / n + unlabeled_fit.var(ddof=1) / big_n))
        return point, se, {
            'values': {'residuals': residuals, 'unlabeled_predictions': unlabeled_fit},
            'labeled_count': n,
            'unlabeled_count': big_n,
            'fold_estimates': [float(e) for e in estimates],
            'fold_weights': weights,
            'variance_design': 'cross_fit_three_folds',
        }
