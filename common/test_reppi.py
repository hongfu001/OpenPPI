"""RePPI: fold formula, binned learner, gain over PPI++ for discrete predictions, coverage."""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from prediction_inference import BinnedMean, PPIPlusPlus, RePPI, Mean, PPIData
from prediction_inference.methods.reppi import three_folds

TRUTH = np.mean([.1, .2, .9])


def discrete_rater(n, rng):
    """A three-level autorater whose levels 0, .5, 1 have P(Y = 1) = .1, .2, .9 (nonlinear)."""
    level = rng.integers(0, 3, n)
    return (rng.random(n) < np.array([.1, .2, .9])[level]).astype(float), level / 2


class Identity:
    def fit(self, features, labels):
        return self

    def predict(self, features):
        return np.ravel(features)


def test_fold_formula_by_hand():
    rng = np.random.default_rng(1)
    y, f = discrete_rater(300, rng)
    _, f_u = discrete_rater(900, rng)
    data = PPIData(y, .99 * f + rng.uniform(0, .01, 300), f_u)
    result = RePPI(learner_factory=Identity, random_state=7)(data, target=Mean())
    folds = three_folds(300, np.random.default_rng(7))
    estimates = []
    for infer in (folds[2], folds[0], folds[1]):
        g, labels = data.labeled_predictions[infer], y[infer]
        a = np.cov(labels, g)[0, 1] / g.var(ddof=1) / (1 + 300 / 900)
        estimates.append(labels.mean() + a * (f_u.mean() - g.mean()))
    assert result.estimate == pytest.approx(np.mean(estimates))
    np.testing.assert_allclose(result.diagnostics['fold_estimates'], estimates)


def test_binned_mean_learns_conditional_means():
    rng = np.random.default_rng(2)
    y, f = discrete_rater(30000, rng)
    model = BinnedMean().fit(f[:, None], y)
    np.testing.assert_allclose(model.predict(np.array([[0.], [.5], [1.]])), [.1, .2, .9], atol=.015)
    continuous = BinnedMean().fit(rng.random((1000, 1)), rng.random(1000))
    assert 2 <= len(continuous.means) <= 11


def test_gain_over_ppi_plus_plus_matches_theory():
    rng = np.random.default_rng(3)
    y, f = discrete_rater(20000, rng)
    _, f_u = discrete_rater(180000, rng)
    data = PPIData(y, f, f_u)
    reppi, ppi = RePPI(random_state=0)(data, target=Mean()), PPIPlusPlus()(data, target=Mean())
    r, var_y = 20000 / 180000, TRUTH * (1 - TRUTH)
    cov_yf = np.mean(np.array([.1, .2, .9]) * [0, .5, 1]) - TRUTH * .5
    assert ppi.standard_error == pytest.approx(np.sqrt((var_y - cov_yf ** 2 / np.var([0, .5, 1]) / (1 + r)) / 20000), rel=.02)
    assert reppi.standard_error == pytest.approx(np.sqrt((var_y - np.var([.1, .2, .9]) / (1 + r)) / 20000), rel=.02)
    assert reppi.standard_error < ppi.standard_error


def test_sklearn_learner_and_coverage():
    rng = np.random.default_rng(4)
    y, f = discrete_rater(600, rng)
    _, f_u = discrete_rater(3000, rng)
    forest = RePPI(learner_factory=lambda: RandomForestRegressor(n_estimators=30, min_samples_leaf=20, random_state=0),
                   random_state=0)(PPIData(y, f, f_u), target=Mean())
    assert abs(forest.estimate - TRUTH) < 4 * forest.standard_error
    covered = 0
    for rep in range(200):
        rng = np.random.default_rng(100 + rep)
        y, f = discrete_rater(500, rng)
        _, f_u = discrete_rater(4500, rng)
        low, high = RePPI(random_state=rep)(PPIData(y, f, f_u), target=Mean()).confidence_interval
        covered += low <= TRUTH <= high
    assert .90 < covered / 200 <= 1.0, f'coverage={covered / 200:.3f}'
