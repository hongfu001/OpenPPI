"""PPI++ power tuning: plug-in lambda, limits lam=0/1, anticorrelation, coverage."""
import numpy as np
import pytest

from openppi import PPI, PPIPlusPlus, Mean, PPIData
from openppi.common.stats import wald_interval
from openppi.inference.ppi_plus_plus import tuned_lambda
from config import Config
from utils import rng_for
from dataset.judge_sim import DGP

DATA = PPIData(labels=[1., 0., 1., 1.],
               labeled_predictions=[.8, .4, .6, .2],
               unlabeled_predictions=[.1, .2, .3, .4, .5])


def test_formula_by_hand():
    y, f, g = DATA.labels, DATA.labeled_predictions, DATA.unlabeled_predictions
    pooled = np.r_[f, g]
    lam = np.cov(y, f)[0, 1] / ((1 + 4 / 5) * pooled.var(ddof=1))
    assert tuned_lambda(y, f, g) == pytest.approx(lam)
    captured = {}

    def interval(point, se, alpha, *, values):
        captured.update(values)
        return wald_interval(point, se, alpha, values=values)

    result = PPIPlusPlus(interval=interval)(DATA, target=Mean())
    assert result.estimate == pytest.approx(y.mean() + lam * (g.mean() - f.mean()))
    se = np.sqrt(lam ** 2 * pooled.var(ddof=1) / 5 + (y - lam * f).var(ddof=1) / 4)
    assert result.standard_error == pytest.approx(se)
    np.testing.assert_allclose(captured['residuals'], y - lam * f)
    assert result.diagnostics['prediction_weight'] == pytest.approx(lam)
    assert result.diagnostics['weight_strategy'] == 'tuned' and 'values' not in result.diagnostics


def test_fixed_lambda_limits():
    classical = PPIPlusPlus(lam=0)(DATA, target=Mean())
    assert classical.estimate == pytest.approx(DATA.labels.mean())
    assert classical.standard_error == pytest.approx(DATA.labels.std(ddof=1) / 2)
    assert PPIPlusPlus(lam=1)(DATA, target=Mean()).estimate == pytest.approx(PPI()(DATA, target=Mean()).estimate)
    assert PPIPlusPlus(lam=1)(DATA, target=Mean()).diagnostics['weight_strategy'] == 'fixed'
    constant = PPIData(DATA.labels, [.5] * 4, [.5] * 5)
    assert PPIPlusPlus()(constant, target=Mean()).diagnostics['prediction_weight'] == 0.
    with pytest.raises(ValueError):
        PPIPlusPlus(lam=np.nan)


def test_tuning_beats_both_baselines_and_handles_anticorrelation():
    data = DGP(Config()).draw(6000, np.random.default_rng(0))
    y, m = data['y'], data['oracle']
    for f in (m, 1 - m):    # anticorrelated predictions need lam < 0 (paper appendix B)
        record = PPIData(y[:300], f[:300], f[300:])
        tuned = PPIPlusPlus()(record, target=Mean())
        assert np.sign(tuned.diagnostics['prediction_weight']) == np.sign(np.corrcoef(y, f)[0, 1])
        for lam in (0, 1):
            assert tuned.standard_error <= PPIPlusPlus(lam=lam)(record, target=Mean()).standard_error


@pytest.mark.slow
def test_ppi_plus_interval_coverage():
    config = Config(n=800, reps=200)
    dgp = DGP(config)
    truth, _ = dgp.truth()
    covered = 0
    for rep in range(config.reps):
        data = dgp.draw(config.n, rng_for(config.seed, rep, 1))
        noise = rng_for(config.seed, rep, 2).normal(0, .1, config.n)
        f = np.clip(data['oracle'] + noise, 0, 1)
        record = PPIData(data['y'][:200], f[:200], f[200:])
        low, high = PPIPlusPlus()(record, target=Mean(), alpha=config.alpha).confidence_interval
        covered += low <= truth <= high
    assert .90 < covered / config.reps <= 1.0, f'coverage={covered / config.reps:.3f}'
