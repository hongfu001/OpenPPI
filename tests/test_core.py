"""Core formulas, fixed-budget ASI selection, PPI, and DGP coverage."""
import numpy as np
import pytest

from openppi import (
    ASI, PPI, ActiveData, Mean, PPIData, Uncertainty, select_for_labeling, LabelRequest, tune_tau,
)
from openppi.common.stats import ipw_correct, wald_interval
from config import Config
from openppi import split
from utils import rng_for
from dataset.judge_sim import DGP


def test_correction_and_interval():
    p = np.array([.2, .6, .8, .4])
    q = np.array([True, False, True, False])
    y = np.array([1., np.nan, 0., np.nan])
    pi = np.array([.5, .3, .8, .5])
    values = ipw_correct(p, y, q, pi)
    np.testing.assert_allclose(values, [1.8, .6, -.2, .4])
    result = ASI()(ActiveData(p, y, q, pi), target=Mean())
    assert result.estimate == pytest.approx(.65)
    assert result.standard_error == pytest.approx(np.std(values, ddof=1) / 2)
    assert result.confidence_interval == wald_interval(result.estimate, result.standard_error)
    assert result.diagnostics['audits'] == 2
    assert 'values' not in result.diagnostics

    def interval(point, se, alpha, *, values):
        np.testing.assert_allclose(values, [1.8, .6, -.2, .4])
        return wald_interval(point, se, alpha)

    assert ASI(interval=interval)(ActiveData(p, y, q, pi), target=Mean()) == result


def test_sampling_budget_and_floor():
    p = np.linspace(.01, .99, 200)
    budget, tau = 40, .5
    request = select_for_labeling(
        p, target=Mean(), design=Uncertainty(tau=tau), budget=budget, random_state=1)
    np.testing.assert_allclose(request.propensities.sum(), budget, atol=1e-10)
    assert request.propensities.min() >= tau * budget / len(p)
    assert request.queried.dtype == bool


def test_with_labels_requires_one_label_per_index():
    request = select_for_labeling(np.linspace(.05, .95, 50), target=Mean(),
                                  design=Uncertainty(), budget=10, random_state=0)
    k = len(request.indices)
    for wrong in (np.ones(k + 1), np.ones(max(k - 1, 0)), np.ones((k, 1))):
        with pytest.raises(ValueError):
            request.with_labels(wrong)


def test_label_request_has_no_labels_until_attached():
    p = np.linspace(.05, .95, 60)
    request = select_for_labeling(p, target=Mean(), design=Uncertainty(), budget=15, random_state=3)
    assert isinstance(request, LabelRequest)
    assert {f for f in request.__dataclass_fields__} == {'predictions', 'propensities', 'queried', 'design'}
    assert isinstance(request.design, Uncertainty) and request.design.rho == 15 / 60
    assert not request.queried.flags.writeable
    np.testing.assert_array_equal(request.indices, np.flatnonzero(request.queried))
    labels = (np.arange(len(request.indices)) % 2).astype(float)
    record = request.with_labels(labels)
    assert isinstance(record, ActiveData)
    np.testing.assert_array_equal(record.observed_labels[request.indices], labels)
    assert np.isnan(record.observed_labels[~request.queried]).all()
    np.testing.assert_array_equal(record.propensities, request.propensities)
    np.testing.assert_array_equal(record.predictions, p)


def test_eta_matches_paper_scaling():
    scores = np.array([.1, .2, .4, 1.3])
    rho = .3
    design = Uncertainty(tau=.25).calibrate(scores, rho)
    raw = np.clip(rho / scores.mean() * scores, 0, 1)
    np.testing.assert_allclose(design.probability(scores), .75 * raw + .25 * rho)
    # Zero scores carry no ranking and fall back to uniform sampling.
    np.testing.assert_allclose(Uncertainty().calibrate(np.zeros(4), rho).probability(np.zeros(4)), rho)


def test_tune_tau_minimizes_eq12():
    rng = np.random.default_rng(0)
    p = rng.uniform(.01, .99, 500)
    y = rng.binomial(1, p).astype(float)
    taus = np.linspace(.05, .95, 19)
    scores = Mean().sampling_scores(np.empty((len(p), 0)), p)
    risks = [np.mean((y - p) ** 2 / Uncertainty(tau=t).calibrate(scores, .2).probability(scores))
             for t in taus]
    assert tune_tau(p, y, target=Mean(), rho=.2, taus=taus) == taus[int(np.argmin(risks))]
    # The tuned rule is never worse than uniform sampling (tau = 1) on the history.
    best = tune_tau(p, y, target=Mean(), rho=.2, taus=np.append(taus, 1.))
    risk = lambda t: np.mean((y - p) ** 2 / Uncertainty(tau=t).calibrate(scores, .2).probability(scores))
    assert risk(best) <= risk(1.)


@pytest.mark.slow
def test_dgp_interval_coverage():
    config = Config(n=800, reps=200)
    dgp = DGP(config)
    truth, _ = dgp.truth()
    covered = 0
    for rep in range(config.reps):
        data = dgp.draw(config.n, rng_for(config.seed, rep, 1))
        request = select_for_labeling(
            data['oracle'], target=Mean(), design=Uncertainty(tau=config.asi_tau),
            budget=config.rho * config.n, random_state=rng_for(config.seed, rep, 2))
        record = request.with_labels(data['y'][request.indices])
        low, high = ASI()(record, target=Mean(), alpha=config.alpha).confidence_interval
        covered += low <= truth <= high
    coverage = covered / config.reps
    # A loose Monte Carlo tolerance around nominal 95% coverage.
    assert .90 < coverage <= 1.0, f'coverage={coverage:.3f}'


def test_ppi_formula_and_split():
    data = PPIData(labels=[1., 0., 1., 1.],
                   labeled_predictions=[.8, .4, .6, .2],
                   unlabeled_predictions=[.1, .2, .3, .4, .5])
    captured = {}

    def interval(point, se, alpha, *, values):
        captured.update(values)
        return wald_interval(point, se, alpha, values=values)

    result = PPI(interval=interval)(data, target=Mean())
    np.testing.assert_allclose(captured['residuals'], [.2, -.4, .4, .8])
    assert result.estimate == pytest.approx(.55)
    assert result.standard_error == pytest.approx(np.sqrt(.005 + .0625))
    assert result.confidence_interval == wald_interval(result.estimate, result.standard_error)
    assert 'values' not in result.diagnostics
    assert result == PPI()(data, target=Mean())
    assert not data.labels.flags.writeable
    with pytest.raises(ValueError):
        PPIData([1., 0.], [.5], [.2, .8])

    # Row IDs expose overlap, missing rows, and label misalignment.
    x = np.arange(30)[:, None]
    train, labeled, unlabeled = split(x, np.arange(30) % 2,
                                     train_size=8, labeled_size=6, random_state=4)
    np.testing.assert_array_equal(train[1], train[0][:, 0] % 2)
    np.testing.assert_array_equal(labeled[1], labeled[0][:, 0] % 2)
    rows = np.concatenate([train[0], labeled[0], unlabeled])[:, 0]
    np.testing.assert_array_equal(np.sort(rows), np.arange(30))
    assert len(train[0]) == 8 and len(labeled[0]) == 6 and len(unlabeled) == 16
