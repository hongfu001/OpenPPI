"""FTRL query probabilities: reference recursion, obliviousness, budget, coverage."""
import numpy as np
import pytest

from openppi import ASI, FTRLDesign, Mean, select_for_labeling
from config import Config
from utils import rng_for
from dataset.judge_sim import DGP


def reference_probabilities(phi, budget, n):
    """compute_pt loop from the reference notebooks with a fixed oracle Phi."""
    beta, gamma, theta, out = budget / 8, 1 / np.sqrt(n), 0., []
    for loss in phi:
        pt = max(beta, min(budget, -gamma * theta))
        out.append(pt)
        theta -= loss / pt ** 2
    return np.array(out)


def test_matches_reference_and_respects_budget():
    phi = np.random.default_rng(0).uniform(0, .3, 500)
    design = FTRLDesign().calibrate(phi, .2)
    p = design.probability(phi)
    np.testing.assert_array_equal(p, reference_probabilities(phi, .2, 500))
    assert p[0] == .2 / 8 and (np.diff(p) >= 0).all() and p[-1] == .2
    assert p.sum() <= 100 + 1e-9


def test_probability_ignores_the_current_row():
    phi = np.full(50, .01)
    design = FTRLDesign(gamma=1e-4).calibrate(phi, .2)
    base = design.probability(phi)
    changed = phi.copy()
    changed[10] = 5.
    p = design.probability(changed)
    np.testing.assert_array_equal(p[:11], base[:11])   # p_t uses rows before t only
    assert (p[11:] >= base[11:]).all() and p[11] > base[11]


@pytest.mark.slow
def test_ftrl_interval_coverage():
    config = Config(n=800, reps=200)
    dgp = DGP(config)
    truth, _ = dgp.truth()
    covered = 0
    for rep in range(config.reps):
        data = dgp.draw(config.n, rng_for(config.seed, rep, 1))
        f = data['oracle']
        request = select_for_labeling(f, target=Mean(), design=FTRLDesign(), scores=f * (1 - f),
                                      budget=config.rho * config.n, random_state=rng_for(config.seed, rep, 2))
        record = request.with_labels(data['y'][request.indices])
        low, high = ASI()(record, target=Mean(), alpha=config.alpha).confidence_interval
        covered += low <= truth <= high
    assert .90 < covered / config.reps <= 1.0, f'coverage={covered / config.reps:.3f}'
