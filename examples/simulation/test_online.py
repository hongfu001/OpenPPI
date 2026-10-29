"""Sequential ASI used by coverage.py (not part of the core library): online flow, Exploring, BudgetCapped.

Run with: python -m pytest examples/simulation/test_online.py
"""
import numpy as np
import pytest

from openppi import IPW, Mean, Uniform, Uncertainty, select_for_labeling
from online import BudgetCapped, Exploring, active_label


class Fixed:
    def predict(self, x):
        return x[:, 0]


class CountingMean(Mean):
    def sampling_scores(self, features, predictions):
        self.calls.append(len(predictions))
        return super().sampling_scores(features, predictions)


def test_target_and_oracle_only_at_queries():
    x = np.linspace(.05, .95, 100)[:, None]
    calls = []
    target = CountingMean()
    object.__setattr__(target, 'calls', [])

    def oracle(i):
        calls.append(i)
        return float(i % 2)

    record = active_label(x, oracle=oracle, target=target, model=Fixed(),
                          budget=30, random_state=7, update_every=20)
    assert calls == np.flatnonzero(record.queried).tolist()
    assert len(calls) == len(set(calls))
    assert len(target.calls) == 10  # score each block and reference before querying
    assert np.isnan(record.observed_labels[~record.queried]).all()
    np.testing.assert_allclose(record.predictions, x[:, 0])
    assert IPW()(record, target=target).diagnostics['audits'] == len(calls)


def test_online_fit_uses_past_queries_and_keeps_history():
    x = np.arange(120, dtype=float)[:, None]
    fits, calls = [], []

    class Learner:
        def update(self, features, labels, sample_weight):
            assert set(features[:, 0].astype(int)) == set(calls)
            np.testing.assert_array_equal(labels, features[:, 0] % 2)
            assert (sample_weight >= 1).all()
            fits.append((len(calls), len(fits)))
            self.p = .2 if len(fits) == 1 else .8
            return self

        def predict(self, features):
            return np.full(len(features), self.p)

    def oracle(i):
        calls.append(i)
        return float(i % 2)

    record = active_label(x, oracle=oracle, target=Mean(), model=None, update="refit", model_factory=Learner,
                          budget=100, random_state=2, update_every=40, min_train=2)
    assert len(fits) == 2
    np.testing.assert_array_equal(record.predictions[:40], .5)
    np.testing.assert_array_equal(record.predictions[40:80], .2)
    np.testing.assert_array_equal(record.predictions[80:], .8)


@pytest.mark.parametrize('mode', ['refit', 'finetune'])
def test_pretrained_updates_preserve_history_and_use_only_past_queries(mode):
    x = np.arange(120, dtype=float)[:, None]
    calls, batches = [], []

    class Online:
        p = .3

        def predict(self, features):
            return np.full(len(features), self.p)

        def update(self, features, labels, sample_weight):
            rows = features[:, 0].astype(int).tolist()
            expected = calls if mode == 'refit' else calls[sum(map(len, batches)):]
            assert rows == expected
            np.testing.assert_array_equal(labels, features[:, 0] % 2)
            assert (sample_weight >= 1).all()
            batches.append(rows)
            self.p = .7

    def oracle(i):
        calls.append(i)
        return i % 2

    original = Online()
    record = active_label(x, oracle=oracle, target=Mean(), model=original,
                          update=mode, budget=100, update_every=40,
                          min_train=2, random_state=2)
    assert len(batches) == 2
    assert original.p == .3
    np.testing.assert_array_equal(record.predictions[:40], .3)
    np.testing.assert_array_equal(record.predictions[40:], .7)



def test_fixed_model_online_flow_matches_batch_selection():
    # With a frozen model and one block, the online flow reproduces select_for_labeling.
    p = np.linspace(.01, .99, 200)
    for design in (Uniform(), Uncertainty()):
        request = select_for_labeling(p, target=Mean(), design=design, budget=40, random_state=1)
        record = active_label(p[:, None], oracle=lambda i: i % 2, target=Mean(),
                              model=Fixed(), design=design, budget=40,
                              update_every=len(p), random_state=1)
        np.testing.assert_array_equal(request.queried, record.queried)
        np.testing.assert_array_equal(request.propensities, record.propensities)
        assert not hasattr(design, 'rho')


def test_exploring_rows_use_uniform_probability():
    design = Exploring(tau=.5, explore=3).calibrate(np.linspace(.1, 1, 10), .2)
    pi = design.probability(np.linspace(.1, 1, 10), index=np.arange(10))
    np.testing.assert_allclose(pi[:3], .2)
    np.testing.assert_allclose(pi[3:], Uncertainty(tau=.5).calibrate(np.linspace(.1, 1, 10), .2)
                               .probability(np.linspace(.1, 1, 10))[3:])


def test_greedy_steps_use_remaining_budget():
    design = BudgetCapped(tau=.5, greedy_steps=2).calibrate(np.ones(10), .25)
    # Window int(2/.25)=8: rows 6, 7, 14, 15, ... ignore scores and use the remaining budget.
    rows = np.arange(16)
    pi = design.probability(np.zeros(16), index=rows, queried=0)
    greedy = rows % 8 >= 6
    np.testing.assert_allclose(pi[greedy], .5 * np.clip(.25 * (rows[greedy] + 1), 0, 1) + .125)
    np.testing.assert_allclose(pi[~greedy], .125)


def test_label_triggered_updates_rescore_remaining_rows():
    x = np.arange(400, dtype=float)[:, None]
    calls, updates = [], []

    class Learner:
        p = .3

        def predict(self, features):
            return np.full(len(features), self.p)

        def update(self, features, labels, sample_weight):
            updates.append(len(calls))
            self.p = .3 + .1 * len(updates)

    def oracle(i):
        calls.append(i)
        return i % 2

    record = active_label(x, oracle=oracle, target=Mean(), model=Learner(), update='refit',
                          budget=200, update_every=25, update_on='labels',
                          min_train=2, random_state=3)
    assert updates and all(u % 25 == 0 for u in updates)
    # Each update happens right after the 25th new label and changes all later predictions.
    for k, count in enumerate(updates, 1):
        row = calls[count - 1]
        assert record.predictions[row] == pytest.approx(.3 + .1 * (k - 1))
        assert record.predictions[row + 1] == pytest.approx(.3 + .1 * k)
