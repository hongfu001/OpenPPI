"""Simulation adapters for the public active labeling API."""
import numpy as np
from prediction_inference import ASI, Mean, tune_tau
from .online import active_label, BudgetCapped, Exploring
from model.learners import SimulationLearner
from model.judge import JudgeModel
from dataset.judge_sim import DGP
from common.utils import rng_for, seed_sequence


def historical_tau(c, rep, policy):
    """Eq. (12) tau from independent historical labels, as in the batch experiments.

    asi_judge scores the fixed judge model on the historical draw;
    asi_pretrained fits a tuning model on 80% of its pretraining data and
    scores the held-out 20%. The cold-start policy has no history.
    """
    if not c.tune_tau or policy == 'asi':
        return c.asi_tau
    history = DGP(c).draw(c.pretrain_size, rng_for(c.seed, rep, 3))
    if policy == 'asi_judge':
        x = np.column_stack([history['features'], history['j']])
        p = JudgeModel(range(x.shape[1] - c.judges, x.shape[1])).predict(x)
        return tune_tau(p, history['y'], target=Mean(), rho=c.rho)
    cut = int(.8 * c.pretrain_size)
    model = SimulationLearner(c, c.seed + rep).update(
        history['features'][:cut], history['y'][:cut], np.ones(cut))
    return tune_tau(model.predict(history['features'][cut:]), history['y'][cut:],
                    target=Mean(), rho=c.rho)


def sample_asi(c, rep, policy, data, ref):
    if policy not in ('asi', 'asi_judge', 'asi_pretrained'):
        raise ValueError('Unknown ASI simulation policy')
    x, reference = data['features'], ref['features']
    kwargs = dict(model=None, update='refit',
                  model_factory=lambda: SimulationLearner(c, c.seed + rep))
    if policy == 'asi_pretrained':
        initial = DGP(c).draw(c.pretrain_size, rng_for(c.seed, rep, 3))
        model = SimulationLearner(c, c.seed + rep)
        model.retain_pretraining(initial['features'], initial['y'])
        kwargs = dict(model=model, update='refit')
    if policy == 'asi_judge':
        # Raw probabilities are retained even when learner features use logits.
        x = np.column_stack([x, data['j']])
        reference = np.column_stack([reference, ref['j']])
        kwargs = dict(model=JudgeModel(range(x.shape[1] - c.judges, x.shape[1])),
                      update='none')
    tau = historical_tau(c, rep, policy)
    design = (BudgetCapped(tau=tau, explore=c.explore, greedy_steps=c.greedy_steps)
              if c.asi_budget_cap else Exploring(tau=tau, explore=c.explore))
    record = active_label(x, oracle=lambda i: data['y'][i], target=Mean(),
                          budget=c.rho*c.n, update_every=c.update_every, update_on=c.update_on,
                          reference_features=reference, prediction_mode=c.asi_f,
                          design=design, random_state=seed_sequence(c.seed, rep, 777), **kwargs)
    return record, tau


def run_asi(c, rep, policy, data, ref, truth):
    sample, tau = sample_asi(c, rep, policy, data, ref)
    result = ASI()(sample, target=Mean(), alpha=c.alpha)
    low, high = result.confidence_interval
    return dict(pretrain_labels=c.pretrain_size if policy == 'asi_pretrained' else 0, rep=rep, policy=policy, method=result.method, t=c.n, tau=tau,
                estimate=result.estimate, standard_error=result.standard_error,
                low=low, high=high, width=high-low, truth=truth,
                covered=bool(low <= truth <= high), sq_error=(result.estimate-truth)**2,
                audits=result.diagnostics['audits'], target_expected_audits=c.rho*c.n,
                sum_sampling_probabilities=result.diagnostics['sum_sampling_probabilities'],
                min_pi=result.diagnostics['min_pi'], max_pi=result.diagnostics['max_pi'])
