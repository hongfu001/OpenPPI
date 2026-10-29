#!/usr/bin/env python3
"""Synthetic data + terminal ASI only (adapted from the supplied simulation).

Dependencies: numpy scipy pandas scikit-learn.
Run from examples/simulation: python coverage.py --n 2000 --reps 100 --rho .2 --out results_asi
Budget means a target EXPECTED number rho*n of labels; actual counts fluctuate.
Only the terminal Wald interval is evaluated: no confidence sequences, repeated
monitoring, stopping rules, empirical likelihood, PPI++, or other baselines.
Sequential predictor updates are retained; they do not imply sequential inference.
Pretraining labels come from an independent data stream.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from openppi import Mean, get_method, tune_tau
from config import add_config_arguments, config_from_args
from dataset.judge_sim import DGP
from model.judge import JudgeModel
from model.learners import SimulationLearner
from online import BudgetCapped, Exploring, active_label
from utils import rng_for, seed_sequence, summarize, write_config, write_csv


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
    recipe = get_method('ASI')          # the sequential designs above replace its batch sampling
    result = recipe.inference(sample, target=Mean(), alpha=c.alpha)
    low, high = result.confidence_interval
    return dict(pretrain_labels=c.pretrain_size if policy == 'asi_pretrained' else 0, rep=rep, policy=policy, method=recipe.name, t=c.n, tau=tau,
                estimate=result.estimate, standard_error=result.standard_error,
                low=low, high=high, width=high-low, truth=truth,
                covered=bool(low <= truth <= high), sq_error=(result.estimate-truth)**2,
                audits=result.diagnostics['audits'], target_expected_audits=c.rho*c.n,
                sum_sampling_probabilities=result.diagnostics['sum_sampling_probabilities'],
                min_pi=result.diagnostics['min_pi'], max_pi=result.diagnostics['max_pi'])


def validate(c, policies):
    if c.n < 2 or c.reps < 1 or c.ref_size < 1 or c.truth_size < 2:
        raise ValueError('Require n>=2, reps>=1, ref-size>=1, truth-size>=2')
    if not (0 < c.rho < 1 and 0 < c.alpha < 1 and 0 < c.asi_tau <= 1):
        raise ValueError('Require rho,alpha in (0,1), asi-tau in (0,1]')
    if c.update_every < 1 or not 0 <= c.explore <= c.n or c.seed < 0:
        raise ValueError('Invalid update-every, explore or seed')
    if not 1 <= c.judges <= 5 or not 0 <= c.polarized < 1:
        raise ValueError('Require judges in [1,5], polarized in [0,1)')
    if c.kappa_easy <= 0 or c.kappa_hard <= 0 or c.judge_temp <= 0 or c.noise_sd <= 0 or c.common_sd < 0:
        raise ValueError('Invalid DGP scale parameters')
    choices = dict(model=('linear','nonlinear'), learner=('rf','svm','mlp'),
                   regime=('informative','biased','correlated','weak','shared_failure',
                           'overconfident','inverted','confident_failure'),
                   judge_features=('prob','logit'), asi_f=('soft','hard'),
                   update_on=('rows','labels'))
    for name, allowed in choices.items():
        if getattr(c, name) not in allowed:
            raise ValueError(f'{name} must be one of {allowed}')
    if c.asi_budget_cap not in (0,1) or c.tune_tau not in (0,1):
        raise ValueError('asi-budget-cap and tune-tau must be 0 or 1')
    if c.greedy_steps < 0:
        raise ValueError('greedy-steps must be >= 0')
    if not policies or len(set(policies)) != len(policies) or not set(policies) <= {'asi','asi_judge','asi_pretrained'}:
        raise ValueError('Unknown or duplicate policies')
    if c.pretrain_size < 2:
        raise ValueError('pretrain-size must be >= 2')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument('--policies', default='asi,asi_judge,asi_pretrained')
    parser.add_argument('--out', default='results_asi')
    args = parser.parse_args()
    c = config_from_args(args)
    policies = [p.strip() for p in args.policies.split(',')]
    validate(c, policies)
    dgp = DGP(c)
    truth, truth_se = dgp.truth()
    out = Path(args.out)
    write_config(out, c, policies=policies, truth=truth,
                 truth_integration_se=truth_se,
                 budget_definition='online target expected labels; pretraining labels separate')
    rows = []
    for rep in range(c.reps):
        data = dgp.draw(c.n, rng_for(c.seed, rep, 1))
        ref = dgp.draw(c.ref_size, rng_for(c.seed, rep, 2))
        for policy in policies:
            rows.append(run_asi(c, rep, policy, data, ref, truth))
        print(f'Replicate {rep+1}/{c.reps}', flush=True)
    results = pd.DataFrame(rows)
    write_csv(out, 'per_rep_final.csv', results)
    summary = summarize(results)
    write_csv(out, 'fixed_budget_comparison.csv', summary)
    print(summary.to_string(index=False))
    print(f'Results: {out.resolve()}')


if __name__ == '__main__':
    main()
