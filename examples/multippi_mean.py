"""MultiPPI vs. classical, single-rater and vector PPI++ under the same budget, over repeated draws.

The label is column 0; raters 1-3 cost 0.05, 0.3 and 1 per query and rater 3
is the most accurate. As in Appendix C.3, N labeled rows (with all rater
scores) are given, the covariance is estimated on them with Ledoit-Wolf, and
the budget only buys extra rater queries. Each baseline is MultiPPI restricted
to the subsets it may use, so every method spends the same budget.
"""
import itertools

import numpy as np

from dataset.judge_sim import make_multi_rater_data
from prediction_inference import MultiPPI, MultiPPIData, Mean, ledoit_wolf, multi_allocate

LABELED, BUDGET, REPS = 200, 300., 300
PRICE = {1: .05, 2: .3, 3: 1.}
FULL = (0, 1, 2, 3)
designs = {                      # first subset: the labeled rows (given, cost 0, at most LABELED)
    'classical': [(0,)],
    'PPI++ rater 1': [(0, 1), (1,)],
    'PPI++ rater 3': [(0, 3), (3,)],
    'PPI++ vector': [FULL, (1, 2, 3)],
    'MultiPPI': [FULL] + [s for r in (1, 2, 3) for s in itertools.combinations((1, 2, 3), r)],
}
errors = {name: [] for name in designs}
covered = dict.fromkeys(designs, 0)
for rep in range(REPS):
    # 1. Labeled rows, then the covariance estimated on them (reused for estimation).
    rng = np.random.default_rng(rep)
    full = make_multi_rater_data(LABELED, seed=rng.integers(2 ** 32))
    sigma = ledoit_wolf(full)
    for name, subsets in designs.items():
        labeled = subsets[0]
        costs = {s: 0. if s == labeled else sum(PRICE[i] for i in s) for s in subsets}
        allocation = multi_allocate(sigma, costs, BUDGET, caps={labeled: LABELED})
        samples = {labeled: full[:, list(labeled)]}
        for s, n in allocation.counts.items():
            if s != labeled:
                samples[s] = make_multi_rater_data(n, seed=rng.integers(2 ** 32))[:, list(s)]
        # 2. Target: the mean of the label. 3. Inference.
        result = MultiPPI(covariance=sigma)(MultiPPIData(samples), target=Mean(), alpha=.05)
        errors[name].append(result.estimate - .5)
        low, high = result.confidence_interval
        covered[name] += low <= .5 <= high

# 4. Output: RMSE relative to classical, and coverage of the 95% interval (truth 0.5).
base = np.sqrt(np.mean(np.square(errors['classical'])))
for name in designs:
    rmse = np.sqrt(np.mean(np.square(errors[name])))
    print(f'{name:14s} RMSE={rmse:.4f} ({rmse / base:.0%} of classical)  coverage={covered[name] / REPS:.3f}')
print('MultiPPI allocation in the last draw:', allocation.counts)
