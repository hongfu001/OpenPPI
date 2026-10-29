"""RePPI vs. PPI and PPI++ in two settings.

A. A trained model's output turned into a 1-5 rating (as when an LLM gives a category).
   Here E[label | rating] is close to linear in the rating, so PPI++'s linear correction
   is already near-optimal and RePPI matches it: recalibration never costs efficiency.
B. The paper's discrete-prediction model (Section 4.3): a three-level autorater whose
   levels 0, .5, 1 have P(label = 1) = .1, .2, .9. The levels are informative but not
   linear in the label, so RePPI, which learns E[label | level] on held-out folds,
   gives shorter intervals than PPI++ (Proposition 3).
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from common.utils import split
from dataset.judge_sim import make_judge_data
from prediction_inference import PPI, PPIPlusPlus, RePPI, Mean, PPIData

# A1. Split iid rows; fit only on the independent training sample, then discretize its output.
frame = make_judge_data(n=4000, seed=0)
train, labeled, unlabeled = split(
    frame.drop(columns='label').to_numpy(), frame['label'].to_numpy(),
    train_size=400, labeled_size=600, random_state=42)
model = RandomForestRegressor(n_estimators=60, min_samples_leaf=8, max_depth=8, random_state=0, n_jobs=1)
model.fit(*train)
cuts = np.quantile(model.predict(train[0]), [.2, .4, .6, .8])
rate = lambda x: np.digitize(model.predict(x), cuts) / 4              # rating 1-5 mapped to 0, .25, ..., 1
data = PPIData(labels=labeled[1], labeled_predictions=rate(labeled[0]), unlabeled_predictions=rate(unlabeled))
# A2. Target and inference: default binned recalibration and a random-forest recalibration.
methods = {'PPI': PPI(), 'PPI++': PPIPlusPlus(), 'RePPI (binned)': RePPI(random_state=0),
           'RePPI (forest)': RePPI(learner_factory=lambda: RandomForestRegressor(
               n_estimators=100, min_samples_leaf=20, random_state=0), random_state=0)}
print('A. 5-point rating of a trained model (one run)')
for label, method in methods.items():
    low, high = method(data, target=Mean(), alpha=.05).confidence_interval
    print(f'  {label:15s} CI=({low:.4f}, {high:.4f}) width={high - low:.4f}')

# B. Stylized discrete autorater, 200 repetitions with n = 500 labeled and N = 4500 unlabeled.
def draw(n, rng):
    level = rng.integers(0, 3, n)
    return (rng.random(n) < np.array([.1, .2, .9])[level]).astype(float), level / 2

truth, widths, covered = np.mean([.1, .2, .9]), {k: [] for k in ('PPI', 'PPI++', 'RePPI')}, dict.fromkeys(('PPI', 'PPI++', 'RePPI'), 0)
for rep in range(200):
    rng = np.random.default_rng(rep)
    (y, f), (_, f_u) = draw(500, rng), draw(4500, rng)
    sample = PPIData(y, f, f_u)
    for label, method in (('PPI', PPI()), ('PPI++', PPIPlusPlus()), ('RePPI', RePPI(random_state=rep))):
        low, high = method(sample, target=Mean(), alpha=.05).confidence_interval
        widths[label].append(high - low)
        covered[label] += low <= truth <= high
print('B. Discrete autorater (Section 4.3), 200 repetitions')
for label in widths:
    print(f'  {label:6s} mean width={np.mean(widths[label]):.4f}  coverage={covered[label] / 200:.3f}')
