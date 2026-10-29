"""One FAB-PPI run: same split as ppi_mean.py, compared with PPI and PPI++."""
from sklearn.ensemble import RandomForestRegressor

from common.utils import split
from dataset.judge_sim import make_judge_data
from prediction_inference import FABPPI, PPI, PPIPlusPlus, Mean, PPIData

# 1. Split iid rows; fit only on the independent training sample.
frame = make_judge_data(n=2000, seed=0)
train, labeled, unlabeled = split(
    frame.drop(columns='label').to_numpy(), frame['label'].to_numpy(),
    train_size=400, labeled_size=400, random_state=42)
model = RandomForestRegressor(n_estimators=60, min_samples_leaf=8,
                              max_depth=8, random_state=0, n_jobs=1)
model.fit(*train)
data = PPIData(labels=labeled[1], labeled_predictions=model.predict(labeled[0]),
               unlabeled_predictions=model.predict(unlabeled))
# 2. Select the target.
target = Mean()
# 3. Horseshoe prior on the rectifier; the last two ignore the uncertainty in mean(f_unlabeled).
methods = (('PPI', PPI()), ('PPI++', PPIPlusPlus()), ('FAB-PPI', FABPPI()), ('FAB-PPI++', FABPPI(lam=None)),
           ('FAB-PPI (N>>n)', FABPPI(imputed_uncertainty=False)),
           ('FAB-PPI++ (N>>n)', FABPPI(lam=None, imputed_uncertainty=False)))
for label, method in methods:
    result = method(data, target=target, alpha=.05)
    low, high = result.confidence_interval
    # 4. Output.
    print(f'{label:17s} estimate={result.estimate:.4f} CI=({low:.4f}, {high:.4f}) width={high - low:.4f}')
