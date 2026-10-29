"""One PPI++ run: same split as ppi_mean.py, compared with classical (lam=0) and PPI (lam=1)."""
from sklearn.ensemble import RandomForestRegressor

from common.utils import split
from dataset.judge_sim import make_judge_data
from prediction_inference import PPIPlusPlus, Mean, PPIData

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
# 3. Tune lambda by the plug-in estimate; fixed lambdas give the two baselines.
for label, method in (('classical', PPIPlusPlus(lam=0)), ('PPI', PPIPlusPlus(lam=1)), ('PPI++', PPIPlusPlus())):
    result = method(data, target=target, alpha=.05)
    low, high = result.confidence_interval
    # 4. Output.
    print(f'{label:9s} lambda={result.diagnostics["prediction_weight"]:.3f} '
          f'estimate={result.estimate:.4f} CI=({low:.4f}, {high:.4f}) width={high - low:.4f}')
