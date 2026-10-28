"""One PPI run: data and split -> target -> inference -> output."""
from sklearn.ensemble import RandomForestRegressor

from common.utils import split
from dataset.judge_sim import make_judge_data
from prediction_inference import PPI, Mean, PPIData

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
# 3. Combine two-sample residual correction with a Wald interval.
result = PPI()(data, target=target, alpha=.05)
# 4. Output.
print('Estimate:', result.estimate)
print('Confidence interval:', result.confidence_interval)
print('Diagnostics:', result.diagnostics)
