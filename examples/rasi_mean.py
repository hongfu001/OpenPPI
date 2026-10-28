"""One RASI run: data -> fixed-model predictions -> burn-in batch -> robust batch -> inference."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from prediction_inference import (RASI, Mean, RobustDesign, select_burn_in,
                                  select_for_labeling)
from dataset.judge_sim import make_judge_data

# 1. iid rows: an independent training sample for f, the rest is the unlabeled pool.
data = make_judge_data(n=2600, seed=0)
features = data.drop(columns='label').to_numpy()
y = data['label'].to_numpy()
model = RandomForestRegressor(n_estimators=60, min_samples_leaf=8, max_depth=8,
                              random_state=0, n_jobs=1).fit(features[:600], y[:600])
pool, pool_y = features[600:], y[600:]
# 2. Select the target and predict with the fixed model.
target = Mean()
predictions = model.predict(pool)
budget, burn_in = 400, 100
rng = np.random.default_rng(42)             # one stream keeps the two batches' draws independent
# 3. Batch 1: a uniform random burn-in, every row labeled (counts toward the budget).
burn = select_burn_in(len(pool), size=burn_in, random_state=rng)
burn_labels = pool_y[burn.indices]          # in practice, export burn.indices to annotators
# 4. Between batches: fit the error model on burn-in labels; 5-fold CV picks the l2 radius.
rest = burn.remaining
design = RobustDesign.fit(
    pool[burn.indices], burn_labels, predictions[burn.indices],
    error_model_factory=lambda: RandomForestRegressor(
        n_estimators=60, min_samples_leaf=8, max_depth=6, random_state=0, n_jobs=1),
    rho=(budget - burn_in) / len(rest), path='geometric', random_state=rng.integers(2**32),
)
# 5. Batch 2: robust selection on the remaining rows, same interface as ASI.
request = select_for_labeling(predictions[rest], target=target, design=design,
                              budget=budget - burn_in, scores=design.errors(pool[rest]),
                              random_state=rng)
print('Rows to label: burn-in', len(burn.indices), '+ robust', len(request.indices))
labels = pool_y[rest[request.indices]]      # row numbers in the pool are rest[request.indices]
# 6. Merge both batches and run inference.
record = burn.combine(predictions, burn_labels, request, labels)
result = RASI()(record, target=target, alpha=.05)
print('Estimate:', result.estimate)
print('Confidence interval:', result.confidence_interval)
print('Diagnostics:', result.diagnostics)
