"""One FTRL batch: data -> fixed-model predictions -> covariate-oblivious selection -> label -> inference."""
from model.judge import JudgeModel
from prediction_inference import ASI, FTRLDesign, Mean, select_for_labeling
from dataset.judge_sim import make_judge_data

# 1. Separate features and labels from the data provider's DataFrame (rows in arrival order).
data = make_judge_data(n=2000, seed=0)
features = data.drop(columns='label')
judge_cols = [i for i, name in enumerate(features.columns) if name.startswith('judge')]
features = features.to_numpy()
y = data['label'].to_numpy()
# 2. Select the target and predict with a fixed model (any already trained model works).
target = Mean()
model = JudgeModel(columns=judge_cols)
predictions = model.predict(features)
# 3. FTRL sets p_t from the oracle Phi of earlier rows only; Phi ~ E[(Y - f)^2 | X],
#    here f(1 - f) for a calibrated binary model. p_t starts at tau/8 and rises to tau.
request = select_for_labeling(predictions, target=target, design=FTRLDesign(),
                              scores=predictions * (1 - predictions), budget=400, random_state=42)
print('Rows to label:', len(request.indices), '| first p_t:', request.propensities[:3].round(3))
# 4. Human labeling. The simulation looks up y; in practice, export request.indices to annotators.
labels = y[request.indices]
# 5. The estimator and interval are those of ASI.
record = request.with_labels(labels)
result = ASI()(record, target=target, alpha=.05)
print('Estimate:', result.estimate)
print('Confidence interval:', result.confidence_interval)
print('Diagnostics:', result.diagnostics)
