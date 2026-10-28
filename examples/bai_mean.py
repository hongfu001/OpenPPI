"""One BAI batch: data -> fixed-model predictions -> cube-balanced selection -> label -> inference."""
from model.judge import JudgeModel
from prediction_inference import BAI, BalancedDesign, Mean, select_for_labeling
from dataset.judge_sim import make_judge_data

# 1. Separate features and labels from the data provider's DataFrame.
data = make_judge_data(n=2000, seed=0)
features = data.drop(columns='label')
judge_cols = [i for i, name in enumerate(features.columns) if name.startswith('judge')]
features = features.to_numpy()
y = data['label'].to_numpy()
# 2. Select the target and predict with a fixed model (any already trained model works).
target = Mean()
model = JudgeModel(columns=judge_cols)
predictions = model.predict(features)
# 3. Same probabilities as ASI, but the cube method selects rows jointly so that the
#    weighted uncertainty of the batch matches the pool total. No label is read here.
#    With a signed error model e(x) ~ f(x) - y, pass BalancedDesign(balancing=e) and scores=|e|.
request = select_for_labeling(predictions, target=target, design=BalancedDesign(tau=.5),
                              budget=400, random_state=42)
print('Rows to label:', len(request.indices))
# 4. Human labeling. The simulation looks up y; in practice, export request.indices to annotators.
labels = y[request.indices]
# 5. Attach labels (-> BalancedActiveData) and run inference with the cube variance estimator.
record = request.with_labels(labels)
result = BAI()(record, target=target, alpha=.05)
print('Estimate:', result.estimate)
print('Confidence interval:', result.confidence_interval)
print('Diagnostics:', result.diagnostics)
