"""One Cluster-Robust PPI++ run on clustered data, with partially and fully labeled clusters."""
import numpy as np

from dataset.judge_sim import make_clustered_data
from prediction_inference import ClusterRobustPPI, ClusteredPPIData, PPIPlusPlus, Mean, PPIData

# 1. Units nested in iid clusters; the fixed predictor is biased upwards by 0.05. Truth = 0.5.
frame = make_clustered_data(clusters=300, seed=0)
y, f, cluster = frame['label'].to_numpy(), frame['prediction'].to_numpy(), frame['cluster'].to_numpy()
rng = np.random.default_rng(42)
designs = {'partially labeled': rng.random(len(frame)) < .3,          # each unit with probability 0.3
           'whole clusters': (rng.random(cluster.max() + 1) < .3)[cluster]}   # each cluster with probability 0.3
# 2. Select the target.
target = Mean()
for design, labeled in designs.items():
    data = ClusteredPPIData(np.where(labeled, y, np.nan), f, labeled, cluster)
    # 3. Labels only and Cluster-Robust PPI++ use cluster-robust SEs; PPI++ (i.i.d.) ignores the clusters.
    methods = (('labels only', lambda: ClusterRobustPPI(lam=0)(data, target=target)),
               ('PPI++ (i.i.d.)', lambda: PPIPlusPlus()(PPIData(y[labeled], f[labeled], f[~labeled]), target=target)),
               ('Cluster-Robust PPI++', lambda: ClusterRobustPPI()(data, target=target)))
    print(design)
    for label, run in methods:
        result = run()
        low, high = result.confidence_interval
        # 4. Output.
        print(f'  {label:21s} estimate={result.estimate:.4f} CI=({low:.4f}, {high:.4f}) width={high - low:.4f}')
