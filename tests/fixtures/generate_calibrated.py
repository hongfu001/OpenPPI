"""Generate fixtures: python generate_calibrated.py /path/to/ppi-aipw.

Official reference commit: 9b23849970ade47aa4264ce190680ad86e416e46.
Uses its sklearn isotonic backend and disables optional efficiency scaling.
"""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
from ppi_aipw import mean_inference

cases = [
    dict(labels=[0,1,0,1,1,0,1,1], labeled_predictions=[.1,.2,.2,.4,.5,.6,.8,.9],
         unlabeled_predictions=[.05,.15,.3,.5,.7,.85]),
    dict(labels=[0,1,1,0,1,0], labeled_predictions=[.5]*6, unlabeled_predictions=[.5]*8),
    dict(labels=[1,0,1,1,0,0,1,0], labeled_predictions=[.1,.2,.3,.4,.5,.6,.7,.8],
         unlabeled_predictions=[.1,.2,.4,.6,.8]),
]
for case in cases:
    case['expected'] = {}
    for method in ('isotonic', 'linear'):
        r = mean_inference(np.array(case['labels']), np.array(case['labeled_predictions']),
                           np.array(case['unlabeled_predictions']), method=method,
                           isocal_backend='sklearn', alpha=.05, efficiency_maximization=False)
        case['expected'][method] = dict(estimate=float(r.pointestimate), se=float(r.se), ci=list(r.ci))
print(json.dumps(cases, indent=2))
