"""Reference fixtures from PPITSAS commit 5850ddc693bb3d066a18b47881e209df642a2f73.

Usage: python tests/fixtures/generate_time_series.py /path/to/ppitsas/utils/methods.py
Only execute the official numerical functions, not its training or experiment code.
Print JSON to stdout; the checked-in fixture needs no network or reference dependencies.
"""
import ast
import json
from pathlib import Path
import sys
import numpy as np
from scipy import stats

source = Path(sys.argv[1]).read_text()
names = {'newey_west_bartlett_bandwidth', 'default_newey_west_lag',
         'newey_west_var_of_mean', 'ppitsas'}
nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
assert {node.name for node in nodes} == names
reference = {'np': np, 'stats': stats}
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<official ppitsas numerical functions>', 'exec'), reference)
rng = np.random.default_rng(42)

def ar(n, phi):
    x = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi*x[i-1] + np.sqrt(1-phi**2)*x[i]
    return x

cases = []
for name, n, m, phi, lag in [('plugin', 40, 70, .8, None),
                             ('negative', 35, 60, -.7, None),
                             ('fixed', 20, 30, .6, 3),
                             ('lag_zero', 12, 18, .8, 0),
                             ('short', 2, 3, .5, None),
                             ('capped', 3, 4, .5, 100),
                             ('constant', 6, 8, 0., None)]:
    f, g, r = ar(n, phi), ar(m, phi), ar(n, phi) + .3
    if name == 'constant':
        f, g, r = np.full(n, 2.), np.full(m, 2.), np.full(n, .5)
    y = f + r
    point, low, high = reference['ppitsas'](None, y, None, f, g, max_lag=lag)
    lag_f = reference['default_newey_west_lag'](g) if lag is None else lag
    lag_r = reference['default_newey_west_lag'](r) if lag is None else lag
    cases.append(dict(name=name, labels=y.tolist(), labeled_predictions=f.tolist(),
                      unlabeled_predictions=g.tolist(), max_lag=lag,
                      expected=dict(estimate=float(point), ci=[float(low),float(high)],
                                    unlabeled_lag=min(lag_f,m-1), residual_lag=min(lag_r,n-1),
                                    unlabeled_mean_variance=reference['newey_west_var_of_mean'](g,lag),
                                    residual_mean_variance=reference['newey_west_var_of_mean'](r,lag))))
print(json.dumps(dict(commit='5850ddc693bb3d066a18b47881e209df642a2f73', cases=cases), indent=2))
