"""Extract numerical point/covariance blocks from official Census notebooks.

Usage: python generate_active_linear.py /path/to/active-inference /path/to/Robust-Active-Statistical-Inference
ASI commit 0f99f7ee6055b6f86c0ce88ece3cb2bf3c9c19cd.
RASI commit e8037bc113c62f420929f9a7c680a93f036674ec.
No survey data, folktables or model-training dependencies are needed.
"""
import ast
import json
from pathlib import Path
import sys
import textwrap
import numpy as np

rng=np.random.default_rng(191)
fixtures=[]
for index,(root,notebook,point_name,label_name,sigma_name) in enumerate([
    (sys.argv[1],'census-analysis.ipynb','pointest_active','active_labels','Sigma_active'),
    (sys.argv[2],'census_data_analysis.ipynb','pointest_active_robust','active_robust_label','Sigma_active_robust')]):
    root=Path(root)
    tree=ast.parse((root/'utils.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ols')
    scope={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(root/'utils.py'),'exec'),scope)
    raw=rng.normal(size=(80,2)); x=np.c_[np.ones(80),raw]
    y=1+.8*raw[:,0]-.6*raw[:,1]+rng.normal(size=80)
    f=1+.5*raw[:,0]-.4*raw[:,1]
    pi=rng.uniform(.2,.8,80); mask=rng.random(80)<pi
    if index==1: pi[:15]=1.; mask[:15]=True
    pseudo=f+mask/pi*(y-f)
    notebook_data=json.loads((root/notebook).read_text())
    token=f'{point_name} = ols('
    source=next(''.join(c['source']) for c in notebook_data['cells'] if token in ''.join(c['source']))
    start=source.index(token)
    stop=source.index(f'{point_name}_std',start)
    # Dedent the extracted body; its first line starts at the token itself.
    line_start=source.rfind('\n',0,start)+1
    block=textwrap.dedent(source[line_start:stop].rsplit('\n',1)[0])
    scope.update(X=x,X_order=x,n=len(x),Hessian_inv=np.linalg.inv(x.T@x/len(x)))
    scope[label_name]=pseudo
    exec(compile(block,notebook,'exec'),scope)
    observed=[float(y[i]) if mask[i] else None for i in range(len(y))]
    fixtures.append(dict(method='ASI' if index==0 else 'RASI',
        predictions=f.tolist(),observed_labels=observed,queried=mask.tolist(),
        propensities=pi.tolist(),covariates=raw.tolist(),
        expected=dict(estimate=scope[point_name].tolist(),covariance=(scope[sigma_name]/len(x)).tolist())))
print(json.dumps(fixtures,indent=2))
