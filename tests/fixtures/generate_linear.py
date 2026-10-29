"""Generate official fixed-weight OLS fixtures; args: ppi_py checkout, pspa checkout.

PPI reference 3d1f0c668444907b39bd045cb9dd38e479ce7dd6.
PSPA reference 8e77b124d58b884730484d1923d909a61ed70025.
Matched-Gram designs isolate the exact agreement with PPI's sum-of-OLS API.
PSPA fixtures retain its one-step estimates and weights; covariance differs
intentionally because the reference expression has nonsymmetric cross-terms.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
sys.path.insert(0,str(Path(sys.argv[1])))
from ppi_py import ppi_ols_pointestimate, ppi_ols_ci

rng=np.random.default_rng(401)
x=rng.normal(size=(40,2)); z=x.copy()
y=1+x@np.array([.8,-.6])+rng.normal(size=40)
f=1+x@np.array([.5,-.3])+rng.normal(size=40)*.3
g=1+z@np.array([.5,-.3])+rng.normal(size=40)*.3
v,u=np.c_[np.ones(40),x],np.c_[np.ones(40),z]
fixture=dict(labeled_covariates=x.tolist(),unlabeled_covariates=z.tolist(),labels=y.tolist(),labeled_predictions=f.tolist(),unlabeled_predictions=g.tolist(),ppi={})
for lam in [0.,.4,1.]:
 beta=ppi_ols_pointestimate(v,y,f,u,g,lam=lam)
 lo,hi=ppi_ols_ci(v,y,f,u,g,lam=lam,alpha=.05)
 fixture['ppi'][str(lam)]=dict(estimate=beta.tolist(),lower=lo.tolist(),upper=hi.tolist())
# Use a different Gram matrix for the PSPA fixture.
z=rng.normal(size=(70,2))@np.array([[1.,.3],[.2,1.]])
g=1+z@np.array([.5,-.3])+rng.normal(size=70)*.3
pspa=dict(labeled_covariates=x.tolist(),unlabeled_covariates=z.tolist(),labels=y.tolist(),labeled_predictions=f.tolist(),unlabeled_predictions=g.tolist())
with tempfile.TemporaryDirectory() as tmp:
 input_file=Path(tmp)/'input.json'; output_file=Path(tmp)/'output.json'
 input_file.write_text(json.dumps(pspa))
 expr='''args<-commandArgs(trailingOnly=TRUE)
source(file.path(args[1], 'R/utils.R')); source(file.path(args[1], 'R/main.R'))
d<-jsonlite::fromJSON(args[2])
r<-pspa_y(X_lab=d$labeled_covariates,X_unlab=d$unlabeled_covariates,Y_lab=d$labels,Yhat_lab=d$labeled_predictions,Yhat_unlab=d$unlabeled_predictions,method='ols')
jsonlite::write_json(list(estimate=r$Estimate,weights=r$Weight),args[3],digits=16,auto_unbox=FALSE)
'''
 script=Path(tmp)/'generate.R'; script.write_text(expr)
 subprocess.run(['Rscript',str(script),sys.argv[2],str(input_file),str(output_file)],check=True)
 pspa['expected']=json.loads(output_file.read_text())
print(json.dumps(dict(ppi=fixture,pspa=pspa),indent=2))
