"""MultiPPI allocation (Cowen-Breen et al., ICLR 2026): how often to query each subset of sources.

X = (X_1, ..., X_k) stacks the label X_1 and k - 1 predictions. Querying the
subset I (an index tuple) once costs c_I; with n_I queries of each I the
minimum-variance unbiased linear estimator of a^T E[X] has variance

    V(n) = a^T (sum_I n_I P_I^T Sigma_I^-1 P_I)^+ a,

which is convex in n (Theorem 2 / Appendix G). MultiAllocate minimizes it
over n >= 0 subject to linear budgets sum_I c_I^(l) n_I <= B^(l) and optional
per-subset caps (e.g. n_full <= N labeled rows), then floors n as in
Appendix C.2. The single-budget SOCP and the multi-budget SDP of the paper
are both this problem; it is solved here with SciPy instead of cvxpy.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .._stats import multi_weights


@dataclass(frozen=True)
class Allocation:
    """counts: floored sample sizes (subsets with fewer than 2 dropped); weights: lambda_I
    for those counts under the covariance used; variance: a^T S a for the floored counts;
    fractional: the continuous optimum before flooring."""
    counts: dict
    weights: dict
    variance: float
    fractional: dict


def _subsets(costs, k):
    subsets = [tuple(sorted(s)) for s in costs]
    if len(set(subsets)) != len(subsets) or any(not s or s[0] < 0 or s[-1] >= k for s in subsets):
        raise ValueError(f'subsets must be distinct, non-empty index tuples within range({k})')
    return subsets


def multi_allocate(covariance, costs, budget, a=None, caps=None):
    """Return the Allocation minimizing V(n) under the budgets.

    covariance: k x k matrix (known or estimated, e.g. ledoit_wolf on fully labeled rows).
    costs: {subset: cost} with a scalar cost, or a length-m vector for m budget constraints.
    budget: scalar or length-m vector. a: target weights, default e_1 = mean of X_1.
    caps: optional {subset: maximum count}, e.g. {full_set: N} for N available labels.
    """
    sigma = np.asarray(covariance, dtype=float)
    k = len(sigma)
    if sigma.shape != (k, k) or not np.allclose(sigma, sigma.T) or np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError('covariance must be a symmetric positive-definite matrix')
    a = np.eye(k)[0] if a is None else np.asarray(a, dtype=float)
    subsets = _subsets(costs, k)
    cost = np.array([np.atleast_1d(np.asarray(costs[s], dtype=float)) for s in subsets])   # |I| x m
    limit = np.atleast_1d(np.asarray(budget, dtype=float))
    if cost.shape[1] != len(limit) or (cost < 0).any() or (limit <= 0).any():
        raise ValueError('costs and budget need matching dimensions, nonnegative costs and positive budgets')
    caps = {tuple(sorted(s)): float(v) for s, v in (caps or {}).items()}
    upper = np.array([min([caps.get(s, np.inf)] + [b / c for b, c in zip(limit, row) if c > 0])
                      for s, row in zip(subsets, cost)])
    if not np.isfinite(upper).all():
        raise ValueError('every subset needs a positive cost or a cap')
    blocks = []
    for s in subsets:
        block = np.zeros((k, k))
        block[np.ix_(s, s)] = np.linalg.inv(sigma[np.ix_(s, s)])
        blocks.append(block)
    ridge = 1e-12 * np.trace(sum(blocks))

    def objective(n):
        # log V is scale-free, and its minimizer over the polytope is that of the convex V.
        precision = sum(ni * b for ni, b in zip(n, blocks)) + ridge * np.eye(k)
        u = np.linalg.solve(precision, a)
        value = a @ u
        return np.log(value), -np.array([u @ b @ u for b in blocks]) / value

    start = upper / (2 * len(subsets))    # each c_I * upper_I <= B, so this point is feasible
    result = minimize(objective, start, jac=True, method='SLSQP', bounds=list(zip(np.zeros(len(upper)), upper)),
                      constraints=[{'type': 'ineq', 'fun': lambda n: limit - n @ cost, 'jac': lambda n: -cost.T}],
                      options={'ftol': 1e-12, 'maxiter': 500})
    fractional = {s: float(v) for s, v in zip(subsets, np.maximum(result.x, 0))}
    counts = {s: int(np.floor(v + 1e-9)) for s, v in fractional.items()}
    counts = {s: n for s, n in counts.items() if n >= 2}
    weights, variance = multi_weights(sigma, counts, a)
    return Allocation(counts, weights, variance, fractional)
