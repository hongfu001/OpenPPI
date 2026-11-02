"""EPI (Wang, Wen & Zou, 2025), equations (4)-(5), Theorems 1-3.

Just-identified mean and least-squares scores. No parametric likelihood or
explicit power tuning is used. Auxiliary constraints determine EL weights.
"""
from itertools import combinations_with_replacement

import numpy as np
from scipy.stats import chi2

from .base import InferenceMethod
from ..common.data import PPIData, RegressionData
from ..common.targets import Mean, LinearRegression
from ..common.regression import checked_inverse, designs, covariance, diagnostics


def empirical_weights(moments):
    """Maximize sum(log(w)) subject to sum(w)=1 and w @ moments=0.

    Solve the concave EL dual by damped Newton steps in an SVD-whitened
    coordinate system. Redundant/zero constraints are removed without changing
    their span. Infeasible or numerically unresolved constraints raise rather
    than silently falling back to a different estimator.
    """
    a = np.asarray(moments, dtype=float)
    if a.ndim != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('EL moments must be a finite matrix with at least two rows')
    n = len(a)
    if a.shape[1] == 0:
        return np.full(n, 1 / n), 0., 0
    scale = np.max(np.abs(a), axis=0)
    active = scale > scale.max() * max(a.shape) * np.finfo(float).eps
    a = a[:, active] / scale[active]
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    rank = int(np.sum(s > (s[0] * max(a.shape) * np.finfo(float).eps))) if len(s) else 0
    z = u[:, :rank] * np.sqrt(n)
    if rank == 0:
        return np.full(n, 1 / n), 0., 0
    if rank >= n:
        raise ValueError('EL constraints are infeasible: need an interior convex hull containing zero')
    lam = np.zeros(rank)
    for _ in range(150):
        denominator = 1 + z @ lam
        gradient = (z / denominator[:, None]).mean(axis=0)
        if np.max(np.abs(gradient)) < 1e-10:
            weights = 1 / denominator
            weights /= weights.sum()
            if np.max(np.abs(weights @ z)) < 1e-9:
                return weights, float(-np.log(n * weights).sum()), rank
        rows = z / denominator[:, None]
        try:
            step = np.linalg.solve(rows.T @ rows / n, gradient)
        except np.linalg.LinAlgError:
            break
        slope = z @ step
        rate = min(1., float(np.min(-.99 * denominator[slope < 0] / slope[slope < 0]))) if (slope < 0).any() else 1.
        objective = np.log(denominator).mean()
        while rate > 1e-14:
            trial = denominator + rate * slope
            if (trial > 0).all() and np.log(trial).mean() >= objective + 1e-4 * rate * (gradient @ step):
                break
            rate *= .5
        if rate <= 1e-14 or not np.isfinite(step).all():
            break
        lam += rate * step
    raise ValueError('EL constraints are infeasible or the dual solver did not converge; '
                     'reduce the basis dimension or collect more labeled observations')


def polynomial_basis(features, degree):
    """All monomials of total degree 1..degree, without a constant column."""
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    z = (features - center) / np.where(scale > 0, scale, 1.)
    columns = [np.prod(z[:, indices], axis=1)
               for power in range(1, degree + 1)
               for indices in combinations_with_replacement(range(z.shape[1]), power)]
    return np.column_stack(columns)


class _LeastSquares:
    def fit(self, features, labels):
        self.coef = np.linalg.lstsq(np.column_stack([np.ones(len(features)), features]), labels, rcond=None)[0]
        return self

    def predict(self, features):
        return np.column_stack([np.ones(len(features)), features]) @ self.coef


class EPI(InferenceMethod):
    """Empirical likelihood prediction-powered inference with finite unlabeled data.

    auxiliary='basis' uses polynomial features of predictions and, when passed
    RegressionData, covariates. auxiliary='crossfit' learns supervised scores
    with fresh learner_factory() objects supporting fit/predict; the default
    is NumPy least squares on the same polynomial features. Cross-fitting
    uses equal folds as in eq. (5), requiring n and m divisible by folds.

    interval_method='auto' uses corrected profile EL intervals for a mean and
    coordinatewise Wald intervals for regression. 'wald' works for both;
    'el' is mean-only. Finite-m EL uses Theorem 2's scaled chi-square cutoff,
    not an ordinary Wilks cutoff. standard_error always uses Theorem 1.
    Only just-identified scores are supported; distribution learning and
    over-identified scores are outside this implementation.
    """
    data_type = PPIData
    supported_targets = (Mean, LinearRegression)
    target_data_types = {LinearRegression: RegressionData}

    def __init__(self, auxiliary='basis', degree=1, folds=2, learner_factory=None,
                 random_state=0, interval_method='auto'):
        super().__init__(interval=self._interval)
        if auxiliary not in ('basis', 'crossfit'):
            raise ValueError("auxiliary must be 'basis' or 'crossfit'")
        if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or degree < 1:
            raise ValueError('degree must be a positive integer')
        if isinstance(folds, (bool, np.bool_)) or not isinstance(folds, (int, np.integer)) or folds < 2:
            raise ValueError('folds must be an integer >= 2')
        if learner_factory is not None and (auxiliary != 'crossfit' or not callable(learner_factory)):
            raise ValueError('learner_factory requires crossfit and must be callable')
        if interval_method not in ('auto', 'el', 'wald'):
            raise ValueError("interval_method must be 'auto', 'el' or 'wald'")
        self.auxiliary, self.degree, self.folds = auxiliary, degree, folds
        self.learner_factory, self.random_state = learner_factory, random_state
        self.interval_method = interval_method
        self.name = 'EPI' if auxiliary == 'basis' else 'EPI-CF'

    def _auxiliary(self, data, target, x):
        n, m = len(data.labels), len(data.unlabeled_predictions)
        features = np.concatenate([data.labeled_predictions, data.unlabeled_predictions])[:, None]
        if isinstance(data, RegressionData):
            features = np.column_stack([features, np.vstack([data.labeled_covariates, data.unlabeled_covariates])])
        basis = polynomial_basis(features, self.degree)
        if self.auxiliary == 'basis':
            return basis[:n] - basis.mean(axis=0), None
        k = self.folds
        if n % k or m % k or n // k < 2 or m // k < 2:
            raise ValueError('Cross-fitted EPI needs n and m divisible by folds and >= 2 rows per fold')
        rng = np.random.default_rng(self.random_state)
        labeled = rng.permutation(n).reshape(k, -1)
        unlabeled = rng.permutation(m).reshape(k, -1)
        h = None
        fold_ids = np.empty(n, dtype=int)
        for fold, (infer, unlab) in enumerate(zip(labeled, unlabeled)):
            train = np.setdiff1d(np.arange(n), infer)
            responses = data.labels[train]
            if isinstance(target, LinearRegression):
                if len(train) <= x.shape[1] or np.linalg.matrix_rank(x[train]) < x.shape[1]:
                    raise ValueError('Cross-fit training design needs full rank and more rows than coefficients')
                beta = np.linalg.lstsq(x[train], responses, rcond=None)[0]
                responses = x[train] * (x[train] @ beta - responses)[:, None]
            learner = (self.learner_factory or _LeastSquares)()
            learner.fit(basis[train], responses)
            hl = np.asarray(learner.predict(basis[infer]), dtype=float)
            hu = np.asarray(learner.predict(basis[n + unlab]), dtype=float)
            hl = hl[:, None] if hl.ndim == 1 else hl
            hu = hu[:, None] if hu.ndim == 1 else hu
            expected = 1 if isinstance(target, Mean) else x.shape[1]
            if hl.shape != (len(infer), expected) or hu.shape != (len(unlab), expected) or not all(np.isfinite(a).all() for a in (hl, hu)):
                raise ValueError('Learner must return finite predictions aligned with the supervised score')
            if h is None:
                h = np.empty((n, expected))
            h[infer] = hl - np.vstack([hl, hu]).mean(axis=0)
            fold_ids[infer] = fold
        return h, fold_ids

    def _estimate(self, data, target):
        n, m = len(data.labels), len(data.unlabeled_predictions)
        regression = isinstance(target, LinearRegression)
        if regression and self.interval_method == 'el':
            raise ValueError("Regression currently supports Wald intervals; use interval_method='wald' or 'auto'")
        x = designs(data, target)[0] if regression else None
        h, fold_ids = self._auxiliary(data, target, x)
        weights, base_loss, rank = empirical_weights(h)
        if regression:
            gram = x.T @ (weights[:, None] * x)
            inverse, condition = checked_inverse(gram, target.max_condition, name='EL weighted Gram matrix', positive=True)
            point = inverse @ (x.T @ (weights * data.labels))
            g = x * (x @ point - data.labels)[:, None]
        else:
            point = float(weights @ data.labels)
            inverse = np.ones((1, 1))
            g = (data.labels - point)[:, None]
        # Use a projection on centered h to estimate Cov(g,h) Cov(h)^-1 Cov(h,g).
        # This preserves PSD even with redundant polynomial columns.
        hc, gc = h - h.mean(axis=0), g - g.mean(axis=0)
        if fold_ids is not None:
            # Fold-specific centering constants have no within-fold variance.
            for fold in range(self.folds):
                mask = fold_ids == fold
                hc[mask] -= hc[mask].mean(axis=0)
        u, s, _ = np.linalg.svd(hc, full_matrices=False)
        keep = s > (s[0] * max(hc.shape) * np.finfo(float).eps) if len(s) else np.zeros(0, dtype=bool)
        fitted = u[:, keep] @ (u[:, keep].T @ gc)
        reduction = fitted.T @ fitted / (n - 1)
        var_g = covariance(g)
        gamma = n / (n + m)
        meat = var_g - (1 - gamma) * reduction
        cov = inverse @ meat @ inverse.T / n
        cov = (cov + cov.T) / 2
        se = np.sqrt(np.maximum(np.diag(cov), 0.))
        info = dict(labeled_count=n, unlabeled_count=m, auxiliary=self.auxiliary,
                    auxiliary_rank=rank, weights=weights, auxiliary_moments=h,
                    moment_residual=weights @ h, labeled_fraction=gamma,
                    variance_design='empirical_likelihood_pooled_centering',
                    interval_method='wald' if regression or self.interval_method == 'wald' else 'el')
        if fold_ids is not None:
            info.update(folds=self.folds, fold_ids=fold_ids)
        if regression:
            info.update(diagnostics(x, np.empty((m, x.shape[1])), target, condition, cov))
            info['variance_design'] = 'empirical_likelihood_pooled_centering'
            return point, se, info
        residual_var = float((var_g - reduction)[0, 0])
        if info['interval_method'] == 'el':
            if residual_var <= np.finfo(float).eps * max(float(var_g[0, 0]), np.finfo(float).tiny):
                raise ValueError('EL ratio inference needs positive residual score variance; use Wald for degenerate scores')
            ratio_scale = float(meat[0, 0] / residual_var)
            info['el_ratio_scale'] = ratio_scale
            info['values'] = dict(labels=data.labels, h=h, base_loss=base_loss, ratio_scale=ratio_scale)
        return point, float(se[0]), info

    def _interval(self, point, se, alpha, values=None):
        if values is None:
            from ..common.stats import wald_interval
            return wald_interval(point, se, alpha)
        y, h, base = values['labels'], values['h'], values['base_loss']
        cutoff = values['ratio_scale'] * chi2.ppf(1 - alpha, 1)

        def objective(theta):
            try:
                _, loss, _ = empirical_weights(np.column_stack([y - theta, h]))
                return max(0., 2 * (loss - base)) - cutoff
            except ValueError:
                return np.inf

        # EL feasible means are within the labeled outcome range. Bisection
        # tolerates infeasible (infinite LR) points near the convex-hull boundary.
        def endpoint(boundary):
            inside, outside = point, float(boundary)
            for _ in range(70):
                mid = inside + (outside - inside) / 2
                if mid == inside or mid == outside:
                    break
                if objective(mid) > 0:
                    outside = mid
                else:
                    inside = mid
                if abs(outside - inside) <= 1e-10 * max(abs(point), float(np.ptp(y)), np.finfo(float).tiny):
                    break
            return float((inside + outside) / 2)
        return endpoint(y.min()), endpoint(y.max())
