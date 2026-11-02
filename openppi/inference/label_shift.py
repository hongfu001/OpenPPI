"""Lee, Ma & Zhao (JMLR 2026): efficient inference under label shift.

Algorithms 1-2 for the target mean, eqs. (9)-(12), (16), and discrete
specialization in Section 5. Continuous Fredholm solves use a configurable
truncated SVD; their finite-sample accuracy depends on smoothing/regularization.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax

from .base import InferenceMethod
from ..common.data import LabelShiftData


def _kernel_rows(left, right, bandwidth):
    return softmax(-cdist(left, right, 'sqeuclidean') / (2 * bandwidth ** 2), axis=1)


def _inverse(operator, rcond, *, discrete):
    u, s, vt = np.linalg.svd(operator, full_matrices=False)
    keep = s > s[0] * rcond
    rank = int(keep.sum())
    if rank == 0 or (discrete and rank != len(s)) or (not discrete and rank < 2):
        raise ValueError('Label-shift operator is not identifiable at the requested tolerance; '
                         'features/conditional probabilities must distinguish label distributions')
    inverse = (vt[keep].T / s[keep]) @ u[:, keep].T
    return inverse, dict(rank=rank, dimension=len(s),
                        retained_condition_number=float(s[0] / s[keep][-1]))


class EfficientLabelShift(InferenceMethod):
    """Estimate E_Q[Y] using labeled P and unlabeled Q with P(X|Y)=Q(X|Y).

    outcome='discrete' (default) uses exact class-conditional averages and a
    finite matrix solve (Section 5). 'continuous' uses Gaussian kernels in Y
    and a truncated-SVD Fredholm solve. No discretization into label bins is
    used for the continuous case. Only the Mean target is currently supported.

    conditional_bandwidth is in source-standardized feature units (default
    3*n**(-1/7), following the paper's simulation bandwidth). Y bandwidths
    default to h=.5*n**(-1/16) and l=1.5*n**(-1/3), in original outcome units.
    rcond defaults to 1e-10 for discrete and 1e-3 for continuous outcomes.
    refine_ratio adds Section 3.2's efficient density-ratio refinement before
    the final mean solve; the initial consistent ratio suffices for Algorithm 2.

    ratio_bounds=None rejects nonpositive density estimates. An explicit
    (positive_lower, upper) opts into clipping followed by source normalization;
    diagnostics report this stabilization, which can affect inference when
    active. Efficiency and Wald validity require the paper's nuisance-rate,
    identifiability, support, and smoothing conditions; numerical inversion
    alone does not ensure them.
    """
    name = 'LabelShift'
    data_type = LabelShiftData

    def __init__(self, outcome='discrete', conditional_bandwidth=None,
                 density_bandwidth=None, inverse_bandwidth=None, rcond=None,
                 initial_ratio=None, refine_ratio=False, ratio_bounds=None, **kwargs):
        super().__init__(**kwargs)
        if outcome not in ('discrete', 'continuous'):
            raise ValueError("outcome must be 'discrete' or 'continuous'")
        for name, value in (('conditional_bandwidth', conditional_bandwidth),
                            ('density_bandwidth', density_bandwidth), ('inverse_bandwidth', inverse_bandwidth)):
            if value is not None and (not np.isscalar(value) or not np.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive')
        rcond = (1e-10 if outcome == 'discrete' else 1e-3) if rcond is None else rcond
        if not np.isscalar(rcond) or not np.isfinite(rcond) or not 0 < rcond < 1:
            raise ValueError('rcond must lie in (0, 1)')
        if ratio_bounds is not None:
            limits = np.asarray(ratio_bounds, dtype=float)
            if limits.shape != (2,) or not np.isfinite(limits).all() or not 0 < limits[0] < 1 < limits[1]:
                raise ValueError('ratio_bounds must satisfy 0 < lower < 1 < upper, both finite')
            ratio_bounds = tuple(limits)
        if not isinstance(refine_ratio, (bool, np.bool_)):
            raise ValueError('refine_ratio must be boolean')
        if initial_ratio is not None and not callable(initial_ratio):
            raise ValueError('initial_ratio must be a callable of label values or None')
        self.outcome, self.rcond = outcome, float(rcond)
        self.conditional_bandwidth = conditional_bandwidth
        self.density_bandwidth, self.inverse_bandwidth = density_bandwidth, inverse_bandwidth
        self.initial_ratio, self.refine_ratio, self.ratio_bounds = initial_ratio, refine_ratio, ratio_bounds

    def _estimate(self, data, target):
        y = data.labels
        n, m = len(y), len(data.unlabeled_features)
        discrete = self.outcome == 'discrete'
        if discrete:
            support, index, counts = np.unique(y, return_inverse=True, return_counts=True)
            if len(support) < 2 or counts.min() < 2:
                raise ValueError('Discrete label shift needs at least two labels and two source observations per class')
            indicator = np.eye(len(support))[index]
            reverse = indicator.T / counts[:, None]
            source_mass = counts / n
            response = support
        else:
            if np.unique(y).size < 3:
                raise ValueError('Continuous label shift needs at least three distinct outcomes')
            support, index = y, np.arange(n)
            l = self.inverse_bandwidth or 1.5 * n ** (-1 / 3)
            reverse = _kernel_rows(y[:, None], y[:, None], l)
            response = y
        bw = self.conditional_bandwidth or 3 * n ** (-1 / 7)
        conditional = data.conditional_probabilities
        if conditional is None:
            source_x, target_x = data.labeled_features, data.unlabeled_features
            scale = source_x.std(axis=0)
            scale = np.where(scale > 0, scale, 1.)
            source_z = (source_x - source_x.mean(axis=0)) / scale
            target_z = (target_x - source_x.mean(axis=0)) / scale
            conditional = _kernel_rows(np.vstack([source_z, target_z]), source_z, bw)
            if discrete:
                conditional = conditional @ indicator
        if conditional.shape[1] != len(support):
            raise ValueError('Conditional probability columns must match sorted source classes '
                             '(discrete) or source label row order (continuous)')
        if not discrete:
            h = self.density_bandwidth or .5 * n ** (-1 / 16)
            density_kernel = np.exp(-cdist(y[:, None], y[:, None], 'sqeuclidean') / (2 * h ** 2))
            # The 1/(h sqrt(2 pi)) factors cancel in the q/p density ratio.
        solves = []
        stabilizations = []

        def source_values(ratio):
            return ratio[index] if discrete else ratio

        def calibrated(ratio):
            denominator = conditional @ (ratio ** 2 + n / m * ratio)
            if not np.isfinite(denominator).all() or (denominator <= 0).any():
                raise ValueError('Conditional density-ratio denominator must be finite and positive')
            b_operator = conditional * ratio[None, :] / denominator[:, None]
            operator = reverse @ b_operator[:n]
            inverse, info = _inverse(operator, self.rcond, discrete=discrete)
            solves.append(info)
            return b_operator, inverse

        def stabilize(ratio):
            if not np.isfinite(ratio).all():
                raise ValueError('Estimated density ratio is not finite')
            if self.ratio_bounds is None:
                if (ratio <= 0).any():
                    raise ValueError('Estimated target density is nonpositive; use more informative data, '
                                     'adjust smoothing, or explicitly choose ratio_bounds for stabilization')
                stabilizations.append(0.)
                return ratio
            clipped = np.clip(ratio, *self.ratio_bounds)
            stabilizations.append(float(np.mean(clipped != ratio)))
            # This normalization makes E_P[rho] = 1 after numerical clipping.
            return clipped / np.mean(source_values(clipped))

        def estimate_ratio(working):
            b_operator, inverse = calibrated(working)
            source_working = source_values(working)
            contrast = b_operator[n:].mean(axis=0) - (source_working[:, None] * b_operator[:n]).mean(axis=0)
            if discrete:
                # Apply Algorithm 1 simultaneously to all class indicators.
                q = source_mass * working + contrast @ inverse
                return stabilize(q / source_mass)
            coefficients = source_working / n + contrast @ inverse
            return stabilize((coefficients @ density_kernel) / density_kernel.mean(axis=0))

        working = np.ones(len(support)) if self.initial_ratio is None else np.asarray(self.initial_ratio(support), dtype=float)
        if working.shape != support.shape or not np.isfinite(working).all() or (working <= 0).any():
            raise ValueError('initial_ratio must return a positive finite vector aligned with source label support')
        initial = estimate_ratio(working)
        ratio = estimate_ratio(initial) if self.refine_ratio else initial
        b_operator, inverse = calibrated(ratio)
        auxiliary = b_operator @ (inverse @ response)
        residual = source_values(ratio) * (y - auxiliary[:n])
        point = float(auxiliary[n:].mean() + residual.mean())
        source_variance = float(residual.var(ddof=1) / n)
        target_variance = float(auxiliary[n:].var(ddof=1) / m)
        se = float(np.sqrt(source_variance + target_variance))
        info = dict(labeled_count=n, unlabeled_count=m, outcome=self.outcome,
                    label_support=support, initial_density_ratio=initial, density_ratio=ratio,
                    source_density_ratio=source_values(ratio), refined_ratio=self.refine_ratio,
                    ratio_clipped_fractions=stabilizations, operator_solves=solves,
                    conditional_model='supplied' if data.conditional_probabilities is not None else 'nadaraya_watson',
                    conditional_bandwidth=bw if data.conditional_probabilities is None else None,
                    predictions=auxiliary, source_residual_mean=float(residual.mean()),
                    fredholm_residual=float(np.max(np.abs(reverse @ auxiliary[:n] - response))),
                    source_variance=source_variance, target_variance=target_variance,
                    variance_design='label_shift_influence_function',
                    rcond=self.rcond)
        if discrete:
            info['estimated_target_probabilities'] = source_mass * ratio
        else:
            info.update(density_bandwidth=h, inverse_bandwidth=l)
        return point, se, info
