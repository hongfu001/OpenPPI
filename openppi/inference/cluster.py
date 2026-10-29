"""Cluster-Robust PPI++ (Broska & Howes 2026) for the unit-level mean."""
import numpy as np
from .base import InferenceMethod
from ..common.data import ClusteredPPIData


def cluster_moments(data, theta):
    """Return Sigma-hat (2 x 2) from the per-cluster sums (eps_g, Z_g), eqs. 6-7.

    eps_g = sum xi (Y - theta); Z_g = sum (xi - n/N (1 - xi)) (f - mean f), with
    mean f over all units. Sigma-hat = sum_g [eps_g, Z_g]^T [eps_g, Z_g] / G.
    """
    xi, f = data.labeled, data.predictions
    n, big_n = xi.sum(), (~xi).sum()
    eps = np.where(xi, np.nan_to_num(data.labels) - theta, 0.)
    z = np.where(xi, 1., -n / big_n) * (f - f.mean())
    _, cluster = np.unique(data.clusters, return_inverse=True)
    sums = np.stack([np.bincount(cluster, eps), np.bincount(cluster, z)], axis=1)
    return sums.T @ sums / len(sums), sums


class ClusterRobustPPI(InferenceMethod):
    """Binary unit-level mean PPI++ with cluster-robust standard errors.

    Point estimate lam * mean(f_unlabeled) + mean(Y - lam f) over labeled units.
    SE^2 = G / n^2 (S11 - 2 lam S12 + lam^2 S22) (Theorem 1). lam=None uses
    lam = S12 / S22 with Sigma-hat evaluated at the lam=1 estimate (eq. 12);
    lam=0 is the labels-only mean with a cluster-robust SE. Singleton clusters
    recover PPI++ with sample moments taken without ddof.
    """
    name = 'Cluster-Robust PPI++'
    data_type = ClusteredPPIData

    def __init__(self, lam=None, **kwargs):
        super().__init__(**kwargs)
        if lam is not None and not np.isfinite(lam):
            raise ValueError('lam must be finite or None')
        self.lam = lam

    @staticmethod
    def _point(data, lam):
        xi, f = data.labeled, data.predictions
        return float(lam * f[~xi].mean() + (data.labels[xi] - lam * f[xi]).mean())

    def _estimate(self, data, target):
        if self.lam is None:
            sigma, _ = cluster_moments(data, self._point(data, 1.))
            lam = float(sigma[0, 1] / sigma[1, 1]) if sigma[1, 1] > 0 else 0.
        else:
            lam = float(self.lam)
        point = self._point(data, lam)
        sigma, sums = cluster_moments(data, point)
        n, groups = int(data.labeled.sum()), len(sums)
        meat = sigma[0, 0] - 2 * lam * sigma[0, 1] + lam ** 2 * sigma[1, 1]
        cluster = np.unique(data.clusters, return_inverse=True)[1]
        labeled_per_cluster, sizes = np.bincount(cluster, data.labeled), np.bincount(cluster)
        return point, float(np.sqrt(groups * max(meat, 0.)) / n), {
            'values': {'cluster_rectifier': sums[:, 0], 'cluster_control': sums[:, 1]},
            'labeled_count': n,
            'unlabeled_count': int((~data.labeled).sum()),
            'cluster_count': groups,
            'partially_labeled_clusters': int(((labeled_per_cluster > 0) & (labeled_per_cluster < sizes)).sum()),
            'prediction_weight': lam,
            'weight_strategy': 'tuned' if self.lam is None else 'fixed',
            'ppi_correlation': float(sigma[0, 1] / np.sqrt(sigma[0, 0] * sigma[1, 1]))
                               if sigma[0, 0] * sigma[1, 1] > 0 else 0.,
            'variance_design': 'cluster_robust',
        }
