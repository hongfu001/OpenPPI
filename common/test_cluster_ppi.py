"""Cluster-Robust PPI++: hand formulas, singleton clusters, labels-only limit, coverage."""
import numpy as np
import pytest

from prediction_inference import ClusterRobustPPI, ClusteredPPIData, PPIPlusPlus, Mean, PPIData
from dataset.judge_sim import make_clustered_data


def record(frame, labeled):
    return ClusteredPPIData(np.where(labeled, frame['label'], np.nan), frame['prediction'],
                            labeled, frame['cluster'])


def test_formula_by_hand():
    clusters = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    labels = np.array([1., np.nan, 0., 1., np.nan, np.nan, 1., 0.])
    f = np.array([.9, .2, .3, .7, .6, .4, .8, .1])
    xi = ~np.isnan(labels)
    data = ClusteredPPIData(labels, f, xi, clusters)
    for lam in (0., 1., .4):
        theta = lam * f[~xi].mean() + (labels[xi] - lam * f[xi]).mean()
        eps = np.where(xi, np.nan_to_num(labels) - theta, 0)
        z = np.where(xi, 1, -5 / 3) * (f - f.mean())
        sums = np.array([[eps[clusters == g].sum(), z[clusters == g].sum()] for g in range(3)])
        s = sums.T @ sums / 3
        result = ClusterRobustPPI(lam=lam)(data, target=Mean())
        assert result.estimate == pytest.approx(theta)
        assert result.standard_error == pytest.approx(np.sqrt(3 * (s[0, 0] - 2 * lam * s[0, 1] + lam ** 2 * s[1, 1])) / 5)
    tuned = ClusterRobustPPI()(data, target=Mean()).diagnostics
    theta1 = f[~xi].mean() + (labels[xi] - f[xi]).mean()
    eps = np.where(xi, np.nan_to_num(labels) - theta1, 0)
    sums = np.array([[eps[clusters == g].sum(), z[clusters == g].sum()] for g in range(3)])
    assert tuned['prediction_weight'] == pytest.approx(sums[:, 0] @ sums[:, 1] / (sums[:, 1] @ sums[:, 1]))
    assert tuned['partially_labeled_clusters'] == 3 and tuned['cluster_count'] == 3
    with pytest.raises(ValueError):
        ClusteredPPIData(labels, f, xi.astype(int), clusters)
    with pytest.raises(ValueError):
        ClusteredPPIData(np.nan_to_num(labels), f, xi, clusters)


def test_singleton_clusters_recover_ppi_plus_plus():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < .4).astype(float)
    f = np.clip(.4 * y + .3 + rng.normal(0, .1, 20000), 0, 1)
    xi = rng.random(20000) < .2
    clustered = ClusterRobustPPI()(ClusteredPPIData(np.where(xi, y, np.nan), f, xi, np.arange(20000)), target=Mean())
    iid = PPIPlusPlus()(PPIData(y[xi], f[xi], f[~xi]), target=Mean())
    assert clustered.estimate == pytest.approx(iid.estimate, abs=1e-4)
    assert clustered.standard_error == pytest.approx(iid.standard_error, rel=.01)
    assert clustered.diagnostics['prediction_weight'] == pytest.approx(iid.diagnostics['prediction_weight'], rel=.02)


def test_labels_only_and_tuning():
    frame = make_clustered_data(clusters=400, seed=1)
    xi = np.random.default_rng(1).random(len(frame)) < .3
    data = record(frame, xi)
    classical = ClusterRobustPPI(lam=0)(data, target=Mean())
    assert classical.estimate == pytest.approx(frame['label'][xi].mean())
    tuned = ClusterRobustPPI()(data, target=Mean())
    assert tuned.standard_error <= classical.standard_error
    assert tuned.standard_error <= ClusterRobustPPI(lam=1)(data, target=Mean()).standard_error


@pytest.mark.parametrize('whole_clusters', [False, True])
def test_cluster_robust_coverage_and_iid_undercoverage(whole_clusters):
    reps, covered, covered_iid = 200, 0, 0
    for rep in range(reps):
        frame = make_clustered_data(clusters=300, seed=1000 + rep)
        rng = np.random.default_rng(rep)
        if whole_clusters:
            xi = (rng.random(300) < .3)[frame['cluster']]
        else:
            xi = rng.random(len(frame)) < .3
        low, high = ClusterRobustPPI()(record(frame, xi), target=Mean()).confidence_interval
        covered += low <= .5 <= high
        f, y = frame['prediction'].to_numpy(), frame['label'].to_numpy()
        low, high = PPIPlusPlus()(PPIData(y[xi], f[xi], f[~xi]), target=Mean()).confidence_interval
        covered_iid += low <= .5 <= high
    assert .90 < covered / reps <= 1.0, f'coverage={covered / reps:.3f}'
    assert covered_iid / reps < .85
