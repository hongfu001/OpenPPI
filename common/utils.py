"""Reproducible random streams and experiment output helpers."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np



def seed_sequence(seed, rep, stream):
    return np.random.SeedSequence([seed, rep, stream])


def rng_for(seed, rep, stream):
    return np.random.default_rng(seed_sequence(seed, rep, stream))


def write_config(out, config, **metadata):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(
        dict(config=asdict(config), **metadata), indent=2))


def write_csv(out, name, frame):
    frame.to_csv(Path(out) / name, index=False)


def summarize(results):
    summary = results.groupby(['policy','method']).agg(
        reps=('rep','count'), coverage=('covered','mean'), avg_width=('width','mean'),
        mse=('sq_error','mean'), mean_audits=('audits','mean'),
        mean_sum_pi=('sum_sampling_probabilities','mean'), mean_tau=('tau','mean')).reset_index()
    summary['coverage_mc_se'] = np.sqrt(summary.coverage*(1-summary.coverage)/summary.reps)
    return summary


def split(features, labels, *, train_size, labeled_size, random_state=None):
    """Randomly partition rows for independent training and two-sample PPI.

    Returns (X_train, y_train), (X_labeled, y_labeled), X_unlabeled.
    Sizes are integer row counts. Shuffling ignores labels; use iid source
    rows and train only on the returned training set. Unlabeled outcomes
    are deliberately omitted from the return value.
    """
    x = np.asarray(features)
    y = np.asarray(labels)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError('Require aligned feature matrix and label vector')
    for size in (train_size, labeled_size):
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 2:
            raise ValueError('train_size and labeled_size must be integer counts >= 2')
    if len(x) - train_size - labeled_size < 2:
        raise ValueError('Leave at least two rows for the unlabeled sample')
    indices = np.random.default_rng(random_state).permutation(len(x))
    train = indices[:train_size]
    labeled = indices[train_size:train_size + labeled_size]
    unlabeled = indices[train_size + labeled_size:]
    return (x[train], y[train]), (x[labeled], y[labeled]), x[unlabeled]
