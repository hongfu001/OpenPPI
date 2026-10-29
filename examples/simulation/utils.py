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
