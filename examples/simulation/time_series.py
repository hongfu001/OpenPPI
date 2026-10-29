"""PPITSAS vs iid PPI on ordered spatial time series (mean only).

Run from examples/simulation/: python time_series.py --reps 200 --out results_time_series
Chronology: old unlabeled history -> recent rectification -> latest fitting.
This is a lightweight stationary simulation, not a reproduction of the paper's weather GP.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from openppi import Mean, PPIData, get_method
from dataset.time_series_sim import TimeSeriesDGP


def prepare_records(data, *, unlabeled_size, rectification_size, train_size):
    """Fit one shared model on the latest block; return one PPIData per site."""
    for size in (unlabeled_size, rectification_size, train_size):
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 2:
            raise ValueError('Block sizes must be integer row counts >= 2')
    features, labels = data['features'], data['y']
    n = unlabeled_size + rectification_size + train_size
    if features.ndim != 3 or labels.shape != features.shape[:2] or len(features) != n:
        raise ValueError('Expected features[time, site, feature] and labels[time, site] matching block sizes')
    end = unlabeled_size + rectification_size
    # Pool locations only for training; keep each inference series ordered and separate.
    model = LinearRegression().fit(features[end:].reshape(-1, features.shape[-1]),
                                   labels[end:].reshape(-1))
    predictions = model.predict(features[:end].reshape(-1, features.shape[-1])).reshape(end, -1)
    return [PPIData(labels[unlabeled_size:end, site],
                    predictions[unlabeled_size:end, site], predictions[:unlabeled_size, site])
            for site in range(labels.shape[1])]


def run_simulation(*, reps=200, unlabeled_size=1600, rectification_size=800,
                   train_size=200, sites=9, seed=20261029, phi=.7, error_phi=.5):
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 2:
        raise ValueError('reps must be an integer >= 2 for Monte Carlo standard errors')
    dgp = TimeSeriesDGP(sites=sites, phi=phi, error_phi=error_phi)
    target = Mean(outcome_type='continuous')
    methods = {name: get_method(name).inference for name in ('PPI', 'PPITSAS')}
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
        data = dgp.draw(unlabeled_size + rectification_size + train_size, rng)
        records = prepare_records(data, unlabeled_size=unlabeled_size,
                                  rectification_size=rectification_size, train_size=train_size)
        for site, record in enumerate(records):
            truth = float(data['truth'][site])
            for name, method in methods.items():
                result = method(record, target=target)
                low, high = result.confidence_interval
                rows.append(dict(rep=rep, site=site, method=name, truth=truth,
                                 estimate=result.estimate, standard_error=result.standard_error,
                                 low=low, high=high, covered=low <= truth <= high,
                                 width=high-low, sq_error=(result.estimate-truth)**2))
    results = pd.DataFrame(rows)
    # Sites are correlated: Monte Carlo uncertainty uses independent replicates,
    # not the individual site outcomes as if they were independent draws.
    per_rep = results.groupby(['method', 'rep'])[['covered', 'width', 'sq_error']].mean()
    summary = per_rep.groupby('method').agg(
        reps=('covered', 'size'), coverage=('covered', 'mean'),
        coverage_mc_sd=('covered', 'std'), avg_width=('width', 'mean'), mse=('sq_error', 'mean'))
    summary['coverage_mc_se'] = summary.pop('coverage_mc_sd') / np.sqrt(summary['reps'])
    summary['rmse'] = np.sqrt(summary.pop('mse'))
    return results, summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in dict(reps=200, unlabeled_size=1600, rectification_size=800,
                              train_size=200, sites=9, seed=20261029).items():
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    parser.add_argument('--phi', type=float, default=.7)
    parser.add_argument('--error-phi', type=float, default=.5)
    parser.add_argument('--out', type=Path)
    args = vars(parser.parse_args())
    out = args.pop('out')
    results, summary = run_simulation(**args)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'config.json').write_text(json.dumps(args, indent=2) + '\n')
        results.to_csv(out / 'per_rep_site.csv', index=False)
        summary.to_csv(out / 'summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
