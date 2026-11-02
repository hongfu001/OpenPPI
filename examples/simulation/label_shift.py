"""Target-mean inference under label shift, with a no-shift control.

Run from examples/simulation/:
  python label_shift.py --reps 300 --out results_label_shift
  python label_shift.py --case continuous_shift --reps 50 --out results_label_shift_continuous
The binary case uses the fixed source Bayes predictor to isolate shift correction.
The continuous case follows Section 6's DGP and uses numerically regularized NW
conditional expectations; it is not a reproduction of the paper's full study.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from openppi import EfficientLabelShift, Mean, TwoSample
from dataset.label_shift_sim import make_label_shift_case, make_continuous_label_shift_case


def run_simulation(*, reps=300, n=300, m=1500, seed=20261102,
                   case='binary', alpha=.05, source_prevalence=.3,
                   target_prevalence=.7, separation=1., rcond=1e-3):
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 2:
        raise ValueError('reps must be an integer >= 2')
    for count in (n, m):
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError('n and m must be integer counts >= 2')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0, 1)')
    if case not in ('binary', 'no_shift', 'label_shift', 'continuous_shift', 'all'):
        raise ValueError('Unknown case')
    cases = (['no_shift', 'label_shift'] if case == 'binary' else
             ['no_shift', 'label_shift', 'continuous_shift'] if case == 'all' else [case])
    rows = []
    for scenario in cases:
        continuous = scenario == 'continuous_shift'
        # Pair the binary scenarios with common random seeds for comparison.
        for rep in range(reps):
            draw_seed = int(np.random.SeedSequence([seed, rep, int(continuous)]).generate_state(1)[0])
            if continuous:
                data, standard, truth = make_continuous_label_shift_case(n=n, m=m, seed=draw_seed)
                options = dict(outcome='continuous', rcond=rcond, ratio_bounds=(.01, 100.))
            else:
                q = source_prevalence if scenario == 'no_shift' else target_prevalence
                data, standard, truth = make_label_shift_case(n=n, m=m, seed=draw_seed,
                    source_prevalence=source_prevalence, target_prevalence=q, separation=separation)
                options = {}
            methods = {'Source-only': (TwoSample(lam=0), standard),
                       'PPI': (TwoSample(), standard), 'PPI++': (TwoSample(lam='tuned'), standard),
                       'LabelShift': (EfficientLabelShift(**options), data),
                       'LabelShift-refined': (EfficientLabelShift(refine_ratio=True, **options), data)}
            target = Mean(outcome_type='continuous' if continuous else 'binary')
            for name, (estimator, record) in methods.items():
                row = dict(case=scenario, rep=rep, method=name, truth=truth, source_count=n, target_count=m)
                try:
                    result = estimator(record, target=target, alpha=alpha)
                    lo, hi = result.confidence_interval
                    fractions = result.diagnostics.get('ratio_clipped_fractions', [0.])
                    row.update(status='ok', message='', estimate=result.estimate,
                               standard_error=result.standard_error, low=lo, high=hi,
                               error=result.estimate-truth, sq_error=(result.estimate-truth)**2,
                               covered=lo <= truth <= hi, width=hi-lo,
                               max_ratio_clipped_fraction=max(fractions),
                               fredholm_residual=result.diagnostics.get('fredholm_residual', np.nan))
                except ValueError as error:
                    # Keep failed repetitions in the exported data and failure-rate denominator.
                    row.update(status='failed', message=str(error), estimate=np.nan,
                               standard_error=np.nan, low=np.nan, high=np.nan,
                               error=np.nan, sq_error=np.nan, covered=np.nan, width=np.nan,
                               max_ratio_clipped_fraction=np.nan, fredholm_residual=np.nan)
                rows.append(row)
    results = pd.DataFrame(rows)
    results['covered'] = pd.to_numeric(results['covered']).astype(float)
    results['covered_all'] = results['covered'].eq(1)
    results['successful'] = results['status'].eq('ok')
    summary = results.groupby(['case', 'method'], sort=False).agg(
        reps=('rep', 'size'), successful_reps=('successful', 'sum'),
        bias=('error', 'mean'), mse=('sq_error', 'mean'), empirical_sd=('estimate', 'std'),
        average_se=('standard_error', 'mean'), coverage=('covered', 'mean'),
        coverage_all=('covered_all', 'mean'), average_length=('width', 'mean'),
        average_clipped_fraction=('max_ratio_clipped_fraction', 'mean'))
    summary['failure_rate'] = 1 - summary['successful_reps'] / summary['reps']
    summary['rmse'] = np.sqrt(summary.pop('mse'))
    summary['coverage_mc_se'] = np.sqrt(summary['coverage'] * (1 - summary['coverage']) / summary['successful_reps'].where(summary['successful_reps'] > 0))
    return results, summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in dict(reps=300, n=300, m=1500, seed=20261102).items():
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in dict(alpha=.05, source_prevalence=.3, target_prevalence=.7,
                              separation=1., rcond=1e-3).items():
        parser.add_argument('--' + name.replace('_', '-'), type=float, default=default)
    parser.add_argument('--case', choices=['binary', 'no_shift', 'label_shift', 'continuous_shift', 'all'], default='binary')
    parser.add_argument('--out', type=Path)
    options = vars(parser.parse_args())
    out = options.pop('out')
    results, summary = run_simulation(**options)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'config.json').write_text(json.dumps(options, indent=2) + '\n')
        results.to_csv(out / 'per_rep.csv', index=False)
        summary.to_csv(out / 'summary.csv', index=False)
    print(summary.to_string(index=False, float_format=lambda value: f'{value:.4f}'))


if __name__ == '__main__':
    main()
