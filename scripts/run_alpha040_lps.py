#!/usr/bin/env python3
"""Finite LPs for Figure 2: alpha = 0.40, N = 2048, M = 819, three rounds.

Seeds 9401-9404, each with a Gaussian and a Rademacher matrix. The matrix is
drawn from numpy.random.default_rng(seed); the three objectives are drawn from
the independent generator default_rng(SeedSequence([seed, 1])), so the
Gaussian and the Rademacher run of a seed use the same objectives. All runs
are kept, whatever their outcome.

The first LP takes about half a minute per run; runs are distributed over
--workers processes (default 3), each with one BLAS thread.

Example:
    python3 scripts/run_alpha040_lps.py --output-dir results/alpha040_lps
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iterlp  # noqa: E402,F401

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

from iterlp.finite_lp import (MARGIN_TOLERANCE, freeze, gaussian_matrix, rademacher_matrix,  # noqa: E402
                              solve_round)
from iterlp.io import REPO_ROOT, environment, load_config, write_json  # noqa: E402

MATRICES = {'gaussian': gaussian_matrix, 'rademacher': rademacher_matrix}


def objectives_for(seed, rounds, n):
    """Objectives v_1, ..., v_rounds, shared by the two ensembles of a seed."""
    return np.random.default_rng(np.random.SeedSequence([seed, 1])).standard_normal((rounds, n))


def run_case(ensemble, seed, n, alpha, margins, lp_time_limit, output_dir):
    started = time.monotonic()
    m = int(np.floor(alpha * n))
    matrix = MATRICES[ensemble](np.random.default_rng(seed), m, n)
    objectives = objectives_for(seed, len(margins), n)
    x = np.zeros(n)
    free = np.ones(n, dtype=bool)
    xs, us, frees, trajectory = [], [], [], []
    record = dict(ensemble=ensemble, seed=seed, n=n, m=m,
                  matrix_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
                  objectives_sha256=hashlib.sha256(objectives.tobytes()).hexdigest(),
                  trajectory=trajectory, status='running')
    for t, kappa in enumerate(margins, 1):
        round_started = time.monotonic()
        free_index, fit = solve_round(matrix, x, free, objectives[t - 1], kappa, lp_time_limit)
        entry = dict(round=t, kappa=kappa, lp_success=bool(fit.success), lp_status=int(fit.status),
                     message=fit.message, lp_iterations=int(fit.nit),
                     elapsed_seconds=time.monotonic() - round_started)
        trajectory.append(entry)
        if not fit.success:
            record['status'] = 'lp_failure'
            break
        old_x = x.copy()
        old_free = free.copy()
        newly_frozen = freeze(x, free, free_index, fit.x)
        u = matrix @ x
        entry.update(free_count=int(free.sum()), free_fraction=float(free.mean()),
                     newly_frozen_count=len(newly_frozen), minimum_lp_margin=float(u.min()),
                     maximum_margin_violation=float(max(0, kappa - u.min())),
                     row_active_fraction=float(np.mean(np.abs(u - kappa) <= MARGIN_TOLERANCE)),
                     negative_endpoint_fraction=float(np.mean(x == -1)),
                     positive_endpoint_fraction=float(np.mean(x == 1)),
                     box_maximum=float(np.max(np.abs(x))),
                     frozen_unchanged=bool(np.array_equal(x[~old_free], old_x[~old_free])),
                     free_set_nested=bool(np.all(~free | old_free)),
                     frozen_exact_endpoints=bool(np.all(np.abs(x[~free]) == 1)))
        xs.append(x.copy())
        us.append(u.copy())
        frees.append(free.copy())
        if (u.min() < kappa - MARGIN_TOLERANCE or entry['box_maximum'] > 1 + 1e-12
                or not entry['frozen_unchanged'] or not entry['free_set_nested']
                or not entry['frozen_exact_endpoints']):
            record['status'] = 'check_failed'
            break
    else:
        record['status'] = 'completed_all_rounds'
    record['elapsed_seconds'] = time.monotonic() - started
    record['saved_rounds'] = len(xs)
    record['states_file'] = f'{ensemble}_seed{seed}.npz'
    np.savez_compressed(output_dir / record['states_file'],
                        x=np.asarray(xs).reshape((-1, n)),
                        u=np.asarray(us).reshape((-1, m)),
                        free=np.asarray(frees, dtype=bool).reshape((-1, n)))
    return record


def main():
    config = load_config('alpha040')
    settings = config['finite_lps']
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'results' / 'alpha040_lps')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--seeds', type=int, nargs='+', default=settings['seeds'])
    parser.add_argument('--n', type=int, default=settings['n'])
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    alpha = config['alpha']
    margins = config['margins']
    m = int(np.floor(alpha * args.n))
    report = dict(description='Finite LPs for Figure 2 (alpha = 0.40, three rounds).',
                  alpha=alpha, n=args.n, m=m, realized_alpha=m / args.n, margins=margins, seeds=args.seeds,
                  ensembles=settings['ensembles'],
                  matrix_rng='numpy.random.default_rng(seed); standard_normal or 2*integers(0, 2) - 1, divided by sqrt(n)',
                  objective_rng='numpy.random.default_rng(numpy.random.SeedSequence([seed, 1])).standard_normal((3, n))',
                  lp_time_limit_seconds=settings['lp_time_limit_seconds'],
                  environment=environment(), runs=[])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, ensemble, seed, args.n, alpha, margins,
                               settings['lp_time_limit_seconds'], out): (ensemble, seed)
                   for seed in args.seeds for ensemble in settings['ensembles']}
        for future in as_completed(futures):
            ensemble, seed = futures[future]
            try:
                record = future.result()
            except Exception as error:
                record = dict(ensemble=ensemble, seed=seed, status='error', error=repr(error))
            report['runs'].append(record)
            report['runs'].sort(key=lambda r: (r['seed'], r['ensemble']))
            write_json(out / 'report.json', report)
            print(json.dumps(dict(ensemble=ensemble, seed=seed, status=record['status'],
                                  seconds=round(record.get('elapsed_seconds', 0), 1))), flush=True)
    report['all_completed'] = all(r['status'] == 'completed_all_rounds' for r in report['runs'])
    write_json(out / 'report.json', report)


if __name__ == '__main__':
    main()
