#!/usr/bin/env python3
"""Section 8 finite LPs: Iterated-LP on Gaussian matrices, N = 1024, M = 614.

For each seed, the matrix is drawn first from numpy.random.default_rng(seed)
and the objective v_t of each round is drawn from the same generator just
before the round is solved. The schedule is the 140-round Section 8 schedule.
A run ends when an LP fails (e.g. is infeasible), when a row margin falls
below kappa_t - 1e-7, when the output becomes integral, after 140 rounds, or
after 600 seconds. All outcomes are kept.

The script first recomputes the population state after rounds 1-3 for the
comparison of frozen fractions in Section 8.3 (seed 2047, 2^14 column and
2^10 row particles per stratum) and compares the free fractions with the
saved seed-2047 run.

Example:
    python3 scripts/run_finite_lps.py --output-dir results/finite_lps
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iterlp  # noqa: E402,F401

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from iterlp.finite_lp import (MARGIN_TOLERANCE, freeze, gaussian_matrix, sign_vector,  # noqa: E402
                              solve_round)
from iterlp.io import REPO_ROOT, environment, load_config, read_json, write_json  # noqa: E402
from iterlp.population import Population  # noqa: E402
from iterlp.schedules import schedule_from_config  # noqa: E402


def population_reference(config, saved_run, tolerance):
    """Population particles after rounds 1, 2, 3 for the distributions figure."""
    reference = config['finite_lps']['population_reference']
    spec = next(s for s in config['population']['runs'] if s['name'] == reference['run'])
    margins = schedule_from_config(config)
    pop = Population(config['alpha'], margins[0], spec['column_power'], spec['row_power'], spec['seed'])
    saved_q = {r['round']: r['q'] for r in saved_run['rounds']}
    saved_q[1] = saved_run['first_round']['q']
    snapshots = {}
    differences = {}
    for t in reference['rounds']:
        if t > 1:
            record = pop.solve(margins[t - 1], max_nfev=reference['max_nfev'])
            if not record['accepted']:
                raise RuntimeError('population round %d was rejected' % t)
        differences[t] = abs(pop.q - saved_q[t]) / saved_q[t]
        if differences[t] > tolerance:
            raise RuntimeError('population round %d: q = %r differs from the saved %r' % (t, pop.q, saved_q[t]))
        snapshots.update({f'x_{t}': pop.X[:, -1], f'u_{t}': pop.U[:, -1],
                          f'w_{t}': pop.row_weights.copy(), f'q_{t}': np.array(pop.q)})
    return snapshots, differences


def run_case(seed, n, alpha, margins, lp_time_limit, case_time_cap):
    """Iterated-LP on one Gaussian matrix. Returns the run record and the saved states."""
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    m = int(np.floor(alpha * n))
    A = gaussian_matrix(rng, m, n)
    x = np.zeros(n)
    free = np.ones(n, dtype=bool)
    xs, us, frees, trajectory = [], [], [], []
    objective_hash = hashlib.sha256()
    run = dict(seed=seed, n=n, m=m, matrix_sha256=hashlib.sha256(A.tobytes()).hexdigest(),
               trajectory=trajectory, status='running')
    for t, kappa in enumerate(margins, 1):
        objective = rng.standard_normal(n)
        objective_hash.update(objective.tobytes())
        free_index, fit = solve_round(A, x, free, objective, kappa, lp_time_limit)
        if not fit.success:
            run['status'] = 'lp_failure'
            trajectory.append(dict(round=t, kappa=kappa, lp_status=int(fit.status), message=fit.message))
            break
        newly_frozen = freeze(x, free, free_index, fit.x)
        u = A @ x
        trajectory.append(dict(round=t, kappa=kappa, free_count=int(free.sum()), free_fraction=float(free.mean()),
                               newly_frozen_count=len(newly_frozen), minimum_lp_margin=float(u.min()),
                               sign_minimum_margin=float((A @ sign_vector(x)).min()),
                               row_active_fraction=float(np.mean(np.abs(u - kappa) <= MARGIN_TOLERANCE)),
                               lp_iterations=int(fit.nit)))
        xs.append(x.copy())
        us.append(u)
        frees.append(free.copy())
        if t <= 3 or t % 10 == 0 or not free.any():
            print(json.dumps(dict(seed=seed, **trajectory[-1])), flush=True)
        if u.min() < kappa - MARGIN_TOLERANCE:
            run['status'] = 'margin_violation'
            break
        if not free.any():
            run['status'] = 'integral'
            break
        if time.monotonic() - started > case_time_cap:
            run['status'] = 'time_cap'
            break
    else:
        run['status'] = 'completed_all_rounds'
    run['elapsed_seconds'] = time.monotonic() - started
    run['objectives_sha256'] = objective_hash.hexdigest()
    run['first_positive_sign_round'] = next((r['round'] for r in trajectory
                                             if r.get('sign_minimum_margin', -1) > MARGIN_TOLERANCE), None)
    run['states_file'] = 'seed%d.npz' % seed
    states = dict(x=np.array(xs), u=np.array(us), free=np.array(frees))
    return run, states


def main():
    config = load_config('section8')
    settings = config['finite_lps']
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'results' / 'finite_lps')
    parser.add_argument('--seeds', type=int, nargs='+', default=settings['seeds'])
    parser.add_argument('--n', type=int, default=settings['n'])
    parser.add_argument('--population-run', type=Path,
                        default=REPO_ROOT / 'data' / 'section8' / 'population' / 'seed2047.json',
                        help='saved run whose free fractions q_1, q_2, q_3 are compared with the recomputed ones')
    parser.add_argument('--tolerance', type=float, default=1e-10,
                        help='largest accepted relative difference of q_1, q_2, q_3 (0 demands equality)')
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    margins = schedule_from_config(config)

    snapshots, differences = population_reference(config, read_json(args.population_run), args.tolerance)
    np.savez_compressed(out / 'population_first_three.npz', **snapshots)
    print('population rounds 1-3: relative differences of q from the saved run:', differences, flush=True)

    report = dict(description='Section 8 finite LPs (Gaussian matrices, quadratic schedule).',
                  alpha=config['alpha'], n=args.n, m=int(np.floor(config['alpha'] * args.n)), seeds=args.seeds,
                  margins=margins, lp_time_limit_seconds=settings['lp_time_limit_seconds'],
                  case_time_cap_seconds=settings['case_time_cap_seconds'],
                  matrix_rng='numpy.random.default_rng(seed).standard_normal((m, n)) / sqrt(n)',
                  objective_rng='same generator, standard_normal(n) drawn before each round',
                  population_reference=dict(settings['population_reference'], file='population_first_three.npz',
                                            q_relative_difference_from_saved={str(k): v for k, v in differences.items()}),
                  environment=environment(), runs=[])
    for seed in args.seeds:
        run, states = run_case(seed, args.n, config['alpha'], margins,
                               settings['lp_time_limit_seconds'], settings['case_time_cap_seconds'])
        np.savez_compressed(out / run['states_file'], **states)
        report['runs'].append(run)
        write_json(out / 'report.json', report)
        print(json.dumps(dict(seed=seed, status=run['status'], rounds=len(states['x']),
                              seconds=round(run['elapsed_seconds'], 1))), flush=True)
    print(out / 'report.json')


if __name__ == '__main__':
    main()
