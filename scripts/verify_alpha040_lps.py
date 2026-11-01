#!/usr/bin/env python3
"""Independent check of the saved alpha = 0.40 finite-LP states (Figure 2).

Regenerates each matrix and the shared objectives from the seeds, checks
their hashes, and checks every saved state: A @ x against the saved margins
(to 1e-12), u >= kappa_t - 1e-7, the box, frozen coordinates at +-1 that never
change, nested free sets, and no free coordinate at +-1. Also reports the
empirical free fractions, endpoint masses and margin atoms, pooled by ensemble.

Example:
    python3 scripts/verify_alpha040_lps.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iterlp  # noqa: E402,F401

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from iterlp.finite_lp import MARGIN_TOLERANCE, gaussian_matrix, rademacher_matrix  # noqa: E402
from iterlp.io import REPO_ROOT, display_path, read_json, write_json  # noqa: E402

MATRICES = {'gaussian': gaussian_matrix, 'rademacher': rademacher_matrix}


def check_run(record, report, folder, tolerance):
    seed, ensemble = record['seed'], record['ensemble']
    n, m = report['n'], report['m']
    margins = np.asarray(report['margins'])[:, None]
    rounds = len(report['margins'])
    matrix = MATRICES[ensemble](np.random.default_rng(seed), m, n)
    objectives = np.random.default_rng(np.random.SeedSequence([seed, 1])).standard_normal((rounds, n))
    state = np.load(folder / record['states_file'])
    x, u, free = state['x'], state['u'], state['free']
    reconstruction_error = float(np.max(np.abs(x @ matrix.T - u)))
    checks = dict(
        matrix_hash=hashlib.sha256(matrix.tobytes()).hexdigest() == record['matrix_sha256'],
        objective_hash=hashlib.sha256(objectives.tobytes()).hexdigest() == record['objectives_sha256'],
        shapes=x.shape == (rounds, n) and u.shape == (rounds, m) and free.shape == (rounds, n),
        finite=bool(np.isfinite(x).all() and np.isfinite(u).all()),
        margins_match_matrix=reconstruction_error < tolerance,
        margin_constraints=bool(np.all(u >= margins - MARGIN_TOLERANCE)),
        box_constraints=bool(np.all(np.abs(x) <= 1)),
        frozen_at_endpoints=bool(np.all(np.abs(x[~free]) == 1)),
        free_sets_nested=bool(np.all(~free[1:] | free[:-1])),
        frozen_unchanged=all(np.array_equal(x[t, ~free[t - 1]], x[t - 1, ~free[t - 1]]) for t in range(1, rounds)),
        no_free_endpoint=bool(np.all(np.abs(x[free]) < 1)))
    return dict(seed=seed, ensemble=ensemble, all_checks_passed=all(checks.values()), checks=checks,
                maximum_reconstruction_error=reconstruction_error,
                maximum_margin_violation=float(max(0, np.max(margins - u))),
                free_fractions=free.mean(axis=1).tolist(),
                lower_endpoint_masses=(x == -1).mean(axis=1).tolist(),
                upper_endpoint_masses=(x == 1).mean(axis=1).tolist(),
                margin_atom_masses=(np.abs(u - margins) <= MARGIN_TOLERANCE).mean(axis=1).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folder', type=Path, default=REPO_ROOT / 'data' / 'alpha040' / 'finite_lps')
    parser.add_argument('--tolerance', type=float, default=1e-12)
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'output' / 'alpha040_finite_lps_check.json',
                        help='where to write the check record')
    args = parser.parse_args()
    report = read_json(args.folder / 'report.json')
    results = [check_run(record, report, args.folder, args.tolerance) for record in report['runs']]
    summary = dict(description='Regenerated matrices and objectives; checked the saved alpha = 0.40 LP states.',
                   folder=display_path(args.folder), tolerance=args.tolerance,
                   all_checks_passed=all(r['all_checks_passed'] for r in results),
                   run_count=len(results), runs=results, pooled_by_ensemble={})
    for ensemble in report['ensembles']:
        rows = [r for r in results if r['ensemble'] == ensemble]
        summary['pooled_by_ensemble'][ensemble] = {
            field: np.mean([r[field] for r in rows], axis=0).tolist()
            for field in ('free_fractions', 'lower_endpoint_masses', 'upper_endpoint_masses', 'margin_atom_masses')}
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2))
    if not summary['all_checks_passed']:
        sys.exit('some checks failed')


if __name__ == '__main__':
    main()
