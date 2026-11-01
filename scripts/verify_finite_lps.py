#!/usr/bin/env python3
"""Independent check of the saved Section 8 finite-LP states.

Regenerates every matrix and objective from its seed and checks, round by
round: the matrix and objective hashes, the row products A @ x against the
saved margins u, the box constraint, that frozen coordinates are +-1 and never
change, that the free sets are nested, the recorded free counts, the minimum
LP margin and the minimum margin of the sign vector. No LP is solved.

Row products are compared with a tolerance (default 1e-12), because the last
bits of A @ x depend on the BLAS build. Use --tolerance 0 for exact equality.

Example:
    python3 scripts/verify_finite_lps.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iterlp  # noqa: E402,F401

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from iterlp.finite_lp import MARGIN_TOLERANCE, gaussian_matrix, sign_vector  # noqa: E402
from iterlp.io import REPO_ROOT, display_path, read_json, write_json  # noqa: E402


def check_run(run, folder, tolerance):
    seed, n, m = run['seed'], run['n'], run['m']
    rng = np.random.default_rng(seed)
    A = gaussian_matrix(rng, m, n)
    assert hashlib.sha256(A.tobytes()).hexdigest() == run['matrix_sha256'], 'matrix hash'
    objectives = hashlib.sha256()
    for _ in run['trajectory']:
        objectives.update(rng.standard_normal(n).tobytes())
    assert objectives.hexdigest() == run['objectives_sha256'], 'objective hash'

    states = np.load(folder / run['states_file'])
    solved = [r for r in run['trajectory'] if 'free_count' in r]
    assert len(states['x']) == len(solved) == len(states['u']) == len(states['free'])
    previous_free = np.ones(n, dtype=bool)
    previous_x = np.zeros(n)
    largest_deficit = 0.
    largest_product_error = 0.
    for x, u, free, r in zip(states['x'], states['u'], states['free'], solved):
        product = A @ x
        product_error = float(np.max(np.abs(product - u)))
        largest_product_error = max(largest_product_error, product_error)
        assert product_error <= tolerance, 'row products, round %d' % r['round']
        assert np.max(np.abs(x)) <= 1 + 1e-12, 'box'
        assert np.array_equal(x[~previous_free], previous_x[~previous_free]), 'frozen coordinates changed'
        assert not np.any(free & ~previous_free), 'free sets not nested'
        assert np.all(np.abs(x[~free]) == 1), 'frozen coordinates not at +-1'
        assert int(free.sum()) == r['free_count'] and float(free.mean()) == r['free_fraction']
        assert int(previous_free.sum() - free.sum()) == r['newly_frozen_count']
        assert float(u.min()) == r['minimum_lp_margin']
        assert abs(float((A @ sign_vector(x)).min()) - r['sign_minimum_margin']) <= tolerance
        largest_deficit = max(largest_deficit, r['kappa'] - r['minimum_lp_margin'])
        assert largest_deficit <= MARGIN_TOLERANCE, 'margin below kappa_t - 1e-7'
        previous_free = free
        previous_x = x
    result = dict(seed=seed, status=run['status'], saved_rounds=len(solved),
                  largest_margin_deficit=largest_deficit, largest_row_product_error=largest_product_error)
    if solved:
        result.update(final_round=solved[-1]['round'], final_free_count=solved[-1]['free_count'],
                      final_sign_minimum_margin=solved[-1]['sign_minimum_margin'],
                      first_positive_sign_round=run['first_positive_sign_round'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folder', type=Path, default=REPO_ROOT / 'data' / 'section8' / 'finite_lps')
    parser.add_argument('--tolerance', type=float, default=1e-12)
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'output' / 'section8_finite_lps_check.json',
                        help='where to write the check record')
    args = parser.parse_args()

    report = read_json(args.folder / 'report.json')
    cases = [check_run(run, args.folder, args.tolerance) for run in report['runs']]
    result = dict(description='Regenerated matrices and objectives; checked all saved states of the Section 8 finite LPs.',
                  folder=display_path(args.folder), tolerance=args.tolerance, all_checks_passed=True, cases=cases)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
