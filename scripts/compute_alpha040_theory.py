#!/usr/bin/env python3
"""Predicted laws for Figure 2: three rounds of the population recursion at alpha = 0.40.

Margins (kappa_1, kappa_2, kappa_3) = (0.60, 0.30, 0.03). For each round t the
script evaluates, on fixed grids,

  * the continuous part of the law of a coordinate x_j^(t) on (-1, 1); it
    integrates to q_t, and each endpoint +-1 carries mass (1 - q_t)/2;
  * the continuous part of the law of a row margin u_i^(t) on (kappa_t, 3.8];
    the atom at kappa_t (tight rows) is stored separately.

Given the history, the new coordinate is a clipped Gaussian and the new
margin a Gaussian projected onto [kappa_t, infinity), so both densities are
Gaussian mixtures over the history particles. No LP data enter.

Two evaluations are made: the primary one used in the figure (2^16 column and
2^12 row particles, seed 1729) and a replicate (2^15 / 2^12, seed 2718); the
differences between them are written to stability.json.

Example:
    python3 scripts/compute_alpha040_theory.py --output-dir results/alpha040_theory
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iterlp  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from iterlp.io import REPO_ROOT, environment, load_config, write_json  # noqa: E402
from iterlp.population import Population  # noqa: E402

trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # NumPy >= 2.0 renamed trapz


def gaussian_density(z):
    return np.exp(-z * z / 2) / math.sqrt(2 * math.pi)


def mixture_density(grid, means, sd, weights):
    """Density of sum_k weights[k] N(means[k], sd^2) on a grid, in blocks of 32 grid points."""
    chunks = []
    for begin in range(0, len(grid), 32):
        z = (grid[begin:begin + 32, None] - means[None, :]) / sd
        chunks.append(gaussian_density(z) @ weights / sd)
    return np.concatenate(chunks)


def compute(config, spec):
    """Evaluate the three rounds for one particle configuration."""
    started = time.monotonic()
    alpha = config['alpha']
    margins = config['margins']
    settings = config['population']
    u_max = settings['margin_grid_max']
    xx = np.linspace(-1, 1, settings['grid_points'])
    pop = Population(alpha, margins[0], spec['column_power'], spec['row_power'], spec['seed'])
    rounds = []
    checks = []
    for t, kappa in enumerate(margins, 1):
        ug = np.linspace(kappa, u_max, settings['grid_points'])
        if t == 1:
            # Round 1 in closed form: x = H/a on |H| < a, U = max(kappa, rho G).
            f = pop.first
            a = f['a']
            xf = a * gaussian_density(a * xx)
            means = np.zeros(1)
            weights = np.ones(1)
            rho = f['rho']
            q = pop.q
            residual = 0.
        else:
            # The densities of round t are mixtures over the particles of round t - 1,
            # so keep q_{t-1} and the row weights before solve() advances the state.
            q_before = pop.q
            weights = pop.row_weights.copy()
            record = pop.solve(kappa, max_nfev=settings['max_nfev'])
            if not record['accepted']:
                raise RuntimeError('%s: round %d rejected' % (spec['name'], t))
            result = pop.last
            score, sd, tau = result['mean_score'], result['sd_score'], result['tau']
            xf = q_before * tau * mixture_density(tau * xx, score, sd, np.full(len(score), 1 / len(score)))
            # The exact law is symmetric under x -> -x; symmetrize to remove odd sampling noise.
            xf = (xf + xf[::-1]) / 2
            means = result['row_mean'].copy()
            rho = result['rho']
            q = result['q']
            assert abs(pop.q - q) < 1e-13
            residual = record['maximum_scaled_residual']
        uf = mixture_density(ug, means, rho, weights)
        atom = float(weights @ ndtr((kappa - means) / rho))
        tail = float(weights @ ndtr((means - u_max) / rho))
        check = dict(round=t, kappa=kappa, q=q,
                     coordinate_endpoint_mass_each=(1 - q) / 2,
                     margin_atom=atom,
                     balance_atom_minus_q_over_alpha=atom - q / alpha,
                     coordinate_density_integral=float(trapezoid(xf, xx)),
                     coordinate_density_integral_error=float(trapezoid(xf, xx) - q),
                     margin_density_integral=float(trapezoid(uf, ug)),
                     margin_tail_beyond_grid=tail,
                     margin_total_mass=float(trapezoid(uf, ug) + tail + atom),
                     maximum_scaled_residual=residual,
                     row_weight_sum=float(weights.sum()))
        assert abs(check['coordinate_density_integral_error']) < 3e-5
        assert abs(check['margin_total_mass'] - 1) < 3e-5
        assert abs(atom - q / alpha) < 3e-6
        checks.append(check)
        rounds.append(dict(t=t, k=kappa, q=q, x=xx, xf=xf, u=ug, uf=uf, atom=atom))
        print(spec['name'], json.dumps(check), flush=True)
    curves = {f'{key}_{r["t"]}': r[key] for r in rounds for key in ['x', 'xf', 'u', 'uf', 'k', 'q', 'atom']}
    report = dict(description='Population recursion at alpha = 0.40 for Figure 2; no LP data enter.',
                  name=spec['name'], alpha=alpha, margins=margins, margin_grid_max=u_max,
                  seed=spec['seed'], column_power=spec['column_power'], row_power=spec['row_power'],
                  max_nfev=settings['max_nfev'], first_round=pop.first, checks=checks, rounds=pop.records,
                  elapsed_seconds=time.monotonic() - started, environment=environment(),
                  curves='%s_curves.npz: for t = 1, 2, 3, x_t/xf_t (coordinate grid and density, integrating to q_t), '
                         'u_t/uf_t (margin grid and density of the continuous part), k_t, q_t, atom_t (mass at kappa_t). '
                         'The coordinate density is symmetrized; each endpoint +-1 has mass (1 - q_t)/2.' % spec['name'])
    return rounds, curves, report


def main():
    config = load_config('alpha040')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'results' / 'alpha040_theory')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluations = []
    for spec in config['population']['runs']:
        rounds, curves, report = compute(config, spec)
        np.savez_compressed(args.output_dir / (spec['name'] + '_curves.npz'), **curves)
        write_json(args.output_dir / (spec['name'] + '_report.json'), report)
        evaluations.append(rounds)

    primary, replicate = evaluations
    stability = []
    for a, b in zip(primary, replicate):
        stability.append(dict(round=a['t'], q_difference=b['q'] - a['q'],
                              margin_atom_difference=b['atom'] - a['atom'],
                              coordinate_density_max_abs_difference=float(np.max(np.abs(a['xf'] - b['xf']))),
                              coordinate_density_l1_difference=float(trapezoid(np.abs(a['xf'] - b['xf']), a['x'])),
                              margin_density_max_abs_difference=float(np.max(np.abs(a['uf'] - b['uf']))),
                              margin_density_l1_difference=float(trapezoid(np.abs(a['uf'] - b['uf']), a['u']))))
    write_json(args.output_dir / 'stability.json', stability)
    print('stability', json.dumps(stability, indent=2), flush=True)


if __name__ == '__main__':
    main()
