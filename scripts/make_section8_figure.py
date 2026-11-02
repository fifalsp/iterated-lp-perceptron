#!/usr/bin/env python3
"""Figure 4 of the paper (Section 8): the density-0.60 schedule, built from saved data.

The free fraction q_t along the schedule, for the three population runs of
Table 1 (gray) and the two feasible finite LP paths with N = 1024 (blue). The
script also checks that all three runs first satisfy the stopping rule
kappa_t - D_alpha(q_t) > 10^-3 at round 119.

Fonts match the paper (LaTeX with mathpazo). If no LaTeX installation is
found, Matplotlib's own math rendering is used instead.

Example:
    python3 scripts/make_section8_figure.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iterlp  # noqa: E402,F401

import argparse  # noqa: E402
import shutil  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from iterlp.io import REPO_ROOT, display_path, load_config, read_json  # noqa: E402
from iterlp.rounding import MINIMUM_RESERVE, rounding_loss  # noqa: E402
from iterlp.schedules import quadratic_schedule_exact  # noqa: E402

DATA = REPO_ROOT / 'data' / 'section8'
BLUE, BLACK = '#2a78d6', '#20252B'
PREDICTION_ALPHA, PREDICTION_LW = .6, 1.0
RUN_ALPHA = .27  # three nearly identical runs overlap to about the gray of a single prediction curve


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def set_style():
    if shutil.which('latex'):
        plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 9,
                             'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{mathpazo}',
                             'axes.linewidth': .6, 'pdf.fonttype': 42})
    else:
        print('LaTeX not found; using Matplotlib math text (fonts will differ from the paper).')
        plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'font.size': 9,
                             'axes.linewidth': .6, 'pdf.fonttype': 42})


def style_axis(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, pad=2, labelsize=8)
    ax.set_xlim(0, 126)
    ax.set_xticks([1, 40, 80, 119])
    ax.set_xlabel(r'Round $t$', fontsize=9)


def population_runs(config, folder):
    """Free fractions of the accepted, resolved prefix of each population run."""
    runs = []
    for spec in config['population']['runs']:
        record = read_json(folder / (spec['name'] + '.json'))
        rows = [r for r in record['rounds'] if r['accepted'] and r.get('resolved', False)]
        require([r['round'] for r in rows] == list(range(2, len(rows) + 2)), 'gap in the population prefix')
        q = np.array([record['first_round']['q']] + [r['q'] for r in rows])
        cost = rounding_loss(q)
        require(np.allclose(cost[1:], [r['rounding_loss'] for r in rows], rtol=1e-12, atol=0),
                'rounding cost does not match the saved values')
        require(record['endpoint']['round'] == len(q), 'endpoint is not the last resolved round')
        runs.append(dict(seed=record['seed'], q=q, cost=cost))
    return runs


def finite_runs(folder):
    """Free fractions of the finite LP paths whose first LP is feasible."""
    report = read_json(folder / 'report.json')
    runs = []
    for run in report['runs']:
        if run['status'] == 'lp_failure':
            continue
        f = np.array([r['free_fraction'] for r in run['trajectory']])
        runs.append(dict(seed=run['seed'], free=f, n=report['n']))
    return runs


def make_figure(config, args):
    margins = np.array(quadratic_schedule_exact(**{k: config['schedule'][k]
                                                   for k in ('offset', 'scale', 'horizon', 'rounds')}))
    pop = population_runs(config, args.population_dir)
    fin = finite_runs(args.finite_dir)
    stop = len(pop[0]['q'])
    require(all(len(r['q']) == stop for r in pop), 'runs stop at different rounds')
    for r in pop:
        reserve = margins[:stop] - r['cost']
        require(reserve[-1] > MINIMUM_RESERVE and np.all(reserve[:-1] <= MINIMUM_RESERVE),
                'the stopping round is not the first round passing the test')

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    fig.subplots_adjust(left=.16, right=.97, top=.96, bottom=.18)

    for r in fin:
        t = np.arange(1, len(r['free']) + 1)
        positive = r['free'] > 0
        ax.step(t[positive], r['free'][positive], where='post', color=BLUE, lw=1.1, zorder=3)
    for r in pop:
        ax.plot(np.arange(1, stop + 1), r['q'], color=BLACK, alpha=RUN_ALPHA, lw=PREDICTION_LW, zorder=2)
    ax.set_yscale('log')
    ax.set_ylim(5e-7, 1)
    ax.set_yticks([1, 1e-2, 1e-4, 1e-6])
    ax.set_ylabel(r'Free fraction', fontsize=9)
    style_axis(ax)

    handles = [Line2D([], [], color=BLACK, alpha=PREDICTION_ALPHA, lw=PREDICTION_LW),
               Line2D([], [], color=BLUE, lw=1.1)]
    ax.legend(handles, ['State evolution', r'Finite LPs, $N=1024$'], loc='lower left',
              frameon=False, fontsize=8, handlelength=2.2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        path = args.output_dir / ('free-fraction.' + extension)
        metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {}
        fig.savefig(path, dpi=220, metadata=metadata)
        print('wrote', display_path(path))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--population-dir', type=Path, default=DATA / 'population')
    parser.add_argument('--finite-dir', type=Path, default=DATA / 'finite_lps')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'output' / 'figures')
    args = parser.parse_args()
    set_style()
    make_figure(load_config('section8'), args)


if __name__ == '__main__':
    main()
