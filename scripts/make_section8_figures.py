#!/usr/bin/env python3
"""The three figures of Section 8, built from saved data (no LP or population solve).

    numerics-trajectories   Figure "Population trajectories for the density-0.60 schedule"
    numerics-distributions  Figure "Distributions after the first three LP rounds"
    numerics-endpoints      Figure "Endpoints at density 0.60"

Each is written as PDF and PNG to --output-dir (default output/figures).

Example:
    python3 scripts/make_section8_figures.py
    python3 scripts/make_section8_figures.py --figure trajectories
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iterlp  # noqa: E402,F401

import argparse  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from iterlp.io import REPO_ROOT, display_path, load_config, read_json  # noqa: E402
from iterlp.rounding import MINIMUM_RESERVE, rounding_loss  # noqa: E402
from iterlp.schedules import quadratic_schedule_exact  # noqa: E402

DATA = REPO_ROOT / 'data' / 'section8'
BLUE, SKY, GREEN, PURPLE = '#0072B2', '#56B4E9', '#009E73', '#CC79A7'
ORANGE, GOLD, GREY = '#D55E00', '#E69F00', '#666666'
RUN_STYLES = {'seed2047': (BLUE, '-', 'o'), 'seed2718': (GREEN, '--', 's'), 'seed3141': (PURPLE, ':', 'D')}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def schedule(config):
    spec = config['schedule']
    return np.array(quadratic_schedule_exact(spec['offset'], spec['scale'], spec['horizon'], spec['rounds']))


def set_style():
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9,
                         'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def light_grid(ax):
    ax.grid(axis='y', alpha=.16, lw=.6)
    ax.set_axisbelow(True)


def save(fig, name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        path = output_dir / (name + '.' + extension)
        metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {}
        fig.savefig(path, dpi=220, metadata=metadata)
        print('wrote', display_path(path))
    plt.close(fig)


def load_finite(config, folder):
    """Finite-LP report and saved states (keyed by seed)."""
    report = read_json(folder / 'report.json')
    require(report['alpha'] == config['alpha'], 'wrong finite density')
    require(np.allclose(report['margins'], schedule(config), rtol=0, atol=1e-15), 'wrong finite schedule')
    states = {run['seed']: np.load(folder / run['states_file']) for run in report['runs']}
    return report, states


def load_population(config, folder):
    """Accepted and resolved prefix of each population run."""
    margins = schedule(config)
    rounds_max = len(margins)
    cases = []
    for spec in config['population']['runs']:
        record = read_json(folder / (spec['name'] + '.json'))
        require(record['status'] != 'running', 'population run still active: ' + spec['name'])
        require(np.allclose(record['schedule'][:rounds_max], margins, rtol=0, atol=1e-15), 'schedule mismatch')
        selected, omitted = [], []
        for row in record['rounds']:
            if row['round'] > rounds_max or not row['accepted'] or not row.get('resolved', False):
                omitted.append(row['round'])
                continue
            require(row['round'] == len(selected) + 2, 'gap in the resolved population prefix')
            selected.append(row)
        q = np.array([record['first_round']['q']] + [row['q'] for row in selected])
        require(len(q) <= rounds_max and np.all(q > 0), 'invalid population free fractions')
        previous = np.concatenate(([1.], q[:-1]))
        freezing = 1 - q / previous
        require(np.all((freezing >= 0) & (freezing <= 1)), 'invalid freezing proportions')
        passed = record['status'] == 'passed_finishing_test'
        if passed:
            require(record['endpoint']['passes_finishing_test'], 'endpoint does not pass the finishing test')
            require(selected[-1]['round'] == record['endpoint']['round'], 'endpoint not in the resolved prefix')
            require(selected[-1]['kappa'] - rounding_loss(q[-1]) > MINIMUM_RESERVE, 'finishing test fails')
            require(not any(row['passes_finishing_test'] for row in selected[:-1]),
                    'an earlier resolved round already passed the finishing test')
        color, line, marker = RUN_STYLES[spec['name']]
        cases.append(dict(seed=record['seed'], q=q, freezing=freezing, rows=selected, complete=passed,
                          color=color, line=line, marker=marker,
                          label=f"{record['seed']}: {2 ** record['column_power']:,} / {2 ** record['row_power']:,}"))
    return cases


def distributions(config, args):
    report, states = load_finite(config, args.finite_dir)
    pop = np.load(args.finite_dir / 'population_first_three.npz')
    valid = [r for r in report['runs'] if states[r['seed']]['x'].shape[0] >= 3]
    require(len(valid) > 0, 'no finite trajectory has three rounds')
    fig, axs = plt.subplots(2, 3, figsize=(7.05, 4.5))
    fig.subplots_adjust(left=.085, right=.985, bottom=.15, top=.84, hspace=.48, wspace=.3)
    bins = np.linspace(-1, 1, 21)
    margins = schedule(config)
    for t in range(1, 4):
        # Top row: free coordinates, conditional on being free.
        finite_values = np.concatenate([states[r['seed']]['x'][t - 1][states[r['seed']]['free'][t - 1]] for r in valid])
        ax = axs[0, t - 1]
        ax.hist(finite_values, bins=bins, density=True, color=ORANGE, alpha=.24,
                label=f'Finite LPs ({len(valid)} feasible paths)')
        density, _ = np.histogram(pop[f'x_{t}'], bins=bins, density=True)
        ax.stairs(density, edges=bins, color=BLUE, lw=1.55, label='Population approximation')
        free = np.mean([states[r['seed']]['free'][t - 1].mean() for r in valid])
        ax.text(.04, .95, f'Frozen: {1 - free:.1%}\nPopulation: {1 - float(pop[f"q_{t}"]):.1%}',
                transform=ax.transAxes, va='top', fontsize=7.5)
        ax.set(title=f'Round {t}', xlabel='Free coordinate', xlim=(-1, 1), ylim=(0, 1.02))
        ax.set_yticks([0, .25, .5, .75, 1])
        if t == 1:
            ax.set_ylabel('Conditional density')

        # Bottom row: distribution function of the excess margin u - kappa_t, with its atom at 0.
        ax = axs[1, t - 1]
        grid = np.r_[0, np.linspace(.003, 2.6, 420)]
        excess = np.concatenate([states[r['seed']]['u'][t - 1] for r in valid]) - margins[t - 1]
        finite_cdf = np.array([(excess <= value + 1e-7).mean() for value in grid])
        population_excess = pop[f'u_{t}'] - margins[t - 1]
        weights = pop[f'w_{t}']
        weights = weights / weights.sum()
        require(np.all(weights >= 0), 'negative population row weights')
        population_cdf = np.array([weights[population_excess <= value + 1e-12].sum() for value in grid])
        ax.plot(grid, finite_cdf, color=ORANGE, lw=1.3, label=f'Finite LPs ({len(valid)} feasible paths)')
        ax.plot(grid, population_cdf, color=BLUE, lw=1.55, ls='--', label='Population approximation')
        ax.plot([0, 0], [0, finite_cdf[0]], color=ORANGE, lw=1.6, alpha=.6)
        ax.set(xlabel=r'Excess row margin $u-\kappa_t$', xlim=(-.05, 2.6), ylim=(0, 1.02))
        if t == 1:
            ax.set_ylabel('Cumulative probability')
    for ax in axs.flat:
        light_grid(ax)
    fig.suptitle(r'$\alpha=0.60$: distributions after the first three LP rounds', y=.97, fontsize=11)
    fig.text(.5, .905, f'Finite matrices: {len(valid)} of {len(report["runs"])} paths feasible at the first LP',
             ha='center', fontsize=8, color=GREY)
    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .02), ncol=2, frameon=False)
    save(fig, 'numerics-distributions', args.output_dir)


def trajectories(config, args):
    cases = load_population(config, args.population_dir)
    rounds_max = config['schedule']['rounds']
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.65))
    fig.subplots_adjust(left=.075, right=.985, bottom=.28, top=.80, wspace=.47)
    t = np.arange(1, rounds_max + 1)
    margins = schedule(config)
    axs[0].plot(t, margins, color='#444444', lw=1.65)
    axs[0].scatter([1, rounds_max], margins[[0, -1]], s=14, color='#444444')
    axs[0].set(title='(a) Prescribed margin', ylabel=r'$\kappa_t$', ylim=(0, .62))
    axs[0].text(.97, .91, 'Shared schedule', transform=axs[0].transAxes, ha='right', fontsize=7.5, color=GREY)
    axs[0].text(.97, .72, 'Quadratic decrease\nMaximum 140 rounds', transform=axs[0].transAxes,
                ha='right', fontsize=7.5, color=GREY)
    for c in cases:
        tt = np.arange(1, len(c['q']) + 1)
        for ax, values in [(axs[1], c['q']), (axs[2], c['freezing'])]:
            ax.plot(tt, values, color=c['color'], ls=c['line'], lw=1.4, label=c['label'])
            if len(tt):
                ax.scatter(tt[-1], values[-1], s=23, marker=c['marker'] if c['complete'] else 'x',
                           color=c['color'], zorder=4)
    axs[1].set(title='(b) Remaining free fraction', ylabel=r'$\widehat q_t$', yscale='log', ylim=(1e-7, 1))
    axs[1].set_yticks([1, 1e-2, 1e-4, 1e-6])
    axs[2].set(title='(c) Freezing at each LP', ylabel='Fraction of previous free set', ylim=(0, .8))
    axs[2].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axs:
        ax.set(xlabel='LP round', xlim=(0, 144))
        ax.set_xticks([1, 40, 80, 120])
        light_grid(ax)
    completed = sum(c['complete'] for c in cases)
    fig.suptitle(r'$\alpha=0.60$: numerical population trajectories', y=.96, fontsize=11)
    subtitle = f'{completed} of {len(cases)} runs reach the preset cold-reserve test; each stops at its first passing round'
    fig.text(.5, .86, subtitle, ha='center', fontsize=8, color=GREY)
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .005), ncol=2, frameon=False,
               title='Seed: conditional columns / rows per last-active stratum', title_fontsize=8)
    save(fig, 'numerics-trajectories', args.output_dir)


def endpoints(config, args):
    cases = load_population(config, args.population_dir)
    report, states = load_finite(config, args.finite_dir)
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.6))
    fig.subplots_adjust(left=.085, right=.985, bottom=.27, top=.79, wspace=.54)

    # (a) Population endpoints and the stopping region kappa > D(q) + 0.001.
    observed = [c for c in cases if c['rows']]
    min_q = min(c['q'][-1] for c in observed) * .62
    max_q = max(c['q'][-1] for c in observed) * 1.55
    q_grid = np.geomspace(min_q, max_q, 200)
    boundary = rounding_loss(q_grid) + MINIMUM_RESERVE
    max_kappa = max([boundary[-1]] + [c['rows'][-1]['kappa'] for c in observed]) * 1.22
    min_kappa = min(boundary[0], min(c['rows'][-1]['kappa'] for c in observed)) * .8
    axs[0].fill_between(q_grid, boundary, max_kappa, color=GREEN, alpha=.09)
    axs[0].plot(q_grid, boundary, color=GREY, lw=1, ls='--')
    for i, c in enumerate(observed):
        x, y = c['q'][-1], c['rows'][-1]['kappa']
        axs[0].scatter(x, y, facecolors=c['color'] if c['complete'] else 'none', edgecolors=c['color'],
                       s=32, linewidths=1.3, marker=c['marker'], zorder=4)
        axs[0].annotate(str(c['seed']), (x, y), xytext=[(4, 8), (-28, 8), (4, -12)][i],
                        textcoords='offset points', fontsize=7, color=c['color'])
    axs[0].set(title='(a) Numerical completion test', xlabel=r'Endpoint $\widehat q_t$',
               ylabel=r'Endpoint margin $\kappa_t$', xscale='log', xlim=(min_q, max_q), ylim=(min_kappa, max_kappa))
    axs[0].text(.03, .96, r'Shaded: $\kappa>D(\widehat q)+0.001$', transform=axs[0].transAxes,
                fontsize=7, color=GREY, va='top')

    # (b) Finite free fractions and (c) sign-vector margins from round 35 on.
    plotted = 0
    for run in report['runs']:
        rows = [r for r in run['trajectory'] if 'free_fraction' in r]
        if not rows:
            continue
        color = [ORANGE, GOLD][plotted % 2]
        line = ['-', '--'][plotted % 2]
        plotted += 1
        tt = np.array([r['round'] for r in rows])
        q = np.array([r['free_fraction'] for r in rows])
        nonzero = q > 0
        axs[1].step(tt[nonzero], q[nonzero], where='post', color=color, ls=line, lw=1.35, label=f'Seed {run["seed"]}')
        if q[-1] == 0:
            axs[1].plot(tt[-1], .015, marker='v', color=color, ms=5,
                        transform=axs[1].get_xaxis_transform(), clip_on=False)
        else:
            axs[1].scatter(tt[-1], q[-1], color=color, s=26, marker='x', zorder=4)
        late = [r for r in rows if r['round'] >= 35]
        axs[2].plot([r['round'] for r in late], [r['sign_minimum_margin'] for r in late],
                    color=color, ls=line, lw=1.15, label=f'Seed {run["seed"]}')
        axs[2].scatter(tt[-1], rows[-1]['sign_minimum_margin'], color=color,
                       marker='o' if q[-1] == 0 else 'x', s=27, zorder=4)
    n = report['n']
    axs[1].axhline(1 / n, color=GREY, lw=.75, ls=':')
    axs[1].set(title='(b) Finite free fractions', xlabel='LP round', ylabel=f'Fraction of {n:,} coordinates',
               xlim=(0, 144), yscale='log', ylim=(.00065, 1))
    axs[1].set_xticks([1, 60, 120])
    axs[2].axhline(0, color=GREY, lw=.9, ls=':')
    axs[2].set(title='(c) Late binary checkpoints', xlabel='LP round',
               ylabel='Minimum row margin after signs', xlim=(33, 144))
    for ax in axs:
        light_grid(ax)
    fig.suptitle(r'$\alpha=0.60$: endpoint and finite-matrix diagnostics', y=.96, fontsize=11)
    failed = [str(r['seed']) for r in report['runs'] if not any('free_fraction' in x for x in r['trajectory'])]
    fig.text(.5, .855, 'Population points are estimates; finite seed ' + ', '.join(failed) + ' failed at the first LP',
             ha='center', fontsize=7.8, color=GREY)
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.62, .025), ncol=2, frameon=False,
               title='Finite matrices in panels (b,c)', title_fontsize=8)
    save(fig, 'numerics-endpoints', args.output_dir)


FIGURES = {'distributions': distributions, 'trajectories': trajectories, 'endpoints': endpoints}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--figure', choices=['all'] + list(FIGURES), default='all')
    parser.add_argument('--population-dir', type=Path, default=DATA / 'population')
    parser.add_argument('--finite-dir', type=Path, default=DATA / 'finite_lps')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'output' / 'figures')
    args = parser.parse_args()
    config = load_config('section8')
    set_style()
    for name, function in FIGURES.items():
        if args.figure in ('all', name):
            function(config, args)


if __name__ == '__main__':
    main()
