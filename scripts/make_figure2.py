#!/usr/bin/env python3
"""Figure 2 of the paper (Section 1.4): state evolution and universality at alpha = 0.40.

Top row: row margins u_i^(t); bottom row: coordinates x_j^(t); rounds 1-3 with
(kappa_1, kappa_2, kappa_3) = (0.60, 0.30, 0.03). Histograms pool the four
Gaussian (filled blue) and the four Rademacher (orange outline) instances at
N = 2048. The black curves are the predicted densities and the black stems
the predicted atoms (tight rows at kappa_t, frozen coordinates at +-1); the
markers on the stems are the empirical atom masses. The predictions come from
the population recursion alone (scripts/compute_alpha040_theory.py).

Fonts match the paper (LaTeX with mathpazo). If no LaTeX installation is
found, Matplotlib's own math rendering is used instead.

Example:
    python3 scripts/make_figure2.py
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
from matplotlib.legend_handler import HandlerBase, HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from iterlp.io import REPO_ROOT, display_path, load_config  # noqa: E402

DATA = REPO_ROOT / 'data' / 'alpha040'
ATOM_TOLERANCE = 1e-7
BLUE, ORANGE, BLACK = '#2a78d6', '#eb6834', '#20252B'
PREDICTION_ALPHA, PREDICTION_LW = .6, 1.0
CAP = .07  # half-width of the cap on an atom stem, in data units


def set_style():
    if shutil.which('latex'):
        plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 9,
                             'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage{mathpazo}',
                             'axes.linewidth': .6, 'pdf.fonttype': 42})
    else:
        print('LaTeX not found; using Matplotlib math text (fonts will differ from the paper).')
        plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'font.size': 9,
                             'axes.linewidth': .6, 'pdf.fonttype': 42})


def style_axis(ax, xlim):
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(*xlim)
    ax.tick_params(axis='x', length=2.5, pad=2, labelsize=8)


def histogram_pair(ax, gaussian, rademacher, bins):
    """Gaussian histogram as filled bars, Rademacher as an outline; both normalized by the total count."""
    for (values, total), color, filled in ((gaussian, BLUE, True), (rademacher, ORANGE, False)):
        heights, edges = np.histogram(values, bins=bins)
        heights = heights / (total * np.diff(edges))
        if filled:
            ax.bar(edges[:-1], heights, width=np.diff(edges), align='edge', color=color, alpha=.35,
                   edgecolor='white', linewidth=.3, zorder=2)
        else:
            ax.step(np.r_[edges[:-1], edges[-1]], np.r_[heights, heights[-1]], where='post',
                    color=color, lw=1.2, zorder=4)


def atom(ax, x, predicted, gaussian_mass, rademacher_mass, cap):
    """Predicted atom as a capped stem; empirical masses as markers."""
    ax.plot([x, x], [0, predicted], color=BLACK, alpha=PREDICTION_ALPHA, lw=PREDICTION_LW,
            solid_capstyle='butt', zorder=3)
    ax.plot([x - cap, x + cap], [predicted, predicted], color=BLACK, alpha=PREDICTION_ALPHA, lw=PREDICTION_LW,
            solid_capstyle='butt', zorder=3)
    ax.plot([x], [gaussian_mass], 'o', ms=4.2, color=BLUE, mec=BLUE, zorder=6, clip_on=False)
    ax.plot([x], [rademacher_mass], 'o', ms=7.2, mfc='none', mec=ORANGE, mew=1.0, zorder=7, clip_on=False)


class StemHandler(HandlerBase):
    """Legend entry drawn as a capped stem."""

    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        x = xd + w / 2
        return [Line2D([x, x], [yd - .15 * h, yd + 1.05 * h], color=BLACK, alpha=PREDICTION_ALPHA,
                       lw=PREDICTION_LW, transform=trans),
                Line2D([x - .22 * w, x + .22 * w], [yd + 1.05 * h] * 2, color=BLACK, alpha=PREDICTION_ALPHA,
                       lw=PREDICTION_LW, transform=trans)]


def main():
    config = load_config('alpha040')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--theory', type=Path, default=DATA / 'population' / 'theory_curves.npz')
    parser.add_argument('--finite-dir', type=Path, default=DATA / 'finite_lps')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'output' / 'figures')
    args = parser.parse_args()
    set_style()

    theory = np.load(args.theory)
    rounds = [dict(t=t, **{k: theory[f'{k}_{t}'] for k in ('x', 'xf', 'u', 'uf', 'k', 'q', 'atom')})
              for t in (1, 2, 3)]
    seeds = config['finite_lps']['seeds']
    gaussian = [np.load(args.finite_dir / f'gaussian_seed{s}.npz') for s in seeds]
    rademacher = [np.load(args.finite_dir / f'rademacher_seed{s}.npz') for s in seeds]

    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.55))
    fig.subplots_adjust(left=.055, right=.99, top=.83, bottom=.075, wspace=.1, hspace=.42)
    ytop_u = 1.1 * max(max(r['uf'].max(), r['atom']) for r in rounds)
    ytop_x = 1.12 * max(max(r['xf'].max(), (1 - r['q']) / 2) for r in rounds)
    for j, r in enumerate(rounds):
        t, k = r['t'], float(r['k'])

        # Margins: density above kappa_t and the atom at kappa_t.
        ax = axs[0, j]
        ug = np.concatenate([d['u'][t - 1] for d in gaussian])
        ur = np.concatenate([d['u'][t - 1] for d in rademacher])
        histogram_pair(ax, (ug[ug > k + ATOM_TOLERANCE], len(ug)), (ur[ur > k + ATOM_TOLERANCE], len(ur)),
                       np.arange(k, 3.41, .15))
        ax.plot(r['u'], r['uf'], color=BLACK, alpha=PREDICTION_ALPHA, lw=PREDICTION_LW, zorder=3)
        atom(ax, k, float(r['atom']), np.mean(ug <= k + ATOM_TOLERANCE), np.mean(ur <= k + ATOM_TOLERANCE), CAP)
        style_axis(ax, (-.12, 3.4))
        ax.set_ylim(0, ytop_u)
        ax.set_xticks([k, 1, 2, 3])
        ax.set_xticklabels([rf'$\kappa_{t}$', '$1$', '$2$', '$3$'])
        ax.set_title(f'Round {t}', fontsize=10, pad=6)

        # Coordinates: density on (-1, 1) and the frozen atoms at -1 and +1.
        ax = axs[1, j]
        xg = np.concatenate([d['x'][t - 1] for d in gaussian])
        xr = np.concatenate([d['x'][t - 1] for d in rademacher])
        histogram_pair(ax, (xg[np.abs(xg) < 1 - ATOM_TOLERANCE], len(xg)), (xr[np.abs(xr) < 1 - ATOM_TOLERANCE], len(xr)),
                       np.linspace(-1, 1, 15))
        ax.plot(r['x'], r['xf'], color=BLACK, alpha=PREDICTION_ALPHA, lw=PREDICTION_LW, zorder=3)
        for sign in (-1, 1):
            atom(ax, sign, (1 - float(r['q'])) / 2, np.mean(sign * xg >= 1 - ATOM_TOLERANCE),
                 np.mean(sign * xr >= 1 - ATOM_TOLERANCE), CAP * 2.3 / 3.52)
        style_axis(ax, (-1.12, 1.12))
        ax.set_ylim(0, ytop_x)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['$-1$', '$0$', '$1$'])

    axs[0, 0].set_ylabel(r'Margins $u_i^{(t)}$', fontsize=9.5, labelpad=4)
    axs[1, 0].set_ylabel(r'Coordinates $x_j^{(t)}$', fontsize=9.5, labelpad=4)
    stem = object()
    fig.legend(handles=[(Patch(facecolor=BLUE, alpha=.35), Line2D([], [], ls='none', marker='o', ms=4.2, color=BLUE)),
                        (Line2D([], [], color=ORANGE, lw=1.2),
                         Line2D([], [], ls='none', marker='o', ms=7.2, mfc='none', mec=ORANGE)),
                        Line2D([], [], color=BLACK, alpha=PREDICTION_ALPHA, lw=PREDICTION_LW), stem],
               labels=['Gaussian LP', 'Rademacher LP', 'Predicted density', 'Predicted atom'],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=.6), stem: StemHandler()},
               loc='upper center', bbox_to_anchor=(.52, 1.0), ncol=4, frameon=False, fontsize=8.5,
               handlelength=2.6, columnspacing=1.8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension, options in (('pdf', {'metadata': {'CreationDate': None}}), ('png', {'dpi': 220})):
        path = args.output_dir / ('state-evolution.' + extension)
        fig.savefig(path, bbox_inches='tight', pad_inches=.03, **options)
        print('wrote', display_path(path))
    plt.close(fig)
    for r in rounds:
        print('round', r['t'], 'predicted margin atom', round(float(r['atom']), 4),
              'predicted mass at each of +-1', round((1 - float(r['q'])) / 2, 4))


if __name__ == '__main__':
    main()
