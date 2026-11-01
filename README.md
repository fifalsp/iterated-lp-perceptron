# Iterated linear programs for the binary perceptron: code and data

This repository contains the code and the saved results behind the numerical
parts of

> Jianan Zhang and Shuangping Li,
> *State Evolution and Universality of Iterated Linear Programs for the Binary Perceptron*, 2026.

It covers Figure 2 (Section 1.4, density 0.40) and the numerical evaluation
of Section 8 (density 0.60): Table 1 and Figure 4.
Every table and figure can be rebuilt from the saved data in about 15
seconds, and every experiment can be rerun from scratch (about 20 minutes on
a laptop).

## What is reproduced

| Paper item | Rebuild from saved data | Saved data used | Rerun the computation |
|---|---|---|---|
| **Figure 2** (Section 1.4): predicted and empirical laws of margins and coordinates, alpha = 0.40, N = 2048, Gaussian and Rademacher | `python3 scripts/make_figure2.py` | `data/alpha040/population/theory_curves.npz`, `data/alpha040/finite_lps/{gaussian,rademacher}_seed940{1..4}.npz` | `scripts/compute_alpha040_theory.py`, `scripts/run_alpha040_lps.py` |
| **Table 1** (Section 8.2): stopping results of the three population runs | `python3 scripts/stopping_table.py` | `data/section8/population/seed{2047,2718,3141}.json` | `scripts/run_population.py` |
| **Figure 4** (Section 8.2): (a) free fraction of the population runs and the finite LPs, (b) margin versus rounding cost | `python3 scripts/make_section8_figure.py` | `data/section8/population/*.json`, `data/section8/finite_lps/report.json` | `scripts/run_population.py`, `scripts/run_finite_lps.py` |

Figure numbers refer to the current version of the paper. The figures as
they appear in the paper are in `figures/` (`state-evolution.pdf` and
`numerics.pdf`, with PNG previews).

## Installation

Python 3.9 to 3.12 with NumPy, SciPy and Matplotlib:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`requirements.txt` pins numpy 2.0.2, scipy 1.13.1 and matplotlib 3.9.4, the
versions used for Section 8. Figure 2 uses LaTeX (with the `mathpazo`
package) for its fonts when a LaTeX installation is found, and Matplotlib's
own math text otherwise. The population runs use `signal.SIGALRM` for their
time cap, so they need macOS or Linux. All scripts limit BLAS to one thread.

## Rebuilding the table and figures from saved data

These commands take about 15 seconds in total and solve no LP or population
problem. They write to `output/` (ignored by git).

```bash
python3 scripts/stopping_table.py          # output/stopping_table.tex and output/stopping_summary.json
python3 scripts/make_section8_figure.py    # output/figures/numerics.{pdf,png}
python3 scripts/make_figure2.py            # output/figures/state-evolution.{pdf,png}
python3 scripts/verify_finite_lps.py       # check of the saved Section 8 LP states
python3 scripts/verify_alpha040_lps.py     # check of the saved alpha = 0.40 LP states
```

`stopping_table.py` recomputes the rounding loss D_{1,21/5}(q_t), the
reserve and the resolved-scale checks for every accepted round, confirms that
each run stopped at its first qualifying round, and writes Table 1 exactly as
it appears in the paper. `make_section8_figure.py` checks that the saved
rounding losses match D_{1,21/5}(q_t) and that every run first passes the
test at its last round. In our check with the pinned versions, Figure 4 came
out byte-for-byte identical to `figures/numerics.pdf`. Figure 2 was drawn for the paper with Matplotlib 3.10.8; with that
version its PNG is identical to `figures/state-evolution.png`, and with 3.9.4
it is visually identical. Image bytes can also depend on installed fonts.

The two `verify_*` scripts regenerate every matrix and objective from its
seed, check their hashes, and check each saved LP state: the row products
`A @ x` against the saved margins (to 1e-12; the last bits depend on the BLAS
build), the box and margin constraints, that frozen coordinates are +-1 and
never change, that the free sets are nested, and the recorded counts.

## Rerunning the experiments

Each script writes to `results/` by default (`--output-dir` changes this).
Timings were measured on an Apple M1 Pro laptop with one BLAS thread.

**Section 8 population runs** (about 15 minutes in total: 3.5 to 4 minutes
each for seeds 2047 and 2718, about 7 minutes for seed 3141):

```bash
python3 scripts/run_population.py --output-dir results/population
python3 scripts/stopping_table.py --input-dir results/population --output-dir results
python3 scripts/make_section8_figure.py --population-dir results/population --output-dir results/figures
```

The settings are in `configs/section8.json`: density 0.60, the schedule
kappa_t = 0.01 + 0.5789153654719907 ((150 - t)/149)^2 for t = 1, ..., 140,
2^14 column particles and 2^10 (seeds 2047 and 2718) or 2^11 (seed 3141) row
particles per stratum, at most 100 function evaluations per root attempt, and
a cap of 900 seconds per run. A run stops at the first accepted, numerically
resolved round with kappa_t - D_{1,21/5}(q_t) > 10^-3 (status
`passed_finishing_test`); otherwise it ends with the first rejected round
(`rejected_round`), after round 140 (`reached_max_rounds`), at the time cap
(`time_cap`) or on an error (`error`). As in the paper, seed 3141 is run only
after seed 2047 has passed. Use `--runs seed2047` for a single run. Each run
is executed in a fresh process and also saves particle snapshots
(`results/population/<run>_snapshots/`, 8 to 10 MB per run), which no table
or figure needs and which are not included in `data/`. The late rounds need
up to about 5 GB of memory (seed 3141). On a slow machine a run can reach the
900-second cap (status `time_cap`); `--time-cap` raises it, which departs from
the setting of the paper.

**Section 8 finite LPs** (about one minute):

```bash
python3 scripts/run_finite_lps.py --output-dir results/finite_lps
python3 scripts/verify_finite_lps.py --folder results/finite_lps
python3 scripts/make_section8_figure.py --finite-dir results/finite_lps --output-dir results/figures
```

Gaussian matrices with N = 1024 and M = 614 for the seeds 6101, 6102 and 6103,
solved with HiGHS (`scipy.optimize.linprog`, primal and dual feasibility
tolerances 1e-9). The script first recomputes the population state after
rounds 1 to 3 (seed 2047) for the comparison of frozen fractions in
Section 8.3.

**Figure 2** (population curves in a few seconds; LPs in about one minute with
three worker processes):

```bash
python3 scripts/compute_alpha040_theory.py --output-dir results/alpha040_theory
python3 scripts/run_alpha040_lps.py --output-dir results/alpha040_lps --workers 3
python3 scripts/verify_alpha040_lps.py --folder results/alpha040_lps
python3 scripts/make_figure2.py --theory results/alpha040_theory/theory_curves.npz \
    --finite-dir results/alpha040_lps --output-dir results/figures
```

The population curves use 2^16 column and 2^12 row particles (seed 1729);
a replicate with 2^15 column particles (seed 2718) is computed alongside, and
`stability.json` records the differences between the two. No LP data enter
the predicted curves.

### What to expect from a rerun

The scripts are deterministic, but the population recursion runs for up to
140 rounds, and differences in the last bits of BLAS and LAPACK results
(between library builds, operating systems or processors) grow along the way.
The saved runs were made on 13 September 2026 with Python 3.9, numpy 2.0.2
and scipy 1.13.1 on macOS. Rerunning them with the same versions on a
later macOS release, the computed q_t, tau_t, rho_t and V_t^2 differed from
the saved records by at most 4e-9 in relative terms for seeds 2047 and 2718,
and by at most 5e-7 for seed 3141, over all rounds. All three runs again
passed the finishing test at round 119, and Table 1 was unchanged at the
printed precision. The finite LPs depend on the population runs only through
the schedule. Rerun with the pinned versions, they freeze exactly the same
coordinates in every round and end in the same way (seed 6101 infeasible at
the first LP; integral outputs at rounds 120 and 109; sign vectors feasible
from rounds 44 and 40), while the LP iterates differ by at most 1e-8. The
alpha = 0.40 data were made with numpy 1.26.4 and scipy 1.11.4; with those
versions the predicted curves and all eight LP runs were reproduced bit for
bit; with the pinned versions they agree to about 1e-14 and 4e-8,
respectively, with identical frozen sets.

## Repository layout

```
iterlp/                    the numerical code
  population.py            particle approximation of the population recursion (state evolution)
  clipped_gaussian.py      closed-form moments of the clipped Gaussian coordinate update
  rounding.py              rounding loss D_{a,T}(q), resolved-scale checks and the finishing test
  schedules.py             the margin schedule kappa_t
  finite_lp.py             Iterated-LP on finite matrices with HiGHS
  io.py                    reading and writing records
scripts/                   command-line entry points (see above)
configs/                   section8.json and alpha040.json: all run settings
data/                      saved results (about 18 MB)
  section8/population/     seed2047.json, seed2718.json, seed3141.json: complete per-round records
  section8/stopping_table.tex, stopping_summary.json
  section8/finite_lps/     report.json, seed610{1,2,3}.npz, population_first_three.npz, check.json
  alpha040/population/     theory_curves.npz, replicate_curves.npz, their reports, stability.json
  alpha040/finite_lps/     report.json, {gaussian,rademacher}_seed940{1..4}.npz, check.json
figures/                   the figures as they appear in the paper
```

**Population records** (`data/section8/population/*.json`) hold the settings,
the schedule, the closed-form first round, one record per solved round, the
final status and the endpoint. Each round record contains, in the notation of
the paper, the free fraction `q` (q_t), `kappa`, `tau`, `rho`,
`value_squared` (V_t^2), `gamma`, `width` (c_t = q_t / tau_t), the history
coefficients under `parameters` (`eta_x`, `eta_r`, `eta_lambda`, `a`, `b`),
the residuals (`maximum_scaled_residual` and its parts), the solver attempts,
Gram-matrix diagnostics, and for accepted rounds `rounding_loss`
(D_{1,21/5}(q_t)), `reserve`, `resolved` and `passes_finishing_test`. Full
covariance matrices of the unclipped scores are kept every tenth round and
for the last two rounds.

**Finite-LP states** (`*.npz`) hold arrays `x` (rounds x N), `u = A x`
(rounds x M) and `free` (rounds x N) after each solved round; `report.json`
records the per-round statistics, the hashes of the matrices and objectives,
and the outcome of each run (`lp_failure`, `integral`,
`completed_all_rounds`, ...). `population_first_three.npz` holds, for
t = 1, 2, 3, the column particles `x_t`, the row margins `u_t` with weights
`w_t`, and `q_t`. `theory_curves.npz` holds, for t = 1, 2, 3, the coordinate
grid and density `x_t`, `xf_t` (integrating to q_t; each endpoint +-1 carries
(1 - q_t)/2), the margin grid and density `u_t`, `uf_t`, the margin `k_t`,
`q_t`, and the margin atom `atom_t`.

## Numerical status

The population computations are numerical approximations, not certified
results. Each represents the population state by finitely many weighted
particles, with scrambled Sobol sampling, importance resampling and
floating-point quadrature, and solves the variational problem of each round
by nonlinear least squares. Nothing bounds the accumulated discretization,
sampling or solver error along the long schedule (up to 140 rounds). Small
residuals and agreement across seeds or particle counts are not error bounds.
Passing the finishing test is numerical evidence for the schedule at
density 0.60, not a rigorous threshold.

The finite LP runs are floating-point illustrations with a few prespecified
seeds. They do not estimate a success probability, and they do not simulate
the final rounding stage. Every prespecified case is retained, including
failures: the first LP is infeasible for seed 6101.

## Citation

> **TODO:** replace `XXXX.XXXXX` by the arXiv identifier once it is available.

```bibtex
@misc{ZhangLi2026IteratedLP,
  author = {Zhang, Jianan and Li, Shuangping},
  title  = {State Evolution and Universality of Iterated Linear Programs for the Binary Perceptron},
  year   = {2026},
  note   = {arXiv preprint arXiv:XXXX.XXXXX}
}
```

## License

MIT; see `LICENSE`.
