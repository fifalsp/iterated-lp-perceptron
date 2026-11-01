#!/usr/bin/env python3
"""Run the population recursion along the Section 8 schedule (density 0.60).

Each run starts from the closed-form first round and solves rounds
t = 2, ..., 140 of the particle approximation. It stops at the first accepted,
numerically resolved round whose reserve kappa_t - D_{1,21/5}(q_t) exceeds
10^-3 (status "passed_finishing_test"). Otherwise it ends with the first
rejected round ("rejected_round"), after round 140 ("reached_max_rounds"),
at the 900-second cap ("time_cap") or on an error ("error"). Every outcome is
written to <output-dir>/<run>.json together with particle snapshots in
<output-dir>/<run>_snapshots/.

As in the paper, seed 3141 is run only if seed 2047 has passed the test.

Example:
    python3 scripts/run_population.py --runs seed2047 --output-dir results/population
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iterlp  # noqa: E402,F401  (limits BLAS to one thread before NumPy loads)

import argparse  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from iterlp.io import REPO_ROOT, environment, load_config, read_json, write_json  # noqa: E402
from iterlp.population import Population  # noqa: E402
from iterlp.rounding import add_finishing_test  # noqa: E402
from iterlp.schedules import schedule_from_config  # noqa: E402

ENDPOINT_FIELDS = ['round', 'q', 'kappa', 'rho', 'tau', 'effective_tau', 'free_squared_norm', 'value_squared',
                   'maximum_scaled_residual', 'rounding_loss', 'reserve', 'resolved', 'passes_finishing_test']
COVARIANCE_FIELDS = ['unclipped_score_covariance', 'unclipped_score_objective_covariance']


class TimeCap(Exception):
    pass


def on_alarm(signum, frame):
    raise TimeCap('time cap reached')


def save_snapshot(pop, directory, round_number, kappa, report, endpoint=False):
    """Save the current particle state (not needed for any table or figure)."""
    directory.mkdir(parents=True, exist_ok=True)
    name = 'endpoint.npz' if endpoint else 'round_%03d.npz' % round_number
    np.savez(directory / name, column_current=pop.X[:, -1], margin=pop.U[:, -1], weight=pop.row_weights,
             last_active_round=pop.row_strata, q=pop.q, kappa=kappa, round=round_number)
    report['snapshots'].append(dict(round=round_number, file=directory.name + '/' + name, q=pop.q, kappa=kappa))


def run(config, spec, output_dir):
    alpha = config['alpha']
    settings = config['population']
    schedule = schedule_from_config(config)
    snapshot_rounds = set(settings['snapshot_rounds'])
    path = output_dir / (spec['name'] + '.json')
    snapshot_dir = output_dir / (spec['name'] + '_snapshots')
    report = dict(
        description='Population recursion at density %g along the quadratic schedule of Section 8.' % alpha,
        name=spec['name'], alpha=alpha, seed=spec['seed'],
        column_power=spec['column_power'], row_power=spec['row_power'],
        max_nfev=settings['max_nfev'], time_cap_seconds=settings['time_cap_seconds'],
        schedule_parameters=config['schedule'], schedule=schedule,
        status='running', first_round=None, rounds=[], snapshots=[],
        environment=environment())
    started = time.monotonic()

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(path, report)

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(settings['time_cap_seconds'])
    try:
        pop = Population(alpha, schedule[0], spec['column_power'], spec['row_power'], spec['seed'])
        report['first_round'] = pop.first
        report['rounds'] = pop.records
        save_snapshot(pop, snapshot_dir, 1, schedule[0], report)
        save()
        print(json.dumps(dict(run=spec['name'], round=1, q=pop.q)), flush=True)
        for kappa in schedule[1:]:
            record = pop.solve(kappa, max_nfev=settings['max_nfev'])
            if record['accepted']:
                add_finishing_test(record)
                if record['round'] in snapshot_rounds:
                    save_snapshot(pop, snapshot_dir, record['round'], kappa, report)
                if record['passes_finishing_test']:
                    save_snapshot(pop, snapshot_dir, record['round'], kappa, report, endpoint=True)
                    report['status'] = 'passed_finishing_test'
                    break
            # Keep the full covariance matrices only every tenth round.
            if len(pop.records) > 1 and pop.records[-2]['round'] % 10:
                for field in COVARIANCE_FIELDS:
                    pop.records[-2].pop(field, None)
            if record['round'] % 10 == 0:
                save()
            if record['round'] % 20 == 0 or record['round'] in (2, 3, 134, 140) or not record['accepted']:
                print(json.dumps(dict(run=spec['name'], round=record['round'], q=record['q'], kappa=kappa,
                                      tau=record['tau'], accepted=record['accepted'],
                                      residual=record['maximum_scaled_residual'], reserve=record.get('reserve'),
                                      seconds=round(time.monotonic() - started, 1))), flush=True)
            if not record['accepted']:
                report['status'] = 'rejected_round'
                break
        else:
            report['status'] = 'reached_max_rounds'
    except TimeCap as exc:
        report['status'] = 'time_cap'
        report['error'] = repr(exc)
    except Exception as exc:
        report['status'] = 'error'
        report['error'] = repr(exc)
    finally:
        signal.alarm(0)

    accepted = [r for r in report['rounds'] if r['accepted']]
    report['accepted_rounds'] = len(accepted)
    if accepted:
        report['endpoint'] = {field: accepted[-1].get(field) for field in ENDPOINT_FIELDS}
        report['maximum_scaled_residual'] = max(r['maximum_scaled_residual'] for r in accepted)
    save()
    print(json.dumps(dict(run=spec['name'], status=report['status'], endpoint=report.get('endpoint'),
                          seconds=round(report['elapsed_seconds'], 1))), flush=True)
    return report


def main():
    config = load_config('section8')
    runs = {spec['name']: spec for spec in config['population']['runs']}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', choices=list(runs), default=list(runs),
                        help='runs to execute, in this order (default: all three)')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'results' / 'population')
    parser.add_argument('--ignore-gate', action='store_true',
                        help='run seed3141 even if seed2047 has not passed the finishing test')
    parser.add_argument('--overwrite', action='store_true', help='replace existing output files')
    parser.add_argument('--time-cap', type=int, default=None,
                        help='seconds per run (default 900, as in the paper); raise it on a slow machine')
    parser.add_argument('--worker', choices=list(runs), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.time_cap is not None:
        config['population']['time_cap_seconds'] = args.time_cap
    if args.worker:
        run(config, runs[args.worker], args.output_dir)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.runs:
        spec = runs[name]
        path = args.output_dir / (name + '.json')
        if path.exists() and not args.overwrite:
            sys.exit('%s exists; use --overwrite or another --output-dir' % path)
        gate = spec.get('run_only_if_passed')
        if gate and not args.ignore_gate:
            gate_path = args.output_dir / (gate + '.json')
            if not gate_path.exists() or read_json(gate_path).get('status') != 'passed_finishing_test':
                print('%s not run: %s has not passed the finishing test in %s' % (name, gate, args.output_dir))
                continue
        # Each run gets a fresh process: the late rounds use several GB of
        # memory, and a fresh heap keeps the run time (and the cap) predictable.
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', name,
                   '--output-dir', str(args.output_dir)]
        if args.time_cap is not None:
            command += ['--time-cap', str(args.time_cap)]
        subprocess.run(command, check=False)


if __name__ == '__main__':
    main()
