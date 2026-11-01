#!/usr/bin/env python3
"""Table 1 of the paper: stopping results of the three population runs.

Reads the run records, recomputes the rounding loss D_{1,21/5}(q_t), the
reserve and the resolved-scale checks for every accepted round, confirms that
each run that passed stopped at its first qualifying round, and writes

    <output-dir>/stopping_table.tex     the LaTeX table of Section 8
    <output-dir>/stopping_summary.json  the endpoints and the checks

Example (saved data, takes a second):
    python3 scripts/stopping_table.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iterlp  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402

from iterlp.io import REPO_ROOT, display_path, load_config, read_json, write_json  # noqa: E402
from iterlp.rounding import MINIMUM_RESERVE, is_resolved, rounding_loss  # noqa: E402

CAPTION = (r'Population evaluations of the density-$0.60$ schedule. Each run stops at its first round '
           r'satisfying~\eqref{eq:num-cold-stop}. Particle counts list column particles and row particles '
           r'per stratum, and all entries are floating-point estimates.')


def scientific(x):
    """1.16306e-06 -> 1.16306\\cdot10^{-6}"""
    mantissa, exponent = f'{x:.5e}'.split('e')
    return mantissa + r'\cdot10^{' + str(int(exponent)) + '}'


def particles(power):
    """14 -> 16\\,384"""
    return f'{2 ** power:,}'.replace(',', r'\,')


def check_run(record):
    """Recompute the finishing test for every accepted round of one run."""
    rounds = record['rounds']
    if [r['round'] for r in rounds] != list(range(2, len(rounds) + 2)):
        raise ValueError('%s: the recorded rounds are not contiguous' % record['name'])
    first_passing_round = None
    for r in rounds:
        if not r['accepted']:
            continue
        loss = rounding_loss(r['q'])
        reserve = r['kappa'] - loss
        if abs(loss - r['rounding_loss']) > 1e-12 or abs(reserve - r['reserve']) > 1e-12:
            raise ValueError('%s, round %d: stored rounding loss does not match' % (record['name'], r['round']))
        resolved = is_resolved(r)
        if resolved != r['resolved']:
            raise ValueError('%s, round %d: stored resolved flag does not match' % (record['name'], r['round']))
        if resolved and reserve > MINIMUM_RESERVE and first_passing_round is None:
            first_passing_round = r['round']
    passed = record['status'] == 'passed_finishing_test'
    if passed and (first_passing_round is None or rounds[-1]['round'] != first_passing_round):
        raise ValueError('%s: the run did not stop at its first passing round' % record['name'])
    return passed, first_passing_round


def main():
    config = load_config('section8')
    names = [spec['name'] for spec in config['population']['runs']]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, default=REPO_ROOT / 'data' / 'section8' / 'population',
                        help='folder with seed2047.json, seed2718.json, seed3141.json')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'output')
    args = parser.parse_args()

    cases = []
    for name in names:
        path = args.input_dir / (name + '.json')
        record = read_json(path)
        if record['status'] == 'running':
            raise RuntimeError(name + ' is still running')
        passed, first_passing_round = check_run(record)
        cases.append(dict(name=name, seed=record['seed'], column_power=record['column_power'],
                          row_power=record['row_power'], status=record['status'], passed=passed,
                          first_passing_round=first_passing_round, endpoint=record.get('endpoint'),
                          elapsed_seconds=record.get('elapsed_seconds')))

    summary = dict(description='Stopping results of the Section 8 population runs (Table 1).',
                   input_dir=display_path(args.input_dir), cases=cases,
                   all_passed=all(c['passed'] for c in cases))
    if summary['all_passed']:
        q_values = [c['endpoint']['q'] for c in cases]
        summary['minimum_reserve'] = min(c['endpoint']['reserve'] for c in cases)
        summary['endpoint_q_ratio'] = max(q_values) / min(q_values)
    write_json(args.output_dir / 'stopping_summary.json', summary)

    lines = [r'\begin{table}[htbp]', r'\centering', r'\caption{' + CAPTION + '}',
             r'\label{tab:num-060-stopping-replicas}', r'\begin{tabular}{rrrrrr}', r'\toprule',
             r'Seed & Particles & Stop $t$ & $\widehat q_t$ & $\kappa_t$ & Reserve\\', r'\midrule']
    for c in cases:
        e = c['endpoint']
        if not c['passed']:
            lines.append(f"{c['seed']} & ${particles(c['column_power'])}/{particles(c['row_power'])}$ "
                         f"& \\multicolumn{{4}}{{l}}{{{c['status'].replace('_', ' ')}}}\\\\")
            continue
        lines.append(f"{c['seed']} & ${particles(c['column_power'])}/{particles(c['row_power'])}$ & ${e['round']}$ "
                     f"& ${scientific(e['q'])}$ & ${e['kappa']:.7f}$ & ${e['reserve']:.7f}$" + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    tex = args.output_dir / 'stopping_table.tex'
    tex.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(table=display_path(tex), all_passed=summary['all_passed'],
                          stops={c['name']: c['first_passing_round'] for c in cases},
                          minimum_reserve=summary.get('minimum_reserve')), indent=2))


if __name__ == '__main__':
    main()
