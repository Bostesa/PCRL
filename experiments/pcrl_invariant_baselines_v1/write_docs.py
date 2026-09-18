"""Stage 8b: turn the committed JSON/CSV evidence into the required markdown reports.

Reads only artifacts this study already wrote. Computes no new endpoint and fits
nothing, so a report can always be regenerated from the evidence without re-running a
model job.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .report import ERASURE, OPTNET, REPAIRED
from .run_fit import OUT

FAMILY_ENDPOINTS = ('utility/same_residence', 'recovery/A/SEX', 'recovery/AB/SEX',
                    'recovery/A/RAC1P', 'recovery/AB/RAC1P')
SHORT = {'utility/same_residence': 'residence', 'recovery/A/SEX': 'A/SEX',
         'recovery/AB/SEX': 'AB/SEX', 'recovery/A/RAC1P': 'A/RAC1P',
         'recovery/AB/RAC1P': 'AB/RAC1P'}


def read_csv(path: Path) -> list:
    if not Path(path).exists():
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def cell(row: dict) -> str:
    """Estimate with its adjusted simultaneous interval, and a significance mark."""
    est = float(row['estimate'])
    low, high = float(row['adjusted_low']), float(row['adjusted_high'])
    mark = ''
    if row['significantly_better'] in ('True', True):
        mark = ' **-**'
    elif row['significantly_worse'] in ('True', True):
        mark = ' **+**'
    return f'{est:+.4f} [{low:+.4f}, {high:+.4f}]{mark}'


def contrast_table(rows, left, right, weight) -> list:
    picked = {r['endpoint']: r for r in rows
              if r['left'] == left and r['right'] == right and r['weight'] == weight}
    if not picked:
        return []
    line = [f'| `{left}` vs `{right}` ({weight}) |']
    for endpoint in FAMILY_ENDPOINTS:
        line.append(f' {cell(picked[endpoint]) if endpoint in picked else "--"} |')
    return [''.join(line)]


def header() -> list:
    return ['| contrast | ' + ' | '.join(SHORT[e] for e in FAMILY_ENDPOINTS) + ' |',
            '|---|' + '---|' * len(FAMILY_ENDPOINTS)]


def development_report(out: Path) -> Path:
    record = json.loads((out / 'DEVELOPMENT_2018.json').read_text())
    intervals = read_csv(out / 'PAIRED_INTERVALS.csv')
    gate = json.loads((out / 'MECHANISM_GATE.json').read_text())

    lines = [
        '# DEVELOPMENT_2018 — repaired objective and external baselines',
        '',
        '**These are DEVELOPMENT numbers on pools that have been used repeatedly.** The',
        'paired household-cluster bootstrap quantifies sampling variability for fixed fitted',
        'systems. It cannot undo repeated use, and it is neither Census replicate-weight',
        'variance nor retraining variability. Every seed and both weightings are reported.',
        '',
        '## Reading the tables',
        '',
        'Each cell is `estimate [adjusted simultaneous 95% interval]` for **left minus right**.',
        'For `recovery/*` a **negative** estimate means the left arm leaks **less**; for',
        '`utility/same_residence` a **positive** estimate means the left arm has **more**',
        'residence capability. `**-**` marks an adjusted interval entirely below zero,',
        '`**+**` one entirely above. The adjustment is single-step studentized max-|t| across',
        'the five family endpoints within each contrast and weighting, exactly the completed',
        "study's procedure.",
        '',
        f'Bootstrap: {record["bootstrap"]["replicates"]} replicates, '
        f'{record["bootstrap"]["clusters"]} cohort-household clusters, seed '
        f'{record["bootstrap"]["seed"]}.',
        '',
        '## Mechanism gate, recorded before any score below was read',
        '',
        f'* Rotation-only share of the training gain: **max abs '
        f'{gate["gate"]["max_abs_share"]:.2e}** across {gate["gate"]["cells"]} cells '
        f'(gate `< 0.10`, registered forecast `< 0.01`). **Passes: '
        f'{gate["gate"]["passes_gate"]}.**',
        f'* Direct invariance identity: the penalty moves by at most '
        f'**{gate["max_direct_invariance_change"]:.2e}** under random rotations.',
        f'* Predecessor, same diagnostic, same cells: mean **0.628**, range 0.367-0.907 — and',
        '  that share is an **attained** value, not a supremum, because every rotation search',
        '  stopped on budget exhaustion or line-search failure.',
        '',
        '## Q3 — does removing the provably inert slack change MEASURED recovery?',
        '',
        'Repaired arm minus the defective predecessor, at matched rank and policy.',
        '',
    ]
    lines += header()
    for arm in REPAIRED:
        rank = '16' if 'riv16' in arm else '8'
        policy = arm.rsplit('_', 1)[1]
        for weight in ('unweighted', 'person_weighted'):
            lines += contrast_table(intervals, arm, f'spectral_nlr{rank}_{policy}', weight)
    lines += ['', '## The repaired arm against its own linear-moment control', '']
    lines += header()
    for arm in REPAIRED:
        rank = '16' if 'riv16' in arm else '8'
        policy = arm.rsplit('_', 1)[1]
        right = f'spectral_{policy}' if rank == '16' else f'spectral_lin8_{policy}'
        for weight in ('unweighted', 'person_weighted'):
            lines += contrast_table(intervals, arm, right, weight)
    lines += ['', '## Q5 — the repaired arm against the frozen neural channel J', '']
    lines += header()
    for arm in REPAIRED:
        for weight in ('unweighted', 'person_weighted'):
            lines += contrast_table(intervals, arm, 'J', weight)
    lines += ['', '## External adaptations against J', '']
    lines += header()
    for arm in ERASURE + OPTNET:
        for weight in ('unweighted', 'person_weighted'):
            lines += contrast_table(intervals, arm, 'J', weight)
    lines += ['', '## External adaptations against the repaired arm', '']
    lines += header()
    for arm in ERASURE + OPTNET:
        for weight in ('unweighted', 'person_weighted'):
            lines += contrast_table(intervals, arm, 'spectral_riv16_C1', weight)
    lines += ['', '## Erasure baselines against the source channel they transformed', '']
    lines += header()
    for arm in ERASURE:
        for weight in ('unweighted', 'person_weighted'):
            lines += contrast_table(intervals, arm, 'A0', weight)

    lines += ['', '## Q4 — the registered coordination cells', '',
              'The coordination rule is a one-sided **point-estimate** rule, reused for',
              'continuity. **A pass is not statistical equivalence and not noninferiority',
              'within .001.** Equivalence and noninferiority are reported separately per',
              'contrast in `DEVELOPMENT_2018.json`; absence of significance is neither.', '',
              '| cell | supported | advantage on all | residence within .001 on all | local race significantly worse |',
              '|---|---|---|---|---|']
    for name, decision in record['registered_decisions'].items():
        lines.append(f'| `{name}` | **{decision.get("supported")}** | '
                     f'{decision.get("advantage_all")} | '
                     f'{decision.get("residence_within_001_all")} | '
                     f'{decision.get("local_race_significantly_worse_any")} |')

    lines += ['', '## Numerical integrity', '',
              f'* Independent score replay: **{record["score_replay"]["checks"]} checks**, '
              f'maximum absolute difference '
              f'**{record["score_replay"]["max_abs_difference"]:.2e}**.',
              '* Every arm\'s `H_A` and `H_B` coordinates were asserted **bitwise** on all seven',
              '  released pools, at release time and again at evaluation time against the frozen',
              '  historical anchors.', '']
    path = out / 'DEVELOPMENT_2018.md'
    path.write_text('\n'.join(lines) + '\n')
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    print('wrote', development_report(a.out))


if __name__ == '__main__':
    main()
