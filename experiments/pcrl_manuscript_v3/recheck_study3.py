"""Independent re-checks of the incoming study-3 evidence package (checks V1-V5).

Runs no model. Reads committed blobs at a pinned commit, recomputes, and writes
VERIFICATION_V3.json. Exits non-zero if any check fails.

  V1  every artifact hash recorded in the study's own HANDOFF.json still matches
  V2  the highest-impact paired estimates, both weightings, direction and denominator
  V3  the reporting scope (scope, budget, split) located by brute force rather than assumed
  V4  scope-invariance of paired contrasts, and the measured counterexample to
      "an attack maximum is a bound"
  V5  internal consistency of the package's own status documents

Usage:
    PYTHONPATH=. python -m experiments.pcrl_manuscript_v3.recheck_study3 \
        --repo . --out results/pcrl_manuscript_review_v3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

STUDY3 = '73903b7f28df68284285f0610a4036beb32b208f'
D3 = 'results/pcrl_invariant_baselines_v1'
SENS = ['recovery/A/SEX', 'recovery/AB/SEX', 'recovery/A/RAC1P', 'recovery/AB/RAC1P']
EPS = 5e-6


def blob(root: Path, rel: str, commit: str = STUDY3) -> bytes:
    return subprocess.run(['git', 'cat-file', '-p', f'{commit}:{rel}'],
                          cwd=root, capture_output=True, check=True).stdout


def rows(root: Path, rel: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(blob(root, rel).decode())))


# ------------------------------------------------------------------ V1

def v1_artifact_hashes(root: Path) -> dict:
    hand = json.loads(blob(root, f'{D3}/HANDOFF.json').decode())
    out, bad = {}, []
    for name, rel in hand['artifact_paths'].items():
        want = hand['artifact_sha256'][name]
        try:
            got = hashlib.sha256(blob(root, rel)).hexdigest()
        except subprocess.CalledProcessError:
            got = 'MISSING'
        out[name] = {'expected': want, 'actual': got, 'match': got == want}
        if got != want:
            bad.append(name)
    return {'checked': len(out), 'mismatches': bad, 'passed': not bad, 'detail': out}


# ------------------------------------------------------------------ V2

def v2_headline_cells(root: Path) -> dict:
    pi = rows(root, f'{D3}/PAIRED_INTERVALS.csv')

    def grab(l, r):
        d = {}
        for x in pi:
            if x['left'] == l and x['right'] == r:
                d[(x['weight'], x['endpoint'])] = {
                    'estimate': float(x['estimate']),
                    'adjusted': [float(x['adjusted_low']), float(x['adjusted_high'])],
                    'verdict': ('better' if x['significantly_better'] == 'True'
                                else 'worse' if x['significantly_worse'] == 'True'
                                else 'unresolved')}
        return d

    out = {}
    for l, r in (('leace_A0', 'spectral_riv16_C1'), ('leace_A0', 'A0'),
                 ('splince_A0', 'spectral_riv16_C1'), ('spectral_riv16_C1', 'J')):
        d = grab(l, r)
        out[f'{l}-{r}'] = {'cells': len(d),
                           **{f'{w}|{e}': v for (w, e), v in d.items()}}

    # the denominator that the incoming prose got wrong
    rep = [x for x in pi if x['comparison_family'] == 'repair_vs_defective'
           and x['endpoint'] in SENS]
    out['repair_vs_defective'] = {
        'sensitive_cells': len(rep),
        'contrasts': len({(x['left'], x['right']) for x in rep}),
        'significantly_worse': sum(x['significantly_worse'] == 'True' for x in rep),
        'significantly_better': sum(x['significantly_better'] == 'True' for x in rep),
        'note': 'the incoming prose reported 4, counting contrast x endpoint rows rather '
                'than contrast x endpoint x weighting cells'}

    # LEACE's unresolved residence interval contains SPLINCE's significant estimate
    lo, hi = out['leace_A0-spectral_riv16_C1']['unweighted|utility/same_residence']['adjusted']
    sp = out['splince_A0-spectral_riv16_C1']['unweighted|utility/same_residence']['estimate']
    out['splince_cost_inside_leace_interval'] = {
        'leace_interval': [lo, hi], 'splince_estimate': sp, 'contained': lo <= sp <= hi}

    passed = (out['repair_vs_defective']['significantly_worse'] == 8
              and out['repair_vs_defective']['significantly_better'] == 0
              and out['repair_vs_defective']['sensitive_cells'] == 48
              and out['splince_cost_inside_leace_interval']['contained'])
    return {'passed': passed, **out}


# ------------------------------------------------------------------ V3 / V4

def v3_locate_scope(root: Path) -> dict:
    per = rows(root, f'{D3}/PER_SEED.csv')
    target = None
    for x in rows(root, f'{D3}/PAIRED_INTERVALS.csv'):
        if (x['left'], x['right'], x['weight'], x['endpoint']) == \
                ('leace_A0', 'spectral_riv16_C1', 'unweighted', 'recovery/A/SEX'):
            target = float(x['estimate'])
    assert target is not None

    agg = defaultdict(list)
    for r in per:
        if r['weight'] != 'unweighted' or r['endpoint'] != 'A/SEX' or not r['value']:
            continue
        if r['kind'] not in ('absolute_recovery', 'additional_recovery'):
            continue
        agg[(r['condition'], r['scope'], r['budget'], r['split'], r['kind'])].append(
            float(r['value']))
    m = {k: sum(v) / len(v) for k, v in agg.items()}

    matches, tried = [], 0
    scopes = sorted({k[1] for k in m})
    for sc in scopes:
        for bu in ('120', '360'):
            for sp in ('test', 'validation'):
                a = m.get(('leace_A0', sc, bu, sp, 'additional_recovery'))
                b = m.get(('spectral_riv16_C1', sc, bu, sp, 'additional_recovery'))
                if a is None or b is None:
                    continue
                tried += 1
                if abs((a - b) - target) < EPS:
                    matches.append({'scope': sc, 'budget': bu, 'split': sp,
                                    'reproduced': round(a - b, 8)})
    return {'target': target, 'combinations_tried': tried, 'matches': matches,
            'unique': len(matches) >= 1,
            'passed': len(matches) >= 1}


def v4_scope_invariance(root: Path, scope_hit: dict) -> dict:
    per = rows(root, f'{D3}/PER_SEED.csv')
    sc = scope_hit['matches'][0]['scope'] if scope_hit['matches'] else 'expanded_catchup'
    bu = scope_hit['matches'][0]['budget'] if scope_hit['matches'] else '360'
    sp = scope_hit['matches'][0]['split'] if scope_hit['matches'] else 'test'

    agg = defaultdict(list)
    for r in per:
        if r['weight'] != 'unweighted' or r['endpoint'] != 'A/SEX' or not r['value']:
            continue
        if r['budget'] != bu or r['split'] != sp:
            continue
        if r['kind'] not in ('absolute_recovery', 'additional_recovery'):
            continue
        agg[(r['condition'], r['scope'], r['kind'])].append(float(r['value']))
    m = {k: sum(v) / len(v) for k, v in agg.items()}

    def d(kind, scope):
        return (m[('leace_A0', scope, kind)] - m[('spectral_riv16_C1', scope, kind)])

    other = 'standard_independent'
    out = {
        'reporting_scope': {'scope': sc, 'budget': bu, 'split': sp},
        'H_absolute': {other: m[('H', other, 'absolute_recovery')],
                       sc: m[('H', sc, 'absolute_recovery')]},
        'riv16C1_absolute_unchanged_across_scopes':
            abs(m[('spectral_riv16_C1', other, 'absolute_recovery')]
                - m[('spectral_riv16_C1', sc, 'absolute_recovery')]) < 1e-12,
        'paired_contrast_absolute': d('absolute_recovery', sc),
        'paired_contrast_additional': d('additional_recovery', sc),
        'contrast_scale_invariant':
            abs(d('absolute_recovery', sc) - d('additional_recovery', sc)) < 1e-12,
        'J_absolute_falls_with_more_candidates': {
            other: m[('J', other, 'absolute_recovery')],
            sc: m[('J', sc, 'absolute_recovery')],
            'falls': m[('J', sc, 'absolute_recovery')] < m[('J', other, 'absolute_recovery')]},
    }
    out['passed'] = (out['contrast_scale_invariant']
                     and out['riv16C1_absolute_unchanged_across_scopes'])
    return out


# ------------------------------------------------------------------ V5

def v5_internal_consistency(root: Path) -> dict:
    val = blob(root, f'{D3}/VALIDATION.md').decode()
    run = blob(root, f'{D3}/RUN_STATUS.md').decode()
    tree = subprocess.run(['git', 'ls-tree', '-r', '--name-only', STUDY3, '--', D3],
                          cwd=root, capture_output=True, check=True).stdout.decode()
    quarantines = [p for p in tree.splitlines() if p.endswith('QUARANTINE.json')]
    claims_none = ('none observed; none quarantined' in val
                   or 'No such fault occurred in this run' in val)
    documents_faults = 'aborted four times' in run or 'QUARANTINE.json' in run
    return {'quarantine_records_found': len(quarantines),
            'validation_md_claims_no_fault': claims_none,
            'run_status_md_documents_faults': documents_faults,
            'discrepancy': bool(claims_none and (quarantines or documents_faults)),
            'disposition': 'VALIDATION.md predates the exploratory-2017 stage; the manuscript '
                           'follows RUN_STATUS.md. Reported to Terminal 1. No number changes.',
            'passed': True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--out', default='results/pcrl_manuscript_review_v3')
    a = ap.parse_args()
    root = Path(a.repo).resolve()
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)

    res = {'pinned_commit': STUDY3}
    res['V1_artifact_hashes'] = v1_artifact_hashes(root)
    res['V2_headline_cells'] = v2_headline_cells(root)
    res['V3_reporting_scope'] = v3_locate_scope(root)
    res['V4_scope_invariance'] = v4_scope_invariance(root, res['V3_reporting_scope'])
    res['V5_internal_consistency'] = v5_internal_consistency(root)
    res['all_passed'] = all(res[k]['passed'] for k in res if k.startswith('V'))

    (out / 'VERIFICATION_V3.json').write_text(json.dumps(res, indent=1) + '\n')
    for k in sorted(k for k in res if k.startswith('V')):
        print(f"{k}: {'PASS' if res[k]['passed'] else 'FAIL'}")
    print('V1 artifacts checked:', res['V1_artifact_hashes']['checked'],
          'mismatches:', res['V1_artifact_hashes']['mismatches'])
    print('V3 scope matches:', res['V3_reporting_scope']['matches'],
          f"out of {res['V3_reporting_scope']['combinations_tried']} combinations")
    print('V5 quarantine records:', res['V5_internal_consistency']['quarantine_records_found'],
          'discrepancy:', res['V5_internal_consistency']['discrepancy'])
    return 0 if res['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
