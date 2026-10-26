#!/usr/bin/env python3
"""RFA1 verifier, wave 2, task 2: the 28 usefulness conditions (G1 at the point), the confidence conjunction inputs and the paired
MODEL_SELECT costs of the saved RF configurations and of the references NEURAL_RAW / H_LJ, recomputed with the verifier's own metric
code (vlib) from the saved release arrays and the MODEL_SELECT labels, compared with runs/FAM/<cfg>/EVAL_MS.json and
runs/REFERENCES/EVAL_MS.json.

Data access: MODEL_SELECT labels through rfa.common.load(..., operation='verify') from THIS __main__ module, inside a queue unit;
releases from runs/FAM/<cfg>/V<v>/RELEASE.npz (file SHA-256 against RELEASE.json and every array's SHA against its wire_sha256);
U0 / Ucal / NEURAL_RAW / H_LJ from the admitted frozen bank (file SHA-256 against both the source admission record and the
frozen-hybrids manifest; MODEL_SELECT row order against the export).

    python3 -m rfa.jobqueue run --unit RFA_VER_USEFULNESS_v1 --stage verify --rss-gib 2 -- \\
        <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v2_usefulness.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vlib as V  # noqa: E402

OUT = HERE.parent / 'out' / 'V2_USEFULNESS.json'
CAMPAIGN = 'pcrl_reader_fit_aware_release_v1'
RUNS = ROOT / '.private' / CAMPAIGN / 'runs'
CONFIGS = ('RF_FULL', 'RF_TASK', 'RF_STOP_LO', 'RF_STOP_HI', 'RF_ALTC_HI', 'RF_RAWJ_HI')
REFERENCES = ('NEURAL_RAW', 'H_LJ')
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_ms():
    from rfa import common as C
    S = C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify',
               'verify: independent recomputation of the 28 usefulness conditions, the confidence conjunction inputs and the paired '
               'MODEL_SELECT costs of the RFA_v1 configurations and of NEURAL_RAW / H_LJ (verifier wave 2)')
    S.pop('X', None)
    return S


def own_release(config, version):
    d = RUNS / 'FAM' / config / f'V{int(version)}'
    rec = json.loads((d / 'RELEASE.json').read_text())
    file_ok = sha256(d / 'RELEASE.npz') == rec['release_sha256']
    if not file_ok:
        raise PermissionError(f'{config}/V{version}: RELEASE.npz differs from its record')
    arrays_ok, out = True, {}
    with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
        for role in ROLES6:
            for t in V.TASKS:
                a = z[f'{role}__{t}']
                arrays_ok &= V.arr_sha(a) == rec['wire_sha256'][f'{role}__{t}']
                if role == 'MODEL_SELECT':
                    out[t] = np.asarray(a, dtype=np.float64).copy()
    return out, {'release_sha256': rec['release_sha256'], 'file_hash_ok': file_ok, 'all_12_arrays_hash_ok': bool(arrays_ok),
                 'kind': rec['kind'], 'checkpoint': rec['checkpoint']}


def own_bank(version, row_id):
    from ohn import common as OC
    from scm_bench import common as SB
    rel = f'admitted/bank/FROZEN_BANK_S{int(version)}.npz'
    p = OC.MIRROR / rel
    got = sha256(p)
    adm = OC.admission()['copies'][rel]['sha256']
    man = json.loads((SB.HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())['bank_files_sha256'][f'S{int(version)}']
    if not (got == adm == man):
        raise PermissionError(f'frozen bank S{version}: {got} vs admission {adm} vs manifest {man}')
    out = {}
    with np.load(p, allow_pickle=False) as z:
        if not np.array_equal(z['row_id__MODEL_SELECT'], row_id):
            raise PermissionError('frozen bank MODEL_SELECT row order differs from the export')
        for name in ('U0', 'Ucal') + REFERENCES:
            out[name] = {t: np.asarray(z[f'{name}__MODEL_SELECT__{t}'], dtype=np.float64).copy() for t in V.TASKS}
    return out, {'bank_sha256': got, 'admission_and_manifest_agree': True}


def confidence(u):
    """RULES 2: per condition PASS iff value + 2 se <= cap; VIOLATION iff value - 2 se > cap; UNRESOLVED otherwise."""
    lab = {}
    for part in ('clauses', 'guards'):
        for k, c in u[part].items():
            se = (c['se'] ** 2 + c.get('se_mc', 0.0) ** 2) ** 0.5
            lab[f'{part}|{k}'] = 'PASS' if c['value'] + 2 * se <= c['cap'] else 'VIOLATION' if c['value'] - 2 * se > c['cap'] else 'UNRESOLVED'
    return lab


def per_person_losses(labels, rel):
    out = {}
    for t in V.TASKS:
        pp = V.per_person(labels[t], rel[t], V.K[t])
        out[t] = {'acc_loss': 1.0 - pp['accuracy'], 'nll': pp['nll'], 'brier': pp['brier']}
    return out


def paired_costs(a, b, sex, unit):
    out = {}
    for t in V.TASKS:
        for m in ('acc_loss', 'nll', 'brier'):
            d = a[t][m] - b[t][m]
            out[f'{t}|{m}'] = {'value': float(d.mean()), 'se': V.cluster_se(d, unit)}
            if m == 'brier':
                for g, name in ((0, 'men_brier'), (1, 'women_brier')):
                    mask = np.asarray(sex) == g
                    out[f'{t}|{name}'] = {'value': float(d[mask].mean()), 'se': V.subgroup_se(d, mask, unit)}
    tot = (a['income']['nll'] + a['occupation']['nll']) - (b['income']['nll'] + b['occupation']['nll'])
    out['total|nll'] = {'value': float(tot.mean()), 'se': V.cluster_se(tot, unit)}
    return out


def main():
    if not os.environ.get('RFA_UNIT_ID'):
        sys.exit('refused: labels are read only inside an rfa.jobqueue verify unit')
    t0 = time.process_time()
    S = load_ms()
    labels = {t: S[t] for t in V.TASKS}
    sex, unit, row_id = S['sex'], S['unit'], S['row_id']
    res = {'schema': 'RFA1-verifier-usefulness-v2', 'unit_id': os.environ.get('RFA_UNIT_ID'), 'people': int(len(row_id)),
           'households': int(len(np.unique(unit))), 'checks': {}, 'configs': {}, 'references': {}, 'provenance': {}}
    chk = res['checks']
    bank, rels = {}, {c: {} for c in CONFIGS}
    for v in (0, 1):
        bank[v], res['provenance'][f'bank_S{v}'] = own_bank(v, row_id)
        for c in CONFIGS:
            rels[c][v], res['provenance'][f'{c}_V{v}'] = own_release(c, v)
    chk['release_files_and_all_arrays_hash_checked'] = all(p.get('all_12_arrays_hash_ok', True) and p.get('file_hash_ok', True)
                                                           for p in res['provenance'].values())
    refs = {v: {'U0': bank[v]['U0'], 'Ucal': bank[v]['Ucal']} for v in (0, 1)}
    # usefulness of the configurations and the references
    def evaluate(name, by_v, campaign_util):
        per_v = {v: V.usefulness_version(labels, by_v[v], refs[v], sex, unit) for v in (0, 1)}
        g = V.g1(per_v)
        cmpv = {str(v): V.compare_with_campaign_utility(per_v[v], campaign_util[v]['utility']) for v in (0, 1)}
        camp_pt = {str(v): campaign_util[v].get('point', {}).get('point_pass_12_plus_2') for v in (0, 1)}
        rec = {'G1': g, 'per_version': {str(v): {'all_pass': pv['all_pass'], 'failed': pv['failed'],
                                                 'max_value_over_cap': max(max(c['value'] / c['cap'] for c in pv['clauses'].values()),
                                                                           max(x['value'] / x['cap'] for x in pv['guards'].values())),
                                                 'clauses': pv['clauses'], 'guards': pv['guards'], 'confidence': confidence(pv)}
                                        for v, pv in per_v.items()},
               'vs_campaign': cmpv, 'campaign_point_pass_12_plus_2': camp_pt,
               'G1_agrees_with_campaign_point': g['G1'] == all(bool(x) for x in camp_pt.values())}
        return rec
    ev_fam = {c: json.loads((RUNS / 'FAM' / c / 'EVAL_MS.json').read_text()) for c in CONFIGS}
    for c in CONFIGS:
        r = evaluate(c, rels[c], {v: ev_fam[c]['versions'][str(v)] for v in (0, 1)})
        res['configs'][c] = r
        chk[f'{c}|28_conditions_match_EVAL_MS (abs <= 1e-12, flags)'] = all(x['agree'] for x in r['vs_campaign'].values())
        chk[f'{c}|G1_agrees_with_campaign'] = r['G1_agrees_with_campaign_point']
    ev_ref = json.loads((RUNS / 'REFERENCES' / 'EVAL_MS.json').read_text())
    for name in REFERENCES:
        rr = ev_ref['references'][f'REF:{name}']
        r = evaluate(name, {v: bank[v][name] for v in (0, 1)}, {v: rr[str(v)] for v in (0, 1)})
        res['references'][name] = r
        chk[f'REF:{name}|28_conditions_match_REFERENCES_EVAL_MS (abs <= 1e-12, flags)'] = all(x['agree'] for x in r['vs_campaign'].values())
        chk[f'REF:{name}|G1_agrees_with_campaign'] = r['G1_agrees_with_campaign_point']
    # income identity: the occupation-only configurations release START's income vector
    inc = {}
    for v in (0, 1):
        base = rels['RF_FULL'][v]['income']
        inc[str(v)] = {c: bool(np.array_equal(rels[c][v]['income'], base)) for c in CONFIGS}
        inc[str(v)]['max_abs_RF_FULL_income_minus_NEURAL_RAW_bank'] = float(np.max(np.abs(base - bank[v]['NEURAL_RAW']['income'])))
        inc[str(v)]['RF_FULL_occupation_differs_from_RF_TASK'] = not np.array_equal(rels['RF_FULL'][v]['occupation'], rels['RF_TASK'][v]['occupation'])
        inc[str(v)]['max_abs_RF_FULL_minus_RF_STOP_HI_occupation'] = float(np.max(np.abs(rels['RF_FULL'][v]['occupation'] - rels['RF_STOP_HI'][v]['occupation'])))
    res['income_identity'] = inc
    chk['occupation_only_configs_release_identical_income'] = all(all(inc[str(v)][c] for c in CONFIGS if c != 'RF_RAWJ_HI') for v in (0, 1))
    chk['RF_FULL_income_equals_NEURAL_RAW_bank_income (abs < 1e-12)'] = all(inc[str(v)]['max_abs_RF_FULL_income_minus_NEURAL_RAW_bank'] < 1e-12 for v in (0, 1))
    # paired costs (the frontier / Q* input): every comparator in each configuration's EVAL_MS
    losses = {(c, v): per_person_losses(labels, rels[c][v]) for c in CONFIGS for v in (0, 1)}
    for nm in ('NEURAL_RAW', 'H_LJ'):
        for v in (0, 1):
            losses[(f'REF:{nm}', v)] = per_person_losses(labels, bank[v][nm])
    pc, worst = {}, 0.0
    for c in CONFIGS:
        for v in (0, 1):
            for spec, rec in ev_fam[c]['versions'][str(v)]['paired_costs'].items():
                if rec.get('missing'):
                    continue
                key = (spec.split('/')[1], v) if spec.startswith('FAM/') else (spec, v)
                if key not in losses:
                    pc[f'{c}|V{v}|{spec}'] = {'skipped': 'comparator not loaded by the verifier'}
                    continue
                mine = paired_costs(losses[(c, v)], losses[key], sex, unit)
                d = max(max(abs(mine[m]['value'] - rec['costs'][m]['value']), abs(mine[m]['se'] - rec['costs'][m]['se'])) for m in mine)
                worst = max(worst, d)
                pc[f'{c}|V{v}|{spec}'] = {'max_abs_diff': d, 'total_nll': mine['total|nll'], 'recorded_total_nll': rec['costs']['total|nll']['value']}
    res['paired_costs'] = pc
    res['paired_costs_worst_abs_diff'] = worst
    chk['paired_costs_match_EVAL_MS (abs <= 1e-12)'] = worst <= 1e-12 and not any('skipped' in x for x in pc.values())
    res['all_pass'] = all(bool(x) for x in chk.values())
    res['cpu_s'] = time.process_time() - t0
    res['versions'] = {'numpy': np.__version__, 'python': sys.version.split()[0]}
    res['code_sha256'] = {f: sha256(HERE / f) for f in ('v2_usefulness.py', 'vlib.py')}
    if OUT.exists():
        raise PermissionError(f'{OUT.name} exists (write-once)')
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'failed': [k for k, x in chk.items() if not x], 'cpu_s': res['cpu_s'],
                      'G1': {k: r['G1']['G1'] for k, r in list(res['configs'].items()) + list(res['references'].items())}}, indent=1))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
