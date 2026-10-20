"""V3 actual transmitted outputs and training / calibration exposure of saved releases (label-free: release files, records, the
configuration ledger, the access log, the queue ledger and git; no label is read).

Per (family, config, version):
 outputs   all six roles x two tasks are float64 probability vectors of the admitted row counts ((R, n, k) stacks for randomized
           releases with R = record draws): finite, non-negative, rows sum to 1 (max deviation reported), decisions = first argmax
           (tied maxima counted), per-key array hashes equal RELEASE.json wire_sha256, file hash equals release_sha256, distinct
           vectors per task and distinct linked (income, occupation) cells on MODEL_SELECT (finite-cell cardinalities);
 record    required fields, kind, randomized fields, card committed before the release (bytes at the card commit = logged SHA-256,
           unchanged now), configuration ledger entry (same hash, number <= 60);
 exposure  the saving unit's label reads (access log): no MODEL_SELECT / closed-role label read by a fitting unit, fit / calibrate
           reads carry the release's card, every role actually read appears in roles_read and vice versa, training sizes stated.

    python3 .../v3_outputs.py --tag V1 [--only FAMILY/CONFIG ...]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
REQUIRED = ('mechanism_card', 'kind', 'description', 'n_model_fits', 'n_calibrator_fits', 'n_crossfit_fits', 'n_critic_fits',
            'roles_read', 'exposure', 'training_people', 'training_households', 'fit_cpu_s', 'randomized')
KINDS = ('twin', 'candidate', 'control', 'ablation', 'perturbation', 'reference_adaptation')
FULL_POOL = (20000, 13337)


def role_sizes():
    sys.path.insert(0, str(V.ROOT))
    import nrm.common as C
    return {r: int(len(C.features(r)['row_id'])) for r in ROLES6}


def ties(p):
    p = np.asarray(p, dtype=np.float64)
    mx = p.max(axis=-1, keepdims=True)
    return int(((p == mx).sum(axis=-1) > 1).sum())


def output_checks(arrs, rec, sizes):
    probs, info = [], {}
    randomized = bool(rec.get('randomized'))
    R = int(rec.get('draws', 1)) if randomized else None
    for role in ROLES6:
        for t in V.TASKS:
            key = f'{role}__{t}'
            a = arrs.get(key)
            if a is None:
                probs.append(f'{key} missing'); continue
            if a.dtype != np.float64:
                probs.append(f'{key} dtype {a.dtype}')
            want = (R, sizes[role], V.K[t]) if randomized else (sizes[role], V.K[t])
            if a.shape != want:
                probs.append(f'{key} shape {a.shape} expected {want}'); continue
            flat = a.reshape(-1, V.K[t])
            p = V.check_prob(flat, V.K[t])
            if p:
                probs.append(f'{key}: {p}')
            dev = float(np.abs(flat.sum(1) - 1.0).max())
            rh = rec.get('wire_sha256', {}).get(key)
            if rh != V.arr_sha(a):
                probs.append(f'{key} array hash differs from RELEASE.json')
            info[key] = {'max_row_sum_dev': dev, 'tied_argmax_rows': ties(flat), 'min_prob': float(flat.min())}
    extra = sorted(set(arrs) - {f'{r}__{t}' for r in ROLES6 for t in V.TASKS})
    if extra:
        probs.append(f'unexpected keys {extra}')
    ms = {t: arrs.get(f'MODEL_SELECT__{t}') for t in V.TASKS}
    card = {}
    if all(x is not None for x in ms.values()):
        draws = [{t: (ms[t][r] if randomized else ms[t]) for t in V.TASKS} for r in range(R if randomized else 1)]
        for i, d in enumerate(draws[:3]):
            pair = np.concatenate([d['income'], d['occupation']], axis=1)
            card[f'draw{i}'] = {'income': V.distinct_rows(d['income']), 'occupation': V.distinct_rows(d['occupation']),
                                'pair': V.distinct_rows(pair),
                                'wire_sha256': {t: V.wire_sha(d[t]) for t in V.TASKS}}
            if i == 0:
                occ = {}
                for nm, arr in (('income', d['income']), ('occupation', d['occupation']), ('pair', pair)):
                    keys = {}
                    for row in np.ascontiguousarray(arr):
                        k = row.tobytes(); keys[k] = keys.get(k, 0) + 1
                    sizes = np.array(sorted(keys.values()))
                    if len(sizes) <= 2000:                 # finite release: people per released vector / joint value
                        occ[nm] = {'n_values': int(len(sizes)), 'min_people': int(sizes.min()), 'median_people': float(np.median(sizes)),
                                   'max_people': int(sizes.max()), 'singletons': int((sizes == 1).sum())}
                card['people_per_value_draw0'] = occ
    return probs, info, card


def record_checks(rec, d, ledger):
    probs = []
    miss = [k for k in REQUIRED if k not in rec]
    if miss:
        probs.append(f'missing record fields {miss}')
    if rec.get('kind') not in KINDS:
        probs.append(f'kind {rec.get("kind")!r}')
    if rec.get('randomized') and not all(k in rec for k in ('draws', 'seed_handling', 'noise_coupling')):
        probs.append('randomized record lacks draws / seed_handling / noise_coupling')
    h = V.file_sha(d / 'RELEASE.npz')
    if h != rec.get('release_sha256'):
        probs.append('RELEASE.npz hash differs from release_sha256')
    card = rec.get('card') or rec.get('mechanism_card')
    cinfo = {}
    if card:
        cp = V.ROOT / card
        commit = rec.get('card_commit')
        cinfo = {'card': card, 'card_commit': commit, 'card_sha256': rec.get('card_sha256'),
                 'sha_at_commit': V.git_blob_sha256(commit, card) if commit else None,
                 'sha_now': V.file_sha(cp) if cp.is_file() else None,
                 'commit_utc': (V.commit_time(commit).isoformat() if commit and V.commit_time(commit) else None),
                 'ancestor': V.is_ancestor(commit) if commit else False}
        if cinfo['sha_at_commit'] != rec.get('card_sha256'):
            probs.append('card bytes at the card commit differ from card_sha256')
        if cinfo['sha_now'] != rec.get('card_sha256'):
            probs.append('card changed after the release')
        if not cinfo['ancestor']:
            probs.append('card commit is not an ancestor of HEAD')
        if cinfo['commit_utc'] is None or V.parse_utc(cinfo['commit_utc']) > V.parse_utc(rec['utc']):
            probs.append('card committed after the release was saved')
        if not str(card).startswith(f'results/{V.CAMPAIGN}/cards/'):
            probs.append('card outside cards/')
    else:
        probs.append('no card in the record')
    led = [e for e in ledger if (e['family'], e['config_id'], int(e['version'])) == (rec.get('family'), rec.get('config_id'), int(rec.get('version', -1)))]
    if len(led) != 1:
        probs.append(f'{len(led)} configuration-ledger entries')
    else:
        e = led[0]
        if e['release_sha256'] != h or e.get('card_sha256') != rec.get('card_sha256') or int(e['configuration_number']) > 60:
            probs.append('configuration-ledger entry differs (hash, card or number > 60)')
    return probs, cinfo


def exposure_checks(rec, events, units):
    probs, att = [], []
    uid = rec.get('unit_id')
    mine = [e for e in events if uid is not None and e.get('unit_id') == uid]
    read = {}
    for e in mine:
        if e.get('fields'):
            read.setdefault(e['role'], set()).update(e['fields'])
    if uid is None:
        att.append('release saved outside the queue (null unit id): its fitting reads cannot be attributed by unit')
    elif uid not in units:
        probs.append(f'unit {uid} not in the queue ledger')
    for e in mine:
        role, op = e.get('role'), e.get('operation')
        if role in ('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT') and e.get('fields'):
            probs.append(f'saving unit {uid} read {role} labels ({op}, {e.get("entry_point")})')
        if op in ('fit', 'calibrate') and e.get('card') != rec.get('card'):
            att.append(f'unit {uid} read {role} under card {e.get("card")} while the release names {rec.get("card")}')
    rr = str(rec.get('roles_read', ''))
    stated = set(re.findall(r'(MODEL_FIT|CALIBRATION|MODEL_SELECT|ATTACK_FIT|ATTACK_VALIDATION|DEV_REPORT)', rr))
    if mine:
        actual = set(read)
        if actual - stated:
            probs.append(f'roles read but not stated in roles_read: {sorted(actual - stated)}')
        if stated - actual:
            att.append(f'roles stated but not read in the saving unit (fit may have read them in another unit): {sorted(stated - actual)}')
        if 'sex' in read.get('MODEL_FIT', set()) and rec.get('kind') == 'twin':
            att.append('a task-only twin unit read MODEL_FIT SEX (allowed only if the twin itself never uses it; see card)')
        if 'sex' not in read.get('MODEL_FIT', set()) and 'sex' in rr.lower() and rec.get('kind') != 'twin':
            att.append('roles_read mentions SEX but the saving unit did not read MODEL_FIT SEX')
    elif uid is not None:
        att.append(f'no label read logged for the saving unit {uid} (fit in another unit or label-free derivation)')
    tp, th = rec.get('training_people'), rec.get('training_households')
    sizes = {'training_people': tp, 'training_households': th, 'full_pool': (tp, th) == FULL_POOL}
    if not sizes['full_pool']:
        att.append(f'training size {tp} / {th} differs from the full MODEL_FIT pool {FULL_POOL} (stated in the record)')
    return probs, att, {'unit_id': uid, 'labels_read_by_saving_unit': {k: sorted(v) for k, v in read.items()},
                        'roles_read_stated': rr, 'sizes': sizes}


def releases(only=None):
    out = []
    for e in V.jsonl(V.CONFIG_LEDGER):
        name = f"{e['family']}/{e['config_id']}"
        if only and name not in only:
            continue
        out.append((e['family'], e['config_id'], int(e['version'])))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--only', action='append', default=[])
    a = ap.parse_args()
    sizes = role_sizes()
    ledger = V.jsonl(V.CONFIG_LEDGER)
    events = V.jsonl(V.ACCESS_LOG)
    units = {e['unit_id'] for e in V.jsonl(V.QUEUE_LEDGER) if e.get('event') == 'unit_start'}
    rels = releases(a.only or None)
    n_conf = len({(e['family'], e['config_id'], int(e['version'])) for e in ledger})
    checks, record = [], {'tag': a.tag, 'utc': V.utc(), 'n_configurations_in_ledger': n_conf, 'releases': {}}
    for fam, cfg, v in rels:
        d = V.RUNS / fam / cfg / f'V{v}'
        rec = json.loads((d / 'RELEASE.json').read_text())
        with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
            arrs = {k: z[k].copy() for k in z.files}
        p_out, info, card = output_checks(arrs, rec, sizes)
        p_rec, cinfo = record_checks(rec, d, ledger)
        p_exp, a_exp, einfo = exposure_checks(rec, events, units)
        name = f'{fam}/{cfg}/V{v}'
        record['releases'][name] = {'kind': rec.get('kind'), 'randomized': bool(rec.get('randomized')), 'draws': rec.get('draws'),
                                    'outputs': info, 'cardinality_MODEL_SELECT': card, 'card': cinfo, 'exposure': einfo,
                                    'problems': {'outputs': p_out, 'record': p_rec, 'exposure': p_exp}, 'attention': a_exp}
        checks.append({'id': f'V3.OUTPUTS.{fam}.{cfg}.V{v}__{a.tag}',
                       'what': f'transmitted outputs of {name}: probability vectors on all six roles, decisions, array hashes, distinct '
                               'vectors and linked cells (MODEL_SELECT)',
                       'inputs': {V.rel(d / 'RELEASE.npz'): V.file_sha(d / 'RELEASE.npz'), V.rel(d / 'RELEASE.json'): V.file_sha(d / 'RELEASE.json')},
                       'result': {'problems': p_out, 'cardinality_MODEL_SELECT': card,
                                  'max_row_sum_dev': max(x['max_row_sum_dev'] for x in info.values()) if info else None,
                                  'tied_argmax_rows_MODEL_SELECT': {t: info.get(f'MODEL_SELECT__{t}', {}).get('tied_argmax_rows') for t in V.TASKS}},
                       'numerical_status': 'PASS' if not p_out else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'})
        checks.append({'id': f'V3.EXPOSURE.{fam}.{cfg}.V{v}__{a.tag}',
                       'what': f'record, card order, configuration ledger and training / calibration exposure of {name} against the access '
                               'and queue logs',
                       'inputs': {**V.inputs_sha([d / 'RELEASE.json', V.CONFIG_LEDGER, V.ACCESS_LOG]), 'card': cinfo},
                       'result': {'record_problems': p_rec, 'exposure_problems': p_exp, 'attention': a_exp, 'exposure': einfo},
                       'numerical_status': 'NOT_APPLICABLE',
                       'procedural_status': 'FAIL' if (p_rec or p_exp) else ('ATTENTION' if a_exp else 'PASS')})
    nums = {}
    for e in ledger:
        nums.setdefault(int(e['configuration_number']), []).append(f"{e['family']}/{e['config_id']}/V{e['version']}")
    dup = {n: x for n, x in nums.items() if len(x) > 1}
    budget_ok = n_conf <= 60 and all((V.RUNS / e['family'] / e['config_id'] / f"V{e['version']}" / 'RELEASE.npz').is_file() for e in ledger)
    checks.append({'id': f'V3.CONFIG_BUDGET__{a.tag}', 'what': 'configuration ledger: distinct (family, config, version) <= 60, every entry '
                   'has its release directory; duplicate configuration numbers (concurrent saves; the budget check is not atomic) '
                   'are reported', 'inputs': V.inputs_sha([V.CONFIG_LEDGER]),
                   'result': {'n_configurations': n_conf, 'n_ledger_lines': len(ledger), 'duplicate_configuration_numbers': dup,
                              'missing_dirs': [f"{e['family']}/{e['config_id']}/V{e['version']}" for e in ledger
                                               if not (V.RUNS / e['family'] / e['config_id'] / f"V{e['version']}" / 'RELEASE.npz').is_file()]},
                   'numerical_status': 'NOT_APPLICABLE',
                   'procedural_status': ('ATTENTION' if dup else 'PASS') if budget_ok else 'FAIL'})
    V.write_record(f'V3_OUTPUTS__{a.tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], c['procedural_status'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
