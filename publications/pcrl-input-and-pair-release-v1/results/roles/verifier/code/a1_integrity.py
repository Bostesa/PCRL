"""Addendum checks A1 (integrity), A2 (selection and brackets) and A8 (code drift); label-free.

A1. For every configuration saved since the first verifier report (configuration numbers >= 24): RELEASE.npz SHA-256 = RELEASE.json =
ledger; MODEL_FIT / CALIBRATION / MODEL_SELECT wire hashes = own array hashes; MODEL.pkl = MODEL.json = RELEASE.json model hash; the
card blob at its recorded commit has the recorded SHA-256 and the commit precedes the save and the first fitting read with that card
(access log). The deployment function recomputes the MODEL_SELECT arrays bitwise for every saved configuration (all 39, so that the
registered releases are re-checked under the current code). The income output of every IA_OSG* configuration equals CTL_TWIN's in
every role: decoded arrays for MODEL_FIT / CALIBRATION / MODEL_SELECT; for ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT the stored
'<ROLE>__income.npy' members are compared as raw bytes (SHA-256), never decoded.
A2. From ia/FIT_R1_V<v>.json (label-free arithmetic): k* by the scan rule under S3; the SBC / SPEND / RO bracket points by the
PROTOCOL 4 rule; the BLIND rung; the fineness classes; the consistency record (own comparison of the recomputed single-step occupation
dT and dI with ia/FIT_V<v>.json); the OSG, BLIND and registered SFG ladders re-derived from the recorded single-step statistics with an
own implementation of the greedy rule; the saved rungs' steps against the ladders.
A8. The queue ledger since 03:11Z: code_changed_during_run of every unit, launch-time drift records, and the campaign code files whose
hash changed between consecutive unit launches.

    python3 -m ipr.jobqueue run --unit IPR_VER_A1_INTEGRITY_V1 --stage verify --rss-gib 2 -- python3 <this file>
"""
from __future__ import annotations
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vlib as V  # noqa: E402
import a_lib as A  # noqa: E402

OPEN_ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT')
CLOSED_ROLES = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
FIRST_NEW = 24


def git(*a):
    r = subprocess.run(['git', '-C', str(V.ROOT), *a], capture_output=True)
    return r.returncode, r.stdout


def bitwise(a, b):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64)); b = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    return bool(a.shape == b.shape and a.tobytes() == b.tobytes())


def member_bytes_sha(npz, key):
    with zipfile.ZipFile(npz) as z:
        return hashlib.sha256(z.read(f'{key}.npy')).hexdigest()


def a1(out, F, names):
    import ipr.ia as IA
    import ipr.ib as IB
    import ipr.ctl as CTL
    PRED = {'IA': IA.predict, 'IB': IB.predict, 'CTL': CTL.predict}
    fam_of, rows = A.ledger_family()
    acc = [json.loads(l) for l in V.ACCESS_LOG.read_text().splitlines() if l.strip()]
    first_fit = {}
    for e in acc:
        if e.get('operation') in ('fit', 'calibrate') and e.get('card'):
            t = V.parse_utc(e['utc'])
            first_fit[e['card']] = min(first_fit.get(e['card'], t), t)
    card_blob, recs = {}, []
    twin_ms = {}
    for r in rows:
        fam, cfg, v = r['family'], r['config_id'], int(r['version'])
        new = int(r['configuration_number']) >= FIRST_NEW
        rec = {'family': fam, 'config': cfg, 'version': v, 'configuration_number': r['configuration_number'], 'new': new}
        try:
            d = V.RUNS / fam / cfg / f'V{v}'
            R = json.loads((d / 'RELEASE.json').read_text())
            arrays = {}
            with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
                keys = list(z.files)
                for k in keys:
                    if k.split('__')[0] in (OPEN_ROLES if new else ('MODEL_SELECT',)):
                        arrays[k] = z[k].copy()
            if new:
                fsha = V.file_sha(d / 'RELEASE.npz')
                rec['npz_sha_eq_json'] = fsha == R['release_sha256']
                rec['npz_sha_eq_ledger'] = fsha == r['release_sha256']
                wires = R.get('wire_sha256', {})
                rec['wire_keys_match'] = sorted(wires) == sorted(keys)
                rec['opened_wire_hashes_match'] = all(V.arr_sha(arrays[k]) == wires.get(k) for k in arrays)
                meta = json.loads((d / 'MODEL.json').read_text())
                msha = V.file_sha(d / 'MODEL.pkl')
                rec['model_sha_eq_meta'] = msha == meta['model_sha256']
                rec['model_sha_eq_release_json'] = msha == R.get('model_sha256')
                key = (R['card'], R['card_commit'])
                if key not in card_blob:
                    rc, blob = git('show', f"{R['card_commit']}:{R['card']}")
                    card_blob[key] = hashlib.sha256(blob).hexdigest() if rc == 0 else None
                rec['card'] = R['card']
                rec['card_commit'] = R['card_commit']
                rec['card_sha_eq_commit_blob'] = card_blob[key] == R['card_sha256'] == r['card_sha256']
                rec['card_sha_eq_current_file'] = V.file_sha(V.ROOT / R['card']) == R['card_sha256']
                cu = V.parse_utc(R['card_commit_utc'])
                rec['card_commit_utc'] = str(cu)
                rec['save_utc'] = R['utc']
                rec['card_commit_before_save'] = cu <= V.parse_utc(R['utc'])
                ff = first_fit.get(R['card'])
                rec['first_fitting_read_with_card'] = str(ff) if ff else None
                rec['card_commit_before_first_fitting_read'] = bool(ff and cu <= ff)
                P = pickle.loads((d / 'MODEL.pkl').read_bytes())
                rec['payload'] = {k: (P[k] if k in ('op', 'outputs', 'delta', 'alpha', 'dq_tag') else None) for k in
                                  ('op', 'outputs', 'delta', 'alpha', 'dq_tag') if k in P}
                if P.get('op') == 'osg':
                    rec['payload']['steps_occ'] = list(P['steps_occ'])
                rec['record_strength'] = R.get('strength')
                rec['kind'] = R.get('kind')
            got = PRED[fam](cfg, v, F['X'], names)
            eq = {t: bitwise(got[t], arrays[f'MODEL_SELECT__{t}']) for t in V.TASKS}
            rec['deploy_ms_bitwise'] = eq
            rec['deploy_all_bitwise'] = all(eq.values())
            rec['ms_wire_sha'] = {t: V.wire_sha(arrays[f'MODEL_SELECT__{t}']) for t in V.TASKS}
            if cfg == 'CTL_TWIN':
                twin_ms[v] = {'npz': d / 'RELEASE.npz', 'arrays': arrays}
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'
            out['errors'].append({'config': f'{fam}/{cfg}/V{v}', 'traceback': traceback.format_exc()[-2000:]})
        recs.append(rec)
    # income of every IA_OSG* configuration = CTL_TWIN's, every role
    tw_files = {v: V.RUNS / 'CTL' / 'CTL_TWIN' / f'V{v}' / 'RELEASE.npz' for v in (0, 1)}
    tw_open = {}
    for v in (0, 1):
        with np.load(tw_files[v], allow_pickle=False) as z:
            tw_open[v] = {r: z[f'{r}__income'].copy() for r in OPEN_ROLES}
    inc = {}
    for r in rows:
        if not r['config_id'].startswith('IA_OSG'):
            continue
        v = int(r['version'])
        npz = V.RUNS / 'IA' / r['config_id'] / f'V{v}' / 'RELEASE.npz'
        e = {}
        with np.load(npz, allow_pickle=False) as z:
            for role in OPEN_ROLES:
                e[role] = bitwise(z[f'{role}__income'], tw_open[v][role])
        for role in CLOSED_ROLES:
            e[role + '_raw_bytes'] = member_bytes_sha(npz, f'{role}__income') == member_bytes_sha(tw_files[v], f'{role}__income')
        e['all'] = all(e.values())
        inc[f"{r['config_id']}/V{v}"] = e
    out['A1'] = {'configs': recs, 'income_equals_twin': inc,
                 'n_new': sum(1 for x in recs if x.get('new')),
                 'budget': {'ledger_rows': len(rows), 'distinct': len({(r['family'], r['config_id'], r['version']) for r in rows}),
                            'limit': 48}}
    al = {}
    for v in (0, 1):
        by = {}
        for c in recs:
            if c['version'] == v and 'ms_wire_sha' in c:
                by.setdefault((c['ms_wire_sha']['income'], c['ms_wire_sha']['occupation']), []).append(f"{c['family']}/{c['config']}")
        al[f'V{v}'] = [sorted(x) for x in by.values() if len(x) > 1]
    out['A1']['ms_exact_aliases_pair'] = al


def a2(out):
    res = {}
    for v in (0, 1):
        d = json.loads((V.RESULTS / 'ia' / f'FIT_R1_V{v}.json').read_text())
        reg = json.loads((V.RESULTS / 'ia' / f'FIT_V{v}.json').read_text())
        s = d['selection']
        e = {}
        k, ff, ratio = A.scan_select(s['path'], s['budget'])
        e['k_star'] = {'recomputed': None if k is None else int(k), 'recorded': s['k_star'],
                       'match': k is not None and int(k) == int(s['k_star']), 'first_failing_rung': ff, 'first_failing_ratio': ratio,
                       'income_components_zero': all(s['path'][g][m]['estimate'] == 0.0 and s['path'][g][m]['se'] == 0.0
                                                     for g in s['path'] for m in ('inc_nll', 'inc_acc', 'inc_brier'))}
        br = {}
        for key, b in (('SPEND', d['spend_path_bracket']), ('RO', d['ro_path_bracket']), ('SBC', d['sbc_tries'][-1]['bracket'])):
            lo, hi, flag = A.bracket(b)
            rec_lo = None if b['lo'] is None else float(b['lo'])
            br[key] = {'recomputed': [lo, hi, flag], 'recorded': [rec_lo, float(b['hi']), b['flag']],
                       'match': [lo, hi, flag] == [rec_lo, float(b['hi']), b['flag']]}
        e['brackets'] = br
        e['sbc_tries'] = [t['target'] for t in d['sbc_tries']]
        bl = d['blind']
        kb = None
        for g in sorted(bl['scan'], key=int):
            if bl['scan'][g] >= d['cand_spend']:
                kb = int(g)
                break
        e['blind'] = {'recomputed_k': kb, 'recorded_k': bl['k'], 'match': kb == bl['k'], 'scan': bl['scan'], 'cand_spend': d['cand_spend']}
        e['fineness_own'] = {k_: A.fineness(fi) for k_, fi in d['fineness_inputs'].items()}
        ts, tr = s['single_step_stats'], reg['selection']['single_step_stats']
        e['consistency'] = {'own_max_abs_diff_dT_occ': max(abs(ts[x]['dT_occ'] - tr[x]['dT_occ']) for x in ts),
                            'own_max_abs_diff_dI': max(abs(ts[x]['dI'] - tr[x]['dI']) for x in ts),
                            'steps_compared': len(ts), 'recorded': s.get('registered_stats_consistency')}
        osg_ladder = A.build_ladder(ts, 'target', 'dT')
        blind_ladder = A.build_ladder(ts, 'blind', 'dT')
        sfg_ladder = A.build_ladder(tr, 'target', 'dT')
        sfg_blind = A.build_ladder(tr, 'blind', 'dT')
        e['ladders'] = {'osg': {'recomputed': osg_ladder, 'recorded': s['ladder'], 'match': osg_ladder == s['ladder']},
                        'osg_blind': {'recomputed': blind_ladder, 'recorded': bl['ladder'], 'match': blind_ladder == bl['ladder']},
                        'sfg_registered': {'recomputed': sfg_ladder, 'recorded': reg['selection']['ladder'],
                                           'match': sfg_ladder == reg['selection']['ladder']},
                        'sfg_blind_registered': {'recomputed': sfg_blind, 'recorded': reg['blind']['ladder'],
                                                 'match': sfg_blind == reg['blind']['ladder']},
                        'osg_dT_is_dT_occ': all(ts[x]['dT'] == ts[x]['dT_occ'] for x in ts)}
        steps = {}
        for g, st in s['steps'].items():
            steps[g] = {'recorded': st, 'own_compose': A.compose(s['ladder'], int(g)), 'match': sorted(st) == A.compose(s['ladder'], int(g))}
        e['rung_steps'] = steps
        sv = d['saved']
        e['saved'] = {c: {k_: x.get(k_) for k_ in ('k', 'strength', 'steps_occ', 'op', 'flag')} for c, x in sv.items()}
        if 'IA_OSG' in sv:
            e['cand_steps_match'] = sorted(sv['IA_OSG']['steps_occ']) == A.compose(s['ladder'], int(s['k_star']))
        if 'IA_OSG_BLIND' in sv:
            e['blind_steps_match'] = sorted(sv['IA_OSG_BLIND'].get('steps_occ') or []) == A.compose(bl['ladder'], int(bl['k']))
        e['path_points'] = d.get('path_points')
        res[f'V{v}'] = e
    out['A2'] = res


def a8(out):
    q = [json.loads(l) for l in V.QUEUE_LEDGER.read_text().splitlines() if l.strip()]
    since = V.parse_utc('2026-10-22T03:11:00+00:00')
    units = []
    for e in q:
        if e.get('event') == 'unit_end' and V.parse_utc(e['utc']) >= since:
            units.append({'unit': e['unit_id'], 'end': e['utc'], 'status': e['status'], 'cpu': e['charged_cpu_s'],
                          'code_changed_during_run': e.get('code_changed_during_run')})
    starts = {e['unit_id']: e for e in q if e.get('event') == 'unit_start'}
    root = V.PRIV / 'compute' / 'units'
    drift, man = {}, []
    for u in units:
        s = starts.get(u['unit'])
        u['start'] = s['utc'] if s else None
        u['launch_drift'] = bool(s and s.get('code_digest_at_launch') != s.get('code_digest_at_registration'))
        p = root / u['unit'] / 'CODE_AT_LAUNCH.json'
        if p.is_file():
            drift[u['unit']] = json.loads(p.read_text()).get('files_changed')
        p0 = root / u['unit'] / 'CODE_AT_START.json'
        if p0.is_file():
            man.append((u['start'] or '', u['unit'], json.loads(p0.read_text())['files']))
    man.sort()
    changes = []
    for (t0, u0, f0), (t1, u1, f1) in zip(man[:-1], man[1:]):
        diff = sorted(k for k in set(f0) | set(f1) if f0.get(k) != f1.get(k))
        if diff:
            changes.append({'between': [u0, u1], 'at_or_before': t1, 'files': diff})
    out['A8'] = {'units': units, 'launch_drift_files': drift,
                 'flagged': {u['unit']: u['code_changed_during_run'] for u in units if u['code_changed_during_run']},
                 'manifest_changes_between_units': changes}


def main():
    t0 = time.process_time()
    from ipr import common as C                     # label-free features only
    F = C.features('MODEL_SELECT')
    names = tuple(map(str, F['feature_names']))
    out = {'unit_id': V.unit_id(), 'utc_start': V.utc(), 'errors': []}
    for name, fn in (('A1', lambda: a1(out, F, names)), ('A2', lambda: a2(out)), ('A8', lambda: a8(out))):
        try:
            fn()
        except Exception as e:
            out['errors'].append({'part': name, 'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()[-3000:]})
    out['cpu_s'] = round(time.process_time() - t0, 2)
    out['utc_end'] = V.utc()
    p = V.write_record(f'A1_INTEGRITY__{V.unit_id() or "NOQUEUE"}.json', out)
    s = {'record': str(p), 'errors': [x.get('part', x.get('config')) for x in out['errors']], 'cpu_s': out['cpu_s']}
    if 'A1' in out:
        new = [c for c in out['A1']['configs'] if c.get('new')]
        s['A1'] = {'new': len(new), 'all_configs': len(out['A1']['configs']),
                   'hashes': all(c.get('npz_sha_eq_json') and c.get('npz_sha_eq_ledger') and c.get('opened_wire_hashes_match')
                                 and c.get('wire_keys_match') and c.get('model_sha_eq_meta') and c.get('model_sha_eq_release_json')
                                 for c in new),
                   'cards': all(c.get('card_sha_eq_commit_blob') and c.get('card_commit_before_save')
                                and c.get('card_commit_before_first_fitting_read') for c in new),
                   'deploy_all_39_bitwise': all(c.get('deploy_all_bitwise') for c in out['A1']['configs']),
                   'income_eq_twin_all': all(x['all'] for x in out['A1']['income_equals_twin'].values()),
                   'n_income_checked': len(out['A1']['income_equals_twin']), 'aliases': out['A1']['ms_exact_aliases_pair']}
    if 'A2' in out:
        s['A2'] = {v: {'k': e['k_star']['match'], 'br': {k: x['match'] for k, x in e['brackets'].items()}, 'blind': e['blind']['match'],
                       'ladders': {k: x['match'] for k, x in e['ladders'].items() if isinstance(x, dict)},
                       'consistency': [e['consistency']['own_max_abs_diff_dT_occ'], e['consistency']['own_max_abs_diff_dI']],
                       'fineness': e['fineness_own']} for v, e in out['A2'].items()}
    if 'A8' in out:
        s['A8'] = {'flagged': out['A8']['flagged'], 'launch_drift': out['A8']['launch_drift_files'],
                   'n_manifest_changes': len(out['A8']['manifest_changes_between_units'])}
    print(json.dumps(s, default=V._default))
    return 0 if not out['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
