"""Verifier check 1 (label-free): release integrity and deployment recomputation; card declarations the rules cannot verify (IB
operation_sex_blind and the D5 collapse; IA ro = GENERIC_ZTILT and the substituted ladder); the MODEL_FIT strength selection, control
bracket points and fineness classes re-derived from the recorded MODEL_FIT spends (label-free arithmetic on the fit records); the role
members and the bank's array members.

For every row of runs/CONFIG_LEDGER.jsonl: RELEASE.npz SHA-256 == RELEASE.json release_sha256 == ledger release_sha256; the
MODEL_FIT / CALIBRATION / MODEL_SELECT wire hashes in RELEASE.json == own array hashes (ATTACK_* / DEV_REPORT arrays are not opened);
MODEL.pkl SHA-256 == MODEL.json == RELEASE.json model_sha256; the card SHA-256 equals the card blob at its recorded commit and the
commit precedes the configuration's save time; the family deployment function (ipr.ia.predict, ipr.ib.predict, ipr.ctl.predict)
recomputes the MODEL_SELECT arrays bitwise from the admitted MODEL_SELECT features.

IB SEX-blindness: the CSF / IND payload state holds only (pi, mode, tag, T_inc, T_occ); T_inc / T_occ are recomputed here from the
cached twin's MODEL_FIT out-of-fold vectors alone (TASK.pkl, admission-pinned; no SEX), and the MODEL_SELECT release is recomputed
with own code from the saved twin vectors, the SHA-256 key of the compact six fields and pi only: bitwise equality means no
SEX-fitted quantity enters the release. D5: both sequential orders are built from that recomputation (second stage = the people whose
first-stage vector is a decision-cell row, identified from the first stage's release) and compared wire by wire with IB_CSF.

    python3 -m ipr.jobqueue run --unit IPR_VER_INTEGRITY_V1 --stage verify --rss-gib 2 -- python3 <this file>
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
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vlib as V  # noqa: E402

OPEN_ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT')     # ATTACK_* / DEV_REPORT arrays are never opened by the verifier


def git(*a):
    r = subprocess.run(['git', '-C', str(V.ROOT), *a], capture_output=True)
    return r.returncode, r.stdout


def own_clip(P, floor=1e-6):
    P = np.clip(np.asarray(P, dtype=np.float64), floor, None)
    return P / P.sum(-1, keepdims=True)


def own_clip_hi(P, floor=1e-6):
    P = np.clip(np.asarray(P, dtype=np.float64), floor, 1.0)
    return P / P.sum(-1, keepdims=True)


def own_tables(P, K):
    """Decision-cell means of clipped twin vectors (card IB_CSF_v1 section 1); flags any cell whose mean loses its argmax."""
    P = own_clip(P)
    dec = P.argmax(1)
    T = np.zeros((K, K))
    issues = []
    for k in range(K):
        if not np.any(dec == k):
            issues.append(f'empty cell {k}')
            continue
        m = P[dec == k].mean(0)
        if m.argmax() != k:
            issues.append(f'cell {k} mean loses its argmax')
        T[k] = m / m.sum()
    return T, issues, np.bincount(dec, minlength=K).tolist()


def own_key(Z, tag):
    Z = np.ascontiguousarray(np.asarray(Z, np.float64))
    return np.array([int(hashlib.sha256(f'{tag}|{r.tobytes().hex()}'.encode()).hexdigest()[:13], 16) / float(16 ** 13) for r in Z])


def is_table_row(R, T):
    """Per person: the released vector equals (bitwise) one of the clipped table rows."""
    rows = own_clip_hi(T)
    hit = np.zeros(len(R), dtype=bool)
    for k in range(rows.shape[0]):
        hit |= np.all(R == rows[k][None, :], axis=1)
    return hit


def bitwise(a, b):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64)); b = np.ascontiguousarray(np.asarray(b, dtype=np.float64))
    return bool(a.shape == b.shape and a.tobytes() == b.tobytes())


def main():
    t0 = time.process_time()
    from ipr import common as C              # label-free features only (no load call in this script)
    import ipr.ia as IA
    import ipr.ib as IB
    import ipr.ctl as CTL
    PRED = {'IA': IA.predict, 'IB': IB.predict, 'CTL': CTL.predict}
    rows = [json.loads(l) for l in V.CONFIG_LEDGER.read_text().splitlines() if l.strip()]
    F = C.features('MODEL_SELECT')
    names = tuple(map(str, F['feature_names']))
    out = {'unit_id': V.unit_id(), 'utc_start': V.utc(), 'configs': [], 'errors': []}
    card_blob = {}
    ms = {}
    for r in rows:
        fam, cfg, v = r['family'], r['config_id'], int(r['version'])
        rec = {'family': fam, 'config': cfg, 'version': v, 'configuration_number': r['configuration_number'], 'kind': r['kind']}
        try:
            d = V.RUNS / fam / cfg / f'V{v}'
            R = json.loads((d / 'RELEASE.json').read_text())
            fsha = V.file_sha(d / 'RELEASE.npz')
            rec['npz_sha_eq_json'] = fsha == R['release_sha256']
            rec['npz_sha_eq_ledger'] = fsha == r['release_sha256']
            arrays = {}
            with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
                keys = list(z.files)
                for k in keys:
                    if k.split('__')[0] in OPEN_ROLES:
                        arrays[k] = z[k].copy()
            wires = R.get('wire_sha256', {})
            rec['wire_keys_match'] = sorted(wires) == sorted(keys)
            rec['opened_wire_hashes_match'] = all(V.arr_sha(arrays[k]) == wires.get(k) for k in arrays)
            rec['n_roles'] = len({k.split('__')[0] for k in keys})
            meta = json.loads((d / 'MODEL.json').read_text())
            msha = V.file_sha(d / 'MODEL.pkl')
            rec['model_sha_eq_meta'] = msha == meta['model_sha256']
            rec['model_sha_eq_release_json'] = msha == R.get('model_sha256')
            key = (R['card'], R['card_commit'])
            if key not in card_blob:
                rc, blob = git('show', f"{R['card_commit']}:{R['card']}")
                card_blob[key] = hashlib.sha256(blob).hexdigest() if rc == 0 else None
            rec['card_sha_eq_commit_blob'] = card_blob[key] == R['card_sha256'] == r['card_sha256']
            rec['card_sha_eq_current_file'] = V.file_sha(V.ROOT / R['card']) == R['card_sha256']
            rec['card_commit_before_save'] = V.parse_utc(R['card_commit_utc']) <= V.parse_utc(R['utc'])
            rec['card_commit_utc'] = str(V.parse_utc(R['card_commit_utc']))
            rec['save_utc'] = R['utc']
            rec['randomized'] = bool(R.get('randomized'))
            rec['strength'] = R.get('strength')
            rec['op'] = R.get('op')
            got = PRED[fam](cfg, v, F['X'], names)
            eq = {t: bitwise(got[t], arrays[f'MODEL_SELECT__{t}']) for t in V.TASKS}
            rec['deploy_ms_bitwise'] = eq
            rec['deploy_all_bitwise'] = all(eq.values())
            rec['ms_wire_sha'] = {t: V.wire_sha(arrays[f'MODEL_SELECT__{t}']) for t in V.TASKS}
            ms[(fam, cfg, v)] = {t: arrays[f'MODEL_SELECT__{t}'] for t in V.TASKS}
        except Exception as e:  # recorded, never hidden
            rec['error'] = f'{type(e).__name__}: {e}'
            out['errors'].append({'config': f'{fam}/{cfg}/V{v}', 'traceback': traceback.format_exc()[-2000:]})
        out['configs'].append(rec)

    keys = {(r['family'], r['config_id'], int(r['version'])) for r in rows}
    out['budget'] = {'distinct_configurations': len(keys), 'limit': 48, 'ledger_rows': len(rows),
                     'per_family': {f: sum(1 for k in keys if k[0] == f) for f in ('CTL', 'IA', 'IB', 'FS')}}
    al = {}
    for v in (0, 1):
        by = {}
        for c in out['configs']:
            if c['version'] == v and 'ms_wire_sha' in c:
                by.setdefault((c['ms_wire_sha']['income'], c['ms_wire_sha']['occupation']), []).append(f"{c['family']}/{c['config']}")
        al[f'V{v}'] = [sorted(x) for x in by.values() if len(x) > 1]
    out['ms_exact_aliases_pair'] = al
    # release-sha level duplicates (whole RELEASE.npz identical)
    dup = {}
    for r in rows:
        dup.setdefault(r['release_sha256'], []).append(f"{r['family']}/{r['config_id']}/V{r['version']}")
    out['identical_release_files'] = [x for x in dup.values() if len(x) > 1]

    # ------------------------------------------------------------ IB: SEX-blindness and the D5 collapse
    ib = {}
    adm = json.loads((V.RESULTS / 'SOURCE_ADMISSION.json').read_text())
    Zms = None
    try:
        from ipr import famkit as FK            # label-free compact six-field coding (ohn.fam_a.compact_po), for the key only
        Zms = FK.compact(F['X'], names)
    except Exception as e:
        out['errors'].append({'config': 'compact', 'traceback': f'{type(e).__name__}: {e}'})
    for v in (0, 1):
        ent = {}
        try:
            tp = V.ROOT / '.private' / 'pcrl_new_release_mechanisms_v1' / 'fa_cache' / f'V{v}' / 'TASK.pkl'
            b = tp.read_bytes()
            want = adm['source_private_artefacts'][f'fa_cache/V{v}/TASK.pkl']['sha256']
            ent['task_cache_sha_ok'] = hashlib.sha256(b).hexdigest() == want
            T = pickle.loads(b)
            ent['task_cache_keys'] = sorted(T)
            TI, iss_i, n_i = own_tables(T['oof']['inc'], 2)
            TO, iss_o, n_o = own_tables(T['oof']['occ'], 5)
            ent['own_table_issues'] = iss_i + iss_o
            ent['decision_cell_counts_modelfit'] = {'income': n_i, 'occupation': n_o}
            twin = ms[('CTL', 'CTL_TWIN', v)]
            for cfg in ('IB_CSF', 'IB_IND', 'IB_CSFT'):
                d = V.RUNS / 'IB' / cfg / f'V{v}'
                buf = (d / 'MODEL.pkl').read_bytes()
                meta = json.loads((d / 'MODEL.json').read_text())
                if hashlib.sha256(buf).hexdigest() != meta['model_sha256']:
                    raise RuntimeError(f'{cfg} V{v}: MODEL.pkl hash differs')
                P = pickle.loads(buf)
                st = P['state']
                e2 = {'payload_keys': sorted(P), 'state_keys': sorted(st), 'pi': st.get('pi'), 'mode': st.get('mode'),
                      'tag': st.get('tag'),
                      'tables_equal_own_from_twin_oof': {'income_max_abs': float(np.max(np.abs(st['T_inc'] - TI))),
                                                         'occupation_max_abs': float(np.max(np.abs(st['T_occ'] - TO))),
                                                         'income_bitwise': bitwise(st['T_inc'], TI),
                                                         'occupation_bitwise': bitwise(st['T_occ'], TO)},
                      'twin_models_are_ctl_twin': None}
                # twin models inside the payload equal CTL_TWIN's (pickled bytes)
                Pt = pickle.loads((V.RUNS / 'CTL' / 'CTL_TWIN' / f'V{v}' / 'MODEL.pkl').read_bytes())
                e2['twin_models_are_ctl_twin'] = (pickle.dumps(P['twin_inc'], protocol=5) == pickle.dumps(Pt['twin_inc'], protocol=5)
                                                  and pickle.dumps(P['twin_occ'], protocol=5) == pickle.dumps(Pt['twin_occ'], protocol=5))
                if cfg in ('IB_CSF', 'IB_IND') and Zms is not None:
                    pI, pO = own_clip(twin['income']), own_clip(twin['occupation'])
                    u = own_key(Zms, st['tag'])
                    fI = u < st['pi']
                    fO = fI if cfg == 'IB_CSF' else own_key(Zms, st['tag'] + '|O') < st['pi']
                    rI, rO = pI.copy(), pO.copy()
                    rI[fI] = TI[pI[fI].argmax(1)]
                    rO[fO] = TO[pO[fO].argmax(1)]
                    rI, rO = own_clip_hi(rI), own_clip_hi(rO)
                    saved = ms[('IB', cfg, v)]
                    e2['own_recompute_from_sex_free_inputs_bitwise'] = {'income': bitwise(rI, saved['income']),
                                                                        'occupation': bitwise(rO, saved['occupation'])}
                    e2['flattened_fraction_ms'] = {'income': float(fI.mean()), 'occupation': float(fO.mean())}
                    e2['tag_matches_card'] = st['tag'] == f'IPR_B_CSF|V{v}'
                    nf = ~fI
                    e2['non_flattened_income_bitwise_twin'] = bitwise(saved['income'][nf], twin['income'][nf])
                    e2['non_flattened_income_max_abs_vs_twin'] = float(np.max(np.abs(saved['income'][nf] - twin['income'][nf])))
                    nfo = ~fO
                    e2['non_flattened_occupation_bitwise_twin'] = bitwise(saved['occupation'][nfo], twin['occupation'][nfo])
                    e2['non_flattened_occupation_max_abs_vs_twin'] = float(np.max(np.abs(saved['occupation'][nfo] - twin['occupation'][nfo])))
                    e2['decisions_kept'] = {'income': bool(np.array_equal(V.argmax_first(saved['income']), V.argmax_first(twin['income']))),
                                            'occupation': bool(np.array_equal(V.argmax_first(saved['occupation']),
                                                                              V.argmax_first(twin['occupation'])))}
                    if cfg == 'IB_CSF':
                        # D5: sequential orders. Income first: stage 1 flattens income for u < pi; stage 2 (occupation) flattens the
                        # people whose released income vector is a decision-cell row. Occupation first: the mirror image.
                        idI = is_table_row(saved['income'], TI)
                        idO = is_table_row(saved['occupation'], TO)
                        e2['d5'] = {'income_first_stage2_set_equals_key_set': bool(np.array_equal(idI, fI)),
                                    'occupation_first_stage2_set_equals_key_set': bool(np.array_equal(idO, fI)),
                                    'income_rows_identified': int(idI.sum()), 'occupation_rows_identified': int(idO.sum()),
                                    'key_set_size': int(fI.sum())}
                        sO = pO.copy(); sO[idI] = TO[pO[idI].argmax(1)]; sO = own_clip_hi(sO)
                        sI = pI.copy(); sI[idO] = TI[pI[idO].argmax(1)]; sI = own_clip_hi(sI)
                        e2['d5']['income_first_wire_equals_csf'] = {'income': bitwise(saved['income'], saved['income']),
                                                                    'occupation': bitwise(sO, saved['occupation'])}
                        e2['d5']['occupation_first_wire_equals_csf'] = {'income': bitwise(sI, saved['income']),
                                                                        'occupation': bitwise(saved['occupation'], saved['occupation'])}
                        e2['d5']['note'] = ('own construction: stage 2 may read the first stage release; with the shared key it reads '
                                            'u < pi directly, which is the same set when the identification is exact')
                ent[cfg] = e2
        except Exception as e:
            ent['error'] = f'{type(e).__name__}: {e}'
            out['errors'].append({'config': f'IB_V{v}', 'traceback': traceback.format_exc()[-2000:]})
        ib[f'V{v}'] = ent
    out['ib_sex_blind_and_d5'] = ib

    # ------------------------------------------------------------ CTL points: op and strength
    ctl = {}
    for c in out['configs']:
        if c['family'] == 'CTL' and c['config'] != 'CTL_TWIN':
            d = V.RUNS / 'CTL' / c['config'] / f"V{c['version']}"
            P = pickle.loads((d / 'MODEL.pkl').read_bytes())
            ctl[f"{c['config']}/V{c['version']}"] = {'payload_op': P.get('op'), 'payload_keys': sorted(P),
                                                      'record_strength': c.get('strength'),
                                                      'name_strength': float(c['config'].split('_')[-1].replace('p', '.'))}
    out['ctl_points'] = ctl

    # ------------------------------------------------------------ MODEL_FIT selection, brackets and fineness (label-free arithmetic)
    def meets(sp, budget):
        return all(sp[m]['estimate'] + 2 * sp[m]['se'] <= lim for m, lim in budget.items())

    def bracket(b):
        grid = b['grid']
        gs = sorted(grid, key=float)
        lo = [g for g in gs if grid[g]['spend'] <= b['cand_spend']]
        hi = [g for g in gs if grid[g]['diff_vs_cand'] >= 2 * grid[g]['se_diff']]
        return (float(lo[-1]) if lo else None, float(hi[0]) if hi else float(gs[-1]),
                None if hi else 'CONTROL_PATH_MAY_NOT_BRACKET')

    def fineness(fi):
        s, lo, up, se, mx = fi['cand'], fi['lower'], fi['upper'], fi['upper_se'], fi['max']
        if s <= 0:
            return 'NOT_APPLICABLE_CAND_NOT_ABOVE_TWIN'
        if mx < s + 2 * se:
            return 'CONTROL_PATH_MAY_NOT_BRACKET'
        if lo > s or up < s + 2 * se:
            return 'SAVED_POINT_RULE_NOT_FOLLOWED'
        below = lo >= 0.75 * s
        above = up <= s + 2 * se + 0.25 * s
        return 'FINE' if below and above else 'COARSE_BOTH' if not below and not above else 'COARSE_BELOW' if not below else 'COARSE_ABOVE'

    sel = {}
    for fam in ('ia', 'ib'):
        for v in (0, 1):
            d = json.loads((V.RESULTS / fam / f'FIT_V{v}.json').read_text())
            s = d['selection']
            budget = s['budget']
            ent = {'budget': budget}
            if fam == 'ia':
                order = sorted(s['path'], key=int)
                pick, first_fail = None, None
                for g in order:
                    if meets(s['path'][g], budget):
                        pick = int(g)
                    else:
                        first_fail = int(g)
                        break
                ent.update({'recomputed_k_star': pick, 'recorded_k_star': s['k_star'], 'match': pick == s['k_star'],
                            'first_failing_rung': first_fail, 'ladder_first_steps': s['ladder'][:max(1, pick or 0)]})
                if first_fail is not None:
                    sp = s['path'][str(first_fail)]
                    ent['first_failing_ratio'] = {m: (sp[m]['estimate'] + 2 * sp[m]['se']) / lim for m, lim in budget.items()
                                                  if sp[m]['estimate'] + 2 * sp[m]['se'] > lim}
                ent['cand_spend_recorded'] = d['cand_spend']
                ent['cand_spend_eq_path_tot_nll'] = abs(s['path'][str(s['k_star'])]['tot_nll']['estimate'] - d['cand_spend']) < 1e-15
                br = {}
                for key, b in (('SPEND', d['spend_path_bracket']), ('RO', d['ro_path_bracket']),
                               ('SBC', d['sbc_tries'][-1]['bracket'])):
                    lo, hi, flag = bracket(b)
                    rec_lo = None if b['lo'] is None else float(b['lo'])
                    br[key] = {'recomputed': [lo, hi, flag], 'recorded': [rec_lo, float(b['hi']), b['flag']],
                               'match': [lo, hi, flag] == [rec_lo, float(b['hi']), b['flag']]}
                ent['brackets'] = br
                ent['sbc_tries'] = [t['target'] for t in d['sbc_tries']]
                ent['sbc_construction'] = d['sbc_construction']
                ent['shared_points'] = d['shared_points']
                bl = d['blind']
                kb = None
                for g in sorted(bl['scan'], key=int):
                    if bl['scan'][g] >= d['cand_spend']:
                        kb = int(g)
                        break
                ent['blind'] = {'recomputed_k': kb, 'recorded_k': bl['k'], 'match': kb == bl['k'],
                                'blind_first_steps': bl['ladder'][:bl['k']], 'cand_steps': s['ladder'][:s['k_star']]}
                ent['substituted_ladder'] = d['substituted_ladder']
                ent['ro_record'] = {k: d['ro_record'][k] for k in ('cov_qz_sex', 'fallback_strata')}
            else:
                order = sorted(s['path'], key=float)
                pick, first_fail = None, None
                for g in order:
                    if meets(s['path'][g], budget):
                        pick = float(g)
                    else:
                        first_fail = float(g)
                        break
                ent.update({'recomputed_pi_star': pick, 'recorded_pi_star': s['pi_star'], 'match': pick == s['pi_star'],
                            'first_failing_pi': first_fail})
                if first_fail is not None:
                    sp = s['path'][str(first_fail)] if str(first_fail) in s['path'] else s['path'][[g for g in s['path'] if float(g) == first_fail][0]]
                    ent['first_failing_ratio'] = {m: (sp[m]['estimate'] + 2 * sp[m]['se']) / lim for m, lim in budget.items()
                                                  if sp[m]['estimate'] + 2 * sp[m]['se'] > lim}
                lo, hi, flag = bracket(d['spend_path_bracket'])
                b = d['spend_path_bracket']
                rec_lo = None if b['lo'] is None else float(b['lo'])
                ent['brackets'] = {'SPEND': {'recomputed': [lo, hi, flag], 'recorded': [rec_lo, float(b['hi']), b['flag']],
                                             'match': [lo, hi, flag] == [rec_lo, float(b['hi']), b['flag']]}}
                ent['shared_points'] = d['shared_points']
                ent['csft_rates'] = d.get('csft_rates')
            ent['fineness_own'] = {k: fineness(fi) for k, fi in d['fineness_inputs'].items()}
            sel[f'{fam.upper()}_V{v}'] = ent
    out['selection'] = sel

    # ------------------------------------------------------------ role members and bank array members
    roles = json.loads((V.RESULTS / 'paths' / 'SHARED_ROLES.json').read_text())['roles']
    spec = json.loads((V.RESULTS / 'grid_specs' / 'IPR1_MAIN.json').read_text())
    rm = {}
    for name, per in roles.items():
        fam_key = 'ia' if name.startswith('IA') else 'ib'
        path = 'RO' if '_RO_' in name else 'SPEND'
        side = name.split('_')[-1]
        e = {}
        for v in ('0', '1'):
            fitrec = json.loads((V.RESULTS / fam_key / f'FIT_V{v}.json').read_text())
            exp = 'CTL/' + fitrec['shared_points'][path][side]
            mem = spec['members'][name]['seeds'][v]
            npz = V.ROOT / mem['income']['path']
            e[v] = {'role_resolves_to': per[v], 'fit_record_point': exp, 'match': per[v] == exp,
                    'bank_path_is_role': str(Path(mem['income']['path']).parent.parent.name) == per[v].split('/')[1],
                    'bank_sha_eq_file': V.file_sha(npz) == mem['income']['sha256'] == mem['occupation']['sha256'],
                    'bank_keys': [mem['income']['key'], mem['occupation']['key']]}
        rm[name] = e
    out['role_members'] = rm
    plan_members = {}
    for pid in ('IA_REG', 'IB_REG'):
        p = json.loads((V.RESULTS / 'claims' / f'PLAN_{pid}.json').read_text())
        plan_members[pid] = {'paths': p['paths'], 'members': p['members'], 'ro': p['ro'], 'operation_sex_blind': p['operation_sex_blind'],
                             'joint': p['joint'], 'strengths': p['strengths'], 'alias_of': p['alias_of'],
                             'controls_not_run': p['controls_not_run']}
    out['plans'] = plan_members
    saved_not_in_bank = sorted({f"{r['family']}/{r['config_id']}" for r in rows if r['family'] in ('IA', 'IB')} -
                               {f"{m['family']}/{m['config']}" for m in spec['members'].values() if m.get('type') == 'ipr'})
    out['saved_family_configs_not_in_bank'] = saved_not_in_bank
    ctl_in_roles = {x for per in roles.values() for x in per.values()}
    out['saved_ctl_points_not_in_roles'] = sorted({f"CTL/{r['config_id']}/V{r['version']}" for r in rows
                                                   if r['family'] == 'CTL' and r['config_id'] != 'CTL_TWIN'}
                                                  - {f'{x}/V{v}' for name, per in roles.items() for v, x in per.items()})
    out['cpu_s'] = round(time.process_time() - t0, 2)
    out['utc_end'] = V.utc()
    p = V.write_record(f'V1_INTEGRITY__{V.unit_id() or "NOQUEUE"}.json', out)
    summ = {'configs': len(out['configs']), 'errors': len(out['errors']),
            'all_npz_json_ledger': all(c.get('npz_sha_eq_json') and c.get('npz_sha_eq_ledger') for c in out['configs']),
            'all_wires': all(c.get('opened_wire_hashes_match') and c.get('wire_keys_match') for c in out['configs']),
            'all_models': all(c.get('model_sha_eq_meta') and c.get('model_sha_eq_release_json') for c in out['configs']),
            'all_cards': all(c.get('card_sha_eq_commit_blob') and c.get('card_commit_before_save') for c in out['configs']),
            'all_deploy_bitwise': all(c.get('deploy_all_bitwise') for c in out['configs']),
            'selection_matches': {k: e['match'] for k, e in sel.items()},
            'aliases': al, 'identical_files': out['identical_release_files'], 'not_in_bank': saved_not_in_bank,
            'ctl_not_in_roles': out['saved_ctl_points_not_in_roles'], 'cpu_s': out['cpu_s'], 'record': str(p)}
    print(json.dumps(summ, default=V._default))
    return 0 if not out['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
