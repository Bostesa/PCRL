"""Original-benchmark evaluation, development reserve and paired training-size contrasts on MODEL_SELECT for the PCRL controlled
training-size campaign (operation 'benchmark', logged; MODEL_SELECT was used adaptively by earlier campaigns: development evidence).

Unchanged definitions (ohn.evaluate.complete_eval -> ihs.evaluate.utility): per version v the 12 point clauses (income and occupation
x U0 and Ucal x accuracy loss <= 0.010, NLL excess <= 0.010 nats with the inherited 1e-12 floor, summed Brier excess <= 0.005;
household-clustered paired SEs) and the men's and women's occupation summed-Brier excess vs U0 (<= 0.005); the frozen references
U0_S<v> / Ucal_S<v> and the frozen H3 income S<v>; the development reserve via ohn.rules.reserve (k = 2, floor = H_LJ's MODEL_SELECT
SE of the same condition and seed, rescaled by sqrt(floor households / households)). A release VERSION is (occupation output of
version v, H3 income S<v>, references of seed v); the two versions together give 24 clauses + 4 guards.

Paired contrasts are reference-free: per-person occupation accuracy, NLL and summed Brier of two outputs on the SAME MODEL_SELECT
people, differenced, with household-clustered SEs (men's / women's: linearized subgroup SEs). Nothing here selects a model.

    python3 -m tsz.evaluate trigger       W100_0 and W100_1 (A1_PO_16 on S100): utility, reserve and the frozen trigger inputs
    python3 -m tsz.evaluate size          every size fit and control: SIZE_TABLE, paired contrasts, reproduction, encoding
    python3 -m tsz.evaluate release --name N --spec SPEC.json     a release given by arrays per version (conditional stage)
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C

OUT = C.RESULTS / 'size'
METRICS = ('accuracy', 'nll', 'brier', 'men_brier', 'women_brier')


def ms(purpose):
    return C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'benchmark', purpose)


def refs_of(v, S):
    from ohn import common as OC
    bank, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
    if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
        raise AssertionError('frozen bank MODEL_SELECT row order differs')
    refs = {r: {t: bank[r]['MODEL_SELECT'][t] for t in C.TASKS} for r in ('U0', 'Ucal')}
    others = {'T_TREE': bank['T_TREE']['MODEL_SELECT']['occupation'], 'TASK_REFIT': bank['TASK_REFIT']['MODEL_SELECT']['occupation'],
              'JOINT': bank['H_LJ']['MODEL_SELECT']['occupation']}
    return refs, others, bank


def h3_income(v, role='MODEL_SELECT'):
    from ohn import devrun as OD
    return OD.anchor_income('H3', v, role)


def eval_version(v, occ_ms, S, ctx=None):
    from ohn import evaluate as OE
    refs, others, _ = ctx[v] if ctx else refs_of(v, S)
    return OE.complete_eval(v, h3_income(v), occ_ms, S, refs, others)


def reserve(evs):
    """ohn.rules.reserve over the given versions {v: eval}; unchanged source rule (RULES.json of the occupation-headroom campaign)."""
    from ohn import rules as OR
    return OR.reserve({v: e['utility'] for v, e in evs.items()})


def per_person(y, p, sex):
    t = C.task_vectors(y, p, C.K_OCC)
    return {'accuracy': t['accuracy'], 'nll': t['nll'], 'brier': t['brier'], 'men': sex == 0, 'women': sex == 1}


def paired_diff(a, b, unit):
    """B - A on the same people (positive nll / brier = B worse; accuracy positive = B better); clustered SEs."""
    from ihs.evaluate import cluster_se, subgroup_se
    out = {}
    for m in ('accuracy', 'nll', 'brier'):
        d = b[m] - a[m]
        out[m] = {'value': float(d.mean()), 'se': cluster_se(d, unit)}
    d = b['brier'] - a['brier']
    out['men_brier'] = {'value': float(d[a['men']].mean()), 'se': subgroup_se(d, a['men'], unit)}
    out['women_brier'] = {'value': float(d[a['women']].mean()), 'se': subgroup_se(d, a['women'], unit)}
    return out


def condition_rows(ev, v):
    """Flat per-condition rows of one version's utility (value, SE, cap, point status, margin, margin in SE units)."""
    u = ev['utility']
    rows = {}
    for c in u['clauses']:
        name = f"{c['task']}|{c['reference']}|{c['metric']}"
        se = float(c.get('se', float('nan')))
        rows[f's{v}|{name}'] = {'value': float(c['value']), 'se': se, 'cap': float(c['cap']), 'point_pass': float(c['value']) <= float(c['cap']),
                                'margin': float(c['cap']) - float(c['value']),
                                'margin_se_units': (float(c['cap']) - float(c['value'])) / se if se > 0 else None}
    for g in u['guards']:
        se = float(g.get('se', float('nan')))
        rows[f"s{v}|guard|{g['group']}"] = {'value': float(g['value']), 'se': se, 'cap': float(g['cap']),
                                             'point_pass': float(g['value']) <= float(g['cap']), 'margin': float(g['cap']) - float(g['value']),
                                             'margin_se_units': (float(g['cap']) - float(g['value'])) / se if se > 0 else None}
    return rows


# ------------------------------------------------------------------ release loaders (MODEL_SELECT arrays of saved fits)
def a1_release(fit_id, role='MODEL_SELECT'):
    d = C.RUNS / 'SIZE' / fit_id
    rec = json.loads((d / 'FIT.json').read_text())
    p = d / 'RELEASE.npz'
    if C.sha(p) != rec['files_sha256']['RELEASE.npz']:
        raise PermissionError(f'{fit_id}: RELEASE.npz differs from its record')
    with np.load(p) as z:
        return z[role].copy(), rec


def control_release(sid, v, stage, role='MODEL_SELECT'):
    """stage 'TR' (TASK_REFIT occupation, the primary control) or 'TT' (T_TREE occupation, its first stage); hash-checked against
    the saved record. FileNotFoundError if the cell has not been fitted."""
    from . import control as CT
    d = CT.cell_dir(sid, v)
    if stage == 'TT':
        sub, recn, fname, key = 'TEACHERS', 'TEACHERS.json', 'TEACHER_OUTPUTS.npz', f'T_TREE__{role}__occupation'
    elif stage == 'TR':
        sub, recn, fname, key = 'STAGE_B', 'STAGE_B.json', 'STAGE_B_OUTPUTS_FINAL.npz', f'TASK_REFIT__{role}__occupation'
    else:
        raise KeyError(stage)
    rec = json.loads((d / sub / recn).read_text())
    p = d / sub / fname
    if C.sha(p) != rec['files_sha256'][fname]:
        raise PermissionError(f'{sid} v{v} {fname} differs from its record')
    with np.load(p) as z:
        arr, rid = z[key].copy(), z[f'row_id__{role}'].copy()
    if not np.array_equal(rid, C.features(role)['row_id']):
        raise AssertionError(f'{sid} v{v}: {role} row order differs')
    trec = rec if stage == 'TT' else json.loads((d / 'TEACHERS' / 'TEACHERS.json').read_text())
    info = {'n_people': trec.get('n_people'), 'n_households': trec.get('n_households'),
            'tree_counts': {t: (trec.get('teachers') or {}).get(t, {}).get('tree_count') for t in ('income', 'occupation')}}
    if stage == 'TR':
        info.update({k: rec.get(k) for k in ('n', 'epoch_order', 'final_label')})
    return arr, info


def fit_ids(recipe):
    if recipe == 'A1':
        return {(sid, v): f'A1__{sid}__V{v}' for sid in C.subset_ids() for v in C.VERSIONS}
    if recipe == 'A1_ADM':
        return {(f'S50_{o}', 0): f'A1__S50_{o}__V0__ADM' for o in C.ORDERINGS}
    raise KeyError(recipe)


def releases_available():
    """{(recipe, sid, v): MODEL_SELECT occupation array} for every saved fit."""
    out, recs = {}, {}
    for recipe in ('A1', 'A1_ADM'):
        for key, fid in fit_ids(recipe).items():
            if (C.RUNS / 'SIZE' / fid / 'FIT.json').is_file():
                out[(recipe,) + key], recs[(recipe,) + key] = a1_release(fid)
    for sid in C.subset_ids():
        for v in C.VERSIONS:
            for stage in ('TR', 'TT'):
                try:
                    arr, rec = control_release(sid, v, stage)
                except FileNotFoundError:
                    continue
                out[(stage, sid, v)], recs[(stage, sid, v)] = arr, rec
    return out, recs


def source_a1(v, role='MODEL_SELECT'):
    from ohn import devrun as OD
    occ, _, _ = OD.load_occupation('A', 'A1_PO_16', v, roles=(role,))
    return occ[role]


def _slim(ev):
    """The utility block and the occupation diagnostics of an evaluation (no per-person arrays)."""
    return {k: ev[k] for k in ('utility', 'occupation_per_sex', 'occupation_versus_task_only_references', 'wire_sha256',
                               'distinct_vectors') if k in ev}


def _deployment_record():
    from . import rules as R
    p = C.ROOT / R.DEPLOYMENT_RECORD
    if not p.is_file():
        return {}, None
    dep = json.loads(p.read_text())
    per = {}
    for v in C.VERSIONS:
        e = dep.get(str(v)) or dep.get(v) or (dep.get('versions') or {}).get(str(v)) or (dep.get('per_version') or {}).get(str(v))
        if isinstance(e, dict):
            per[v] = {f: e.get(f) for f in R.DEPLOYMENT_FIELDS}
    return per, C.sha(p)


def trigger_cmd():
    """The registered task-stage trigger (RULES.md section 7), evaluated once."""
    from . import rules as R
    t0 = time.process_time()
    out = C.RESULTS / 'trigger' / 'TRIGGER.json'
    if out.exists():
        raise PermissionError('the trigger is evaluated once (TRIGGER.json exists)')
    S = ms('benchmark: registered task-stage trigger: W100_0 / W100_1 (A1_PO_16 refitted on all MODEL_FIT) with the frozen H3 '
           'income on MODEL_SELECT (24 clauses + 4 guards, occupation-side reserve)')
    ctx = {v: refs_of(v, S) for v in C.VERSIONS}
    evs, h3ok, h3sha, w = {}, {}, {}, {}
    for v in C.VERSIONS:
        occ, rec = a1_release(f'A1__S100__V{v}')
        inc = h3_income(v)                                        # ohn.devrun checks the admitted H3 release hash
        h3ok[v] = True
        h3sha[v] = C.arr_sha(inc)
        evs[v] = eval_version(v, occ, S, ctx)
        w[v] = {'fit_id': f'A1__S100__V{v}', 'release_sha256_MODEL_SELECT': C.arr_sha(occ), 'T_star': rec.get('T_star')}
    dep, dep_sha = _deployment_record()
    trig = R.trigger({v: evs[v]['utility'] for v in C.VERSIONS}, dep, h3ok)
    rec = {'schema': 'TSZ1-trigger-v1', 'utc': C.now(), 'witness': w, 'h3_income_wire_sha256': h3sha, 'deployment_record': {
               'path': R.DEPLOYMENT_RECORD, 'sha256': dep_sha, 'per_version': dep},
           'coordinator_deployment_check': 'deploy/DEPLOYMENT_CHECKS_W100.json (non-gating second record)',
           'trigger': trig, 'evaluations': {str(v): _slim(evs[v]) for v in C.VERSIONS}, 'cpu_s': time.process_time() - t0,
           'unit_id': __import__('os').environ.get(C.UNIT_ENV)}
    C.write_json_x(out, rec)
    print(json.dumps({'decision': trig['decision'], 'failing_parts': trig['failing_parts'],
                      'min_occ_ratio': {v: trig['occupation_reserve']['per_version'][v]['min_ratio'] for v in C.VERSIONS}
                      if trig.get('occupation_reserve') else None}, default=str))
    return rec


def _loss_change(a, b, unit):
    """loss(b) - loss(a) per metric (accuracy as error rate), with paired SEs: the rules' loss orientation."""
    d = paired_diff(a, b, unit)
    return {'acc_loss': {'value': -d['accuracy']['value'], 'se': d['accuracy']['se']}, 'nll': d['nll'], 'brier': d['brier'],
            'men_brier': d['men_brier'], 'women_brier': d['women_brier']}


def _size_cells():
    """Label-free completeness of the registered size design: (cells recorded missing by SUPPORT_FAILURE.json, cells neither fitted
    nor recorded missing). A missing cell is reported and its labels are INCOMPLETE (RULES 2); it is never silently dropped."""
    from . import control as CT
    missing, not_ready = [], []
    for recipe in ('A1', 'A1_ADM'):
        for fid in fit_ids(recipe).values():
            d = C.RUNS / 'SIZE' / fid
            if not (d / 'FIT.json').is_file():
                (missing if (d / 'SUPPORT_FAILURE.json').is_file() else not_ready).append(fid)
    for sid in C.subset_ids():
        for v in C.VERSIONS:
            d = CT.cell_dir(sid, v)
            for stage, rec in (('TT', d / 'TEACHERS' / 'TEACHERS.json'), ('TR', d / 'STAGE_B' / 'STAGE_B.json')):
                if not rec.is_file():
                    not_ready.append(f'{stage}:{sid}__V{v}')
    return missing, not_ready


def _nan_cell(vals, o):
    for m in vals:                                               # missing cell -> non-finite -> INCOMPLETE (RULES 2, 3.3)
        vals[m]['values'][o] = float('nan'); vals[m]['se'][o] = float('nan')


def size_cmd():
    """Every size fit and control on MODEL_SELECT: SIZE_TABLE, occupation-side reserve, paired contrasts with labels, control
    comparison, reproduction and encoding sensitivity (RULES.md sections 2-6)."""
    from . import rules as R
    from . import sizefit as SF
    t0 = time.process_time()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'SIZE_RESULTS.json').exists():
        raise PermissionError('SIZE_RESULTS.json exists (write-once)')
    missing_cells, not_ready = _size_cells()
    if not_ready:
        raise PermissionError('SIZE_RESULTS.json is write-once: every registered cell must be fitted or recorded missing '
                              f'(SUPPORT_FAILURE.json) before MODEL_SELECT is read; not ready: {not_ready}')
    S = ms('benchmark: training-size table on MODEL_SELECT: every A1_PO_16 size fit, the encoding-sensitivity fits and the '
           'equally exposed TASK_REFIT / T_TREE controls with the frozen H3 income (24 clauses + 4 guards, reserve, paired contrasts)')
    y, sex, unit = S['occupation'], S['sex'], S['unit']
    ctx = {v: refs_of(v, S) for v in C.VERSIONS}
    rels, recs = releases_available()
    vec = {k: per_person(y, p, sex) for k, p in rels.items()}
    # 1. per-fit evaluation and occupation-side reserve (per version), absolute metrics
    table, evals, occres = [], {}, {}
    for key in sorted(rels, key=str):
        recipe, sid, v = key
        ev = eval_version(v, rels[key], S, ctx)
        evals[key] = ev
        occ = R.occupation_reserve({v: ev['utility']})
        occres[key] = occ['per_version'][v]
        pv = occres[key]
        row = {'recipe': recipe, 'subset': sid, 'version': v, 'fraction': int(sid[1:].split('_')[0]),
               'ordering': sid.split('_')[1] if '_' in sid else 'ALL',
               'accuracy': float(vec[key]['accuracy'].mean()), 'nll': float(vec[key]['nll'].mean()), 'brier': float(vec[key]['brier'].mean()),
               'men_brier': float(vec[key]['brier'][vec[key]['men']].mean()), 'women_brier': float(vec[key]['brier'][vec[key]['women']].mean()),
               'occ_point_pass': pv['point_pass'], 'occ_reserve_met': pv['met'], 'min_occ_ratio': pv['min_ratio'], 'binding': pv['binding'],
               'income_reserve_met': pv['income']['income_reserve_met'], 'min_income_ratio': pv['income']['min_income_ratio']}
        for c, r in pv['conditions'].items():
            row[f'{c}|value'] = r['value']; row[f'{c}|own_se'] = r['own_se']; row[f'{c}|margin'] = r['margin']
            row[f'{c}|ratio'] = r['ratio']; row[f'{c}|rho'] = r['rho']; row[f'{c}|point_pass'] = r['point_pass']
        rr = recs.get(key) or {}
        for f_ in ('n_people', 'n_households', 'T_star', 'min_class_count'):
            if f_ in rr:
                row[f_] = rr[f_]
        if 'class_support' in rr:                                  # A1_PO_16 FIT.json (RULES 2 item 5)
            row['min_class_count'] = rr['class_support']['subset_min_class_count']
            row['inner_training_min_class_count'] = rr['class_support']['inner_training_min_class_count']
        table.append(row)
    # 2. paired size contrasts per recipe (loss orientation), summarized by the rules
    def changes_for(recipe):
        out = {}
        for c, (fa, fb) in R.CONTRASTS.items():
            out[c] = {m: {} for m in R.METRICS}
            for v in C.VERSIONS:
                vals = {m: {'values': {}, 'se': {}} for m in R.METRICS}
                for o in C.ORDERINGS:
                    ka = (recipe, 'S100' if fa == 100 else f'S{fa}_{o}', v)
                    kb = (recipe, 'S100' if fb == 100 else f'S{fb}_{o}', v)
                    if ka not in vec or kb not in vec:
                        _nan_cell(vals, o)
                        continue
                    d = _loss_change(vec[ka], vec[kb], unit)
                    for m in R.METRICS:
                        vals[m]['values'][o] = d[m]['value']; vals[m]['se'][o] = d[m]['se']
                for m in R.METRICS:
                    out[c][m][v] = vals[m]
        return out
    size = {recipe: R.size_summary(changes_for(recipe)) for recipe in ('A1', 'TR', 'TT')}
    size_changes = {recipe: changes_for(recipe) for recipe in ('A1', 'TR', 'TT')}
    # 3. control comparison: level (loss(A1) - loss(control)) per size, DiD per contrast
    control = {}
    for ctl in ('TR', 'TT'):
        lev = {}
        for f in (50, 70, 100):
            diffs = {m: {} for m in R.METRICS}
            for v in C.VERSIONS:
                vals = {m: {'values': {}, 'se': {}} for m in R.METRICS}
                for o in (C.ORDERINGS if f != 100 else ('ALL',)):
                    sid = 'S100' if f == 100 else f'S{f}_{o}'
                    ka, kc = ('A1', sid, v), (ctl, sid, v)
                    if ka not in vec or kc not in vec:
                        _nan_cell(vals, o)
                        continue
                    d = _loss_change(vec[kc], vec[ka], unit)              # loss(A1) - loss(control)
                    for m in R.METRICS:
                        vals[m]['values'][o] = d[m]['value']; vals[m]['se'][o] = d[m]['se']
                for m in R.METRICS:
                    diffs[m][v] = vals[m]
            lev[str(f)] = R.control_level(diffs)
        did = {}
        for c, (fa, fb) in R.CONTRASTS.items():
            diffs = {m: {} for m in R.METRICS}
            for v in C.VERSIONS:
                vals = {m: {'values': {}, 'se': {}} for m in R.METRICS}
                for o in C.ORDERINGS:
                    sa = 'S100' if fa == 100 else f'S{fa}_{o}'
                    sb = 'S100' if fb == 100 else f'S{fb}_{o}'
                    keys = [('A1', sa, v), ('A1', sb, v), (ctl, sa, v), (ctl, sb, v)]
                    if not all(k in vec for k in keys):
                        _nan_cell(vals, o)
                        continue
                    a1a, a1b, ca, cb = (vec[k] for k in keys)
                    dd = {'accuracy': (a1b['accuracy'] - a1a['accuracy']) - (cb['accuracy'] - ca['accuracy']),
                          'nll': (a1b['nll'] - a1a['nll']) - (cb['nll'] - ca['nll']),
                          'brier': (a1b['brier'] - a1a['brier']) - (cb['brier'] - ca['brier'])}
                    from ihs.evaluate import cluster_se, subgroup_se
                    for m, arr, mask in (('acc_loss', -dd['accuracy'], None), ('nll', dd['nll'], None), ('brier', dd['brier'], None),
                                         ('men_brier', dd['brier'], a1a['men']), ('women_brier', dd['brier'], a1a['women'])):
                        if mask is None:
                            vals[m]['values'][o] = float(arr.mean()); vals[m]['se'][o] = cluster_se(arr, unit)
                        else:
                            vals[m]['values'][o] = float(arr[mask].mean()); vals[m]['se'][o] = subgroup_se(arr, mask, unit)
                for m in R.METRICS:
                    diffs[m][v] = vals[m]
            did[c] = R.control_did(diffs)
        explains = {m: R.control_explains(size['A1']['contrasts']['50->100'][m]['across'], size[ctl]['contrasts']['50->100'][m]['across'],
                                          did['50->100'][m]['across']) for m in R.METRICS}
        also = {}
        for f in (50, 70, 100):
            cells = [(o, v) for o in (C.ORDERINGS if f != 100 else ('ALL',)) for v in C.VERSIONS]
            sidof = lambda o: 'S100' if f == 100 else f'S{f}_{o}'
            rs = {f'{o}|{v}': occres[('A1', sidof(o), v)]['met'] for o, v in cells if ('A1', sidof(o), v) in occres}
            cs = {f'{o}|{v}': occres[(ctl, sidof(o), v)]['met'] for o, v in cells if (ctl, sidof(o), v) in occres}
            also[str(f)] = (R.control_also_meets_reserve(rs, cs) if len(rs) == len(cells) and len(cs) == len(cells)
                            else 'INCOMPLETE')                    # a missing cell is never dropped (RULES 2)
        control[ctl] = {'level': lev, 'did': did, 'control_explains_50_100': explains, 'control_also_meets_reserve': also}
    # 4. margin pattern, explanation and readings (A1_PO_16)
    st = {'W100': {v: occres[('A1', 'S100', v)]['met'] for v in C.VERSIONS if ('A1', 'S100', v) in occres},
          'S50': {o: {v: occres[('A1', f'S50_{o}', v)]['met'] for v in C.VERSIONS if ('A1', f'S50_{o}', v) in occres} for o in C.ORDERINGS},
          'S70': {o: {v: occres[('A1', f'S70_{o}', v)]['met'] for v in C.VERSIONS if ('A1', f'S70_{o}', v) in occres} for o in C.ORDERINGS}}
    pattern = R.margin_pattern(st)
    pat = pattern['pattern'] if isinstance(pattern, dict) else pattern
    readings = R.interpretation(pat, size['A1']['size_effect'], control['TR']['control_explains_50_100'],
                                {int(f): ok is True for f, ok in control['TR']['control_also_meets_reserve'].items()})
    # 5. reproduction (W100_v vs the source A1_PO_16 S<v>, every role the source stored)
    from ohn import devrun as OD
    repro = {}
    for v in C.VERSIONS:
        fid = f'A1__S100__V{v}'
        if not (C.RUNS / 'SIZE' / fid / 'FIT.json').is_file():
            continue
        rel, rec = SF.release(fid)                                   # RELEASE.npz hash-checked against FIT.json
        src, src_rec, _ = OD.load_occupation('A', 'A1_PO_16', v)     # OCC.npz hash-checked against OCC.json
        recs_v = []
        for role in src:
            if role not in rel:                                      # every role the source stored (RULES 5): a gap is a failure
                recs_v.append({'role': role, 'missing_in_new_release': True, 'sha_equal': False, 'max_abs_diff': float('inf'),
                               'decisions_equal': False, 'tstar_equal': rec.get('T_star') == src_rec.get('T_star')})
                continue
            r = R.reproduction_record(rel[role], src[role], rec.get('T_star'), src_rec.get('T_star'))
            r['role'] = role
            recs_v.append(r)
        repro[v] = recs_v
    reproduction = {'label': R.reproduction(repro), 'records': repro,
                    'margin': R.margin_reproduced({'status': 'OCC_RESERVE_MET_BOTH' if all(st['W100'].get(v) for v in C.VERSIONS)
                                                   and len(st['W100']) == 2 else 'NOT'})}
    # 6. encoding sensitivity (v = 0, S50): loss(FIT-only) - loss(admitted)
    E = {m: {'values': {}, 'se': {}} for m in R.METRICS}
    for o in C.ORDERINGS:
        ka, kb = ('A1_ADM', f'S50_{o}', 0), ('A1', f'S50_{o}', 0)
        if ka in vec and kb in vec:
            d = _loss_change(vec[ka], vec[kb], unit)                   # loss(FIT-only) - loss(admitted)
            for m in R.METRICS:
                E[m]['values'][o] = d[m]['value']; E[m]['se'][o] = d[m]['se']
        else:
            _nan_cell(E, o)
    enc = R.encoding_summary(E)
    # 7. subset-to-subset variation: pairwise differences at a fixed size (A1, both versions)
    pairwise = {}
    for f in (50, 70):
        for v in C.VERSIONS:
            for i, oa in enumerate(C.ORDERINGS):
                for ob in C.ORDERINGS[i + 1:]:
                    ka, kb = ('A1', f'S{f}_{oa}', v), ('A1', f'S{f}_{ob}', v)
                    if ka in vec and kb in vec:
                        pairwise[f'S{f}|V{v}|{ob}-{oa}'] = _loss_change(vec[ka], vec[kb], unit)
    vdiff = {}
    for sid in C.subset_ids():
        ka, kb = ('A1', sid, 0), ('A1', sid, 1)
        if ka in vec and kb in vec:
            vdiff[sid] = _loss_change(vec[ka], vec[kb], unit)            # version 1 - version 0 (procedure repeat, shared rows)
    res = {'schema': 'TSZ1-size-results-v1', 'utc': C.now(), 'fits_evaluated': len(table), 'missing_cells': missing_cells,
           'size_summary': size, 'size_changes': size_changes, 'control': control, 'margin_pattern': pattern,
           'interpretation': readings, 'reproduction': reproduction, 'encoding': {'summary': enc, 'values': E},
           'subset_to_subset_pairwise': pairwise, 'version_difference_same_rows': vdiff,
           'occupation_reserve': {f'{k[0]}|{k[1]}|V{k[2]}': occres[k] for k in occres},
           'income_conditions_H3': {str(v): occres[('A1', 'S100', v)]['income'] for v in C.VERSIONS if ('A1', 'S100', v) in occres},
           'evaluations': {f'{k[0]}|{k[1]}|V{k[2]}': _slim(e) for k, e in evals.items()},
           'records': {f'{k[0]}|{k[1]}|V{k[2]}': {f_: r.get(f_) for f_ in ('n_people', 'n_households', 'T_star', 'min_class_count',
                                                                            'tree_counts', 'dictionary', 'class_support', 'n',
                                                                            'epoch_order')} for k, r in recs.items()},
           'rules_sha256': R.load_rules()['_sha256'], 'cpu_s': time.process_time() - t0,
           'note': 'development evidence on adaptively used MODEL_SELECT; orderings are nested draws from one pool; versions share rows'}
    C.write_json_x(OUT / 'SIZE_RESULTS.json', res)
    keys = sorted({k for r in table for k in r})
    lead = ['recipe', 'subset', 'version', 'fraction', 'ordering', 'n_people', 'n_households', 'T_star', 'accuracy', 'nll', 'brier',
            'men_brier', 'women_brier', 'occ_point_pass', 'occ_reserve_met', 'min_occ_ratio', 'binding', 'income_reserve_met',
            'min_income_ratio']
    cols = [k for k in lead if k in keys] + [k for k in keys if k not in lead]
    with open(OUT / 'SIZE_TABLE.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(table)
    print(json.dumps({'fits': len(table), 'size_effect': size['A1']['size_effect'], 'pattern': pat,
                      'explanation': readings['size_explanation'], 'reproduction': reproduction['label'].get('label'),
                      'cpu_s': round(time.process_time() - t0, 1)}))
    return res


REFERENCES = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')
COND_OUT = C.RESULTS / 'conditional'


def _selection_record(v):
    p = COND_OUT / f'SELECTION_V{v}.json'
    if not C.committed(p):
        raise PermissionError(f'{p.name} must be committed before any MODEL_SELECT read of a corrected release')
    return json.loads(p.read_text()), C.sha(p)


def _point_and_reserve(ev, v):
    """Point status of the complete release (12 clauses + 2 guards of version v) and the occupation-side reserve (RULES 1)."""
    from . import rules as R
    u = ev['utility']
    pts = [float(c['value']) <= float(c['cap']) for c in u['clauses']] + [float(g['value']) <= float(g['cap']) for g in u['guards']]
    occ = R.occupation_reserve({v: u})['per_version'][v]
    return {'point_pass': bool(all(pts)) and len(pts) == 14, 'occ_reserve_met': occ['met'], 'occ_point_pass': occ['point_pass'],
            'min_occ_ratio': occ['min_ratio'], 'binding': occ['binding'], 'failed_occ_reserve': occ['failed'],
            'income': occ['income'], 'occ_conditions': occ['conditions']}


def conditional_cmd():
    """MODEL_SELECT evaluation of the conditional stage, after the committed selections (RULES.md sections 8.3 last bullet, 10,
    11): the actual selected releases (identity included), matched continuations, the witness, LOC at PAIR's strength
    (descriptive) and the inherited references; realized increments at every strength (recorded, never gating) and realized minus
    the nested prediction."""
    from . import correct as CR
    from . import rules as R
    from cts import trackf as TF
    from ohn import devreaders as ODR, rules as OR
    t0 = time.process_time()
    out = COND_OUT / 'COND_UTILITY.json'
    if out.exists():
        raise PermissionError('COND_UTILITY.json exists (write-once)')
    sels = {v: _selection_record(v) for v in C.VERSIONS}
    S = ms('benchmark: conditional stage on MODEL_SELECT after the committed strength selections: utility, reserve and realized '
           'costs of the selected corrected releases, matched continuations, witness and inherited references (never a selection input)')
    y, sex, unit = S['occupation'], S['sex'], S['unit']
    ctx = {v: refs_of(v, S) for v in C.VERSIONS}
    res = {'releases': {}, 'realized_costs': {}, 'realized_minus_predicted': {}, 'references': {}}
    for v in C.VERSIONS:
        sel, sel_sha = sels[v]
        s_loc, s_pair = float(sel['selected']['LOC']), float(sel['selected']['PAIR'])
        W, _ = a1_release(f'A1__S100__V{v}')
        named = {'W': ('W', 0.0), 'LOC_SEL': ('LOC', s_loc), 'PAIR_SEL': ('PAIR', s_pair), 'CONT_AT_LOC': ('CONT', s_loc),
                 'CONT_AT_PAIR': ('CONT', s_pair), 'LOC_AT_PAIR': ('LOC', s_pair)}
        for name, (arm, s) in named.items():
            occ = W if (arm == 'W' or s == 0.0) else CR.load_release(v, arm, s, 'MODEL_SELECT')
            ev = eval_version(v, occ, S, ctx)
            rec = {'arm': arm, 'strength': s, 'identity_equals_W': bool(np.array_equal(occ, W)), **_point_and_reserve(ev, v),
                   'evaluation': _slim(ev), 'occ_sha256': C.arr_sha(occ)}
            res['releases'].setdefault(name, {})[str(v)] = rec
        # realized increments at every registered strength (never gating) and realized minus the nested prediction
        for arm in ('CONT', 'LOC', 'PAIR'):
            nested_p = C.RUNS / 'NESTED' / f'V{v}' / f'NESTED_{arm}.json'
            pred = json.loads(nested_p.read_text())['per_strength'] if nested_p.is_file() else None
            rc, rmp = {}, {}
            for s in R.STRENGTHS:
                p = W if s == 0.0 else CR.load_release(v, arm, s, 'MODEL_SELECT')
                cs = TF.cost_summary(p, W, y, sex, unit)
                rc[CR.skey(s)] = cs
                if pred is not None and CR.skey(s) in pred:
                    rmp[CR.skey(s)] = {j: {'realized': cs['stats'][j], 'predicted': pred[CR.skey(s)]['stats'][j],
                                           'difference': cs['stats'][j] - pred[CR.skey(s)]['stats'][j],
                                           'realized_se': cs['se'][j], 'predicted_se': pred[CR.skey(s)]['se'][j]}
                                       for j in ('men_brier', 'women_brier', 'brier', 'nll', 'acc_loss')}
            res['realized_costs'].setdefault(arm, {})[str(v)] = rc
            res['realized_minus_predicted'].setdefault(arm, {})[str(v)] = rmp
        res.setdefault('selection_sha256', {})[str(v)] = sel_sha
        res.setdefault('selected', {})[str(v)] = {'LOC': s_loc, 'PAIR': s_pair}
        # inherited references (their own complete releases, own income)
        for r in REFERENCES:
            rel = ODR.load_member({'type': 'reference', 'name': r}, v)
            from ohn import evaluate as OE
            refs, others, _ = ctx[v]
            ev = OE.complete_eval(v, rel['income'], rel['occupation'], S, refs, others)
            whole = OR.reserve({v: ev['utility']})
            occ = R.occupation_reserve({v: ev['utility']})['per_version'][v]
            pts = [float(c['value']) <= float(c['cap']) for c in ev['utility']['clauses']] + \
                  [float(g['value']) <= float(g['cap']) for g in ev['utility']['guards']]
            res['references'].setdefault(r, {})[str(v)] = {
                'point_pass_12_plus_2': bool(all(pts)), 'whole_release_reserve_met': whole['per_seed'][v]['status'] == OR.MET,
                'min_whole_ratio': whole['min_ratio'], 'whole_binding': whole['binding_condition'],
                'occ_reserve_met': occ['met'], 'min_occ_ratio': occ['min_ratio'], 'occ_binding': occ['binding'],
                'evaluation': _slim(ev)}
    # whole-release reserve of every candidate release (28-condition reserve per version, for the equal-reserve table)
    for name, per in res['releases'].items():
        for v in C.VERSIONS:
            u = per[str(v)]['evaluation']['utility']
            w = OR.reserve({v: u})
            per[str(v)]['whole_release_reserve_met'] = w['per_seed'][v]['status'] == OR.MET
            per[str(v)]['min_whole_ratio'] = w['min_ratio']
    rec = {'schema': 'TSZ1-conditional-utility-v1', 'utc': C.now(), **res, 'rules_sha256': R.load_rules()['_sha256'],
           'cpu_s': time.process_time() - t0, 'unit_id': __import__('os').environ.get(C.UNIT_ENV),
           'note': 'development evidence on adaptively used MODEL_SELECT; realized costs are recorded after selection and never '
                   'enter it; the two-SE reserve is a development heuristic reported separately from the original point limits'}
    C.write_json_x(out, rec)
    print(json.dumps({'selected': res['selected'],
                      'point_and_reserve': {n: {v: (p['point_pass'], p['occ_reserve_met'], round(p['min_occ_ratio'], 2))
                                                for v, p in per.items()} for n, per in res['releases'].items()}}, default=str))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('trigger')
    sub.add_parser('size')
    sub.add_parser('conditional')
    a = ap.parse_args(argv)
    if a.cmd == 'conditional':
        conditional_cmd(); return 0
    if a.cmd == 'trigger':
        trigger_cmd(); return 0
    if a.cmd == 'size':
        size_cmd(); return 0
    if a.cmd == 'selftest':
        rng = np.random.default_rng(1)
        n = 400; unit = np.arange(n) // 2; sex = rng.integers(0, 2, n); y = rng.integers(0, 5, n)
        p = rng.dirichlet(np.ones(5), n)
        A = per_person(y, p, sex)
        d = paired_diff(A, A, unit)
        ok = all(abs(v['value']) == 0.0 for v in d.values())
        print(json.dumps({'ok': ok, 'zero_difference_of_identical_outputs': ok}))
        return 0 if ok else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
