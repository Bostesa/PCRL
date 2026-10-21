"""Stages of the PCRL finite-output solver-adequacy campaign (card cards/FA_FSA_v1.md; queue units only).

    python3 -m fsa.jobqueue run --unit FSA_<STAGE>_V<v>_V<n> --stage <stage> --rss-gib 3 -- python3 -m fsa.fa.solve <stage> --version <v>

Stages:
- replay  (P1 replay; MODEL_FIT income, occupation, SEX as in the original P1): the previous campaign's nested estimator
          frc.fa.route.nested_c, called UNCHANGED with the recorded inputs of results/pcrl_finite_routing_comparison_v1/fa/
          SELECTION_V<v>.json (plan, construction B1, capacity 16 x 64, T_fixed, 5 outer folds), saves the per-person held-out
          outputs (private) and checks the recorded nested means; the independent verifier recomputes spends, clustered SEs,
          the selection and F4 from the saved per-person file with its own code.
- audit   (cached MODEL_FIT out-of-fold statistics; no label file is read): the objective audit of the saved frc releases and the
          original chain (traced): monotone acceptance, the refresh step's effect, the routing proposal score against the
          acceptance objective along the tilt line, the emitted B1 tables against the search prototypes, runtime routing
          consistency.
- source  (cached statistics; no label file is read): the source-specific solver comparison: the original, SV1 and SV1_first
          chains of PRC and HQ over the registered strength grid on the full MODEL_FIT pool (objective, partitions, re-routing,
          fitted MI, emitted-table training cross-entropy, effort).
- nested  (MODEL_FIT income, occupation, SEX): the nested estimator with the original and SV1 chains of PRC and HQ in the SAME
          outer folds and teacher fits (per-person held-out outputs saved; spends and held-out LOOKUP AUCs recorded).
Every stage refuses an existing output (write-once records).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from nrm.fa import prc as P
from frc.fa import route as FR                 # the previous campaign's family module: pure computation, called unchanged
from fsa import common as C
from fsa.fa import solver as S

CARD = 'results/pcrl_finite_solver_adequacy_v1/cards/FA_FSA_v2.md'
GRID = tuple(P.LAMBDA_GRID)                   # (0.25, 0.5, 1, 2, 4, 8): the registered strength grid, unchanged
FAMS = {'PRC': 'PRC_S2', 'HQ': 'HQ_S2'}         # route parameters of each family (nrm.fa.prc.route_params)


def _utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _admission():
    return json.loads((C.RESULTS / 'SOURCE_ADMISSION.json').read_text())


def _write_once(path, obj):
    path = Path(path)
    if path.exists():
        raise PermissionError(f'{path.name} exists (write-once)')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'x') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return path


def _priv(name):
    d = C.PRIV / 'solver'
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d / name


# ---------------------------------------------------------------- read-only sources (pinned in SOURCE_ADMISSION.json)
def frc_model(cfg, version):
    pin = _admission()['frc_private_artefacts'][f'frc/runs/FA/{cfg}/V{int(version)}']
    d = C.FRC_RUNS / 'FA' / cfg / f'V{int(version)}'
    if _sha(d / 'MODEL.pkl') != pin['MODEL.pkl'] or _sha(d / 'RELEASE.npz') != pin['RELEASE.npz']:
        raise PermissionError(f'frc {cfg} V{version} differs from its pin')
    return FR.load_model(cfg, version)


def frc_sexside(version, row_sha):
    pin = _admission()['frc_private_artefacts'][f'frc/fa_cache/V{int(version)}/SEXSIDE.pkl']['sha256']
    p = C.FRC_FA_CACHE / f'V{int(version)}' / 'SEXSIDE.pkl'
    buf = p.read_bytes()
    if hashlib.sha256(buf).hexdigest() != pin:
        raise PermissionError(f'frc SEXSIDE V{version} differs from its pin')
    obj = pickle.loads(buf)
    if obj['row_id_sha256'] != row_sha:
        raise PermissionError('SEXSIDE was fitted on other rows')
    return obj


def frc_record(rel):
    pin = _admission()['frc_private_artefacts'][f'frc/results/{rel}']['sha256']
    p = C.FRC_RESULTS / rel
    if _sha(p) != pin:
        raise PermissionError(f'frc record {rel} differs from its pin')
    return json.loads(p.read_text())


def statistics(version, mf):
    """Full-pool out-of-fold statistics of MODEL_FIT (cached, label-derived): NRM1 task teachers + P1 SEX-side teachers."""
    from fsa.fa import route as RT
    Z, _, unit, names, row_sha = RT._prep(mf)
    task = RT.source_task(version, row_sha)
    ss = frc_sexside(version, row_sha)
    E = {'qI': task['oof']['inc'], 'qO': task['oof']['occ'], 'Z': Z}
    E.update(P.embed_oof({'oof': {'inc': E['qI'], 'occ': E['qO'], 'sex': ss['oof']['sex'], 'occ_s': ss['oof']['occ_s']}}))
    return E, Z, unit, names, row_sha


def twin_state(version, E):
    """The repaired twin of the qualification decision (frc TWB_C64, B1) and its partition on E."""
    dec = frc_record('decisions/QUAL_DECISION.json')
    if dec.get('outcome') != 'QUALIFIED' or dec.get('twin', {}).get('config') != 'TWB_C64':
        raise PermissionError('unexpected source qualification decision')
    tb = frc_model('TWB_C64', version)['book']
    k0, l0 = P.route(E, tb)
    return tb, k0, l0


# ---------------------------------------------------------------- shared metrics of one partition
def partition_summary(E, book, k, l, k0, l0, tables=None):
    KO = book['CO'].shape[0]
    out = {'objective': P.objective(E, book, k, l), 'partition_sha256': P.array_sha(k, l),
           'rerouted_vs_twin': float(((k != k0) | (l != l0)).mean()), 'income_rerouted': float((k != k0).mean()),
           'occupation_rerouted': float((l != l0).mean()), 'tilt': float(book.get('tilt', 0.0)),
           'occupied_pair_cells': int(len(np.unique(k * KO + l)))}
    if 'pi' in E:
        out['fitted_pi_pair_mi'] = P.plugin_mi(k * KO + l, E['pi'])
        out['fitted_pi_income_mi'] = P.plugin_mi(k, E['pi'])
        out['fitted_pi_occupation_mi'] = P.plugin_mi(l, E['pi'])
    n = len(k)
    out['proto_ce_income'] = float(-(E['qI'] * np.log(book['CI'][k])).sum() / n)
    out['proto_ce_occupation'] = float(-(E['qO'] * np.log(book['CO'][l])).sum() / n)
    if tables is not None:
        out['emitted_ce_income'] = float(-(E['qI'] * np.log(tables['I'][k])).sum() / n)
        out['emitted_ce_occupation'] = float(-(E['qO'] * np.log(tables['O'][l])).sum() / n)
        out['emitted_minus_proto_max_abs'] = float(max(np.abs(tables['I'] - book['CI']).max(), np.abs(tables['O'] - book['CO']).max()))
        if 'qO0' in E:
            g = {}
            for nm, CO in (('proto', book['CO']), ('emitted', tables['O'])):
                gg = 0.0
                for key, w in (('qO0', (1 - E['pi']) / book['p0']), ('qO1', E['pi'] / book['p1'])) if 'p0' in book else ():
                    gg = gg + w * ((E[key] - CO[l]) ** 2).sum(1)
                g[nm] = float(np.mean(gg)) if not np.isscalar(gg) else float('nan')
            out['guard_proto'], out['guard_emitted'] = g['proto'], g['emitted']
    return out


# ---------------------------------------------------------------- stage: P1 replay
def stage_replay(version, mf_lab, log=print):
    t0 = time.process_time()
    rec_path = C.RESULTS / 'replay' / f'P1_REPLAY_V{int(version)}.json'
    if rec_path.exists():
        raise PermissionError(f'{rec_path.name} exists')
    sel = frc_record(f'fa/SELECTION_V{int(version)}.json')
    Z, lab, unit, names, row_sha = FR._prep(mf_lab)
    if sorted(lab) != ['income', 'occupation', 'sex']:
        raise PermissionError('the P1 replay reads income, occupation and SEX of MODEL_FIT')
    plan = FR._plan(sel['capacity_id'], tuple(map(tuple, sel['ladder'])))
    if sorted(plan) != sorted(sel['nested_means']):
        raise AssertionError('the replay plan differs from the recorded P1 plan')
    T_fixed = sel['T_fixed']
    nest = FR.nested_c(Z, lab, unit, version, plan, sel['construction'], T_fixed=T_fixed, capacity=tuple(sel['capacity']),
                       outer=int(sel['outer_folds']), log=log)
    diffs = {}
    for name, d in sel['nested_means'].items():
        for key, v in d.items():
            got = nest[name].get(key)
            if isinstance(v, (int, float)) and got is not None and np.isfinite(v):
                diffs[f'{name}|{key}'] = abs(float(got) - float(v))
    arrays = {'unit': np.asarray(unit), 'sex': lab['sex'], 'income': lab['income'], 'occupation': lab['occupation']}
    for name in plan:
        for key, arr in nest[name]['_per_person'].items():
            arrays[f'{name}|{key}'] = np.asarray(arr)
    npz = _priv(f'P1_REPLAY_V{int(version)}.npz')
    if npz.exists():
        raise PermissionError(f'{npz.name} exists')
    np.savez_compressed(npz, **arrays)
    # the six P1 fields, computed exactly as frc.fa.route.stage_paths computes them (the verifier recomputes independently)
    B = sel['budgets']
    sel2, audit = {}, {}
    for fam, bud in (('PRC', B), ('HQ', {'S2': B['S2']}), ('SHUF', {'S2': B['S2']}), ('ORD', B)):
        cands = [n for n in plan if n.startswith(fam + '@')]
        chosen, aud = P.select_path(nest, lab, unit, cands, budgets=bud)
        sel2[fam] = chosen
        audit[fam] = {c: {k2: list(v2) for k2, v2 in a.items()} for c, a in aud.items()}
    br = FR.brackets_for(nest, plan, sel2)
    look = FR.nested_lookup(nest, lab, unit, version, [n for n in plan if n != 'PRC_TWIN'])
    s2 = sel2['PRC'].get('S2')
    f4 = None
    if s2 is not None and s2 in look:
        r = look[s2]
        f4 = {'entry': s2, 'diff_twin_minus_prc': r['diff_twin_minus_entry'], 'se_boot': r['se_boot'],
              'pass_2se': bool(r['diff_twin_minus_entry'] - 2 * r['se_boot'] > 0)}
    prc_sha = _sha(C.ROOT / 'nrm' / 'fa' / 'prc.py')
    route_sha = _sha(C.ROOT / 'frc' / 'fa' / 'route.py')
    rec = {'stage': 'P1_replay', 'version': int(version), 'utc': _utc(), 'card': CARD,
           'replayed_record': f'results/pcrl_finite_routing_comparison_v1/fa/SELECTION_V{int(version)}.json',
           'code': 'frc.fa.route.nested_c, select_path, brackets_for, nested_lookup (unchanged; the selection loop of stage_paths)',
           'nested_means': {n: {k2: v2 for k2, v2 in d.items() if k2 != '_per_person'} for n, d in nest.items()},
           'audit_spend_over_twin': audit, 'brackets': br, 'nested_lookup_pair': look, 'F4': f4, 'selection': sel2,
           'budgets': B, 'T_fixed': T_fixed, 'outer_folds': int(sel['outer_folds']), 'capacity': sel['capacity'],
           'construction': sel['construction'], 'ladder': sel['ladder'],
           'estimator_identity': {'nrm/fa/prc.py': prc_sha, 'nrm_prc_expected': '1192f34dc931fee3426be3423b807296b7fd79c47b098f35ffbea3f1f7ad210a',
                                  'frc/fa/route.py': route_sha, 'frc_route_expected': '933d0d6f9d8c97bf04bfd075ac4b1db166a81ae5921524748bb74bb414eb5988'},
           'estimator_unchanged': prc_sha == '1192f34dc931fee3426be3423b807296b7fd79c47b098f35ffbea3f1f7ad210a'
           and route_sha == '933d0d6f9d8c97bf04bfd075ac4b1db166a81ae5921524748bb74bb414eb5988',
           'per_person_outputs_saved': True,
           'max_abs_diff_recorded_nested_means': max(diffs.values()) if diffs else None, 'n_compared': len(diffs),
           'worst': sorted(diffs.items(), key=lambda x: -x[1])[:5],
           'selection_equal_recorded': sel2 == sel['selection'], 'brackets_equal_recorded': br == sel['brackets'],
           'per_person_file_sha256': _sha(npz), 'per_person_keys': sorted(arrays),
           'cpu_s': round(time.process_time() - t0, 1)}
    _write_once(rec_path, rec)
    log(json.dumps({k: rec[k] for k in ('max_abs_diff_recorded_nested_means', 'selection_equal_recorded', 'brackets_equal_recorded', 'cpu_s')}))
    return rec


# ---------------------------------------------------------------- stage: objective audit (cached statistics)
def proposal_vs_objective(E, nb, lam):
    """Along the tilt line of the fixed refreshed book: per interval, the proposal score (mean A + t D at the chosen cells and its
    dependence part mean D) against the exact objective and its dependence part (objective minus the prototype cross-entropy)."""
    A, D = S.cost_terms(E, nb)
    c0, ev_t, ev_i, ev_c = S.envelope(A, D, lam)
    KO = nb['CO'].shape[0]
    inc = S.IncrementalObjective(E, nb, c0 // KO, c0 % KO)
    inc.init_mi()
    cur = c0.copy()
    idx = np.arange(len(c0))
    sumA, sumD = float(A[idx, cur].sum()), float(D[idx, cur].sum())
    rows, lo, pos = [], 0.0, 0
    n = len(c0)

    def snap(lo_, hi_):
        J = inc.value_fast()
        rows.append({'t_lo': lo_, 't_hi': hi_, 'objective': J, 'objective_dependence': J - inc.S_ce / n,
                     'proposal_mean_A': sumA / n, 'proposal_mean_D': sumD / n})
    for b in np.unique(ev_t):
        if b > lo or not rows:
            snap(lo, float(b))
        while pos < len(ev_t) and ev_t[pos] == b:
            i, c = int(ev_i[pos]), int(ev_c[pos])
            sumA += A[i, c] - A[i, cur[i]]
            sumD += D[i, c] - D[i, cur[i]]
            cur[i] = c
            inc.move(i, c)
            pos += 1
        lo = float(b)
    snap(lo, float(lam))
    from scipy.stats import spearmanr
    dep_p = np.array([r['proposal_mean_D'] for r in rows])
    dep_o = np.array([r['objective_dependence'] for r in rows])
    tot_o = np.array([r['objective'] for r in rows])
    tot_p = np.array([r['proposal_mean_A'] + 0.5 * (r['t_lo'] + r['t_hi']) * r['proposal_mean_D'] for r in rows])
    best = int(np.argmin(tot_o))
    return {'n_intervals': len(rows), 'n_events': int(len(ev_t)),
            'spearman_dependence_proposal_vs_objective': float(spearmanr(dep_p, dep_o)[0]) if len(rows) > 2 else None,
            'spearman_total_proposal_vs_objective': float(spearmanr(tot_p, tot_o)[0]) if len(rows) > 2 else None,
            'best_interval': rows[best], 'objective_at_t0': rows[0]['objective'],
            'intervals_sample': rows[:: max(1, len(rows) // 40)]}


def stage_audit(version, mf, log=print):
    t0 = time.process_time()
    rec_path = C.RESULTS / 'audit' / f'OBJECTIVE_AUDIT_V{int(version)}.json'
    if rec_path.exists():
        raise PermissionError(f'{rec_path.name} exists')
    E, Z, unit, names, row_sha = statistics(version, mf)
    tb, k0, l0 = twin_state(version, E)
    out = {'stage': 'objective_audit', 'version': int(version), 'utc': _utc(), 'card': CARD, 'labels_read': False,
           'statistics': 'cached MODEL_FIT out-of-fold teacher predictions (label-derived): NRM1 task teachers, P1 SEX-side teachers'}
    # 1. the original chain, traced, reproduces the saved PRC_S2 / HQ_S2 partitions; monotone acceptance; the refresh effect
    for fam, cfg in FAMS.items():
        rp = P.route_params(cfg, version=version)
        tr = {}
        path = S.chain(E, tb, k0, l0, list(GRID), rp, solver='original_traced', trace=tr)
        saved = frc_model(cfg, version)
        ks, ls = P.route(E, saved['book'])
        b8, k8, l8 = path[8.0]
        steps = [s for lam in GRID for s in tr[lam]['steps']]
        acc = [s for s in steps if s.get('accepted_tilt') is not None]
        out[f'{fam}_chain'] = {
            'reproduces_saved_partition': bool(np.array_equal(k8, ks) and np.array_equal(l8, ls)),
            'per_strength': {str(lam): {'objective': tr[lam]['objective'], 'tilt': tr[lam]['tilt'], 'reset_to_twin': tr[lam]['reset_to_twin'],
                                        'steps': len(tr[lam]['steps']),
                                        'accepted': sum(1 for s in tr[lam]['steps'] if s.get('accepted_tilt') is not None),
                                        'tried_tilts_last_step': [(round(x['tilt'], 6), x['J'], x['moved'])
                                                                  for x in (tr[lam]['steps'][-1]['tried'] if tr[lam]['steps'] else [])]}
                             for lam in GRID},
            'accepted_objective_strictly_decreasing': all(s['J_after'] < s['J_before'] for s in acc),
            'refresh_delta_max': max((s['J_refresh_same_partition'] - s['J_before'] for s in steps), default=None),
            'refresh_raised_objective_steps': sum(1 for s in steps if s['J_refresh_same_partition'] > s['J_before'] + 1e-12),
            'n_steps': len(steps), 'n_accepted': len(acc)}
        # 2. proposal score vs acceptance objective along the tilt line of the refreshed final book (lam = 8)
        nb = P.update_book(E, dict(b8), k8, l8)
        out[f'{fam}_tilt_line_at_8'] = proposal_vs_objective(E, nb, 8.0)
        out[f'{fam}_tilt_line_at_8']['current_objective'] = P.objective(E, b8, k8, l8)
        # 3. emitted tables vs prototypes (the saved release's B1 tables)
        out[f'{fam}_emitted_vs_proto'] = partition_summary(E, b8, k8, l8, k0, l0, tables=saved['dec'])
    out['TWIN_emitted_vs_proto'] = partition_summary(E, tb, k0, l0, k0, l0, tables=frc_model('TWB_C64', version)['dec'])
    # 4. runtime routing consistency: full-pool teachers on MODEL_FIT rows (in-sample) vs the out-of-fold cells
    for cfg in ('TWB_C64', 'PRC_S2', 'HQ_S2'):
        M = frc_model(cfg, version)
        Ef = P.embed(M['teachers'], Z)
        kf, lf = FR.route_any(Ef, M['book'])
        ko, lo_ = FR.route_any(E, M['book'])
        out[f'runtime_{cfg}'] = {'model_fit_rows_joint_cell_agreement_insample_vs_oof': float(((kf == ko) & (lf == lo_)).mean()),
                                 'income_agreement': float((kf == ko).mean()), 'occupation_agreement': float((lf == lo_).mean())}
    # 5. expressivity (statistical reviewer's challenge 6): do the deployed task scores identify the six-field support states?
    M = frc_model('TWB_C64', version)
    Ef = P.embed(M['teachers'], Z)
    rows = np.ascontiguousarray(Z)
    _, xs_inv = np.unique(rows, axis=0, return_inverse=True)
    q = np.round(np.column_stack([Ef['qI'], Ef['qO']]), 12)
    _, q_inv = np.unique(q, axis=0, return_inverse=True)
    n_states = int(xs_inv.max() + 1)
    n_q = int(q_inv.max() + 1)
    state_q = np.zeros(n_states, dtype=np.int64)
    state_q[xs_inv.ravel()] = q_inv.ravel()
    q_states = np.bincount(state_q, minlength=n_q)          # support states per distinct q value
    shared = q_states[q_inv.ravel()] > 1
    out['expressivity'] = {'people': int(len(Z)), 'distinct_support_states': n_states, 'distinct_task_score_vectors': n_q,
                           'share_of_people_whose_task_scores_are_shared_with_another_support_state': float(shared.mean()),
                           'note': 'support state = distinct compact six-field encoding; task scores = the deployed full-pool task '
                                   'teachers (rounded to 1e-12); where the share is about 0, an unrestricted task-score mapping can '
                                   'express any partition PRC can produce on this support'}
    out['cpu_s'] = round(time.process_time() - t0, 1)
    _write_once(rec_path, out)
    log(json.dumps({'version': version, 'cpu_s': out['cpu_s']}))
    return out


# ---------------------------------------------------------------- stage: source-specific solver comparison (cached statistics)
def stage_source(version, mf, solvers=('original', 'SV1', 'SV1_first'), log=print):
    t0 = time.process_time()
    rec_path = C.RESULTS / 'source' / f'SOURCE_CHECK_V{int(version)}.json'
    if rec_path.exists():
        raise PermissionError(f'{rec_path.name} exists')
    E, Z, unit, names, row_sha = statistics(version, mf)
    tb, k0, l0 = twin_state(version, E)
    out = {'stage': 'source_specific_solver_comparison', 'version': int(version), 'utc': _utc(), 'card': CARD, 'labels_read': False,
           'statistics': 'cached MODEL_FIT out-of-fold teacher predictions (label-derived)', 'grid': list(GRID), 'runs': {}}
    keep, books = {}, {}
    for fam, cfg in FAMS.items():
        rp = P.route_params(cfg, version=version)
        for solver in solvers:
            c0 = time.process_time()
            tr = {}
            path = S.chain(E, tb, k0, l0, list(GRID), rp, solver=solver if solver != 'original' else 'original_traced', trace=tr)
            cpu = time.process_time() - c0
            per = {}
            for lam in GRID:
                b, k, l = path[lam]
                tabs = FR.tables('B1', b, k, l, E)[0]
                per[str(lam)] = partition_summary(E, b, k, l, k0, l0, tables=tabs)
                st = tr[lam]['steps']
                per[str(lam)]['lloyd_steps'] = len(st)
                per[str(lam)]['route_evaluations'] = sum(len(s['tried']) if 'tried' in s else s.get('route_evaluations', 0) for s in st)
                per[str(lam)]['line_search_events'] = sum(sum(s['n_events'].values()) if isinstance(s.get('n_events'), dict)
                                                          else s.get('n_events', 0) for s in st)
                per[str(lam)]['accepted_steps'] = sum(1 for s in st if s.get('accepted_tilt') is not None)
                per[str(lam)]['stopping_reason'] = ('no improving proposal' if st and st[-1].get('accepted_tilt') is None
                                                    else 'path_steps reached' if st else 'no step')
                per[str(lam)]['reset_to_twin'] = tr[lam]['reset_to_twin']
                per[str(lam)]['rule_consistent'] = bool(np.array_equal(FR.route_any(E, b)[0], k) and np.array_equal(FR.route_any(E, b)[1], l))
                keep[(fam, solver, lam)] = (k, l)
                books[f'{fam}|{solver}|{lam}'] = b
            out['runs'][f'{fam}|{solver}'] = {'cpu_s': round(cpu, 1), 'per_strength': per}
    # paired summaries: SV1 against the original at every strength
    cmp = {}
    for fam in FAMS:
        for lam in GRID:
            o, s = out['runs'][f'{fam}|original']['per_strength'][str(lam)], out['runs'][f'{fam}|SV1']['per_strength'][str(lam)]
            cmp[f'{fam}|{lam}'] = {'objective_sv1_minus_original': s['objective'] - o['objective'],
                                   'same_partition': s['partition_sha256'] == o['partition_sha256'],
                                   'fitted_pair_mi_sv1_minus_original': s.get('fitted_pi_pair_mi', 0) - o.get('fitted_pi_pair_mi', 0),
                                   'emitted_ce_total_sv1_minus_original': (s['emitted_ce_income'] + s['emitted_ce_occupation'])
                                   - (o['emitted_ce_income'] + o['emitted_ce_occupation']),
                                   'rerouted_sv1': s['rerouted_vs_twin'], 'rerouted_original': o['rerouted_vs_twin']}
    out['sv1_vs_original'] = cmp
    out['cpu_s'] = round(time.process_time() - t0, 1)
    _write_once(rec_path, out)
    bp = _priv(f'SOURCE_BOOKS_V{int(version)}.pkl')
    with open(bp, 'xb') as f:
        pickle.dump(books, f, protocol=5)
    np.savez_compressed(_priv(f'SOURCE_PARTITIONS_V{int(version)}.npz'),
                        **{f'{f}|{s}|{lam}|k': v[0] for (f, s, lam), v in keep.items()},
                        **{f'{f}|{s}|{lam}|l': v[1] for (f, s, lam), v in keep.items()})
    log(json.dumps({'version': version, 'cpu_s': out['cpu_s'], 'cmp': {k: round(v['objective_sv1_minus_original'], 6) for k, v in cmp.items()}}))
    return out


# ---------------------------------------------------------------- stage: nested estimator with both solvers in the same folds
def fit_pipeline_s(Z, lab, unit, version, plan, construction='B1', T_fixed=None, capacity=(16, 64), log=None):
    """frc.fa.route.fit_pipeline_c with the chain's solver taken from each plan entry: ('prc', cfg, lam, override, solver)."""
    tch = P.fit_teachers(Z, lab, unit, version, P.HGB, ('inc', 'occ', 'sex', 'occ_s'), T_fixed=T_fixed, log=log)
    E = P.embed_oof(tch)
    E['Z'] = Z
    rp0 = P.route_params('PRC_TWIN')
    KI, KO = capacity
    tb, k0, l0 = P.twin_book(E, KI, KO, version, rp0)
    state = {'version': int(version), 'T': tch['T'], 'models': tch['models'], 'books': {}, 'dec': {}, 'fit_cells': {},
             'capacity': [int(KI), int(KO)], 'E_fit': E, 'twin_fit': (tb, k0, l0), 'construction': construction}
    chains = {}
    for p in plan.values():
        if p[0] == 'prc':
            chains.setdefault((p[1], p[4]), set()).add(float(p[2]))
    paths = {}
    for (cfg, solver), ls in chains.items():
        Ec, rp, extra = FR.chain_inputs(cfg, {}, E, version)
        paths[(cfg, solver)] = (S.chain(Ec, tb, k0, l0, sorted(ls), rp, solver=solver), extra)
    for name, p in plan.items():
        if p[0] == 'twin':
            book, k, l = tb, k0, l0
        else:
            chain, extra = paths[(p[1], p[4])]
            book, k, l = chain[float(p[2])]
            book = {**book, **extra}
        state['books'][name] = book
        state['dec'][name] = FR.tables(construction, book, k, l, E, lab)[0]
        state['fit_cells'][name] = (k, l)
    return state


def nested_s(Z, lab, unit, version, plan, construction='B1', T_fixed=None, capacity=(16, 64), outer=P.N_OUTER, log=None):
    """frc.fa.route.nested_c with fit_pipeline_s (same outer folds NRM_FA_ncv_S<v>; per-person held-out outputs)."""
    of = P.household_folds(unit, outer, FR.OUTER_TAG.format(v=int(version)))
    store = {name: {} for name in plan}
    for o in range(outer):
        tr, te = of != o, of == o
        st = fit_pipeline_s(Z[tr], {k: v[tr] for k, v in lab.items()}, unit[tr], version, plan, construction, T_fixed=T_fixed,
                            capacity=capacity, log=log)
        E = P.embed(st['models'], Z[te])
        for name in plan:
            rel, k, l = FR.apply_c(st, name, E)
            m = P.person_metrics(rel, {k2: v[te] for k2, v in lab.items()})
            ktr, ltr = st['fit_cells'][name]
            KO = st['books'][name]['CO'].shape[0]
            vtr, vte = P.views(ktr, ltr, KO), P.views(k, l, KO)
            for vw in vtr:
                m[f'look_{vw}'] = P.lookup_scores(vtr[vw], lab['sex'][tr], vte[vw])
            m['fit_pi_pair_mi'] = np.full(int(te.sum()), P.plugin_mi(vtr['pair'], st['E_fit']['pi']))
            for key, v in m.items():
                store[name].setdefault(key, np.zeros(len(Z)))[te] = v
    out = {}
    for name in plan:
        d = store[name]
        out[name] = {key: float(v.mean()) for key, v in d.items() if not key.startswith('look_')}
        for vw in ('pair', 'income', 'occupation'):
            out[name][f'auc_{vw}'] = P.auc(d[f'look_{vw}'], lab['sex'])
        sx = lab['sex'].astype(bool)
        out[name]['occ_brier_men'] = float(d['occ_brier'][~sx].mean())
        out[name]['occ_brier_women'] = float(d['occ_brier'][sx].mean())
        out[name]['_per_person'] = d
    return out


def nested_plan(solvers=('original', 'SV1')):
    plan = {'PRC_TWIN': ('twin',)}
    for solver in solvers:
        tag = '' if solver == 'original' else solver.lower()
        for fam, cfg in FAMS.items():
            plan.update({f'{fam}{tag}@{g}': ('prc', cfg, g, {}, solver) for g in GRID})
    return plan


def stage_nested(version, mf_lab, log=print):
    t0 = time.process_time()
    rec_path = C.RESULTS / 'nested' / f'NESTED_SOLVERS_V{int(version)}.json'
    if rec_path.exists():
        raise PermissionError(f'{rec_path.name} exists')
    sel = frc_record(f'fa/SELECTION_V{int(version)}.json')
    Z, lab, unit, names, row_sha = FR._prep(mf_lab)
    plan = nested_plan()
    nest = nested_s(Z, lab, unit, version, plan, 'B1', T_fixed=sel['T_fixed'], capacity=tuple(sel['capacity']),
                    outer=int(sel['outer_folds']), log=log)
    arrays = {'unit': np.asarray(unit), 'sex': lab['sex']}
    for name in plan:
        for key, arr in nest[name]['_per_person'].items():
            arrays[f'{name}|{key}'] = np.asarray(arr)
    npz = _priv(f'NESTED_SOLVERS_V{int(version)}.npz')
    if npz.exists():
        raise PermissionError(f'{npz.name} exists')
    np.savez_compressed(npz, **arrays)
    spend = {}
    for name in plan:
        if name == 'PRC_TWIN':
            continue
        spend[name] = {k2: list(v2) for k2, v2 in P.spend(nest, name, 'PRC_TWIN', lab, unit).items()}
    replay_equal = None
    rp = _priv(f'P1_REPLAY_V{int(version)}.npz')
    if rp.exists():                              # the original-solver entries must equal the unchanged P1 replay exactly
        with np.load(rp) as z:
            replay_equal = {}
            for name in plan:
                if '@' in name and not any(t in name for t in ('sv1',)):
                    for key in ('occ_nll', 'inc_nll', 'look_pair'):
                        rk = f'{name}|{key}'
                        if rk in z.files:
                            replay_equal[rk] = float(np.max(np.abs(z[rk] - nest[name]['_per_person'][key])))
    rec = {'stage': 'nested_solver_comparison', 'version': int(version), 'utc': _utc(), 'card': CARD,
           'original_entries_max_abs_diff_vs_replay': (max(replay_equal.values()) if replay_equal else None),
           'means': {n: {k2: v2 for k2, v2 in d.items() if k2 != '_per_person'} for n, d in nest.items()},
           'spend_over_twin': spend, 'per_person_file_sha256': _sha(npz), 'T_fixed': sel['T_fixed'],
           'outer_folds': int(sel['outer_folds']), 'cpu_s': round(time.process_time() - t0, 1)}
    _write_once(rec_path, rec)
    log(json.dumps({'version': version, 'cpu_s': rec['cpu_s']}))
    return rec


# ---------------------------------------------------------------- queue entry
def _features():
    """Label-free features for the cached-statistics stages, which still need the verified registration and the committed card."""
    if not C.registered():
        raise PermissionError('the cached-statistics stages run only after the verified registration receipt')
    C.card_info(CARD, 'FA')
    return C.features('MODEL_FIT')


def _labelled(purpose):
    return C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), operation='fit', purpose=purpose, card=CARD)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('replay', 'audit', 'source', 'nested'))
    ap.add_argument('--version', type=int, required=True, choices=C.VERSIONS)
    a = ap.parse_args(argv)
    if a.stage == 'replay':
        stage_replay(a.version, _labelled('P1 replay: the previous campaign\'s nested P1 estimator rerun unchanged with per-person '
                                          'outputs saved (income, occupation, SEX as a training and evaluation label, as in P1)'))
    elif a.stage == 'audit':
        stage_audit(a.version, _features())
    elif a.stage == 'source':
        stage_source(a.version, _features())
    else:
        stage_nested(a.version, _labelled('nested MODEL_FIT comparison of the original and SV1 chains of PRC and HQ in the same '
                                          'outer folds (income, occupation, SEX as training and evaluation labels)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
