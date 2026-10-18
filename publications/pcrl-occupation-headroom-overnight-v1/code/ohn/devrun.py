"""Common release format, anchors, references and the development scoring entry point of the overnight occupation-headroom
campaign (EXPLORATORY DEVELOPMENT on previously used records).

Occupation release (private), written by a family module through `save_occupation`:
  runs/<FAMILY>/<config_id>/S<seed>/OCC.npz    '<ROLE>__occupation' (n, 5) float64 probability vectors for ALL six roles in
                                               the admitted row order, computed from the six permitted fields only (decisions
                                               are always the argmax);
  runs/<FAMILY>/<config_id>/S<seed>/OCC.json   the record: mechanism, inputs, roles read, interpreter, exposure, individual fit
                                               counts (model / calibrator / cross-fitting), fit CPU;
  runs/<FAMILY>/<config_id>/S<seed>/CROSSFIT.npz (task models; optional for pure post-processing controls)
                                               'MODEL_FIT__oof' (n_fit, 5): household cross-fitted predictions (each household
                                               predicted by the pipeline refitted without its fold, ohn.common.crossfit_folds),
                                               'MODEL_SELECT__foldavg' (n_ms, 5): average of the 5 fold pipelines on MODEL_SELECT
                                               (to compare cross-fitted and deployed predictions), plus 'fold' (n_fit,).
A complete release pairs one INCOME ANCHOR with one occupation output:
  H3   income of H/H3_STRONG_LJDIR_PB (the source campaign's income component), frozen;
  TW   income of H/HC_STRONG (its matched task-only strong-income twin), frozen.
References (no fit): U0, Ucal, T_TREE, TASK_REFIT, NEURAL_RAW, H_LJ, H_JJ, H_LL (frozen bank), B2TWIN, RETIRED_B3CTL (lineage
releases), SUCC = H3 + JOINT_TREE occupation (the source successor), TWJ = HC_STRONG + JOINT_TREE occupation, O1 = the source
family-O recalibration of the JOINT_TREE occupation (income per the anchor when combined).
MODEL_SELECT labels and SEX are read (and logged) only by `score` (utility) and `ohn.devreaders` (reader grid).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C

ANCHORS = {'H3': ('H', 'H3_STRONG_LJDIR_PB'), 'TW': ('H', 'HC_STRONG')}
SOURCE_OCC = {'JOINT': 'H_LJ occupation (JOINT_TREE E20, frozen bank)', 'T_TREE': 'CatBoost teacher occupation (frozen bank)',
              'TASK_REFIT': 'task-only leaf corrections on the teacher routing (frozen bank)',
              'O1': 'source family-O O1_SEXBAL recalibration of the JOINT_TREE occupation output',
              'OC_TCAL': 'source family-O temperature control', 'OC_BFIT': 'source family-O unweighted bias control'}


def occ_dir(family, config_id, seed):
    return C.RUNS / family / config_id / f'S{seed}'


def _check_prob(p, k, what):
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    if p.ndim != 2 or p.shape[1] != k or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
        raise ValueError(f'{what}: not a valid {k}-class probability-vector release')
    return p


def role_sizes():
    out = {}
    for role in C.ROLES6:
        F = C.load_role(role, ('features',))
        out[role] = len(F['row_id'])
    return out


def save_occupation(family, config_id, seed, occ_by_role, record, crossfit=None):
    """Write OCC.npz / OCC.json (and CROSSFIT.npz). Refuses to overwrite: a revision gets a new config id."""
    d = occ_dir(family, config_id, seed)
    if (d / 'OCC.npz').exists():
        raise PermissionError(f'{family}/{config_id}/S{seed} already exists (append-only; use a new configuration id)')
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    missing = set(C.ROLES6) - set(occ_by_role)
    if missing:
        raise ValueError(f'occupation release must cover all six roles; missing {sorted(missing)}')
    arrays = {f'{role}__occupation': _check_prob(occ_by_role[role], 5, f'{role} occupation') for role in C.ROLES6}
    with open(d / 'OCC.npz', 'wb') as f:
        np.savez(f, **arrays)
    rec = {**record, 'family': family, 'config_id': config_id, 'seed': seed, 'utc': C.now(), 'interpreter': sys.executable,
           'python': sys.version.split()[0], 'numpy': np.__version__, 'occ_sha256': C.sha(d / 'OCC.npz'),
           'wire_sha256': {k: C.wire_sha(v) for k, v in arrays.items()}, 'unit_id': os.environ.get(C.UNIT_ENV)}
    for key in ('n_model_fits', 'n_calibrator_fits', 'n_crossfit_fits', 'operation', 'roles_read', 'exposure', 'fit_cpu_s', 'kind'):
        if key not in rec:
            raise ValueError(f'OCC.json record needs {key!r}')
    if crossfit is not None:
        oof = _check_prob(crossfit['MODEL_FIT__oof'], 5, 'cross-fitted MODEL_FIT')
        fa = _check_prob(crossfit['MODEL_SELECT__foldavg'], 5, 'fold-average MODEL_SELECT')
        with open(d / 'CROSSFIT.npz', 'wb') as f:
            np.savez(f, MODEL_FIT__oof=oof, MODEL_SELECT__foldavg=fa, fold=np.asarray(crossfit['fold'], dtype=np.int64))
        rec['crossfit_sha256'] = C.sha(d / 'CROSSFIT.npz')
    (d / 'OCC.json').write_text(json.dumps(rec, indent=1, sort_keys=True, default=float) + '\n')
    return d


def load_occupation(family, config_id, seed, roles=None):
    d = occ_dir(family, config_id, seed)
    rec = json.loads((d / 'OCC.json').read_text())
    if C.sha(d / 'OCC.npz') != rec['occ_sha256']:
        raise PermissionError('occupation release file differs from its record')
    out = {}
    with np.load(d / 'OCC.npz') as z:
        for k in z.files:
            role = k.split('__')[0]
            if roles is None or role in roles:
                out[role] = z[k].copy()
    cf = None
    if (d / 'CROSSFIT.npz').is_file():
        if C.sha(d / 'CROSSFIT.npz') != rec.get('crossfit_sha256'):
            raise PermissionError('cross-fitting file differs from its record')
        with np.load(d / 'CROSSFIT.npz') as z:
            cf = {k: z[k].copy() for k in z.files}
    return out, rec, cf


def anchor_income(anchor, seed, role):
    fam, cfg = ANCHORS[anchor]
    rel, _ = C.ihs_release(fam, cfg, seed, roles=(role,))
    return rel[role]['income']


def source_occupation(name, seed, role):
    """Occupation output of a frozen source release on one role (no fit)."""
    if name in ('JOINT', 'T_TREE', 'TASK_REFIT'):
        bank, _ = C.frozen_bank(seed, roles=(role,)) if role in ('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT') else (None, None)
        if bank is not None:
            return bank['H_LJ' if name == 'JOINT' else name][role]['occupation']
        from ihs import bases as B                                   # roles outside the frozen bank: recompute (checked vs storage)
        arm = {'JOINT': 'JOINT_TREE', 'T_TREE': 'T_TREE', 'TASK_REFIT': 'TASK_REFIT'}[name]
        return B.base_outputs(seed, arm, (role,))[role]['prob']['occupation']
    if name in ('O1', 'OC_TCAL', 'OC_BFIT'):
        cfg = {'O1': 'O1_SEXBAL', 'OC_TCAL': 'OC_TCAL', 'OC_BFIT': 'OC_BFIT'}[name]
        rel, _ = C.ihs_release('O', cfg, seed, roles=(role,))
        return rel[role]['occupation']
    raise KeyError(name)


def unit_stats(**counts):
    """Add counters (model / calibrator / cross-fitting / reader fits, cache hits) to this queue unit's UNIT_STATS.json."""
    p = os.environ.get('OHN_UNIT_STATS')
    if not p:
        return
    p = Path(p)
    cur = json.loads(p.read_text()) if p.is_file() else {}
    for k, v in counts.items():
        cur[k] = cur.get(k, 0) + v if isinstance(v, (int, float)) else v
    p.write_text(json.dumps(cur, sort_keys=True))


# ---------------------------------------------------------------- references and scoring (MODEL_SELECT labels: logged)
FROZEN_COMPLETE = ('U0', 'Ucal', 'T_TREE', 'TASK_REFIT', 'NEURAL_RAW', 'H_LJ', 'H_JJ', 'H_LL')
AMD_RUN_OF = {'RETIRED_B3CTL': 'B3CTL_UNCOUPLED_MIX50_C6A', 'B2TWIN': 'B2CTL_UNCOUPLED_C6A'}
COMBINED = {'SUCC': ('H3', 'JOINT'), 'TWJ': ('TW', 'JOINT'), 'H3+T_TREE': ('H3', 'T_TREE'), 'H3+TASK_REFIT': ('H3', 'TASK_REFIT'),
            'H3+O1': ('H3', 'O1'), 'TW+T_TREE': ('TW', 'T_TREE'), 'TW+TASK_REFIT': ('TW', 'TASK_REFIT'), 'TW+O1': ('TW', 'O1')}
REFERENCE_SET = FROZEN_COMPLETE + tuple(AMD_RUN_OF) + tuple(COMBINED)


def reference_release(name, seed, role):
    """A frozen complete reference release {'income', 'occupation'} on one role (no fit)."""
    if name in FROZEN_COMPLETE:
        bank, _ = C.frozen_bank(seed, roles=(role,))
        return {t: bank[name][role][t] for t in C.TASKS}
    if name in AMD_RUN_OF:
        return C.amd_release(AMD_RUN_OF[name], seed, roles=(role,))[role]
    anchor, occ = COMBINED[name]
    return {'income': anchor_income(anchor, seed, role), 'occupation': source_occupation(occ, seed, role)}


def references(seed):
    """Unit OHN_REFERENCES_S<s>: MODEL_SELECT utility of every frozen reference (comparator utility and SE floors)."""
    from . import evaluate as EV
    from . import ledger as L
    S, refs, others = EV.ms_context(seed, purpose=f'reference utility evaluation, seed {seed}')
    out = {}
    for name in REFERENCE_SET:
        rel = reference_release(name, seed, 'MODEL_SELECT')
        m = EV.complete_eval(seed, rel['income'], rel['occupation'], S, refs, others)
        from . import rules as RU
        m['reserve'] = RU.reserve({seed: m})
        out[name] = m
        L.append({'family': 'REF', 'version': 'frozen', 'config_id': name, 'seed': seed, 'anchor': '', 'kind': 'reference',
                  'budget_class': 'NONE', 'counts': 0, 'n_model_fits': 0, 'unit_id': os.environ.get(C.UNIT_ENV, ''),
                  'hypothesis': 'historical or combined frozen release (no fit)', 'roles_read': 'MODEL_SELECT (labels, SEX: logged)',
                  **L.summarize(m)})
    d = C.RUNS / 'REF' / f'S{seed}'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'EVAL_MS_REFERENCES.json').write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + '\n')
    return out


def reference_eval(seed):
    return json.loads((C.RUNS / 'REF' / f'S{seed}' / 'EVAL_MS_REFERENCES.json').read_text())


def score(family, config_id, seed, version, kind, budget_class, counts, hypothesis, change, training, selection, notes='',
          code_paths=(), anchors=('H3', 'TW')):
    from . import evaluate as EV
    from . import ledger as L
    t0 = time.process_time()
    occ, rec, cf = load_occupation(family, config_id, seed, roles=('MODEL_SELECT',))
    purpose = f'development utility evaluation {family}/{config_id}/S{seed}'
    S, refs, others = EV.ms_context(seed, purpose=purpose)
    results = {}
    for i, anchor in enumerate(anchors):
        inc = anchor_income(anchor, seed, 'MODEL_SELECT')
        m = EV.complete_eval(seed, inc, occ['MODEL_SELECT'], S, refs, others, cf=cf if i == 0 else None, purpose=purpose)
        if i > 0 and 'crossfit' in results[anchors[0]]:
            m['crossfit'] = results[anchors[0]]['crossfit']
        from . import rules as RU                      # the statistical reviewer's registered rule (one seed: per-seed status)
        m['reserve'] = RU.reserve({seed: m})
        results[anchor] = m
        d = occ_dir(family, config_id, seed)
        (d / f'EVAL_MS_{anchor}.json').write_text(json.dumps(m, indent=1, sort_keys=True, default=float) + '\n')
    m = results[anchors[0]]
    paths = [Path(__file__), Path(EV.__file__)] + [Path(p) for p in code_paths]
    row = {'family': family, 'version': version, 'config_id': config_id, 'seed': seed, 'anchor': anchors[0], 'kind': kind,
           'budget_class': budget_class, 'counts': int(counts), 'n_model_fits': rec.get('n_model_fits', ''),
           'n_calibrator_fits': rec.get('n_calibrator_fits', ''), 'n_crossfit_fits': rec.get('n_crossfit_fits', ''),
           'code_sha256': L.code_sha(*paths), 'unit_id': os.environ.get(C.UNIT_ENV, ''), 'interpreter': rec.get('python', ''),
           'hypothesis': hypothesis, 'change_from_previous': change,
           'roles_read': str(rec.get('roles_read', '')) + ' | MODEL_SELECT (labels, SEX: development evaluation, logged)',
           'training_effort': training, 'selection_effort': selection,
           'cpu_s': round(float(rec.get('fit_cpu_s', 0.0)) + time.process_time() - t0, 1), 'notes': notes, **L.summarize(m)}
    n = L.append(row)
    return results, n


def load_complete(family, config_id, seed, roles=None):
    """A complete release (both tasks, all six roles) saved as RELEASE.npz / RELEASE.json (income branch)."""
    d = occ_dir(family, config_id, seed)
    rec = json.loads((d / 'RELEASE.json').read_text())
    if C.sha(d / 'RELEASE.npz') != rec['release_sha256']:
        raise PermissionError('release file differs from its record')
    out = {}
    with np.load(d / 'RELEASE.npz') as z:
        for k in z.files:
            role, t = k.split('__')
            if roles is None or role in roles:
                out.setdefault(role, {})[t] = z[k].copy()
    return out, rec


def save_complete(family, config_id, seed, rel_by_role, record):
    """Write RELEASE.npz / RELEASE.json for a complete release (income branch); refuses overwrites."""
    d = occ_dir(family, config_id, seed)
    if (d / 'RELEASE.npz').exists():
        raise PermissionError(f'{family}/{config_id}/S{seed} already exists (append-only; use a new configuration id)')
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    if set(C.ROLES6) - set(rel_by_role):
        raise ValueError('complete release must cover all six roles')
    arrays = {f'{role}__{t}': _check_prob(rel_by_role[role][t], C.K[t], f'{role} {t}') for role in C.ROLES6 for t in C.TASKS}
    with open(d / 'RELEASE.npz', 'wb') as f:
        np.savez(f, **arrays)
    rec = {**record, 'family': family, 'config_id': config_id, 'seed': seed, 'utc': C.now(), 'interpreter': sys.executable,
           'python': sys.version.split()[0], 'numpy': np.__version__, 'release_sha256': C.sha(d / 'RELEASE.npz'),
           'wire_sha256': {k: C.wire_sha(v) for k, v in arrays.items()}, 'unit_id': os.environ.get(C.UNIT_ENV)}
    for key in ('n_model_fits', 'n_calibrator_fits', 'n_crossfit_fits', 'operation', 'roles_read', 'exposure', 'fit_cpu_s', 'kind'):
        if key not in rec:
            raise ValueError(f'RELEASE.json record needs {key!r}')
    (d / 'RELEASE.json').write_text(json.dumps(rec, indent=1, sort_keys=True, default=float) + '\n')
    return d


def score_complete(family, config_id, seed, version, kind, budget_class, counts, hypothesis, change, training, selection, notes=''):
    """Score a complete release (both tasks from RELEASE.npz) with the registered evaluator -> EVAL_MS_COMPLETE.json + ledger row."""
    from . import evaluate as EV
    from . import ledger as L
    from . import rules as RU
    t0 = time.process_time()
    rel, rec = load_complete(family, config_id, seed, roles=('MODEL_SELECT',))
    S, refs, others = EV.ms_context(seed, purpose=f'development utility evaluation {family}/{config_id}/S{seed} (complete release)')
    m = EV.complete_eval(seed, rel['MODEL_SELECT']['income'], rel['MODEL_SELECT']['occupation'], S, refs, others)
    m['reserve'] = RU.reserve({seed: m})
    (occ_dir(family, config_id, seed) / 'EVAL_MS_COMPLETE.json').write_text(json.dumps(m, indent=1, sort_keys=True, default=float) + '\n')
    row = {'family': family, 'version': version, 'config_id': config_id, 'seed': seed, 'anchor': 'COMPLETE', 'kind': kind,
           'budget_class': budget_class, 'counts': int(counts), 'n_model_fits': rec.get('n_model_fits', ''),
           'n_calibrator_fits': rec.get('n_calibrator_fits', ''), 'n_crossfit_fits': rec.get('n_crossfit_fits', ''),
           'code_sha256': L.code_sha(Path(__file__), Path(EV.__file__)), 'unit_id': os.environ.get(C.UNIT_ENV, ''),
           'interpreter': rec.get('python', ''), 'hypothesis': hypothesis, 'change_from_previous': change,
           'roles_read': str(rec.get('roles_read', '')) + ' | MODEL_SELECT (labels, SEX: development evaluation, logged)',
           'training_effort': training, 'selection_effort': selection,
           'cpu_s': round(float(rec.get('fit_cpu_s', 0.0)) + time.process_time() - t0, 1), 'notes': notes, **L.summarize(m)}
    return m, L.append(row)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sc = sub.add_parser('score-complete')
    for a in ('--family', '--config', '--hypothesis', '--change', '--training', '--selection'):
        sc.add_argument(a, required=True)
    sc.add_argument('--seed', type=int, required=True); sc.add_argument('--version', default='')
    sc.add_argument('--kind', default='income_branch', choices=('income_branch', 'control', 'privacy', 'stability'))
    sc.add_argument('--budget-class', default='INC8', choices=('OCC36', 'PRIV24', 'INC8', 'CTRL12', 'NONE'))
    sc.add_argument('--counts', type=int, default=1); sc.add_argument('--notes', default='')
    e = sub.add_parser('score')
    for a in ('--family', '--config', '--hypothesis', '--change', '--training', '--selection'):
        e.add_argument(a, required=True)
    e.add_argument('--seed', type=int, required=True); e.add_argument('--version', default='')
    e.add_argument('--kind', default='task_model', choices=('task_model', 'privacy', 'control', 'income_branch', 'stability'))
    e.add_argument('--budget-class', default='OCC36', choices=('OCC36', 'PRIV24', 'INC8', 'CTRL12', 'NONE'))
    e.add_argument('--counts', type=int, default=1, help='1 on the first scoring of a new seed-specific configuration, else 0')
    e.add_argument('--anchor', default='H3', choices=tuple(ANCHORS))
    e.add_argument('--notes', default=''); e.add_argument('--code', action='append', default=[])
    r = sub.add_parser('references'); r.add_argument('--seed', type=int, required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'score-complete':
        from . import ledger as L
        m, n = score_complete(a.family, a.config, a.seed, a.version, a.kind, a.budget_class, a.counts, a.hypothesis, a.change,
                              a.training, a.selection, a.notes)
        print(json.dumps({'ledger_entry': n, 'complete': L.summarize(m)}, default=float))
        return
    if a.cmd == 'references':
        out = references(a.seed)
        print(json.dumps({k: {'point_pass_12': v['utility']['point_pass_12'], 'guard_pass': v['utility']['guard_pass']} for k, v in out.items()}))
        return
    anchors = (a.anchor,) + tuple(x for x in ANCHORS if x != a.anchor)
    res, n = score(a.family, a.config, a.seed, a.version, a.kind, a.budget_class, a.counts, a.hypothesis, a.change, a.training,
                   a.selection, a.notes, a.code, anchors)
    from . import ledger as L
    print(json.dumps({'ledger_entry': n, 'anchors': {k: L.summarize(v) for k, v in res.items()}}, default=float))


if __name__ == '__main__':
    main()
