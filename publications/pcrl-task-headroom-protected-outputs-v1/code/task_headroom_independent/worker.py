"""Independent verification worker: runs the configured stages inside one queue unit.

Every byte is opened through the queue's verify(path, sha256) (the only task_headroom
import, permitted queue infrastructure). All numerics are from task_headroom_independent.core.
Receipts: results/pcrl_task_headroom_protected_outputs_v1/independent/STAGE_<name>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from . import core

W = Path(__file__).resolve().parents[1]
OUT = W / 'results/pcrl_task_headroom_protected_outputs_v1/independent'
PRIV_OUT = W / '.private/task_headroom/independent'
ROLES = ('UPDATE', 'GUARD', 'FIT2', 'ATT', 'INNER')
EXPECTED_ROWS = {'UPDATE': 8180, 'GUARD': 2043, 'FIT2': 5211, 'ATT': 4064, 'INNER': 2235}
FORBIDDEN = ('sex', 'race', 'income', 'occupation', 'fnlwgt', 'row_id', 'unit', 'hidden_flag')
TOL_PROB = 1e-12
TOL_INTERVAL = 0.003


def _f(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


class Receipt:
    def __init__(self, name):
        self.name = name; self.checks = []; self.details = {}; self.t0 = time.process_time()

    def check(self, label, ok, **detail):
        self.checks.append({'check': label, 'pass': bool(ok), **detail})
        return bool(ok)

    def write(self, unit_id):
        failed = [c['check'] for c in self.checks if not c['pass']]
        rec = {'schema': 'THPV-independent-stage-v1', 'stage': self.name, 'queue_unit_id': unit_id,
               'status': 'PASS' if not failed and self.checks else 'FAIL', 'n_checks': len(self.checks),
               'failed_checks': failed, 'checks': self.checks, 'details': self.details,
               'cpu_s': time.process_time() - self.t0,
               'maxrss_bytes_self': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'imports': 'numpy/scipy/sklearn/torch/catboost + task_headroom.evaluation_queue.queue_config only'}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'STAGE_{self.name}.json').write_text(json.dumps(rec, indent=2, sort_keys=True, default=_f))
        return rec


class Ctx:
    def __init__(self, cfg, verify, guard):
        self.cfg, self.verify, self.guard = cfg, verify, guard
        self._roles = {}; self._teachers = {}; self._leaves = {}

    def path(self, item):
        return self.verify(item['path'], item['sha256'])

    def json(self, item):
        return json.loads(self.path(item).read_text())

    def npz(self, item):
        with np.load(self.path(item), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def npz_keys(self, item):
        with np.load(self.path(item), allow_pickle=False) as z:
            return sorted(z.files)

    def role(self, name):
        if name not in self._roles:
            self._roles[name] = self.npz(self.cfg['roles'][name]['features'])
        return self._roles[name]

    def labels(self, name, field):
        z = self.npz(self.cfg['roles'][name][field])
        f = self.role(name)
        assert np.array_equal(z['row_id'], f['row_id']) and np.array_equal(z['unit'], f['unit']), f'{name} {field} pairing'
        return z[field].astype(np.int64)

    def X(self, role):
        if role == 'POOL':
            return np.concatenate([self.role('UPDATE')['X'], self.role('GUARD')['X']])
        return self.role(role)['X']

    def teacher(self, seed, task, depth=4):
        key = (seed, task, depth)
        if key not in self._teachers:
            self._teachers[key] = core.Teacher(self.json(self.cfg['stage_a'][str(seed)][f'TREE_TEACHER_{task}_d{depth}.json']))
        return self._teachers[key]

    def leaves(self, seed, task, role, depth=4):
        key = (seed, task, role, depth)
        if key not in self._leaves:
            t = self.teacher(seed, task, depth)
            L = t.leaves(self.X(role))
            self._leaves[key] = (L, t.teacher_raw(L))
        return self._leaves[key]

    def tree_release(self, seed, role, theta=None):
        """[(p1, d1), (p2, d2)] for a tree service with optional theta pair."""
        out = []
        for i, task in enumerate(core.TASKS):
            t = self.teacher(seed, task)
            L, raw = self.leaves(seed, task, role)
            if theta is not None:
                raw = raw + t.correction(L, theta[i])
            out.append(core.release(task, raw))
        return out

    def theta(self, item):
        z = self.npz(item)
        assert sorted(z) == ['theta_income', 'theta_occupation']
        return [z['theta_income'], z['theta_occupation']]

    def neq_release(self, seed, role):
        """Own torch forward of the stored N_EQ state dicts."""
        import torch
        torch.set_num_threads(1)
        payload = torch.load(self.path(self.cfg['stage_a'][str(seed)]['NEQ_MODEL.pt']), map_location='cpu', weights_only=True)
        assert list(payload['feature_names']) == [str(x) for x in self.role('FIT2')['feature_names']]
        out = []
        for task, k in (('income', 2), ('occupation', 6)):
            sd = payload[task]
            x = torch.from_numpy(np.ascontiguousarray(self.X(role), dtype=np.float32))
            lin = torch.nn.functional.linear
            with torch.no_grad():
                h = torch.relu(lin(x, sd['0.weight'], sd['0.bias']))
                h = torch.relu(lin(h, sd['2.weight'], sd['2.bias']))
                h = lin(h, sd['4.weight'], sd['4.bias'])
                z = lin(h, sd['5.weight'], sd['5.bias']).to(torch.float64).numpy()
            z = z - z.mean(1, keepdims=True); z = z - z.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            out.append((p, p.argmax(1).astype(np.int64)))
        return out

    def raw_proj_release(self, seed, role):
        z = self.npz(self.cfg['raw_proj_stored'][str(seed)][role])
        r1, r2 = z['r1'], z['r2']
        return [(np.ascontiguousarray(r1[:, 18:20]), r1[:, 20].astype(np.int64)),
                (np.ascontiguousarray(r2[:, 22:28]), r2[:, 28].astype(np.int64))]

    def hist_release(self, name, seed, role):
        rid = self.role(role)['row_id']
        out = []
        for part in self.cfg['historical'][name][str(seed)]['parts']:
            z = self.npz(part)
            pos = {int(r): j for j, r in enumerate(z['row_id'])}
            ix = np.array([pos[int(r)] for r in rid])
            p = np.asarray(z[part['probability_field']], dtype=np.float64)[ix]
            d = np.asarray(z[part['decision_field']], dtype=np.int64)[ix]
            assert np.array_equal(p.argmax(1), d), 'historical decision is not argmax'
            out.append((p, d))
        return out

    def service_release(self, label, key, seed, role):
        """label: roster label; key: FIT2-table key (e.g. 'J_TREE|3.0|E5', 'T_TREE', 'RAW_PROJ')."""
        if key == 'T_TREE':
            return self.tree_release(seed, role)
        if key == 'N_EQ':
            return self.neq_release(seed, role)
        if key == 'RAW_PROJ':
            return self.raw_proj_release(seed, role)
        if key in self.cfg.get('historical', {}):
            return self.hist_release(key, seed, role)
        return self.tree_release(seed, role, self.theta(self.cfg['checkpoints'][str(seed)][key]))


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


# =============================================================================== 1. SOURCE
def stage_source(ctx, R):
    cfg = ctx.cfg
    adm = ctx.json(cfg['admission_receipt'])
    lock = ctx.json(cfg['sci_lock'])
    R.check('admission_receipt_sha_matches_science_lock', cfg['admission_receipt']['sha256'] == lock['admission']['sha256'])
    R.check('admission_status_PASS', adm['status'] == 'PASS')
    admcfg = ctx.json(cfg['admission_config'])
    for role, fields in cfg['roles'].items():
        for field, item in fields.items():
            with open(ctx.path(item), 'rb') as fh:
                actual = hashlib.file_digest(fh, 'sha256').hexdigest()
            expected = admcfg['roles'][role][field]['sha256'] if field in admcfg['roles'].get(role, {}) else (
                adm['fit2_sex_export']['sha256'] if (role, field) == ('FIT2', 'sex') else None)
            R.check(f'rehash_{role}_{field}', actual == expected and admcfg['roles'].get(role, {}).get(field, {'path': item['path']})['path'] == item['path'],
                    actual=actual, expected=expected)
    names0 = None; ids = {}; groups = {}
    for role in ROLES:
        f = ctx.role(role)
        n = len(f['row_id'])
        facts = adm['facts'][role]
        R.check(f'{role}_rows', n == EXPECTED_ROWS[role] == facts['rows'], rows=n)
        R.check(f'{role}_unique_row_ids', len(np.unique(f['row_id'])) == n)
        R.check(f'{role}_groups', len(np.unique(f['unit'])) == facts['groups'], groups=int(len(np.unique(f['unit']))))
        R.check(f'{role}_X_digest', core.array_digest(f['X']) == facts['X_sha256'])
        R.check(f'{role}_row_id_digest', core.array_digest(f['row_id']) == facts['row_id_sha256'])
        X = f['X']
        R.check(f'{role}_X_float32_finite_83', X.dtype == np.float32 and X.shape == (n, 83) and bool(np.isfinite(X).all()))
        names = [str(x) for x in f['feature_names']]
        R.check(f'{role}_no_forbidden_inputs', not any(x.lower().split('=')[0] in FORBIDDEN for x in names))
        if names0 is None:
            names0 = names
        R.check(f'{role}_feature_schema_identical', names == names0)
        # duplicate groups carry identical features
        order = np.argsort(f['unit'], kind='stable'); u = f['unit'][order]; Xs = X[order]
        same = u[1:] == u[:-1]
        R.check(f'{role}_duplicate_groups_identical_features', bool(np.all(Xs[1:][same] == Xs[:-1][same])), duplicate_pairs=int(same.sum()))
        ids[role] = set(map(int, f['row_id'])); groups[role] = set(map(int, f['unit']))
        for field in cfg['roles'][role]:
            if field in ('income', 'occupation', 'sex'):
                y = ctx.labels(role, field)
                key = f'{field}_counts'
                if key in facts:
                    R.check(f'{role}_{field}_counts', np.bincount(y, minlength=core.NCLASS.get(field, 2)).tolist() == facts[key])
            elif field.startswith('references'):
                z = ctx.npz(cfg['roles'][role][field])
                R.check(f'{role}_{field}_pairing', np.array_equal(z['row_id'], f['row_id']) and np.array_equal(z['unit'], f['unit']))
    for i, a in enumerate(ROLES):
        for b in ROLES[i + 1:]:
            R.check(f'rows_disjoint_{a}_{b}', not (ids[a] & ids[b]))
            R.check(f'groups_disjoint_{a}_{b}', not (groups[a] & groups[b]))
    R.check('FIT2_is_5211_rows', len(ids['FIT2']) == 5211)
    R.check('POOL_is_10223_rows', len(ids['UPDATE']) + len(ids['GUARD']) == 10223)
    # stored RAW_PROJ outputs align with role order (digests recorded at admission)
    for s in (0, 1):
        for role in ('FIT2', 'ATT', 'INNER'):
            w = ctx.raw_proj_release(s, role)
            fx = adm['raw_proj'][f's{s}_{role}']
            R.check(f'RAW_PROJ_stored_s{s}_{role}_digest', core.array_digest(w[0][0]) == fx['income_sha256'] and core.array_digest(w[1][0]) == fx['occupation_sha256'])
            R.check(f'RAW_PROJ_stored_s{s}_{role}_decisions_are_argmax', all(np.array_equal(p.argmax(1), d) for p, d in w))
    for name in cfg.get('historical', {}):
        for s in (0, 1):
            for role in ('FIT2', 'ATT', 'INNER'):
                w = ctx.hist_release(name, s, role)
                fx = adm['historical'][f'{name}|s{s}|{role}']
                R.check(f'HIST_{name}_s{s}_{role}_digest', core.array_digest(w[0][0]) == fx['income_sha256'] and core.array_digest(w[1][0]) == fx['occupation_sha256'])


# =============================================================================== 2. TEACHER
def stage_teacher(ctx, R):
    import catboost
    R.details['catboost_version'] = catboost.__version__
    R.check('catboost_1.2.8', catboost.__version__ == '1.2.8')
    lockB = ctx.json(ctx.cfg['stage_b_lock'])
    for s in (0, 1):
        A = ctx.cfg['stage_a'][str(s)]
        seedrec = ctx.json(A['STAGE_A_SEED.json'])
        for task in core.TASKS:
            for depth in ctx.cfg.get('teacher_depths', [4, 6]):
                ctx.guard()
                tag = f'{task}_d{depth}'
                tj = A[f'TREE_TEACHER_{tag}.json']
                R.check(f's{s}_{tag}_payload_sha_in_seed_record', seedrec['files'][f'TREE_TEACHER_{tag}.json'] == tj['sha256'])
                t = ctx.teacher(s, task, depth)
                R.check(f's{s}_{tag}_tree_count', t.T == seedrec['trees'][tag]['tree_count'], T=t.T)
                if depth == 4:
                    sel = lockB['selected_teachers'][f'{s}|{task}']
                    R.check(f's{s}_{tag}_bound_in_STAGE_B_LOCK', sel['teacher_payload_sha256'] == tj['sha256'] and sel['tree_count'] == t.T
                            and sel['catboost_model_sha256'] == A[f'CATBOOST_{tag}.cbm']['sha256'])
                stored = ctx.npz(A[f'TREE_OUTPUTS_{tag}.npz'])
                model = catboost.CatBoostClassifier()
                model.load_model(str(ctx.path(A[f'CATBOOST_{tag}.cbm'])))
                for role in ('POOL', 'FIT2', 'ATT', 'INNER'):
                    X = ctx.X(role)
                    L, raw = ctx.leaves(s, task, role, depth)
                    p, d = core.release(task, raw)
                    diff = max_abs(p, stored[role])
                    R.check(f's{s}_{tag}_{role}_vs_TREE_OUTPUTS', diff <= TOL_PROB and np.array_equal(d, stored[role].argmax(1)), max_abs_diff=diff)
                    cl = np.asarray(model.calc_leaf_indexes(catboost.Pool(X, thread_count=1), thread_count=1), dtype=np.int64)
                    R.check(f's{s}_{tag}_{role}_routing_vs_catboost_bitwise', np.array_equal(cl, L))
                    cp = np.asarray(model.predict_proba(X, thread_count=1), dtype=np.float64)
                    craw = np.asarray(model.predict(X, prediction_type='RawFormulaVal', thread_count=1), dtype=np.float64).reshape(len(X), -1)
                    dc = max_abs(p, cp)
                    R.check(f's{s}_{tag}_{role}_vs_catboost_predict_proba', np.array_equal(d, cp.argmax(1)) and dc <= 1e-12,
                            max_abs_prob_diff=dc, max_abs_raw_diff=max_abs(raw, craw))
                    if depth == 4:
                        zero = core.release(task, raw + t.correction(L, np.zeros((t.T * t.L, t.dim))))
                        R.check(f's{s}_{tag}_{role}_zero_theta_identity', max_abs(zero[0], p) == 0.0 and np.array_equal(zero[1], d))
                ctx._leaves = {k: v for k, v in ctx._leaves.items() if k[3] == 4 and k[2] != 'POOL'}
                ctx._teachers = {k: v for k, v in ctx._teachers.items() if k[2] == 4}
                del model, stored


# =============================================================================== 3. STAGE_A
def stage_stage_a(ctx, R):
    rep = ctx.json(ctx.cfg['stage_a_report'])
    seeds = {s: ctx.json(ctx.cfg['stage_a'][str(s)]['STAGE_A_SEED.json']) for s in (0, 1)}
    for task in core.TASKS:
        tot = {d: sum(seeds[s]['trees'][f'{task}_d{d}']['internal_best_nll'] for s in (0, 1)) for d in (4, 6)}
        depth = 4 if tot[4] <= tot[6] else 6
        R.check(f'depth_selection_{task}', depth == rep['selection'][task]['depth'] == 4, totals=tot)
        R.check(f'depth_totals_{task}', all(abs(tot[d] - rep['selection'][task]['internal_nll_sum_by_depth'][str(d)]) <= 1e-12 for d in (4, 6)))
        R.check(f'tree_counts_{task}', all(seeds[s]['trees'][f'{task}_d{depth}']['tree_count'] == rep['selection'][task]['tree_count_by_seed'][str(s)] for s in (0, 1)))
    y = {t: ctx.labels('FIT2', t) for t in core.TASKS}
    unit = ctx.role('FIT2')['unit']
    Wb, G = core.group_multiplicities(unit, 2048, ctx.cfg.get('bootstrap_seed', 20261015))
    R.details['bootstrap'] = {'draws': 2048, 'groups': G, 'seed': ctx.cfg.get('bootstrap_seed', 20261015), 'rng': 'PCG64 multinomial'}
    diffs = {}; mine_clauses = []; mine_diag = []
    for s in (0, 1):
        neq_store = ctx.npz(ctx.cfg['stage_a'][str(s)]['NEQ_OUTPUTS.npz'])
        neq = ctx.neq_release(s, 'FIT2')
        for i, t in enumerate(core.TASKS):
            dd = max_abs(neq[i][0], neq_store[f'FIT2__{t}'])
            R.check(f's{s}_N_EQ_{t}_own_forward_vs_NEQ_OUTPUTS', dd <= TOL_PROB and np.array_equal(neq[i][1], neq_store[f'FIT2__{t}'].argmax(1)), max_abs_diff=dd)
        tree = ctx.tree_release(s, 'FIT2')
        P_t = [tree[0][0], tree[1][0]]; P_n = [neq_store['FIT2__income'], neq_store['FIT2__occupation']]
        mine_clauses += core.clause_rows(y, P_t, P_n, s, 'N_EQ')
        refs = ctx.npz(ctx.cfg['roles']['FIT2'][f'references_s{s}'])
        for ref in ('U0', 'Ucal'):
            mine_diag += core.clause_rows(y, P_t, [refs[f'{ref}_1'], refs[f'{ref}_2']], s, ref)
        for i, t in enumerate(core.TASKS):
            a, b = core.per_row(y[t], P_t[i]), core.per_row(y[t], P_n[i])
            diffs[(t, 'nll', s)] = b['nll'] - a['nll']
            diffs[(t, 'brier', s)] = b['brier'] - a['brier']
            diffs[(t, 'accuracy', s)] = a['accuracy'] - b['accuracy']
    H = rep['headroom_N_EQ_minus_tree_positive_is_tree_better']
    for t in core.TASKS:
        for m in ('nll', 'brier', 'accuracy'):
            for s in ('0', '1', 'mean_over_seeds'):
                v = (diffs[(t, m, 0)] + diffs[(t, m, 1)]) / 2 if s == 'mean_over_seeds' else diffs[(t, m, int(s))]
                iv = core.pct_interval(v.mean(), core.weighted_means(v, Wb))
                ref = H[f'{t}|{m}'][s]
                R.check(f'headroom_{t}_{m}_{s}', abs(iv['point'] - ref['point']) <= 1e-12 and abs(iv['lower'] - ref['lower']) <= TOL_INTERVAL
                        and abs(iv['upper'] - ref['upper']) <= TOL_INTERVAL, mine=iv, reported=ref)
    def cmp(mine, theirs, label):
        key = lambda r: (r['seed'], r['task'], r['reference'], r['metric'])
        tm = {key(r): r for r in theirs}
        R.check(f'{label}_count', len(mine) == len(theirs), n=len(mine))
        for r in mine:
            o = tm.get(key(r))
            R.check(f'{label}_{"_".join(map(str, key(r)))}', o is not None and abs(r['value'] - o['value']) <= 1e-12 and r['point_pass'] == o['point_pass'],
                    mine=r['value'], reported=None if o is None else o['value'], point_pass=r['point_pass'])
    cmp(mine_clauses, rep['internal_clauses_vs_N_EQ'], 'internal_clause')
    cmp(mine_diag, rep['diagnostic_clauses_vs_in_sample_U0_Ucal'], 'diagnostic_clause')
    prereq = 'TASK_MODEL_PREREQUISITE_MET' if all(r['point_pass'] for r in mine_clauses) else 'TASK_MODEL_PREREQUISITE_NOT_MET'
    R.check('prerequisite_label', prereq == rep['prerequisite'], mine=prereq)
    occ = (diffs[('occupation', 'nll', 0)] + diffs[('occupation', 'nll', 1)]).mean() / 2
    R.details['occupation_nll_headroom_point'] = float(occ)
    R.check('occupation_practical_target_point', bool(occ >= 0.010) == rep['occupation_nll_practical_target_0.010']['met_at_point'])


# =============================================================================== 4. COUNTERFACTUALS
def _ck_label_epoch(label):
    return int(label.rsplit('E', 1)[1])


def stage_counterfactuals(ctx, R):
    fams = ctx.cfg['families']
    summary = {}
    for s in (0, 1):
        ctx.guard()
        recs = {f: ctx.json(ctx.cfg['family_records'][str(s)][f]) for f in fams}
        for f, rec in recs.items():
            zi = rec['zero_identity']
            R.check(f's{s}_{f}_zero_identity_recorded', len(zi) == 8 and all(v['max_abs_prob_diff'] <= 1e-12 and v['decisions_identical'] for v in zi.values()))
            # checkpoint files registered in the record hash-match the files we open
            for tag, cks in rec['checkpoints'].items():
                for lab, info in cks.items():
                    key = f'{f}|{tag}|{lab}'
                    item = ctx.cfg['checkpoints'][str(s)].get(key)
                    R.check(f's{s}_{key}_file_sha_matches_record', item is not None and item['sha256'] == info['sha256'])
        cf = recs['J_TREE']['runs']['3.0']['counterfactuals']
        steps_ok = len(cf) == 8 and all(c['step'] == i for i, c in enumerate(cf))
        per = {r: {'probe_changed_steps': sum(c['recipients'][r]['probe_probability_max_abs_change'] > 0 for c in cf),
                   'decision_changes_total': sum(c['recipients'][r]['probe_decision_changes'] for c in cf),
                   'min_privacy_grad_norm': min(c['recipients'][r]['privacy_grad_norm'] for c in cf),
                   'max_probe_prob_change': max(c['recipients'][r]['probe_probability_max_abs_change'] for c in cf)} for r in ('0', '1')}
        R.check(f's{s}_J_TREE_rho3_first8_counterfactuals_recorded', steps_ok, per_recipient=per)
        R.check(f's{s}_J_TREE_rho3_privacy_moves_transmitted_probe_every_step',
                all(per[r]['probe_changed_steps'] == 8 and per[r]['min_privacy_grad_norm'] > 0 for r in per), per_recipient=per)
        # Released FIT2 outputs vs T_REFIT at matched epoch (and vs the teacher)
        teacher = ctx.tree_release(s, 'FIT2')
        refit = {}
        for key, item in ctx.cfg['checkpoints'][str(s)].items():
            if key.startswith('T_REFIT|'):
                refit[_ck_label_epoch(key.split('|')[2])] = ctx.tree_release(s, 'FIT2', ctx.theta(item))
        for key, item in sorted(ctx.cfg['checkpoints'][str(s)].items()):
            fam, rho, lab = key.split('|')
            th = ctx.theta(item)
            e = _ck_label_epoch(lab)
            if lab.startswith('P2'):
                # sequential rule: the first-trained recipient is frozen bitwise at its phase-1 result
                first = 0 if fam == 'S12_TREE' else 1
                p1 = ctx.theta(ctx.cfg['checkpoints'][str(s)][f'{fam}|{rho}|P1E20'])
                R.check(f's{s}_{key}_first_recipient_frozen_bitwise_at_P1E20', np.array_equal(th[first], p1[first]))
                if e == 0:
                    R.check(f's{s}_{key}_second_recipient_starts_at_zero', not np.any(th[1 - first]))
                    continue
            elif e == 0:
                R.check(f's{s}_{key}_epoch0_theta_is_zero', all(not np.any(x) for x in th))
                continue
            if fam == 'T_REFIT':
                continue
            rel = ctx.tree_release(s, 'FIT2', th)
            m = refit[e]
            row = {'vs_T_REFIT_matched': [{'max_abs_prob_diff': max_abs(rel[i][0], m[i][0]),
                                           'decision_changes': int(np.sum(rel[i][1] != m[i][1]))} for i in range(2)],
                   'vs_T_REFIT_E20': [max_abs(rel[i][0], refit[20][i][0]) for i in range(2)],
                   'vs_teacher': [max_abs(rel[i][0], teacher[i][0]) for i in range(2)],
                   'theta_max_abs': [float(np.max(np.abs(x))) for x in th]}
            summary[f's{s}|{key}'] = row
            # sequential first phase: the inactive recipient stays at the teacher; privacy movement applies to the active one
            active = [0, 1]
            if lab.startswith('P1'):
                active = [0] if fam == 'S12_TREE' else [1]
            moved = all(row['vs_T_REFIT_matched'][i]['max_abs_prob_diff'] > 1e-9 for i in active)
            if lab.startswith('P1'):
                inactive = 1 - active[0]
                R.check(f's{s}_{key}_inactive_recipient_is_teacher', row['vs_teacher'][inactive] == 0.0)
            R.check(f's{s}_{key}_released_FIT2_differs_from_T_REFIT_matched', moved, **{k: row[k] for k in ('vs_T_REFIT_matched',)})
    R.details['fit2_release_movement'] = summary
    nominees = ctx.cfg.get('nominees')
    if nominees:
        for s in (0, 1):
            for fam, key in nominees.items():
                row = summary.get(f's{s}|{key}')
                R.check(f'nominee_s{s}_{fam}_{key}_differs_from_T_REFIT_in_both_recipients', row is not None and all(
                    r['max_abs_prob_diff'] > 1e-9 for r in row['vs_T_REFIT_matched']), row=row)


# =============================================================================== 5. FIT2 / NOMINATION
def proxy_value(views, sex, folds, seed=2026101502):
    out = {}
    for name, x in views.items():
        vals = {}
        for reader in ('LR_C1', 'HGB_15x200'):
            oof = np.zeros(len(sex))
            for f in (0, 1):
                tr, te = folds != f, folds == f
                oof[te] = core.fit_predict(reader, seed, x[tr], sex[tr], x[te])[0]
            vals[reader] = core.auc(sex, oof)
        out[name] = {'readers': vals, 'value': max(vals.values())}
    return out


def pair_views(rel):
    a = core.view_features(*rel[0], 'out'); b = core.view_features(*rel[1], 'out')
    return {'income': a, 'occupation': b, 'pair': np.column_stack([a, b])}


def stage_fit2(ctx, R):
    cfg = ctx.cfg
    tabs = {s: ctx.json(cfg['fit2_tables'][str(s)])['table'] for s in (0, 1)}
    sel = ctx.json(cfg['selection'])
    lock = ctx.json(cfg['sel_lock']) if cfg.get('sel_lock') else {}
    y = {t: ctx.labels('FIT2', t) for t in core.TASKS}
    sx = ctx.npz(cfg['roles']['FIT2']['sex'])
    f2 = ctx.role('FIT2')
    R.check('FIT2_sex_pairing', np.array_equal(sx['row_id'], f2['row_id']) and np.array_equal(sx['unit'], f2['unit']))
    sex = sx['sex'].astype(np.int64)
    folds = core.proxy_fold(f2['unit'])
    R.details['proxy_fold_sizes'] = np.bincount(folds).tolist()
    # (a) replay nomination on the reported FIT2 tables
    fams = {f: core.replay_family(tabs, f) for f in core.PROTECTED_ORDER}
    for f in core.PROTECTED_ORDER:
        st, best, _ = fams[f]
        R.check(f'replay_{f}_nominee', st == sel['families'][f]['status'] and best['key'] == sel['families'][f]['nominee']['key'],
                mine={'status': st, 'key': best['key']}, reported={'status': sel['families'][f]['status'], 'key': sel['families'][f]['nominee']['key']})
    primary = core.replay_primary(fams)
    R.check('replay_primary_service', primary == sel['primary_family'], mine=primary, reported=sel['primary_family'])
    if lock:
        lp = lock.get('primary_service', lock.get('primary_family'))
        R.check('SELECTION_LOCK_primary_matches', lp == primary, lock_primary=lp)
    jl = fams['J_TREE'][1]['key'].split('|')[2]
    R.check('replay_T_REFIT_match', sel['T_REFIT'] == f'T_REFIT|none|{jl}')
    broster = cfg['B_roster']
    elig = {c: all(tabs[s][c]['internal_point_feasible'] for s in (0, 1)) for c in broster}
    required = [c for c in broster if c == 'RAW_PROJ' or elig[c]]
    R.check('replay_B_required', required == list(sel['B_required']), mine=required, reported=sel['B_required'])
    K = core.family_size(len(required), primary)
    R.check('replay_K', K == sel['K_primary_family'] and (not lock or lock.get('K_primary_family', K) == K), mine=K)
    # (b) independent recomputation of clauses (every candidate + roster) and proxies
    mine = {0: {}, 1: {}}
    nominee_keys = {f: fams[f][1]['key'] for f in core.PROTECTED_ORDER}
    for s in (0, 1):
        neq = ctx.neq_release(s, 'FIT2')
        Pn = [neq[0][0], neq[1][0]]
        keys = [k for k in tabs[s] if k in cfg['checkpoints'][str(s)] or k in broster or k in ('FARE_Z1',)]
        for key in sorted(keys):
            ctx.guard()
            rel = ctx.service_release(key, key, s, 'FIT2')
            rows = core.clause_rows(y, [rel[0][0], rel[1][0]], Pn, s, 'N_EQ')
            m = {'internal_point_feasible': all(r['point_pass'] for r in rows), 'max_normalized_violation': max(r['normalized'] for r in rows),
                 'summed_nll': sum(core.means(y[t], rel[i][0])['nll'] for i, t in enumerate(core.TASKS))}
            th = tabs[s][key]
            tr = {(r['task'], r['metric']): r['value'] for r in th['internal']}
            dv = max(abs(r['value'] - tr[(r['task'], r['metric'])]) for r in rows)
            R.check(f's{s}_{key}_clauses', dv <= 1e-12 and m['internal_point_feasible'] == th['internal_point_feasible']
                    and abs(m['summed_nll'] - th['summed_nll']) <= 1e-12, max_clause_diff=dv, feasible=m['internal_point_feasible'])
            want_proxy = (key in nominee_keys.values() or key in broster or key == sel['T_REFIT']
                          or (cfg.get('proxy_scope') == 'feasible' and m['internal_point_feasible'] and '|' in key))
            if want_proxy:
                px = proxy_value(pair_views(rel), sex, folds)
                m['proxy'] = px
                d = {v: abs(px[v]['value'] - th['proxy'][v]['value']) for v in px}
                R.check(f's{s}_{key}_proxy', max(d.values()) <= 1e-9, mine={v: px[v]['value'] for v in px},
                        reported={v: th['proxy'][v]['value'] for v in px})
            mine[s][key] = m
    # (c) fully independent nomination where proxies were recomputed
    for f in core.PROTECTED_ORDER:
        keys = [f'{f}|{rho}|{lab}' for rho in ('1.0', '3.0', '10.0') for lab in core.candidate_labels(f)]
        feas = [k for k in keys if all(mine[s][k]['internal_point_feasible'] for s in (0, 1))]
        if feas and all('proxy' in mine[s][k] for s in (0, 1) for k in feas):
            st, best, _ = core.replay_family(mine, f)
            R.check(f'independent_{f}_nominee', best['key'] == sel['families'][f]['nominee']['key'], mine=best['key'])
        elif not feas:
            R.check(f'independent_{f}_no_feasible', sel['families'][f]['status'] == 'NO_FEASIBLE_FALLBACK_DIAGNOSTIC')
    elig_mine = {c: all(mine[s][c]['internal_point_feasible'] for s in (0, 1)) for c in broster}
    R.check('independent_B_eligibility', elig_mine == elig, mine=elig_mine)
    R.details['nominees'] = nominee_keys; R.details['primary'] = primary; R.details['K'] = K; R.details['B_required'] = required
    R.details['independent_fit2'] = {f's{s}|{k}': v for s in (0, 1) for k, v in mine[s].items()}


# =============================================================================== 6. WIRES
def stage_wires(ctx, R):
    cfg = ctx.cfg
    sel = ctx.json(cfg['selection'])
    roster = sel['attack_roster']
    inner_ids = ctx.role('INNER')['row_id']
    for s in (0, 1):
        rec = ctx.json(cfg['readers'][str(s)]['receipt'])
        R.check(f's{s}_services_equal_roster', sorted(rec['services']) == sorted(cfg['service_labels']), services=sorted(rec['services']))
        for label in cfg['service_labels']:
            ctx.guard()
            key = roster[label]
            item = cfg['readers'][str(s)]['releases'][label]
            keys = ctx.npz_keys(item)
            R.check(f's{s}_{label}_INNER_RELEASE_fields_only', keys == ['d1', 'd2', 'p1', 'p2', 'row_id'], fields=keys)
            z = ctx.npz(item)
            mine_i = ctx.service_release(label, key, s, 'INNER')
            mine_a = ctx.service_release(label, key, s, 'ATT')
            dp = [max_abs(mine_i[0][0], z['p1']), max_abs(mine_i[1][0], z['p2'])]
            R.check(f's{s}_{label}_INNER_release_recomputed', np.array_equal(z['row_id'], inner_ids) and max(dp) <= TOL_PROB
                    and np.array_equal(mine_i[0][1], z['d1']) and np.array_equal(mine_i[1][1], z['d2']),
                    key=key, max_abs_prob_diff=dp)
            R.check(f's{s}_{label}_shapes', z['p1'].shape == (len(inner_ids), 2) and z['p2'].shape == (len(inner_ids), 6))
            ws = rec['services'][label]['wire_sha256']
            stored_pairs = [(z['p1'], z['d1']), (z['p2'], z['d2'])]
            # the stored INNER release is bitwise the wire the readers scored
            R.check(f's{s}_{label}_stored_INNER_release_digest_equals_reader_wire', core.wire_digest(stored_pairs) == ws['INNER'])
            exact_a = core.wire_digest(mine_a) == ws['ATT']
            if key == 'RAW_PROJ' or key in cfg.get('historical', {}):
                # stored/historical releases are read, not recomputed: bitwise identity required
                R.check(f's{s}_{label}_ATT_wire_digest_bitwise', exact_a)
            else:
                # float64 recomputation (own summation order) is equal to ~1e-15, not bitwise; recorded only
                R.details.setdefault('att_wire_bitwise_equal_recomputed', {})[f's{s}|{label}'] = bool(exact_a)


# =============================================================================== 7. READER REFITS
def stage_readers(ctx, R):
    cfg = ctx.cfg
    sel = ctx.json(cfg['selection'])
    roster = sel['attack_roster']
    att = ctx.role('ATT')
    ysex = ctx.labels('ATT', 'sex')
    tr, va = core.att_split(att['unit'])
    seed_base = 2026101205
    saved = {}
    for s in (0, 1):
        rec = ctx.json(cfg['readers'][str(s)]['receipt'])
        R.check(f's{s}_ATT_split_sizes', rec['ATT_split'] == {'train_rows': int(len(tr)), 'validation_rows': int(len(va))},
                mine=[int(len(tr)), int(len(va))])
        R.check(f's{s}_model_seed', rec['model_seed'] == seed_base + s)
        scores = ctx.npz(cfg['readers'][str(s)]['scores'])
        # Replay winner selection for every service and view from the validation table.
        for label, srec in rec['services'].items():
            for view in ('income', 'occupation', 'pair'):
                cands = {n: v['fixed_orientation_auc'] for n, v in srec['validation'][view].items()}
                if view == 'pair':
                    for single in ('income', 'occupation'):
                        other = 'occupation' if single == 'income' else 'income'
                        for n, v in srec['validation'][single].items():
                            cands[f'IGNORE_{other}__{n}'] = v['fixed_orientation_auc']
                # candidate order: slate order (as in the attack code), then pair aliases income-view first
                slate = list(core.SLATE)
                order = slate if view != 'pair' else slate + [f'IGNORE_occupation__{n}' for n in slate] + [f'IGNORE_income__{n}' for n in slate]
                best = max(cands.values())
                tied = [n for n in order if cands[n] == best]
                win = tied[0]
                R.check(f's{s}_{label}_{view}_winner_replay', win == srec['winner'][view]['reader'] and set(order) == set(cands),
                        mine=win, reported=srec['winner'][view]['reader'], tied=tied)
                if len(tied) > 1:
                    R.details.setdefault('validation_ties', {})[f's{s}|{label}|{view}'] = tied
                R.check(f's{s}_{label}_{view}_winner_scores_present', f'{label}__{view}__{win}' in scores)
        # Independent refits of the pair winner for the predeclared services.
        for label in cfg['refit_services']:
            ctx.guard()
            key = roster[label]
            win = rec['services'][label]['winner']['pair']['reader']
            if win.startswith('IGNORE_'):
                ignored, reader = win[len('IGNORE_'):].split('__')
                view = 'occupation' if ignored == 'income' else 'income'
            else:
                view, reader = 'pair', win
            kind = core.SLATE[reader]
            rel_a = ctx.service_release(label, key, s, 'ATT')
            rel_i = ctx.service_release(label, key, s, 'INNER')
            xa = core.reader_input(rel_a, view, kind); xi = core.reader_input(rel_i, view, kind)
            sv, si = core.fit_predict(reader, seed_base + s, xa[tr], ysex[tr], xa[va], xi)
            a_val = core.auc(ysex[va], sv)
            sign = 1 if a_val >= 0.5 else -1
            if sign < 0:
                sv, si = 1 - sv, 1 - si
            v_rep = rec['services'][label]['validation'][view][reader]
            stored = scores[f'{label}__pair__{win}']
            d = max_abs(si, stored)
            R.check(f's{s}_{label}_pair_winner_{win}_refit_validation_auc', abs(core.auc(ysex[va], sv) - v_rep['fixed_orientation_auc']) <= 1e-9
                    and sign == v_rep['orientation'], mine=core.auc(ysex[va], sv), reported=v_rep['fixed_orientation_auc'])
            R.check(f's{s}_{label}_pair_winner_{win}_refit_INNER_scores', d <= cfg.get('refit_tol', 1e-6), max_abs_diff=d)
            saved[f's{s}__{label}'] = si
    PRIV_OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    p = PRIV_OUT / f'READER_REFITS_{cfg["unit_id"]}.npz'
    with p.open('wb') as fh:
        np.savez(fh, row_id=ctx.role('INNER')['row_id'], **saved)
    R.details['refit_scores_private'] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


# =============================================================================== 8. INNER
VIEWS = ('income', 'occupation', 'pair')


def stage_inner(ctx, R):
    cfg = ctx.cfg
    rep = ctx.json(cfg['inner_report'])
    sel = ctx.json(cfg['sel_lock'])
    sci = ctx.json(cfg['sci_lock'])
    truth = {k: ctx.npz(cfg['inner_truth'][k]) for k in ('income', 'occupation', 'sex')}
    inner = ctx.role('INNER')
    for k, z in truth.items():
        R.check(f'truth_{k}_pairing', np.array_equal(z['row_id'], inner['row_id']) and np.array_equal(z['unit'], inner['unit']))
    y = {k: truth[k][k].astype(np.int64) for k in truth}
    unit = inner['unit']
    Wb, G = core.group_multiplicities(unit, 2048, cfg.get('bootstrap_seed', 20261015))
    R.check('groups', G == rep['groups'] == 2234, G=G)
    labels = list(cfg['service_labels'])
    primary = sel.get('primary_service', sel.get('primary_family'))
    R.check('primary_matches_lock', rep['primary_service'] == primary, mine=primary)
    Kf = int(sel['K_primary_family'])
    R.check('K_formula', Kf == core.family_size(len(sel['B_required']), primary) == rep['K'], K=Kf)
    crit = core.t_bonferroni(Kf, G)
    R.check('critical_value', abs(crit - rep['critical_value']) <= 1e-12, mine=crit, reported=rep['critical_value'])
    rec_ = {}; releases = {}; receipts = {}
    for s in (0, 1):
        receipts[s] = ctx.json(cfg['readers'][str(s)]['receipt'])
        sc = ctx.npz(cfg['readers'][str(s)]['scores'])
        R.check(f's{s}_scores_row_order', np.array_equal(sc['row_id'], inner['row_id']))
        for lab in labels:
            z = ctx.npz(cfg['readers'][str(s)]['releases'][lab])
            releases[(lab, s)] = [(z['p1'], z['d1']), (z['p2'], z['d2'])]
            for v in VIEWS:
                win = receipts[s]['services'][lab]['winner'][v]['reader']
                score = sc[f'{lab}__{v}__{win}']
                env = max(core.auc(y['sex'], sc[k]) for k in sc if k.startswith(f'{lab}__{v}__'))
                rec_[(lab, s, v)] = {'reader': win, 'point': core.auc(y['sex'], score), 'draws': core.auc_weighted(y['sex'], score, Wb), 'envelope': env}
                val = receipts[s]['services'][lab]['validation']
                cands = {n: x['fixed_orientation_auc'] for n, x in val[v].items()}
                if v == 'pair':
                    for single, other in (('income', 'occupation'), ('occupation', 'income')):
                        cands.update({f'IGNORE_{other}__{n}': x['fixed_orientation_auc'] for n, x in val[single].items()})
                tied = [n for n in cands if cands[n] == max(cands.values())]
                if len(tied) > 1:
                    R.details.setdefault('validation_tie_inner_auc', {})[f's{s}|{lab}|{v}'] = {
                        'winner': win, 'inner_auc_by_tied_reader': {n: core.auc(y['sex'], sc[f'{lab}__{v}__{n}']) for n in tied}}
        # controls
        sh_y = None
    # replay of every winner AUC vs the service table
    st = {(r['service'], r['seed']): r for r in rep['service_table']}
    for lab in labels:
        for s in (0, 1):
            for v in VIEWS:
                r = st[(lab, s)]
                m = rec_[(lab, s, v)]
                R.check(f'replay_auc_{lab}_s{s}_{v}', abs(m['point'] - r[f'{v}_auc']) <= 1e-12 and m['reader'] == r[f'{v}_reader']
                        and abs(m['envelope'] - r[f'{v}_envelope']) <= 1e-12, mine=m['point'], reported=r[f'{v}_auc'])
    # reader-refit AUCs (stage 7 predictions)
    if cfg.get('reader_refits'):
        rf = ctx.npz(cfg['reader_refits'])
        for k, v in rf.items():
            if k == 'row_id':
                continue
            s = int(k[1]); lab = k.split('__', 1)[1]
            a = core.auc(y['sex'], v)
            R.check(f'refit_INNER_auc_{k}', abs(a - rec_[(lab, s, 'pair')]['point']) <= 1e-9, refit_auc=a, stored=rec_[(lab, s, 'pair')]['point'])
    # utility vectors
    refs = {}
    for s in (0, 1):
        r = ctx.npz(cfg['roles']['INNER'][f'references_s{s}'])
        refs[s] = {'U0': [r['U0_1'], r['U0_2']], 'Ucal': [r['Ucal_1'], r['Ucal_2']]}

    def A_rows(name):
        out = []
        for s in (0, 1):
            for i, t in enumerate(core.TASKS):
                a = core.per_row(y[t], releases[(name, s)][i][0])
                for ref in ('U0', 'Ucal'):
                    b = core.per_row(y[t], refs[s][ref][i])
                    for metric, d in (('accuracy_loss', b['accuracy'] - a['accuracy']), ('nll_excess', a['nll'] - b['nll']), ('brier_excess', a['brier'] - b['brier'])):
                        iv = core.sim_interval(d.mean(), core.weighted_means(d, Wb), crit)
                        cap = core.CAPS[metric]
                        out.append({'endpoint': f'A|{name}|s{s}|{t}|{ref}|{metric}', **iv, 'cap': cap, 'point_pass': bool(iv['point'] <= cap),
                                    'status': core.status_cap(iv, cap)})
        return out

    def rdiff(a, b, v):
        pt = np.mean([rec_[(a, s, v)]['point'] - rec_[(b, s, v)]['point'] for s in (0, 1)])
        dr = np.mean([rec_[(a, s, v)]['draws'] - rec_[(b, s, v)]['draws'] for s in (0, 1)], axis=0)
        return pt, dr

    alias = {lab: all(np.array_equal(releases[(lab, s)][i][0], releases[(primary, s)][i][0]) and np.array_equal(releases[(lab, s)][i][1], releases[(primary, s)][i][1])
                      for s in (0, 1) for i in range(2)) for lab in labels} if primary else {}
    R.check('aliases_of_primary', {k: bool(v) for k, v in alias.items()} == {k: bool(v) for k, v in rep.get('aliases_of_primary', {}).items()}, mine=alias)

    def BC(P, comps, margin, prefix):
        rows = []
        for c in comps:
            if alias.get(c) and P == primary:
                rows.append({'endpoint': f'{prefix}|{P}_vs_{c}|alias', 'status': 'EXCLUDED_EXACT_ALIAS'}); continue
            pt, dr = rdiff(c, P, 'pair'); iv = core.sim_interval(pt, dr, crit)
            rows.append({'endpoint': f'{prefix}|{P}_vs_{c}|pair_benefit', **iv, 'margin': margin, 'status': core.status_margin(iv, margin)})
            for v in ('income', 'occupation'):
                pt, dr = rdiff(P, c, v); iv = core.sim_interval(pt, dr, crit)
                rows.append({'endpoint': f'{prefix}|{P}_vs_{c}|{v}_excess', **iv, 'cap': 0.010, 'status': core.status_cap(iv, 0.010)})
        return rows
    fam = []
    if primary:
        fam += A_rows(primary) + BC(primary, sel['B_required'], 0.020, 'B')
    fam += BC('J_TREE', sel['C_roster'], 0.0, 'C')
    if primary != 'J_TREE':
        fam += A_rows('J_TREE')
    theirs = {r['endpoint']: r for r in rep['family']}
    R.check('family_endpoint_set', sorted(r['endpoint'] for r in fam) == sorted(theirs), mine_n=len(fam), reported_n=len(theirs))
    disagreements = []
    for r in fam:
        o = theirs.get(r['endpoint'])
        if o is None:
            R.check(f'endpoint_{r["endpoint"]}', False, reason='missing in report'); continue
        if r['status'] == 'EXCLUDED_EXACT_ALIAS':
            R.check(f'endpoint_{r["endpoint"]}', o['status'] == 'EXCLUDED_EXACT_ALIAS'); continue
        bound = r.get('cap', r.get('margin'))
        dp = abs(r['point'] - o['point']); dl = abs(r['lower'] - o['lower']); du = abs(r['upper'] - o['upper'])
        same_status = r['status'] == o['status']
        borderline = (min(abs(o['lower'] - bound), abs(o['upper'] - bound)) <= TOL_INTERVAL)
        ok = dp <= 1e-10 and dl <= TOL_INTERVAL and du <= TOL_INTERVAL and (same_status or borderline)
        if not same_status:
            disagreements.append({'endpoint': r['endpoint'], 'mine': r['status'], 'reported': o['status'], 'borderline': borderline})
        R.check(f'endpoint_{r["endpoint"]}', ok, point_diff=dp, lower=[r['lower'], o['lower']], upper=[r['upper'], o['upper']],
                status=[r['status'], o['status']], borderline=borderline)
    R.details['status_disagreements'] = disagreements
    # labels from SCIENCE_LOCK rules (using the reported statuses for the final label replay, and own statuses as a second replay)
    def labels_from(rows):
        A = [r for r in rows if r['endpoint'].startswith(f'A|{primary}|')] if primary else []
        Bp = [r for r in rows if r['endpoint'].startswith('B|') and r['status'] != 'EXCLUDED_EXACT_ALIAS']
        Cr = [r for r in rows if r['endpoint'].startswith('C|') and r['status'] != 'EXCLUDED_EXACT_ALIAS']
        if not primary:
            s_label = 'NO_POINT_FEASIBLE_PROTECTED_NOMINEE'
        else:
            a_point = all(r['point_pass'] for r in A)
            b_ok = all(r['status'] == ('ESTABLISHED' if r['endpoint'].endswith('pair_benefit') else 'PASS') for r in Bp)
            if sel.get('B_unavailable'):
                s_label = 'COMPETITIVE_PROTECTION_INCOMPLETE'
            elif a_point and b_ok:
                s_label = 'COMPETITIVE_PROTECTION_ESTABLISHED'
            elif a_point:
                s_label = 'POINT_FEASIBLE_NO_COMPETITIVE_CRITERION'
            else:
                s_label = 'NOT_POINT_FEASIBLE_ON_INNER_NO_COMPETITIVE_CRITERION'
        j_int = sel['families']['J_TREE']['status'] == 'POINT_FEASIBLE_NOMINEE'
        c_ok = j_int and all(r['status'] == ('ESTABLISHED' if r['endpoint'].endswith('pair_benefit') else 'PASS') for r in Cr)
        return s_label, 'JOINT_COMPONENT_ESTABLISHED' if c_ok else 'JOINT_COMPONENT_NOT_ESTABLISHED'
    s_mine, j_mine = labels_from(fam)
    s_rep, j_rep = labels_from(list(rep['family']))
    R.check('scientific_label_own_statuses', s_mine == rep['scientific_label'], mine=s_mine, reported=rep['scientific_label'])
    R.check('scientific_label_replayed_from_reported_rows', s_rep == rep['scientific_label'], replay=s_rep)
    R.check('joint_label_own_statuses', j_mine == rep['joint_label'], mine=j_mine, reported=rep['joint_label'])
    R.check('joint_label_replayed_from_reported_rows', j_rep == rep['joint_label'], replay=j_rep)
    R.check('labels_in_SCIENCE_LOCK_vocabulary', rep['scientific_label'] in sci['labels']['scientific'] and rep['joint_label'] in sci['labels']['joint'])
    # controls
    cr = rep.get('controls', {})
    for s in (0, 1):
        sc = ctx.npz(cfg['readers'][str(s)]['scores'])
        for v in VIEWS:
            pl = core.auc(y['sex'], np.where(y['sex'] == 1, sc[f'CONTROL__{v}__PLANTED_1'], sc[f'CONTROL__{v}__PLANTED_0']))
            o = cr.get(f's{s}|{v}', {})
            R.check(f'control_planted_s{s}_{v}', pl >= 0.99 and abs(pl - o.get('planted_auc', -1)) <= 1e-12, mine=pl)
            const = core.auc(y['sex'], np.full(len(unit), 0.5))
            R.check(f'control_constant_s{s}_{v}', const == 0.5)
            R.check(f'control_shuffled_reported_gate_s{s}_{v}', o.get('shuffled_auc_abs', 1) <= 0.60, reported=o.get('shuffled_auc_abs'))
    R.details['labels'] = {'scientific': s_mine, 'joint': j_mine}
    R.details['recovery_points'] = {f'{lab}|s{s}|{v}': rec_[(lab, s, v)]['point'] for lab in labels for s in (0, 1) for v in VIEWS}


STAGES = {'SOURCE': stage_source, 'TEACHER': stage_teacher, 'STAGE_A': stage_stage_a, 'COUNTERFACTUALS': stage_counterfactuals,
          'FIT2': stage_fit2, 'WIRES': stage_wires, 'READERS': stage_readers, 'INNER': stage_inner}


def run(spec):
    from task_headroom.evaluation_queue import queue_config
    cfg0 = json.loads(Path(spec).read_text())
    active, cfg, verify, guard = queue_config(spec, 'verification', allow_inner_truth='INNER' in cfg0['stages'])
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    ctx = Ctx(cfg, verify, guard)
    results = {}
    for name in cfg['stages']:
        guard()
        R = Receipt(name + cfg.get('receipt_suffix', ''))
        try:
            STAGES[name](ctx, R)
        except Exception as exc:  # recorded as a FAIL receipt, then re-raised
            R.check('stage_completed_without_exception', False, error=f'{type(exc).__name__}: {exc}')
            R.write(active['unit_id'])
            raise
        results[name] = R.write(active['unit_id'])['status']
    print(json.dumps(results))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    run(ap.parse_args().spec)
