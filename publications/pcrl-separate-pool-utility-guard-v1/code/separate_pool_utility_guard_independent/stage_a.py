"""Independent numerical verifier for pcrl_separate_pool_utility_guard_v1 Stage A.

Queue-only worker (stage 'verification'). Every byte read is bound through the
queue's verify(path, sha). Arithmetic lives in .numerics (own numpy code);
runner modules are not imported except the queue infrastructure.
Writes aggregate numbers only (no person-level arrays).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse
import csv
import hashlib
import io
import json
import math
import time
import traceback
from pathlib import Path

ARMS = ('SEPARATE_GUARD', 'GLOBAL_UPDATE_GUARD', 'OLD_BATCH_GUARD', 'ORDINARY', 'TASK_ONLY')
GLOBAL = {'SEPARATE_GUARD': 'GUARD', 'GLOBAL_UPDATE_GUARD': 'UPDATE'}
EPOCHS = (0, 1, 3, 5)
TRACE_ROW = {0: 0, 1: 32, 3: 96, 5: 160}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
BUD = {'nll': 0.010, 'brier': 0.005, 'accuracy': 0.010}
FINITE_TOL = 1e-10
MAXNORM = 0.01
TASKS = ('income', 'occupation')


def st(ok):
    return 'PASS' if ok else 'FAIL'


def jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not math.isfinite(x):
        return repr(x)
    return x


class Ctx:
    def __init__(self, cfg, verify):
        self.cfg = cfg; self.verify = verify; self.read = []

    def path(self, item):
        p = self.verify(item['path'], item['sha256']); self.read.append(str(p)); return p

    def json(self, item):
        return json.loads(self.path(item).read_text())

    def npz(self, item):
        import numpy as np
        with np.load(self.path(item), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def npy(self, item):
        import numpy as np
        return np.load(self.path(item), allow_pickle=False)

    def ckpt(self, item):
        import torch
        p = self.path(item)
        try:
            payload = torch.load(p, map_location='cpu', weights_only=True)
        except Exception:
            payload = torch.load(p, map_location='cpu', weights_only=False)
        model = {k: v.detach().cpu().numpy().copy() for k, v in payload['model'].items()}
        meta = {k: payload.get(k) for k in ('seed', 'epoch', 'wire_variant')}
        return model, meta


# ---------------------------------------------------------------- check 1
def check_roles(c, D):
    import numpy as np
    from .numerics import fold_of, h, SPLIT_SALT, PROBE_SALT
    cfg = c.cfg
    inh = {k: c.npz(v) for k, v in cfg['inherited'].items()}
    F = inh['features']; unit = F['unit']; rid = F['row_id']; n = len(rid)
    align = all(np.array_equal(inh[k]['row_id'], rid) and np.array_equal(inh[k]['unit'], unit) for k in inh)
    folds = np.array([fold_of(u) for u in unit])
    ustr = np.array([str(u) for u in unit])
    gfold = {}
    for s, f in zip(ustr, folds):
        gfold.setdefault(s, set()).add(int(f))
    crossing = sum(1 for v in gfold.values() if len(v) > 1)
    pool = np.flatnonzero(folds < 2); val = np.flatnonzero(folds == 2)
    groups = sorted({ustr[i] for i in pool}, key=lambda g: (h(SPLIT_SALT + g), g))
    n_guard = math.floor(0.2 * len(groups))
    gset = set(groups[:n_guard])
    isg = np.array([s in gset for s in ustr])
    upd = pool[~isg[pool]]; grd = pool[isg[pool]]
    sets = {'UPDATE': upd, 'GUARD': grd, 'VALIDATION': val}
    ex = {r: {k: c.npz(v) for k, v in d.items()} for r, d in cfg['exports'].items()}
    D['exports'] = ex
    roster_ok = (set(ex['UPDATE']) == {'features', 'occupation', 'sex'} and set(ex['GUARD']) == {'features', 'occupation'}
                 and set(ex['VALIDATION']) == {'features', 'income', 'occupation', 'references_s0', 'references_s1'})
    def same(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    byte = {}
    for role, idx in sets.items():
        for name, arrs in ex[role].items():
            src = inh[name]
            ok = set(arrs) == set(src)
            for k in arrs:
                if k == 'feature_names':
                    ok = ok and same(arrs[k], src[k])
                else:
                    ok = ok and same(arrs[k], src[k][idx])
            byte[f'{role}/{name}'] = ok
    eg = {r: set(map(str, ex[r]['features']['unit'])) for r in ex}
    overlap = {f'{a}__{b}': len(eg[a] & eg[b]) for a in eg for b in eg if a < b}
    # Public / private split records.
    pub = c.json(cfg['files']['role_split']); priv = c.json(cfg['files']['role_split_private'])
    rpaths = c.json(cfg['files']['role_paths'])
    probe = sorted(range(len(upd)), key=lambda i: (h(PROBE_SALT + str(rid[upd[i]])), i))[:512]
    D['probe_positions'] = probe
    counts_ok = {
        'eligible_groups': pub['eligible_groups'] == len(groups),
        'guard_groups': pub['guard_groups'] == n_guard == len(groups) // 5,
        'inherited_fit_rows': pub['inherited_fit_rows'] == n,
        'inherited_fit_groups': pub['inherited_fit_groups'] == len(gfold),
        'inherited_fold_rows': pub['inherited_fold_rows'] == {str(k): int((folds == k).sum()) for k in range(3)},
        'duplicate_groups_crossing': pub['duplicate_groups_crossing_inherited_folds'] == crossing == 0,
        'group_overlap_recorded_zero': pub['group_overlap'] == overlap and not any(overlap.values()),
        'probe_position_sha256': pub['probe_position_sha256'] == hashlib.sha256(json.dumps(probe).encode()).hexdigest(),
        'private_split_sha256': pub['private_split_sha256'] == cfg['files']['role_split_private']['sha256'],
    }
    for role, idx in sets.items():
        sup = pub['support'][role]
        counts_ok[f'{role}_rows'] = sup['rows'] == len(idx)
        counts_ok[f'{role}_groups'] = sup['groups'] == len(set(ustr[idx]))
        counts_ok[f'{role}_occ_counts'] = sup['occupation_class_counts'] == np.bincount(inh['occupation']['occupation'][idx], minlength=6).tolist()
        if role == 'VALIDATION':
            counts_ok['VALIDATION_income_counts'] = sup['income_class_counts'] == np.bincount(inh['income']['income'][idx], minlength=2).tolist()
        if role == 'UPDATE':
            counts_ok['UPDATE_sex_counts'] = sup['sex_class_counts'] == np.bincount(inh['sex']['sex'][idx], minlength=2).tolist()
        counts_ok[f'{role}_index_sha256'] = pub['role_index_sha256'][role] == hashlib.sha256(json.dumps(idx.tolist()).encode()).hexdigest()
        counts_ok[f'{role}_group_sha256'] = pub['role_group_sha256'][role] == hashlib.sha256(json.dumps(sorted(set(ustr[idx]))).encode()).hexdigest()
        counts_ok[f'{role}_private_indices'] = priv['fit_row_indices'][role] == idx.tolist()
        for name in ex[role]:
            counts_ok[f'{role}_{name}_export_sha_matches_ROLE_PATHS'] = (
                rpaths['sha256'][role][name] == cfg['exports'][role][name]['sha256'] == pub['exports'][role][name]['sha256'])
    counts_ok['private_guard_groups'] = priv['guard_groups'] == sorted(gset)
    counts_ok['private_probe_positions'] = priv['update_probe_positions'] == probe
    # Trainer exposure: specs/configs (hashed) plus live queue state (infrastructure read, mutable, not hashed).
    trainer = {}
    state = json.loads((Path(os.environ['PCRL_QUEUE_ROOT']) / 'state.json').read_text())
    for s in ('0', '1'):
        uid = f'SPG_TRAIN_S{s}_V1'
        spec = c.json(cfg['files'][f'trainer_spec_s{s}']); tcfg = c.json(cfg['files'][f'trainer_cfg_s{s}'])
        live = state['units'][uid]
        def novalid(paths):
            return not any('validation' in p.lower() for p in paths)
        trainer[uid] = {
            'spec_file_sources_without_VALIDATION': novalid([x['path'] for x in spec['sources']]),
            'state_json_sources_without_VALIDATION': novalid([x['path'] for x in live['spec']['sources']]),
            'state_json_sources_equal_spec_file': live['spec']['sources'] == spec['sources'],
            'config_roles_only_UPDATE_GUARD': set(tcfg['roles']) == {'UPDATE', 'GUARD'} and novalid(json.dumps(tcfg['roles']).split('"')),
            'GUARD_sex_not_a_trainer_input': 'sex' not in tcfg['roles']['GUARD'],
            'status': live['status'], 'source_count': len(spec['sources'])}
    trainer_ok = all(all(v for k, v in d.items() if k not in ('status', 'source_count')) and d['status'] == 'done' for d in trainer.values())
    ok = align and roster_ok and all(byte.values()) and not any(overlap.values()) and all(counts_ok.values()) and trainer_ok and crossing == 0
    return {'status': st(ok), 'kind': 'recomputation from inherited FIT arrays (no model forward)',
            'inherited_label_feature_pairing': align, 'export_rosters_ok': roster_ok,
            'byte_identical_exports': byte, 'export_group_overlap': overlap,
            'rows': {k: int(len(v)) for k, v in sets.items()}, 'eligible_groups': len(groups), 'guard_groups': n_guard,
            'duplicate_groups_crossing_folds': crossing, 'record_agreement': counts_ok,
            'failed_record_items': [k for k, v in counts_ok.items() if not v], 'trainer_exposure': trainer}


# ---------------------------------------------------------------- check 2
def check_income(c, D):
    import numpy as np
    from .numerics import INC_KEYS, OCC_KEYS, income_wire
    cfg = c.cfg
    X = D['exports']['VALIDATION']['features']['X']
    out = {'seeds': {}}; ok_all = True
    D['src'] = {}; D['B'] = {}; D['ckpt'] = {}
    for s in ('0', '1'):
        src, _ = c.ckpt(cfg['source_checkpoints'][s]); D['src'][s] = src
        g = c.json(cfg['geometry'][s]); g0 = g['heads'][0]
        B = np.asarray(g0['B'], dtype=np.float64).reshape(g0['rank'], g0['source_dimension']); D['B'][s] = B
        head_match_geom = (np.array_equal(np.asarray(g0['weight'], dtype=np.float64), src['head.0.weight'])
                           and np.array_equal(np.asarray(g0['bias'], dtype=np.float64), src['head.0.bias']))
        ref = {m: income_wire(src, X, B, m) for m in ('f32', 'f64')}
        full = {m: income_wire(src, X, B, m, project=False) for m in ('f32', 'f64')}
        proj = {m: {'max_abs_prob_diff_projected_vs_unprojected': float(np.max(np.abs(ref[m]['probability'] - full[m]['probability']))),
                    'max_abs_centered_logit_diff': float(np.max(np.abs(ref[m]['centered_logits'] - full[m]['centered_logits']))),
                    'decision_mismatches': int(np.sum(ref[m]['decision'] != full[m]['decision']))} for m in ref}
        tensor_ok = {}; release_identical = {}; epoch0_occ = {}; ids = {}
        for arm in ARMS:
            for e in EPOCHS:
                model, meta = c.ckpt(cfg['training'][s][arm]['checkpoints'][str(e)])
                D['ckpt'][(s, arm, e)] = model
                key = f'{arm}|{e}'
                ids[key] = meta['seed'] == int(s) and meta['epoch'] == e and meta['wire_variant'] == arm
                tensor_ok[key] = all(model[k].dtype == src[k].dtype and model[k].shape == src[k].shape
                                     and model[k].tobytes() == src[k].tobytes() for k in INC_KEYS)
                w = income_wire(model, X, B, 'f32')
                release_identical[key] = all(np.array_equal(w[k], ref['f32'][k]) for k in w)
                if e == 0:
                    epoch0_occ[arm] = all(model[k].tobytes() == src[k].tobytes() and model[k].dtype == src[k].dtype for k in OCC_KEYS)
        D['income_ref'] = ref
        proj_ok = all(v['max_abs_prob_diff_projected_vs_unprojected'] <= 1e-10 and v['decision_mismatches'] == 0 for v in proj.values())
        ok = all(tensor_ok.values()) and all(release_identical.values()) and all(epoch0_occ.values()) and all(ids.values()) and proj_ok and B.shape[0] == 1
        ok_all &= ok
        out['seeds'][s] = {'checkpoints_checked': len(tensor_ok), 'income_tensors_bitwise_equal_source': sum(tensor_ok.values()),
                           'own_VALIDATION_income_release_bitwise_identical_to_source': sum(release_identical.values()),
                           'epoch0_occupation_tensors_bitwise_equal_source': epoch0_occ,
                           'checkpoint_identity_fields_ok': sum(ids.values()), 'basis_rank': int(B.shape[0]),
                           'source_head0_equals_geometry_weight_bias': bool(head_match_geom),
                           'rank1_projection_parity': proj, 'f32_vs_f64_income_prob_max_abs': float(np.max(np.abs(ref['f32']['probability'] - ref['f64']['probability'])))}
    out['status'] = st(ok_all); out['kind'] = 'new model forward (own numpy) + bitwise tensor comparison'
    return out


# ---------------------------------------------------------------- check 3
def check_accounting(c, D):
    cfg = c.cfg
    lock = c.json({'path': cfg['training_complete_lock'], 'sha256': cfg['training_complete_lock_sha256']}); D['lock'] = lock
    nU = len(D['exports']['UPDATE']['features']['row_id']); nG = len(D['exports']['GUARD']['features']['row_id'])
    wU = nU / (nU + nG); wG = nG / (nU + nG)
    P = math.ceil(nU / 256); expected_steps = (80 + 5 * 160) * 6
    out = {'N_UPDATE': nU, 'N_GUARD': nG, 'weights': {'update': wU, 'guard': wG}, 'proposals_per_epoch': P,
           'expected_critic_steps': expected_steps, 'seeds': {}}
    ok_all = P == 32 and lock.get('proposals_per_epoch', 32) == 32
    D['logs'] = {}; D['train'] = {}
    for s in ('0', '1'):
        T = c.json(cfg['files'][f'train_s{s}']); D['train'][s] = T
        coef_ok = T['coefficient'] == lock['coefficients'][s]
        arms = {}
        for arm in ARMS:
            a = T['arms'][arm]; item = cfg['training'][s][arm]
            lines = [json.loads(l) for l in c.path(item['UPDATE_LOG']).read_text().splitlines() if l.strip()]
            D['logs'][(s, arm)] = lines
            summ = c.json(item['SUMMARY'])
            counts = {}
            for r in lines:
                counts[r['status']] = counts.get(r['status'], 0) + 1
            per_epoch = {}
            for r in lines:
                d = per_epoch.setdefault(r['epoch'], {'accepted': 0, 'rejected': 0, 'n': 0})
                d['n'] += 1; d['accepted'] += r['status'] == 'ACCEPTED'; d['rejected'] += r['status'] == 'REJECTED_ZERO_STEP'
            ep_ok = all(per_epoch.get(row['epoch'], {}).get('accepted') == row['accepted'] and per_epoch[row['epoch']]['rejected'] == row['rejected']
                        and per_epoch[row['epoch']]['n'] == row['proposals'] == P for row in a['epoch_accounting'])
            checks = {
                'task_weights_exact': a['task_weights']['update'] == wU and a['task_weights']['guard'] == wG,
                'task_weight_max_abs_diff': max(abs(a['task_weights']['update'] - wU), abs(a['task_weights']['guard'] - wG)),
                'proposals_160': len(lines) == 160 and [r['update_number'] for r in lines] == list(range(1, 161)) and a['model_updates'] == 160,
                'epoch_fields': [r['epoch'] for r in lines] == [1 + (k // P) for k in range(160)],
                'critic_steps_summary': a['critic_optimizer_steps'] == expected_steps and summ['critic_optimizer_steps'] == expected_steps,
                'critic_steps_per_line': all(r['critic_optimizer_steps'] == 480 + 30 * (i + 1) for i, r in enumerate(lines)),
                'status_counts_match_summary': counts == a['update_status_counts'] == summ['update_status_counts'],
                'status_counts_match_lock': counts == lock['proposal_accounting'][s][arm],
                'epoch_accounting_matches_log': ep_ok,
                'coefficient_shared': a['coefficient'] == T['coefficient'],
                'private_summary_sha_in_lock': item['SUMMARY']['sha256'] == lock['logs'][s][arm]['SUMMARY'],
            }
            ok = all(v for k, v in checks.items() if k != 'task_weight_max_abs_diff')
            ok_all &= ok
            arms[arm] = dict(checks, status_counts=counts, status=st(ok))
        ok_all &= coef_ok
        out['seeds'][s] = {'coefficient_matches_lock': coef_ok, 'coefficient': T['coefficient'], 'arms': arms}
    out['status'] = st(ok_all); out['kind'] = 'stored-record arithmetic (no model forward)'
    return out


# ---------------------------------------------------------------- check 4
def _runner_state_metrics(lines):
    """Runner metric at trace state k: log k+1 'current'; state 160 from the last post/current."""
    m = [r['result'].get('current') for r in lines]
    last = lines[-1]['result']
    m.append(last['post_metrics'] if lines[-1]['status'] == 'ACCEPTED' else last.get('current'))
    return m


def check_global(c, D):
    import numpy as np
    from .numerics import unflatten_occ, flatten_occ, occupation_wire, pool_metrics, l2, OCC_KEYS
    cfg = c.cfg
    out = {'seeds': {}}; ok_all = True; marginal = []
    D['traces'] = {}
    csv_rows = list(csv.DictReader(io.StringIO(c.path(cfg['files']['budget_csv']).read_text())))
    for s in ('0', '1'):
        T = D['train'][s]; seed_out = {}
        for arm in ARMS:
            item = cfg['training'][s][arm]
            tr = c.npy(item['STATE_TRACE']); D['traces'][(s, arm)] = tr
            lines = D['logs'][(s, arm)]
            r = {'trace_shape': list(tr.shape)}
            enc_roundtrip = bool(np.array_equal(tr[:, :10576], tr[:, :10576].astype(np.float32).astype(np.float64)))
            ck_eq = {str(e): bool(np.array_equal(tr[TRACE_ROW[e]], flatten_occ(D['ckpt'][(s, arm, e)]))) for e in EPOCHS}
            diffs = np.diff(tr, axis=0); norms = np.sqrt(np.sum(diffs * diffs, axis=1))
            stat = [x['status'] for x in lines]
            rej_unchanged = all(np.array_equal(tr[k + 1], tr[k]) for k in range(160) if stat[k] != 'ACCEPTED')
            acc_moved = all(np.any(tr[k + 1] != tr[k]) for k in range(160) if stat[k] == 'ACCEPTED')
            lognorm = max(abs(norms[k] - lines[k]['actual_norm']) for k in range(160))
            r.update(encoder_coordinates_float32_exact=enc_roundtrip, trace_rows_equal_checkpoints=ck_eq,
                     max_step_norm=float(norms.max()), step_norm_margin=float(MAXNORM - norms.max()),
                     all_steps_within_norm=bool(np.all(norms <= MAXNORM + 1e-12)),
                     rejected_unchanged_bitwise=rej_unchanged, accepted_moved=acc_moved,
                     logged_actual_norm_max_abs_diff=float(lognorm),
                     statuses=sorted(set(stat)))
            ok = enc_roundtrip and all(ck_eq.values()) and r['all_steps_within_norm'] and rej_unchanged and acc_moved and lognorm <= 1e-15
            if arm in GLOBAL:
                pool = GLOBAL[arm]; X = D['exports'][pool]['features']['X']; y = D['exports'][pool]['occupation']['occupation']
                anc = T['anchors'][pool]; rec = T['arms'][arm]['anchor']
                N = len(y)
                lim = {'nll': anc['nll'] + BUD['nll'], 'brier': anc['brier'] + BUD['brier']}
                drop = math.floor(BUD['accuracy'] * N + 1e-9); floor_c = anc['correct'] - drop
                limits_ok = (rec['nll_limit'] == lim['nll'] and rec['brier_limit'] == lim['brier'] and rec['allowed_correct_drop'] == drop
                             and rec['correct_floor'] == floor_c and rec['start'] == anc and T['anchors'][pool]['rows'] == N
                             and D['lock']['anchors'][s][pool] == anc)
                mine = {m: [] for m in ('f32', 'f64')}
                for k in range(161):
                    stt = unflatten_occ(tr[k])
                    for m in mine:
                        w = occupation_wire(stt, X, m)
                        mine[m].append(pool_metrics(w['probability'], w['decision'], y))
                runner = _runner_state_metrics(lines)
                # anchors
                a32 = mine['f32'][0]; a64 = mine['f64'][0]
                anchor_diff = {'nll_f32': abs(a32['nll'] - anc['nll']), 'brier_f32': abs(a32['brier'] - anc['brier']),
                               'nll_f64': abs(a64['nll'] - anc['nll']), 'brier_f64': abs(a64['brier'] - anc['brier']),
                               'correct_f32': a32['correct'] - anc['correct'], 'correct_f64': a64['correct'] - anc['correct']}
                # slacks
                slack_diff = {m: max(max(abs(lines[k]['result']['raw_slacks'][0] - (lim['nll'] - mine[m][k]['nll'])),
                                         abs(lines[k]['result']['raw_slacks'][1] - (lim['brier'] - mine[m][k]['brier']))) for k in range(160))
                              for m in mine}
                # runner vs mine per state
                rv = {m: {'nll': max(abs(runner[k]['nll'] - mine[m][k]['nll']) for k in range(161)),
                          'brier': max(abs(runner[k]['brier'] - mine[m][k]['brier']) for k in range(161)),
                          'correct_mismatch_states': sum(runner[k]['correct'] != mine[m][k]['correct'] for k in range(161))} for m in mine}
                prec = {'nll': max(abs(mine['f32'][k]['nll'] - mine['f64'][k]['nll']) for k in range(161)),
                        'brier': max(abs(mine['f32'][k]['brier'] - mine['f64'][k]['brier']) for k in range(161)),
                        'correct_mismatch_states': sum(mine['f32'][k]['correct'] != mine['f64'][k]['correct'] for k in range(161))}
                post_cons = all({q: lines[k]['result']['post_metrics'][q] for q in ('nll', 'brier', 'correct')} == lines[k + 1]['result']['current'] for k in range(159) if stat[k] == 'ACCEPTED')
                cur_const = all(lines[k + 1]['result']['current'] == lines[k]['result']['current'] for k in range(159) if stat[k] != 'ACCEPTED')
                # budgets with margins
                margins = {'nll': [], 'brier': [], 'correct': []}; nstat = {'PASS': 0, 'FAIL': 0, 'NUMERICALLY_MARGINAL': 0}
                discs = {'nll': [], 'brier': []}
                for k in range(161):
                    for q in ('nll', 'brier'):
                        mg = min(lim[q] - mine['f32'][k][q], lim[q] - mine['f64'][k][q])
                        disc = max(abs(mine['f32'][k][q] - mine['f64'][k][q]), abs(runner[k][q] - mine['f32'][k][q]), 1e-15)
                        margins[q].append(mg); discs[q].append(disc)
                        if mg > disc: nstat['PASS'] += 1
                        elif mg < -(FINITE_TOL + disc): nstat['FAIL'] += 1; marginal.append((s, arm, k, q, 'FAIL', mg))
                        else: nstat['NUMERICALLY_MARGINAL'] += 1; marginal.append((s, arm, k, q, 'NUMERICALLY_MARGINAL', mg, disc))
                    cm = min(mine['f32'][k]['correct'], mine['f64'][k]['correct'], runner[k]['correct']) - floor_c
                    margins['correct'].append(cm)
                    if cm >= 0: nstat['PASS'] += 1
                    else: nstat['FAIL'] += 1; marginal.append((s, arm, k, 'correct', 'FAIL', cm))
                ratio = {q: float(min(m / d for m, d in zip(margins[q], discs[q]))) for q in discs}
                disc_at_min = {q: float(discs[q][int(np.argmin(margins[q]))]) for q in discs}
                tight = {}
                for k in range(161):
                    if margins['correct'][k] <= 1:
                        w = occupation_wire(unflatten_occ(tr[k]), X, 'f64'); ps = np.sort(w['probability'], axis=1)
                        tight[str(k)] = {'correct_margin': int(margins['correct'][k]), 'min_top2_probability_gap': float(np.min(ps[:, -1] - ps[:, -2]))}
                runner_margin = {'nll': min(lim['nll'] - runner[k]['nll'] for k in range(161)),
                                 'brier': min(lim['brier'] - runner[k]['brier'] for k in range(161)),
                                 'correct': min(runner[k]['correct'] for k in range(161)) - floor_c}
                rows = [x for x in csv_rows if x['seed'] == s and x['arm'] == arm]
                csv_ok = (len(rows) == 160 and {float(x['nll_limit']) for x in rows} == {lim['nll']}
                          and {float(x['brier_limit']) for x in rows} == {lim['brier']} and {int(x['correct_floor']) for x in rows} == {floor_c}
                          and all(float(x['slack_nll']) == lines[i]['result']['raw_slacks'][0] and float(x['slack_brier']) == lines[i]['result']['raw_slacks'][1]
                                  for i, x in enumerate(sorted(rows, key=lambda z: int(z['update'])))))
                anc_ok = (anchor_diff['nll_f32'] <= 1e-6 and anchor_diff['brier_f32'] <= 1e-6 and anchor_diff['correct_f32'] == 0)
                gok = (limits_ok and anc_ok and nstat['FAIL'] == 0 and slack_diff['f32'] <= 1e-6 and post_cons and cur_const and csv_ok)
                ok = ok and gok
                r['global_guard'] = {'pool': pool, 'pool_rows': N, 'limits': dict(lim, correct_floor=floor_c, allowed_drop=drop),
                                     'limits_and_anchor_record_exact': limits_ok, 'anchor_abs_diff_vs_TRAIN': anchor_diff,
                                     'raw_slack_max_abs_diff': slack_diff, 'runner_state_metric_vs_mine': rv,
                                     'f32_vs_f64_max_abs': prec, 'post_metrics_equal_next_current': post_cons,
                                     'current_constant_across_rejections': cur_const, 'csv_limits_constant_and_slacks_equal': csv_ok,
                                     'min_margin_mine': {k: (float(min(v)) if k != 'correct' else int(min(v))) for k, v in margins.items()},
                                     'argmin_state': {k: int(np.argmin(v)) for k, v in margins.items()},
                                     'min_margin_runner_logged': runner_margin, 'discrepancy_at_min_margin_state': disc_at_min,
                                     'min_margin_to_discrepancy_ratio': ratio, 'states_with_correct_margin_le_1': tight, 'state_budget_classification_counts': nstat,
                                     'status': st(gok)}
            ok_all &= ok
            r['status'] = st(ok)
            seed_out[arm] = r
        out['seeds'][s] = seed_out
    out['marginal_or_failing_states'] = [list(x) for x in marginal][:50]
    out['status'] = st(ok_all) if not any(x[4] == 'NUMERICALLY_MARGINAL' for x in marginal) or not ok_all else 'PASS_WITH_NUMERICALLY_MARGINAL_STATES'
    out['kind'] = 'new model forward (own numpy, f32 and f64) on the whole fixed pool at all 161 stored trace states; trace/norm checks on stored vectors'
    return out


# ---------------------------------------------------------------- check 5
def _change(a, b):
    import numpy as np
    out = {k: float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) for k in ('representation', 'centered_logits', 'probability')}
    out['decision_changes'] = int(np.sum(a['decision'] != b['decision']))
    return out


def check_privacy(c, D):
    import numpy as np
    from .numerics import unflatten_occ, occupation_wire, l2
    cfg = c.cfg
    Xp = D['exports']['UPDATE']['features']['X'][D['probe_positions']]
    out = {'seeds': {}}; ok_all = True; D['privacy_evidence'] = {}
    tol = 1e-10
    for s in ('0', '1'):
        T = D['train'][s]; lines = D['logs'][(s, 'SEPARATE_GUARD')]; tr = D['traces'][(s, 'SEPARATE_GUARD')]
        scans = c.json(cfg['training'][s]['SEPARATE_GUARD']['COUNTERFACTUAL_SCAN'])
        same_as_train = scans == json.loads(json.dumps(T['arms']['SEPARATE_GUARD']['counterfactual_scan']))
        first8 = [r['update_number'] for r in lines if r['status'] == 'ACCEPTED' and r['actual_norm'] > 0][:8]
        rows = []
        for sc in scans:
            u = sc['update_number']
            dn = l2(tr[u] - tr[u - 1])
            pre = occupation_wire(unflatten_occ(tr[u - 1]), Xp, 'f32'); post = occupation_wire(unflatten_occ(tr[u]), Xp, 'f32')
            ch = _change(post, pre); lg = sc['update_probe_wire_change']
            moved = ch['decision_changes'] > 0 or max(ch['representation'], ch['centered_logits'], ch['probability']) > tol
            beyond = sc['privacy_minus_counterfactual_max_abs'] > sc['contribution_tolerance']
            rows.append({'update_number': u, 'status_in_log': lines[u - 1]['status'],
                         'trace_norm_minus_privacy_actual_norm': dn - sc['privacy_actual_norm'],
                         'own_probe_change': ch, 'own_minus_logged_change': {k: ch[k] - lg[{'probability': 'probabilities'}.get(k, k)] for k in ('representation', 'centered_logits', 'probability')},
                         'decision_changes_logged': lg['decision_changes'], 'own_wire_changed_beyond_1e-10': moved,
                         'privacy_minus_counterfactual_max_abs': sc['privacy_minus_counterfactual_max_abs'],
                         'contribution_tolerance': sc['contribution_tolerance'],
                         'tolerance_rule_consistent': sc['contribution_tolerance'] == max(1e-10, 10 * sc['repeat_computation_proposal_max_abs_noise']),
                         'contribution_margin_ratio': sc['privacy_minus_counterfactual_max_abs'] / sc['contribution_tolerance'],
                         'stored_beyond_tolerance': beyond, 'counterfactual_status': sc['counterfactual_status']})
        ok = (same_as_train and len(scans) == 8 and [x['update_number'] for x in rows] == first8
              and all(x['status_in_log'] == 'ACCEPTED' and abs(x['trace_norm_minus_privacy_actual_norm']) <= 1e-15
                      and x['own_wire_changed_beyond_1e-10'] and x['stored_beyond_tolerance'] and x['tolerance_rule_consistent'] for x in rows))
        D['privacy_evidence'][s] = bool(ok and any(x['own_wire_changed_beyond_1e-10'] and x['stored_beyond_tolerance'] for x in rows))
        # Checkpoint-level probe flags vs lock (all arms).
        flags = {}; flags_ok = True
        for arm in ARMS:
            w0 = occupation_wire(D['ckpt'][(s, arm, 0)], Xp, 'f32')
            for e in EPOCHS:
                ch = _change(occupation_wire(D['ckpt'][(s, arm, e)], Xp, 'f32'), w0)
                changed = ch['decision_changes'] > 0 or max(ch['representation'], ch['centered_logits'], ch['probability']) > tol
                flags[f'{arm}|{e}'] = changed
                flags_ok &= changed == D['lock']['probe_wire_changed'][s][arm][str(e)]
        ok = ok and flags_ok
        ok_all &= ok
        out['seeds'][s] = {'scan_file_equals_TRAIN_json': same_as_train, 'scanned_updates': [x['update_number'] for x in rows],
                           'first_eight_accepted_nonzero_in_log': first8, 'scans': rows,
                           'max_abs_trace_norm_minus_privacy_actual_norm': max(abs(x['trace_norm_minus_privacy_actual_norm']) for x in rows),
                           'min_contribution_margin_ratio': min(x['contribution_margin_ratio'] for x in rows),
                           'max_abs_own_minus_logged_probe_change': max(max(abs(v) for v in x['own_minus_logged_change'].values()) for x in rows),
                           'decision_change_count_mismatches': sum(x['own_probe_change']['decision_changes'] != x['decision_changes_logged'] for x in rows),
                           'checkpoint_probe_wire_changed_own': flags, 'checkpoint_probe_flags_match_lock': flags_ok,
                           'privacy_bearing_accepted_wire_changing_update': D['privacy_evidence'][s],
                           'status': st(ok)}
    out['status'] = st(ok_all)
    out['kind'] = ('STORED-VECTOR check for the counterfactual displacement (privacy_minus_counterfactual and its tolerance are runner-stored; '
                   'counterfactual QP not re-solved); trace diff norms from stored STATE_TRACE; probe pre/post wire is a NEW own-numpy forward')
    return out


# ---------------------------------------------------------------- check 6
def check_validation(c, D):
    import numpy as np
    from .numerics import occupation_wire, income_wire, full_metrics, deltas
    cfg = c.cfg
    V = c.json(cfg['files']['validation_full']); lock = D['lock']
    ex = D['exports']['VALIDATION']; X = ex['features']['X']
    y = {'income': ex['income']['income'], 'occupation': ex['occupation']['occupation']}
    pair_ok = all(np.array_equal(ex[k]['row_id'], ex['features']['row_id']) for k in ('income', 'occupation', 'references_s0', 'references_s1'))
    keys = ('accuracy', 'nll', 'brier', 'balanced_accuracy', 'confidence')
    mx = {'metrics': 0.0, 'per_class_recall': 0.0, 'incremental': 0.0, 'U0': 0.0, 'Ucal': 0.0, 'f32_vs_f64_delta': 0.0}
    class_counts_ok = True; pass_agree = True; table = {}; marg = []; wire = {}
    for s in ('0', '1'):
        B = D['B'][s]; src = D['src'][s]
        def mets(model, mode):
            wi = income_wire(model, X, B, mode); wo = occupation_wire(model, X, mode)
            return {'income': full_metrics(wi['probability'], y['income']), 'occupation': full_metrics(wo['probability'], y['occupation'])}, wo
        start = {m: mets(src, m)[0] for m in ('f32', 'f64')}
        start_wo = occupation_wire(src, X, 'f32')
        refm = {}
        for ref in ('U0', 'Ucal'):
            refm[ref] = {t: full_metrics(np.asarray(ex[f'references_s{s}'][f'{ref}_{i + 1}'], dtype=np.float64), y[t]) for i, t in enumerate(TASKS)}
        for arm in ARMS:
            for e in EPOCHS:
                key = f'{s}|{arm}|{e}'
                m32, wo = mets(D['ckpt'][(s, arm, e)], 'f32'); m64, _ = mets(D['ckpt'][(s, arm, e)], 'f64')
                ch = {k: float(np.max(np.abs(wo[k] - start_wo[k]))) for k in ('representation', 'centered_logits', 'probability')}
                ch['decision_changes'] = int(np.sum(wo['decision'] != start_wo['decision']))
                wire[key] = ch
                rv = V['table'][key]; ent = {}
                for t in TASKS:
                    mine = m32[t]; run = rv[t]
                    for k in keys:
                        mx['metrics'] = max(mx['metrics'], abs(mine[k] - run['metrics'][k]))
                    mx['per_class_recall'] = max(mx['per_class_recall'], max(abs(a - b) for a, b in zip(mine['per_class_recall'], run['metrics']['per_class_recall'])))
                    class_counts_ok &= mine['class_counts'] == run['metrics']['class_counts']
                    inc = deltas(mine, start['f32'][t]); inc64 = deltas(m64[t], start['f64'][t])
                    for k in CAPS:
                        mx['incremental'] = max(mx['incremental'], abs(inc[k] - run['incremental_vs_start'][k]))
                        mx['f32_vs_f64_delta'] = max(mx['f32_vs_f64_delta'], abs(inc[k] - inc64[k]))
                    for ref in ('U0', 'Ucal'):
                        od = deltas(mine, refm[ref][t])
                        for k in CAPS:
                            mx[ref] = max(mx[ref], abs(od[k] - run['original_reference_deltas'][ref][k]))
                    passes = True; cls = {}
                    for k, cap in CAPS.items():
                        mg = cap - inc[k]
                        disc = max(abs(inc[k] - inc64[k]), abs(inc[k] - run['incremental_vs_start'][k]), 1e-15)
                        cls[k] = {'delta': inc[k], 'margin': mg, 'discrepancy': disc,
                                  'class': 'PASS' if mg > disc else ('FAIL' if mg < -disc else 'NUMERICALLY_MARGINAL')}
                        if cls[k]['class'] == 'NUMERICALLY_MARGINAL':
                            marg.append([key, t, k, mg, disc])
                        passes &= inc[k] <= cap
                    pass_agree &= passes == run['incremental_proxy_pass']
                    ent[t] = {'pass': passes, 'caps': cls}
                table[key] = ent
    sel = {}
    for arm in ARMS:
        feas = [e for e in EPOCHS if all(table[f'{s}|{arm}|{e}'][t]['pass'] for s in ('0', '1') for t in TASKS)]
        sel[arm] = {'feasible_common_epochs': feas, 'chosen_epoch': max(feas) if feas else 0}
    sel_agree = all(sel[a]['feasible_common_epochs'] == V['selection'][a]['feasible_common_epochs']
                    and sel[a]['chosen_epoch'] == V['selection'][a]['chosen_epoch'] for a in ARMS)
    sep = sel['SEPARATE_GUARD']['chosen_epoch']
    def wchanged(ch):
        return ch['decision_changes'] > 0 or max(ch['representation'], ch['centered_logits'], ch['probability']) > 1e-10
    crit = {'chosen_common_epoch_positive': sep > 0,
            'released_occupation_wire_changed_both_seeds': sep > 0 and all(lock['probe_wire_changed'][s]['SEPARATE_GUARD'][str(sep)] for s in ('0', '1')),
            'validation_wire_changed_both_seeds': sep > 0 and all(wchanged(wire[f'{s}|SEPARATE_GUARD|{sep}']) for s in ('0', '1')),
            'privacy_bearing_accepted_update_both_seeds': all(D['privacy_evidence'].get(s, False) for s in ('0', '1')),
            'all_incremental_validation_proxies_pass_both_seeds': sep > 0}
    gate = 'PASS_PENDING_INDEPENDENT_REPLAY' if all(crit.values()) else 'FAIL'
    crit_agree = crit == V['separate_guard_criteria']
    # SEPARATE_GUARD cap failures, per epoch, for context.
    sep_fail = {}
    for s in ('0', '1'):
        for e in EPOCHS[1:]:
            for t in TASKS:
                for k, v in table[f'{s}|SEPARATE_GUARD|{e}'][t]['caps'].items():
                    if v['delta'] > CAPS[k]:
                        sep_fail.setdefault(f'{s}|{e}', []).append({'task': t, 'cap': k, 'delta': v['delta'], 'margin': v['margin']})
    wire_rv = max(abs(wire[k][q] - V['validation_wire_change_vs_start'][k][q]) for k in wire for q in ('representation', 'centered_logits', 'probability'))
    wire_dec = sum(wire[k]['decision_changes'] != V['validation_wire_change_vs_start'][k]['decision_changes'] for k in wire)
    tol_ok = mx['metrics'] <= 1e-6 and mx['per_class_recall'] == 0 and mx['incremental'] <= 1e-6 and mx['U0'] <= 1e-6 and mx['Ucal'] <= 1e-6
    ok = (pair_ok and tol_ok and class_counts_ok and pass_agree and sel_agree and crit_agree and gate == V['provisional_gate'])
    return {'status': st(ok) if not marg or not ok else 'PASS_WITH_NUMERICALLY_MARGINAL_CAPS',
            'kind': 'new model forward (own numpy, f32 and f64) of all 40 checkpoints on VALIDATION; own metrics, deltas, selection and gate',
            'validation_rows': int(len(X)), 'pairing_ok': pair_ok, 'max_abs_diff_vs_runner': mx,
            'class_counts_agree': class_counts_ok, 'per_entry_pass_agree': pass_agree,
            'own_selection': sel, 'selection_agrees': sel_agree, 'own_gate_criteria': crit, 'criteria_agree': crit_agree,
            'own_gate': gate, 'runner_gate': V['provisional_gate'], 'numerically_marginal_caps': marg,
            'separate_guard_cap_failures': sep_fail,
            'validation_wire_change_max_abs_diff_vs_runner': wire_rv, 'validation_wire_decision_change_mismatches': wire_dec,
            'separate_guard_validation_wire_change': {k: v for k, v in wire.items() if '|SEPARATE_GUARD|' in k},
            'cap_margins': {k: {t: {q: v['caps'][q]['margin'] for q in CAPS} for t, v in e.items()} for k, e in table.items()}}


def run(spec_path):
    from separate_pool_utility_guard.evaluation_queue import queue_config
    t0 = time.monotonic(); c0 = time.process_time()
    active, cfg, verify, guard = queue_config(spec_path, 'verification')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    c = Ctx(cfg, verify); D = {}
    result = {'schema': 'SPG1-independent-stage-a-v1', 'campaign': 'pcrl_separate_pool_utility_guard_v1',
              'queue_unit_id': active['unit_id'], 'verifier': 'separate_pool_utility_guard_independent (own numpy arithmetic)',
              'training_complete_lock_sha256': cfg['training_complete_lock_sha256'], 'checks': {}}
    steps = [('1_role_partition_and_exposure', check_roles), ('2_frozen_income_parity', check_income),
             ('3_task_weights_and_accounting', check_accounting), ('4_global_anchors_slacks_displacement', check_global),
             ('5_privacy_counterfactual', check_privacy), ('6_validation', check_validation)]
    for name, fn in steps:
        guard()
        try:
            result['checks'][name] = fn(c, D)
        except Exception as exc:
            result['checks'][name] = {'status': 'ERROR', 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-3000:]}
    sts = [v['status'] for v in result['checks'].values()]
    result['overall_status'] = ('FAIL' if any(x in ('FAIL', 'ERROR') for x in sts) else
                                'PASS_WITH_NUMERICALLY_MARGINAL_ITEMS' if any('MARGINAL' in x for x in sts) else 'PASS')
    result['files_read'] = len(set(c.read))
    result['files_read_sha256_of_sorted_list'] = hashlib.sha256(json.dumps(sorted(set(c.read))).encode()).hexdigest()
    result['worker_cpu_s'] = time.process_time() - c0; result['worker_wall_s'] = time.monotonic() - t0
    result['person_level_arrays_written'] = False
    out = Path(cfg['output']); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + '\n')
    print(json.dumps({'overall': result['overall_status'], 'checks': {k: v['status'] for k, v in result['checks'].items()}}))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True)
    run(p.parse_args().spec)
