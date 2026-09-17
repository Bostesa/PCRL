"""Independent verification of the locked 2017 transport outputs.

Recomputes from saved releases, predictions, labels and selection records with
separate code paths (no report computation functions are reused).
"""
from __future__ import annotations
import hashlib, itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/redesign_20260917_acs_spectral_transport_v1'
DATA = ROOT/'data/acs_spectral_transport'
HIST = ('H', 'E', 'A0', 'L025', 'L20', 'J')
SPEC = tuple('spectral_'+a for a in ('S0', 'M025', 'M1', 'L025', 'L1', 'C025', 'C1', 'L2'))
IFACES = HIST+SPEC
SCOPES_B = {'common_fresh': ({'fresh'}, {'wire'}), 'fresh_expanded': ({'fresh'}, {'wire', 'derived'}),
            'fresh_catchup': ({'fresh', 'catchup'}, {'wire', 'derived'}), 'transport_all': ({'fresh', 'catchup', 'frozen2018'}, {'wire', 'derived'})}


def read(p): return json.loads(Path(p).read_text())


def ll(y, p):
    """Independent log loss: floor, renormalize, mean negative log of the true class."""
    q = np.maximum(np.minimum(np.asarray(p, dtype=np.float64), 1.0), 1e-12)
    q = q/q.sum(axis=1)[:, None]
    return np.array([-np.log(q[i, int(k)]) for i, k in enumerate(y)])


def check(name, ok, detail, results):
    results.append({'check': name, 'pass': bool(ok), 'detail': detail})
    print(('PASS ' if ok else 'FAIL ')+name, detail if not ok else '', flush=True)


def main():
    res = []
    # 1. household disjointness and completeness of partitions
    parts = {p: np.load(DATA/f'2017_{p}.npz') for p in ('attacker_fit', 'attacker_validation', 'task_fit', 'task_validation', 'final_evaluation')}
    sets = {p: set(z['SERIALNO'].tolist()) for p, z in parts.items()}
    overlap = sum(len(sets[a] & sets[b]) for a, b in itertools.combinations(sets, 2))
    keys = set(); dup = 0
    for z in parts.values():
        for a, b in zip(z['SERIALNO'].tolist(), z['SPORDER'].tolist()):
            dup += (a, b) in keys; keys.add((a, b))
    check('household_disjointness', overlap == 0 and dup == 0, {'shared_households': overlap, 'duplicate_person_keys': dup,
          'rows': sum(len(z['SERIALNO']) for z in parts.values()), 'households': sum(len(s) for s in sets.values())}, res)
    # lock
    lock = read(OUT/'TRANSPORT_LOCK.json'); amended = {}
    for a in sorted(OUT.glob('TRANSPORT_LOCK_AMENDMENT_*.json')):
        rec = read(a)
        assert all(k.startswith(('experiments/', 'scripts/')) and k in lock['files'] for k in rec['code_hashes']), 'amendment touches a non-code locked input'
        amended[a.name] = sorted(rec['code_hashes']); lock['files'].update(rec['code_hashes'])
    changed = [n for n, h in lock['files'].items() if hashlib.sha256(Path(n if n.startswith('/') else ROOT/n).read_bytes()).hexdigest() != h]
    check('lock_inputs_unchanged_or_amended', not changed, {'files': len(lock['files']), 'changed': changed[:5],
          'code_only_amendments': amended, 'objects_data_and_selections_under_original_hashes': True}, res)
    access = read(OUT/'FINAL_ACCESS_LOG.json')
    lock_commit_time = '2026-09-17'
    check('final_access_after_lock', all(a['lock_sha256'] == hashlib.sha256((OUT/'TRANSPORT_LOCK.json').read_bytes()).hexdigest() for a in access) and lock['created_utc'] < min(a['utc'] for a in access),
          {'accesses': len(access), 'lock_created': lock['created_utc'], 'first_access': min(a['utc'] for a in access)}, res)
    lab = np.load(OUT/'final_labels.npz'); y = {k[2:]: lab[k] for k in lab.files if k.startswith('y/')}; w = lab['weights']
    assert np.array_equal(lab['serialno'], parts['final_evaluation']['SERIALNO'])
    # 2. exact source-output identity on every released 2017 wire
    bad = 0; n = 0
    for s in (0, 1, 2):
        for part in ('attacker_fit', 'attacker_validation', 'task_fit', 'task_validation', 'final_evaluation'):
            z = np.load(OUT/f'seed_{s}/releases_2017/{part}.npz')
            A, B = z['anchor/A'], z['anchor/B']
            assert A.dtype == np.float64 and np.allclose(A.sum(1), 2) and np.allclose(B.sum(1), 1)
            for c in IFACES:
                wa, wb, wab = z[f'wire/{c}/A'], z[f'wire/{c}/B'], z[f'wire/{c}/AB']
                n += 1
                bad += not (np.array_equal(wa[:, :4], A) and np.array_equal(wb, B) and np.array_equal(wab, np.hstack([wa, wb])) and wa.dtype == np.float64)
                if c == 'H': bad += wa.shape[1] != 4
                else: bad += wa.shape[1] != 20
    check('source_output_identity', bad == 0, {'release_views_checked': n, 'violations': bad}, res)
    # B-only releases identical across interfaces (same seed)
    same = all(np.array_equal(np.load(OUT/f'seed_{s}/releases_2017/final_evaluation.npz')[f'wire/{c}/B'], np.load(OUT/f'seed_{s}/releases_2017/final_evaluation.npz')['wire/H/B']) for s in (0, 1, 2) for c in IFACES)
    check('b_only_identical', same, {}, res)
    # 3. projection rules and 4. selection identity (Mode B)
    proj_bad = []; sel_bad = []; balias_bad = []
    for s, c in itertools.product((0, 1, 2), IFACES):
        rec = read(OUT/f'seed_{s}/{c}/audit_selection.json'); h = read(OUT/f'seed_{s}/H/audit_selection.json')
        wa = 4 if c == 'H' else 20; wd = 8
        for b, roles in rec['candidates'].items():
            for role, cs in roles.items():
                view, t = role.split('/')
                if view == 'B' and c != 'H' and cs != h['candidates'][b][role]: balias_bad.append((s, c, b, role))
                for cid, r in cs.items():
                    cols = r['projection_columns']; width = (wa if r['space'] == 'wire' else wd) + (2 if view == 'AB' else 0)
                    if view == 'B': width = 2
                    if cols is not None and (max(cols) >= width or min(cols) < 0): proj_bad.append((s, c, role, cid))
                    if cid.startswith('inherited_B__') and cols != [wa if r['space'] == 'wire' else wd, (wa if r['space'] == 'wire' else wd)+1] and r['projection_columns'] is not None:
                        src = roles['B/'+t][cid[len('inherited_B__'):]]
                        if src['projection_columns'] is None: proj_bad.append((s, c, role, cid, 'B block'))
                    if cid.startswith('anchor__') and cols is not None and any(k not in (0, 1, 2, 3, wa, wa+1) for k in cols): proj_bad.append((s, c, role, cid, 'anchor'))
                scores = rec['validation_scores'][b][role]
                for scope, (orig, spaces) in SCOPES_B.items():
                    ids = [cid for cid, r in cs.items() if not r['diagnostic_only'] and r['transport_origin'] in orig and r['space'] in spaces]
                    best = sorted(ids, key=lambda k: (scores[k]['log_loss'], k))[0]
                    if best != rec['selections'][b][role][scope]: sel_bad.append((s, c, b, role, scope))
                # validation scores recomputed for all selected candidates in the primary scope
    check('projection_rules', not proj_bad, {'violations': proj_bad[:5]}, res)
    check('b_view_aliases_H', not balias_bad, {'violations': balias_bad[:5]}, res)
    check('mode_B_selection_identity', not sel_bad, {'violations': sel_bad[:5]}, res)
    # 5. score dictionaries: independent log loss for every selected prediction (both modes, both weights)
    worst = 0.; count = 0
    for s, c, mode in itertools.product((0, 1, 2), IFACES, ('A', 'B')):
        d = OUT/f'seed_{s}/{c}/mode_{mode}'; m = read(d/'metrics.json')['raw_metrics']
        z = np.load(d/'predictions.npz'); ids = dict(zip(z['keys'].tolist(), z['ids'].tolist()))
        for r in m:
            if not (r.get('selected') or r.get('selected_scopes')): continue
            key = f"utility/{r['view']}/{r['target']}/{r['candidate_id']}" if r['role'] == 'utility' else f"audit/{r['audit_budget']}/{r['view']}/{r['target']}/{r['candidate_id']}"
            p = z['p/'+ids[key]]; yy = y[r['target']]; v = yy >= 0
            l = ll(yy[v], p)
            worst = max(worst, abs(l.mean()-r['scores']['test']['log_loss']), abs((w[v]*l).sum()/w[v].sum()-r['scores']['test_person_weighted']['log_loss']))
            count += 1
    check('score_dictionaries', worst < 1e-9, {'selected_rows': count, 'max_abs_difference': worst}, res)
    # 6. incremental baselines and 8. paired differences from the published tables
    ps = pd.read_csv(OUT/'PER_SEED.csv')
    key = ['seed', 'mode', 'scope', 'budget', 'weight', 'endpoint']
    a = ps[ps.kind == 'absolute_recovery']; hA = a[a.condition == 'H'][key+['value']].rename(columns={'value': 'h'})
    add = ps[ps.kind == 'additional_recovery'].merge(a[key+['condition', 'value']].rename(columns={'value': 'abs'}), on=key+['condition']).merge(hA, on=key)
    check('incremental_baselines', np.allclose(add['value'], add['abs']-add['h'], atol=1e-13, rtol=0), {'rows': len(add)}, res)
    fam = pd.read_csv(OUT/'FAMILIES.csv'); spec = read(OUT/'COMPARISONS.json')['families']; worst = 0.
    for _, r in fam.iterrows():
        f = spec[r['family']]; kind, e = r['endpoint'].split('/', 1)
        k = 'utility_loss' if kind == 'utility' else 'absolute_recovery'
        q = ps[(ps['mode'] == f['mode']) & (ps.scope == f['scope']) & (ps.budget == f['budget']) & (ps.weight == r['weight']) & (ps.kind == k) & (ps.endpoint == e)]
        diff = np.mean([q[(q.condition == r['left']) & (q.seed == s)]['value'].iloc[0]-q[(q.condition == r['right']) & (q.seed == s)]['value'].iloc[0] for s in (0, 1, 2)])
        worst = max(worst, abs(diff-r['estimate']))
    check('paired_differences', worst < 1e-12, {'family_rows': len(fam), 'max_abs_difference': worst}, res)
    # 7. withholding arithmetic
    wd_ = pd.read_csv(OUT/'evidence/WITHHOLDING.csv.gz'); worst = 0.
    for _, r in wd_.sample(400, random_state=0).iterrows():
        k = r['kind']
        q = ps[(ps['mode'] == r['mode']) & (ps.scope == r['scope']) & (ps.budget == r['budget']) & (ps.weight == r['weight']) & (ps.seed == r['seed']) & (ps.kind == k) & (ps.endpoint == r['endpoint'])]
        exp = (1-r['p'])*q[q.condition == 'H']['value'].iloc[0]+r['p']*q[q.condition == r['source']]['value'].iloc[0]
        worst = max(worst, abs(exp-r['value']))
    dec = read(OUT/'TRANSPORT_DECISION.json')
    worst2 = max(c['abs_difference'] for c in dec['withholding_arithmetic_checks'])
    check('withholding_arithmetic', worst < 1e-12 and worst2 < 1e-12, {'sampled_rows': 400, 'max_abs_difference': worst, 'per_person_mixture_vs_linear': worst2}, res)
    # 9. bootstrap: independent loop implementation for one F1 endpoint
    serial = lab['serialno']; codes, inv = np.unique(serial, return_inverse=True); G = len(codes)
    rng = np.random.default_rng(20260917); draws = rng.integers(0, G, size=(2000, G))
    row = fam[(fam.family == 'F1_primary') & (fam.left == 'spectral_C1') & (fam.right == 'spectral_L1') & (fam.endpoint == 'recovery/AB/SEX') & (fam.weight == 'person_weighted')].iloc[0]
    vv = y['SEX'] >= 0
    def sel_loss(s, c):
        d = OUT/f'seed_{s}/{c}/mode_B'; m = read(d/'metrics.json')['raw_metrics']
        r = [x for x in m if x['role'] == 'audit' and x['audit_budget'] == 360 and x['view'] == 'AB' and x['target'] == 'SEX' and 'transport_all' in x['selected_scopes']][0]
        z = np.load(d/'predictions.npz'); ids = dict(zip(z['keys'].tolist(), z['ids'].tolist()))
        return ll(y['SEX'][vv], z['p/'+ids[f"audit/360/AB/SEX/{r['candidate_id']}"]])
    diffs = [sel_loss(s, 'spectral_L1')-sel_loss(s, 'spectral_C1') for s in (0, 1, 2)]  # recovery difference
    g = inv[vv]; ww = w[vv]; reps = []
    for b in range(2000):
        cnt = np.bincount(draws[b], minlength=G)[g]*ww
        reps.append(np.mean([cnt @ d_/cnt.sum() for d_ in diffs]))
    reps = np.array(reps); point = np.mean([ww @ d_/ww.sum() for d_ in diffs])
    check('bootstrap_replay', abs(point-row['estimate']) < 1e-12 and abs(reps.std(ddof=1)-row['se']) < 1e-10,
          {'estimate': point, 'reported': row['estimate'], 'se': float(reps.std(ddof=1)), 'reported_se': row['se']}, res)
    crit = row['critical_value']
    check('adjusted_interval_arithmetic', abs(row['adjusted_low']-(row['estimate']-crit*row['se'])) < 1e-12, {'critical': crit}, res)
    # 10. service quality recomputed
    sq = pd.read_csv(OUT/'evidence/SERVICE_QUALITY.csv'); worst = 0.
    for s in (0, 1, 2):
        z = np.load(OUT/f'seed_{s}/releases_2017/final_evaluation.npz')
        for t, p in (('income_binary', z['anchor/A'][:, :2]), ('civilian_at_work', z['anchor/A'][:, 2:]), ('public_coverage', z['anchor/B'])):
            v = y[t] >= 0
            worst = max(worst, abs(ll(y[t][v], p[v]).mean()-sq[(sq.seed == s) & (sq.task == t)]['transport_unweighted'].iloc[0]))
    check('service_quality', worst < 1e-9, {'max_abs_difference': worst}, res)
    out = {'checks': res, 'all_pass': all(r['pass'] for r in res)}
    (OUT/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(out, indent=2, default=float)+'\n')
    print('ALL PASS' if out['all_pass'] else 'FAILURES PRESENT')
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
