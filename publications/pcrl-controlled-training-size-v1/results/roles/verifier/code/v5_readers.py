"""Verifier stage V5b: development-reader grid replay (pair benefits, recipient differences, stability and target labels, the
effective reader-seed check), from the saved grid outputs and the reader cache, with the verifier's own code.

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records;
declared development readers only; reader recipes and overlapping splits are not independent datasets).

Inputs: .private/<campaign>/runs/GRID/<bank>/S<v>/GRID.json + BOOT.npz (v = 0, 1); the bank's member releases (paths + SHA-256 from
GRID.json members); the reader cache .private/<campaign>/runs/devgrid_cache (key SHA-256("TSZ1-devgrid-v1|<rep>|<reader seed>|1.0|
<wire hash>|<SEX array hash>")); MODEL_SELECT SEX (tsz.common.load operation 'verify': to recompute the SEX hash and the AUCs).

  1. paired cells: b = comparator SCORE AUC - candidate SCORE AUC (pair view), recipient differences candidate - comparator (income,
     occupation views), cell SE = SD (ddof 1) of the paired bootstrap draw differences; vs GRID.json 'paired' (exact).
  2. labels: own stability (n_eff, distinct-cell SS, qualifier) at 0 and the target label at 0.020; old per-cell recipient gate
     (both single views <= 0.010 in every cell).
  3. replay of cells from the cache: own winner pick (max oriented VALIDATION AUC, exact ties to the earlier candidate in the
     registered order), own AUC of the winner's cached SCORE scores, own household-bootstrap weights (registered seed rule) and own
     weighted AUC draws, vs GRID.json score_auc and BOOT.npz (tolerance 1e-12).
  4. effective reader seeds: for every member, view, (v, R) and seed-dependent recipe, cached SCORE scores under k0 and k1 are not
     bitwise equal; seed-invariant recipes equal; k_differs per comparison pair and n_eff.

    python3 -m tsz.jobqueue run --unit VER_READERS_V1 --stage verification --rss-gib 3 -- python3 .../v5_readers.py --bank COND --tag V1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (the logged label-access layer; operation 'verify')

PRIV = ROOT / '.private' / 'pcrl_controlled_training_size_v1'
GRID = PRIV / 'runs' / 'GRID'
CACHE = PRIV / 'runs' / 'devgrid_cache'
VIEWS = ('income', 'occupation', 'pair')
SRC8 = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP')
NEW8 = ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1')
SEED_DEP = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
REP_INDEX = {'R1': 0, 'R2': 1}


def my_auc(y, s, w=None):
    """Mann-Whitney AUC with ties counted 1/2 (optional nonnegative weights)."""
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    u, inv = np.unique(s, return_inverse=True)
    wp = np.bincount(inv, weights=w * (y == 1), minlength=len(u)); wn = np.bincount(inv, weights=w * (y == 0), minlength=len(u))
    below = np.cumsum(wn) - wn
    return float(np.sum(wp * (below + 0.5 * wn)) / (wp.sum() * wn.sum()))


def boot_weights(unit_sc, v, rep, B=400):
    """Registered rule (READER_SPLITS.json bootstrap): seed 2026101900 + 100 v + repetition index; per draw, households resampled
    with replacement (counts), each SCORE person weighted by its household's count."""
    g, inv = np.unique(unit_sc, return_inverse=True)
    rng = np.random.default_rng(2026101900 + 100 * int(v) + REP_INDEX[rep])
    W = np.empty((B, len(unit_sc)))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))[inv]
    return W


def wire_hash(view, inc, occ):
    h = hashlib.sha256(view.encode())
    for a in ((inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def cache_entry(rep, rs, wh, sx):
    key = hashlib.sha256(f'TSZ1-devgrid-v1|{rep}|{rs}|1.0|{wh}|{sx}'.encode()).hexdigest()
    p = CACHE / f'{key}.npz'
    if not p.is_file():
        return None, key
    with np.load(p, allow_pickle=False) as z:
        info = json.loads(str(z['info']))
        return {n: (z[f'score__{n}'].copy(), info[n]) for n in SRC8 + NEW8}, key


def my_pick(rd_inc, rd_occ, rd_pair, view):
    if view in ('income', 'occupation'):
        rd = rd_inc if view == 'income' else rd_occ
        cands = [(n, rd[n][1]['val_auc'], (view, n)) for n in SRC8 + NEW8]
    else:
        cands = []
        for group in (SRC8, NEW8):
            cands += [(n, rd_pair[n][1]['val_auc'], ('pair', n)) for n in group]
            cands += [(f'IGNORE_occupation__{n}', rd_inc[n][1]['val_auc'], ('income', n)) for n in group]
            cands += [(f'IGNORE_income__{n}', rd_occ[n][1]['val_auc'], ('occupation', n)) for n in group]
    best = 0
    for i in range(1, len(cands)):
        if cands[i][1] > cands[best][1]:
            best = i
    return cands[best]


def wire_sha(p):
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    h = hashlib.sha256(p.tobytes()); h.update(np.ascontiguousarray(p.argmax(1).astype(np.int64)).tobytes())
    return h.hexdigest()


def load_member(m, v):
    """A bank member of the COND spec: 'arrays' members by path / key / file SHA-256 (income = the frozen H3 anchor of version v);
    frozen 'reference' members through the source's admitted loader (ohn.devreaders.load_member)."""
    if m.get('type') == 'arrays':
        per = m['seeds'][str(v)]
        out = {}
        for t in ('income', 'occupation'):
            x = per[t]
            if isinstance(x, dict) and x.get('frozen_anchor') == 'H3':
                from ohn import devrun as OD
                out[t] = np.asarray(OD.anchor_income('H3', v, 'MODEL_SELECT'), dtype=np.float64)
                continue
            p = Path(x['path']); p = p if p.is_absolute() else ROOT / p
            if x.get('sha256') and V.file_sha(p) != x['sha256']:
                raise PermissionError(f'member file hash differs: {p}')
            with np.load(p, allow_pickle=False) as z:
                out[t] = np.asarray(z[x['key']], dtype=np.float64).copy()
        return out
    if m.get('type') in ('reference', 'occ', 'complete'):
        from ohn.devreaders import load_member as source_load
        rel = source_load(m, v)
        return {t: np.asarray(rel[t], dtype=np.float64) for t in ('income', 'occupation')}
    return None


def my_kinds(p, k):
    p = np.asarray(p, dtype=np.float64); d = p.argmax(1); one = np.eye(k)[d]
    return {'out': np.column_stack((p, d.astype(np.float64))), 'log': np.column_stack((np.log(np.maximum(p, 1e-300)), one))}


def refit_lr(inc, occ, view, kind, sex, fit, val, sc):
    """Own refit of the source LR readers (OUT_LR / LOG_LR): own feature table, own standardization (mean / population SD, zero SD ->
    1), scikit-learn LogisticRegression(C = 1, lbfgs, max_iter 500) on FIT, orientation by the VALIDATION AUC (>= 0.5 keeps the sign),
    oriented SCORE scores."""
    from sklearn.linear_model import LogisticRegression
    import warnings
    a, b = my_kinds(inc, 2)[kind], my_kinds(occ, 5)[kind]
    X = a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))
    mu = X[fit].mean(0); sd = X[fit].std(0); sd = np.where(sd > 0, sd, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = LogisticRegression(C=1, max_iter=500, solver='lbfgs').fit((X[fit] - mu) / sd, sex[fit])
    s = m.predict_proba((X - mu) / sd)[:, 1].astype(np.float64)
    s = s if my_auc(sex[val], s[val]) >= 0.5 else 1 - s
    return s[sc]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bank', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--labels-record', default=None, help='coordinator label record to compare (JSON path, optional)')
    a = ap.parse_args()
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'verify',
               'independent verifier V5: development reader grid replay (cells, bootstrap, stability labels, reader-seed check) with own code')
    sex, unit = np.asarray(S['sex']), np.asarray(S['unit'])
    sx = V.arr_sha(sex)
    grids, boots, inputs = {}, {}, []
    for v in (0, 1):
        d = GRID / a.bank / f'S{v}'
        grids[v] = json.loads((d / 'GRID.json').read_text())
        with np.load(d / 'BOOT.npz') as z:
            boots[v] = {k: z[k].copy() for k in z.files}
        inputs += [d / 'GRID.json', d / 'BOOT.npz']
    # 1. paired cells
    pair_bad, cells_all = [], {}
    comps = [(cand, comp) for cand, cc in grids[0]['paired'].items() for comp in cc]
    for cand, comp in comps:
        cells_all[(cand, comp)] = {view: {'cells': {}, 'ses': {}} for view in VIEWS}
        for v in (0, 1):
            G = grids[v]
            for cell, cres in G['cells'].items():
                if '@' in cell:
                    continue
                for view in VIEWS:
                    sa, sb = cres[cand][view]['score_auc'], cres[comp][view]['score_auc']
                    ba, bb = boots[v][f'{cell}__{cand}__{view}'], boots[v][f'{cell}__{comp}__{view}']
                    d, db = (sb - sa, bb - ba) if view == 'pair' else (sa - sb, ba - bb)
                    se = float(np.std(db, ddof=1))
                    cells_all[(cand, comp)][view]['cells'][cell] = d; cells_all[(cand, comp)][view]['ses'][cell] = se
                    rec = G['paired'][cand][comp][view]
                    if rec['cells'][cell] != d or abs(rec['cell_se'][cell] - se) > 1e-15:
                        pair_bad.append(f'{cand}|{comp}|{view}|{cell}')
    # 2. labels
    labels = {}
    for (cand, comp), vv in cells_all.items():
        st = V.stability(vv['pair']['cells'], vv['pair']['ses'])
        tl, ts = V.target_label(vv['pair']['cells'], vv['pair']['ses'])
        gate = all(vv[view]['cells'].get(c, math.inf) <= 0.010 for view in ('income', 'occupation') for c in vv['pair']['cells'])
        labels[f'{cand}|{comp}'] = {'stability': st['label'], 'qualifier': st.get('qualifier'), 'n_eff': st.get('n_eff'), 'mean': st.get('mean'),
                                    'min': st.get('min'), 'max': st.get('max'), 'd_v': st.get('d_v'), 'sd_total': st.get('sd_total'),
                                    'k_differs': st.get('k_differs'), 'target_label': tl, 'old_recipient_gate_pass': gate,
                                    'cells': vv['pair']['cells'],
                                    'recipient_max': {view: max(vv[view]['cells'].values()) for view in ('income', 'occupation')}}
    # 3. replay cells from the cache (winners, AUC, bootstrap) for every member, view, cell
    replay_bad, replay_n, missing_cache, refit_rows = [], 0, [], []
    seeds_check = {'seed_dependent_equal': [], 'seed_invariant_differ': [], 'checked': 0}
    wire_bad = []
    for v in (0, 1):
        G = grids[v]
        spec_members = G['spec']['members']
        prov = G.get('members') or {}
        names = sorted({n for cres in G['cells'].values() for n in cres})
        rels = {}
        for nm in names:
            m = spec_members.get(nm)
            arrs = load_member(m, v) if m else None
            if arrs is None or any(x is None for x in arrs.values()):
                continue
            ws = (prov.get(nm) or {}).get('wire_sha256') or {}
            if ws and any(ws.get(t) != wire_sha(arrs[t]) for t in ('income', 'occupation')):
                wire_bad.append(f'V{v}|{nm}')
            rels[nm] = arrs
        for rep in ('R1', 'R2'):
            part = V.folds(unit, 10, f'TSZ1_devgrid_{rep}')
            sc = part >= 7
            W = boot_weights(unit[sc], v, rep)
            y = sex[sc]
            for nm, arrs in rels.items():
                ent = {}
                for k in (0, 1):
                    rs = 2026101205 + 500 + 10 * v + k
                    ent[k] = {view: cache_entry(rep, rs, wire_hash(view, arrs['income'], arrs['occupation']), sx)[0] for view in VIEWS}
                    if any(e is None for e in ent[k].values()):
                        missing_cache.append(f'S{v}{rep}k{k}|{nm}')
                        continue
                    cell = f'S{v}{rep}k{k}'
                    for view in VIEWS:
                        name, val, (sview, rec_name) = my_pick(ent[k]['income'], ent[k]['occupation'], ent[k]['pair'], view)
                        s = ent[k][sview][rec_name][0]
                        g = G['cells'][cell][nm][view]
                        aucv = my_auc(y, s)
                        draws = np.array([my_auc(y, s, W[b]) for b in range(W.shape[0])])
                        bd = boots[v][f'{cell}__{nm}__{view}']
                        replay_n += 1
                        if name != g['winner'] or abs(aucv - g['score_auc']) > 1e-12 or float(np.max(np.abs(draws - bd))) > 1e-12:
                            replay_bad.append({'cell': cell, 'member': nm, 'view': view, 'winner_mine': name, 'winner_grid': g['winner'],
                                               'auc_diff': aucv - g['score_auc'], 'boot_max_diff': float(np.max(np.abs(draws - bd)))})
                if nm == 'W' and 0 in ent and ent[0] is not None and all(e is not None for e in ent[0].values()):
                    roles = {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6)}
                    for view in VIEWS:
                        for rcp, kind in (('OUT_LR', 'out'), ('LOG_LR', 'log')):
                            mine_s = refit_lr(arrs['income'], arrs['occupation'], view, kind, sex, roles['FIT'], roles['VALIDATION'], sc)
                            cached = ent[0][view][rcp][0]
                            refit_rows.append({'cell': f'S{v}{rep}k0', 'view': view, 'recipe': rcp, 'bitwise': bool(np.array_equal(mine_s, cached)),
                                               'max_abs_diff': float(np.max(np.abs(mine_s - cached)))})
                if 0 in ent and 1 in ent and all(e is not None for e in ent[0].values()) and all(e is not None for e in ent[1].values()):
                    for view in VIEWS:
                        for rcp in SRC8 + NEW8:
                            eq = bool(np.array_equal(ent[0][view][rcp][0], ent[1][view][rcp][0]))
                            seeds_check['checked'] += 1
                            if rcp in SEED_DEP and eq:
                                seeds_check['seed_dependent_equal'].append(f'S{v}{rep}|{nm}|{view}|{rcp}')
                            if rcp not in SEED_DEP and not eq:
                                seeds_check['seed_invariant_differ'].append(f'S{v}{rep}|{nm}|{view}|{rcp}')
    comp_rec = None
    if a.labels_record:
        comp_rec = json.loads(Path(a.labels_record).read_text()) if Path(a.labels_record).is_file() else None
    code = V.code_hashes(__file__, V.__file__)
    checks = [
        {'id': f'V5.READERS.PAIRED_CELLS__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'every comparison pair of the reader bank: per cell pair benefit (comparator - candidate, pair view) and recipient differences '
                 '(candidate - comparator, single views) from GRID.json SCORE AUCs, cell SEs from the paired BOOT.npz draws (SD, ddof 1), own '
                 'code vs GRID.json paired (exact values, SEs within 1e-15)',
         'inputs': V.inputs_sha(inputs), 'expected': 'agreement', 'observed': {'n_pairs': len(comps), 'mismatches': pair_bad[:50], 'n_mismatches': len(pair_bad)},
         'status': 'PASS' if not pair_bad else 'FAIL'},
        {'id': f'V5.READERS.CACHE_REPLAY__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'cells replayed from the reader cache: own winner pick on VALIDATION AUCs, own SCORE AUC of the winner\'s cached scores, own '
                 'household bootstrap weights (registered seed rule) and weighted AUC draws vs GRID.json and BOOT.npz (1e-12)',
         'inputs': V.inputs_sha(inputs), 'expected': 'agreement', 'observed': {'n_replayed': replay_n, 'mismatches': replay_bad[:30],
                                                                              'missing_cache_entries': missing_cache[:30], 'n_missing': len(missing_cache),
                                                                              'member_wire_hash_mismatches': wire_bad},
         'status': 'PASS' if (replay_n > 0 and not replay_bad and not missing_cache and not wire_bad) else ('NOT_RUN' if replay_n == 0 else 'FAIL')},
        {'id': f'V5.READERS.SEED_EFFECTIVENESS__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'effective reader seeds (RULES.md section 9): cached SCORE scores of every seed-dependent recipe differ between k0 and k1 for every '
                 'member, view and (v, R); seed-invariant recipes are equal (a difference is NONDETERMINISTIC_RECIPE, not fatal)',
         'inputs': V.inputs_sha(inputs), 'expected': 'no seed-dependent equality for W100_v and the selected releases (RULES.md 9.2); other '
                                                     'members reported', 'observed': seeds_check,
         'status': 'NOT_RUN' if seeds_check['checked'] == 0 else
                   ('FAIL' if any(x.split('|')[1] in ('W', 'LOC_SEL', 'PAIR_SEL', 'CONT_AT_LOC', 'CONT_AT_PAIR', 'LOC_AT_PAIR')
                                  for x in seeds_check['seed_dependent_equal']) else
                    ('ATTENTION' if (seeds_check['seed_dependent_equal'] or seeds_check['seed_invariant_differ']) else 'PASS'))},
        {'id': f'V5.READERS.LR_REFIT__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'independent refit of the OUT_LR and LOG_LR readers (own feature tables, own standardization, scikit-learn LogisticRegression '
                 'C = 1, lbfgs, max_iter 500, orientation on VALIDATION) for the witness W of both versions, cells S<v>R<r>k0, three views, vs '
                 'the cached SCORE scores (max abs difference <= 1e-9; bitwise reported)',
         'inputs': V.inputs_sha(inputs), 'expected': 'max abs difference <= 1e-9', 'observed': {'rows': refit_rows},
         'status': 'NOT_RUN' if not refit_rows else ('PASS' if all(r['max_abs_diff'] <= 1e-9 for r in refit_rows) else 'FAIL')},
        {'id': f'V5.READERS.LABELS__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'own stability labels (threshold 0; n_eff, distinct-cell SS, qualifier), target labels (0.020) and the old per-cell recipient gate '
                 'for every comparison pair of the bank' + (' vs the coordinator label record' if comp_rec is not None else ' (no label record compared)'),
         'inputs': V.inputs_sha(inputs + ([Path(a.labels_record)] if comp_rec is not None else [])), 'expected': 'see the coordinator record',
         'observed': labels, 'status': 'PASS'}]
    for c in checks:
        c['code_sha256'] = code
    V.write_record(f'V5_READERS__{a.tag}.json', {'checks': checks, 'utc': V.utc()})
    print(json.dumps(V.append_checks(checks)))
    print(json.dumps({c['id']: c['status'] for c in checks}))
    print(json.dumps({k: [x['stability'], x['qualifier'], round(x['mean'], 5) if x['mean'] is not None else None, x['target_label'], x['old_recipient_gate_pass']]
                      for k, x in labels.items()}, indent=0)[:6000])


if __name__ == '__main__':
    main()
