"""VER check 6: development reader grid (DEV_GRID.json), recomputed with the verifier's own code.

Modes
  paired  (no label read)  from runs/GRID/<BANK>/S<s>/GRID.json (winner SCORE AUCs) and BOOT.npz (bootstrap draws of each winner's
          SCORE AUC): every paired cell value (pair benefit = comparator - candidate; recipient difference = candidate - comparator)
          and its bootstrap SE, the 8-cell stability statistics and labels (RULES.json stability section), the recipient point gate,
          the material-target label and the learning-curve means, compared with GRID.json `paired` and runs/GRID/<BANK>/SUMMARY.json
          (and any other summary JSON given with --summary).
  cache   (MODEL_SELECT SEX, logged)  from the cached reader scores (runs/devgrid_cache/<key>.npz, key recomputed with own code):
          every recipe's SCORE AUC with own AUC code, the VALIDATION winners re-selected from the cached VALIDATION AUCs under the
          registered order and tie rule, the winners' SCORE AUCs and the household-bootstrap draws (own weights and weighted AUC)
          against GRID.json / BOOT.npz.
  refit   (MODEL_SELECT SEX, logged)  independent reader refits with own feature, scaler, recipe and selection code for chosen
          (bank, member, cell) entries: per-recipe VALIDATION AUC and SCORE scores against the cache, winners and SCORE AUCs against
          GRID.json.
Readers see only released probability vectors and argmax decisions; SEX is the evaluation label (sex0 = men, sex1 = women).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcommon as V  # noqa: E402

CELLS = [f'S{s}{r}k{k}' for s in (0, 1) for r in ('R1', 'R2') for k in (0, 1)]
VIEWS = ('income', 'occupation', 'pair')
SRC8 = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP')
NEW8 = ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1')
GRID_VERSION = 'OHN1-devgrid-v1'
RULES = V.load_rules()


# ------------------------------------------------------------------------------------------------ stability rule (own code)
def stability_stats(cells, cell_se, threshold):
    vals = np.array([cells[c] for c in CELLS], dtype=np.float64)
    ses = np.array([cell_se[c] for c in CELLS], dtype=np.float64)
    d = {0: float(vals[:4].mean()), 1: float(vals[4:].mean())}
    ss = float(np.sum((vals[:4] - d[0]) ** 2) + np.sum((vals[4:] - d[1]) ** 2))
    v_rep = ss / 6.0
    v_samp = float(np.mean(ses ** 2))
    sd = math.sqrt(v_rep + v_samp)
    lower = {s: d[s] - 2.0 * sd for s in (0, 1)}
    upper = {s: d[s] + 2.0 * sd for s in (0, 1)}
    return {'per_seed_mean': d, 'v_rep': v_rep, 'v_samp': v_samp, 'sd_total': sd, 'lower': lower, 'upper': upper,
            'mean': float(vals.mean()), 'min': float(vals.min()), 'max': float(vals.max()),
            'all_cells_above': bool(np.all(vals > threshold)), 'n_above': int(np.sum(vals > threshold))}


def benefit_label(cells, cell_se):
    if any(c not in cells or c not in cell_se or not math.isfinite(cells[c]) or not math.isfinite(cell_se[c]) for c in CELLS):
        return 'INCOMPLETE', None
    st = stability_stats(cells, cell_se, 0.0)
    if st['all_cells_above'] and all(st['lower'][s] > 0 for s in (0, 1)):
        return 'STABLE_POSITIVE', st
    if st['all_cells_above']:
        return 'POSITIVE_NOT_STABLE', st
    if st['mean'] > 0:
        return 'SIGN_CHANGES', st
    return 'NOT_POSITIVE', st


def target_label(cells, cell_se):
    st = stability_stats(cells, cell_se, 0.020)
    if st['all_cells_above'] and all(st['lower'][s] > 0.020 for s in (0, 1)):
        return 'DEV_GRID_TARGET_STABLE'
    if st['mean'] > 0.020:
        return 'DEV_GRID_TARGET_POINT_ONLY'
    return 'DEV_GRID_TARGET_NOT_MET'


def recipient_label(cells, cell_se):
    vals = [cells[c] for c in CELLS]
    if any(v > 0.010 for v in vals):
        return 'RECIPIENT_POINT_FAILED', max(vals)
    st = stability_stats(cells, cell_se, 0.010)
    return ('RECIPIENT_STABLE' if all(st['upper'][s] <= 0.010 for s in (0, 1)) else 'RECIPIENT_EDGE'), max(vals)


# ------------------------------------------------------------------------------------------------ mode: paired (no labels)
def load_grids():
    out = {}
    for p in sorted(V.RUNS.glob('GRID/*/S*/GRID.json')):
        bank, s = p.parts[-3], int(p.parts[-2][1:])
        out.setdefault(bank, {})[s] = (json.loads(p.read_text()), p.parent / 'BOOT.npz')
    return out


def collect_cells():
    """Per bank: every paired cell recomputed from GRID.json winner SCORE AUCs and BOOT.npz draws (own code), with the gaps to the
    stored paired values. Returns (per_bank entries, per_bank learning-curve values, per-bank / per-seed gap records)."""
    grids = load_grids()
    per_bank, per_bank_lc, gaps = {}, {}, {}
    for bank, per_seed in grids.items():
        b = {}
        for s, (g, bootp) in per_seed.items():
            boot = np.load(bootp)
            md_v = md_se = 0.0
            for cand, comps in g['paired'].items():
                for comp, views in comps.items():
                    for view in VIEWS:
                        for cell, val in views[view]['cells'].items():
                            a = g['cells'][cell][cand][view]['score_auc']
                            c = g['cells'][cell][comp][view]['score_auc']
                            ba = boot[f'{cell}__{cand}__{view}']
                            bc = boot[f'{cell}__{comp}__{view}']
                            mine = (c - a) if view == 'pair' else (a - c)
                            mb = (bc - ba) if view == 'pair' else (ba - bc)
                            se = float(np.sqrt(np.sum((mb - mb.mean()) ** 2) / (len(mb) - 1)))
                            md_v = max(md_v, abs(mine - val))
                            md_se = max(md_se, abs(se - views[view]['cell_se'][cell]))
                            if '@' not in cell:
                                e = per_bank.setdefault(bank, {}).setdefault((cand, comp), {v: {'cells': {}, 'cell_se': {}} for v in VIEWS})
                                e[view]['cells'][cell] = mine
                                e[view]['cell_se'][cell] = se
                            elif view == 'pair':
                                per_bank_lc.setdefault(bank, {}).setdefault((cand, comp), {}).setdefault(cell.split('@')[1], []).append(mine)
            b[s] = {'max_abs_cell_value_diff': md_v, 'max_abs_cell_se_diff': md_se, 'n_candidates': len(g['paired']),
                    'unit_id': g.get('unit_id'), 'stats': g.get('stats')}
        gaps[bank] = b
    return per_bank, per_bank_lc, gaps


def build_labels(per_bank, per_bank_lc, banks):
    """8-cell labels from the cells of the given banks only (a cell present in several banks carries the same value: cached readers)."""
    merged, lcm = {}, {}
    for bank in banks:
        for key, e in per_bank.get(bank, {}).items():
            m = merged.setdefault(key, {v: {'cells': {}, 'cell_se': {}} for v in VIEWS})
            for view in VIEWS:
                m[view]['cells'].update(e[view]['cells'])
                m[view]['cell_se'].update(e[view]['cell_se'])
        for key, fr in per_bank_lc.get(bank, {}).items():
            for f, vals in fr.items():
                lcm.setdefault(key, {}).setdefault(f, []).extend(vals)
    labels = {}
    for (cand, comp), e in merged.items():
        if not all(c in e['pair']['cells'] for c in CELLS):
            labels.setdefault(cand, {})[comp] = {'benefit': 'INCOMPLETE', 'cells_present': sorted(e['pair']['cells'])}
            continue
        bl, st = benefit_label(e['pair']['cells'], e['pair']['cell_se'])
        ri, ri_max = recipient_label(e['income']['cells'], e['income']['cell_se'])
        ro, ro_max = recipient_label(e['occupation']['cells'], e['occupation']['cell_se'])
        full = [e['pair']['cells'][c] for c in CELLS]
        curve = {f: float(np.mean(v)) for f, v in lcm.get((cand, comp), {}).items()}
        curve['1.0'] = float(np.mean(full))
        row = {'benefit': bl, 'mean': st['mean'], 'min': st['min'], 'per_seed_mean': st['per_seed_mean'], 'sd_total': st['sd_total'],
               'lower': st['lower'], 'recipient_income': ri, 'recipient_income_max': ri_max, 'recipient_occupation': ro,
               'recipient_occupation_max': ro_max, 'income_cells': e['income']['cells'], 'learning_curve': curve,
               'pair_cells': e['pair']['cells'], 'pair_cell_se': e['pair']['cell_se'],
               'lc_flag': ('BENEFIT_SHRINKS_WITH_READER_DATA' if curve['1.0'] < curve['0.25'] - 0.005 else 'NO_SHRINKAGE_FLAG')
               if '0.25' in curve else None}
        if comp in RULES['comparators']['target_references']:
            row['target'] = target_label(e['pair']['cells'], e['pair']['cell_se'])
        labels.setdefault(cand, {})[comp] = row
    return labels


def compare_summary(sm, labels):
    mism, n = [], 0
    for cand, comps in sm.get('entries', {}).items():
        for comp, v in comps.items():
            st = v.get('stability', {})
            mine = labels.get(cand, {}).get(comp)
            if mine is None:
                mism.append(f'{cand} vs {comp}: not in my merge')
                continue
            n += 1
            theirs_b = st.get('benefit', {}).get('label')
            if theirs_b != mine['benefit']:
                mism.append(f'{cand} vs {comp}: benefit {theirs_b} vs mine {mine["benefit"]}')
            if 'benefit' in st and 'sd_total' in st['benefit'] and mine['benefit'] != 'INCOMPLETE':
                if abs(st['benefit']['sd_total'] - mine['sd_total']) > 1e-12 or abs(st['benefit']['mean'] - mine['mean']) > 1e-12:
                    mism.append(f'{cand} vs {comp}: sd_total/mean differ')
            rc = st.get('recipient', {})
            for view, key in (('income', 'recipient_income'), ('occupation', 'recipient_occupation')):
                if view in rc and rc[view].get('label') != mine.get(key):
                    mism.append(f'{cand} vs {comp}: recipient {view} {rc[view].get("label")} vs mine {mine.get(key)}')
            if 'target' in mine and st.get('target', {}).get('label') != mine['target']:
                mism.append(f'{cand} vs {comp}: target {st.get("target", {}).get("label")} vs mine {mine["target"]}')
            if v.get('learning_curve_mean_pair_benefit'):
                for f, val in v['learning_curve_mean_pair_benefit'].items():
                    if f in mine.get('learning_curve', {}) and abs(mine['learning_curve'][f] - val) > 1e-12:
                        mism.append(f'{cand} vs {comp}: learning curve {f} differs')
    return n, mism


def run_paired(summaries):
    """Change 07:52Z (records/CHECK_CHANGES.json): each summary is compared with labels built from ITS OWN banks (the summary's
    'banks' field); before, every summary was compared with a merge over all banks, so a per-bank summary written before another
    bank added the missing seed showed INCOMPLETE-vs-complete differences. The all-banks labels are still recorded (labels)."""
    per_bank, per_bank_lc, gaps = collect_cells()
    rec = {'check': 'VER_GRID_PAIRED', 'utc': V.utc(), 'banks': gaps}
    worst_v = max((x['max_abs_cell_value_diff'] for b in gaps.values() for x in b.values()), default=0.0)
    worst_se = max((x['max_abs_cell_se_diff'] for b in gaps.values() for x in b.values()), default=0.0)
    labels = build_labels(per_bank, per_bank_lc, sorted(per_bank))
    rec['labels'] = labels
    rec['labels_scope'] = 'all banks under runs/GRID'
    rec['max_abs_cell_value_diff'] = worst_v
    rec['max_abs_cell_se_diff'] = worst_se
    comp_res = {}
    sums = ([p for p in sorted(V.RUNS.glob('GRID/*/SUMMARY.json'))] + [p for p in sorted(V.RES.glob('grid/*.json'))]
            + [Path(x) for x in summaries])
    for sp in sums:
        if not sp.is_file():
            continue
        sm = json.loads(sp.read_text())
        scope = list(sm.get('banks') or sorted(per_bank))
        n, mism = compare_summary(sm, build_labels(per_bank, per_bank_lc, scope))
        comp_res[str(sp.relative_to(V.ROOT)) if str(sp).startswith(str(V.ROOT)) else str(sp)] = {'n_compared': n, 'mismatches': mism,
                                                                                             'banks_used': scope}
    rec['summary_comparisons'] = comp_res
    rec['status'] = 'PASS' if (worst_v < 1e-12 and worst_se < 1e-12 and all(not v['mismatches'] for v in comp_res.values())) else 'ATTENTION'
    V.write_record('VER_GRID_PAIRED', rec)
    print(json.dumps({'status': rec['status'], 'max_abs_cell_value_diff': worst_v, 'max_abs_cell_se_diff': worst_se,
                      'summary_comparisons': {k: (v['n_compared'], v['mismatches'][:5]) for k, v in comp_res.items()}}, indent=1))
    return labels


# ------------------------------------------------------------------------------------------------ registered split (own code)
def grid_roles(unit, rep):
    part = V.household_parts(unit, 10, f'OHN1_devgrid_{rep}')
    return part <= 4, (part >= 5) & (part <= 6), part >= 7


def lc_mask(unit, fit, frac):
    if frac == 1.0:
        return fit
    part = V.household_parts(unit, 4, 'OHN1_devgrid_LC')
    return fit & ((part == 0) if frac == 0.25 else (part <= 1))


def view_wire_hash(view, inc, occ):
    h = hashlib.sha256(view.encode())
    arrs = (inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)
    for a in arrs:
        h.update(np.array(a, dtype=np.float64, order='C', copy=True).tobytes())
    return h.hexdigest()


def reader_seed(rel_seed, k):
    base = None
    for line in (V.ROOT / 'task_conditioned_representation/attacks.py').read_text().splitlines():
        if line.replace(' ', '').startswith('MODEL_SEED_BASE='):
            base = int(line.split('=')[1].split('#')[0].strip())
    return base + 300 + 10 * int(rel_seed) + int(k)


def cache_key(rep, rs, frac, view, inc, occ, sex):
    return hashlib.sha256(f'{GRID_VERSION}|{rep}|{rs}|{frac}|{view_wire_hash(view, inc, occ)}|{V.array_hash(np.asarray(sex))}'.encode()).hexdigest()


def select(cands):
    """cands: [(name, val_auc, key)]; maximum VALIDATION AUC, exact ties to the earlier candidate."""
    best = 0
    for i in range(1, len(cands)):
        if cands[i][1] > cands[best][1]:
            best = i
    others = sorted((c[1] for i, c in enumerate(cands) if i != best), reverse=True)
    return cands[best], (cands[best][1] - others[0]) if others else None


def winners_from(info):
    """info[view][recipe] = val_auc."""
    w = {}
    for view in ('income', 'occupation'):
        w[view] = select([(n, info[view][n], f'{view}__{n}') for n in SRC8 + NEW8])
    cands = []
    for group in (SRC8, NEW8):
        cands += [(n, info['pair'][n], f'pair__{n}') for n in group]
        cands += [(f'IGNORE_occupation__{n}', info['income'][n], f'income__{n}') for n in group]
        cands += [(f'IGNORE_income__{n}', info['occupation'][n], f'occupation__{n}') for n in group]
    w['pair'] = select(cands)
    return w


def boot_weights_own(unit_sc, rel_seed, rep, B=400, seed=2026101806):
    """Registered bootstrap: B resamples of SCORE households with numpy's default_rng(seed + 100 s + rep index)."""
    groups = sorted(set(np.asarray(unit_sc).tolist()))
    idx = {g: i for i, g in enumerate(groups)}
    inv = np.array([idx[g] for g in np.asarray(unit_sc).tolist()])
    rng = np.random.default_rng(seed + 100 * int(rel_seed) + (0 if rep == 'R1' else 1))
    G = len(groups)
    W = np.empty((B, len(unit_sc)))
    for b in range(B):
        draw = rng.integers(0, G, G)
        cnt = np.zeros(G)
        np.add.at(cnt, draw, 1.0)
        W[b] = cnt[inv]
    return W


def load_member(member, seed):
    from ohn import devrun as D
    if member['type'] == 'reference':
        return D.reference_release(member['name'], seed, 'MODEL_SELECT')
    if member['type'] == 'occ':
        occ, _, _ = D.load_occupation(member['family'], member['config'], seed, roles=('MODEL_SELECT',))
        return {'income': D.anchor_income(member['anchor'], seed, 'MODEL_SELECT'), 'occupation': occ['MODEL_SELECT']}
    if member['type'] == 'complete':
        rel, _ = D.load_complete(member['family'], member['config'], seed, roles=('MODEL_SELECT',))
        return {t: rel['MODEL_SELECT'][t] for t in V.TASKS}
    raise ValueError(member)


def ms_sex(purpose):
    from ohn import common as C
    S = C.load_role('MODEL_SELECT', ('features', 'sex'), purpose=V.PURPOSE + purpose)
    return S['sex'], np.asarray(S['unit'])


def run_cache(banks, max_boot_checks):
    sex, unit = ms_sex('dev-grid recomputation from cached reader scores (VER check 6)')
    rec = {'check': 'VER_GRID_CACHE', 'utc': V.utc(), 'unit_id': V.unit_id(), 'banks': {}}
    worst = {'score_auc': 0.0, 'val_auc_order': 0, 'boot': 0.0}
    n_boot = 0
    masks = {rep: grid_roles(unit, rep) for rep in ('R1', 'R2')}
    for bank in banks:
        for s in (0, 1):
            gp = V.RUNS / 'GRID' / bank / f'S{s}' / 'GRID.json'
            if not gp.is_file():
                continue
            g = json.loads(gp.read_text())
            boot = np.load(gp.parent / 'BOOT.npz')
            rel = {name: load_member(m, s) for name, m in g['spec']['members'].items()}
            out = {'cells': {}, 'missing_cache': [], 'winner_mismatch': [], 'score_auc_max_abs_diff': 0.0, 'boot_max_abs_diff': 0.0,
                   'per_recipe_score_auc': {}}
            for cell in g['cells']:
                base, frac = (cell.split('@') + ['1.0'])[:2]
                frac = float(frac)
                rep, k = base[2:4], int(base[-1])
                fit, val, sc = masks[rep]
                rs = reader_seed(s, k)
                W = None
                for name, r in rel.items():
                    info, scores = {}, {}
                    ok = True
                    for view in VIEWS:
                        key = cache_key(rep, rs, frac, view, r['income'], r['occupation'], sex)
                        p = V.RUNS / 'devgrid_cache' / f'{key}.npz'
                        if not p.is_file():
                            out['missing_cache'].append(f'{cell}/{name}/{view}')
                            ok = False
                            continue
                        with np.load(p, allow_pickle=False) as z:
                            inf = json.loads(str(z['info']))
                            info[view] = {n: inf[n]['val_auc'] for n in SRC8 + NEW8}
                            scores[view] = {n: z[f'score__{n}'].copy() for n in SRC8 + NEW8}
                    if not ok:
                        continue
                    w = winners_from(info)
                    ycell = sex[sc]
                    for view in VIEWS:
                        (wname, wval, wkey), gap = w[view]
                        e = g['cells'][cell][name][view]
                        if wname != e['winner'] or abs(wval - e['val_auc']) > 1e-15:
                            out['winner_mismatch'].append(f'{cell}/{name}/{view}: mine {wname} {wval} vs {e["winner"]} {e["val_auc"]}')
                        sv, sn = wkey.split('__', 1)
                        a = V.auc_mw(ycell, scores[sv][sn])
                        d = abs(a - e['score_auc'])
                        out['score_auc_max_abs_diff'] = max(out['score_auc_max_abs_diff'], d)
                        if bool(gap is not None and gap < 0.010) != bool(e['near_tie']):
                            out['winner_mismatch'].append(f'{cell}/{name}/{view}: near-tie flag differs')
                        if n_boot < max_boot_checks and '@' not in cell:
                            if W is None:
                                W = boot_weights_own(unit[sc], s, rep)
                            mine_b = np.array([V.auc_mw(ycell, scores[sv][sn], W[b]) for b in range(W.shape[0])])
                            out['boot_max_abs_diff'] = max(out['boot_max_abs_diff'], float(np.max(np.abs(mine_b - boot[f'{cell}__{name}__{view}']))))
                            n_boot += 1
                    if '@' not in cell:
                        out['per_recipe_score_auc'].setdefault(name, {})[cell] = {
                            view: {n: V.auc_mw(sex[sc], scores[view][n]) for n in SRC8 + NEW8} for view in VIEWS}
            worst['score_auc'] = max(worst['score_auc'], out['score_auc_max_abs_diff'])
            worst['boot'] = max(worst['boot'], out['boot_max_abs_diff'])
            out.pop('per_recipe_score_auc')
            rec['banks'][f'{bank}/S{s}'] = out
    rec['n_boot_vectors_checked'] = n_boot
    rec['score_auc_max_abs_diff'] = worst['score_auc']
    rec['boot_max_abs_diff'] = worst['boot']
    rec['status'] = 'PASS' if (worst['score_auc'] < 1e-12 and worst['boot'] < 1e-12 and
                               all(not v['missing_cache'] and not v['winner_mismatch'] for v in rec['banks'].values())) else 'ATTENTION'
    V.write_record('VER_GRID_CACHE' + ('_' + '_'.join(banks) if banks else ''), rec)
    print(json.dumps({k: v for k, v in rec.items() if k != 'banks'}, indent=1))
    print(json.dumps({b: {k: (v if not isinstance(v, list) else v[:5]) for k, v in o.items()} for b, o in rec['banks'].items()}, indent=1))


# ------------------------------------------------------------------------------------------------ mode: refit (own readers)
def kinds_own(p, k):
    p = np.asarray(p, dtype=np.float64)
    d = np.argmax(p, axis=1)
    one = np.zeros((len(p), k)); one[np.arange(len(p)), d] = 1.0
    return {'out': np.hstack([p, d[:, None].astype(np.float64)]), 'log': np.hstack([np.log(np.clip(p, 1e-300, None)), one]),
            'dec': one, 'conf': p.max(axis=1)[:, None]}


def rich_own(p):
    p = np.asarray(p, dtype=np.float64)
    one = np.zeros_like(p); one[np.arange(len(p)), np.argmax(p, axis=1)] = 1.0
    return np.hstack([p, np.log(np.clip(p, 1e-300, None)), one])


class Std:
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.m = x.mean(axis=0)
        sd = x.std(axis=0)
        self.s = np.where(sd > 0, sd, 1.0)
        return self

    def tr(self, x):
        return (np.asarray(x, dtype=np.float64) - self.m) / self.s


def spline_own(x, n_knots):
    from sklearn.preprocessing import SplineTransformer
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    knots = np.unique(np.quantile(x[:, 0], np.linspace(0, 1, n_knots)))
    if len(knots) < 2:
        return None
    return SplineTransformer(knots=knots[:, None], degree=3, extrapolation='constant', include_bias=True).fit(x)


class CatiOwn:
    """Own implementation of the registered categorical-interaction design (fitted on FIT outputs only)."""

    def __init__(self, view):
        self.view = view

    def _inc_fit(self, inc):
        p = np.ascontiguousarray(np.asarray(inc, dtype=np.float64))
        uniq = {r.tobytes(): r for r in p}
        if len(uniq) <= 32:
            self.inc_kind = 'cat'
            keys = sorted(uniq, key=lambda b: (float(uniq[b][1]), b))
            self.inc_index = {b: j for j, b in enumerate(keys)}
        else:
            self.inc_kind = 'q'
            self.cuts = np.quantile(p[:, 1], np.arange(1, 6) / 6)

    def _inc_ind(self, inc):
        p = np.ascontiguousarray(np.asarray(inc, dtype=np.float64))
        if self.inc_kind == 'cat':
            M = np.zeros((len(p), len(self.inc_index)))
            for i in range(len(p)):
                j = self.inc_index.get(p[i].tobytes())
                if j is not None:
                    M[i, j] = 1.0
            return M
        return np.eye(6)[np.searchsorted(self.cuts, p[:, 1], side='left')]

    def _inc_cont(self, inc):
        if self.inc_kind == 'cat':
            return np.zeros((len(inc), 0))
        p = np.asarray(inc, dtype=np.float64)
        return np.hstack([p, np.log(np.clip(p, 1e-300, None))])

    @staticmethod
    def _occ9(occ):
        p = np.asarray(occ, dtype=np.float64)
        lp = np.log(np.clip(p, 1e-300, None))
        return np.hstack([p, lp[:, 1:] - lp[:, :1]])

    @staticmethod
    def _logit(inc):
        p = np.asarray(inc, dtype=np.float64)
        return (np.log(np.clip(p[:, 1], 1e-300, None)) - np.log(np.clip(p[:, 0], 1e-300, None))).reshape(-1, 1)

    def fit(self, inc, occ):
        if self.view in ('income', 'pair'):
            self._inc_fit(inc)
            if self.view == 'income' and self.inc_kind == 'q':
                self.inc_spl = spline_own(self._logit(inc), 6)
        if self.view == 'occupation':
            lc = self._occ9(occ)[:, 5:]
            self.occ_spl = [spline_own(lc[:, j], 5) for j in range(4)]
        return self

    def transform(self, inc, occ):
        if self.view == 'pair':
            M = self._inc_ind(inc); O = self._occ9(occ)
            inter = np.einsum('ij,ik->ijk', M, O).reshape(len(M), -1)
            return np.hstack([M, self._inc_cont(inc), O, inter])
        if self.view == 'income':
            M = self._inc_ind(inc)
            if self.inc_kind == 'cat':
                return M
            blocks = [M, self._inc_cont(inc)]
            if self.inc_spl is not None:
                blocks.append(self.inc_spl.transform(self._logit(inc)))
            return np.hstack(blocks)
        O = self._occ9(occ); lc = O[:, 5:]
        spl = [self.occ_spl[j].transform(lc[:, [j]]) for j in range(4) if self.occ_spl[j] is not None]
        prods = np.stack([lc[:, i] * lc[:, j] for i in range(4) for j in range(i + 1, 4)], axis=1)
        dec = np.eye(5)[np.argmax(np.asarray(occ), axis=1)]
        return np.hstack([O] + spl + [prods, dec])


def model_own(name, rs):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    if name in ('OUT_MLP', 'LOG_MLP'):
        return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=1e-4, max_iter=300, early_stopping=False, random_state=rs)
    if name in ('OUT_HGB', 'CONF_HGB'):
        return HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, early_stopping=False, random_state=rs)
    if name in ('OUT_LR', 'LOG_LR', 'DEC_LR'):
        return LogisticRegression(C=1, max_iter=500, solver='lbfgs')
    if name in ('ET5', 'ET25'):
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5 if name == 'ET5' else 25, max_features=1.0, n_jobs=1, random_state=rs)
    if name in ('HGB15', 'HGB31'):
        return HistGradientBoostingClassifier(max_leaf_nodes=15 if name == 'HGB15' else 31, max_iter=200, learning_rate=0.05, min_samples_leaf=30,
                                              l2_regularization=1.0, early_stopping=False, random_state=rs)
    if name in ('MLP64', 'MLP6432'):
        return MLPClassifier(hidden_layer_sizes=(64,) if name == 'MLP64' else (64, 32), alpha=0.01, batch_size=256, max_iter=300,
                             early_stopping=False, random_state=rs)
    if name in ('CATI01', 'CATI1'):
        return LogisticRegression(C=0.1 if name == 'CATI01' else 1.0, solver='lbfgs', max_iter=5000)
    raise ValueError(name)


SRC_KIND = {'OUT_LR': 'out', 'OUT_MLP': 'out', 'OUT_HGB': 'out', 'LOG_LR': 'log', 'LOG_MLP': 'log', 'DEC_LR': 'dec', 'CONF_HGB': 'conf'}


def features_own(name, view, inc, occ, fit):
    if name in SRC_KIND:
        a, b = kinds_own(inc, 2)[SRC_KIND[name]], kinds_own(occ, 5)[SRC_KIND[name]]
        return a if view == 'income' else b if view == 'occupation' else np.hstack([a, b])
    if name.startswith('CATI'):
        d = CatiOwn(view).fit(inc[fit], occ[fit])
        return d.transform(inc, occ)
    return rich_own(inc) if view == 'income' else rich_own(occ) if view == 'occupation' else np.hstack([rich_own(inc), rich_own(occ)])


def lookup_own(inc, occ, view, sex, fit):
    arrs = [inc] if view == 'income' else [occ] if view == 'occupation' else [inc, occ]
    Z = np.ascontiguousarray(np.hstack([np.asarray(a, dtype=np.float64) for a in arrs]))
    keys = [Z[i].tobytes() for i in range(len(Z))]
    pi = float(np.mean(sex[fit]))
    tab = {}
    for i in np.flatnonzero(fit):
        n1, n = tab.get(keys[i], (0, 0))
        tab[keys[i]] = (n1 + int(sex[i]), n + 1)
    return np.array([(tab[k][0] + 2.0 * pi) / (tab[k][1] + 2.0) if k in tab else pi for k in keys])


def fit_one_own(name, view, inc, occ, sex, fit, val, rs):
    if name == 'LOOKUP':
        raw = lookup_own(inc, occ, view, sex, fit)
    else:
        X = features_own(name, view, inc, occ, fit)
        sc = Std().fit(X[fit])
        m = model_own(name, rs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.fit(sc.tr(X[fit]), sex[fit])
        if list(m.classes_) != [0, 1]:
            raise ValueError('class order')
        raw = m.predict_proba(sc.tr(X))[:, 1].astype(np.float64)
    sign = 1 if V.auc_mw(sex[val], raw[val]) >= 0.5 else -1
    s = raw if sign == 1 else 1.0 - raw
    return s, V.auc_mw(sex[val], s[val])


def run_refit(entries, tag):
    """entries: list of 'BANK:MEMBER:CELL' (CELL like S0R1k0)."""
    sex, unit = ms_sex('independent dev-grid reader refits (VER check 6): ' + ', '.join(entries))
    masks = {rep: grid_roles(unit, rep) for rep in ('R1', 'R2')}
    prev = V.RECORDS / 'VER_GRID_REFIT.json'
    db = json.loads(prev.read_text()) if prev.is_file() else {'check': 'VER_GRID_REFIT', 'entries': {}, 'units': []}
    t_all = time.process_time()
    for ent in entries:
        bank, name, cell = ent.split(':')
        s, rep, k = int(cell[1]), cell[2:4], int(cell[-1])
        g = json.loads((V.RUNS / 'GRID' / bank / f'S{s}' / 'GRID.json').read_text())
        r = load_member(g['spec']['members'][name], s)
        fit, val, sc = masks[rep]
        rs = reader_seed(s, k)
        t0 = time.process_time()
        info, scores, diffs = {}, {}, {}
        for view in VIEWS:
            key = cache_key(rep, rs, 1.0, view, r['income'], r['occupation'], sex)
            with np.load(V.RUNS / 'devgrid_cache' / f'{key}.npz', allow_pickle=False) as z:
                cinfo = json.loads(str(z['info']))
                cscore = {n: z[f'score__{n}'].copy() for n in SRC8 + NEW8}
            info[view], scores[view] = {}, {}
            for n in SRC8 + NEW8:
                s_all, vauc = fit_one_own(n, view, r['income'], r['occupation'], sex, fit, val, rs)
                info[view][n] = vauc
                scores[view][n] = s_all[sc]
                diffs[f'{view}__{n}'] = {'val_auc_mine': vauc, 'val_auc_grid': cinfo[n]['val_auc'], 'val_auc_abs_diff': abs(vauc - cinfo[n]['val_auc']),
                                         'score_max_abs_diff': float(np.max(np.abs(s_all[sc] - cscore[n]))),
                                         'score_auc_mine': V.auc_mw(sex[sc], s_all[sc]), 'score_auc_cached': V.auc_mw(sex[sc], cscore[n]),
                                         'orientation_equal': None}
        w = winners_from(info)
        e = g['cells'][cell][name]
        win = {}
        for view in VIEWS:
            (wname, wval, wkey), gap = w[view]
            sv, sn = wkey.split('__', 1)
            a = V.auc_mw(sex[sc], scores[sv][sn])
            win[view] = {'mine': wname, 'grid': e[view]['winner'], 'same_winner': wname == e[view]['winner'], 'score_auc_mine': a,
                         'score_auc_grid': e[view]['score_auc'], 'score_auc_abs_diff': abs(a - e[view]['score_auc']),
                         'val_auc_mine': wval, 'val_auc_grid': e[view]['val_auc'], 'gap_mine': gap}
        db['entries'][ent] = {'winners': win, 'recipes': diffs, 'cpu_s': round(time.process_time() - t0, 1), 'unit_id': V.unit_id(), 'utc': V.utc(),
                              'max_val_auc_abs_diff': max(d['val_auc_abs_diff'] for d in diffs.values()),
                              'max_score_abs_diff': max(d['score_max_abs_diff'] for d in diffs.values()),
                              'recipes_bitwise_equal': sum(d['score_max_abs_diff'] == 0.0 for d in diffs.values()),
                              'n_recipes': len(diffs)}
        print(ent, json.dumps({v: (x['mine'], x['grid'], round(x['score_auc_abs_diff'], 6)) for v, x in win.items()}),
              'max val diff', round(db['entries'][ent]['max_val_auc_abs_diff'], 6), 'bitwise', db['entries'][ent]['recipes_bitwise_equal'], '/48', flush=True)
    db['units'].append({'unit_id': V.unit_id(), 'utc': V.utc(), 'entries': entries, 'cpu_s': round(time.process_time() - t_all, 1), 'tag': tag})
    V.write_record('VER_GRID_REFIT', db)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('paired', 'cache', 'refit'))
    ap.add_argument('--summary', nargs='*', default=[])
    ap.add_argument('--bank', nargs='*', default=[])
    ap.add_argument('--max-boot', type=int, default=60)
    ap.add_argument('--entries', nargs='*', default=[])
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    if a.mode == 'paired':
        run_paired(a.summary)
    elif a.mode == 'cache':
        run_cache(a.bank, a.max_boot)
    else:
        run_refit(a.entries, a.tag)


if __name__ == '__main__':
    main()
