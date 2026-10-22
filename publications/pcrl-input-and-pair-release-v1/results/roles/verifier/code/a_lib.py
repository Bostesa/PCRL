"""Addendum helpers of the independent verifier (campaign `pcrl_input_and_pair_release_v1`; post-registration readings).

Own code, written from PROTOCOL.md, RULES.md, the cards IA_OSG_R1 / IA_SFG_v1 / CTL_PATHS_v1 and the NRM1 definitions quoted in RULES.md.
Nothing here imports ipr.evaluate, ipr.devreaders, ipr.rules, ipr.claims or nrm.rules. No label is read here (label reads happen only
in the verifier's own scripts through ipr.common.load, operation 'verify', inside queue units).

Contents: member resolution (family configurations from the configuration ledger, shared role members, frozen references); cache
entries; registered pair-candidate order; the aggregate screen with n_eff and the reader-seed qualifier; the two-SE reserve; the
greedy ladder rule; the PROTOCOL 4 bracket rule and the fineness classes.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

import vlib as V

SRC8 = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP')
NEW8 = ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1')
SEED_DEPENDENT = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
REFS_FROZEN = ('NEURAL_RAW', 'H_LJ')
NINE = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100')
SECONDARY = ('income|acc_loss', 'income|brier', 'income|nll', 'occupation|acc_loss', 'occupation|brier', 'occupation|nll',
             'occupation|men_brier', 'occupation|women_brier')
FIELDS_MENU = {'AGEP': ('AGE5', 'AGE10'), 'WKHP': ('WK6', 'WK5', 'WK3'), 'COW': ('COW5', 'COW4', 'COW3'),
               'SCHL': ('SCH5', 'SCH4', 'SCH3'), 'MAR': ('MAR3', 'MAR2', 'MAR1'), 'POBP': ('POB4', 'POB2', 'POB1')}
FIELD_OF = {s: f for f, lv in FIELDS_MENU.items() for s in lv}
OHN_RULES = V.ROOT / 'results' / 'pcrl_occupation_headroom_overnight_v1' / 'roles' / 'statistical_reviewer' / 'RULES.json'


def ledger_family():
    rows = [json.loads(l) for l in V.CONFIG_LEDGER.read_text().splitlines() if l.strip()]
    return {r['config_id']: r['family'] for r in rows}, rows


def roles_map():
    return json.loads((V.RESULTS / 'paths' / 'SHARED_ROLES.json').read_text())['roles']


def load_ms(fam, cfg, v):
    d = V.RUNS / fam / cfg / f'V{int(v)}'
    rec = json.loads((d / 'RELEASE.json').read_text())
    if V.file_sha(d / 'RELEASE.npz') != rec['release_sha256']:
        raise RuntimeError(f'{fam}/{cfg}/V{v}: RELEASE.npz hash differs from RELEASE.json')
    with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
        return {t: z[f'MODEL_SELECT__{t}'].copy() for t in V.TASKS}, rec


def member_ms(name, v, fam_of, roles, bank=None):
    """MODEL_SELECT vectors of a bank member: a saved configuration of this campaign, a shared role member, or a frozen reference."""
    if name in roles:
        fam, cfg = roles[name][str(v)].split('/')
        return load_ms(fam, cfg, v)[0]
    if name in fam_of:
        return load_ms(fam_of[name], name, v)[0]
    if bank is not None and name in REFS_FROZEN:
        return {t: np.asarray(bank[v][name]['MODEL_SELECT'][t], dtype=np.float64) for t in V.TASKS}
    raise KeyError(name)


def read_entry(path):
    with np.load(path, allow_pickle=False) as z:
        info = json.loads(str(z['info']))
        scores = {k[len('score__'):]: z[k].copy() for k in z.files if k.startswith('score__')}
    return scores, info


def resolve(w):
    if w.startswith('IGNORE_occupation__'):
        return 'income', w[len('IGNORE_occupation__'):]
    if w.startswith('IGNORE_income__'):
        return 'occupation', w[len('IGNORE_income__'):]
    return 'pair', w


def pair_candidates(ent, allowed=None):
    out = []
    for group in (SRC8, NEW8):
        g = [n for n in group if allowed is None or n in allowed]
        out += [(n, ent['pair'][1][n]['val_auc']) for n in g]
        out += [(f'IGNORE_occupation__{n}', ent['income'][1][n]['val_auc']) for n in g]
        out += [(f'IGNORE_income__{n}', ent['occupation'][1][n]['val_auc']) for n in g]
    return out


def cell_parts(c):
    return int(c[1]), c[2:4], int(c[5:])


def aggregate(cells, ses, which):
    """Own aggregate screen over the named cells (RULES.md 0 / NRM1): blocks (version, repetition); x = mean of block means; se_hh =
    RMS of cell SEs with the weights of x; se_rs = sqrt(sum_b (m_b - x)^2 / n_b); se_agg = max(raw, 0.001) unless an exact alias;
    per-version quantities; n_eff = distinct cell values within blocks; the reader-seed qualifier."""
    names = [c for c in which]
    if not all(c in cells and c in ses for c in names):
        return {'complete': False, 'missing': sorted(c for c in names if c not in cells or c not in ses)}
    blocks = {}
    for c in names:
        v, r, k = cell_parts(c)
        blocks.setdefault((v, r), {})[k] = (float(cells[c]), float(ses[c]))

    def summ(keys):
        nb = len(keys)
        mb = {b: sum(t[0] for t in blocks[b].values()) / len(blocks[b]) for b in keys}
        x = sum(mb.values()) / nb
        hh = math.sqrt(sum(blocks[b][k][1] ** 2 / (nb * len(blocks[b])) for b in keys for k in blocks[b]))
        rs = math.sqrt(sum((m - x) ** 2 for m in mb.values()) / nb)
        return {'mean': x, 'se_hh': hh, 'se_rs': rs, 'raw': math.sqrt(hh * hh + rs * rs)}
    keys = sorted(blocks)
    if sorted({b[0] for b in keys}) != [0, 1]:
        return {'complete': False, 'missing': ['a version']}
    tot = summ(keys)
    per = {v: summ([b for b in keys if b[0] == v]) for v in (0, 1)}
    alias = all(t[0] == 0.0 and t[1] == 0.0 for b in blocks.values() for t in b.values())
    for s in [tot] + list(per.values()):
        s['se_agg'] = 0.0 if alias else max(s['raw'], V.SE_FLOOR)
        s['lower'], s['upper'] = s['mean'] - V.Z * s['se_agg'], s['mean'] + V.Z * s['se_agg']
    kd = {b: len({t[0] for t in blocks[b].values()}) > 1 for b in keys if len(blocks[b]) > 1}
    n_eff = sum(len({t[0] for t in blocks[b].values()}) for b in keys)
    nk = sum(kd.values())
    qual = ('READER_SEEDS_EFFECTIVE_ALL_BLOCKS' if kd and nk == len(kd) else
            'READER_SEED_REPLICATION_ABSENT' if nk == 0 else 'READER_SEEDS_PARTLY_EFFECTIVE')
    return {'complete': True, 'alias': alias, **tot, 'per_version': per, 'n_eff': n_eff, 'qualifier': qual, 'n_cells': len(names)}


def benefit_label(a, threshold=0.0):
    return V.benefit_label(a, threshold)


def recipient_label(a):
    return V.recipient_label(a)


def resolved_negative(a):
    if not a.get('complete') or a['alias']:
        return False if a.get('complete') else None
    return a['upper'] < 0 and all(a['per_version'][v]['mean'] < 0 for v in (0, 1))


def version_means(cells):
    out = {}
    for v in (0, 1):
        blocks = {}
        for c in V.REG_CELLS:
            if int(c[1]) == v:
                blocks.setdefault(c[2:4], []).append(cells[c])
        out[v] = sum(sum(b) / len(b) for b in blocks.values()) / len(blocks)
    return out


# ---------------------------------------------------------------- the two-SE reserve (OHN1 rule, own arithmetic)
def reserve(clauses, guards, households, v, floor=None):
    """clauses: {'<task>|<ref>|<metric>': (value, se, cap)}; guards: {'sex0'|'sex1': (value, se, cap)}. Per condition: margin =
    cap - value >= k max(own SE, floor SE x sqrt(G0 / G)), k = 2; floor = the frozen H_LJ MODEL_SELECT SE of the same condition and
    version (ohn RULES.json, read as data). Returns per-version whole / occupation-side / income-side status and minimum ratios."""
    rr = json.loads(OHN_RULES.read_text())['reserve'] if floor is None else floor
    k = float(rr['multiplier_k'])
    G0 = int(rr['floor']['households'])
    fl = rr['floor']['se'][str(int(v))]
    scale = math.sqrt(G0 / int(households))
    conds = {**{c: x for c, x in clauses.items()}, **{f'guard|{g}': x for g, x in guards.items()}}
    rec = {}
    for name, (value, se, cap) in conds.items():
        f = float(fl[name]) * scale
        used = max(float(se), f)
        margin = float(cap) - float(value)
        rec[name] = {'ratio': margin / used, 'met': bool(margin >= k * used), 'se_source': 'own' if se >= f else 'floor'}
    occ = [n for n in rec if n.startswith('occupation|') or n.startswith('guard|')]
    inc = [n for n in rec if n.startswith('income|')]
    worst = min(rec, key=lambda n: rec[n]['ratio'])
    worst_o = min(occ, key=lambda n: rec[n]['ratio'])
    return {'whole_met': all(r['met'] for r in rec.values()), 'min_whole_ratio': rec[worst]['ratio'], 'whole_binding': worst,
            'occ_met': all(rec[n]['met'] for n in occ), 'min_occ_ratio': rec[worst_o]['ratio'], 'occ_binding': worst_o,
            'income_met': all(rec[n]['met'] for n in inc), 'min_income_ratio': min(rec[n]['ratio'] for n in inc),
            'failed': sorted(n for n, r in rec.items() if not r['met'])}


def reserve_status(per_version):
    met = [v for v in (0, 1) if per_version[v]['whole_met']]
    return 'RESERVE_MET' if len(met) == 2 else ('RESERVE_ONE_SEED_ONLY' if met else 'RESERVE_NOT_MET')


# ---------------------------------------------------------------- MODEL_FIT selection, ladders, brackets, fineness
def meets(sp, budget):
    return all(sp[m]['estimate'] + 2 * sp[m]['se'] <= lim for m, lim in budget.items())


def scan_select(path, budget):
    """k* = the largest rung such that it and every smaller rung meet the budget; the scan stops at the first failing rung."""
    pick, first_fail = None, None
    for g in sorted(path, key=float):
        if meets(path[g], budget):
            pick = g
        else:
            first_fail = g
            break
    ratio = None
    if first_fail is not None:
        sp = path[first_fail]
        ratio = {m: (sp[m]['estimate'] + 2 * sp[m]['se']) / lim for m, lim in budget.items()
                 if sp[m]['estimate'] + 2 * sp[m]['se'] > lim}
    return pick, first_fail, ratio


def build_ladder(stats, order, dkey='dT'):
    """Greedy rule of card IA_SFG_v1 / IA_OSG_R1 (own implementation): each field moves only to a coarser level; incremental ratio
    against the field's current level; 'target': largest incremental dI / dT first (dT floored at 1e-5); 'blind': smallest incremental
    dT first; ties broken by (-dT, step name)."""
    cur = {f: None for f in FIELDS_MENU}
    ladder = []
    while True:
        cand = []
        for f, lv in FIELDS_MENU.items():
            i0 = -1 if cur[f] is None else lv.index(cur[f])
            T0 = 0.0 if cur[f] is None else stats[cur[f]][dkey]
            I0 = 0.0 if cur[f] is None else stats[cur[f]].get('dI', 0.0)
            for st in lv[i0 + 1:]:
                dT = max(stats[st][dkey] - T0, 1e-5)
                dI = stats[st].get('dI', 0.0) - I0
                key = -dT if order == 'blind' else dI / dT
                cand.append((key, -dT, st))
        if not cand:
            return ladder
        st = max(cand)[2]
        cur[FIELD_OF[st]] = st
        ladder.append(st)


def compose(ladder, k):
    """The steps of rung k: each field at its last level among the first k ladder steps."""
    cur = {}
    for st in ladder[:k]:
        cur[FIELD_OF[st]] = st
    return sorted(cur.values())


def bracket(b):
    grid = b['grid']
    gs = sorted(grid, key=float)
    lo = [g for g in gs if grid[g]['spend'] <= b['cand_spend']]
    hi = [g for g in gs if grid[g]['diff_vs_cand'] >= 2 * grid[g]['se_diff']]
    return (float(lo[-1]) if lo else None, float(hi[0]) if hi else float(gs[-1]), None if hi else 'CONTROL_PATH_MAY_NOT_BRACKET')


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


def seed_fingerprint_checks(cells, members, pairs=((0, 1),)):
    """From GRID.json cells (merged over versions): for every member, view, version, repetition and seed pair, the seed-dependent
    recipes whose recorded SCORE fingerprints are equal across the pair (an equality on the minimum set needs the extra-trees
    explanation; none means the reader seeds are effective)."""
    equal, n = [], 0
    for c in cells:
        v, r, k = cell_parts(c)
        for a, b in pairs:
            if k != a:
                continue
            other = f'S{v}{r}k{b}'
            if other not in cells:
                continue
            for m in members:
                for view in V.VIEWS:
                    f0 = (((cells[c].get(m) or {}).get(view) or {}).get('extras') or {}).get('score_sha256') or {}
                    f1 = (((cells[other].get(m) or {}).get(view) or {}).get('extras') or {}).get('score_sha256') or {}
                    for rc in SEED_DEPENDENT:
                        if rc in f0 and rc in f1:
                            n += 1
                            if f0[rc] == f1[rc]:
                                equal.append(f'{c}|{m}|{view}|{rc}')
    return {'checks': n, 'equal': equal}
