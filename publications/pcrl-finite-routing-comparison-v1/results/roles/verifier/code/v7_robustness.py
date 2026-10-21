"""V7 robustness grid and classifier-stability replay (label-free), own code.

Inputs: runs/GRID/<rob bank>/S<v>/GRID.json + BOOT.npz, results/<campaign>/claims/STABILITY/CLASSIFIER_STABILITY.json,
READER_SPLITS.json, the admitted MODEL_SELECT households (frc.common.features, label-free). Recomputed with my own code
(v6_grid_claims functions):
- every paired cell value and household-bootstrap SE of the robustness bank from its BOOT.npz;
- robustness labels of PRC_S2 against TWIN, HQ_S2, SHUF_S2, UNT_BR, ORD_S2, NEURAL_RAW, H_LJ: ROBUSTNESS_CONFIRMED iff the
  aggregate benefit is AGG_POSITIVE on the robustness cells (every cell other than the 8 registered ones) and on all cells of the
  entry; NOT_RUN if the entry or a version is missing;
- the ablations ABL_LOC / ABL_PAIR against PRC_S2 and TWIN: registered comparison (benefit label) and robustness label;
- the split identities: households ordered by SHA-256('FRC1_devgrid_R<r>|<household>') dealt into 10 parts (FIT 0-4, VALIDATION
  5-6, SCORE 7-9), assignment hash = SHA-256 of the int8 role code (0 / 1 / 2) per admitted MODEL_SELECT row, SCORE household
  overlaps between repetitions;
- refit effectiveness: (a) R3 / R4 assignment hashes distinct from each other and from R1 / R2, every overlap involving R3 / R4 in
  [0, 1); (b) the STRICT grid validity (no ER1) on all eight seeds S<v>k<k>, k = 0..3, with k0 / k1 and k2 / k3 fingerprint pairs,
  minimum set PRC_S2, TWIN, HQ_S2, UNT_BR, ORD_S2.

    python3 .../v7_robustness.py --tag V1 [--rob FRC1_ROB] [--main FRC1_MAIN]
    python3 .../v7_robustness.py --selftest
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402
import v6_grid_claims as G  # noqa: E402

COMPS = ('TWIN', 'HQ_S2', 'SHUF_S2', 'UNT_BR', 'ORD_S2', 'NEURAL_RAW', 'H_LJ')
MIN_SET = ('PRC_S2', 'TWIN', 'HQ_S2', 'UNT_BR', 'ORD_S2')
REPS = ('R1', 'R2', 'R3', 'R4')
TOL = 1e-12


def cells_of(entry, which):
    names = [c for c in (entry or {}).get('cells', {})]
    if which == 'robustness':
        names = [c for c in names if c not in G.REG]
    return sorted(names)


def screen(entry, which):
    cells = cells_of(entry, which)
    if not cells or sorted({int(c[1]) for c in cells}) != [0, 1]:
        return {'label': 'INCOMPLETE', 'resolved': False, 'cells': cells}
    a = G.aggregate(entry, cells)
    if a is None:
        return {'label': 'INCOMPLETE', 'resolved': False}
    if a['alias']:
        lab = 'ALIAS_ZERO'
    elif a['lower'] > 0 and all(a['per_version'][v]['mean'] > 0 for v in G.VERS):
        lab = 'AGG_POSITIVE'
    elif a['mean'] > 0:
        lab = 'AGG_POSITIVE_POINT_ONLY'
    else:
        lab = 'AGG_NOT_POSITIVE'
    return {'label': lab, 'resolved': lab == 'AGG_POSITIVE', 'mean': a['mean'], 'se_agg': a['se_agg'], 'n_cells': len(cells)}


def robustness(entry):
    rob = screen((entry or {}).get('pair'), 'robustness')
    if rob['label'] == 'INCOMPLETE':
        return {'label': 'NOT_RUN'}
    allc = screen(entry.get('pair'), 'all')
    return {'label': 'ROBUSTNESS_CONFIRMED' if (rob['resolved'] and allc['resolved']) else 'ROBUSTNESS_NOT_CONFIRMED',
            'robustness_cells': rob, 'all_cells': allc}


def splits(unit):
    unit = np.asarray(unit)
    roles = {}
    for r in REPS:
        part = V.household_folds(unit, 10, f'FRC1_devgrid_{r}')
        code = np.where(part <= 4, 0, np.where(part <= 6, 1, 2)).astype(np.int8)
        roles[r] = (code, set(np.unique(unit[part >= 7]).tolist()))
    out = {}
    for r, (code, sc) in roles.items():
        out[r] = {'assignment_sha256': hashlib.sha256(code.tobytes()).hexdigest(), 'score_households': len(sc),
                  'score_overlap': {q: len(sc & roles[q][1]) / len(sc) for q in REPS if q != r}}
    return out


def strict_validity(bank, distinct):
    man = json.loads((V.RESULTS / 'READER_SPLITS.json').read_text())
    seeds = {f'S{v}k{k}': man['reader_seeds'][f'V{v}k{k}'] for v in G.VERS for k in range(4)}
    errors = []
    forb = {G.SEED_BASE + off + 10 * v + k for off in (300, 400, 500, 600) for v in G.VERS for k in range(10)}
    sv = list(seeds.values())
    if len(set(sv)) != len(sv) or set(sv) & forb or any(seeds[f'S{v}k{k}'] != G.SEED_BASE + 700 + 10 * v + k
                                                        for v in G.VERS for k in range(4)):
        errors.append('seed rule')
    checks = []
    for cell, cres in bank['cells'].items():
        v, rep, k = G.parse(cell)
        if k not in (0, 2):
            continue
        other = cell[:-1] + str(k + 1)
        for member, ent in cres.items():
            for vw, e in ent.items():
                f0 = (e.get('extras') or {}).get('score_sha256') or {}
                f1 = (((bank['cells'].get(other) or {}).get(member) or {}).get(vw, {}).get('extras') or {}).get('score_sha256') or {}
                for rcp in G.SEED_DEP + G.SEED_INV:
                    if rcp in f0 and rcp in f1:
                        checks.append((rcp, vw, member, v, rep, f0[rcp] == f1[rcp], k))
    eq = [c for c in checks if c[0] in G.SEED_DEP and c[5]]
    diff = {(c[2], c[1], c[3], c[4], c[0]) for c in checks if c[0] in G.SEED_DEP and not c[5]}
    on_min = [c for c in eq if c[2] in MIN_SET]
    if on_min:
        errors.append(f'{len(on_min)} seed-dependent equalities on the minimum set')
    unexpl = []
    for c in eq:
        if c[2] in MIN_SET:
            continue
        a = any((c[2], c[1], c[3], c[4], r) in diff for r in G.SEED_DEP if r != c[0])
        b = any((c[2], x, c[3], c[4], c[0]) in diff for x in G.VIEWS if x != c[1])
        n = (distinct.get(f'{c[2]}|V{c[3]}') or {}).get(c[1])
        if not (a and b and isinstance(n, int)):
            unexpl.append(c)
    if unexpl:
        errors.append(f'{len(unexpl)} unexplained seed-dependent equalities')
    return {'effective': not errors, 'errors': errors, 'n_checks': len(checks), 'minimum_set_equalities': len(on_min)}


def run(tag, rob_id, main_id):
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    import frc.common as C
    checks, record = [], {'tag': tag, 'utc': V.utc(), 'rob': rob_id}
    rob = G.load_bank(rob_id)
    worst_d = worst_s = 0.0
    n = 0
    for cand, comps in rob['paired'].items():
        for comp, rec in comps.items():
            mine = G.my_paired(rob, cand, comp)
            for vw in G.VIEWS:
                for c, x in rec[vw]['cells'].items():
                    worst_d = max(worst_d, abs(x - mine[vw]['cells'][c]))
                    worst_s = max(worst_s, abs(rec[vw]['cell_se'][c] - mine[vw]['cell_se'][c]))
                    n += 1
    checks.append({'id': f'V7.GRID_PAIRED.{rob_id}__{tag}', 'what': f'{rob_id}: every paired cell value and bootstrap SE from BOOT.npz',
                   'inputs': V.inputs_sha([V.RUNS / 'GRID' / rob_id / f'S{v}' / f for v in G.VERS for f in ('GRID.json', 'BOOT.npz')]),
                   'result': {'n_cell_values': n, 'max_abs_value_diff': worst_d, 'max_abs_se_diff': worst_s},
                   'numerical_status': 'PASS' if (n and worst_d <= TOL and worst_s <= TOL) else 'FAIL',
                   'procedural_status': 'NOT_APPLICABLE'})
    rb = {c: robustness(G.my_paired(rob, 'PRC_S2', c)) for c in COMPS if 'PRC_S2' in rob['paired'] and c in rob['paired']['PRC_S2']}
    abl = {}
    for a in ('ABL_LOC', 'ABL_PAIR'):
        for comp in ('PRC_S2', 'TWIN'):
            if a in rob['paired'] and comp in rob['paired'][a]:
                e = G.my_paired(rob, a, comp)
                abl[f'{a}_vs_{comp}'] = {'registered': G.comparison(e)['benefit']['label'], 'robustness': robustness(e)['label']}
    unit = C.features('MODEL_SELECT')['unit']
    sp = splits(unit)
    dist = G.distinct_counts(rob['members'])
    gv = strict_validity(rob, dist)
    errs = []
    hs = [sp[r]['assignment_sha256'] for r in REPS]
    if len(set(hs)) != len(hs):
        errs.append('SPLITS_NOT_DIFFERENT_OR_NOT_RECORDED')
    for r in ('R3', 'R4'):
        for q in REPS:
            if q != r and not 0 <= sp[r]['score_overlap'][q] < 1:
                errs.append(f'SCORE_OVERLAP_{r}_{q}_NOT_RECORDED_OR_IDENTICAL')
    if not gv['effective']:
        errs.append('READER_SEEDS_NOT_EFFECTIVE')
    refit = 'REFIT_GENUINELY_EFFECTIVE' if not errs else 'REFIT_NOT_GENUINELY_EFFECTIVE'
    mine = {'robustness': {c: x['label'] for c, x in rb.items()}, 'ablation': abl, 'splits': sp, 'refit': refit, 'refit_reasons': errs,
            'strict_validity': gv}
    sp_path = V.RESULTS / 'claims' / 'STABILITY' / 'CLASSIFIER_STABILITY.json'
    mm = []
    if sp_path.is_file():
        st = json.loads(sp_path.read_text())
        for c, lab in mine['robustness'].items():
            if lab != (st.get('robustness', {}).get(c) or {}).get('label'):
                mm.append(f'robustness {c} mine {lab} recorded {(st.get("robustness", {}).get(c) or {}).get("label")}')
        for k, x in abl.items():
            r = (st.get('ablation') or {}).get(k) or {}
            if x['registered'] != ((r.get('registered') or {}).get('benefit') or {}).get('label') or \
                    x['robustness'] != (r.get('robustness') or {}).get('label'):
                mm.append(f'ablation {k} mine {x} recorded {r and (((r.get("registered") or {}).get("benefit") or {}).get("label"), (r.get("robustness") or {}).get("label"))}')
        for r in REPS:
            rs = (st.get('splits') or {}).get(r) or {}
            if rs.get('assignment_sha256') != sp[r]['assignment_sha256'] or any(abs(rs.get('score_overlap', {}).get(q, -1) - x) > TOL
                                                                                 for q, x in sp[r]['score_overlap'].items()):
                mm.append(f'split {r} differs')
        re_ = st.get('refit_effectiveness') or {}
        if refit != re_.get('label') or sorted(errs) != sorted(re_.get('reasons') or []):
            mm.append(f'refit mine {refit} {errs} recorded {re_.get("label")} {re_.get("reasons")}')
    else:
        mm.append('CLASSIFIER_STABILITY.json not present')
    record['mine'] = mine
    record['mismatches'] = mm
    checks.append({'id': f'V7.STABILITY.{rob_id}__{tag}',
                   'what': 'robustness labels of PRC_S2 vs its matched controls and target references, ablation comparisons, own '
                           'split identities, refit effectiveness (strict eight-seed validity) vs CLASSIFIER_STABILITY.json',
                   'inputs': V.inputs_sha([sp_path, V.RESULTS / 'READER_SPLITS.json']), 'result': {**mine, 'mismatches': mm},
                   'numerical_status': 'PASS' if not mm else ('NOT_RUN' if mm == ['CLASSIFIER_STABILITY.json not present'] else 'FAIL'),
                   'procedural_status': 'NOT_APPLICABLE'})
    V.write_record(f'V7_ROBUSTNESS__{tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], (c['result'].get('mismatches') or [])[:6])
    return 0


def selftest():
    reg = {c: 0.01 for c in G.REG}
    robc = {f'S{v}{r}k{k}': 0.012 for v in G.VERS for r in ('R3', 'R4') for k in range(4)}
    e = {'pair': {'cells': {**reg, **robc}, 'cell_se': {c: 0.002 for c in {**reg, **robc}}}}
    ok = {'confirmed_when_positive': robustness(e)['label'] == 'ROBUSTNESS_CONFIRMED'}
    e2 = {'pair': {'cells': dict(reg), 'cell_se': {c: 0.002 for c in reg}}}
    ok['not_run_without_robustness_cells'] = robustness(e2)['label'] == 'NOT_RUN'
    e3 = {'pair': {'cells': {**reg, **{c: -0.01 for c in robc}}, 'cell_se': {c: 0.002 for c in {**reg, **robc}}}}
    ok['not_confirmed_when_negative'] = robustness(e3)['label'] == 'ROBUSTNESS_NOT_CONFIRMED'
    u = np.repeat(np.arange(500), 2)
    s = splits(u)
    ok['split_hashes_distinct'] = len({s[r]['assignment_sha256'] for r in REPS}) == 4
    ok['overlaps_in_unit_interval'] = all(0 <= x < 1 for r in REPS for x in s[r]['score_overlap'].values())
    ok['all_ok'] = all(ok.values())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--rob', default='FRC1_ROB')
    ap.add_argument('--main', default='FRC1_MAIN')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag, a.rob, a.main)


if __name__ == '__main__':
    sys.exit(main())
