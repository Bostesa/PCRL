"""Decisive-phase rule applications of the PCRL new-release-mechanisms campaign (coordinator). Label-free: it reads only committed
claims outputs (claims/<id>/CLAIMS.json), saved reader grids (runs/GRID/<bank>/S<v>/GRID.json, BOOT.npz), saved evaluation records
and the committed plans, and applies the statistical reviewer's frozen decisive-phase rules unchanged:

  fb2p          RULING_FB_V3 2(a): FB2P runs at the highest FB2_SEX strength that is TASK_FEASIBLE in both versions AND whose
                registered-cell pair benefit against the same-strength FB2_MINF is AGG_POSITIVE on NRM1_MAIN; else NOT_RUN.
  f1            RULING_FB_R1_FINAL C6: F1 holds unless frontier_comparison gives DOMINATES_CONTROL_FRONTIER at BOTH S3 and S5 on
                every declared path (card v2 section 6; MINF in PLAN_MAIN, MINF and CTL in PLAN_FINAL), with the MINF-path reading
                beside it (ENG2-04); an INCOMPLETE or missing record is completed first, never read as F1 or as a pass.
  mean          C5: MEAN is fitted at the highest FB2_SEX strength that is TASK_FEASIBLE in both versions and DOMINATES on both
                declared paths (MINF and CTL); otherwise not fitted.
  f2            RULING_FB_V3 1: F2 holds iff benefit_screen (pair view, registered cells) of FB2_SEX_S<j> vs FB2_SEX_S<j>_MEAN is
                NOT AGG_POSITIVE; the aggregate, both version means, MEAN's G1 and the single-view differences are reported beside.
                An INCOMPLETE screen is reported NOT_READ_INCOMPLETE (completed first, as C6 treats INCOMPLETE records), not F2_HOLDS.
  perturbation  RULING_FB_V3 3 (descriptive only) with the reviewer's 16:16Z clarification (CLARIFICATION_V3_S3_1616.md):
                B0 = FB2_SEX vs FB2_MINF and B1 = FB2P_SEX vs FB2P_MINF, both read once per cell under the declared instance map
                in the shared bank (no Monte Carlo replicates, ruling 2(c)); delta = x1 - x0; se_delta = se_agg of the per-cell
                difference entry whose cell SE is the SD of the paired bootstrap-draw differences (all releases of a cell share the
                household-bootstrap weights); c1 = pooled paired total-NLL cost of FB2P_SEX vs FB2P_MINF. Applicability
                (NOT_APPLICABLE_ORIGINAL_NOT_POSITIVE) follows the registered NRM1_MAIN Monte Carlo reading of B0 (the one rule 2(a)
                reads); if the single-instance B0 is not AGG_POSITIVE while that one is, the flag B0_SINGLE_INSTANCE_NOT_RESOLVED is
                attached (it never changes the label).

    python3 -m nrm.decisive fb2p --claims <CLAIMS.json> --plan <plan> --bank NRM1_MAIN --out <json>
    python3 -m nrm.decisive f1 --claims <CLAIMS.json> --out <json>
    python3 -m nrm.decisive mean --claims <CLAIMS.json> --out <json>
    python3 -m nrm.decisive f2 --claims <CLAIMS.json> --plan <plan> --bank <bank> --strength S5 --out <json>
    python3 -m nrm.decisive perturbation --plan <plan> --bank <bank> --strength S5 --main-claims <CLAIMS.json> --main-plan <plan>
                                         --out <json>
    python3 -m nrm.decisive selftest
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import claims as CL
from . import common as C
from . import rules as R

STRENGTHS = ('S5', 'S3')                                      # highest strength first
DOM = 'DOMINATES_CONTROL_FRONTIER'
LIMITATION = ("the frozen base models are not refitted; the perturbation covers the channel's fitted parameters and the release "
              "key only")


def _read(p):
    return json.loads(Path(p).read_text())


def _src(p):
    p = Path(p)
    return {'path': str(p.resolve().relative_to(C.ROOT.resolve())), 'sha256': C.sha(p)}


def _screen(entry):
    b = R.benefit_screen((entry or {}).get('pair'))
    a = b['aggregate']
    return {'label': b['label'], 'mean': a.get('mean'), 'se_agg': a.get('se_agg'),
            'per_version_means': {v: a['per_version'][v]['mean'] for v in R.VERSIONS} if a.get('complete') else None,
            'flags': b.get('flags', [])}


def _recipient(entry):
    out = {}
    for view in R.SINGLE_VIEWS:
        r = R.recipient_screen((entry or {}).get(view))
        a = r['aggregate']
        out[view] = {'label': r['label'], 'gate_pass': r['gate_pass'],
                     'per_version_means': {v: a['per_version'][v]['mean'] for v in R.VERSIONS} if a.get('complete') else None}
    return out


def _g1(claims, member):
    rec = claims['records'].get(member)
    return rec['summary']['G1'] if rec else 'NOT_IN_CLAIMS'


def _reps(plan):
    return {m: c.get('replicates', []) for m, c in plan['configs'].items()}


def fb2p(claims_path, plan_path, bank):
    claims, plan = _read(claims_path), _read(plan_path)
    if claims['bank'] != bank:
        raise ValueError(f"claims were assembled on {claims['bank']}, not {bank}")
    grid = CL.load_grid(bank)
    per = {}
    for j in STRENGTHS:
        sex, minf = f'FB2_SEX_{j}', f'FB2_MINF_{j}'
        scr = _screen(CL.grid_entry(grid, sex, minf, _reps(plan)))
        g1 = _g1(claims, sex)
        per[j] = {'candidate': sex, 'comparator': minf, 'G1': g1, 'benefit_vs_same_strength_minf': scr,
                  'qualifies': g1 == 'TASK_FEASIBLE' and scr['label'] == 'AGG_POSITIVE'}
    chosen = next((j for j in STRENGTHS if per[j]['qualifies']), None)
    return {'rule': 'RULING_FB_V3 2(a)', 'bank': bank, 'sources': {'claims': _src(claims_path), 'plan': _src(plan_path)},
            'per_strength': per, 'chosen_strength': chosen,
            'decision': (f'GO: fit FB2P_SEX_{chosen} and FB2P_MINF_{chosen}' if chosen else
                         'NOT_RUN: no strength qualifies; the 4 FB2P configurations are not spent')}


def _f1_verdict(labels):
    if not labels or any(x in ('INCOMPLETE', 'MISSING', None) for x in labels):
        return 'NOT_READ_INCOMPLETE'
    return 'F1_DOES_NOT_HOLD' if all(x == DOM for x in labels) else 'F1_HOLDS'


def f1(claims_path):
    """C6 against every path declared in the plan (card v2 section 6: 'against the declared paths'; ENG2-04), with the MINF-path
    reading of PLAN_MAIN beside it; an INCOMPLETE or missing path record is never read."""
    claims = _read(claims_path)
    paths = {j: {p: (f or {}).get('label') for p, f in (claims['records'][f'FB2_SEX_{j}'].get('frontier_per_path') or {}).items()}
             for j in ('S3', 'S5')}
    minf = {j: paths[j].get('MINF', 'MISSING') for j in paths}
    return {'rule': 'RULING_FB_R1_FINAL C6 (F1 holds unless DOMINATES_CONTROL_FRONTIER at both S3 and S5 on every declared path)',
            'sources': {'claims': _src(claims_path)}, 'path_labels': paths, 'minf_path_labels': minf,
            'F1_minf_path_reading': _f1_verdict(list(minf.values())),
            'F1': _f1_verdict([x for j in paths for x in paths[j].values()] + [minf[j] for j in minf if minf[j] == 'MISSING'])}


def mean(claims_path):
    claims = _read(claims_path)
    per = {}
    for j in STRENGTHS:
        rec = claims['records'][f'FB2_SEX_{j}']
        paths = {p: (f or {}).get('label') for p, f in (rec.get('frontier_per_path') or {}).items()}
        missing = [p for p in ('MINF', 'CTL') if p not in paths]
        incomplete = [p for p, x in paths.items() if x == 'INCOMPLETE']
        per[j] = {'G1': rec['summary']['G1'], 'paths': paths, 'missing_paths': missing, 'incomplete_paths': incomplete,
                  'qualifies': not missing and not incomplete and rec['summary']['G1'] == 'TASK_FEASIBLE'
                  and all(paths[p] == DOM for p in ('MINF', 'CTL'))}
    if any(per[j]['missing_paths'] or per[j]['incomplete_paths'] for j in STRENGTHS):
        decision, chosen = 'NOT_READ: a declared path is missing or INCOMPLETE (complete it first)', None
    else:
        chosen = next((j for j in STRENGTHS if per[j]['qualifies']), None)
        decision = (f'GO: fit FB2_SEX_{chosen}_MEAN (both versions)' if chosen else
                    'NOT_FITTED: no strength is TASK_FEASIBLE and DOMINATES both declared paths')
    return {'rule': 'RULING_FB_R1_FINAL C5', 'sources': {'claims': _src(claims_path)}, 'per_strength': per,
            'chosen_strength': chosen, 'decision': decision}


def f2(claims_path, plan_path, bank, j):
    claims, plan = _read(claims_path), _read(plan_path)
    if claims['bank'] != bank:
        raise ValueError(f"claims were assembled on {claims['bank']}, not {bank}")
    sex, mn = f'FB2_SEX_{j}', f'FB2_SEX_{j}_MEAN'
    e = CL.grid_entry(CL.load_grid(bank), sex, mn, _reps(plan))
    scr = _screen(e)
    verdict = ('NOT_READ_INCOMPLETE' if scr['label'] == 'INCOMPLETE' else
               'F2_DOES_NOT_HOLD' if scr['label'] == 'AGG_POSITIVE' else 'F2_HOLDS')
    return {'rule': 'RULING_FB_V3 1 (F2)', 'bank': bank, 'sources': {'claims': _src(claims_path), 'plan': _src(plan_path)},
            'candidate': sex, 'comparator': mn, 'benefit_vs_own_mean': scr, 'single_view_differences': _recipient(e),
            'MEAN_G1': _g1(claims, mn), 'MEAN_G2': 'NO_MATCHED_CONTROLS (not selectable)', 'F2': verdict,
            'meaning': 'F2_DOES_NOT_HOLD = the randomized release is resolved as less recoverable than its own mean vector'}


def _diff_entry(bank, e0, e1, pairs):
    """Per registered cell: b1 - b0 with SE = SD of the paired bootstrap-draw differences (shared weights per cell)."""
    (s0, m0), (s1, m1) = pairs
    boots = {v: np.load(C.RUNS / 'GRID' / bank / f'S{v}' / 'BOOT.npz') for v in R.VERSIONS}
    cells, ses = {}, {}
    for cell in R.REGISTERED:
        bz = boots[R.parse_cell(cell)[0]]

        def ben(cand, comp):
            return bz[f'{cell}__{comp}__pair'] - bz[f'{cell}__{cand}__pair']
        cells[cell] = e1['pair']['cells'][cell] - e0['pair']['cells'][cell]
        ses[cell] = float(np.std(ben(s1, m1) - ben(s0, m0), ddof=1))
    return {'cells': cells, 'cell_se': ses}


def perturbation_label(b0_registered, b1, agg_delta, c1):
    """RULING_FB_V3 3 labels from the registered (Monte Carlo) B0 screen, the perturbed screen, the single-instance difference
    aggregate and the paired total-NLL cost (value, se)."""
    if b0_registered['label'] != 'AGG_POSITIVE':
        return 'NOT_APPLICABLE_ORIGINAL_NOT_POSITIVE', []
    if c1 is None or b1['label'] == 'INCOMPLETE':                # an input is missing: never read (ENG2-05)
        return 'INCOMPLETE', ['paired total-NLL cost of FB2P_SEX vs FB2P_MINF missing' if c1 is None else 'B1 incomplete']
    fails = []
    if b1['label'] != 'AGG_POSITIVE':
        fails.append('perturbed benefit not resolved')
    d, s = agg_delta['mean'], agg_delta['se_agg']
    if abs(d) > 2 * s:
        fails.append(f"benefit changed by more than 2*se_delta ({'up' if d > 0 else 'down'})")
    if c1['value'] - 2 * c1['se'] > 0:
        fails.append('SEX now spends more than MINF')
    return ('PERTURBATION_CONSISTENT' if not fails else 'PERTURBATION_NOT_CONSISTENT'), fails


def perturbation(plan_path, bank, j, main_claims, main_plan):
    sex, minf, psex, pminf = f'FB2_SEX_{j}', f'FB2_MINF_{j}', f'FB2P_SEX_{j}', f'FB2P_MINF_{j}'
    if not all((C.RUNS / 'FB' / x / f'V{v}' / 'RELEASE.json').exists() for x in (psex, pminf) for v in C.VERSIONS):
        return {'rule': 'RULING_FB_V3 3', 'label': 'NOT_RUN', 'reason': 'FB2P not fitted', 'limitation': LIMITATION}
    b0_reg = _screen(CL.grid_entry(CL.load_grid(_read(main_claims)['bank']), sex, minf, _reps(_read(main_plan))))
    grid = CL.load_grid(bank)
    e0, e1 = CL.grid_entry(grid, sex, minf, {}), CL.grid_entry(grid, psex, pminf, {})
    b0, b1 = _screen(e0), _screen(e1)
    dent = _diff_entry(bank, e0, e1, ((sex, minf), (psex, pminf)))
    agg = R.aggregate(dent)
    _, costs = CL.eval_records('FB', psex)
    c1, flags = R._primary({v: costs[v].get(f'FB/{pminf}') for v in C.VERSIONS}) \
        if all(costs[v].get(f'FB/{pminf}') for v in C.VERSIONS) else (None, [])
    label, fails = perturbation_label(b0_reg, b1, agg, c1)
    qual = ['B0_SINGLE_INSTANCE_NOT_RESOLVED'] if b0_reg['label'] == 'AGG_POSITIVE' and b0['label'] != 'AGG_POSITIVE' else []
    return {'rule': 'RULING_FB_V3 3 with CLARIFICATION_V3_S3_1616 (descriptive only; never changes a decision, tier, selection or '
                    'falsifier)', 'bank': bank,
            'sources': {'plan': _src(plan_path), 'main_claims': _src(main_claims), 'main_plan': _src(main_plan)}, 'strength': j,
            'B0_registered_monte_carlo': b0_reg, 'B0_single_instance': b0, 'B1': b1,
            'delta': {'value': agg['mean'], 'se_delta': agg['se_agg'], 'per_version_means': {v: agg['per_version'][v]['mean']
                                                                                            for v in R.VERSIONS}, **dent},
            'c1_total_nll_FB2P_SEX_vs_FB2P_MINF': c1, 'c1_flags': flags, 'label': label, 'failing_parts': fails, 'flags': qual,
            'single_view_differences': {'original': _recipient(e0), 'perturbed': _recipient(e1)},
            'reading': ('applicability from the registered NRM1_MAIN Monte Carlo B0; delta and se_delta from B0 and B1 read once per '
                        'cell (declared instance map R1 -> 0, R2 -> 1, k0 and k1 sharing) in the shared bank'),
            'limitation': LIMITATION}


def selftest():
    """Invented data only: rule logic and the difference-of-differences SE."""
    chk = {}
    pos = {'label': 'AGG_POSITIVE'}
    neg = {'label': 'AGG_POINT_ONLY'}
    agg = {'mean': 0.004, 'se_agg': 0.003}
    chk['consistent'] = perturbation_label(pos, pos, agg, {'value': -0.001, 'se': 0.001})[0] == 'PERTURBATION_CONSISTENT'
    chk['not_applicable'] = perturbation_label(neg, pos, agg, None)[0] == 'NOT_APPLICABLE_ORIGINAL_NOT_POSITIVE'
    chk['missing_cost_incomplete'] = perturbation_label(pos, pos, agg, None)[0] == 'INCOMPLETE'
    chk['f1_reading'] = (_f1_verdict([DOM, DOM, DOM, DOM]) == 'F1_DOES_NOT_HOLD' and _f1_verdict([DOM, 'INCOMPLETE']) ==
                         'NOT_READ_INCOMPLETE' and _f1_verdict([DOM, 'NOT_SEPARATED_FROM_TASK_SPENDING']) == 'F1_HOLDS')
    lab, f = perturbation_label(pos, neg, {'mean': -0.01, 'se_agg': 0.002}, {'value': 0.01, 'se': 0.001})
    chk['all_three_fail'] = lab == 'PERTURBATION_NOT_CONSISTENT' and len(f) == 3 and 'down' in f[1]
    rng = np.random.default_rng(0)
    w = rng.normal(size=400)
    a, b, c, d = 0.60 + 0.01 * w, 0.62 + 0.01 * w, 0.61 + 0.012 * w, 0.625 + 0.01 * w   # shared weights -> correlated draws
    dd = (d - c) - (b - a)
    chk['paired_sd_small'] = float(np.std(dd, ddof=1)) < 0.003
    ok = all(chk.values())
    print(json.dumps({'ok': ok, 'checks': chk}, indent=1))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('fb2p', 'f1', 'mean', 'f2', 'perturbation'):
        p = sub.add_parser(name); p.add_argument('--out', required=True)
        if name in ('fb2p', 'f1', 'mean', 'f2'):
            p.add_argument('--claims', required=True)
        if name in ('fb2p', 'f2', 'perturbation'):
            p.add_argument('--plan', required=True); p.add_argument('--bank', required=True)
        if name in ('f2', 'perturbation'):
            p.add_argument('--strength', required=True, choices=('S3', 'S5'))
        if name == 'perturbation':
            p.add_argument('--main-claims', required=True); p.add_argument('--main-plan', required=True)
    sub.add_parser('selftest')
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return selftest()
    rec = {'fb2p': lambda: fb2p(a.claims, a.plan, a.bank), 'f1': lambda: f1(a.claims), 'mean': lambda: mean(a.claims),
           'f2': lambda: f2(a.claims, a.plan, a.bank, a.strength),
           'perturbation': lambda: perturbation(a.plan, a.bank, a.strength, a.main_claims, a.main_plan)}[a.cmd]()
    rec = {'schema': 'NRM1-decisive-v1', 'utc': C.now(), 'rules_sha256': R.load_rules()['_sha256'], **rec}
    C.write_json_x(Path(a.out), rec)
    print(json.dumps(rec, indent=1, default=str)[:4000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
