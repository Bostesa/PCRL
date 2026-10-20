"""Frontier and matched-control tables of the PCRL new-release-mechanisms campaign (coordinator; label-free). Reads only the committed
final claims (claims/FINAL_V2/CLAIMS.json), the saved registered grids of its bank, the saved evaluation records and the reference
evaluation; applies nrm.rules screens unchanged; writes results/<campaign>/TABLES.md and TABLES.json.

    python3 -m nrm.tables

Development evidence on adaptively used MODEL_SELECT, for the declared readers only. Uncertainty: se_agg of nrm.rules.aggregate
(household bootstrap, reader-split spread and Monte Carlo in quadrature, floor 0.001).
"""
from __future__ import annotations
import json
import sys

from . import claims as CL
from . import common as C
from . import rules as R

CLAIMS = C.RESULTS / 'claims' / 'FINAL_V2' / 'CLAIMS.json'
PLAN = C.RESULTS / 'claims' / 'PLAN_FINAL.json'
FAMILY_ROWS = ('FB2_BASE', 'FB2_SEX_S3', 'FB2_SEX_S5', 'FB2_MINF_S3', 'FB2_MINF_S5', 'CTL_REM_0p50', 'CTL_REM_1p00', 'FB_TWIN',
               'FB_SEX_S2', 'FB_MINF_S2', 'PRC_TWIN')
REF_ROWS = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100', 'U0', 'Ucal')
REF_SPEC = {**CL.REF_SPEC, 'U0': 'REF:U0', 'Ucal': 'REF:Ucal'}
COMPARISONS = (('FB2_SEX_S3', 'FB2_BASE', 'twin'), ('FB2_SEX_S3', 'FB2_MINF_S3', 'same-strength SEX-blind alternative (G2)'),
               ('FB2_SEX_S3', 'CTL_REM_0p50', 'CTL path bracket'), ('FB2_SEX_S3', 'NEURAL_RAW', 'target reference'),
               ('FB2_SEX_S3', 'H_LJ', 'target reference'),
               ('FB2_SEX_S5', 'FB2_BASE', 'twin'), ('FB2_SEX_S5', 'FB2_MINF_S5', 'same-strength SEX-blind alternative (G2)'),
               ('FB2_SEX_S5', 'CTL_REM_1p00', 'CTL path bracket'), ('FB2_SEX_S5', 'NEURAL_RAW', 'target reference'),
               ('FB2_SEX_S5', 'H_LJ', 'target reference'),
               ('FB2_MINF_S5', 'FB2_BASE', 'control vs twin (descriptive)'), ('CTL_REM_0p50', 'FB2_BASE', 'control vs twin (descriptive)'),
               ('CTL_REM_1p00', 'FB2_BASE', 'control vs twin (descriptive)'))


def _f(x, n=4):
    return '—' if x is None else f'{x:+.{n}f}' if isinstance(x, float) and n == 4 else f'{x:.{n}f}'


def _auc(grid, m, view):
    """Registered-cell mean SCORE AUC of the base instance per cell (R1 -> 0, R2 -> 1)."""
    xs = [grid['cells'][c][m][view]['score_auc'] for c in R.REGISTERED if m in grid['cells'].get(c, {})]
    return sum(xs) / len(xs) if len(xs) == len(R.REGISTERED) else None


def _auc_pooled(grid, m, view):
    """The same with the Monte Carlo replicate members (+2, +4) averaged in per cell, as nrm.rules.mc_cell pools the benefits
    (ENG2-08); None for a deterministic member."""
    reps = [m, f'{m}__d2', f'{m}__d4']
    if not all(r in grid['cells'].get(R.REGISTERED[0], {}) for r in reps[1:]):
        return None
    xs = [sum(grid['cells'][c][r][view]['score_auc'] for r in reps) / len(reps) for c in R.REGISTERED]
    return sum(xs) / len(xs)


def _spend(u):
    v = {f"{c['task']}|{c['reference']}|{c['metric']}": float(c['value']) for c in u['clauses']}
    return v['income|U0|nll_excess'] + v['occupation|U0|nll_excess']


def build():
    claims = json.loads(CLAIMS.read_text())
    plan = json.loads(PLAN.read_text())
    grid = CL.load_grid(claims['bank'])
    reps = {m: c.get('replicates', []) for m, c in plan['configs'].items()}
    refs = json.loads((C.ROOT / plan['references_eval'][0]).read_text())['references']
    rows, utils = [], {}
    for m in FAMILY_ROWS + REF_ROWS:
        if m in REF_ROWS:
            u = {v: refs[REF_SPEC[m]][str(v)]['utility'] for v in C.VERSIONS}
            kind, g1, reserve = 'reference', None, None
        else:
            cfg = plan['configs'].get(m) or {'family': 'CTL' if m.startswith('CTL') else 'FA' if m.startswith('PRC') else 'FB',
                                             'config': m}
            base, _ = CL.eval_records(cfg['family'], cfg['config'])
            u = {v: base['versions'][str(v)]['utility'] for v in C.VERSIONS}
            rec = claims['records'].get(m)
            kind = cfg.get('kind')
            g1 = rec['summary']['G1'] if rec else 'not in the final plan (descriptive)'
            reserve = (rec.get('reserve') or {}).get('qualified') if rec else None
        utils[m] = u
        worst = {v: max([float(c['value']) / float(c['cap']) for c in u[v]['clauses']] +
                        [float(g['value']) / float(g['cap']) for g in u[v]['guards']]) for v in C.VERSIONS}
        rows.append({'member': m, 'kind': kind, 'G1': g1, 'reserve_qualified': reserve,
                     'worst_value_over_cap': worst, 'total_nll_excess_vs_U0': {v: _spend(u[v]) for v in C.VERSIONS},
                     'pair_auc': _auc(grid, m, 'pair'), 'income_auc': _auc(grid, m, 'income'),
                     'occupation_auc': _auc(grid, m, 'occupation'), 'pair_auc_mc_pooled': _auc_pooled(grid, m, 'pair')})
    comps = []
    for cand, comp, role in COMPARISONS:
        e = CL.grid_entry(grid, cand, comp, reps)
        b = R.benefit_screen(e['pair'])
        a = b['aggregate']
        rs = {v: R.recipient_screen(e[v])['label'] for v in R.SINGLE_VIEWS}
        comps.append({'candidate': cand, 'comparator': comp, 'role': role, 'benefit': a['mean'], 'se_agg': a['se_agg'],
                      'v0': a['per_version'][0]['mean'], 'v1': a['per_version'][1]['mean'], 'label': b['label'],
                      'recipient_income': rs['income'], 'recipient_occupation': rs['occupation']})
    frontier = []
    for cand in ('FB2_SEX_S3', 'FB2_SEX_S5'):
        for pid, f in (claims['records'][cand].get('frontier_per_path') or {}).items():
            frontier.append({'candidate': cand, 'path': pid, 'label': f.get('label'), 'reason': f.get('reason'),
                             'bracketing_control': f.get('bracketing_control'), 'benefit': f.get('benefit_mean'),
                             'se': f.get('benefit_se'),
                             'primary_cost_differences': {q: round(x['value'], 5)
                                                          for q, x in (f.get('primary_cost_differences') or {}).items()}})
    return {'schema': 'NRM1-tables-v1', 'utc': C.now(), 'claims': str(CLAIMS.relative_to(C.ROOT)), 'claims_sha256': C.sha(CLAIMS),
            'bank': claims['bank'], 'selection': claims['selection']['outcome'], 'rows': rows, 'comparisons': comps,
            'frontier': frontier, 'reserve_frontier': reserve_section(claims, grid, reps, refs, rows, utils)}


def reserve_section(claims, grid, reps, refs, rows, utils):
    """nrm.rules.reserve_frontier over the plan's members and the nine benchmark references (reserve_status, identical rule; the
    utility references U0 / Ucal and members outside the final plan are left out). A member whose claims record carries no reserve
    (the invalid-grid branch: PRC_TWIN) gets reserve_status computed from its own evaluation record (DOC2-13)."""
    pts = []
    for r in rows:
        m = r['member']
        if m in ('U0', 'Ucal'):
            continue
        if m in REF_ROWS:
            rs, kind = R.reserve_status({v: refs[REF_SPEC[m]][str(v)]['utility'] for v in C.VERSIONS}), 'reference'
        else:
            rec = claims['records'].get(m)
            if rec is None:
                continue
            rs, kind = rec.get('reserve') or {}, ('candidate' if rec['config'].get('kind') == 'candidate' else 'control')
            if not rs.get('min_ratio_by_version'):         # the invalid-grid branch records only {'qualified': False}
                rs = R.reserve_status(utils[m])
        mr = rs.get('min_ratio_by_version') or {}
        auc = r['pair_auc_mc_pooled'] if r.get('pair_auc_mc_pooled') is not None else r['pair_auc']   # as the paired screens
        pts.append({'id': m, 'kind': kind, 'reserve_qualified': rs.get('qualified') is True, 'pair_auc': auc,
                    'min_ratio': min(mr.values()) if mr else None})
    paired = {(p['id'], o['id']): CL.grid_entry(grid, p['id'], o['id'], reps) for p in pts if p['kind'] == 'candidate'
              for o in pts if o['id'] != p['id']}
    return R.reserve_frontier(pts, {k: e for k, e in paired.items() if e is not None})


def markdown(t):
    L = ['# Frontier and matched-control tables', '',
         f"Generated by `python3 -m nrm.tables` from `{t['claims']}` (bank {t['bank']}; selection **{t['selection']}**). Development "
         'evidence on adaptively used MODEL_SELECT records, for the declared readers only; lower recovery for these readers is not '
         'protection against every reader. AUC = registered-cell mean SCORE AUC for the recorded SEX column (8 cells: 2 versions × '
         '2 repetitions × 2 reader seeds). Spend = income + occupation NLL excess vs U0 (nats).', '',
         '## 1. Utility and recovery per release', '',
         'AUC columns: the base instance per cell (randomized releases read instance 0 in R1 cells and 1 in R2 cells, as registered). '
         'For randomized releases the last column pools the Monte Carlo replicate instances (+2, +4) per cell, as the claims and '
         'section 2 do, so its differences match the section-2 benefits.', '',
         '| Release | Kind | G1 | Worst clause / cap (V0 / V1) | Spend vs U0 (V0 / V1) | Pair AUC | Income AUC | Occupation AUC | '
         'Pair AUC, MC pooled |',
         '|---|---|---|---|---|---|---|---|---|']
    for r in t['rows']:
        w, s = r['worst_value_over_cap'], r['total_nll_excess_vs_U0']
        pooled = '—' if r.get('pair_auc_mc_pooled') is None else f"{r['pair_auc_mc_pooled']:.4f}"
        L.append(f"| {r['member']} | {r['kind'] or 'family member'} | {r['G1'] or '(reference)'} | {w[0]:.3f} / {w[1]:.3f} | "
                 f"{s[0]:+.4f} / {s[1]:+.4f} | {_f(r['pair_auc'], 4).lstrip('+')} | {_f(r['income_auc'], 4).lstrip('+')} | "
                 f"{_f(r['occupation_auc'], 4).lstrip('+')} | {pooled} |")
    L += ['', 'A worst clause / cap above 1.000 fails the original limits (24 clauses and 4 guards per version, read as the maximum of '
          'value / cap).', '', '## 2. Matched-control comparisons (pair benefit = comparator AUC − candidate AUC; positive favours '
          'the candidate)', '',
          '| Candidate | Comparator | Role | Benefit | se_agg | V0 | V1 | Screen | Income recipient | Occupation recipient |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for c in t['comparisons']:
        L.append(f"| {c['candidate']} | {c['comparator']} | {c['role']} | {c['benefit']:+.4f} | {c['se_agg']:.4f} | "
                 f"{c['v0']:+.4f} | {c['v1']:+.4f} | {c['label']} | {c['recipient_income']} | {c['recipient_occupation']} |")
    L += ['', 'AGG_POSITIVE needs benefit − 2·se_agg > 0 and both version means > 0; AGG_POSITIVE_POINT_ONLY means a positive point '
          'estimate that is not resolved.', '', '## 3. Frontier against the declared SEX-blind control paths (G3)', '',
          '| Candidate | Path | Label | Reason | Bracketing control | Benefit vs bracket | se | Paired total-NLL spend: candidate − '
          'path point |', '|---|---|---|---|---|---|---|---|']
    for f in t['frontier']:
        L.append(f"| {f['candidate']} | {f['path']} | {f['label']} | {f['reason'] or ''} | {f['bracketing_control']} | "
                 f"{f['benefit']:+.4f} | {f['se']:.4f} | {json.dumps(f['primary_cost_differences'])} |")
    L += ['', 'The bracketing control is the cheapest path point that spends at least as much as the candidate (pooled paired total '
          'NLL). DOMINATES_CONTROL_FRONTIER needs a resolved benefit there and no resolved extra secondary spending.', '',
          '## 4. Reserve-qualified frontier (nrm.rules.reserve_frontier; the two-SE reserve is a separate heuristic)', '',
          'Pair AUC here is Monte Carlo pooled for randomized releases (as the paired screens), the base instance otherwise.', '',
          '| Point | Kind | Reserve qualified | Min reserve ratio | Pair AUC | On the reserve Pareto frontier | Label |',
          '|---|---|---|---|---|---|---|']
    rf = t['reserve_frontier']
    for p in sorted(rf['points'], key=lambda p: (not p['reserve_qualified'], p['pair_auc'] if p['pair_auc'] is not None else 9)):
        mr = '—' if p['min_ratio'] is None else f"{p['min_ratio']:.2f}"
        L.append(f"| {p['id']} | {p['kind']} | {p['reserve_qualified']} | {mr} | {p['pair_auc']:.4f} | "
                 f"{p['on_reserve_pareto_frontier']} | {p.get('label', '')} |")
    for p in rf['points']:
        if p.get('paired_vs_qualified'):
            L += ['', f"{p['id']} against every other reserve-qualified point: " +
                  ', '.join(f'{o} {lab}' for o, lab in p['paired_vs_qualified'].items()) + '.']
    L += ['', 'References failing the reserve (still in the original benchmark): ' + ', '.join(rf['references_failing_reserve']) +
          '. ' + rf['note'][:1].upper() + rf['note'][1:] + '.', '']
    return '\n'.join(L)


def main():
    t = build()
    C.write_json(C.RESULTS / 'TABLES.json', t)
    (C.RESULTS / 'TABLES.md').write_text(markdown(t))
    print(json.dumps({'rows': len(t['rows']), 'comparisons': len(t['comparisons']), 'frontier': len(t['frontier'])}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
