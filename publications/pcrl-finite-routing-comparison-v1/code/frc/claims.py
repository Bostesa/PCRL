"""Claims assembly of the PCRL finite-routing comparison (`pcrl_finite_routing_comparison_v1`; coordinator; label-free). It reads only
saved records (evaluation records EVAL_MS*.json, the P1 selection records, the qualification decision, reader-grid outputs
GRID.json) and a committed plan, and applies the statistical reviewer's frozen rules (frc.rules, FRC1-rules-v1) unchanged.

    python3 -m frc.claims run --plan results/<campaign>/claims/PLAN_<id>.json --bank <bank_id> --out results/<campaign>/claims/<id>

PLAN (committed before the grid outputs it governs are read):
  {"plan_id", "prc_s2": member or null (the PRC configuration selected at S2, an alias resolved to its saved member),
   "members": {member: {"eval": "frc:FA/<cfg>" | "nrm:FA/<cfg>" | "ref:<name>", "kind": ...}},
   "paths": {"HQ": [...], "SHUF": [...], "ORD": ["TWIN", ...]}, "alias_of": {member: saved member},
   "every_nonzero_selection_alias_of_twin": bool, "f4": {"version": {v: {"diff", "se"}}},
   "nested_lookup": {member: {v: AUC_member - AUC_prc}}, "reader_budget": {reference: flag}}
Grid members carry the plan's member names (the twin is "TWIN"; references by their benchmark names).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from . import common as C
from . import rules as RU
from nrm import rules as NR

REF_SPEC = {'NEURAL_RAW': 'REF:NEURAL_RAW', 'H_LJ': 'REF:H_LJ', 'B2TWIN': 'REF:B2TWIN', 'RETIRED_B3CTL': 'REF:RETIRED_B3CTL',
            'SUCC': 'REF:SUCC', 'TWJ': 'REF:TWJ', 'W': 'W', 'TR100': 'REF:H3+TASK_REFIT', 'TT100': 'REF:H3+T_TREE'}


def load_grid(bank):
    """Merge the per-version grid runs of a bank: {'paired': {cand: {comp: {view: {'cells', 'cell_se'}}}}, 'cells': {...}}."""
    out = {'paired': {}, 'cells': {}, 'members': {}}
    for v in C.VERSIONS:
        g = json.loads((C.RUNS / 'GRID' / bank / f'S{v}' / 'GRID.json').read_text())
        out['members'][v] = g.get('members') or {}
        for cell, cres in (g.get('cells') or {}).items():
            out['cells'][cell] = cres
        for cand, comps in (g.get('paired') or {}).items():
            for comp, views in comps.items():
                for view, e in views.items():
                    d = out['paired'].setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                    d['cells'].update(e['cells'])
                    d['cell_se'].update(e['cell_se'])
    return out


def nolookup_benefit(grid, cand, comp):
    """Per registered cell: the non-table pair winner's SCORE AUC of comp minus that of cand (positive favours cand)."""
    cells = {}
    for cell in NR.REGISTERED:
        try:
            a = grid['cells'][cell][cand]['pair']['extras']['nolookup']['score_auc']
            b = grid['cells'][cell][comp]['pair']['extras']['nolookup']['score_auc']
        except KeyError:
            return None
        cells[cell] = float(b) - float(a)
    return {'cells': cells}


def _eval_path(spec):
    kind, rest = spec.split(':', 1)
    if kind == 'frc':
        fam, cfg = rest.split('/')
        return C.RUNS / fam / cfg / 'EVAL_MS.json'
    if kind == 'nrm':
        fam, cfg = rest.split('/')
        return C.RUNS / 'SOURCE' / fam / cfg / 'EVAL_MS.json'
    raise ValueError(spec)


def evals_of(spec):
    p = _eval_path(spec)
    if not p.is_file():
        return None
    r = json.loads(p.read_text())
    return {v: r['versions'][str(v)] for v in C.VERSIONS if not r['versions'][str(v)].get('missing')}


def costs_between(cand_spec, comp_spec):
    """MODEL_SELECT paired costs loss(cand) - loss(comp) per version from the candidate's evaluation record (its --compare entry)."""
    ev = evals_of(cand_spec)
    if ev is None:
        return None
    key = comp_spec.split(':', 1)[1] if comp_spec.startswith('frc:') else comp_spec.replace('nrm:', 'NRM:').replace('ref:', 'REF:')
    out = {}
    for v, e in ev.items():
        c = (e.get('paired_costs') or {}).get(key)
        if c is None:
            return None
        out[v] = c['costs']
    return out


def build(plan, grid):
    mem = plan['members']
    s2 = plan.get('prc_s2')
    rec = {'grid_valid': plan.get('grid_valid'), 'capacity_decision': json.loads(
        (C.RESULTS / 'decisions' / 'QUAL_DECISION.json').read_text())['capacity_decision']}
    if s2 is None:
        rec['prc_s2'] = {'status': 'NOT_FEASIBLE'}
        return rec
    ev = evals_of(mem[s2]['eval'])
    rec['prc_s2'] = {'status': 'FITTED' if ev else 'MISSING', 'config_id': s2,
                     'evals': {v: e['utility'] for v, e in (ev or {}).items()},
                     'every_nonzero_selection_alias_of_twin': bool(plan.get('every_nonzero_selection_alias_of_twin'))}
    others = [m for m in mem if m != s2]
    rec['grid'] = {m: grid['paired'].get(s2, {}).get(m) for m in others if grid['paired'].get(s2, {}).get(m) is not None}
    for ref in REF_SPEC:
        if ref in grid['paired'].get(s2, {}):
            rec['grid'][ref] = grid['paired'][s2][ref]
    rec['paths'] = plan['paths']
    rec['alias_of'] = plan.get('alias_of') or {}
    rec['costs_vs'] = {m: costs_between(mem[s2]['eval'], mem[m]['eval']) for m in others if mem[m]['eval'].split(':')[0] != 'ref'}
    rec['f4'] = plan.get('f4')
    rec['nolookup'] = {m: nolookup_benefit(grid, s2, m) for m in list(others) + [r for r in REF_SPEC if r not in others]}
    rec['nested_lookup'] = plan.get('nested_lookup') or {}
    rec['reader_budget'] = plan.get('reader_budget') or {}
    rec['joint'] = None
    return rec


def run(plan_path, bank, out):
    plan = json.loads(Path(plan_path).read_text())
    if not C.committed(Path(plan_path)):
        raise PermissionError('the claims plan must be committed before the grid outputs it governs are read')
    grid = load_grid(bank)
    rec = build(plan, grid)
    dec = RU.decisions(rec)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    C.write_json_x(out / 'DECISIONS.json', {'schema': 'FRC1-decisions-v1', 'utc': C.now(), 'plan': plan.get('plan_id'), 'bank': bank,
                                            'rules_sha256': C.sha(RU.RULES_PATH), 'decisions': dec,
                                            'inputs_summary': {'grid_entries': sorted(rec.get('grid') or {}),
                                                               'costs_vs': sorted(k for k, v in (rec.get('costs_vs') or {}).items()
                                                                                  if v is not None),
                                                               'nolookup': sorted(k for k, v in (rec.get('nolookup') or {}).items()
                                                                                  if v is not None)}})
    print(json.dumps({k: (v.get('label') if isinstance(v, dict) else v) for k, v in dec.items() if k != 'scope'}, indent=1))
    return dec


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--plan', required=True); r.add_argument('--bank', required=True)
    r.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    run(a.plan, a.bank, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
