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


def distinct_inputs(bank):
    """Label-free distinct released-vector counts per (member, version) and view, from the bank's own members (A-SC1)."""
    import numpy as np
    from . import devreaders as DR
    spec = json.loads((C.RESULTS / 'grid_specs' / f'{bank}.json').read_text())
    out = {}
    for name, m in spec['members'].items():
        for v in C.VERSIONS:
            rel, _ = DR.load_member(m, v)
            inc, occ = rel['income'], rel['occupation']
            out[f'{name}|V{v}'] = {'income': int(np.unique(inc, axis=0).shape[0]), 'occupation': int(np.unique(occ, axis=0).shape[0]),
                                   'pair': int(np.unique(np.column_stack((inc, occ)), axis=0).shape[0])}
    return out


def validity(grid, minimum_set, distinct=None, ks=(0, 1)):
    """frc.rules.grid_validity (the NRM1 A-SC1 check with this grid's seed rule) from the extras' score fingerprints: k0 vs k1 per
    recipe, view, member, version and repetition (as nrm.claims.validity)."""
    man = json.loads((C.RESULTS / 'READER_SPLITS.json').read_text())
    seeds = {f'S{v}k{k}': man['reader_seeds'][f'V{v}k{k}'] for v in C.VERSIONS for k in ks}
    checks = []
    for cell, cres in grid['cells'].items():
        if '@' in cell or not cell.endswith('k0'):
            continue
        v, rep = int(cell[1]), cell[2:4]
        other = cell[:-2] + 'k1'
        for member, ent in cres.items():
            for view, e in ent.items():
                f0 = (e.get('extras') or {}).get('score_sha256') or {}
                f1 = (((grid['cells'].get(other) or {}).get(member) or {}).get(view, {}).get('extras') or {}).get('score_sha256') or {}
                for recipe in NR.SEED_DEPENDENT + NR.SEED_INVARIANT:
                    if recipe in f0 and recipe in f1:
                        checks.append({'recipe': recipe, 'view': view, 'wire': member, 'v': v, 'R': rep,
                                       'equal_k0_k1': f0[recipe] == f1[recipe]})
    return RU.grid_validity(seeds, checks, minimum_set, distinct or {}), checks


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
    gv, checks = validity(grid, plan.get('minimum_set') or [], distinct_inputs(bank))
    plan = {**plan, 'grid_valid': bool(gv['effective'])}
    rec = build(plan, grid)
    dec = RU.decisions(rec)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    C.write_json_x(out / 'DECISIONS.json', {'schema': 'FRC1-decisions-v1', 'utc': C.now(), 'plan': plan.get('plan_id'), 'bank': bank,
                                            'rules_sha256': C.sha(RU.RULES_PATH), 'decisions': dec, 'grid_validity': gv,
                                            'grid_validity_checks': len(checks),
                                            'inputs_summary': {'grid_entries': sorted(rec.get('grid') or {}),
                                                               'costs_vs': sorted(k for k, v in (rec.get('costs_vs') or {}).items()
                                                                                  if v is not None),
                                                               'nolookup': sorted(k for k, v in (rec.get('nolookup') or {}).items()
                                                                                  if v is not None)}})
    print(json.dumps({k: (v.get('label') if isinstance(v, dict) else v) for k, v in dec.items() if k != 'scope'}, indent=1))
    return dec


def split_records():
    """Label-free split identities of the FRC1 repetitions: assignment hash (FIT 0 / VALIDATION 1 / SCORE 2 per admitted MODEL_SELECT
    row) and the share of SCORE households each repetition shares with every other (frc.rules.refit_effectiveness input)."""
    import hashlib
    import numpy as np
    from . import devreaders as DR
    unit = C.features('MODEL_SELECT')['unit']
    roles = {r: DR.roles_of(unit, r) for r in DR.ALL_REPS}
    out = {}
    for r, m in roles.items():
        a = np.where(m['FIT'], 0, np.where(m['VALIDATION'], 1, 2)).astype(np.int8)
        sc = set(np.unique(unit[m['SCORE']]).tolist())
        ov = {}
        for q, mq in roles.items():
            if q != r:
                sq = set(np.unique(unit[mq['SCORE']]).tolist())
                ov[q] = len(sc & sq) / len(sc)
        out[r] = {'assignment_sha256': hashlib.sha256(a.tobytes()).hexdigest(), 'score_overlap': ov,
                  'score_households': len(sc)}
    return out


def stability(main_bank, rob_bank, out):
    """Classifier-stability results: nrm.rules.reader_stability of PRC_S2 against its matched controls and target references on the
    registered grid; nrm.rules.robustness on the robustness cells of the refit bank; frc.rules.refit_effectiveness (split identities
    and reader-seed checks on all eight seeds); the ablation comparisons."""
    gm = load_grid(main_bank)
    rec = {'schema': 'FRC1-classifier-stability-v1', 'utc': C.now(), 'main_bank': main_bank, 'rob_bank': rob_bank,
           'reader_stability': {}, 'robustness': {}, 'ablation': {}}
    wires = {m: {v: (gm['members'][v].get(m) or {}).get('wire_sha256') for v in C.VERSIONS} for m in gm['members'][0]}
    for comp in ('TWIN', 'HQ_S2', 'SHUF_S2', 'UNT_BR', 'ORD_S2', 'NEURAL_RAW', 'H_LJ'):
        rec['reader_stability'][comp] = NR.reader_stability(gm['cells'], 'PRC_S2', comp, wires=wires)
    gr = load_grid(rob_bank)
    for comp in ('TWIN', 'HQ_S2', 'SHUF_S2', 'UNT_BR', 'ORD_S2', 'NEURAL_RAW', 'H_LJ'):
        e = gr['paired'].get('PRC_S2', {}).get(comp)
        rec['robustness'][comp] = NR.robustness(e) if e else {'label': 'NOT_RUN'}
    for abl in ('ABL_LOC', 'ABL_PAIR'):
        for comp in ('PRC_S2', 'TWIN'):
            e = gr['paired'].get(abl, {}).get(comp)
            if e:
                rec['ablation'][f'{abl}_vs_{comp}'] = {'registered': NR.comparison(e), 'robustness': NR.robustness(e)}
    man = json.loads((C.RESULTS / 'READER_SPLITS.json').read_text())
    seeds = {f'S{v}k{k}': man['reader_seeds'][f'V{v}k{k}'] for v in C.VERSIONS for k in range(4)}
    checks = []
    for cell, cres in gr['cells'].items():
        if '@' in cell:
            continue
        v, rep, k = NR.parse_cell(cell)
        if k not in (0, 2):
            continue
        other = cell[:-1] + str(k + 1)
        for member, ent in cres.items():
            for view, e in ent.items():
                f0 = (e.get('extras') or {}).get('score_sha256') or {}
                f1 = (((gr['cells'].get(other) or {}).get(member) or {}).get(view, {}).get('extras') or {}).get('score_sha256') or {}
                for recipe in NR.SEED_DEPENDENT + NR.SEED_INVARIANT:
                    if recipe in f0 and recipe in f1:
                        checks.append({'recipe': recipe, 'view': view, 'wire': member, 'v': v, 'R': rep, 'k_pair': [k, k + 1],
                                       'equal_k0_k1': f0[recipe] == f1[recipe]})
    splits = split_records()
    rec['splits'] = splits
    rec['refit_effectiveness'] = RU.refit_effectiveness(splits, seeds, checks, ['PRC_S2', 'TWIN', 'HQ_S2', 'UNT_BR', 'ORD_S2'],
                                                        distinct_inputs(rob_bank))
    rec['refit_checks'] = len(checks)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    C.write_json_x(out / 'CLASSIFIER_STABILITY.json', rec)
    print(json.dumps({'robustness': {k: v.get('label') for k, v in rec['robustness'].items()},
                      'refit': rec['refit_effectiveness']['label'], 'ablation': {k: (v['registered']['benefit']['label'],
                                                                                     v['robustness']['label'])
                                                                                  for k, v in rec['ablation'].items()}}, indent=1))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--plan', required=True); r.add_argument('--bank', required=True)
    r.add_argument('--out', required=True)
    s = sub.add_parser('stability'); s.add_argument('--main', required=True); s.add_argument('--rob', required=True)
    s.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        run(a.plan, a.bank, a.out)
    else:
        stability(a.main, a.rob, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
