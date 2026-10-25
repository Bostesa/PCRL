"""Claims assembly of a DR reading (campaign `pcrl_direct_recovery_frontier_v1`, DRF1; coordinator; label-free). It reads only saved
records (the committed reading plan, the DR fit and optimistic records, the MODEL_SELECT evaluation records EVAL_MS.json, the
reader-grid outputs GRID.json and the access log for the registration status) and applies the statistical reviewer's frozen rules
(`drf.rules`, DRF1-rules-v1) unchanged.

    python3 -m drf.drclaims run --plan results/<campaign>/claims/PLAN_<id>.json --bank <bank_id> --out results/<campaign>/claims/<id>
    python3 -m drf.drclaims d2r --plan PLAN.json --main <bank> --rob <bank> --out DIR
    python3 -m drf.drclaims optimistic --plan PLAN.json         (the MODEL_FIT optimistic diagnostic alone; reported, never gating)

PLAN (committed before the reading's first MODEL_SELECT utility or reader computation; RULES.md 11):
  {"plan_id", "family": "DR", "card", "revision", "revision_note" (revision >= 1), "written_utc",
   "cand": "DR_AUC", "twin": {"id": "TWB_C64"}, "obj_mi": {"config_id": "DR_MI"}, "start": {"id": "SUCC"},
   "members": {id: eval spec} (this reading's DR configurations 'DR/<config>' or role members 'ROLE:<name>'; the twin's own
              utility record 'SRCFRC:FA/TWB_C64'), "compare_specs": {id: the comparator spec used in the paired costs},
   "paths": {"CTRL": [ids]}, "alias_of": {id: 'TWIN' | earlier id}, "selection_data": {"CTRL": "MODEL_FIT_NESTED"},
   "controls_not_run": {"CTRL": false, "OBJ": false}, "fineness_inputs": grid_fineness input, "budgets", "fit_records": [...],
   "optimistic_records": [...], "alias_records": {...}, "distinct_inputs_model_select": {...}, "statement": ...}
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from . import claims as CL
from . import common as C
from . import rules as RU

REFS = CL.REFS
LAW_TAG = 'empirical_true_sex'


def evals_of(spec):
    if spec.startswith('SRCFRC:'):
        fam, cfg = spec[7:].split('/')
        p = C.RUNS / 'SOURCE_FRC' / fam / cfg / 'EVAL_MS.json'
        if not p.is_file():
            return None
        r = json.loads(p.read_text())
        return {v: r['versions'][str(v)] for v in C.VERSIONS if not r['versions'][str(v)].get('missing')}
    return CL.evals_of(spec)


def _utility(spec):
    ev = evals_of(spec)
    return {v: e['utility'] for v, e in (ev or {}).items()}


def optimistic_input(plan):
    """useful_optimistic() input from the committed fit and optimistic records (MODEL_FIT only; RULES.md 5.1)."""
    opt = {'OPT_S': {}, 'OPT_L': {}, 'CTRL': {}}
    for v in C.VERSIONS:
        fp = C.RESULTS / 'dr' / f'FIT_{plan.get("optimistic_ctrl_fit_tag", "")}V{v}.json'
        op = C.RESULTS / 'dr' / f'OPTIMISTIC_V{v}.json'
        if not fp.is_file() or not op.is_file():
            continue
        fit, o = json.loads(fp.read_text()), json.loads(op.read_text())
        ins_path = []
        for p in fit['path_full']:
            e = p['insample']['empirical']
            ob = e['observable_merged_wires']
            ins_path.append({'id': f'PATH_STEP_{p["step"]}', 'law': LAW_TAG, 'spend': e['spend']['inc_nll'] + e['spend']['occ_nll'],
                             'auc': {'pair': ob['auc_pair'], 'income': ob['auc_inc'], 'occupation': ob['auc_occ']}})
        nest_path = []
        npth = fit['nested']['path']
        for step in sorted(npth['spend_over_identity'], key=int):
            ex = npth['heldout_exact_auc_mean'][step]
            nest_path.append({'id': f'PATH_STEP_{step}', 'law': LAW_TAG, 'spend': npth['spend_over_identity'][step]['total_nll'][0],
                              'auc': {'pair': ex['auc_pair'], 'income': ex['auc_inc'], 'occupation': ex['auc_occ']}})
        opt['CTRL'][v] = {'in_sample': ins_path, 'nested': nest_path}
        for kind in ('OPT_S', 'OPT_L'):
            ent = o['entries'][kind]['insample']['empirical']
            ob = ent['observable_merged_wires']
            ne = o['nested']['entries'][kind]
            ex = ne['heldout_exact_auc_mean']
            opt[kind][v] = {
                'in_sample': {'law': LAW_TAG, 'spend': ent['spend']['inc_nll'] + ent['spend']['occ_nll'],
                              'auc': {'pair': ob['auc_pair'], 'income': ob['auc_inc'], 'occupation': ob['auc_occ']},
                              'within_budget': bool(ent['feasible'])},
                'nested': {'law': LAW_TAG, 'spend': ne['spend_over_identity']['total_nll'][0],
                           'auc': {'pair': ex['auc_pair'], 'income': ex['auc_inc'], 'occupation': ex['auc_occ']},
                           'within_budget': bool(ent['feasible'])}}
    return opt


def build(plan, grid, bank):
    """The DRF1 rules' record of one DR reading (drf.rules module docstring; RULES.md 11)."""
    mem, cspec = plan['members'], plan['compare_specs']
    cand = plan['cand']
    twin_id, obj_id = plan['twin']['id'], plan['obj_mi']['config_id']
    ok, reg = CL.registration_status(plan['card'])
    rec = {'family': 'DR', 'card': plan['card'], 'revision': int(plan.get('revision', 0)),
           'card_committed_before_any_model_select_outcome': ok, 'registration_status_detail': reg}
    if plan.get('revision_note'):
        rec['revision_note'] = plan['revision_note']
    rec['cand'] = {'status': 'FITTED', 'config_id': cand, 'evals': _utility(mem[cand]),
                   'alias_of_twin': bool((plan.get('alias_records') or {}).get('cand_alias_of_twin', False))}
    rec['reader_stage'] = plan.get('reader_stage', 'RUN')
    rec['start'] = {'id': 'SUCC', 'evals': _utility('REF:SUCC')}
    rec['twin'] = {'id': twin_id, 'evals': _utility(mem[twin_id])}
    rec['obj_mi'] = {'config_id': obj_id, 'evals': _utility(mem[obj_id])}
    rec['paths'] = {'CTRL': list(plan['paths'].get('CTRL') or [])}
    rec['selection_data'] = plan.get('selection_data') or {}
    rec['controls_not_run'] = plan.get('controls_not_run') or {'CTRL': False, 'OBJ': False}
    rec['alias_of'] = plan.get('alias_of') or {}
    rec['fineness'] = RU.grid_fineness(plan.get('fineness_inputs'))
    pc = grid['paired'].get(cand, {})
    ids = [twin_id, obj_id] + rec['paths']['CTRL']
    rec['grid'] = {k: pc[k] for k in dict.fromkeys(['SUCC'] + ids + list(REFS)) if k in pc}
    rec['costs_vs'] = {k: CL.costs_between(mem[cand], cspec[k]) for k in dict.fromkeys(ids + ['SUCC'])}
    rec['nolookup'] = {k: CL.extra_benefit(grid, cand, k) for k in dict.fromkeys(ids + ['SUCC'] + list(REFS))}
    rec['enhanced'] = {k: CL.extra_benefit(grid, cand, k, 'enhanced') for k in dict.fromkeys(ids + ['SUCC'])}
    rec['reader_budget'] = CL.reader_budget_flags(bank, cand)
    rec['optimistic'] = optimistic_input(plan)
    rec['joint'] = None
    rec['novelty'] = None
    return rec


def run(plan_path, bank, out):
    plan = json.loads(Path(plan_path).read_text())
    if not C.committed(Path(plan_path)):
        raise PermissionError('the reading plan must be committed before the grid outputs it governs are read')
    grid = CL.load_grid(bank)
    seeds, boot = CL.seed_records()
    minimum = [plan['cand'], plan['twin']['id'], plan['obj_mi']['config_id'], 'SUCC'] + list(plan['paths'].get('CTRL') or [])
    distinct = CL.distinct_inputs(bank)
    checks = CL.seed_checks(grid)
    gv = RU.grid_validity(seeds, checks, minimum, distinct, boot_seeds=boot)
    rec = build(plan, grid, bank)
    rec['grid_valid'] = gv['effective']
    dec = RU.decisions(rec)
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    C.write_json_x(outd / 'DECISIONS.json', {'schema': 'DRF1-decisions-v1', 'utc': C.now(), 'plan': plan.get('plan_id'), 'bank': bank,
                                             'plan_sha256': C.sha(Path(plan_path)), 'rules_sha256': C.sha(RU.RULES_PATH),
                                             'decisions': dec, 'grid_validity': gv, 'grid_validity_checks': len(checks),
                                             'seeds': seeds, 'boot_seeds': boot, 'distinct_inputs': distinct,
                                             'registration_status': rec.get('registration_status_detail'),
                                             'reader_budget': rec.get('reader_budget'), 'optimistic_input': rec.get('optimistic'),
                                             'inputs_summary': {'grid_entries': sorted(rec.get('grid') or {}),
                                                                'costs_vs': sorted(k for k, v in (rec.get('costs_vs') or {}).items()
                                                                                   if v is not None),
                                                                'nolookup': sorted(k for k, v in (rec.get('nolookup') or {}).items()
                                                                                   if v is not None)}})
    print(json.dumps({k: (v.get('label') if isinstance(v, dict) else v) for k, v in dec.items() if k != 'scope'}, indent=1,
                     default=str))
    return dec


def run_stopped(plan_path, out):
    """Decision record of a reading whose candidate fails the original usefulness conditions on MODEL_SELECT, with the competitive
    reader stage stopped (RULES.md 2: reader_stage STOPPED_USEFULNESS_FAILURE, allowed only when DR fails G1; the fitted release is
    preserved). No grid is read; the optimistic diagnostic is still computed for the failure stage."""
    plan = json.loads(Path(plan_path).read_text())
    if not C.committed(Path(plan_path)):
        raise PermissionError('the reading plan must be committed before its decision record is written')
    from nrm import rules as NR
    mem = plan['members']
    ev = _utility(mem[plan['cand']])
    if NR.point_eligibility(ev)['label'] == 'POINT_ELIGIBLE':
        raise PermissionError('the candidate passes G1 on MODEL_SELECT: the reader stage must run (RULES.md 2)')
    twin_id, obj_id = plan['twin']['id'], plan['obj_mi']['config_id']
    ok, reg = CL.registration_status(plan['card'])
    rec = {'family': 'DR', 'card': plan['card'], 'revision': int(plan.get('revision', 0)),
           'card_committed_before_any_model_select_outcome': ok, 'registration_status_detail': reg,
           'cand': {'status': 'FITTED', 'config_id': plan['cand'], 'evals': ev,
                    'alias_of_twin': bool((plan.get('alias_records') or {}).get('cand_alias_of_twin', False))},
           'reader_stage': 'STOPPED_USEFULNESS_FAILURE', 'start': {'id': 'SUCC', 'evals': _utility('REF:SUCC')},
           'twin': {'id': twin_id, 'evals': _utility(mem[twin_id])}, 'obj_mi': {'config_id': obj_id, 'evals': _utility(mem[obj_id])},
           'paths': {'CTRL': list(plan['paths'].get('CTRL') or [])}, 'selection_data': plan.get('selection_data') or {},
           'controls_not_run': plan.get('controls_not_run') or {'CTRL': False, 'OBJ': False}, 'alias_of': plan.get('alias_of') or {},
           'fineness': RU.grid_fineness(plan.get('fineness_inputs')), 'grid_valid': None, 'grid': {}, 'costs_vs': {},
           'nolookup': {}, 'optimistic': optimistic_input(plan), 'joint': None, 'novelty': None}
    if plan.get('revision_note'):
        rec['revision_note'] = plan['revision_note']
    dec = RU.decisions(rec)
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    C.write_json_x(outd / 'DECISIONS.json', {'schema': 'DRF1-decisions-v1', 'utc': C.now(), 'plan': plan.get('plan_id'), 'bank': None,
                                             'plan_sha256': C.sha(Path(plan_path)), 'rules_sha256': C.sha(RU.RULES_PATH),
                                             'decisions': dec, 'reader_stage': 'STOPPED_USEFULNESS_FAILURE',
                                             'registration_status': reg, 'optimistic_input': rec['optimistic'],
                                             'point_eligibility': NR.point_eligibility(ev)})
    print(json.dumps({k: (v.get('label') if isinstance(v, dict) else v) for k, v in dec.items() if k != 'scope'}, indent=1,
                     default=str))
    return dec


def d2r(plan_path, main_bank, rob_bank, out):
    """D2R of DR (RULES.md 8; descriptive; never changes D2): registered and robustness cells merged, refit effectiveness, the
    registered decisive comparators (the CTRL bracket Q* read from the reading's DECISIONS.json, and OBJ_MI); never re-chosen."""
    plan = json.loads(Path(plan_path).read_text())
    dec = json.loads((Path(out).parent / plan['plan_id'] / 'DECISIONS.json').read_text())['decisions']
    gm, gr = CL.load_grid(main_bank), CL.load_grid(rob_bank)
    g = CL.merge_grids(gm, gr)
    cand, twin_id, obj_id = plan['cand'], plan['twin']['id'], plan['obj_mi']['config_id']
    d2 = dec.get('D2') or {}
    fr = ((d2.get('comparisons') or {}).get('CTRL_FRONTIER') or {}).get('frontier') or {}
    q = fr.get('bracketing_control')
    comparators = {'CTRL': q, 'OBJ': obj_id}
    seeds, boot = CL.seed_records(ks=(0, 1, 2, 3), reps=('R1', 'R2', 'R3', 'R4'))
    checks = CL.seed_checks(g, pairs=((0, 1), (2, 3)))
    minimum = [m for m in [cand, twin_id, obj_id, 'SUCC', q] if m]
    distinct = CL.distinct_inputs(rob_bank)
    eff = RU.refit_effectiveness(CL.split_records(), seeds, checks, minimum, distinct, boot_seeds=boot)
    keys = [m for m in [q, obj_id, twin_id, 'SUCC'] if m] + [r for r in REFS if r != 'SUCC']
    rob = {k: g['paired'].get(cand, {}).get(k) for k in dict.fromkeys(keys)}
    s_reg, b_reg = CL.seed_records()
    gv = RU.grid_validity(s_reg, CL.seed_checks(gm), minimum, CL.distinct_inputs(main_bank), boot_seeds=b_reg)
    rec = {'grid_valid': gv['effective'], 'release': cand, 'refit': eff, 'twin_id': twin_id,
           'rob': {k: v for k, v in rob.items() if v is not None}, 'comparators': comparators}
    res = RU.d2r(rec)
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    C.write_json_x(outd / 'D2R.json', {'schema': 'DRF1-d2r-v1', 'utc': C.now(), 'plan': plan.get('plan_id'), 'main_bank': main_bank,
                                       'rob_bank': rob_bank, 'comparators': comparators, 'd2r': res, 'refit_effectiveness': eff,
                                       'refit_checks': len(checks), 'seeds': seeds, 'boot_seeds': boot})
    print(json.dumps({'d2r': res.get('label'), 'refit': eff.get('label'), 'comparators': comparators}, indent=1, default=str))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--plan', required=True); r.add_argument('--bank', required=True)
    r.add_argument('--out', required=True)
    d = sub.add_parser('d2r'); d.add_argument('--plan', required=True); d.add_argument('--main', required=True)
    d.add_argument('--rob', required=True); d.add_argument('--out', required=True)
    o = sub.add_parser('optimistic'); o.add_argument('--plan', required=True)
    st = sub.add_parser('stopped'); st.add_argument('--plan', required=True); st.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        run(a.plan, a.bank, a.out)
    elif a.cmd == 'stopped':
        run_stopped(a.plan, a.out)
    elif a.cmd == 'd2r':
        d2r(a.plan, a.main, a.rob, a.out)
    else:
        plan = json.loads(Path(a.plan).read_text())
        print(json.dumps(RU.useful_optimistic(optimistic_input(plan)), indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
