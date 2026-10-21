"""Invented-data tests of fsa.rules (no real data, no label). Run: python3 -m fsa.test_rules (prints JSON; exit 0 iff all pass).

Covers: the rule file and its refusal of altered constants; nrm.rules and frc.rules left unchanged; route D (each documented
approximation, missing specification, no contradiction, outside the real pipeline, no effect, unverified, immaterial, material);
route E (recovery and task usefulness, inclusive / exclusive thresholds, objective ties, proxy-only, revised worse, not deployable,
unrestricted enumeration never advancing, same-class reference gap, not runnable, no distinct source point, registration);
route S (recovery and task rationale, pooled resolution, the surrogate-only SV1 case closing, spend bracket, invalid records);
the advancement decision (supersession of E by S, D alone, the equal-effort control never advancing, revisions, withdrawal);
the FSA1 reader-validity rule (+800, offsets 300-700 forbidden, expected small-support invariance, seed-implementation errors,
splits, bootstrap seeds, recorded checks, n_eff, not retroactive); the operating path; C1; C3 and the outcome classes; diagnostics
never decision inputs; the P1 replay acceptance. Invented utility records and grid entries come from nrm.test_rules (unchanged)."""
from __future__ import annotations
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

from nrm import rules as NR
from nrm import test_rules as NT                 # invented-data helpers (util, evals, entry, reg, grid_entry, costs, ALIAS)
from frc import rules as FR
from fsa import rules as R

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------- rule file and the inherited modules
@test
def t_rules_load_and_refuse_altered_constants():
    r = R.load_rules()
    assert r['schema'] == 'FSA1-rules-v1' and len(r['_sha256']) == 64
    raw = json.loads(R.RULES_PATH.read_text())
    for path, val in ((('tolerances', 'task_no_worse'), 0.001), (('tolerances', 'objective_tie_abs'), 1e-6),
                      (('reader_validity', 'forbidden_offsets'), [300, 400, 500, 600]), (('reader_validity', 'seed_offset'), 700),
                      (('p1_replay', 'reproduction_tolerance_abs'), 1e-9), (('advancement', 'max_registered_revisions'), 3),
                      (('reader_validity', 'bootstrap_seed_base'), 2026102100), (('tolerances', 'auc_material'), 0.002)):
        bad = copy.deepcopy(raw)
        d = bad
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = val
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'RULES.json'
            p.write_text(json.dumps(bad))
            try:
                R.load_rules(p)
            except ValueError:
                continue
            raise AssertionError(f'altered {path} accepted')


@test
def t_inherited_unchanged_not_monkeypatched():
    assert NR.load_rules()['_sha256'] == R.NRM1_RULES_SHA256 and FR.load_rules()['_sha256'] == R.FRC1_RULES_SHA256
    R.reader_validity({}, [], {}, {}, {}, {})
    assert NR.FORBIDDEN_OFFSETS == (300, 500) and FR.FORBIDDEN_OFFSETS == (300, 400, 500, 600) and FR.SEED_OFFSET == 700
    assert R.forbidden_seeds() > FR.forbidden_seeds() > NR.forbidden_seeds() and len(R.forbidden_seeds()) == 100
    assert R.registered_seed(1, 3) == NR.MODEL_SEED_BASE + 813 and R.registered_boot_seed(1, 2) == 2026102302


# ---------------------------------------------------------------- route D
def defect(**kw):
    d = {'id': 'D1', 'location': 'nrm/fa/prc.py:lloyd', 'in_real_pipeline': True,
         'specification': {'source': 'code docstring contract', 'quote': 'every accepted state is (book, route(E, book))'},
         'approximation': None, 'contradiction_demonstrated': True,
         'changes': {'partition_people_changed': 12, 'release_max_abs_change': 0.01, 'decision_inputs': []}, 'verified': True}
    d.update(kw)
    return d


@test
def t_route_d():
    assert R.defect_label(defect()) == 'DEFECT_VERIFIED_MATERIAL'
    assert R.route_d({'claims': [defect()]})['label'] == 'D_ADVANCES'
    for a in R.DOCUMENTED_APPROXIMATIONS:                                   # documented approximations are never defects
        assert R.defect_label(defect(approximation=a)) == 'NOT_A_DEFECT_DOCUMENTED_APPROXIMATION', a
    assert R.defect_label(defect(specification={'source': 'reviewer opinion', 'quote': 'x'})) == \
        'NOT_A_DEFECT_NO_COMMITTED_SPECIFICATION'
    assert R.defect_label(defect(specification={'source': 'FA_PRC_v1 section 1', 'quote': ' '})) == \
        'NOT_A_DEFECT_NO_COMMITTED_SPECIFICATION'
    assert R.defect_label(defect(contradiction_demonstrated=False)) == 'NOT_A_DEFECT_NO_CONTRADICTION'
    assert R.defect_label(defect(in_real_pipeline=False)) == 'NOT_A_DEFECT_NOT_IN_REAL_PIPELINE'
    none = {'partition_people_changed': 0, 'release_max_abs_change': 1e-13,
            'decision_inputs': [{'name': 'auc', 'kind': 'auc', 'old': 0.62, 'new': 0.62 + 1e-13}]}
    assert R.defect_label(defect(changes=none)) == 'NOT_A_DEFECT_NO_EFFECT'
    u = R.route_d({'claims': [defect(verified=False)]})
    assert u['label'] == 'INCOMPLETE' and u['per_claim']['D1'] == 'DEFECT_UNVERIFIED'
    small = {'partition_people_changed': 0, 'release_max_abs_change': 1e-6,
             'decision_inputs': [{'name': 'nested auc', 'kind': 'auc', 'old': 0.6227, 'new': 0.6229}]}
    assert R.defect_label(defect(changes=small)) == 'DEFECT_VERIFIED_IMMATERIAL'
    assert R.route_d({'claims': [defect(changes=small)]})['label'] == 'D_NO_MATERIAL_DEFECT'
    disc = {'partition_people_changed': 0, 'decision_inputs': [{'name': 'selection S2', 'kind': 'discrete', 'old': 'PRC@8.0',
                                                                'new': 'PRC@4.0'}]}
    assert R.defect_label(defect(changes=disc)) == 'DEFECT_VERIFIED_MATERIAL'
    for d, lab in ((0.0005, 'DEFECT_VERIFIED_MATERIAL'), (0.0004, 'DEFECT_VERIFIED_IMMATERIAL')):
        ch = {'partition_people_changed': 0, 'decision_inputs': [{'name': 'spend', 'kind': 'task', 'old': 0.0, 'new': d}]}
        assert R.defect_label(defect(changes=ch)) == lab, d
    assert R.route_d({'claims': []})['label'] == 'D_NO_MATERIAL_DEFECT' and R.route_d({})['label'] == 'INCOMPLETE'
    assert R.route_d({'claims': [defect()]})['old_records_edited'] is False


# ---------------------------------------------------------------- route E
def pt(J, auc=(0.650, 0.560, 0.630), task=None, deployable=True):
    t = {m: 0.5 for m in R.TASK_METRICS}
    t.update(task or {})
    return {'J': J, 'deployable': deployable, 'auc': dict(zip(NR.VIEWS, auc)), 'task': t}


def e_cell(law='L1', lam=8.0, o=None, r=None, ref=None, unr=None, **kw):
    c = {'law': law, 'lam': lam, 'same_start': True, 'same_objective': True, 'same_capacity': True, 'same_rule_family': True,
         'original': o or pt(1.0), 'revised': r or pt(1.0), 'reference': ref, 'unrestricted': unr}
    c.update(kw)
    return c


def e_rec(cells, **kw):
    rec = {'revision_id': 'SV1', 'laws_registered_before_run': True, 'solver_registered_before_run': True,
           'runnable_on_source': True, 'source_distinct': {0: True, 1: True},
           'registered_cells': [[c['law'], c['lam']] for c in cells], 'cells': cells}
    rec.update(kw)
    return rec


USEFUL = pt(0.999, auc=(0.648, 0.560, 0.630))                  # J -0.001, pair AUC -0.002, tasks unchanged


@test
def t_route_e_usefulness_and_thresholds():
    e = R.route_e(e_rec([e_cell(r=USEFUL), e_cell(law='L2')]))
    assert e['label'] == 'E_ADVANCES' and e['useful_cells'] == [['L1', 8.0]]
    proxy = pt(1.0 - 0.0012)                                       # the SV1 trace: objective lower, nothing else moves
    assert R.route_e(e_rec([e_cell(r=proxy)]))['label'] == 'E_PROXY_ONLY'
    assert R.route_e(e_rec([e_cell(r=pt(0.999, auc=(0.649, 0.560, 0.630)))]))['label'] == 'E_ADVANCES'   # -0.001 inclusive
    assert R.route_e(e_rec([e_cell(r=pt(0.999, auc=(0.6491, 0.560, 0.630)))]))['label'] == 'E_PROXY_ONLY'  # -0.0009
    worse_task = pt(0.999, auc=(0.640, 0.560, 0.630), task={'occ_brier_men': 0.5005})                  # +0.0005 is worse
    assert R.route_e(e_rec([e_cell(r=worse_task)]))['label'] == 'E_PROXY_ONLY'
    ok_task = pt(0.999, auc=(0.640, 0.560, 0.630), task={'occ_brier_men': 0.5004})
    assert R.route_e(e_rec([e_cell(r=ok_task)]))['label'] == 'E_ADVANCES'
    view = pt(0.999, auc=(0.640, 0.571, 0.630))                                                        # income +0.011
    assert R.route_e(e_rec([e_cell(r=view)]))['label'] == 'E_PROXY_ONLY'
    task = pt(0.999, task={'inc_nll': 0.4995, 'occ_nll': 0.4995})                                      # total NLL -0.001
    assert R.route_e(e_rec([e_cell(r=task)]))['label'] == 'E_ADVANCES'
    mis = pt(0.999, auc=(0.651, 0.560, 0.630), task={'inc_nll': 0.4990, 'occ_nll': 0.4990})            # pair +0.001
    m = R.route_e(e_rec([e_cell(r=mis)]))
    assert m['label'] == 'E_PROXY_ONLY' and m['flags'] == ['OBJECTIVE_RECOVERY_MISALIGNED_L1@8.0']
    tie = pt(1.0 - 5e-10, auc=(0.640, 0.560, 0.630))                                                   # an objective tie
    assert R.route_e(e_rec([e_cell(r=tie)]))['label'] == 'E_OLD_MATCHES_REFERENCE'


@test
def t_route_e_reference_class_and_validity():
    assert R.route_e(e_rec([e_cell(r=USEFUL), e_cell(law='L2', r=pt(1.001))]))['label'] == 'E_REVISED_WORSE_SOMEWHERE'
    assert R.route_e(e_rec([e_cell(r=pt(0.99, deployable=False))]))['label'] == 'E_REVISED_NOT_DEPLOYABLE'
    unr = R.route_e(e_rec([e_cell(unr=pt(0.8, auc=(0.55, 0.55, 0.55)))]))          # unrestricted enumeration never advances
    assert unr['label'] == 'E_OLD_MATCHES_REFERENCE' and abs(unr['cells'][0]['unrestricted_gap_reported_only'] - 0.2) < 1e-12
    gap = R.route_e(e_rec([e_cell(ref={**pt(0.99), 'same_class': True})]))         # a same-class gap the revision does not close
    assert gap['label'] == 'E_GAP_WITHOUT_RUNNABLE_REVISION' and not gap['cells'][0]['original_matches_reference']
    other = R.route_e(e_rec([e_cell(ref={**pt(0.99), 'same_class': False})]))      # another class is not the reference
    assert other['label'] == 'E_OLD_MATCHES_REFERENCE'
    assert R.route_e(e_rec([e_cell(r=USEFUL)], runnable_on_source=False))['label'] == 'E_GAP_WITHOUT_RUNNABLE_REVISION'
    assert R.route_e(e_rec([e_cell(r=USEFUL)], source_distinct={0: True, 1: False}))['label'] == 'E_NO_DISTINCT_SOURCE_POINT'
    assert R.route_e(e_rec([e_cell(r=USEFUL)], laws_registered_before_run=False))['label'] == 'E_INVALID'
    assert R.route_e(e_rec([e_cell(r=USEFUL)], solver_registered_before_run=None))['label'] == 'E_INVALID'
    rec = e_rec([e_cell(r=USEFUL)])
    rec['registered_cells'].append(['L9', 8.0])                                     # a registered cell not reported
    assert R.route_e(rec)['label'] == 'E_INVALID'
    assert R.route_e(e_rec([e_cell(r=USEFUL, same_objective=False)]))['label'] == 'E_INVALID'
    assert R.route_e(e_rec([e_cell(r=USEFUL), e_cell(r=USEFUL)]))['label'] == 'E_INVALID'      # duplicated cell
    bad = e_cell(r={'J': 0.9, 'deployable': True})                                  # improvement without exact metrics
    assert R.route_e(e_rec([bad]))['label'] == 'E_INVALID'


# ---------------------------------------------------------------- route S
def s_rec(dJ=-0.001, d_auc=-0.003, se_auc=0.001, d_sp=0.0, se_sp=0.0005, r_spend=0.004, o_path=None, views=0.0, **kw):
    fitted = {v: {lam: {'original': 1.0, 'revised': 1.0 + (dJ if lam == 8.0 else 0.0), 'revised_deployable': True,
                        'distinct_from_original': lam == 8.0, 'distinct_from_twin': True} for lam in R.LAMBDA_GRID} for v in (0, 1)}
    path = o_path or [{'id': 'O@8.0', 'spend': 0.004}]
    paired = {p['id']: {'d_auc': (d_auc, se_auc), 'd_spend': (d_sp, se_sp), 'd_auc_income': (views, 0.001),
                        'd_auc_occupation': (views, 0.001)} for p in path}
    nested = {v: {'revised_point': 'SV1@8.0', 'revised_spend': r_spend, 'original_s2': path[-1]['id'], 'original_path': path,
                  'paired': copy.deepcopy(paired)} for v in (0, 1)}
    rec = {'revision_id': 'SV1', 'card_committed_before_run': True, 'model_select_opened': False,
           'label_derived_statistics_disclosed': True, 'original_entries_equal_replay': True, 'fitted': fitted, 'nested': nested}
    rec.update(kw)
    return rec


@test
def t_route_s():
    s = R.route_s(s_rec())
    assert s['label'] == 'S_ADVANCES' and s['recovery_rationale'] and s['improving_strengths_both_versions'] == [8.0]
    sv1 = R.route_s(s_rec(dJ=-0.0012, d_auc=0.0, d_sp=0.0))      # objective lower; nested recovery and task do not improve
    assert sv1['label'] == 'S_PROXY_ONLY' and sv1['label'] in R.CLOSING['S']
    assert R.route_s(s_rec(se_auc=0.002))['label'] == 'S_PROXY_ONLY'                 # -0.003 + 2 x 0.002 > 0: not resolved
    assert R.route_s(s_rec(d_sp=0.0006))['label'] == 'S_PROXY_ONLY'                  # spends more than 'no worse'
    assert R.route_s(s_rec(views=0.011))['label'] == 'S_PROXY_ONLY'                  # a recipient view rises beyond 0.010
    t = R.route_s(s_rec(d_auc=0.0, d_sp=-0.002, se_sp=0.0005))                       # task rationale
    assert t['label'] == 'S_ADVANCES' and t['task_rationale'] and not t['recovery_rationale']
    assert R.route_s(s_rec(d_auc=0.0, d_sp=-0.002, se_sp=0.0011))['label'] == 'S_PROXY_ONLY'
    assert R.route_s(s_rec(d_auc=0.001, d_sp=-0.002, se_sp=0.0005))['label'] == 'S_PROXY_ONLY'   # cheaper but recovery +0.001
    assert R.route_s(s_rec(dJ=-5e-10))['label'] == 'S_OLD_MATCHES_REFERENCE'          # a tie is no improvement
    w = s_rec()
    w['fitted'][1][0.25]['revised'] = 1.0 + 1e-6
    assert R.route_s(w)['label'] == 'S_REVISED_WORSE_SOMEWHERE'
    nd = s_rec()
    nd['fitted'][0][2.0]['revised_deployable'] = False
    assert R.route_s(nd)['label'] == 'S_REVISED_NOT_DEPLOYABLE'
    for k, val in (('model_select_opened', True), ('label_derived_statistics_disclosed', None),
                   ('card_committed_before_run', False), ('original_entries_equal_replay', False)):
        assert R.route_s(s_rec(**{k: val}))['label'] == 'S_INVALID', k
    nf = s_rec()
    nf['nested'][1]['revised_point'] = None
    assert R.route_s(nf)['label'] == 'S_NO_FEASIBLE_REVISED_POINT'
    al = s_rec()
    al['nested'][0]['alias_of_original_s2'] = True
    a = R.route_s(al)
    assert a['label'] == 'S_PROXY_ONLY' and 'ALIAS_OF_ORIGINAL_AT_S2_V0' in a['flags']
    assert R.route_s(s_rec(nested=None))['label'] == 'INCOMPLETE'
    path = [{'id': 'O@1.0', 'spend': 0.001}, {'id': 'O@8.0', 'spend': 0.006}]
    assert R._bracket(path, 0.004) == 'O@8.0' and R._bracket(path, 0.0012) == 'O@1.0' and R._bracket(path, 0.01) == 'O@8.0'
    b = R.route_s(s_rec(o_path=path, r_spend=0.0012))
    assert b['per_version'][0]['bracket'] == 'O@1.0' and b['per_version'][0]['original_s2'] == 'O@8.0'


# ---------------------------------------------------------------- the advancement decision
def adv(e=None, s=None, d=None, att=('D', 'E', 'S'), revs=('SV1',), **kw):
    rec = {'registered_revisions': list(revs), 'attempted': {x: x in att for x in 'DES'}, 'D': d or {'claims': []},
           'E': {'SV1': e or e_rec([e_cell(r=pt(0.9988))])}, 'S': {'SV1': s or s_rec(dJ=-0.0012, d_auc=0.0)}}
    rec.update(kw)
    return rec


@test
def t_advancement():
    c = R.advancement(adv())                                                         # surrogate-only everywhere: closes
    assert c['label'] == 'OPTIMIZER_REPAIR_CLOSED' and c['statement'] == R.CLOSING_STATEMENT
    assert c['routes_evaluated'] == ['D', 'E', 'S'] and c['routes_not_attempted'] == []
    good_e = e_rec([e_cell(r=USEFUL)])
    c = R.advancement(adv(e=good_e))                                                  # S closes for SV1: E is superseded
    assert c['label'] == 'OPTIMIZER_REPAIR_CLOSED' and 'E_SUPERSEDED_BY_S_SV1' in c['flags']
    a = R.advancement(adv(e=good_e, att=('D', 'E')))                                  # S not attempted: E advances
    assert a['label'] == 'ADVANCE' and a['advancing_routes'] == ['E'] and a['pilot_solver'] == 'SV1'
    a = R.advancement(adv(s=s_rec()))
    assert a['label'] == 'ADVANCE' and a['advancing_routes'] == ['S'] and not a['defects_corrected_in_pilot_solver']
    a = R.advancement(adv(d={'claims': [defect()]}))
    assert a['label'] == 'ADVANCE' and a['advancing_routes'] == ['D'] and a['pilot_solver'] == 'CORRECTED_ORIGINAL'
    assert R.advancement(adv(e=good_e, s=s_rec(nested=None)))['label'] == 'INCOMPLETE'   # S pending: no advance on E
    assert R.advancement(adv(d={'claims': [defect()]}, s=s_rec(nested=None)))['label'] == 'INCOMPLETE'   # complete record only
    r = adv()
    del r['S']['SV1']
    assert R.advancement(r)['label'] == 'INCOMPLETE'                                 # every revision on every attempted route
    assert R.advancement(adv(revs=('SV1', 'SV2', 'SV3')))['label'] == 'INCOMPLETE'    # at most two variants
    assert R.advancement(adv(att=()))['label'] == 'INCOMPLETE'
    w = adv(e=e_rec([e_cell(r=USEFUL)], laws_registered_before_run=False))
    assert R.advancement(w)['label'] == 'INCOMPLETE'                                  # an invalid route blocks closure
    w['withdrawn'] = ['E']
    c = R.advancement(w)
    assert c['label'] == 'OPTIMIZER_REPAIR_CLOSED' and 'ROUTE_E_WITHDRAWN_AFTER_INVALID' in c['flags']
    ctl = adv(controls={'S': {'SV1_first': s_rec()}})                                 # the equal-effort control never advances
    c = R.advancement(ctl)
    assert c['label'] == 'OPTIMIZER_REPAIR_CLOSED' and c['equal_effort_controls_reported_only']['S']['SV1_first'] == 'S_ADVANCES'
    objs = {v: {lam: {'original': 1.0, 'revised': 0.998, 'control': 0.999} for lam in R.LAMBDA_GRID} for v in (0, 1)}
    at = R.equal_effort_attribution(objs)
    assert at['counts'] == {'resolution': 12, 'selection': 12, 'selection_hurts': 0} and at['decision_input'] is False
    assert R.advancement(adv(source_objectives=objs))['label'] == 'OPTIMIZER_REPAIR_CLOSED'   # attribution changes nothing


# ---------------------------------------------------------------- reader validity
WIRES = {'PRC': {'income': 16, 'occupation': 64, 'pair': 900}, 'NEURAL_RAW': {'income': 5000, 'occupation': 5000, 'pair': 5000}}


def rv(reps=('R1', 'R2'), ks=(0, 1), offset=R.SEED_OFFSET, eq=None):
    seeds = {f'S{v}k{k}': NR.MODEL_SEED_BASE + offset + 10 * v + k for v in (0, 1) for k in ks}
    checks = []
    for r in NR.SEED_DEPENDENT:
        for w, di in WIRES.items():
            for view in NR.VIEWS:
                for v in (0, 1):
                    for rp in reps:
                        same = r in ('ET5', 'ET25') and w == 'PRC' and view != 'pair'      # expected finite-view invariance
                        checks.append({'recipe': r, 'view': view, 'wire': w, 'v': v, 'R': rp,
                                       'equal_k0_k1': (eq or {}).get((r, w, view, v, rp), same)})
    checks.append({'recipe': 'LOOKUP', 'view': 'pair', 'wire': 'PRC', 'v': 0, 'R': reps[0], 'equal_k0_k1': True})
    distinct = {f'{w}|V{v}': dict(di) for w, di in WIRES.items() for v in (0, 1)}
    splits = {rp: {'tag': f'FSA1_devgrid_{rp}', 'assignment_sha256': f'hash{rp}', 'fit_score_household_overlap': 0,
                   'score_overlap': {q: 0.3 for q in reps if q != rp}} for rp in reps}
    boot = {f'S{v}{rp}': R.registered_boot_seed(v, i) for v in (0, 1) for i, rp in enumerate(reps)}
    return seeds, checks, distinct, splits, boot, {'schema': [], 'access': [], 'score': []}


@test
def t_reader_validity():
    g = R.reader_validity(*rv())
    assert g['label'] == 'GRID_VALID', g['errors']
    assert g['expected_small_support_equalities'] == 2 * 2 * 2 * 2           # ET5, ET25 x income, occupation x 2 v x 2 R
    assert g['stochastic_recipes']['ET5']['small_support'] == {'checks': 12, 'differ': 4}     # the 900-value pair view differs
    assert g['stochastic_recipes']['MLP64']['large_support'] == {'checks': 12, 'differ': 12}
    for off in R.FORBIDDEN_OFFSETS:
        x = R.reader_validity(*rv(offset=off))
        assert x['label'] == 'GRID_INVALID' and any('reuse earlier' in e for e in x['errors']), off
    a = rv()
    a[0]['S1k1'] = NR.MODEL_SEED_BASE + 800 + 19
    assert any('differ from the registered' in e for e in R.reader_validity(*a)['errors'])
    big = R.reader_validity(*rv(eq={('MLP64', 'NEURAL_RAW', 'pair', 1, 'R2'): True}))     # equality on a large-support view
    assert big['label'] == 'GRID_INVALID' and any('seed-implementation error' in e for e in big['errors'])
    s, ch, di, sp, bo, er = rv()
    ch = [c for c in ch if not (c['recipe'] == 'ET5' and c['wire'] == 'NEURAL_RAW' and c['v'] == 1 and c['R'] == 'R2')]
    u = R.reader_validity(s, ch, di, sp, bo, er)
    assert u['label'] == 'GRID_INVALID' and any('untested' in e and 'ET5|V1|R2' in e for e in u['errors'])
    for mut in (lambda x: x[3]['R2'].update(assignment_sha256='hashR1'), lambda x: x[3]['R1'].update(tag='FRC1_devgrid_R1'),
                lambda x: x[3]['R1'].update(fit_score_household_overlap=3), lambda x: x[3]['R1']['score_overlap'].update(R2=1.0),
                lambda x: x[4].update(S0R1=2026102100), lambda x: x[5].update(schema=['cell S0R1k0 missing']),
                lambda x: x[5].pop('access'), lambda x: x[2]['PRC|V0'].pop('income')):
        z = rv()
        mut(z)
        assert R.reader_validity(*z)['label'] == 'GRID_INVALID'
    nd = rv()
    nd[1].append({'recipe': 'OUT_LR', 'view': 'pair', 'wire': 'PRC', 'v': 0, 'R': 'R1', 'equal_k0_k1': False})
    x = R.reader_validity(*nd)
    assert x['label'] == 'GRID_VALID' and x['flags'] == ['NONDETERMINISTIC_RECIPE']
    # duplicate cells count once in the inherited aggregate: k0 = k1 in every block gives 4 distinct of 8 cells
    dup = NT.entry({c: (0.01 if c.endswith('k0') or c.endswith('k1') else 0) + 0.001 * int(c[1]) + 0.0001 * int(c[3])
                    for c in NR.REGISTERED})
    assert NR.aggregate(dup)['n_eff_distinct_cells'] == 4
    # prospective, never retroactive: FRC1's +700 seeds fail here while frc.rules still accepts them unchanged
    s7 = {k: x for k, x in rv(offset=700)[0].items()}
    assert R.reader_validity(s7, *rv()[1:])['label'] == 'GRID_INVALID'
    assert not FR._seed_errors(s7, NR.REG_K)


@test
def t_refit_effective():
    ok = R.refit_effective(*rv(reps=('R1', 'R2', 'R3', 'R4'), ks=(0, 1, 2, 3)))
    assert ok['label'] == 'REFIT_EFFECTIVE' and ok['decision_input'] is False, ok['validity']['errors']
    z = rv(reps=('R1', 'R2', 'R3', 'R4'), ks=(0, 1, 2, 3))
    z[3]['R3']['assignment_sha256'] = 'hashR1'
    assert R.refit_effective(*z)['label'] == 'REFIT_NOT_EFFECTIVE'
    assert R.refit_effective(*rv())['label'] == 'REFIT_NOT_EFFECTIVE'                 # k2 / k3 and R3 / R4 missing


# ---------------------------------------------------------------- the pilot
@test
def t_operating_path():
    sha = {'TWIN': {0: 't0', 1: 't1'}, 'P@1.0': {0: 'a0', 1: 't1'}, 'P@8.0': {0: 'b0', 1: 'b1'}, 'Q@8.0': {0: 'b0', 1: 'c1'}}
    sel = {0: {'S1': 'P@1.0', 'S2': 'P@8.0', 'S3': 'Q@8.0'}, 1: {'S1': 'P@1.0', 'S2': 'P@8.0', 'S3': None}}
    p = R.operating_path(sel, sha, 'MODEL_FIT_NESTED')
    assert p['per_version'][0]['path'] == ['TWIN', 'P@1.0', 'P@8.0'] and p['per_version'][0]['aliases'] == {'Q@8.0': 'P@8.0'}
    assert p['per_version'][1]['path'] == ['TWIN', 'P@8.0'] and p['per_version'][1]['aliases'] == {'P@1.0': 'TWIN'}
    assert p['per_version'][1]['primary'] == 'P@8.0'
    assert R.operating_path(sel, sha, 'MODEL_SELECT')['label'] == 'INVALID'
    del sha['P@8.0']
    assert R.operating_path(sel, sha, 'MODEL_FIT_NESTED')['label'] == 'INCOMPLETE'


@test
def t_c1():
    f = {v: {lam: {'original': 1.0, 'revised': 0.999} for lam in R.LAMBDA_GRID} for v in (0, 1)}
    assert R.c1(f, {0: 8.0, 1: 8.0})['label'] == 'BETTER_OPTIMIZATION'
    g = copy.deepcopy(f)
    g[1][0.5]['revised'] = 1.0 + 2e-9
    assert R.c1(g, {0: 8.0, 1: 8.0})['label'] == 'MIXED'
    h = copy.deepcopy(f)
    h[0][8.0]['revised'] = 1.0 - 5e-10
    assert R.c1(h, {0: 8.0, 1: 8.0})['label'] == 'NOT_BETTER'
    assert R.c1(f, {0: 8.0, 1: 3.0})['label'] == 'INCOMPLETE'


POINTS = ('TWIN', 'HQ_S2r', 'HQ_BRr', 'UNT_BR', 'ORIG_PRC_S2')


def pilot_rec(pair=0.02, refs=0.03, orig=None, orig_cost=-0.0005, g1=True, revised_J=0.999):
    grid = {i: NT.grid_entry(pair) for i in POINTS}
    grid['ORIG_PRC_S2'] = NT.grid_entry(pair if orig is None else orig)
    grid.update({r: NT.grid_entry(refs) for r in NR.BENCHMARK})
    costs = {'TWIN': NT.costs(0.003), 'HQ_S2r': NT.costs(0.001), 'HQ_BRr': NT.costs(-0.001), 'UNT_BR': NT.costs(-0.001),
             'ORIG_PRC_S2': NT.costs(orig_cost)}
    return {'advancement': {'label': 'ADVANCE'}, 'grid_valid': True,
            'c1': {'fitted': {v: {lam: {'original': 1.0, 'revised': revised_J} for lam in R.LAMBDA_GRID} for v in (0, 1)},
                   'operating_lambda': {0: 8.0, 1: 8.0}},
            'revised_prc': {'status': 'FITTED', 'config_id': 'SV1_PRC_S2',
                            'evals': NT.evals() if g1 else NT.evals(occ={('U0', 'nll_excess'): 0.011})},
            'grid': grid, 'paths': {'HQ': ['HQ_S2r', 'HQ_BRr'], 'CONTROL': ['UNT_BR'], 'ORIGINAL': ['TWIN', 'ORIG_PRC_S2']},
            'original_primary': 'ORIG_PRC_S2', 'alias_of': {}, 'costs_vs': costs,
            'f4': {'version': {v: {'diff': 0.01, 'se': 0.003} for v in (0, 1)}},
            'nolookup': {i: {'cells': NT.reg(lambda c: 0.01)} for i in POINTS + NR.BENCHMARK},
            'nested_lookup': {i: {v: 0.01 for v in (0, 1)} for i in POINTS}, 'reader_budget': {},
            'effort': {'PRC': {'settings_sha256': 'a', 'stopped_by': 'CONVERGED'},
                       'HQ': {'settings_sha256': 'a', 'stopped_by': 'CONVERGED'}}, 'joint': None}


@test
def t_pilot_outcome_classes():
    out = R.pilot(pilot_rec())
    assert out['outcome_class']['label'] == 'METHOD_CANDIDATE' and out['C3']['label'] == 'PROTECTION_BEYOND_MATCHED_CONTROLS'
    assert out['C1']['label'] == 'BETTER_OPTIMIZATION' and out['C2']['label'] == 'TASK_FEASIBLE' and out['C4']['met']
    assert 'MATERIAL_0020_MARGIN_MET' in out['outcome_class']['flags'] and out['C6']['label'] == 'NOT_ASSESSED_MISSING_CONTROLS'
    assert out['C7']['label'] == 'NOT_CLAIMED' and abs(out['outcome_class']['bound'] - 0.008) < 1e-12
    assert R.pilot(pilot_rec(refs=0.001))['outcome_class']['label'] == 'COMPONENT'
    r = pilot_rec()
    r['grid']['UNT_BR'] = NT.grid_entry(0.0005)                                     # no resolved benefit at the control bracket
    out = R.pilot(r)
    assert out['C3']['label'] == 'PARTIAL_TWIN+HQ_FRONTIER_ONLY' and out['R_VS_O']['label'] == 'RECOVERY_GAIN_AT_MATCHED_SPEND'
    assert out['outcome_class']['label'] == 'ENGINEERING_RESULT'
    r = pilot_rec(orig=0.0)
    r['grid']['UNT_BR'] = NT.grid_entry(0.0005)
    out = R.pilot(r)
    assert out['R_VS_O']['label'] == 'NO_MEASURED_GAIN_OVER_ORIGINAL' and out['outcome_class']['label'] == 'BOUNDED_NEGATIVE_RESULT'
    r = pilot_rec(orig=0.0, orig_cost=-0.003)
    r['grid']['UNT_BR'] = NT.grid_entry(0.0005)
    assert R.pilot(r)['R_VS_O']['label'] == 'TASK_GAIN_AT_NO_WORSE_RECOVERY'
    assert R.pilot(r)['outcome_class']['label'] == 'ENGINEERING_RESULT'
    r = pilot_rec(revised_J=1.0)                                                     # C1 not met: no engineering result
    r['grid']['UNT_BR'] = NT.grid_entry(0.0005)
    assert R.pilot(r)['outcome_class']['label'] == 'BOUNDED_NEGATIVE_RESULT'
    out = R.pilot(pilot_rec(revised_J=1.0))
    assert out['outcome_class']['label'] == 'METHOD_CANDIDATE' and 'NOT_ATTRIBUTABLE_TO_BETTER_OPTIMIZATION' in \
        out['outcome_class']['flags']
    out = R.pilot(pilot_rec(g1=False))
    assert out['C2']['label'] == 'TASK_INFEASIBLE' and out['C3']['label'] == 'MET_WITHOUT_TASK_FEASIBILITY'
    assert out['outcome_class']['label'] == 'BOUNDED_NEGATIVE_RESULT'
    assert R.pilot({**pilot_rec(), 'advancement': {'label': 'OPTIMIZER_REPAIR_CLOSED'}})['label'] == 'PILOT_NOT_AUTHORIZED'
    r = pilot_rec()
    r['grid_valid'] = False
    out = R.pilot(r)
    assert out['C3']['label'] == 'NOT_COMPUTED_GRID_INVALID' and out['outcome_class']['label'] == 'INCOMPLETE'


@test
def t_c3_conjuncts_effort_and_aliases():
    r = pilot_rec()
    r['effort']['HQ']['settings_sha256'] = 'b'
    assert R.c3(r)['label'] == 'NOT_ASSESSED_UNEQUAL_EFFORT'
    r = pilot_rec()
    r['effort']['HQ']['stopped_by'] = 'CAP'                                          # HQ capped while PRC converged
    assert R.c3(r)['label'] == 'NOT_ASSESSED_UNEQUAL_EFFORT'
    r = pilot_rec()
    r['effort']['PRC']['stopped_by'] = 'CAP'                                         # PRC capped: conservative, allowed
    assert R.c3(r)['met']
    r = pilot_rec()
    del r['effort']
    assert R.c3(r)['label'] == 'INCOMPLETE'
    r = pilot_rec()
    r['f4']['version'][1] = {'diff': 0.005, 'se': 0.003}
    assert R.c3(r)['label'] == 'NOT_MET_F4'
    r = pilot_rec()
    r['nolookup']['HQ_BRr'] = {'cells': NT.reg(lambda c: -0.01 if c.startswith('S1') else 0.01)}
    assert R.c3(r)['label'] == 'READER_TARGETED'
    r = pilot_rec()
    r['paths']['CONTROL'] = []
    d = R.c3(r)
    assert 'CONTROL_AT_IDENTITY' in d['flags'] and d['comparisons']['CONTROL_FRONTIER']['path'] == ['TWIN'] and not d['met']
    r = pilot_rec()
    del r['costs_vs']['HQ_S2r']                                                      # a frontier input missing: never 'not passed'
    assert R.c3(r)['label'] == 'INCOMPLETE'
    r = pilot_rec()
    r['revised_prc']['alias_of_original'] = True
    assert R.c3(r)['label'] == 'NOT_ASSESSED_ALIAS_OF_ORIGINAL'
    r = pilot_rec()
    r['revised_prc']['alias_of_twin'] = True
    assert R.c3(r)['label'] == 'NOT_ASSESSED_EXACT_ALIAS'
    r = pilot_rec()
    r['revised_prc']['status'] = 'NOT_FEASIBLE'
    assert R.c3(r)['label'] == 'NOT_ASSESSED_NO_FEASIBLE_POINT'
    assert R.c3({**pilot_rec(), 'grid_valid': None})['label'] == 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'


@test
def t_reserve_and_diagnostics_never_gating():
    tight = pilot_rec()
    tight['revised_prc']['evals'] = NT.evals(occ={('U0', 'nll_excess'): 0.0095}, se=0.002)    # point feasible, reserve not met
    out = R.pilot(tight)
    assert out['outcome_class']['label'] == 'METHOD_CANDIDATE' and 'POINT_FEASIBLE_NOT_RESERVE_QUALIFIED' in \
        out['outcome_class']['flags']
    base = R.pilot(pilot_rec())
    with_diag = pilot_rec()
    with_diag.update({'configs': [], 'ablation': {'pair': -0.5}, 'refit': {'label': 'REFIT_NOT_EFFECTIVE'},
                      'strongest': 'X', 'robustness': {'pair': 'anything'}})
    assert json.dumps(R.pilot(with_diag), sort_keys=True, default=str) == json.dumps(base, sort_keys=True, default=str)
    dg = R.diagnostics({'configs': [], 'refit': dict(zip(('seeds', 'checks', 'distinct_inputs', 'splits', 'boot_seeds',
                                                          'recorded_errors'), rv()))})
    assert dg['decision_input'] is False and dg['strongest']['label'] == 'NO_FEASIBLE_NONZERO_RELEASE'
    assert dg['refit']['label'] == 'REFIT_NOT_EFFECTIVE'


# ---------------------------------------------------------------- P1 replay acceptance
def sel_rec():
    return {'nested_means': {'PRC_TWIN': {'occ_nll': 1.2, 'auc_pair': 0.63}, 'PRC@8.0': {'occ_nll': 1.205, 'auc_pair': 0.62}},
            'audit_spend_over_twin': {'PRC': {'PRC@8.0': {'occ_nll': [0.005, 0.001], 'inc_nll': [0.001, 0.0005]}}},
            'budgets': {'S2': {'occ_nll': 0.008, 'inc_nll': 0.004}},
            'brackets': {'brackets': {'HQ': 'HQ@8.0'}, 'spend': {'HQ': {'HQ@8.0': 0.006}}, 'target_total_nll_spend': 0.005,
                         'prc_s2': 'PRC@8.0'},
            'nested_lookup_pair': {'PRC_TWIN': {'auc': 0.63}, 'PRC@8.0': {'auc': 0.62, 'diff_twin_minus_entry': 0.01,
                                                                          'se_boot': 0.002}},
            'F4': {'entry': 'PRC@8.0', 'diff_twin_minus_prc': 0.01, 'se_boot': 0.002, 'pass_2se': True, 'rule': 'text'},
            'selection': {'PRC': {'S1': None, 'S2': 'PRC@8.0', 'S3': 'PRC@8.0'}}, 'utc': '2026-10-20T19:59Z', 'cpu_s': 1.0}


def replay(x=None):
    y = copy.deepcopy(x or sel_rec())
    y.update({'estimator_unchanged': True, 'per_person_outputs_saved': True, 'utc': 'later', 'cpu_s': 2.0})
    return y


def verif(x=None):
    y = copy.deepcopy(x or sel_rec())
    return {k: y[k] for k in R.P1_VERIFIER_MIN}


@test
def t_p1_replay():
    rec = {v: sel_rec() for v in (0, 1)}
    ok = R.p1_replay(rec, {v: replay() for v in (0, 1)}, {v: verif() for v in (0, 1)})
    assert ok['label'] == 'P1_REPLAY_VERIFIED' and ok['old_labels_changed'] is False
    near = replay()
    near['nested_means']['PRC@8.0']['occ_nll'] += 5e-13
    v = verif()
    v['audit_spend_over_twin']['PRC']['PRC@8.0']['occ_nll'][1] += 5e-10
    assert R.p1_replay(rec, {0: near, 1: replay()}, {0: v, 1: verif()})['label'] == 'P1_REPLAY_VERIFIED'
    far = replay()
    far['nested_means']['PRC@8.0']['occ_nll'] += 2e-12
    nr = R.p1_replay(rec, {0: far, 1: replay()}, {0: verif(), 1: verif()})
    assert nr['label'] == 'P1_NOT_REPRODUCED' and 'not decision inputs' in nr['consequence'] and not nr['old_labels_changed']
    v = verif()
    v['F4']['se_boot'] += 2e-9
    assert R.p1_replay(rec, {v_: replay() for v_ in (0, 1)}, {0: v, 1: verif()})['label'] == 'P1_VERIFICATION_DISAGREES'
    edge = sel_rec()
    edge['audit_spend_over_twin']['PRC']['PRC@8.0']['occ_nll'] = [0.006, 0.001]        # 0.006 + 2 x 0.001 = 0.008 = budget
    ve = verif(edge)
    ve['selection']['PRC']['S2'] = None                                                  # flips at the boundary: a tie
    t = R.p1_replay({0: edge, 1: sel_rec()}, {0: replay(edge), 1: replay()}, {0: ve, 1: verif()})
    assert t['label'] == 'P1_REPLAY_VERIFIED' and len(t['per_version'][0]['verifier_boundary_ties']) == 1
    off = verif()
    off['selection']['PRC']['S2'] = None                                                 # off the boundary: a disagreement
    assert R.p1_replay(rec, {v_: replay() for v_ in (0, 1)}, {0: off, 1: verif()})['label'] == 'P1_VERIFICATION_DISAGREES'
    f4e = sel_rec()
    f4e['F4'].update(diff_twin_minus_prc=0.004, se_boot=0.002)
    vf = verif(f4e)
    vf['F4']['pass_2se'] = False
    assert R.p1_replay({0: f4e, 1: sel_rec()}, {0: replay(f4e), 1: replay()}, {0: vf, 1: verif()})['label'] == \
        'P1_REPLAY_VERIFIED'
    ch = replay()
    ch['estimator_unchanged'] = False
    assert R.p1_replay(rec, {0: ch, 1: replay()}, {v_: verif() for v_ in (0, 1)})['label'] == 'P1_ESTIMATOR_CHANGED'
    short = verif()
    del short['F4']
    assert R.p1_replay(rec, {v_: replay() for v_ in (0, 1)}, {0: short, 1: verif()})['label'] == 'INCOMPLETE'
    txt = replay()
    txt['F4']['rule'] = 'reworded'                                                       # text fields are not compared
    assert R.p1_replay(rec, {0: txt, 1: replay()}, {v_: verif() for v_ in (0, 1)})['label'] == 'P1_REPLAY_VERIFIED'
    miss = replay()
    del miss['nested_means']['PRC_TWIN']['auc_pair']
    assert R.p1_replay(rec, {0: miss, 1: replay()}, {v_: verif() for v_ in (0, 1)})['label'] == 'P1_NOT_REPRODUCED'
    assert R.p1_replay(rec, {0: replay()}, {v_: verif() for v_ in (0, 1)})['label'] == 'INCOMPLETE'


def main():
    t0 = time.process_time()
    res = []
    for t in TESTS:
        try:
            t()
            res.append({'test': t.__name__, 'result': 'PASS'})
        except Exception as ex:                                            # report every failure, then exit non-zero
            res.append({'test': t.__name__, 'result': 'FAIL', 'error': f'{type(ex).__name__}: {ex}'})
    npass = sum(r['result'] == 'PASS' for r in res)
    try:
        sha = R.load_rules()['_sha256']
    except (ValueError, KeyError, OSError) as ex:                         # a refused rule file fails the run, reported
        sha = f'REFUSED: {ex}'
    out = {'schema': R.SCHEMA, 'rules_sha256': sha, 'tests': res, 'passed': npass, 'total': len(res),
           'all_pass': npass == len(res) and not sha.startswith('REFUSED'), 'cpu_s': round(time.process_time() - t0, 3),
           'note': 'invented data only; no label read, no real-data computation'}
    print(json.dumps(out, indent=1))
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
