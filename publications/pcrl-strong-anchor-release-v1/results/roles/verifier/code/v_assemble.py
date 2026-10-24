"""Verifier (label-free): assemble roles/verifier/CHECKS.json from the verifier's work files (work/ms_recomputed.json from queue unit
SAR_VERIFY_1, work/refit.json from SAR_VERIFY_2, work/decide.json from v_decide.py). Elementary MODEL_SELECT checks are grouped
(per release / comparison / bank and kind) with their counts, failures and largest absolute differences; every elementary value
stays in the work file whose SHA-256 each grouped entry cites.

    python3 results/pcrl_strong_anchor_release_v1/roles/verifier/code/v_assemble.py --wave 1 --utc <date -u>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402


def rel(p):
    return str(Path(p).relative_to(V.ROOT))


def group_key(cid):
    m = re.match(r'U_REF_([A-Za-z0-9_]+?)_v[01]_', cid)
    if m:
        return 'C1_UTILITY_REFERENCE_' + m.group(1)
    m = re.match(r'U_([A-Za-z0-9_]+?)_v[01]_', cid)
    if m:
        return 'C1_UTILITY_' + m.group(1)
    m = re.match(r'PC_([A-Za-z0-9_]+?)_vs_([A-Za-z0-9_]+?)_v[01]_', cid)
    if m:
        return f'C1_PAIRED_COSTS_{m.group(1)}_vs_{m.group(2)}'
    m = re.match(r'G_(SAR1_F[AB]_R1)_v[01]_', cid)
    if m:
        return f'C3_GRID_MEMBER_WIRES_{m.group(1)}'
    m = re.match(r'(G[WVAB])_(SAR1_F[AB]_R1)_S[01]R[12]k[01]_.*_(registered|nolookup)$', cid)
    if m:
        kind = {'GW': 'WINNER_PICK', 'GV': 'WINNER_VALIDATION_AUC', 'GA': 'WINNER_SCORE_AUC', 'GB': 'WINNER_BOOTSTRAP_DRAWS'}[m.group(1)]
        return f'C3_GRID_{kind}_{m.group(3).upper()}_{m.group(2)}'
    m = re.match(r'GP_(SAR1_F[AB]_R1)_', cid)
    if m:
        return f'C3_GRID_PAIRED_CELLS_AND_BOOTSTRAP_SE_{m.group(1)}'
    return 'OTHER_' + cid


WHAT = {
    'C1_UTILITY_': 'the 12 task clauses (accuracy loss, NLL excess, summed Brier excess vs U0_S<v> / Ucal_S<v>) and the 2 within-SEX '
                   'occupation guards per version, values and household-clustered / linearized SEs, recomputed with the verifier\'s '
                   'metric code from the saved MODEL_SELECT arrays and labels, against the EVAL_MS record (and the evaluated wire hashes)',
    'C1_PAIRED_COSTS_': 'paired per-person costs (accuracy loss, NLL, summed Brier, men\'s / women\'s Brier, total NLL) with clustered '
                        'SEs, both versions, against the EVAL_MS record',
    'C3_GRID_MEMBER_WIRES_': 'the bank members\' MODEL_SELECT wires (float64 vectors + argmax) equal the grid record\'s wire hashes',
    'C3_GRID_WINNER_PICK_': 'winner re-picked from the stored per-recipe VALIDATION AUCs (max, exact ties to the earlier candidate; pair: '
                            'pair readers then ignore-occupation then ignore-income per recipe group)',
    'C3_GRID_WINNER_VALIDATION_AUC_': 'the winner\'s VALIDATION AUC in the grid record equals the stored reader record',
    'C3_GRID_WINNER_SCORE_AUC_': 'the winner\'s SCORE AUC recomputed (verifier AUC) from the stored oriented SCORE scores and the SCORE '
                                 'SEX labels',
    'C3_GRID_WINNER_BOOTSTRAP_DRAWS_': 'the 400 household-bootstrap AUC draws recomputed with the verifier\'s own weights (seed 2026102400 '
                                       '+ 100 v + repetition index) against BOOT.npz / BOOT_EXTRAS.npz (max abs difference)',
    'C3_GRID_PAIRED_CELLS_AND_BOOTSTRAP_SE_': 'per-cell paired differences (pair: comparator - candidate; views: candidate - comparator) '
                                              'and their bootstrap SEs (SD of paired draws) against GRID.json "paired"',
}


def what_of(g):
    for k, w in WHAT.items():
        if g.startswith(k):
            return w
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wave', required=True)
    ap.add_argument('--utc', required=True)
    a = ap.parse_args()
    msp, dcp, rfp = V.WORK / 'ms_recomputed.json', V.WORK / 'decide.json', V.WORK / 'refit.json'
    ms, dc, rf = (json.loads(p.read_text()) for p in (msp, dcp, rfp))
    sha = {rel(p): V.fsha(p) for p in (msp, dcp, rfp)}
    checks = []
    sys.path.insert(0, str(V.ROOT))
    from ohn import common as OC                       # label-free: the admitted frozen-bank paths (hash re-checked on access)
    frozen = {}
    for v in (0, 1):
        fp = OC.admitted(f'admitted/bank/FROZEN_BANK_S{v}.npz')
        frozen[str(fp)] = V.fsha(fp)
    la = [json.loads(x) for x in (V.PRIV / 'access' / 'LABEL_ACCESS.jsonl').read_text().splitlines() if x.strip()]
    mine_reads = [{k: e.get(k) for k in ('utc', 'unit_id', 'role', 'fields', 'operation', 'entry_point', 'people', 'households', 'rows_sha256')}
                  for e in la if e.get('operation') == 'verify']
    groups = OrderedDict()
    for c in ms['checks']:
        groups.setdefault(group_key(c['id']), []).append(c)
    for g, cs in groups.items():
        diffs = [abs(float(c['record']) - float(c['recomputed'])) for c in cs
                 if isinstance(c['record'], (int, float)) and isinstance(c['recomputed'], (int, float))
                 and not isinstance(c['record'], bool)]
        fails = [c['id'] for c in cs if c['status'] != 'PASS']
        tag = g.split('_', 3)[-1] if g.startswith('C1_') else g
        inputs = {**ms['inputs_sha256'], **frozen, rel(msp): sha[rel(msp)]}
        rec_entry = {'n_elementary': len(cs), 'n_fail': len(fails), 'failed_ids': fails[:20], 'max_abs_difference': max(diffs) if diffs else None,
                     'elementary_values_in': rel(msp)}
        if g.startswith('C1_UTILITY_'):
            rec_entry['examples'] = [{k: c[k] for k in ('id', 'record', 'recomputed')} for c in cs if c['id'].endswith('_value')][:28]
        checks.append({'id': g, 'what': what_of(g), 'inputs_sha256': inputs, 'recomputed': rec_entry,
                       'record': 'the record values named in each elementary check (EVAL_MS.json / GRID.json / BOOT*.npz)',
                       'status': 'PASS' if not fails else 'FAIL',
                       'tolerance': cs[0]['tolerance'] if all(c['tolerance'] == cs[0]['tolerance'] for c in cs) else 'mixed',
                       'unit': 'SAR_VERIFY_1'})
    for c in dc['checks']:
        checks.append({**c, 'inputs_sha256': {**c['inputs_sha256'], rel(dcp): sha[rel(dcp)]}, 'unit': 'none (label-free script v_decide.py)'})
    for r in rf['refits']:
        ok = (r['score_vector_bitwise_equal_to_stored'] and r['score_auc_refit'] == r['score_auc_record']
              and r['val_auc_refit'] == r['val_auc_record'] and r['orientation'] == r['orientation_record'])
        checks.append({'id': f"C4_REFIT_{r['bank']}_{r['cell']}_{r['member']}_{r['view']}",
                       'what': f"winning reader {r['winner']} ({r['recipe']} on the {r['refit_view']} view) of {r['member']} {r['view']} "
                               f"{r['cell']} refitted from scratch with the locked recipe code on the frozen FIT part, reader seed "
                               f"{r['reader_seed']}; orientation on VALIDATION; SCORE AUC (bitwise)",
                       'inputs_sha256': {**{k: v for k, v in rf['inputs_sha256'].items() if r['bank'] in k or 'grid_specs' in k}, rel(rfp): sha[rel(rfp)]},
                       'recomputed': {'val_auc': r['val_auc_refit'], 'score_auc': r['score_auc_refit'], 'orientation': r['orientation'],
                                      'score_vector_bitwise_equal_to_stored': r['score_vector_bitwise_equal_to_stored'],
                                      'max_abs_score_diff': r['max_abs_score_diff']},
                       'record': {'val_auc': r['val_auc_record'], 'score_auc': r['score_auc_record'], 'orientation': r['orientation_record']},
                       'status': 'PASS' if ok else 'FAIL', 'tolerance': 'bitwise', 'unit': 'SAR_VERIFY_2',
                       'note': f"sampling frame {rf['frame_size']} winners, numpy.random.default_rng({rf['seed']}).choice(size=6, replace=False); "
                               f"interpreter {rf['interpreter']} {rf['versions']}"})
    pm = rf['p_men_model_fit']
    checks.append({'id': 'S2_FA_R1_p_men', 'what': 'the MODEL_FIT share of men p used by FA_TFRO_R1\'s guard estimator (logged MODEL_FIT read)',
                   'inputs_sha256': {rel(rfp): sha[rel(rfp)], 'results/pcrl_strong_anchor_release_v1/fa/FIT_R1_V0.json': V.fsha(V.RES / 'fa' / 'FIT_R1_V0.json')},
                   'recomputed': pm, 'record': 0.5103, 'status': 'PASS' if abs(pm - 0.5103) <= 1e-12 else 'FAIL', 'tolerance': 1e-12,
                   'unit': 'SAR_VERIFY_2'})
    se = rf['seed_effectiveness']
    eq = [x for x in se if x['equal_k0_k1']]
    checks.append({'id': 'C3_GRID_SEED_EFFECTIVENESS_MINIMUM_SET', 'what': 'on every minimum-set wire, view, version and repetition of '
                   'SAR1_FB_R1 / SAR1_FA_R1, every seed-dependent recipe (ET5, ET25, MLP64, MLP6432, OUT_MLP, LOG_MLP) gives different stored '
                   'SCORE scores under k0 and k1 (no equality to explain on the minimum set)',
                   'inputs_sha256': {rel(rfp): sha[rel(rfp)]}, 'recomputed': {'combinations': len(se), 'equalities': eq},
                   'record': {'unexplained_minimum_set': 0, 'grid_validity': 'READER_SEEDS_EFFECTIVE'},
                   'status': 'PASS' if not eq else 'FAIL', 'tolerance': 'bitwise', 'unit': 'SAR_VERIFY_2'})
    n_fail = sum(c['status'] != 'PASS' for c in checks)
    out = {'schema': 'SAR1-verifier-checks-v1', 'wave': int(a.wave), 'utc': a.utc, 'verifier_units': ['SAR_VERIFY_1', 'SAR_VERIFY_2'],
           'totals': {'checks': len(checks), 'pass': len(checks) - n_fail, 'fail': n_fail,
                      'elementary_model_select_checks': len(ms['checks']),
                      'elementary_model_select_fail': sum(c['status'] != 'PASS' for c in ms['checks'])},
           'failed': [c['id'] for c in checks if c['status'] != 'PASS'],
           'verifier_label_reads': mine_reads,
           'code_sha256': {rel(p): V.fsha(p) for p in sorted(HERE.glob('*.py'))},
           'checks': checks,
           'note': 'a failed check is preserved, never relaxed; an adjudication, if any, is a separate later entry'}
    V.jdump(V.VDIR / 'CHECKS.json', out)
    print(json.dumps(out['totals']), json.dumps(out['failed']))


if __name__ == '__main__':
    main()
