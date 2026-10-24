"""Verifier wave 2 (label-free): append the wave-2 entries to roles/verifier/CHECKS.json. The wave-1 file content is checked against
its recorded SHA-256 first; its entries are kept unchanged and the wave-2 entries (field "wave": 2) are appended after them.

    python3 .../roles/verifier/code/v2_assemble.py --utc <date -u> --wave1-sha256 <sha of the wave-1 CHECKS.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402
import v_assemble as A1  # noqa: E402  (wave-1 grouping helpers)


def rel(p):
    return str(Path(p).relative_to(V.ROOT))


def group_key2(cid):
    """Wave-2 grouping: utility / paired costs by release; grid elementary checks by bank and kind (registered or robustness cells)."""
    import re
    m = re.match(r'(G[WVAB])_(SAR1_[A-Z0-9_]+?)_S[01](R[1-4])k([0-3])_.*_(registered|nolookup)$', cid)
    if m:
        kind = {'GW': 'WINNER_PICK', 'GV': 'WINNER_VALIDATION_AUC', 'GA': 'WINNER_SCORE_AUC', 'GB': 'WINNER_BOOTSTRAP_DRAWS'}[m.group(1)]
        cells = 'REGISTERED_CELLS' if m.group(3) in ('R1', 'R2') and m.group(4) in ('0', '1') else 'ROBUSTNESS_CELLS'
        return f'C3_GRID_{kind}_{m.group(5).upper()}_{m.group(2)}_{cells}'
    m = re.match(r'GP_(SAR1_[A-Z0-9_]+?)_S[01](R[1-4])k([0-3])_', cid)
    if m:
        cells = 'REGISTERED_CELLS' if m.group(2) in ('R1', 'R2') and m.group(3) in ('0', '1') else 'ROBUSTNESS_CELLS'
        return f'C3_GRID_PAIRED_CELLS_AND_BOOTSTRAP_SE_{m.group(1)}_{cells}'
    m = re.match(r'G_(SAR1_[A-Z0-9_]+?)_v[01]_', cid)
    if m:
        return f'C3_GRID_MEMBER_WIRES_{m.group(1)}'
    return A1.group_key(cid)


def grouped2(ms, msp, unit, frozen):
    out, groups = [], OrderedDict()
    for c in ms['checks']:
        groups.setdefault(group_key2(c['id']), []).append(c)
    for g, cs in groups.items():
        diffs = [abs(float(c['record']) - float(c['recomputed'])) for c in cs
                 if isinstance(c['record'], (int, float)) and isinstance(c['recomputed'], (int, float)) and not isinstance(c['record'], bool)]
        fails = [c['id'] for c in cs if c['status'] != 'PASS']
        out.append({'id': 'W2_' + g, 'wave': 2, 'what': A1.what_of(g) + (' (robustness cells R3 / R4 and seeds k2 / k3)' if 'ROBUSTNESS' in g else ''),
                    'inputs_sha256': {**ms['inputs_sha256'], **frozen, rel(msp): V.fsha(msp)},
                    'recomputed': {'n_elementary': len(cs), 'n_fail': len(fails), 'failed_ids': fails[:20],
                                   'max_abs_difference': max(diffs) if diffs else None, 'elementary_values_in': rel(msp)},
                    'record': 'the record values named in each elementary check (EVAL_MS.json / GRID.json / BOOT*.npz)',
                    'status': 'PASS' if not fails else 'FAIL', 'tolerance': cs[0]['tolerance'], 'unit': unit})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--utc', required=True)
    ap.add_argument('--wave1-sha256', required=True)
    a = ap.parse_args()
    cp = V.VDIR / 'CHECKS.json'
    if V.fsha(cp) != a.wave1_sha256:
        raise PermissionError('CHECKS.json differs from the wave-1 file (append only to the recorded wave-1 content)')
    W = json.loads(cp.read_text())
    sys.path.insert(0, str(V.ROOT))
    from ohn import common as OC
    frozen = {}
    for v in (0, 1):
        fp = OC.admitted(f'admitted/bank/FROZEN_BANK_S{v}.npz')
        frozen[str(fp)] = V.fsha(fp)
    new = []
    for name, unit in (('ms2_fa_r2_fb_rob.json', 'SAR_VERIFY_3'), ('ms2_fa_r2_rob.json', 'SAR_VERIFY_5')):
        msp = V.WORK / name
        new += grouped2(json.loads(msp.read_text()), msp, unit, frozen)
    dcp = V.WORK / 'decide2.json'
    for c in json.loads(dcp.read_text())['checks']:
        new.append({**c, 'wave': 2, 'inputs_sha256': {**c['inputs_sha256'], rel(dcp): V.fsha(dcp)}, 'unit': 'none (label-free script v2_decide.py)'})
    rfp = V.WORK / 'refit2.json'
    rf = json.loads(rfp.read_text())
    for r in rf['refits']:
        ok = (r['score_vector_bitwise_equal_to_stored'] and r['score_auc_refit'] == r['score_auc_record']
              and r['val_auc_refit'] == r['val_auc_record'] and r['orientation'] == r['orientation_record'])
        new.append({'id': f"W2_C4_REFIT_{r['bank']}_{r['cell']}_{r['member']}_{r['view']}", 'wave': 2,
                    'what': f"winning reader {r['winner']} ({r['recipe']} on the {r['refit_view']} view) of {r['member']} {r['view']} {r['cell']} "
                            f"refitted from scratch (locked recipe code, frozen FIT part, reader seed {r['reader_seed']}); bitwise comparison",
                    'inputs_sha256': {**{k: v for k, v in rf['inputs_sha256'].items() if r['bank'] in k or 'grid_specs' in k}, rel(rfp): V.fsha(rfp)},
                    'recomputed': {'val_auc': r['val_auc_refit'], 'score_auc': r['score_auc_refit'], 'orientation': r['orientation'],
                                   'score_vector_bitwise_equal_to_stored': r['score_vector_bitwise_equal_to_stored'], 'max_abs_score_diff': r['max_abs_score_diff']},
                    'record': {'val_auc': r['val_auc_record'], 'score_auc': r['score_auc_record'], 'orientation': r['orientation_record']},
                    'status': 'PASS' if ok else 'FAIL', 'tolerance': 'bitwise', 'unit': 'SAR_VERIFY_4',
                    'note': f"frames {rf['frame_sizes']}, numpy.random.default_rng({rf['seed']}) indices {rf['sample_indices']}; {rf['interpreter']} {rf['versions']}"})
    eq = [x for x in rf['seed_effectiveness'] if x['equal_k0_k1']]
    new.append({'id': 'W2_C3_GRID_SEED_EFFECTIVENESS_FA_R2_MINIMUM_SET', 'wave': 2,
                'what': 'every seed-dependent recipe differs between k0 and k1 on every FA_R2 minimum-set wire, view, version and repetition',
                'inputs_sha256': {rel(rfp): V.fsha(rfp)}, 'recomputed': {'combinations': len(rf['seed_effectiveness']), 'equalities': eq},
                'record': {'unexplained_minimum_set': 0}, 'status': 'PASS' if not eq else 'FAIL', 'tolerance': 'bitwise', 'unit': 'SAR_VERIFY_4'})
    ctp = V.WORK / 'contrast2.json'
    ct = json.loads(ctp.read_text())
    for v, x in ct['versions'].items():
        new.append({'id': f'W2_S2_FA_R2_v{v}_guard_contrast_recomputed', 'wave': 2,
                    'what': f'FA_R2 v{v}: the out-of-fold guard contrast at rho* {x["rho_star"]} recomputed from the saved bag (five fold networks '
                            'and feature maps in MODEL.pkl) and MODEL_FIT occupation / SEX labels, without importing sar.fa: men\'s and women\'s '
                            'mean Brier change over SUCC, C, the five per-fold contrasts and SE_C_between',
                    'inputs_sha256': {**ct['inputs_sha256'], rel(ctp): V.fsha(ctp)},
                    'recomputed': {k: x[k] for k in ('mf_men', 'mf_women', 'C_mf', 'C_per_fold', 'se_C_between')}, 'record': x['record'],
                    'status': 'PASS' if x['max_abs_diff'] <= 1e-12 else 'FAIL', 'tolerance': 1e-12, 'unit': 'SAR_VERIFY_6',
                    'note': f"max abs difference {x['max_abs_diff']}; SE_C_household (linearized clustered SE) not recomputed"})
    # adjudication of the wave-1 income failure (the wave-1 FAIL stands; deviation 10 corrects the card's statement)
    ms1 = json.loads((V.WORK / 'ms_recomputed.json').read_text())
    mx = max(ms1['fb_income_vs_neural_raw'][v]['max_abs_diff'] for v in ('0', '1'))
    new.append({'id': 'W2_ADJ_S1_FB_R1_income_bitwise_START', 'wave': 2,
                'what': 'adjudication entry for wave-1 S1_FB_R1_income_bitwise_START (that FAIL stands): the corrected statement of deviation 10 '
                        '("equal to START\'s income up to 7e-8; identical decisions") against the verifier\'s measurement',
                'inputs_sha256': {rel(V.WORK / 'ms_recomputed.json'): V.fsha(V.WORK / 'ms_recomputed.json'),
                                  'results/pcrl_strong_anchor_release_v1/PROTOCOL_DEVIATIONS.md': V.fsha(V.RES / 'PROTOCOL_DEVIATIONS.md')},
                'recomputed': {'max_abs_diff': mx, 'argmax_equal': all(ms1['fb_income_vs_neural_raw'][v]['argmax_equal'] for v in ('0', '1'))},
                'record': {'deviation_10_bound': 7e-8}, 'status': 'PASS' if mx <= 7e-8 else 'FAIL', 'tolerance': 0, 'unit': 'SAR_VERIFY_1 (wave 1)'})
    n2f = [c['id'] for c in new if c['status'] != 'PASS']
    W['checks'] = W['checks'] + new
    W.setdefault('waves', {})
    W['waves']['1'] = {'utc': W['utc'], 'totals': W['totals'], 'failed': W['failed'], 'units': W['verifier_units'],
                       'sha256_of_wave1_file': a.wave1_sha256}
    el2 = sum(len(json.loads((V.WORK / n).read_text())['checks']) for n in ('ms2_fa_r2_fb_rob.json', 'ms2_fa_r2_rob.json'))
    el2f = sum(sum(c['status'] != 'PASS' for c in json.loads((V.WORK / n).read_text())['checks']) for n in ('ms2_fa_r2_fb_rob.json', 'ms2_fa_r2_rob.json'))
    W['waves']['2'] = {'utc': a.utc, 'totals': {'checks': len(new), 'pass': len(new) - len(n2f), 'fail': len(n2f),
                                                'elementary_model_select_checks': el2, 'elementary_model_select_fail': el2f},
                       'failed': n2f, 'units': ['SAR_VERIFY_3', 'SAR_VERIFY_4', 'SAR_VERIFY_5', 'SAR_VERIFY_6'],
                       'code_sha256': {rel(p): V.fsha(p) for p in sorted(HERE.glob('*.py'))}}
    allf = [c['id'] for c in W['checks'] if c['status'] != 'PASS']
    W['totals_all_waves'] = {'checks': len(W['checks']), 'pass': len(W['checks']) - len(allf), 'fail': len(allf), 'failed': allf}
    W['note_waves'] = ('the top-level totals / failed / utc / verifier_units / code_sha256 fields are wave 1\'s, unchanged; per-wave summaries '
                       'are under "waves"; entries are append-only (wave-1 entries unchanged, wave-2 entries carry "wave": 2)')
    V.jdump(cp, W)
    print(json.dumps({'wave2': W['waves']['2']['totals'], 'wave2_failed': n2f, 'all': W['totals_all_waves']['checks'],
                      'all_fail': W['totals_all_waves']['fail']}))


if __name__ == '__main__':
    main()
