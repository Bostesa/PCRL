"""Investigator A, R1 design: print the diagnosis inputs from the committed FA_TFR_v1 fit records only (no label read, no model fit).
Run: python3 results/pcrl_strong_anchor_release_v1/roles/investigator_a/r1_diagnose_records.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FA = ROOT / 'results' / 'pcrl_strong_anchor_release_v1' / 'fa'


def fmt(x):
    return f"{x['estimate']:+.5f} (se {x['se']:.5f}; bound {x['estimate'] + 2 * x['se']:+.5f})"


for v in (0, 1):
    r = json.loads((FA / f'FIT_V{v}.json').read_text())
    print(f'==== V{v}: keys {sorted(r)}')
    print(' cand', r.get('cand'), '| stop', r.get('stop'), '| cpu', round(r.get('cpu_s', 0), 1))
    s = r['selection']
    print(' rho_star', s.get('rho_star'), '| per_rho', list(s.get('per_rho', {})))
    for k, row in s.get('per_rho', {}).items():
        print(f'  rho {k}: meets_cal {row["meets_cal"]} meets_guard {row["meets_guard"]} guard_ref_s {row["guard_reference_s"]} '
              f'lo {row["bracket_lo"]} hi {row["bracket_hi"]} flags {row["bracket_flags"]}')
        for m, x in row['cal_spend_over_SUCC'].items():
            print('     CAL', m.ljust(16), fmt(x))
        for m, x in row['mf_guard_spend_over_reference'].items():
            print('     MF guard vs ref', m.ljust(16), fmt(x))
    print(' MF out-of-fold spends over SUCC (biased: SUCC in-sample on MODEL_FIT), every rho:')
    for k, d in s.get('mf_spend_over_SUCC_descriptive_biased', {}).items():
        print(f'  rho {k}: ' + ', '.join(f'{m} {x["estimate"]:+.4f}' for m, x in d.items()))
    for key in ('crossfit', 'task_crossfit'):
        if key in s:
            print(' ', key, {kk: vv for kk, vv in s[key].items() if kk != 'fold_tag'})
    other = {k: r[k] for k in r if k not in ('selection', 'cand', 'stop', 'cpu_s', 'saved', 'frozen_sha256', 'task_cache_sha256')}
    print(' other:', json.dumps(other, default=str)[:1500])
