"""Investigator A, R2 design: print the FA_TFRO_R1 fit records (MODEL_FIT / CALIBRATION aggregates) and list the R1 decision record
(no label read; nothing computed from MODEL_SELECT).
Run: python3 results/pcrl_strong_anchor_release_v1/roles/investigator_a/r2_read_records.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
R = ROOT / 'results' / 'pcrl_strong_anchor_release_v1'


def f(x, k=5):
    return None if x is None else round(float(x), k)


for v in (0, 1):
    p = R / 'fa' / f'FIT_R1_V{v}.json'
    if not p.is_file():
        print('missing', p); continue
    r = json.loads(p.read_text())
    s = r['selection']
    print(f'==== V{v}: rho* {s["rho_star"]} status {r["cand"]["status"]} cpu {f(r.get("cpu_s"), 1)}')
    for k, row in s['per_rho'].items():
        g = row['guard_estimate']
        sp = row['cal_spend_over_SUCC']
        print(f'  rho {k}: passes {row["passes"]} a {row["part_a"]} b {row["part_b"]} lambda {f(row["lambda"], 4)}')
        print(f'     CAL occ_nll {f(sp["occ_nll"]["estimate"])} ({f(sp["occ_nll"]["se"])}) occ_brier {f(sp["occ_brier"]["estimate"])} '
              f'({f(sp["occ_brier"]["se"])}) occ_acc {f(sp["occ_acc"]["estimate"])} ({f(sp["occ_acc"]["se"])})')
        print(f'     guard: L {f(g["L_cal"])} (se {f(g["se_L"])}), C {f(g["C_mf"])} (se {f(g["se_C"])}), p {f(g["p_men"], 4)}, '
              f'MF men {f(g["mf_men"])}, MF women {f(g["mf_women"])}, D_men {[f(x) for x in g["men"]]}, D_women {[f(x) for x in g["women"]]}')
    c = r.get('ctrl') or {}
    print('  cand cal spend', {k: f(v_) for k, v_ in (r['cand'].get('cal_spend_over_SUCC') or {}).items() if k != 'all'})
    print('  ctrl lo/hi/flags', c.get('lo'), c.get('hi'), c.get('flags'), 'saved', json.dumps(c.get('saved'), default=str)[:400])
    if c.get('path'):
        print('  ctrl path spend_over_start:', {k: f(x['spend_over_start'], 4) for k, x in c['path'].items()})
    mf = r.get('mf_reader_check', {}).get('auc', {})
    print('  MF readers (descriptive, biased):', {k: {kk: f(vv, 4) for kk, vv in d.items()} for k, d in mf.items()})
    print('  saved:', sorted(r.get('saved', {})))
d = R / 'claims' / 'FA_R1'
print('claims dir:', sorted(x.name for x in d.iterdir()) if d.is_dir() else 'none')
p = d / 'DECISIONS.json'
if p.is_file():
    print(json.dumps(json.loads(p.read_text()), default=str)[:2500])
