"""Coordinator's in-memory preview of a reading's decisions (ddr.claims logic; label-free; writes nothing). The official record is
written by the queued `ddr.claims run` unit.

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/peek_decisions.py PLAN_DB_REG.json DDR1_MAIN
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ddr import claims as CL      # noqa: E402
from ddr import rules as RU       # noqa: E402
from nrm import rules as NR       # noqa: E402

R = Path(__file__).resolve().parents[2]
plan = json.loads((R / 'claims' / sys.argv[1]).read_text())
bank = sys.argv[2]
grid = CL.load_grid(bank)
seeds, boot = CL.seed_records()
minimum = [m for m in [plan['cand'], plan['twin']] + [x for k in ('SBC', 'SPEND') for x in (plan['paths'].get(k) or [])] if m]
gv = RU.grid_validity(seeds, CL.seed_checks(grid), minimum, CL.distinct_inputs(bank), boot_seeds=boot)
rec = CL.build(plan, grid, bank)
rec['grid_valid'] = gv['effective']
dec = RU.decisions(rec)
print('grid validity', gv['label'], gv['errors'][:2])
for k in ('D1', 'D2', 'D3', 'D4', 'outcome_class'):
    print(k, (dec.get(k) or {}).get('label'))
d2 = dec['D2']
print('passed', d2.get('passed'), 'flags', d2.get('flags'))
for c, x in (d2.get('comparisons') or {}).items():
    fr = x.get('frontier') or {}
    print(f'  {c}: pass={x.get("pass")} label={x.get("label")} reason={x.get("reason")} point={x.get("point")} '
          f'benefit={fr.get("benefit_mean")} se={fr.get("benefit_se")} more={fr.get("secondary_more_spending")}')
cand = plan['cand']
for comp in [plan['twin']] + [x for k in ('SBC', 'SPEND') for x in plan['paths'][k]] + plan.get('beside', []) + ['NEURAL_RAW', 'H_LJ']:
    e = grid['paired'].get(cand, {}).get(comp)
    if e:
        a = NR.aggregate(e['pair'])
        c = NR.comparison(e)
        print(f'  {cand} vs {comp}: mean {a["mean"]:+.4f} se_agg {a["se_agg"]:.4f} V0 {a["per_version"][0]["mean"]:+.4f} '
              f'V1 {a["per_version"][1]["mean"]:+.4f} screen {c["benefit"]["label"]} inc {c["recipient"]["income"]["label"]} '
              f'occ {c["recipient"]["occupation"]["label"]}')
