"""Compact digest of every reading's decision record (coordinator; label-free; for the documents).

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/digest.py DA_REG DB_REG DB_R1 DA_R1
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
for name in sys.argv[1:]:
    d = json.loads((R / 'claims' / name / 'DECISIONS.json').read_text())['decisions']
    d1, d2 = d['D1'], d['D2']
    print(f"===== {name}: class {d['outcome_class'].get('label')}; tag {d.get('adaptive_tag')}")
    print(f"  D1 {d1['label']}: cand {d1.get('cand')} twin {d1.get('twin')} reserve {d1.get('reserve')}")
    tw = d2['comparisons'].get('TWIN') or {}
    print(f"  TWIN: benefit {tw.get('benefit')} mean {tw.get('benefit_mean')} se {tw.get('benefit_se')} recip {tw.get('recipient')} "
          f"gate {tw.get('recipient_gate_pass')} targeting {tw.get('targeting_pass')}")
    for k in ('SBC_FRONTIER', 'SPEND_FRONTIER'):
        c = d2['comparisons'].get(k) or {}
        fr = c.get('frontier') or {}
        print(f"  {k}: {c.get('label')} Q*={fr.get('bracketing_control')} benefit {c.get('benefit_mean')} se {c.get('benefit_se')} "
              f"costs {json.dumps({q: round(x['value'], 5) for q, x in (fr.get('primary_cost_differences') or {}).items()})} "
              f"recip {c.get('recipient_vs_bracket')} targeting {c.get('targeting_pass')} reason {c.get('reason')}")
    print(f"  D2 {d2['label']} flags {d2.get('flags')}")
    print(f"  D3 {d['D3'].get('label')} failing {d['D3'].get('failing')}; D4 {d['D4'].get('label')} {d['D4'].get('level')}")
    det = d['D4'].get('detail') or d['D4']
    print('  D4 detail', json.dumps(det, default=str)[:700])
