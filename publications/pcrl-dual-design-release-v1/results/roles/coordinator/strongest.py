"""Strongest feasible nonzero release per family (RULES.md 8, `ddr.rules.strongest_nonzero`, unchanged), over every reading of the
family whose decisions exist (the registered CAND and each revision's CAND), from the saved utilities and each reading's own grid
(pair entry against the shared twin). A recipient-split strength (DB R1's (alpha_I, alpha_O) per version) enters the frozen
rule as alpha_I + alpha_O, the quantity its own selection maximized (the frozen helpers read a number or a per-version number).
Descriptive: chosen on registered MODEL_SELECT cells; it never changes D1-D6. It names the
release whose D2R (matched classifier refits) and ablation the campaign must report.

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/strongest.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ddr import common as C          # noqa: E402
from ddr import claims as CL         # noqa: E402
from ddr import plan as PL           # noqa: E402
from ddr import rules as RU          # noqa: E402

READINGS = {'DA': [('DA_REG', 'DDR1_MAIN'), ('DA_R1', 'DDR1_DA_R1')],
            'DB': [('DB_REG', 'DDR1_MAIN'), ('DB_R1', 'DDR1_R1')]}
out = {}
for fam, readings in READINGS.items():
    configs, seen = [], {}
    for reading, bank in readings:
        if not (C.RESULTS / 'claims' / reading / 'DECISIONS.json').is_file():
            continue
        plan = json.loads((C.RESULTS / 'claims' / f'PLAN_{reading}.json').read_text())
        cand, twin = plan['cand'], plan['twin']
        if not cand:
            continue
        w = PL.wires(fam, cand)
        key = tuple(w[v] for v in C.VERSIONS)
        alias = 'DA_TWIN' if key == tuple(PL.wires('DA', 'DA_TWIN')[v] for v in C.VERSIONS) else seen.get(key)
        seen.setdefault(key, cand)
        ev = CL.evals_of(plan['members'][cand]) or {}
        g = CL.load_grid(bank)
        st = (plan.get('strengths') or [{}])[0].get('strength')
        if isinstance(st, dict):          # a recipient-split strength (alpha_I, alpha_O) enters the frozen rule as alpha_I + alpha_O,
            st = {v: (sum(x) if isinstance(x, (list, tuple)) else x) for v, x in st.items()}   # the sum its selection maximized
        configs.append({'config_id': cand, 'strength': st,
                        'revision': int(plan.get('revision') or 0), 'twin_id': twin, 'reading': reading, 'bank': bank,
                        'evals': {v: e['utility'] for v, e in ev.items()},
                        'pair_vs_twin': ((g['paired'].get(cand) or {}).get(twin) or {}).get('pair'), 'alias_of': alias})
    s = RU.strongest_nonzero(configs)
    s['readings'] = [{'config_id': c['config_id'], 'reading': c['reading'], 'bank': c['bank'], 'revision': c['revision']}
                     for c in configs]
    out[fam] = s
    print(fam, s['label'], s.get('selected'), s.get('x'), [(r['config_id'], r.get('status'), r.get('x')) for r in s['records']])
C.write_json(C.RESULTS / 'claims' / 'STRONGEST.json', {'schema': 'DDR1-strongest-v1', 'utc': C.now(), 'families': out})
