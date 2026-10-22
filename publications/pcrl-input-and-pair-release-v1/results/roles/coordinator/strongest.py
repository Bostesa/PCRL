"""Strongest feasible nonzero release per family (RULES.md 8, `ipr.rules.strongest_nonzero`, unchanged), over every reading of the
family whose decisions exist (the registered CAND and each revision's CAND), from the saved utilities and each reading's own grid (pair
entry against the shared twin). Descriptive: chosen on registered MODEL_SELECT cells; it never changes D1-D6. It names the release
whose D2R (matched classifier refits) and ablation the campaign must report.

    python3 results/pcrl_input_and_pair_release_v1/roles/coordinator/strongest.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ipr import common as C          # noqa: E402
from ipr import claims as CL         # noqa: E402
from ipr import plan as PL           # noqa: E402
from ipr import rules as RU          # noqa: E402

READINGS = {'IA': [('IA_REG', 'IPR1_MAIN')], 'IB': [('IB_REG', 'IPR1_MAIN')]}
extra = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
for fam, more in extra.items():
    READINGS.setdefault(fam, []).extend([tuple(x) for x in more])
out = {}
for fam, readings in READINGS.items():
    configs, seen = [], {}
    tw = PL.member_wires('CTL/CTL_TWIN', {})
    for reading, bank in readings:
        if not (C.RESULTS / 'claims' / reading / 'DECISIONS.json').is_file():
            continue
        plan = json.loads((C.RESULTS / 'claims' / f'PLAN_{reading}.json').read_text())
        cand, twin = plan['cand'], plan['twin']
        if not cand:
            continue
        w = PL.member_wires(plan['members'][cand], {})
        key = tuple(w[v] for v in C.VERSIONS)
        alias = 'CTL_TWIN' if key == tuple(tw[v] for v in C.VERSIONS) else seen.get(key)
        seen.setdefault(key, cand)
        ev = CL.evals_of(plan['members'][cand]) or {}
        g = CL.load_grid(bank)
        configs.append({'config_id': cand, 'strength': (plan.get('strengths') or [{}])[0].get('strength'),
                        'revision': int(plan.get('revision') or 0), 'twin_id': twin, 'reading': reading, 'bank': bank,
                        'evals': {v: e['utility'] for v, e in ev.items()},
                        'pair_vs_twin': ((g['paired'].get(cand) or {}).get(twin) or {}).get('pair'), 'alias_of': alias})
    s = RU.strongest_nonzero(configs)
    s['readings'] = [{'config_id': c['config_id'], 'reading': c['reading'], 'bank': c['bank'], 'revision': c['revision']}
                     for c in configs]
    out[fam] = s
    print(fam, s['label'], s.get('selected'), s.get('x'), [(r['config_id'], r.get('status'), r.get('x')) for r in s['records']])
C.write_json(C.RESULTS / 'claims' / 'STRONGEST.json', {'schema': 'IPR1-strongest-v1', 'utc': C.now(), 'families': out})
