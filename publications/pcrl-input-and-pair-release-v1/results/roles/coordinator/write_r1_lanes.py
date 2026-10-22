"""Write the lanes of IA revision R1 (card IA_OSG_R1; label-free; coordinator).

    python3 write_r1_lanes.py fit     # FIT_R1_V0 / FIT_R1_V1 (before the fits)
    python3 write_r1_lanes.py read    # EVAL_R1, GRID_R1_V0 / V1, RICH_R1, CLAIMS_R1 from claims/PLAN_IA_R1.json (after `ipr.plan revision`)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # the repository root (results/<campaign>/roles/coordinator/)
RES = ROOT / 'results/pcrl_input_and_pair_release_v1'
LANES = RES / 'lanes'
REL = 'results/pcrl_input_and_pair_release_v1'
CARD = f'{REL}/cards/IA_OSG_R1.md'
BANK = 'IPR1_R1'


def write(name, units):
    p = LANES / f'{name}.json'
    if p.exists():
        raise SystemExit(f'{p.name} exists (lanes are write-once)')
    p.write_text(json.dumps(units, indent=1) + '\n')
    print('lane written:', p.name, [u['unit'] for u in units])


def fit():
    for v in (0, 1):
        write(f'FIT_R1_V{v}', [{"unit": f"IPR_IA_R1_FIT_V{v}", "stage": "fit", "rss_gib": 3.0, "max_wall_s": 7200,
                                "argv": ["PY", "-m", "ipr.ia.osg", "fit", "--version", str(v), "--card", CARD]}])


def read():
    p = json.loads((RES / 'claims' / 'PLAN_IA_R1.json').read_text())
    if not p.get('cand'):
        raise SystemExit('no candidate: the protection comparison is not run')
    rel = sorted(set(m for k, m in p['members'].items() if k != 'CTL_TWIN') | {f"IA/{a}" for a in p.get('ablations', [])})
    comps = sorted(set(['CTL/CTL_TWIN', 'IA/IA_SFG', 'W'] + [m for m in p['members'].values() if m != f"IA/{p['cand']}"]
                       + [f"IA/{a}" for a in p.get('ablations', [])]))
    argv = ["PY", "-m", "ipr.evaluate", "many"]
    for r in rel:
        argv += ["--release", r]
    for c in comps:
        argv += ["--compare", c]
    write('EVAL_R1', [{"unit": "IPR_EVAL_IA_R1_V1", "stage": "evaluate", "rss_gib": 2.5, "argv": argv}])
    for v in (0, 1):
        write(f'GRID_R1_V{v}', [{"unit": f"IPR_GRID_R1_V{v}", "stage": "readers", "rss_gib": 3.0, "max_wall_s": 10800,
                                 "argv": ["TORCH", "-m", "ipr.devreaders", "run", "--spec", f"{REL}/grid_specs/{BANK}.json", "--seed", str(v)]}])
    write('RICH_R1', [{"unit": "IPR_RICH_R1_V1", "stage": "readers", "rss_gib": 2.0,
                       "argv": ["PY", "-m", "ipr.richreaders", "score", "--spec", f"{REL}/rich/RICH_SPEC_{BANK}.json"]}])
    write('CLAIMS_R1', [{"unit": "IPR_CLAIMS_IA_R1_V1", "stage": "claims", "rss_gib": 1.5,
                         "argv": ["PY", "-m", "ipr.claims", "run", "--plan", f"{REL}/claims/PLAN_IA_R1.json", "--bank", BANK,
                                  "--out", f"{REL}/claims/IA_R1"]}])


if __name__ == '__main__':
    {'fit': fit, 'read': read}[sys.argv[1]]()
