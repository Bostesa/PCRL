"""Write the MODEL_SELECT evaluation, registered-grid, rich-reader and claims lanes from the committed plans (label-free; coordinator)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # the repository root (results/<campaign>/roles/coordinator/)
RES = ROOT / 'results/pcrl_input_and_pair_release_v1'
LANES = RES / 'lanes'
REL = 'results/pcrl_input_and_pair_release_v1'


def plan(fam):
    return json.loads((RES / 'claims' / f'PLAN_{fam}_REG.json').read_text())


def main():
    plans = {f: plan(f) for f in ('IA', 'IB')}
    roles = json.loads((RES / 'paths' / 'SHARED_ROLES.json').read_text())['roles']
    eval_units = []
    # the shared twin: utility and the twin's paired costs against nothing (references only)
    eval_units.append({"unit": "IPR_EVAL_TWIN_V1", "stage": "evaluate", "rss_gib": 2.5,
                       "argv": ["PY", "-m", "ipr.evaluate", "many", "--release", "CTL/CTL_TWIN", "--compare", "W"]})
    for fam, p in plans.items():
        if not p.get('cand'):
            continue
        rel = [m for k, m in p['members'].items() if not m.startswith('ROLE:') and k != 'CTL_TWIN']
        abl = [f"{fam}/{a}" for a in p.get('ablations', [])]
        comps = sorted(set(['CTL/CTL_TWIN'] + [m for m in p['members'].values() if m != f"{fam}/{p['cand']}"] + abl + ['W']))
        argv = ["PY", "-m", "ipr.evaluate", "many"]
        for r in sorted(set(rel + abl)):
            argv += ["--release", r]
        for c in comps:
            argv += ["--compare", c]
        eval_units.append({"unit": f"IPR_EVAL_{fam}_V1", "stage": "evaluate", "rss_gib": 2.5, "argv": argv})
    argv = ["PY", "-m", "ipr.evaluate", "roles"]
    for name in sorted(roles):
        argv += ["--role", name]
    argv += ["--compare", "CTL/CTL_TWIN"]
    eval_units.append({"unit": "IPR_EVAL_ROLES_V1", "stage": "evaluate", "rss_gib": 2.5, "argv": argv})
    ref = json.loads((LANES / 'REF_EVAL.json').read_text())
    (LANES / 'EVAL_ALL.json').write_text(json.dumps(eval_units + ref, indent=1) + '\n')
    for v in (0, 1):
        (LANES / f'GRID_MAIN_V{v}.json').write_text(json.dumps([
            {"unit": f"IPR_GRID_MAIN_V{v}", "stage": "readers", "rss_gib": 3.0, "max_wall_s": 10800,
             "argv": ["TORCH", "-m", "ipr.devreaders", "run", "--spec", f"{REL}/grid_specs/IPR1_MAIN.json", "--seed", str(v)]}], indent=1) + '\n')
    (LANES / 'RICH_MAIN.json').write_text(json.dumps([
        {"unit": "IPR_RICH_MAIN_V1", "stage": "readers", "rss_gib": 2.0,
         "argv": ["PY", "-m", "ipr.richreaders", "score", "--spec", f"{REL}/rich/RICH_SPEC_IPR1_MAIN.json"]}], indent=1) + '\n')
    (LANES / 'CLAIMS_REG.json').write_text(json.dumps([
        {"unit": f"IPR_CLAIMS_{fam}_REG_V1", "stage": "claims", "rss_gib": 1.5,
         "argv": ["PY", "-m", "ipr.claims", "run", "--plan", f"{REL}/claims/PLAN_{fam}_REG.json", "--bank", "IPR1_MAIN",
                  "--out", f"{REL}/claims/{fam}_REG"]} for fam in plans if plans[fam].get('cand')], indent=1) + '\n')
    print('lanes written:', [u['unit'] for u in eval_units])


if __name__ == '__main__':
    main()
