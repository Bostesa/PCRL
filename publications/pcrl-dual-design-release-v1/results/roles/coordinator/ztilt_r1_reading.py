"""Declared reading of card DB_ZTILT_R1 (coordinator; aggregate grid outputs only): DB_PTC_R1 against its spend-matched release-only
control DB_ZTILT_R1 on the registered cells (the card's rule) and, beside it, on the refit cells with the inherited robustness
screen (nrm.rules.robustness). Writes claims/ZTILT_R1_READING.json.

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/ztilt_r1_reading.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ddr import common as C          # noqa: E402
from ddr import claims as CL         # noqa: E402
from nrm import rules as NR          # noqa: E402

g = CL.load_grid('DDR1_R1_ZTILT')
out = {}
for comp in ('DB_ZTILT_R1', 'DA_TWIN'):
    e = g['paired']['DB_PTC_R1'][comp]
    reg, rob = NR.benefit_screen(e['pair']), NR.robustness(e)
    ra = reg['aggregate']
    out[comp] = {'registered': {'label': reg['label'], 'mean': ra['mean'], 'se_agg': ra['se_agg'],
                                'lower_2se': ra['mean'] - 2 * ra['se_agg'],
                                'version_means': {str(v): ra['per_version'][v]['mean'] for v in C.VERSIONS}},
                 'robustness': {'label': rob['label'], 'robustness_cells': rob['robustness_cells']['aggregate'],
                                'all_cells': {'label': rob['all_cells']['label'], **rob['all_cells']['aggregate']}}}
z = out['DB_ZTILT_R1']['registered']
declared = ('NOT_RESOLVED_BELOW_ZTILT: the gain is not separated from release-only information (card DB_ZTILT_R1: PRC\'s failure '
            'repeated)' if z['lower_2se'] <= 0 or min(z['version_means'].values()) <= 0 else
            'RESOLVED_BELOW_ZTILT: the gain needs propensity information beyond the release (no removal guarantee)')
spend = {}
for name in ('DB_PTC_R1', 'DB_ZTILT_R1'):
    r = json.loads((C.RUNS / 'DB' / name / 'EVAL_MS.json').read_text())['versions']
    spend[name] = [float(r[v]['paired_costs']['DA/DA_TWIN']['costs']['income|nll']['value'] +
                         r[v]['paired_costs']['DA/DA_TWIN']['costs']['occupation|nll']['value']) for v in ('0', '1')]
fits = {v: json.loads((C.RESULTS / 'db' / f'ZTILT_R1_V{v}.json').read_text()) for v in C.VERSIONS}
rec = {'schema': 'DDR1-ztilt-r1-reading-v1', 'utc': C.now(), 'bank': 'DDR1_R1_ZTILT', 'card': 'cards/DB_ZTILT_R1.md',
       'comparisons': out, 'declared_reading': declared,
       'model_select_total_nll_cost_vs_twin': spend,
       'model_fit_spend_matching': {str(v): {'chosen': f['chosen'], 'cand_spend': f['cand_spend'],
                                             'chosen_spend': f['spends'][f"{f['chosen'][0]}|{f['chosen'][1]}"],
                                             'reached': f['reached_cand_spend']} for v, f in fits.items()},
       'note': 'descriptive, beside D2 (ADAPTIVE_R1 reading); the registered ZTILT scale grid made the chosen point spend about twice '
               'the release on MODEL_FIT; the refit cells use new splits and reader seeds on the same MODEL_SELECT people'}
C.write_json(C.RESULTS / 'claims' / 'ZTILT_R1_READING.json', rec)
print(json.dumps({'declared': declared, 'registered': z, 'robustness': out['DB_ZTILT_R1']['robustness']['label'],
                  'spend': spend}, indent=1, default=str))
