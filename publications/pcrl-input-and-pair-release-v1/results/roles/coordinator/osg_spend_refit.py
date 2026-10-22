"""IA_OSG against its two occupation-only quantization points on the refit cells (coordinator; label-free; descriptive). IA_OSG's
SPEND path has no bracket on MODEL_SELECT (NO_BRACKET_SPEND), so `ipr.claims d2r` passes it as an explicit null and the D2R record
holds no SPEND comparator. This record reports, beside D2R and never as a decision input, `nrm.rules.robustness` of IA_OSG against
IA_OSG_SPD_HI and IA_OSG_SPD_LO on the merged registered (IPR1_R1) and robustness (IPR1_ROB_R1) cells, read from the saved grids.

    python3 results/pcrl_input_and_pair_release_v1/roles/coordinator/osg_spend_refit.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ipr import common as C          # noqa: E402
from ipr import claims as CL         # noqa: E402
from nrm import rules as NR          # noqa: E402


def main():
    g = CL.merge_grids(CL.load_grid('IPR1_R1'), CL.load_grid('IPR1_ROB_R1'))
    out = {}
    for comp in ('IA_OSG_SPD_HI', 'IA_OSG_SPD_LO'):
        e = (g['paired'].get('IA_OSG') or {}).get(comp)
        if e is None:
            out[comp] = {'missing': True}
            continue
        r = NR.robustness(e)
        rob = r['robustness_cells']['aggregate']
        reg = NR.comparison(e)['benefit'].get('aggregate') or {}
        out[comp] = {'cand': 'IA_OSG', 'comparator': comp, 'robustness_label': r['label'],
                     'robustness_mean': rob.get('mean'), 'robustness_se_agg': rob.get('se_agg'),
                     'n_eff_robustness_cells': rob.get('n_eff_distinct_cells'),
                     'reader_seed_qualifier': rob.get('reader_seed_qualifier'),
                     'registered_mean': reg.get('mean'), 'registered_se_agg': reg.get('se_agg')}
        print(comp, r['label'], rob.get('mean'), rob.get('se_agg'), reg.get('mean'), reg.get('se_agg'))
    C.write_json_x(C.RESULTS / 'claims' / 'IA_R1_SPEND_REFIT.json',
                   {'schema': 'IPR1-beside-refit-v1', 'utc': C.now(), 'banks': ['IPR1_R1', 'IPR1_ROB_R1'],
                    'note': "descriptive, beside IA_R1_D2R and never a decision input: IA_OSG's SPEND path has no bracket on MODEL_SELECT "
                            "(NO_BRACKET_SPEND), so the D2R reading holds no SPEND comparator; benefit = comparator AUC - IA_OSG AUC",
                    'comparisons': out})


if __name__ == '__main__':
    main()
