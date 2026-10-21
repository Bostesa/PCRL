"""Verifier notes: explicit NOT_RUN / NOT_APPLICABLE entries for items of the verifier's brief that were not recomputed, so that the
CHECKS.json totals show them (campaign `pcrl_finite_solver_adequacy_v1`). No computation, no data read.

    python3 .../vnotes.py --selftest
    python3 .../vnotes.py --tag V1        (appends the entries below once; ids are unique)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcore as V  # noqa: E402

SCRIPT = f'results/{V.CAMPAIGN}/roles/verifier/code/vnotes.py'


def notes(tag):
    return [
        {'id': f'V2.MEANS_TRAINING_SIDE__{tag}', 'block': 'V2', 'kind': 'numerical', 'status': 'NOT_APPLICABLE',
         'summary': 'nested_means keys that need the training-side partitions of each outer fold (fit_auc_pair_insample, '
                    'fit_mi_pair_labels_insample, fit_mi_pair_pi_fitted, divergence_auc_pair_insample_minus_heldout, '
                    'mean_fit_rerouted_vs_twin, mean_heldout_rerouted_vs_twin) cannot be recomputed from the per-person held-out '
                    'file; they enter the reproduction comparison (replay vs record, max |diff| 0.0) only'},
        {'id': f'V3.BREAKPOINT_ENUMERATION__{tag}', 'block': 'V3', 'kind': 'numerical', 'status': 'NOT_RUN',
         'summary': 'not re-enumerated with own code: the tilt-line breakpoints of the narrow reference class (completeness of '
                    'N_card / N_wide; only the objectives at the reported partitions were recomputed) and the full assignment '
                    'enumeration of the m = 6 / 8 laws (own brute force run for D_witness only, 1,296 assignments, counts and '
                    'optima equal); STEP5_V1.json was checked through route_e_input_SV1.json (V3.ROUTE_E), not separately'},
        {'id': f'V4.AUDIT_NOT_RECOMPUTED__{tag}', 'block': 'V4', 'kind': 'numerical', 'status': 'NOT_RUN',
         'summary': 'audit numbers not recomputed: the tilt-line interval objectives and Spearman ranks of the proposal score '
                    '(PRC / HQ at strength 8), the runtime in-sample vs out-of-fold routing agreement of TWB_C64 / PRC_S2 / HQ_S2, '
                    'the expressivity counts, the refresh-step objective deltas and the tried tilts of the traced chain'},
        {'id': f'V6.ADVANCEMENT_D_E_INPUTS__{tag}', 'block': 'V6', 'kind': 'procedural', 'status': 'NOT_RUN',
         'summary': 'fsa.rules.advancement was applied with the verifier\'s own route S input and the RECORDED route D and route E '
                    'inputs; route D claims are qualitative (not recomputed); route E metrics were recomputed (V3.ROUTE_E) but the '
                    'route E objectives of the original / revised / reference states (which need their books) were not'},
        {'id': f'V6.EQUAL_EFFORT__{tag}', 'block': 'V6', 'kind': 'procedural', 'status': 'NOT_RUN',
         'summary': 'decisions/EQUAL_EFFORT_ATTRIBUTION.json not checked (descriptive; never a decision input under RULES)'},
    ]


def selftest():
    res = []
    with tempfile.TemporaryDirectory() as td:
        st = V.Checks(td)
        st.append(notes('T'))
        doc = json.loads(st.path.read_text())
        res.append(('appended', len(doc['checks']) == len(notes('T')) and st.verify_chain() == [], None))
        res.append(('statuses', all(e['status'] in ('NOT_RUN', 'NOT_APPLICABLE') for e in doc['checks']), None))
        try:
            st.append(notes('T'))
            res.append(('refuses_duplicates', False, None))
        except PermissionError:
            res.append(('refuses_duplicates', True, None))
    n_ok = sum(r[1] for r in res)
    for name, ok, _ in res:
        print(('PASS ' if ok else 'FAIL ') + name)
    print(f'vnotes selftest: {n_ok}/{len(res)}')
    return n_ok == len(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--tag', default='V1')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    items = notes(a.tag)
    for it in items:
        it['script'] = SCRIPT
    V.Checks().append(items)
    print(json.dumps([it['id'] for it in items]))
