"""Write ANALYSIS_LOCK.json: the endpoint roster, claim conjunctions, inference settings and the verifier's refit subset, as frozen
by the statistical reviewer (roles/statistical_reviewer/FROZEN_DESIGN.md) before any final reader fit. Write-once; no label read.

Roles in the final bank (FINALIST_LOCK.json): FINALIST, NEURAL_RAW, H_LJ (eligible references), TASKONLY (matched task-only
control), B2TWIN (the same coarse-code construction with the LOCAL_TREE income statistic), H_JJ (the JOINT_TREE parent,
utility-ineligible), PUB_IBSI (the published IBSI objective as adapted in the SCM follow-up), T_TREE (task-only teacher).
"""
from __future__ import annotations
import json

from . import common as C

CAP_ORDER = ('accuracy_loss', 'nll_excess', 'brier_excess')


def build(design):
    """design: dict with 'inference', 'endpoints' (list), 'claims' (list), 'refit_subset' (list), 'notes'."""
    keys = [e['key'] for e in design['endpoints']]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate endpoint keys')
    fam_sizes = {}
    for e in design['endpoints']:
        fam_sizes[e['family']] = fam_sizes.get(e['family'], 0) + 1
    if fam_sizes != design['inference']['K']:
        raise ValueError(f"family sizes {fam_sizes} differ from the declared K {design['inference']['K']}")
    for cl in design['claims']:
        conds = list(cl.get('conditions', [])) + [c for st in cl.get('steps', []) for c in st['conditions']] + list(cl.get('nested', {}).get('adds', []))
        for c in conds:
            if 'endpoint' in c and c['endpoint'] not in keys:
                raise ValueError(f"claim {cl['name']} refers to an unknown endpoint {c['endpoint']}")
    src = C.RESULTS / 'roles' / 'statistical_reviewer' / 'CLAIMS_AND_ENDPOINTS_DRAFT.md'
    rec = {'schema': 'AMD1-analysis-lock-v1', 'utc': C.now(), 'finalist_lock_sha256': C.sha(C.RESULTS / 'FINALIST_LOCK.json'),
           'frozen_design_source': 'roles/statistical_reviewer/CLAIMS_AND_ENDPOINTS_DRAFT.md', 'frozen_design_sha256': C.sha(src),
           'encoding_check': 'statistical reviewer, four lock corrections and one report correction applied before the commit', **design}
    C.write_json_x(C.RESULTS / 'ANALYSIS_LOCK.json', rec)
    print(json.dumps({'K': fam_sizes, 'claims': [c['name'] for c in design['claims']], 'refit_subset': len(design['refit_subset'])}))
