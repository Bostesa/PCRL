"""Coordinator's label-free pre-check of ddr.claims helpers on the saved releases (no label, no grid output read).

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/test_claims_labelfree.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from ddr import claims as CL   # noqa: E402

t0 = time.process_time()
out = {'budget_flags_DA_STD': CL.reader_budget_flags('DDR1_MAIN', 'DA_STD'),
       'budget_flags_DB_PTC': CL.reader_budget_flags('DDR1_MAIN', 'DB_PTC')}
d = CL.distinct_inputs('DDR1_MAIN')
out['distinct'] = {k: d[k] for k in ('DA_TWIN|V0', 'DA_STD|V0', 'DA_SPD_HI|V0', 'DB_PTC|V1', 'DB_SPD_HI|V1', 'NEURAL_RAW|V0', 'H_LJ|V0')}
s = CL.split_records()
out['splits'] = {r: {'hash': x['assignment_sha256'][:12], 'overlap': {q: round(o, 3) for q, o in x['score_overlap'].items()}}
                 for r, x in s.items()}
out['seeds'], out['boot'] = CL.seed_records()
out['cpu_s'] = time.process_time() - t0
print(json.dumps(out, indent=1, default=str))
