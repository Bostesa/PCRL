"""Bank DDR1_R1_ZTILT (card DB_ZTILT_R1; label-free; written after the control is saved and before any of its grid output):
DA_TWIN, DB_PTC_R1 and DB_ZTILT_R1 on R1-R4 x k0-k3 (DB_PTC_R1's and DA_TWIN's cells are cache hits of the R1 and refit grids).

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/write_ztilt_r1_bank.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ddr import common as C          # noqa: E402
from ddr import devreaders as DR     # noqa: E402
from ddr import banks as BK          # noqa: E402

mem = {k: BK.family_member(v) for k, v in {'DA_TWIN': 'DA/DA_TWIN', 'DB_PTC_R1': 'DB/DB_PTC_R1', 'DB_ZTILT_R1': 'DB/DB_ZTILT_R1'}.items()}
spec = {'bank_id': 'DDR1_R1_ZTILT',
        'purpose': 'card DB_ZTILT_R1: DB_PTC_R1 (DB\'s strongest release) against its spend-matched release-only control DB_ZTILT_R1 and '
                   'the twin, on the registered cells and the refit cells (descriptive, reported beside D2)',
        'members': mem, 'candidates': ['DB_PTC_R1'], 'comparators': list(mem), 'reps': ['R1', 'R2', 'R3', 'R4'],
        'reader_seeds': [0, 1, 2, 3], 'written_utc': C.now(), 'benchmark_references': [], 'diagnostics': []}
DR.validate_spec(spec)
out = C.RESULTS / 'grid_specs' / 'DDR1_R1_ZTILT.json'
if out.exists():
    raise SystemExit(f'{out.name} exists (write-once)')
C.write_json(out, spec)
print(json.dumps({'spec': str(out.relative_to(C.ROOT)), 'members': sorted(mem)}))
