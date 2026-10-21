"""Write the robustness-refit bank DDR1_ROB (coordinator; label-free; written before any grid output is read). PROTOCOL 6.3 / RULES.md 8:
the strongest feasible nonzero release of each family (with one operation configuration per family at revision 0, its CAND, which is
G1-feasible in both versions), its TWIN and its bracketing controls (Q* of each path by the inherited rule on the MODEL_SELECT paired
total-NLL costs), the component ablations, DB's ZTILT and SHUF, and NEURAL_RAW / H_LJ, on repetitions R1-R4 x reader seeds k0-k3 (the
R1 / R2 x k0 / k1 cells are the registered ones and are cache hits).

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/write_rob_bank.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ddr import common as C          # noqa: E402
from ddr import devreaders as DR     # noqa: E402
from ddr import banks as BK          # noqa: E402

MEMBERS = {'DA_TWIN': 'DA/DA_TWIN', 'DA_STD': 'DA/DA_STD', 'DA_SBC_HI': 'DA/DA_SBC_HI', 'DA_SPD_HI': 'DA/DA_SPD_HI',
           'DA_STD_OCC': 'DA/DA_STD_OCC', 'DA_COMP': 'DA/DA_COMP',
           'DB_PTC': 'DB/DB_PTC', 'DB_SBC_HI': 'DB/DB_SBC_HI', 'DB_SPD_HI': 'DB/DB_SPD_HI', 'DB_ZTILT': 'DB/DB_ZTILT',
           'DB_SHUF': 'DB/DB_SHUF', 'DB_PTC_OCC': 'DB/DB_PTC_OCC'}
mem = {k: BK.family_member(v) for k, v in MEMBERS.items()}
mem['NEURAL_RAW'] = BK.REFERENCE_MEMBERS['NEURAL_RAW']
mem['H_LJ'] = BK.REFERENCE_MEMBERS['H_LJ']
spec = {'bank_id': 'DDR1_ROB',
        'purpose': 'robustness to classifier refitting (D2R, descriptive): new household splits R3 / R4 and reader seeds k2 / k3 for the '
                   'strongest feasible nonzero release of each family (its CAND), its TWIN, its bracketing controls on both paths '
                   '(SBC_HI, SPD_HI: the MODEL_SELECT bracket Q* of each path), the component ablations, DB ZTILT and SHUF, and '
                   'NEURAL_RAW / H_LJ; the registered cells R1 / R2 x k0 / k1 are included (cache hits) so that entries merge',
        'members': mem, 'candidates': ['DA_STD', 'DA_STD_OCC', 'DA_COMP', 'DB_PTC', 'DB_PTC_OCC', 'DB_ZTILT', 'DB_SHUF'],
        'comparators': list(mem), 'reps': ['R1', 'R2', 'R3', 'R4'], 'reader_seeds': [0, 1, 2, 3], 'written_utc': C.now(),
        'benchmark_references': ['NEURAL_RAW', 'H_LJ'], 'diagnostics': []}
DR.validate_spec(spec)
out = C.RESULTS / 'grid_specs' / 'DDR1_ROB.json'
if out.exists():
    raise SystemExit('DDR1_ROB.json exists (write-once)')
C.write_json(out, spec)
print(json.dumps({'spec': str(out.relative_to(C.ROOT)), 'members': len(mem)}))
