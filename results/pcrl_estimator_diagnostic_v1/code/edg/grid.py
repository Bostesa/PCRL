"""Write GRID.json: the frozen grid, repetition counts, seeds, bank subset, rules and the SHA-256 of every code file (code lock)."""
import hashlib
import json

from . import CODE, RESULTS
from . import analyze as AN
from . import bank as BK
from . import estimators as E
from . import laws as L
from . import simulate as SIM


def main(argv=None):
    files = sorted(p for p in CODE.rglob('*.py') if '__pycache__' not in p.parts) + [RESULTS / 'edg_run.py']
    rec = {'schema': 'EDG1-grid-v1', 'scenarios': SIM.scenarios(), 'reps': SIM.REPS, 'min_reps': SIM.MIN_REPS,
           'base_seed': SIM.BASE_SEED, 'seed_rule': 'np.random.SeedSequence(base_seed, spawn_key=(cell, rep))',
           'estimators': SIM.ESTIMATORS, 'nested_lookup_a': E.NESTED_A, 'bank_lookup_pseudo': E.BANK_A, 'folds': E.N_FOLDS,
           'split': 'unit permutation % 10: parts 0-4 FIT, 5-6 VALIDATION, 7-9 SCORE (independent of the fold permutation)',
           'families': L.FAMILIES, 'pis': L.PIS, 'occupancies': L.OCCS, 'ns': SIM.NS,
           'bank_cells': [[k, v] for k, v in BK.BANK_CELLS], 'bank_cpu_cap_s': BK.BANK_CPU_CAP, 'bank_max_reps': BK.MAX_REPS,
           'bank_min_reps': BK.MIN_REPS_BANK, 'class_limit_recipes': BK.CLASS_LIMIT_RECIPES, 'n_big': BK.N_BIG,
           'n_big_val': BK.N_BIG_VAL, 'reader_seed_offset': BK.SEED_OFFSET,
           'decision_rule': 'R2SE: X > Y iff d - 2 se > 0; X < Y iff d + 2 se < 0; se = linearized paired AUC SE (rankings fixed), by unit',
           'secondary_rule': 'M20: |d| >= 0.020 when dA = 0', 'null_tolerance': AN.TOL0,
           'code_sha256': {str(p.relative_to(RESULTS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    with open(RESULTS / 'GRID.json', 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(len(rec['scenarios']), 'scenarios;', len(rec['code_sha256']), 'code files locked')
