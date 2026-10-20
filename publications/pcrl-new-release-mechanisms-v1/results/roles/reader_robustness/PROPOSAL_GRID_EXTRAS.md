# Proposal to the coordinator: grid extras before the grid code lock (reader-robustness investigator)

Written about 15:35Z on 2026-10-20, before any candidate outcome (no family release had been graded; I read no label). It changes
nothing until the coordinator integrates it into `nrm/devreaders.py` and adds `nrm/readerextra.py` to `LOCK_FILES`.
Code: `nrm/readerextra.py` (commits 109eb4571 and fc61c797d); selftest 15/15 on invented data (units RR_EXTRA_SELFTEST_V1/V2).

## What it adds per (cell, release, view) (no change to registered winners, cells or decisions)

1. **Score fingerprints**: SHA-256 of each locked recipe's oriented SCORE scores, and each recipe's VALIDATION AUC. The A-SC1 grid-validity
   check (RULES.md section 6, `nrm.rules.grid_validity`: bitwise k0 / k1 equality per seed-dependent recipe, view, wire, version,
   repetition) can then be computed from GRID.json alone. Without it, only code that re-reads SEX can compute `grid_valid`.
2. **NOLOOKUP winner**: chosen exactly as the registered winner (VALIDATION, same candidate order, ties to the earlier candidate) with
   the table readers removed. Its SCORE AUC and bootstrap draws use the cell's weights, so differences are paired. This answers the
   challenger's requirement that a PRC-vs-ORD advantage be credited only if it also holds for smooth readers.
3. **TABLE_CV and the enhanced winner**: a frequency-table reader with cross-validated smoothing for finite output cells.
   - Single view: LOOKUP with a pseudo-count chosen by household-grouped 5-fold CV inside FIT.
   - Pair: an additive key logit (income-cell + occupation-cell indicators, L2 chosen by CV) with the joint table shrunk towards it.
   - It is deterministic and costs about 0.05 CPU-s per view at n = 8,000.
   - The enhanced winner = 16 locked recipes + TABLE_CV, with TABLE_CV last, so exact ties keep the locked recipe.
   - Why: the locked LOOKUP reads up to 1,024 joint values with about 4,000 FIT people, and no locked recipe gives one free parameter per
     occupation cell inside the pair view. On invented additive cell data (16 x 64 cells), the pair AUC is TABLE_CV 0.862 against
     LOOKUP 0.821 (a mechanics check, not an effect-size estimate).
   - It is applied to every release with the same effort. On continuous releases it returns about the FIT prior.

## Exact integration (nrm/devreaders.py)

```python
from . import readerextra as RX                                   # imports
LOCK_FILES = (..., 'nrm/readerextra.py')                          # lock list
EXTRAS = True                                                     # constant
# evaluate_bank, after the per-view `entry[view] = {...}` loop (W, inc, occ, rep, frac are in scope there):
if EXTRAS:
    ex, eb = RX.cell_extras(rd, inc, occ, sex, unit, rep, frac, W, views)
    for view in views:
        entry[view]['extras'] = ex[view]
        entry[view]['extras_boot'] = eb[view]
# strip(): drop 'extras_boot' as well as 'boot'
#     {k: x for k, x in e.items() if k not in ('boot', 'extras_boot')}
# run_bank, after BOOT.npz:
bx = {f'{cell}__{n}__{v}__{k}': b for cell, cres in res.items() for n, ent in cres.items() for v, e in ent.items()
      for k, b in e.get('extras_boot', {}).items()}
if bx:
    with open(out / 'BOOT_EXTRAS.npz', 'wb') as f:
        np.savez(f, **bx)
```

- Cost: about 0.5 CPU-s per release per cell, on top of about 16 CPU-s (+3%).
- Labels: SEX is used exactly as the grid already uses it: FIT for fitting, VALIDATION for orientation and selection, SCORE for AUC.
- Use: descriptive reader-robustness labels only (`HOLDS_WITHOUT_TABLE_READERS` / `NEEDS_TABLE_READERS`, the enhanced-reader benefit,
  grid validity). The frozen decision rules and the registered winners are unchanged.
- If you decline 3 (TABLE_CV), call `cell_extras(..., table_cv=False)`; 1 and 2 still work.
