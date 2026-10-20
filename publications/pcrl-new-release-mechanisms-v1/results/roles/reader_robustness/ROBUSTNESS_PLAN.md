# Reader-robustness plan for the promising bank (reader-robustness investigator; Phase 1 draft, about 15:38Z on 2026-10-20)

Written before any candidate outcome. Every robustness result is a descriptive development label. Under the frozen rules
(RULES.md sections 5.1 and 6) robustness cells never change a decision or a ranking. Lower recovery for these readers is never
protection against every reader.

## 1. Bank (frozen by the coordinator, about 18:30-19:00Z)

The bank holds:
- the selected candidate(s);
- the family's task-only twin and its declared simplest alternatives;
- the bracketing SEX-blind control point(s) Q* on the frontier;
- NEURAL_RAW and H_LJ (the target references);
- the reference with the lowest pair AUC.

Further references only if CPU allows. Exact aliases (identical wires) are run once and counted once.

## 2. Grid design (through `nrm.jobqueue run --stage readers`, one unit per version, `nrm.banks write` specs)

**A, repetitions:**
- `--reps R1 R2 R3 R4 --reader-seeds 0 1`.
- The R1/R2 cells are cache hits; the new cells are R3/R4 x k0/k1.

**B, reader seeds:**
- `--reps R1 R2 --reader-seeds 2 3`.
- New cells: R1/R2 x k2/k3.
- A and B form an L-shaped design: 16 new cells instead of 24.
- The R3/R4 x k2/k3 corner is run only if CPU allows.

**C, learning curve:**
- `--reps R1 R2 --reader-seeds 0 --fractions 0.25 0.5 1.0`.
- Only for finite-cell members and their controls, where sparse joint values could change the interpretation.
- '@' cells never enter a screen.

**D, Monte Carlo:**
- Every randomized member gets replicate members at draw offsets +2 and +4 in A, B and the registered cells (D = 3 per cell).
- Offsets: R1 reads draws 0/2/4, R2 1/3/5, R3 2/4/6, R4 3/5/7 (all < R = 8).
- Without replicates, the registered cells carry MC_SE_NOT_ESTIMABLE.

**Comparator readers** are refitted in every new cell, since the grid refits every member. Exact aliases reuse the cache.

**CPU:**
- About 16 CPU-s per release per cell. A + B cost about 260 CPU-s per member per version.
- A 10-member bank is about 1.4 CPU-h for both versions.
- C adds about 0.2 CPU-h for 6 members.
- D adds about 0.6 CPU-h per randomized member pair.
- I will state the exact bank size against the remaining budget before launching.

## 3. Summaries (implementing RULES.md sections 5-6 through `nrm.rules`; code `nrm/readerstab.py`, still to be written)

**Merging:** one or more GRID.json / BOOT.npz per version, merged by cell. The same cell from two runs must agree bitwise, else it is flagged.

**Per (candidate, comparator) and view:**
- `nrm.rules.robustness` (ROBUSTNESS_CONFIRMED / NOT_CONFIRMED / NOT_RUN);
- `aggregate` on the robustness cells and on all cells;
- sign reversals (cells, blocks, versions);
- `reader_stability`: winner changes, near ties, seed-dependent winners, deterministic cells, k_differs, n_eff and the qualifier;
- exact aliases counted once (`EXACT_ALIAS_COUNTED_ONCE`, from the members' wire hashes).

**Monte Carlo:** `nrm.rules.mc_cell` over the offset replicate members, giving `cell_se_mc` per cell. The household, reader/split
and MC components are reported separately.

**Grid validity:** `nrm.rules.grid_validity`, with checks from the per-recipe score fingerprints (proposal 1). If those are not
integrated, the result is `NOT_COMPUTED_GRID_VALIDITY_UNKNOWN` unless the coordinator or verifier supplies the checks.

**Reader targeting** (needs proposals 2-3):
- the same screen on the NOLOOKUP-winner cells and on the enhanced (TABLE_CV) cells;
- an advantage that holds only with table readers, or vanishes with TABLE_CV, is labelled reader-targeted or budget-sensitive.

**Reader budget** (`nrm.readerbudget`, label-free, on every bank member):
- people per vector, unseen shares, LOOKUP data weight, joint-value structure;
- `contrast` (READER_BUDGET_ASYMMETRIC / COMPARABLE) for candidate vs twin, vs SHUF and vs ORD / Q*.

**Writing rule:** "across reader seeds" is written only with READER_SEEDS_EFFECTIVE_ALL_BLOCKS and a valid grid.

**Output:** `results/pcrl_new_release_mechanisms_v1/readers/ROBUSTNESS.json` plus a short report.

## 4. Baseline already measured (label-free; `readers/BUDGET_REFS_V1.*`, unit RR_BUDGET_REFS_V1)

- **Continuous references** (NEURAL_RAW, H_LJ, SUCC, TWJ, W, TR100, TT100, U0, Ucal):
  - median 1 FIT person per pair vector;
  - 93% of SCORE people's pair vectors unseen in FIT;
  - LOOKUP data weight 0.026, so LOOKUP is about the prior.
- **6-level income codes** (B2TWIN, RETIRED_B3CTL): 650-680 FIT people per income vector, none unseen.
- **Distinct vectors:** MODEL_SELECT has at most 7,500 distinct pair vectors among 8,001 people (identical raw inputs give identical
  outputs).
- **Splits:** the R1-R4 masks rebuilt here equal READER_SPLITS.json.
