# Reader-robustness investigator, Phase 2 report (saved verbatim with attribution)

Author: the reader-robustness investigator of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 16:48Z on
2026-10-20 (its own header). The investigator read no label itself and ran no sub-agents; it asked for this text to be saved verbatim
because a report file of its own was refused by the harness. Saved verbatim; nothing below the line was edited.

---

Reader-robustness investigator, Phase 2 report (16:48Z on 2026-10-20). Every comparison is ROBUSTNESS_NOT_CONFIRMED: the FB2 SEX path's benefit over its twin keeps its sign on every new split and seed but is never resolved, and its edge over the SEX-blind MINF channel is about zero and changes sign. Everything is committed; the last commit is f07a8620f. I read no label myself and ran no sub-agents. Please save this verbatim: the harness refused a report file from me (sub-agents must return text), so this message is the report.

**Scope.** These are descriptive development results on adaptively used MODEL_SELECT records. Robustness cells never change a decision or a ranking (RULES.md sections 5.1 and 6). Versions share training rows, and R3/R4 share about 1,300 FIT households with R1/R2. Lower recovery for these readers is not protection against every reader, and each person gets one released instance. The registered numbers below are my recomputation, averaged over the base, +2 and +4 instances as `nrm.claims` does; your claims records govern.

## What ran
- **Bank.** Frozen outcome-blind at 15:57Z (724a81f71), before any main-grid output was read: FB2_BASE, FB2_SEX_S3/S5, FB2_MINF_S3/S5, NEURAL_RAW, H_LJ, RETIRED_B3CTL. Plan: `readers/ROBUST_PLAN.json`.
- **Design.** L-shaped, both versions: R3/R4 at reader seeds k0/k1 (NRM1_ROBUST_REPS), plus R1/R2 at seeds k2/k3 (NRM1_ROBUST_SEEDS).
  - That gives 8 registered plus 16 robustness cells per comparison.
  - The R3/R4 x k2/k3 corner was not run, and there was no learning curve.
  - Robustness cells read one instance each, so no Monte Carlo SE is estimable there.
- **Code.** `nrm/readerstab.py` was committed, with selftests passing (25/25, then 26/26), before any grid output was read. Final version 5c803c086; earlier fcf55f224, 78aa98106, 29699f02b.
- **Run.** Unit RR_STAB_RUN_V1 wrote `readers/ROBUSTNESS.json`: 56 comparisons over the merged runs.
- **Integrity.**
  - 0 cell conflicts (registered cells re-read bitwise identical).
  - My recomputed paired cells equal the grid's own (PAIRED_MATCH).
  - Grid validity for the bank members is READER_SEEDS_EFFECTIVE: 3,072 k0/k1 checks over R1-R4. All 16 seed-dependent equalities are ET5/ET25 on RETIRED_B3CTL's income view, which has only 6 distinct vectors. They are explained, and none is on the minimum set.
- **Exact alias.** H_LJ and RETIRED_B3CTL share an identical occupation wire (the grid's own alias record). It is counted once.

## Pair benefit (comparator AUC minus candidate AUC; positive favours the candidate)

| Comparison | Registered x | Robustness cells | R3/R4 part | k2/k3 part | All 24 cells, x (se) |
|---|---|---|---|---|---|
| SEX_S3 vs BASE | +0.0069 | +0.0069 | +0.0081 | +0.0057 | +0.0072 (0.0072) |
| SEX_S5 vs BASE | +0.0120 | +0.0128 | +0.0159 | +0.0097 | +0.0134 (0.0080) |
| SEX_S3 vs MINF_S3 | +0.0027 | +0.0006 | -0.0009 | +0.0021 | +0.0007 (0.0087) |
| SEX_S5 vs MINF_S5 | +0.0019 | +0.0023 | +0.0018 | +0.0027 | +0.0020 (0.0080) |
| SEX_S3 vs MINF_S5 | -0.0032 | -0.0037 | -0.0060 | -0.0013 | -0.0041 |
| SEX_S5 vs MINF_S3 | +0.0078 | +0.0065 | +0.0069 | +0.0061 | +0.0069 (0.0076) |
| MINF_S3 vs BASE | +0.0042 | +0.0063 | +0.0090 | +0.0035 | +0.0064 (0.0062) |
| MINF_S5 vs BASE | +0.0101 | +0.0105 | +0.0141 | +0.0070 | +0.0113 (0.0067) |
| SEX_S3 vs NEURAL_RAW / H_LJ / RETIRED_B3CTL | -0.018 / -0.021 / -0.037 | -0.017 / -0.023 / -0.042 | | | |
| SEX_S5 vs NEURAL_RAW / H_LJ / RETIRED_B3CTL | -0.013 / -0.016 / -0.032 | -0.011 / -0.017 / -0.036 | | | |

Every positive registered value above is AGG_POSITIVE_POINT_ONLY; none is resolved.

## Findings
1. **Twin benefits keep their sign but never resolve.** SEX vs BASE and MINF vs BASE have no sign reversal in any cell, block or version, at either strength. But x - 2 se < 0 everywhere; the household-bootstrap part of the SE is about 0.005-0.009.
2. **SEX targeting adds nothing stable over MINF.**
   - At S3 the edge falls from +0.0027 to +0.0006 on the robustness cells. 4 of 8 robustness blocks have the opposite sign, version 1 is negative there, and block S1R3 is -0.0119.
   - At S5 it is +0.0019 to +0.0023, with 3 opposite blocks.
   - Across strengths the order follows noise strength alone: SEX_S3 reads worse than MINF_S5, and SEX_S5 better than MINF_S3. The cost-aware verdict (G3) is yours.
3. **Every FB2 release is easier to read than every reference.** Benefits run from -0.011 to -0.042 with no sign reversal in 24 cells. Mean pair AUC: FB2_BASE about 0.660, NEURAL_RAW 0.636, H_LJ 0.631, RETIRED_B3CTL 0.613.
4. **The registered income-view failure is a reader-selection artifact.**
   - The registered gate for SEX vs BASE reads RECIPIENT_AGG_POINT_FAILED in version 0 (r = +0.0130 at S3, +0.0111 at S5).
   - It comes from block S0R2 alone. There the twin's income winner, picked on VALIDATION, is ET25 (a near tie). It scores 0.532 on SCORE, against 0.554-0.565 in every other block; the SEX path's winners there are MLP readers at about 0.561.
   - The same S0R2 cells also exceed 0.010 for MINF vs BASE, although MINF leaves income almost unchanged.
   - Over all 24 cells it reads RECIPIENT_AGG_EDGE (+0.0074 / +0.0061), and every version-0 income winner is a near tie.
   - Separately, SEX_S5 vs MINF_S5 on the occupation view is POINT_FAILED on the registered cells (r0 = +0.0100) and EDGE on all cells.
5. **No table-reader effect.** On these continuous outputs neither LOOKUP nor TABLE_CV wins in any cell of any member. TABLE_CV reads about 0.50 on FB2, and 0.534 on RETIRED_B3CTL through its 6-level income code. The NOLOOKUP and enhanced benefits equal the main ones; the declared targeting labels are all NO_MAIN_BENEFIT.
6. **Reader seeds are only partly effective.**
   - In the registered grid, k0 and k1 give identical cells in 2 of 4 blocks for the twin comparisons. The winners are deterministic: FB2_SEX_S5's pair winner is CATI01 in 20 of 24 cells.
   - So the qualifier is READER_SEEDS_PARTLY_EFFECTIVE, and "across reader seeds" should not be written.
   - The all-cells reading of EFFECTIVE_ALL_BLOCKS is an artifact of mixing averaged registered cells with single-instance k2/k3 cells. The single-instance variant, declared before any output, reads PARTLY_EFFECTIVE.
7. **Reader budgets are comparable** (label-free, `readers/BUDGET_ROBUST_V1.*`). Every FB2_SEX vector is unique. FB2_MINF's noise lands almost entirely on occupation (7,488 distinct income vectors against the base's 7,484), while FB2_SEX noises both outputs.

## Process
- **Withdrawal (your 16:20Z one-unit rule).**
  - RR_ROB_REPS_S1_V1 had already started at 16:18:38Z, when a slot was freed by the verifier.
  - At 16:20:10Z I sent SIGTERM to its reader process only (PID 99818). The unit failed with return code -15 after 91.2 CPU-s.
  - Its empty default folder `runs/GRID/NRM1_ROBUST_REPS/S1` is kept.
  - I reran it as RR_ROB_REPS_S1_V2 into `.../S1_V2` (read through `readerstab --run-dir`), reusing 19 completed cache entries, then RR_ROB_SEEDS_S1_V1. From then on only one of my units was in the queue at a time.
- **Other corrections and fixes.**
  - A spec draft carried a 15:58Z time, ahead of the clock; I corrected it before the commit.
  - The new instance map broke `nrm.readerbudget`'s randomized path; I fixed it (selftest 19/19).
  - The full timeline is in `roles/reader_robustness/PHASE2_LOG.md`.
- **CPU.** 1,695 CPU-s in the queue over 15 RR units, 2 of them failed (the V1 budget selftest's wrong expectation and the withdrawn unit), plus 5 CPU-s charged. That is about 0.47 CPU-h.

Files are in `results/pcrl_new_release_mechanisms_v1/`:
- `readers/ROBUSTNESS.json`
- `readers/ROBUST_PLAN.json`
- `readers/BUDGET_ROBUST_V1.json` / `.csv`
- `readers/BUDGET_REFS_V1.json` / `.csv`
- `grid_specs/NRM1_ROBUST_REPS.json`
- `grid_specs/NRM1_ROBUST_SEEDS.json`
- `roles/reader_robustness/PHASE2_LOG.md`

Code: `nrm/readerstab.py`, `nrm/readerbudget.py`, `nrm/readerextra.py`.
