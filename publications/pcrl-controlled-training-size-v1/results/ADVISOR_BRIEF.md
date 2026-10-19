# Advisor brief: controlled training-size test with one conditional protection experiment (PCRL, 2026-10-19)

**What we asked.** The previous campaign refitted the occupation recipe on 42-53% of MODEL_FIT and lost most of its subgroup margin.
(1) Does a fixed occupation recipe (A1_PO_16) keep useful task and subgroup margin at 50%, 70% and 100% of the training pool, against an
equally trained ordinary control? (2) If the full-data model keeps it, does one fixed correction recipe use it to lower the declared
development readers' recovery of the recorded SEX column while keeping usefulness?

**What we did.**
- **Data.** Public-use ACS 2019 Florida development roles only. MODEL_SELECT, already used adaptively, served as the common evaluation
  role. The final attack and DEV_REPORT labels stayed closed.
- **Registration.** Everything was registered and published on GitHub before any real-label fit. Five label-free benchmark units on
  invented labels ran earlier on uncommitted code (deviation 7).
- **Size study.** Four household orderings with nested 50 / 70 / 100% subsets. A1_PO_16 and the task-only T_TREE -> TASK_REFIT control
  were refitted on the same households, in two source versions (procedure repeats on the same rows).
- **Conditional stage.** A registered trigger on the full-data refit fired. One correction recipe was fitted on the full pool, with an
  occupation-only critic (LOC, primary), a coalition-aware critic (PAIR) and a matched zero-critic continuation (CONT). Its strength
  was chosen from nested 10-fold cross-fitting on MODEL_FIT only, and fresh development readers read the selected releases.
- **Verification.** An independent verifier replayed the registered checks with its own code: all 22 A1_PO_16 fits refitted bitwise,
  the trigger (within 1.7e-16), the S100 control identity and one T_TREE refit, one nested outer fold end to end, both selections, the
  utility tables and 39 reader-label comparisons. We deployed 11 packages from the six raw fields.

**What we found.**
1. **On this pool the registered size explanation is supported, but the gain is not shown to be specific to the recipe.**
   - From 50% to 100% the men's Brier excess falls by 4.3 / 4.2e-3 on average (v0 / v1; every ordering positive, 3.2-5.3e-3), and NLL
     improves by about 11.4e-3.
   - The occupation reserve holds in all 8 cells at 70% and in both versions at 100%, but in only 4 of 8 cells at 50%. So the earlier
     reduced-training refits did not settle full-data feasibility.
   - The ordinary TASK_REFIT control shows a resolved size gain of its own on NLL, Brier and men's Brier. The difference between the
     two gains is not resolved on those metrics; on women's Brier it is version-dependent (in favour of A1_PO_16 on v1 only).
2. **The full-data refit reproduces the source package bitwise** (both versions; confirmed by the verifier's own refits), with minimum
   occupation reserve ratios 4.88 / 4.30.
   - Against the equally trained TASK_REFIT control it is not resolved better. It is resolved better than T_TREE on NLL, Brier and
     men's Brier.
   - TASK_REFIT also keeps the reserve at 100%.
3. **The correction kept its point utility but did not protect.**
   - Both arms selected s = 0.3 in both versions. At that strength they pass every original point limit (24 clauses + 4 guards) and,
     separately, keep the occupation-side reserve (a development heuristic). They fail the original per-cell recipient gate in the
     occupation view against every strong reference.
   - The occupation-only arm lowers the readers' pair AUC by only about 0.005 against the witness and against the continuation. All 8
     cells are positive (7 distinct; the reader seed changed the result in 3 of 4 version-repetition pairs), but the registered screen
     says not stable.
   - The coalition-aware arm changes sign and is no better than the occupation-only arm.
   - Every strong reference has lower mean recovery (by 2.8e-3 for TWJ against LOC, up to 45e-3), passes the original point limits,
     and fails the two-SE reserve heuristic. The 0.020 target is not met.
   - Reader-label disclosure: the registered effective-reader-seed check, read literally over all 14 checked bank members, returned
     READER_SEEDS_NOT_EFFECTIVE: ExtraTrees readers (ET5, ET25) are seed-invariant on the income views of B2TWIN and RETIRED_B3CTL,
     which take 6 distinct values. On the registered minimum set (W100_v and every selected release) there was no equality. Amendment
     A-SC1, made after this check's outcome and before any reader-stability label was computed or read, limited grid invalidation to
     unexplained equalities. The grid passed the amended check. No stability formula changed.
   - Qualification (deviations 11-13): "before any reader-stability label was computed or read" holds for the coordinator and the
     statistical reviewer. The independent verifier had already computed its own labels (01:32:59-01:33:09Z, after the seed-check
     record and before the ruling at 01:36:00Z) and stored them unread.
4. **A transfer warning.** In point estimates the nested 90% cost estimate under-predicted the deployed model's men's cost 1.9- to
   2.8-fold: realized 0.9-1.9e-3 (SE 0.3-0.4e-3) on MODEL_SELECT, against 0.5-0.8e-3 predicted. Only LOC v0's gap exceeds two SEs. The
   reserve still held at the selected strength.

**Classification.** A controlled training-size result on this development pool, and a closed negative correction recipe
(CORRECTION_RECIPE_CLOSED). It is not a useful component and not competitive, and nothing is confirmed.

**Process notes.**
- The reader-seed check had to be amended after its outcome, before the coordinator or the statistical reviewer had computed or read
  any stability label. Two references' coarse income wires make ExtraTrees seed-invariant. The verifier had computed its own labels
  minutes earlier and stored them unread (deviations 11-12). Under the literal rule the closure would be the same.
- Two selftest label reads seconds after the receipt, and five pre-registration benchmark units on uncommitted code, are disclosed.

**Recommendation.**
- Do not tune this correction further. The recipe is closed.
- If the occupation line continues, treat training size as a property of the pool that the equally exposed T_TREE / TASK_REFIT control
  shares, not as an advantage of A1_PO_16.
- Any successor needs a cost estimator that transfers to the deployed model, and a fresh population before any claim.

Read next: `RESEARCH_DECISION.md`, then `START_HERE.md`.
