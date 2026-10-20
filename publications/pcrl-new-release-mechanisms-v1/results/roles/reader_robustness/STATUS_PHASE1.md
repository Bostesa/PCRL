# Reader-robustness investigator, Phase 1 status (saved verbatim with attribution)

Author: the reader-robustness investigator of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 15:35Z on
2026-10-20 (written 15:34Z). Saved verbatim; nothing below was edited. Its grid-extras proposal was integrated before the grid code lock
(commit 7222b0105; lock 1128afd67).

---

Reader-robustness investigator, Phase 1 status at 15:34Z on 2026-10-20. Handing back early, as you asked. Everything is committed, no label was read, and my numerical work used 9.1 CPU-s in the queue plus 5 CPU-s charged as an upper estimate.

**Done**

1. **`nrm/readerbudget.py`** (commits 92b274003, test fix 8a1d7… see git log; label-free).
   - **What it reports**, for any release (a saved one, a frozen reference or a bank member), by version, repetition R1-R4, FIT fraction 0.25 / 0.5 / 1.0 and view (income, occupation, pair):
     - distinct released vectors, exact and rounded to 6 / 3 / 2 decimals;
     - FIT people per vector, per vector and per person;
     - VALIDATION and SCORE coverage: share never seen in FIT, share in 1-4-person and 15+-person groups, mean LOOKUP data weight;
     - the pair view's joint-value structure.
   - **Randomized releases** are reported per draw, with the draw each repetition reads at offsets +0, +2 and +4.
   - **Budget contrast flag:** `contrast()` gives READER_BUDGET_ASYMMETRIC when group sizes differ by more than 2x or unseen shares by more than 0.10.
   - **Split check:** the R1-R4 masks it rebuilds equal READER_SPLITS.json.
   - **Selftest:** 19/19 on invented data (unit RR_BUDGET_SELFTEST_V2). V1 failed on a wrong test expectation, not a code bug; the fix is recorded in the commit message.
   - **Baseline on the 9 references plus U0 and Ucal** (`readers/BUDGET_REFS_V1.json/.csv`, commit 8ef90449c):
     - Every continuous reference: about 1 FIT person per pair vector, 93% of SCORE people unseen in FIT, LOOKUP data weight 0.026. LOOKUP is effectively the prior on these.
     - B2TWIN and RETIRED_B3CTL income: 6 vectors with 650-680 FIT people each.
     - MODEL_SELECT has at most 7,500 distinct pair vectors among 8,001 people, because identical raw inputs give identical outputs.

2. **Proposal to you, needed before the grid code lock:** `nrm/readerextra.py` (109eb4571, fc61c797d), selftest 15/15 (unit RR_EXTRA_SELFTEST_V2). The exact devreaders diff is in `roles/reader_robustness/PROPOSAL_GRID_EXTRAS.md`; it is about 8 lines plus adding the file to LOCK_FILES and adds about 3% grid CPU. Per cell, release and view it records three things, none of which changes registered winners or decisions:
   - **(a) Score fingerprints:** SHA-256 of each recipe's SCORE scores, plus its VALIDATION AUC. Without these, `grid_valid` (RULES section 6) can only be computed by code that re-reads SEX.
   - **(b) NOLOOKUP winner:** the winner chosen exactly as the registered one, but excluding the table readers, with paired bootstrap draws. This tests the challenger's requirement that an advantage "holds for smooth readers".
   - **(c) TABLE_CV and an enhanced winner:** TABLE_CV is a frequency-table reader with cross-validated smoothing inside FIT; for the pair it uses an additive key logit with a back-off to the joint table. The enhanced winner is chosen over the 16 locked readers plus TABLE_CV, with ties going to the locked readers.
     - It targets the sparse-reading bias: about 4 FIT people per joint value at 16 x 64 cells.
     - It gives the prior on continuous releases and is deterministic.
     - On invented data its pair AUC is 0.862 against LOOKUP's 0.821. That is a mechanics check only.
     - If you decline (c), call it with `table_cv=False`; (a) and (b) still work.

3. **Robustness plan** (`roles/reader_robustness/ROBUSTNESS_PLAN.md`, b61e9d6c9):
   - **Bank:** candidate, twin, declared alternatives, the bracketing control point Q*, NEURAL_RAW, H_LJ and the lowest-AUC reference.
   - **Repetitions and seeds:** an L-shaped design, one run with R1-R4 at seeds k0/k1 and one with R1/R2 at k2/k3. That is 16 new cells per member instead of 24.
   - **Learning curve:** FIT fractions 0.25 / 0.5, for finite-cell members only.
   - **Monte Carlo:** replicates at offsets +2 and +4 for randomized members, including the registered cells; otherwise those cells carry MC_SE_NOT_ESTIMABLE.
   - **Cost:** about 1.4 CPU-h for a 10-member bank over both versions, plus extras.
   - **Summaries** go through `nrm.rules`: robustness, aggregate, reader_stability, mc_cell and grid_validity, with aliases counted once.

**Remaining for Phase 2**
- **`nrm/readerstab.py`**, the stability-summary code over GRID.json and BOOT.npz, is not written yet: merging runs, grouping Monte Carlo replicates, extras-based paired differences, reader-targeting labels, and an invented-data selftest. It only applies the frozen rules through `nrm.rules`. I will write and commit it, with its selftest, before reading any robustness grid output.
- After that: the robustness grids, `readers/ROBUSTNESS.json` and the report.
- I need from you:
  - the frozen bank;
  - your decision on proposal 2;
  - whether grid validity is computed by you, the verifier or my code.

**Process note:** one shell call at about 15:33Z began with a stray `python3 -` reading empty stdin. It executed nothing and is disclosed in my CPU charge.
