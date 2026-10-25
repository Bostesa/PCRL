# EDG1 independent verifier: round 1 handback

Campaign `pcrl_estimator_diagnostic_v1` (EDG1). Role: independent verifier; no agents spawned.

Clock, read with `date -u`:
- work started 2026-10-25T17:20:04Z;
- this file written 2026-10-25T17:54:54Z.

## Bottom line

- Six of the seven requested checks pass: 1, 2, 4, 5, 6 and 7. Check 6 passes after a correction of my own check (D2 below).
- Check 3 passes on merging and on tie handling in the population optimum A, but finds one **study defect** (D1):
  - the known-score estimators B_same and B_indep do not score exact ties as 1/2;
  - the cause is float rounding of the posterior in `edg/laws.py`.
- The defect makes B's paired difference non-zero in **15 of the 32 dA = 0 comparisons**. With exact ties it is exactly 0 on every
  sample.
- A, C_same, C_nested, D_nested and D_bank are not affected.
- I made two defects of my own (D2, D3). The failed outputs are kept, with corrected runs beside them.

## How: the independent method

- **Exact arithmetic.** Rational arithmetic (`fractions.Fraction` and Python integers) throughout.
- **Brute force with ties counted 1/2.** Every AUC is computed by enumeration:
  - population: the sum over every ordered pair of wire values of P1(a) P0(b) ([s(a) > s(b)] + 1/2 [s(a) = s(b)]);
  - sample: every (positive, negative) pair of people, or every pair of wire values weighted by integer class counts. The second
    route was checked equal to the first.
- **Own components, independent of the study code:**
  - wire merging by the exact float64 bytes of the emitted vectors;
  - exact masses from the exact binary values of the study's float p and q;
  - an inverse-CDF sampler with my own seeds. No study seed was used; `simulate.sample_rep` was called only at repetition index
    777777.
- **Own known laws for every family** (IND, REF, RED, COMP), built from PROTOCOL.md section 3 in exact rationals:
  - rational uniform and Zipf occupancies with my own permutations;
  - a 60-digit decimal sigmoid, with bisection for alpha (marginal error below 1e-59);
  - exactly centred u and v for COMP.
- **What I imported from the study.** Only `edg.laws.build`, to get its law arrays. As the objects under test: `edg.estimators`,
  `edg.bank`, `drf.solver`, `drf.finite_law`, `edg.source_prc` and the vendored recipes.
- **Library validation.** My library was first validated on hand-computable cases (CHECK0).

## Results

| check | status | key evidence (own samples / exact arithmetic) |
|---|---|---|
| 0 own library self-test | PASS | AUC 3/5 and 11/20 hand laws, pure wires, ties, the enumeration maximum, merging, the 60-digit solve |
| 1 independence gives exactly 1/2 | PASS | 4 study and 4 own IND laws (12 + 12 views): P1 == P0 exactly, and exact AUC = 1/2 for the optimal ranking, random tied and untied fixed scores, and fitted LOOKUP tables. Study float values (A, `auc_of_score`, d_nested / d_bank pops) are within 3.2e-15 of 1/2 |
| 2 known-score AUC vs brute force | PASS | 1,456 cases over all laws and views, n = 20 to 1,000: `auc_if` / `b_known` within 1.7e-16; IF equals brute-force DeLong placements (5.6e-17); person-unit SE (4.3e-16 relative); a missing class gives NaN, never 1/2 |
| 3a supports and partitions | PASS | Own merging reproduces every support (IND 16/64/1024, REF 16/64/256/1024, RED 16/64/256, COMP 16/15/240) and the study's partition (a bijection) |
| 3b COMP merge (cells 14 and 15) | PASS | The two vectors are bitwise identical; supports 15 / 240; exact A on unmerged internal ids exceeds merged A by 0.0009-0.0019; own uniform COMP gives the study's A(pair) exactly: 0.5828515625 and 0.603564453125 |
| 3c tied likelihood ratios in A | PASS | REF coarse q has 8 levels with atoms 2k / 2k+1 bitwise tied; the study float A equals the exact tied-ratio A within 1.5e-15 |
| 3d ties 1/2 in B | **FAIL (study defect D1)** | See D1 |
| 4 irrelevant refinement | PASS | 4 study and 4 own REF laws: exact A identical (Fraction equality) at L16, L64, L256 and L1024; nested partitions; every child's exact posterior equals its parent's; study A range 0.6708-0.6814; an exact rebuild of the study laws matches within 3.3e-16 |
| 5 RED / COMP relations | PASS | RED: A(pair) == A(income) exactly, A(income) > A(occupation), and every pair posterior equals its income posterior exactly. COMP: single views exactly 1/2 (own laws; study float law within 1.1e-17), A(pair) 0.563-0.604. All 48 registered comparisons are classified correctly (null abs(dA) <= 1.8e-15; non-null >= 0.0081) |
| 6 static label-access review | PASS | 37 cited lines, each verified by script to contain the quoted code; the reviewed files equal the registration lock |
| 6a estimator behaviour, original run | FAIL (kept; my defect D2) | Superseded by r2 |
| 6a corrected run r2 | PASS | See below |
| 6b reader bank | PASS | See below |
| 7 provenance | PASS | See below |

**6a corrected run (r2)**, on 728 own samples over all laws and views, n = 200, 2,000 and 20,000:
- C_same equals `drf.solver.auc_star`, `drf.finite_law.auc_star` and my exact value (within 9.8e-15).
- C_same >= B_same on every sample: 0 violations, exact, for both the float B and the exact-tie B; the smallest C - B was 6.6e-5.
- C_same equals the maximum over all rankings on 93 enumerated tiny tables.
- C_nested is exact (7.8e-16).
- D_nested and D_bank equal brute force on the readers' float scores (1.2e-15).
- Every `lookup_scores` call received exactly the ids and labels of folds != f.
- Relabelling the test fold, or the SCORE part, changed no fit, orientation, VALIDATION AUC or population AUC.
- `sample_rep` never splits a household across folds or parts, and the parts are 50 / 20 / 30% of units.

**6b reader bank**, on RED|0.5|skewed and REF|0.5|skewed L16, n = 2,000, all 16 recipes, a non-study reader seed:
- SCORE-label perturbation left every recipe score, VALIDATION AUC, orientation, convergence flag, population AUC and winner
  bitwise unchanged.
- Refits without the appended support rows gave bitwise-identical scores.
- The recipes' population AUCs equal brute force (within 3.3e-16).
- Winners equal my own implementation of the rule: maximum VALIDATION AUC, exact ties to the earlier candidate.
- The test has power: perturbing VALIDATION labels changed all 16 VALIDATION AUCs.
- `_src_tab` and `_fit_one` are text-equal to `drf/devreaders.py`.
- `run_bank` was not called, because it writes into the study's `out/`; it was reviewed statically.

**7 provenance**:
- The 10 vendored files equal SOURCE_PROVENANCE.json and the blobs at 913eb5c91.
- The 5 lock-listed recipe files (and `drf/devreaders.py`) equal DEVGRID_CODE_LOCK.json.
- The `prc.py` blob hash matches, and both copied functions are text-equal.
- The `drf` files equal provenance and the blobs at c4896479e.
- LAWS.json hashes, supports and A match (A within 1.6e-15 of my exact value).
- Every change since registration is disclosed in A1-A4, and no estimator or analysis file changed.

All of this is in `CHECKS.json` and `out/`.

## Defects

### D1 (study defect): B's known score splits exact ties (CHECK3 3d)

**Cause.** `edg/laws.py` `build` sets `post = m1 / (m1 + m0)` from float masses, and `edg/estimators.py` `b_known` ranks that float.
Wires whose posteriors are exactly equal get posteriors that differ in the last bits, so `b_known` orders pairs it must tie.

**Scope.**
- 18 of 52 views split the float law's own exact ties. Example: REF|0.5|skewed L16 has 8 exact levels but 15 float values.
- At least 24 of 52 views split the specification's ties. This adds the COMP income / occupation views (u and v are centred only in
  float) and the RED|*|skewed occupation views.
- The affected views are in the skewed REF and RED laws, IND|0.2|skewed, and every COMP law.

**Consequence: 15 of 32 dA = 0 comparisons.** In these, B's paired difference is not identically 0 (the exact-tie known score gives 0
on every sample):
- IND|0.2|skewed: 3;
- REF|*|skewed: 6;
- RED|*|skewed pair-income: 2;
- COMP occupation-income: all 4 laws.

**Effect size**, on own samples:

| n | SD of d up to | abs(d) >= 0.020 (M20) up to | R2SE resolved, pooled | R2SE, worst comparison |
|---|---|---|---|---|
| 2,000 | 0.022 | 37% of samples | 89/2,100 (4.2%) | 10% |
| 20,000 | 0.0093 | 5% | 45/780 (5.8%) | 25% (R = 20) |

The exact-tie known score gives 0 for all of these.

**What is not affected.**
- B remains unbiased for A: mean deviations are within about 2 MCSE of 0, because a tie-broken optimal ranking is still
  population-optimal.
- A is not affected (its tie handling only touches terms of order 1e-17).
- C is not affected: integer-count ties divide to identical floats.
- D is not affected (see N1).

**Implication for the study.** In the affected views and comparisons, B rows report extra variance and non-zero false-positive / M20
rates. These are rounding artifacts, not properties of the known optimal score. The B_indep bias column is still valid.

**Suggested fix.** Compute the known posterior so that exactly equal posteriors are equal floats, then recompute the B rows:
- exact rationals from the specification mapped to one float per value;
- or, where every cell of a wire shares q (REF, RED pair / income, IND), the internal q itself.

COMP also needs exactly centred u and v. C and D rows need no change. I have not read the report or POST_HOC.json (that is out of
round-1 scope). Any statement there that relies on B's behaviour in these views should be re-checked, including A4's "B coverage
in the non-null comparisons".

### D2 (my defect): CHECK6a original run, kept as a FAIL

**What went wrong.** I compared D_nested with an exact-rational idealization of the DRF1 LOOKUP reader. They differed by up to
2.2e-5 in 4 of 52 views.

**Diagnosis** (`out/DIAG6a_dnested_ties.json`). Brute force on the reader's own float scores reproduces the study's values exactly.
The difference comes from 10 shrunk rates (s1 + a rbar) / (n + a) that are equal as rationals but round to different floats.

**Why this is my defect.** A fitted reader is its source float code, so the study is right and my check was wrong. The corrected run
`out/CHECK6a_estimators_r2.json` judges D by the float scores and keeps the exact-idealization difference as a NOTE. Its seeds and all
other parts are unchanged.

### D3 (my defect): first CHECKS.json summary text

`CHECKS_v1_superseded.json` (kept) quoted an R2SE range for D1 that I had estimated by eye ("about 2-10%"). The current
`CHECKS.json` computes the ranges from the recorded values. No status changed.

## Notes

- **N1.** The D_nested reader orders some exactly tied cells by float rounding (at most 2.2e-5 AUC here). This is the DRF1 source
  reader's own behaviour, reproduced faithfully, so it is not a study defect.
- **N2.** The static review file `out/CHECK6_code_review.json` reads FAIL in its own status field. The only reason is that its lock
  list included `selftest.py`, which differs from the GRID.json lock. That is the disclosed amendment A3 (two lines, tooling).
  The study's other post-registration changes are also disclosed:
  - `edg_run.py` (A1, A2, A4);
  - new `custody.py`, `publish.py` and `posthoc.py` (A1, A2, A4).

  `estimators.py`, `laws.py`, `simulate.py`, `bank.py`, `analyze.py`, `source_prc.py`, `grid.py`, `__init__.py`, GRID.json and
  LAWS.json are unchanged since registration (c169705c2). The coordinator kept committing during my work (HEAD d3cc386d8 at
  17:49Z). My checks ran on these unchanged files.
- **N3.** PROTOCOL.md section 5 lists six bank scenario cells; `bank.py` has five BANK_CELLS entries, because REF|0.5|skewed carries
  the two paired views L16 and L256. This is consistent.
- **N4 (integrity disclosure).** Early on, one read-only command used the Homebrew `python3` instead of the venv interpreter, to print
  GRID.json fields. It read only that file in W, computed nothing, and wrote nothing. Everything else used
  `<home>/PCRL/.venv/bin/python` with `-B`, PYTHONDONTWRITEBYTECODE=1 and one numerical thread. No bytecode was written.
  Nothing outside W was opened except that interpreter, `date`, read-only `git show` / `git log` / `git diff` / `git status`, and
  one `ps` listing to confirm the two heavy processes. No existing file in W was edited.

## CPU

- 53.4 CPU-s in total, recorded with `resource.getrusage` in each output JSON, over 16 runs, one process at a time.
- The largest single run was 19.1 s (CHECK6b); every run was under 20 s.
- About 0.5 s more went to unrecorded exploratory one-liners: tie counts, a timing probe, AST scans and JSON prints.

## Files (all under `roles/verifier/`)

- `HANDBACK_1.md`.
- `CHECKS.json` (v2) and `CHECKS_v1_superseded.json`.
- `code/`: `vlib.py`, `check0` to `check7` scripts, `check3b`, `check3c`, `check6a_estimators.py` (original),
  `check6a_estimators_r2.py`, `diag6a_dnested_ties.py`, `check6_code_review.py`, `make_checks.py`.
- `out/`: `CHECK0` to `CHECK7` JSONs, including `CHECK6a_estimators.json` (FAIL, kept) and `CHECK6a_estimators_r2.json` (the
  correction), `DIAG6a_dnested_ties.json`, and the two CHECKS build records.

## Not done

Round 2 was not started. No `out/*.jsonl.gz` file was opened, and no grid or bank repetition was re-run.
