# EDG1 coordinator log (append-only; times from `date -u`)

- 2026-10-25T16:55:33Z: start. DRF1 worktree clean at 913eb5c91; 122 GiB free; YOTUO attached (not touched).
- 16:56Z: isolated worktree `.claude/worktrees/pcrl-estimator-diagnostic-v1` on branch `research/pcrl-estimator-diagnostic-v1`,
  branched from the DRF1 publication commit c4896479e.
- 16:57-17:05Z: read the published source. Read:
  - `drf/finite_law.py`, `drf/solver.py`;
  - `drf/dr.py` `nested`, including `LOOKUP_A = 1.0`;
  - `drf/devreaders.py`;
  - `nrm/fa/prc.py` `lookup_scores` / `auc`;
  - the recipe modules (lock-hash equal);
  - the published aggregates: MODEL_FIT 20,000 people / 13,337 households; folds of about 4,000; MODEL_SELECT 8,001 people;
    supports 16 / 64; twin pair 0.7475 nested exact against 0.6303 LOOKUP.
  - No private file, array, model or label was opened.
- 17:05-17:18Z: wrote `edg/` (laws, estimators, simulate, bank, analyze, selftest, grid), vendored the recipes and wrote
  PROTOCOL.md.
  - Smoke tests used repetition indices 900000+ and tiny bank samples. They wrote to the scratchpad only and printed status and
    timing only: about 0.05-0.08 CPU-s per repetition at n = 20,000.
  - Interpreter: the PCRL venv (numpy 2.4.2 / scipy 1.17.1 / scikit-learn 1.8.0, equal to DRF1's lock, and matplotlib 3.10.8).
    The Homebrew python3 has numpy 2.4.4 and no matplotlib.
- 17:18:07Z: LAWS.json (all analytic checks pass), SELFTEST.json (all pass), GRID.json (49 scenarios; 20 code files locked).
  Registration commit follows. No simulation outcome exists yet.
- 17:18:28Z: registration commit c169705c2.
- 17:18:44-17:31:10Z: grid run (49 cells x 500, 743.5 CPU-s). 17:18:50Z: timing-only BENCH (10 repetitions by the declared rule).
  17:19:00-17:24:50Z: bank subset (348.7 CPU-s). Two heavy processes in total, one thread each. Verifier and statistical reviewer
  spawned at about 17:19Z, with light runs only.
- 17:26Z: found the B rounding-tie detail. It is disclosed in ESTIMATOR_DIAGNOSTIC.md; nothing was recomputed.
- 17:31:22Z: frozen analysis (SUMMARY.json, TABLES.md, two figures). Outcomes committed as 88283f433 at 17:33Z.
- 17:34-17:37Z: drafted ESTIMATOR_DIAGNOSTIC.md. Its quoted numbers were re-checked against SUMMARY.json by script, and 8 quotes
  were corrected before review.
- 17:37-17:39Z: custody rehearsal 1 failed on tooling (record kept), which led to amendment A3. Rehearsal 2 passed (17:39:20Z).
- 17:40Z: publication dry run clean; the builder's own patterns were fixed so the sanitizer no longer alters them.
- 17:44-17:48Z: statistical review round 1 (design; 5 REQUIRED). Post hoc summaries (A4) and report revisions followed.
- 17:55Z: verifier round 1: six of seven checks pass; study defect D1 (B rounding ties) confirmed; the verifier's own two defects
  were kept with corrected runs.
- 18:00Z: statistical review round 2: 3 new REQUIRED and 14 RECOMMENDED, all applied by 18:02Z. The OPT_L bracket support was
  re-checked from DRF1's published records.
- 18:04Z: verifier round 2: reproduction and seed re-runs exact; 62 of 64 report items matched; the two rounding slips were fixed.
- 18:06Z: closeout sections written. The final backup, restore check, off-device copy and publication follow.
