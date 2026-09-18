# VERIFICATION_STATUS — what has been checked, by what, and what each check cannot show

Branch `research/pcrl-manuscript-integrated-v2`. Historical artifacts are preserved unedited;
everything below is either a pointer to a committed check or a companion artifact published beside
a stale one, never a silent replacement.

## 1. The stale verifier, resolved

`results/redesign_20260917_acs_spectral_transport_v1/INDEPENDENT_VERIFICATION.json` is a **correct
artifact that became incomplete**. Its `lock_inputs_unchanged_or_amended` check lists
`code_only_amendments` for amendments 1 and 2 and reports `changed: []`. Lock amendment 3 was
created at `2026-09-17T18:30:13Z`, after that artifact was written, so its `changed: []` verdict
was computed against the pre-amendment-3 report code.

`results/pcrl_evidence_review_v1/LOCK_RECHECK.json` was the predecessor's re-run. Its `note` field
states the situation correctly, but two of its own fields contradict that note: it records
`amendments_covered_by_published_verification` as all three and
`amendments_not_covered_by_published_verification` as empty. Read against
`INDEPENDENT_VERIFICATION.json`, which names only amendments 1 and 2, those two fields are wrong.
The file is preserved as committed; the corrected companion supersedes those two fields only.

**Corrected companion:** `CORRECTED_INDEPENDENT_VERIFICATION.json` in this directory, regenerated
locally, read-only, on one worker.

| | |
|---|---|
| Inputs hashed | **17,639 of 17,639** (1.80 GiB / 1,936,177,415 bytes) |
| Inputs changed after all three amendments applied | **0** |
| Inputs missing | **0** |
| Invalid amendment entries | **0** |
| Non-code paths re-bound by any amendment | **0** |
| Amendments covered by the published artifact | 1, 2 |
| Amendments **not** covered by the published artifact | **3** |
| Final-partition accesses | 10, from `2026-09-17T16:46:33Z` to `2026-09-17T17:54:45Z` |
| Final reads after amendment 3 (`18:30:13Z`) | **0** |
| Verdict | `pass_with_all_amendments: true` |
| Measured runtime | 5.1 s wall, one worker, one BLAS thread |

Amendment 3 re-binds `scripts/report_acs_spectral_transport.py` for a CSV → gzip-CSV serialisation
change, affects no score, selection, interval or decision, and post-dates every read of the final
partition. The count is therefore **three amendments, all code-only**, and the pass/fail wording is
reconciled: the source-probe columns count **passing** seeds, and the corrected verification
**passes** with all three applied.

Regeneration, from a checkout where the study's local artifacts resolve:

```bash
PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.recheck_verification \
    --study-root /Users/nathansamson/.config/superpowers/worktrees/PCRL/residual-spectral-20260910 \
    --published results/redesign_20260917_acs_spectral_transport_v1 \
    --out results/pcrl_manuscript_review_v2
```

The `--study-root` is read-only: the script hashes files and writes only into `--out`. The locked
inputs are fitted objects, releases and per-person arrays that stay local and are not tracked, so
they resolve only in the worktree where the study ran.

## 2. Checks inherited from the two completed studies

| Check | Result | Source |
|---|---|---|
| All 90 Study-1 endpoints recomputed by a separately written scorer | max abs diff **1.11e-15** (estimate), 2.40e-15 (interval endpoints); identical critical values | `pcrl_evidence_review_v1/FAMILY_AGREEMENT.json` |
| Study-1 candidate selection re-derived from stored validation losses | **0 mismatches** in 924 cells | `pcrl_evidence_review_v1/SELECTION_RECHECK.json` |
| Cross-interpreter reanalysis (CPython 3.13.7 and 3.14, different NumPy builds) | agree to the tolerances above | `pcrl_evidence_review_v1/VALIDATION.md` §1 |
| Study-1 released service vectors identical bitwise | **210 of 210** views, 0 violations | `INDEPENDENT_VERIFICATION.json` |
| Study-1 post-fix prediction replay | 0 mismatches in 59,874 predictions; 0 selection changes in 37,122 validation predictions | `PREDICTION_REPLAY.json`, `VALIDATION_SCORE_AUDIT.json` |
| Study-2 cross-process prediction replay, machine at swap capacity | **27,540 checked, 0 mismatches**, bitwise | `pcrl_nonlinear_rank_v1/PREDICTION_REPLAY.json` |
| Study-2 score replay | 2,112 checks, max abs diff 4.44e-16 | `pcrl_nonlinear_rank_v1/SCORE_REPLAY.csv` |
| Study-2 comparison arithmetic verified from the published CSVs | `all_clean: true`; additional-recovery identity exact over 34,848 comparisons | `pcrl_nonlinear_rank_v1/COMPARISON_VERIFICATION.json` |
| Study-2 rotation feasibility | `max abs` orthogonality violation ≤ 8.9e-16 | `pcrl_nonlinear_rank_v1/DIAGNOSTICS_SUMMARY.json` |
| Mathematical fixtures and sealed-year loader tests | 21 passed | `tests/pcrl_evidence_review_v1/` |
| Nonlinear-moment fixtures | see `tests/pcrl_nonlinear_rank_v1/test_nonlinear_moment.py` | — |

## 3. Checks performed by this terminal

| Check | Result |
|---|---|
| Corrected lock verification with all three amendments (§1) | pass, 0 changed, 0 missing |
| Every manuscript table and figure regenerated from committed evidence, each source sha256-recorded | `papers/pcrl_manuscript_v2/MANIFEST.json` |
| Denominator audit of the F1 "8 of 8" claim against the frozen decision record | **14 of 16 better, 0 worse, 2 unresolved** — correction A1 |
| Attribution-family denominators recounted from `PAIRED_INTERVALS.csv` | one significantly worse cell found — correction A2 |
| Rotation stop reasons inspected | `budget_exhausted` / `line_search_failed_no_improvement` — correction A3 |
| Nuisance improvement recomputed over all roles and seeds | 0.03–0.66 % (sex), 0.96–2.35 % (race) — correction A8 |
| PDF compiled and every rendered page inspected | 22 pages, 0 overfull/underfull boxes, 0 undefined references or citations, 0 LaTeX warnings |
| Integration: both study trees preserved, tests collected, build reproducible | `INTEGRATION_MANIFEST.json` |

**No model was fitted, refitted, scored or rescored by this terminal**, and 2016 was neither scored
nor inspected.

## 4. What none of these checks establishes

* **That the attack families are strong.** Every recovery number is what the tested families
  achieved — a floor, not a bound.
* **That the moments certify anything.** They do not: vanishing fitted finite moments imply neither
  `Z ⊥ S | H` nor any bound on `I(S;Z|H)`, and no calibration is inherited from KCI or RCoT.
* **That the refined objective was optimised well.** Three checks exclude three specific failures.
  Ky Fan's optimality is for the *original* fixed matrix only.
* **That 2017 respondents differ from 2018 respondents**, or 2016 from either. Public identifiers
  cannot support this.
* **That memory pressure caused the original numerical corruption.** It is the observed condition
  under which it appeared. The repair does not depend on the diagnosis.
* **That the surveyed prior work is exhaustive.** A bounded search cannot establish novelty.
* **Anything about a Terminal 1 experiment.** None exists at the time of writing; see
  `HANDOFF.json` and `PENDING_ADDENDUM.md`.
