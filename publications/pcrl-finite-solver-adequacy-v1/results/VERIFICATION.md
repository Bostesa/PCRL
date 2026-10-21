# Verification (campaign `pcrl_finite_solver_adequacy_v1`)

Summary by the coordinator, 2026-10-21. Four independent checks were made:

| Who | What | Record |
|---|---|---|
| Independent verifier | Recomputed the decisive numbers with its own code. It imported no `fsa`, `frc` or `nrm` numerical function for a verified quantity, and called the frozen `fsa.rules` functions only on its own inputs. | `roles/verifier/CHECKS.json` (57 append-only, hash-chained checks; the chain verifies); final report `roles/verifier/REPORT_FINAL_0345.md` (verbatim) |
| Statistical reviewer | Re-derived every label with the frozen rule functions from the saved inputs, and checked the decision's wording | `roles/statistical_reviewer/CLOSING_STATEMENT.md` |
| Engineering and claims reviewer | Regenerated `tables/` from the records in a scratch copy (`diff -r` empty), ran the solver selftest (9/9) and the rules tests (15/15), and checked every number in the documents | `roles/engineering_review/REVIEW_CLAIMS.md` |
| Custody restore check | Rebuilt from the backup alone: source and private-output hashes, a bitwise re-route of the 72 saved books, the objectives and replay means, the solver selftest, and the route S and advancement labels | `custody/BACKUP_VERIFICATION.json` |

## The verifier's totals

| | PASS | FAIL | ATTENTION | NOT_APPLICABLE | NOT_RUN |
|---|---|---|---|---|---|
| Numerical | 34 | 0 | 0 | 1 | 2 |
| Procedural | 8 | 1 | 9 | 0 | 2 |

There is no numerical FAIL. The one procedural FAIL is the verifier's own path-mapping bug, superseded by a rerun.

## Decisive results

- **P1 replay (V2, both versions).**
  - The replay record matches the previous campaign's `SELECTION_V<v>.json` over 809 leaves (largest difference 0.0; every discrete
    output equal).
  - The verifier's own recomputation from the per-person outputs:
    - 275 nested mean values per version, difference 0.0;
    - spends and household-clustered SEs, 288 leaves, within 2.2e-19;
    - the selection, the brackets, `nested_lookup_pair` (73 leaves) and F4, bitwise equal.
  - No margin lies within 1e-9 of a threshold, so there are no boundary ties.
  - F4: 0.007152675 (SE 0.002245231) and 0.011134730 (SE 0.002313061), entry PRC@8.0, passing in both versions.
  - `fsa.rules.p1_replay` on its recomputation gives `P1_REPLAY_VERIFIED` in both versions.
  - Two label reads, operation `verify`, MODEL_FIT only, after the receipt (FSA_VER_P1_V0_V2 at 03:29:35Z, FSA_VER_P1_V1_V2 at
    03:29:54Z).
- **Route S and the advancement (V6).**
  - The verifier's nested inputs equal `decisions/ROUTE_S_INPUT.json` with 0 mismatches. In both versions the revised point is
    PRCsv1@8.0, the original S2 point PRC@8.0 and the bracket PRC@2.0.
  - Paired differences (revised minus original):

    | Version | Pair AUC vs bracket | Total NLL vs bracket | Total NLL vs the original S2 point |
    |---|---|---|---|
    | 0 | -0.00181 (se 0.00201) | -0.00061 (se 0.00060) | -0.00061 (same partition) |
    | 1 | -0.00382 (se 0.00170) | +0.00034 | -0.00072 |

  - Pooled pair AUC -0.00281 with pooled se 0.00185. The estimate plus 2 SE is +0.0009, not below 0.
  - `route_s` gives `S_PROXY_ONLY` on both the verifier's input and the recorded input. `advancement` gives
    `OPTIMIZER_REPAIR_CLOSED` with the verifier's S input and the recorded D and E inputs.
- **Exact laws (V3).**
  - All five laws pass: 253 release records and 740 objective values recomputed from `laws.json`, largest difference 5.8e-14.
  - The verifier's own enumeration of D_witness (1,296 assignments, 36 feasible) gives the investigator's optima.
  - Route E: 60 cells, 240 reported partitions and 60 unrestricted objectives (largest difference 4.4e-16). The usefulness deltas
    match ADVANCEMENT.json to 2.2e-16.
- **Source check and audit (V4).**
  - The 72 saved states (both versions × PRC / HQ × original / SV1 / SV1_first × 6 strengths) give the recorded objective within
    8.9e-16. Partition hashes, B1 and prototype cross-entropies, information, guards and re-routing shares all equal the record.
  - The verifier's own routing of each saved book reproduces its partition in 72 of 72 states.
  - The smallest gap between a person's best and second-best routing cost is 4.7e-9 (version 0) and 2.5e-9 (version 1). That is
    small, but far above arithmetic noise. The coordinator's knife-edge check (a 1e-9 relative tilt change moves no one) agrees.
  - SV1 lowers PRC's fitted objective at every strength: by 1.2e-4 to 5.6e-3 in version 0 and 3e-6 to 1.05e-2 in version 1. This
    concerns the surrogate objective only.
- **Nested comparison (V5).** Means and AUCs differ by 0.0, and spends with SEs by at most 1.1e-19. The 13 original-solver entries
  equal the replay in all 117 arrays per version.
- **Receipt, admission and access (V1).**
  - The verifier's own `git ls-remote` returns the receipt commit a771587ca. The 8 registration files are byte-equal across the
    published copy, f301965a7, HEAD and the worktree.
  - All 72 pinned artefacts re-hash to their pins.
  - There were 6 reads, all after the receipt and all on MODEL_FIT, with no closed role and no MODEL_SELECT. Card FA_FSA_v2 was
    committed before every fit read. The audit and source units read no label.

## Non-PASS items and their dispositions

| Check | Status | Cause | Disposition |
|---|---|---|---|
| V1.ADMISSION__V1 | procedural FAIL | the verifier's own path-mapping bug | superseded by V1.ADMISSION__V2 |
| V1.ADMISSION__V2 | ATTENTION | an untracked empty lock file left by the previous campaign's verifier | disclosed in that campaign; lock files are not published |
| V1.ACCESS_LOG__V1, __V2 | ATTENTION | the four label-reading units launched while verifier files were uncommitted, outside their import closures | `PROTOCOL_DEVIATIONS.md` entry 3; no stage imported them |
| V2.ORIGINAL_CODE × 2 | ATTENTION | since the original P1 run, `emit` in `frc/fa/route.py` dispatches through `route_any` | the statistical reviewer checked the git diff: behaviour-preserving for every non-UNT book, and P1 has none. It is an erratum to the RULES 8 pin (CLOSING_STATEMENT addendum 1). |
| V2.PROVENANCE × 2 | ATTENTION | uncommitted verifier code at launch; files outside the import closure changed during the run (for example `fsa/fa/advance.py`) | `PROTOCOL_DEVIATIONS.md` entry 3; no effect on the replay |
| V2.ESTIMATOR_TEXT × 2 | ATTENTION | RULES 8 asks for a "textually identical" selection loop. The replay's copy renames three local variables (`sel2`, `k2`, `v2`) and is AST-identical after renaming. | The reviewer ruled that `estimator_unchanged = True` stands, giving `P1_REPLAY_VERIFIED`. This is disclosed as a reading made after the result. The strict text gives `P1_ESTIMATOR_CHANGED`. No decision reads this label. |

## What was not verified independently

- **Numerical NOT_APPLICABLE.** The `nested_means` keys that need training-side partitions (six keys) cannot be recomputed from
  held-out outputs. The verifier's P1-structure record therefore omits `nested_means`. The replay itself reproduces all 413 means
  bitwise.
- **Numerical NOT_RUN.**
  - Re-enumeration of the tilt-line breakpoints, and full enumeration of the m = 6 and m = 8 laws.
  - The audit's tilt-line Spearman and interval values, the runtime in-sample against out-of-fold agreement, the expressivity counts
    and the refresh deltas.
  - These audit numbers rest on the coordinator's audit code. The engineering reviewer confirmed only that the tables reproduce
    from the records.
- **Procedural NOT_RUN.** The advancement's D and E inputs were taken as recorded, and `decisions/EQUAL_EFFORT_ATTRIBUTION.json`
  was not checked. The statistical reviewer re-derived the D and E labels from those inputs, and the engineering reviewer checked
  the effort numbers in the tables.

## The verifier's own disclosures

- **Failed units, all its own bugs, each rerun under a new id.**
  - FSA_VER_P1_V0_V1 and FSA_VER_P1_V1_V1: an import-path bug, before any label read.
  - FSA_VER_ACCESS_V1: exited 3 after recording its checks.
  - FSA_VER_EXACT_V1: crashed before recording anything.
- **Rules draft.** Before registration it read the uncommitted RULES.json draft to plan. Every real run took its tolerances from the
  committed rules.
- **CPU.** 14 queue units used 59.0 CPU-s, and a further 182 CPU-s were charged for out-of-queue selftests, dry runs and rule
  application.
- **Git.** It committed only under `roles/verifier/`, with plain messages, and did not push. `records/.checks.lock` is left
  untracked on purpose.
