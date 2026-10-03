# Validation

## 1. Before any fit

| Check | Result |
|---|---|
| Code freeze and tests | `stored_model_eval` suite 97/97 (pilot tests included). Official LEACE install = upstream tag v0.2.4 tree (sha256 fffac29d…); upstream tests 27/27. |
| Private-path scrub | `LOCK.json`, scripts and notes hold no absolute private paths: they are home-relative or `<drive>`/`<repo>` placeholders. The report stage refuses to write tables that contain the private root. |
| Lock | Built 2026-10-02 20:57:48Z, committed and pushed as 3274fe1 (remote SHA verified) before the first fit at 20:59:19Z. It pins 39 code files, 8 dependencies, the inputs index, the support freeze, 882 unit IDs, the 24-endpoint family and the inference settings. |
| Dry run under the lock | OK (no fits) |
| Shuffled-label sanity (fit and validation rows only; seed 20261005; 6 arm-A units) | Validation AUC: nonlinear 0.47–0.51, linear 0.485–0.511. Nothing flagged; no assessment rows used. |
| Synthetic validation of the independent replay | 64/64 checks (rerun on the final replay code, sha256 3dc83ceb…). 19/19 planted defects detected. Null coverage of the derived worst-class bound is 0.945 (nominal 0.95); the naive bound of the max gives 0.33. |

## 2. Run integrity

| Check | Result |
|---|---|
| Units | 435/882 complete and hash-verified (`COMPLETE.json`). Tier 1: 126/126. Tier 2: E1 72/72, E2 36/36, E3 201/648. |
| Unrun units | 447, all E3, stopped by the frozen rule. The stop is recorded in `logs/BUDGET_LEDGER.json` and `notes/evaluation/CONSUMPTION.json`. |
| Tier-2 trigger | Open (`notes/execution/TIER2_TRIGGER.json`), evaluated by the runner and re-evaluated read-only: every Tier-1 unit complete; arm-A native check reproduced on all 6 seeds (\|Δ\| ≤ 3.4e-6); B/C implementation bound holds on all 12 maps; U1 consistent; σ\* present. |
| Arm-A native check, all pairs | 42/42 reproduce the stored `dominant_axis_audit.json` values (\|Δ\| ≤ 4.3e-6) |
| LEACE maps | 60 pinned (42 B + 18 C), each referenced by its units; implementation bound holds on all 60. A refit that differs from its pin is refused. |
| Aliases | No C map equals its B map (rule: max\|P_B − P_C\| ≤ 1e-10). The closest is HMDA s1 underwriting at 1.2e-10, which is not an alias by the frozen rule. |
| Effective-protocol consumption | 253 leaves: 180 consumed, 72 descriptive, 7 unsupported, 0 not consumed |
| Primary family file | The report stage rewrites `PRIMARY_FAMILY.json` with re-wrapped keys plus B, seed and tail count. Its 24 endpoint definitions are identical to the pre-fit file. The committed pre-fit file is kept unchanged; the report-stage copy (the one the replay read) is `notes/execution/PRIMARY_FAMILY_report_stage.json`. |
| Resume | `STAGE=tier1 RESUME=1` skipped 126/126 hash-verified units with 0 fits and the lock check OK |

## 3. Independent replay

**What the replay is.** `verification/replay_bench.py` runs on numpy only. It does not import `stored_model_eval`, does
no refits, and recomputes every quantity from the saved predictions, maps, roles and labels.

**Result** (`INDEPENDENT_VERIFICATION.json`): 100,954 checks in 83 min.

| Status | Count |
|---|---|
| PASS | 100,511 |
| INFO | 185 |
| NOTE | 34 |
| MC_BORDERLINE | 38 |
| NOT_RUN_BUDGET | 1 (the 447 E3 units, by design) |
| FAIL | 185 |

**The primary family reproduces completely.** All 183 primary checks PASS at B = 20,000: 24 endpoints with their
points, simultaneous bounds and decisions, family size and α. σ\* (Adult 2.0, HMDA 4.0) is reproduced from the
attacker_val predictions alone.

**Every FAIL is in an exploratory statistic.** None changes a primary decision or a conclusion in RESEARCH_DECISION.md.

| Count | Cause | Diagnosis | Effect |
|---|---|---|---|
| 143 | **Runner defect: unpaired seed composition in a budget-truncated pair** | Adult employment_analysis/marital_status, σ ≥ 0.5. The budget stop left encoder seed 2 unrun there, so those D groups average seeds 0–1. The runner's D−A and plus−outputs contrasts subtract an A/CELL mean over seeds 0–2. The runner's error equals that seed-composition gap exactly (0.013953; re-derived to 1e-13). | Exploratory E3 rows for one pair only. RESEARCH_DECISION already treats these values as not comparable. The corrected, seed-paired values are in `verification/replay_results_aggregate.json`. |
| 34 | **Runner numerical loss in ρ₁²** | Adult education_assessment, arm C. The saved canonical scores have \|mean\|/sd up to about 4e6, so the runner's uncentred moments lose about 1 % of precision. For s1 the runner's own interval excludes its point. | Exploratory, values about 0.001–0.005. Conclusions unchanged. |
| 8 | **Monte Carlo exceedance** | 5 distinct exploratory statistics exceed the frozen 6.5-MC-SE tolerance by 0.3–3.5 %. Against a B = 20,000 reference, the runner's B = 2,000 bounds sit 1.3–3.4 SD from the reference quantile, which is ordinary spread over about 37k bound comparisons (`verification/mc_exceedance_reference.json`). | None |

**Other non-PASS items:**
- **38 MC_BORDERLINE:** exploratory decisions whose bound lies within MC tolerance of the bar.
- **34 NOTE:** the descriptive `defense_fit_cov_rank` column depends on the rank tolerance; the replay reproduces each
  map's own `sample_cov_rank`.
- **185 INFO:** informational items.

**Replay mapping adaptations.** These are listed in the aggregate JSON. They changed only how runner rows and files are
mapped; the mathematics was not changed. They cover:
- the real file and ID formats;
- the Lslate seed-mean for rep+outputs "L";
- ddof-0 cross-covariance;
- the 1e-12 log-loss clip;
- the worst point taken as the max of seed-mean class AUCs.

**These defects were not fixed after reading results.** The runner code is locked. Changing it after the run would be
a post-hoc analysis change. The defects are disclosed here, and the corrected exploratory values are kept in the
verification outputs.

## 4. Owner spot checks

These were computed directly from the saved predictions with sklearn, independently of both the runner and the replay:

| Endpoint | Spot check | Runner |
|---|---|---|
| P-adult-Rrep-A | 0.815397 | 0.8153969 |
| P-adult-Rrep-B | 0.817359 | 0.8173594 |
| P-adult-U2NI-B | −0.000444 | −0.000444 |
| P-hmda-Rrep-A | 0.8702396 | 0.8702396 |

P-hmda-Rrep-A is one-vs-rest over the supported classes {0, 1, 2}, scored on all assessment rows, as frozen. My first
attempt restricted the rows to the supported classes and gave 0.8713, a different estimand.

## 5. Anomalies investigated (no defect)

- **Adult untreated nonlinear AUC 0.815 against the pilot's 0.684.** This is a mean over encoder seeds,
  0.669 / 0.912 / 0.866. Seed 0 (0.669) is consistent with the pilot (0.684), allowing for attacker-seed averaging.
- **Constant D−A for rep + clean outputs across σ.** Once the noise destroys the representation, the selected attacker
  is the outputs-only attacker, and the outputs are unchanged across arms.
- **HMDA fair_lending/sex: nonlinear AUC identical across A, B and C on seed 0, and within 0.0002 on seed 1.** The representations have only
  6,703 / 4,742 distinct rows among 77,408. A lookup on those rows is unaffected by an affine eraser.
- **HMDA s2 fair_lending/sex: linear AUC 0.36 (below chance) next to a positive held-out R² (0.069).**
  - About 10 % of rows form a linearly separable cluster that the logistic model predicts at about 1.0, correctly.
  - The remaining 90 % receive probabilities in 0.56–0.60, ranked against the label.
  - Log loss improves on the prior (0.613 vs 0.665), which is the quantity selection uses, while AUC falls below 0.5.
  - This is a property of the selected model, not a defect.
- **Large negative G1 and G2 under LEACE or on rank-deficient seeds** (for example Adult s1 G1 = −0.57). These are the
  correct held-out R² of a nearly unregularised predictor fitted to almost no signal (see INTERPRETATION_CORRECTION.md).
  G1 is unclamped by protocol.

## 6. Backup and restore

- The drive copy is `<drive>/private_bench_v1_20261003/`: 9,764 files.
- Every file was re-read uncached (F_NOCACHE): **9,764/9,764 sha256 match at backup; 9,765/9,765 after the log re-sync**.
- Loaded from the drive copy only, the official LEACE map for HMDA s0 underwriting/race and two attackers
  (`rep__NL__as1`, `rep__GBT`) reproduce the saved transform and predictions exactly (max \|diff\| = 0.0).
- See `notes/execution/BACKUP_AND_RESTORE.json`. Nothing was deleted.
