# Independent numerical verification: PCRL income-headroom successor development campaign

**Role.** Independent verifier with separate code (brief `roles/coordinator/VERIFIER_BRIEF.md`). Statistical machine-learning
research on public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles only. The service
predicts income class (2) and occupation group (5) from AGEP, WKHP, COW, SCHL, MAR and POBP; each recipient receives its own
float64 probability vector and argmax decision. SEX is the dataset's recorded demographic column; evaluation readers estimate it
from the released vectors (each output alone and the linked pair). This is defensive evaluation of releases designed to lower that
recovery. Everything below is development evidence on already-used records, conditional on the fitted releases and readers.

**State verified.** Registration `74376912c` (receipt `77dd6710f`), finalist freeze `930235e9e`, reporting lock `8e559efe8`
(receipt `9ad0fb25d`). Nominee `H/H3_STRONG_LJDIR_PB`, matched task-only twin `H/HC_STRONG`, declared component control
`H/HC_TTREE_LJDIR_PB`; final bank of 8 releases per seed.

**Independence.** Own code in `roles/verifier/code/` for every verified quantity. Not imported: `ihs/evaluate.py`, `ihs/reserve.py`,
`ihs/nominate.py`, `ihs/report.py`, `ihs/claims.py`, `ihs/feasibility.py`, `amd/report.py`, `ffv.readers.pick`/`design_for`/`fit_new`.
Used as allowed: `ihs.common.load_role` / `frozen_bank` / `amd_release` (hash-checked admitted loader for the B2TWIN and
RETIRED_B3CTL source releases) / `dev_report_truth`, `ihs.devrun.load_release`, `task_headroom.inference.group_weights` (only to
check that the verifier's own weight generator reproduces the registered scheme), reader constructors
(`task_conditioned_representation.attacks.fit_reader` / `predict_reader`, `scm_bench.readers.kinds`, `ffv.readers.make_model`,
`ffv.features`).

## Summary

| # | Check | Status | Key numbers |
|---|---|---|---|
| 1 | Source admission and role integrity | **PASS** | 124 / 124 admitted copies re-hashed (0 mismatches); 34 / 34 run releases vs RELEASE.json; both final banks, 6 reader files and 4 locks match; gates confirmed from code and 4 live refusals (0 log lines added); MODEL_SELECT log 40 lines, 0 without purpose |
| 2 | Training and calibration exposure | **PASS** (notes only) | no MODEL_SELECT, ATTACK or DEV_REPORT label used to fit or calibrate the finalist, twin, component control or the best H / P / O / combination configurations; MODEL_SELECT label reads only in score, reference and verifier units (none in the 32 fit or 4 combine units); first ATTACK SEX read 8 s after the FINALIST_LOCK commit; SEX only as a MODEL_FIT training label; runtime inputs the six fields |
| 3 | Task and subgroup metrics and SEs | **PASS** | 21 configurations (34 configuration-seeds) + 4 references x 2 seeds = 42 evaluations; 12 clauses + 2 guards each; max point difference 0.0, max SE difference 0.0 (bitwise); point-pass flags and wire hashes all match |
| 4 | Reserve calculations | **PASS** | own rule code from the rule text: 34 / 34 per-seed statuses, failed lists and min ratios match (ratio difference 0); 21 / 21 two-seed statuses match NOMINATION.json; references reproduce RESERVE_RULE.json (H_LJ 5.5553 / 4.1626, NEURAL_RAW 2.6545 / 1.5575 one seed only, B2TWIN 2.7978 / 3.1148, RETIRED_B3CTL 0.4024 / 0.3489 not met) |
| 5 | Development proxies | **PASS** | 3 configurations + 2 references, all 4 cells x 2 release seeds: 840 / 840 per-reader cell values and 840 per-direction AUCs identical (max difference 0.0); benefits, recipient differences, c1 / c2 identical |
| 6 | Candidate and comparator selection | **PASS** | 21 / 21 labels, flags, scores, min ratios, model-fit counts, seed-0 ledger entries and per-cell pair winners match; 6 qualifying; near-tie set {CMB_P3JR_O1SEXBAL 0.60862, P3_JR_B60 0.60923, PC_JSCALE_B60 0.61012, H3_STRONG_LJDIR_PB 0.61185}; choice H/H3_STRONG_LJDIR_PB (min income ratio 2.862 vs 2.677 / 2.677 / 2.212) |
| 7 | Actual transmitted outputs | **PASS** (1 LOW engineering note) | redeployed from the six raw DEV_REPORT fields through `ihs.deploy.release`: 12 / 12 vectors bitwise (finalist, twin, component control x 2 seeds x 2 tasks) against RELEASE.npz and the final bank; own recompute of the income formula bitwise for all 6 releases; occupation bitwise H_LJ's JOINT_TREE output; refusals for SEX / PINCP / OCCP, extra, reordered and missing fields and hidden outputs; `ihs/deploy.py` edited after the reporting lock (V7-D1, no output change) |
| 8a | Final reader refits | **PASS** | 48 refits per seed (all 16 pair recipes for SUCCESSOR, NEURAL_RAW, H_LJ, covering the 12-reader roster and all 6 ENHANCED / ORIGINAL pair winners): 96 / 96 AV arrays and 96 / 96 DEV_REPORT score arrays bitwise; AV AUC difference 0.0 |
| 8b | Winners | **PASS** | 96 / 96 winners (8 services x 2 seeds x 2 banks x 3 views) re-derived: candidate, AUC, near-tie flags, gap to original; 608 / 608 candidate AUCs recomputed from AV_SCORES (max difference 0.0) |
| 8c | DEV_REPORT replay (one logged event) | **PASS** | 130 / 130 endpoints x 2 scorecards: max point difference 0.0, max SE difference 2.2e-19, max bound difference 1.7e-18, 0 clause-pass mismatches; 216 draw vectors within 5.2e-18 of the stored draws; every claim label E, R2, R3, P, N, M, C, T, O, D and the composite identical in both scorecards; READER_SENSITIVE key identical |
| 9 | Planning feasibility | **PASS (numbers); 2 documentation discrepancies** | 4 binding clauses exact (0.3976 / 0.5497 / 0.8624 / 0.8685 SE; at 17,510 households 0.837 / 1.157 / 1.815 / 1.827); analytic lower bound 113,758 households for the weakest condition below the stored joint D1 128,599; own Gaussian MC and bootstrap rescaling reproduce the stored 80 %-power counts to within one grid step |
| 10 | Labels and limitations | **PARTIAL** | every checked number matches (28-cell DEV table, claim table, MODEL_SELECT figures, 34 configurations, 14 lineage reads; two trivial roundings); required caveats present in RESEARCH_DECISION.md; wording issues: tie-break explanation wrong, "established income headroom" without its Brier / NLL qualifier and as an unregistered status label, declared-readers qualifier missing in two summaries, M / C wording beyond the registered sentence with mandatory M qualifiers missing |

## Check 1: source admission and role integrity (PASS)

Code `ver_admission.py`, unit `VER_ADMISSION_V1`.
- All **124** admitted copies in `SOURCE_ADMISSION.json` re-hashed with own SHA-256 code (bytes and digest): 0 mismatches;
  recorded expected-source hashes equal the recorded copy hashes for every copy (stage_b 40, amd_deploy 34, exports 18,
  amd_runs 12, stage_a 12, banks 2 + 2 + 2, packages 2). Recorded problems: none.
- All **34** run releases (`RELEASE.npz` vs `RELEASE.json release_sha256`): 0 mismatches. Final service banks S0 / S1 match
  `SERVICE_BANK_MANIFEST.json`; READERS_S0 / S1 `AV_SCORES.npz`, `DEV_REPORT_SCORES.npz`, `READERS_SEED.json` and the four locks
  match `REPORTING_LOCK.json`. The manifest's SUCCESSOR and TW release records resolve to `H/H3_STRONG_LJDIR_PB/S{0,1}` and
  `H/HC_STRONG/S{0,1}`.
- Gates (from `ihs/common.py`): `load_role` refuses SEALED_ASSESSMENT and any non-admitted role or field; MODEL_FIT / CALIBRATION /
  MODEL_SELECT labels require the committed registration; MODEL_SELECT labels require a purpose and are logged; ATTACK_FIT /
  ATTACK_VALIDATION SEX require the committed FINALIST_LOCK and a purpose and are logged; `dev_report_truth` requires a registered
  event name, refuses a second use of the same event, requires the committed REPORTING_LOCK and checks the truth-file hashes
  before writing the event. `committed()` requires the file to be tracked, committed and clean. Live tests (no data read):
  SEALED_ASSESSMENT refused; MODEL_SELECT labels without purpose refused; DEV_REPORT `sex` field refused; CALIBRATION `sex`
  field refused; 0 log lines added.
- Logs: `MODEL_SELECT_USE.jsonl` 40 lines, every line with a purpose and a unit id; 2 lines precede the registration commit
  (units `IHS_REFERENCES_S0_V1` / `S1_V1`, historical references only; disclosed in `PROTOCOL_DEVIATIONS.md` item 2).
  `ATTACK_LABEL_READS.jsonl` 9 lines, all with a purpose, 0 before the FINALIST_LOCK commit (`930235e9e`, 03:03:35Z; first read
  03:03:43Z). DEV_REPORT: this campaign's log has exactly 2 events (coordinator 03:08:07Z unit `IHS_REPORT_V1`; verifier
  03:17:54Z unit `VER_DEV_REPLAY_V1`), both after the reporting-lock commit (03:07:29Z) and both recording the same
  REPORTING_LOCK hash `6f7ee4c1...`; the lineage-wide log has 15 lines: the 12 pre-campaign events recorded at admission, one
  CORRECTION line, and this campaign's 2 events (14 DEV_REPORT reads in the lineage).

## Check 2: training and calibration exposure (PASS; notes only)

From each release record (`RELEASE.json`, `STATE.json`), the family code (`ihs/fam_h.py`, `fam_p.py`, `fam_o.py`, `combine.py`),
the queue ledger and the access logs.

| Configuration | Fitting | Calibration | Selection | SEX use | Runtime inputs |
|---|---|---|---|---|---|
| H3_STRONG_LJDIR_PB (finalist) | MODEL_FIT features and income (Step-1 HGB, household 5-fold CV, refit on all); stored T_TREE, LOCAL_TREE, JOINT_TREE corrections | CALIBRATION income (stack, household cross-fitted beta budget, Platt map) | MODEL_SELECT labels only in score units | none in the module; only through the stored corrections (trained on MODEL_FIT) | the six fields |
| HC_STRONG (twin) | the same HGB | CALIBRATION income (Platt map, close to identity) | same | none | the six fields |
| HC_TTREE_LJDIR_PB (component control) | stored trees only | CALIBRATION income | same | through the stored corrections only | the six fields |
| P3_JR_B60 (best of P) | MODEL_FIT income loss; MODEL_FIT SEX for the online critics | none | same | MODEL_FIT, training time | the six fields |
| O1_SEXBAL (best of O) | MODEL_FIT occupation and SEX for the balancing bias (described as in-sample) | CALIBRATION occupation (temperature) | reads H_LJ's MODEL_SELECT outputs (no labels) for an identity check | MODEL_FIT, training time | the six fields |
| CMB_P3JR_O1SEXBAL | no fit (combines two releases) | none | same | through its sources | the six fields |

- `MODEL_SELECT_USE.jsonl`: every label read is in a score, reference or verifier unit; none in the 32 fit units or the 4
  combination units; all fit units started after the registration commit (02:35:11Z). ATTACK SEX: first read 03:03:43Z, 8 s
  after the FINALIST_LOCK commit `930235e9e`; later reads are verifier units, all with purposes. Release fits read DEV_REPORT
  features only, and every `predict` function takes only the features.
- No internal MODEL_FIT fold is described as unseen data (closest: `roles/investigator_h/CARD.md:42`, "honest quality:
  out-of-fold on MODEL_FIT", explicitly limited to the HGB grid).
- Disclosed adaptive use (not a violation): MODEL_SELECT summary values shaped the finalist's design (the B1 budget from
  `REFERENCE_MS_TABLE.csv` and the reserve thresholds, `fam_h.py:70-72`; H3 is a revision after H1's MODEL_SELECT result).

## Check 3: task and subgroup metrics and SEs (PASS)

Code `ver_ms_checks.py`, units `VER_MS_CHECKS_V1` / `V2` (V2 only corrected the verifier's own wire-hash definition to include
the int64 argmax bytes; utility numbers identical). Own task vectors (argmax accuracy, NLL with the 1e-12 floor, summed Brier),
own household-clustered paired SE (5,306 MODEL_SELECT households, cluster sums of centred differences, G/(G-1)) and own
linearized subgroup SE for the within-SEX guards.
- Coverage: every configuration with a release (21 configurations; 13 with both seeds, 8 with one; 34 configuration-seeds),
  including the nominee, twin, component control and the best of each family (H3_STRONG_LJDIR_PB, P3_JR_B60, O1_SEXBAL,
  CMB_P3JR_O1SEXBAL), plus references H_LJ, NEURAL_RAW, B2TWIN and RETIRED_B3CTL on both seeds.
- Every one of the 12 clauses and 2 guards in all 42 evaluations equals the stored `EVAL_MS.json` / `EVAL_MS_REFERENCES.json`
  value and SE exactly (max |difference| 0.0 for points and for SEs; tolerance 1e-12 / 1e-9). Point-pass flags and released
  wire hashes match.

## Check 4: reserve calculations (PASS)

Own implementation of `UTILITY_RESERVE_RULE.md` section 1 with the floors of `RESERVE_RULE.json`: 6 income conditions per seed,
margin >= 2 x max(own SE, H_LJ SE), inclusive; occupation side inherited if all 8 occupation conditions equal H_LJ's to 1e-12,
otherwise each must meet the reserve or keep the inherited margin; two-seed status by the rule's table; `min_income_ratio` over
the 12 income conditions. Inputs: the verifier's own recomputed values and SEs (check 3).

| Configuration | Seed 0 | Seed 1 | Two-seed (own = stored) | min income ratio |
|---|---|---|---|---|
| H3_STRONG_LJDIR_PB (nominee) | MET 3.5111 | MET 2.8619 | RESERVE_MET | 2.8619 |
| HC_STRONG (twin) | MET 4.5777 | MET 3.2955 | RESERVE_MET | 3.2955 |
| HC_TTREE_LJDIR_PB (component control) | MET 3.9024 | MET 2.4003 | RESERVE_MET | 2.4003 |
| P3_JR_B60 | MET 3.7246 | MET 2.6768 | RESERVE_MET | 2.6768 |
| PC_JSCALE_B60 | MET 3.4025 | MET 2.2122 | RESERVE_MET | 2.2122 |
| O1_SEXBAL | MET 5.5553 | MET 4.1626 | RESERVE_MET | 4.1626 |
| CMB_P3JR_O1SEXBAL | MET 3.7246 | MET 2.6768 | RESERVE_MET | 2.6768 |
| OC_BFIT | MET | NOT MET (guard sex0 worsened) | RESERVE_ONE_SEED_ONLY | 4.1626 |
| OC_TCAL | NOT MET (guard sex0) | NOT MET (guard sex0) | RESERVE_NOT_MET | 4.1626 |

All 34 per-seed screens (status, failed conditions, min ratio) and all 21 two-seed statuses equal the stored ones; the four
references reproduce `RESERVE_RULE.json` `reference_screen` exactly.

## Check 5: development proxies (PASS)

Code `ver_proxy_cells.py`, `ver_proxy_summary.py`; units `VER_PROXY_CELLS_V1`, `VER_PROXY_CELLS_S1_V1` (Homebrew python3,
the registered development interpreter). Own halves (A: the source permutation, seed from SHA-256('SCM1proxysplit'); B: SHA-256
order of `IHS1_dev_proxy_rep_B|household`; 2,653 / 2,653 households each), reader seeds MODEL_SEED_BASE + 100 + k (only LOG_MLP and
ET25 depend on k), both directions averaged, own rank AUC, own LOOKUP, own view maxima (pair maximum includes single views).
- Nominee, twin, component control, NEURAL_RAW and H_LJ; all 4 cells (A0, A1, B0, B1) on both release seeds; 7 readers x 3 views:
  **840 / 840 cell values identical** to EVAL_MS (max difference 0.0) and 840 / 840 per-direction AUCs identical to the
  evaluator's cache. Per-cell pair benefits and recipient differences against both references equal EVAL_MS `comparisons` and
  NOMINATION.json `reference_cells` exactly; two-seed mean pair proxies 0.6118549753742035 (nominee), 0.6368789517976022 (twin),
  0.6142582365018991 (component control).
- This also confirms `PROTOCOL_DEVIATIONS.md` item 3 (the S1 / S2 evaluator safeguards changed no reference cell).

## Check 6: candidate and comparator selection (PASS)

Own implementation of `NOMINATION_RULE.md` from the stored EVAL_MS records: admissibility and alias check (wire hashes), (a)
point eligibility, (b) two-seed reserve (own rule code), c1 every-cell pair superiority over NEURAL_RAW and H_LJ, c2 every-cell
recipient screen (<= 0.010), c3 every-cell gain over B2TWIN, the label order, the flags (PAIR_NEAR_TIE, READER_UNSTABLE,
RECIPIENT_EDGE with the maximum cell difference), the score (mean of 8 `max_incl_single` cells), the near-tie set (best + 0.005),
and the tie-breaks (largest min income ratio, fewest model fits, earliest seed-0 ledger entry).
- 21 / 21 labels equal `NOMINATION.json`: 8 NOT_CONSIDERED_ONE_SEED_ONLY; POINT_INELIGIBLE OC_TCAL; RESERVE_NOT_MET OC_BFIT;
  UTILITY_HEADROOM_ONLY HC_STRONG, PC_TWIN_B40, O1_SEXBAL; NO_GAIN_OVER_B2TWIN HC_TTREE_LJDIR_PB, PC_JSCALE_B40; QUALIFYING
  CMB_P2JR_O1SEXBAL, CMB_P3JR_O1SEXBAL, P2_JR_B40, P3_JR_B60, PC_JSCALE_B60, H3_STRONG_LJDIR_PB. Flags, scores (difference 0),
  min ratios, model-fit counts, seed-0 ledger entries and per-cell pair winners match for all 13 two-seed configurations.
- Best score 0.6086170 (CMB_P3JR_O1SEXBAL); near-tie set of four; the largest min income ratio is H3_STRONG_LJDIR_PB's 2.862
  (others 2.677, 2.677, 2.212), so the rule nominates **H/H3_STRONG_LJDIR_PB**, as recorded. CMB_P2JR_O1SEXBAL (+0.00590) and
  P2_JR_B40 (+0.00587) are outside the window.

## Check 7: actual transmitted outputs (PASS; V7-D1 LOW, V7-D2 INFO)

Code `ver_deploy_redeploy.py`, `ver_deploy_mechanism.py`; units `VER_REDEPLOY_V1`, `VER_DEPLOY_MECHANISM_V1`.
- Own driver calls the package wrapper `ihs.deploy.release` on the packages in `runs/DEPLOY/<config>_S{s}`, each input row holding
  exactly the six DEV_REPORT fields (label-free; the raw-input export holds only the six fields plus `row_id` and `unit`);
  re-encoding the six fields reproduces the admitted 145-column DEV features bitwise for every package.
- H3_STRONG_LJDIR_PB, HC_STRONG and HC_TTREE_LJDIR_PB, both seeds, both tasks: the float64 vectors (5,943 x 2 and 5,943 x 5)
  are equal by `np.array_equal` and by own SHA-256 to the stored `RELEASE.npz` and to the final-bank SUCCESSOR / TW / CC entries;
  decisions equal the argmax. Finalist wire-hash prefixes: s0 income `646fcc24c5d6a41c`, occupation `5c5b034bb424c7ac`; s1
  income `994690a00bae4572`, occupation `83198896afb77878`. 13-row and 1-row requests give the same bits.
- Own recompute without `fam_h`: d = a (s_B + D_L + beta D_J) + b from `STATE.json` parameters with own re-coding and the stored
  tree corrections: income bitwise for all 6 configuration-seeds (own sigmoid within 1.1e-16); beta 0.25 (s0) / 0.20 (s1) for the
  finalist and the component control; the HGB model file hash matches `STEP1.json` and the locked `STATE.json`.
- The occupation output is bitwise H_LJ's JOINT_TREE output (frozen bank and service bank) for all three on both seeds.
- Refusals: SEX, PINCP, OCCP (PermissionError); an extra field, reordered fields, a missing POBP (ValueError); a hidden-output
  request (PermissionError), for all six packages. Hashes: banks equal the manifest; FINALIST_LOCK package files 10 / 10 equal on
  disk; 11 dependencies per package equal on re-hash.
- V7-D1: `ihs/deploy.py` was changed in commit `f615cbf53` (03:15:07Z), after the reporting lock (03:07:29Z) and the DEV read
  (03:08:07Z): 3 lines add family H's Step-1 HGB files to the package dependency hashes. The file is in no lock (ANALYSIS_LOCK
  hashes only `ihs/claims.py` and `ihs/report.py`, both unchanged) and no output changed, but the change is not recorded in
  `PROTOCOL_DEVIATIONS.md`.

## Check 8: finalist branch

### 8a. Final reader refits (PASS)

Code `ver_readers_refit.py`; units `VER_READERS_S0_V1`, `VER_READERS_S1_V1` (`.venv` Python 3.13.7, numpy 2.4.2, sklearn 1.8.0,
scipy 1.17.1 asserted). Own driver, own AUC, own orientation sign rule, own rich features (equal to `ffv.features.rich_view`
bitwise), own LOOKUP; constructors as allowed (`ffv.features.CatiDesign` for the CATI design; `design_for` not used). Fit on
ATTACK_FIT SEX, oriented on ATTACK_VALIDATION, seed MODEL_SEED_BASE + s.
- 48 refits per seed: all 16 pair recipes for SUCCESSOR, NEURAL_RAW and H_LJ (the 12-reader READER_PLAN roster and all six
  ENHANCED / ORIGINAL pair winners; no winner was an ignore-one alternative). **96 / 96 AV arrays and 96 / 96 DEV_REPORT score
  arrays bitwise equal**, AV AUC differences 0.0, no orientation mismatch, CATI design 73 columns as stored.
- Roster AV AUCs (stored = refit; s0 / s1): SUCCESSOR ET5 0.5845 / 0.5907, HGB31 0.5891 / 0.6023, MLP6432 0.5924 / 0.5957,
  CATI1 0.5996 / 0.6032; NEURAL_RAW ET5 0.6009 / 0.5983, HGB31 0.6101 / 0.6142, MLP6432 0.6181 / 0.6230, CATI1 0.6238 / 0.6283;
  H_LJ ET5 0.5941 / 0.6001, HGB31 0.6010 / 0.6159, MLP6432 0.6021 / 0.6160, CATI1 0.6132 / 0.6145.
- The stored non-convergence flag on `PAIR__NEURAL_RAW__MLP6432` (seed 1; 300 of 300 iterations) reproduces; it is not a winner.

### 8b. Winners (PASS)

Code `ver_readers_winners.py`, unit `VER_READERS_WINNERS_V1`. 96 / 96 winners re-derived with own selection code (source-first
order, maximum oriented AV AUC, exact ties to the earlier candidate, near tie < 0.010): candidate, reader key, AUC, near-tie flag,
gap to runner-up, `same_as_original`, `gap_to_original_winner`, `AV_NEAR_TIE_TO_ORIGINAL`. All 608 stored candidate AUCs
recomputed from `AV_SCORES.npz` against ATTACK_VALIDATION SEX (max difference 0.0); AV controls identical and inside the bands.
Pair winners (ORIGINAL -> ENHANCED): s0 SUCCESSOR OUT_MLP 0.60286, NEURAL_RAW LOG_MLP 0.62877, H_LJ OUT_MLP 0.62028 (unchanged);
s1 SUCCESSOR OUT_HGB 0.60722, H_LJ OUT_MLP 0.62471 (unchanged), NEURAL_RAW LOG_MLP 0.62812 -> CATI1 0.62826 (near tie to the
original). Bank changes behind the READER_SENSITIVE key: s1 income, SUCCESSOR LOG_MLP -> MLP6432 (+0.0003) and H_LJ LOG_MLP ->
HGB15 (+0.0006), both near ties.

### 8c. DEV_REPORT replay (PASS; one logged event)

Code `ver_dev_replay.py`. Dry run on invented labels (unit `VER_DEV_DRYRUN_INVENTED_V1`, no access event), then the single real
replay: **unit `VER_DEV_REPLAY_V1`, event `DEV_REPORT_TRUTH_REREAD_BY_IHS_INDEPENDENT_VERIFIER` at 2026-10-18T03:17:54Z** through
`ihs.common.dev_report_truth`, exactly once (now refused by the one-use gate). All label-free inputs were loaded and hash-checked
before the event (locks vs REPORTING_LOCK, banks vs manifest, reader files vs REPORTING_LOCK, row orders). Own household weights
(seed 2026101705, B = 2048; identical to the registered generator), own rank AUC, own weighted bootstrap AUC (tie-group sums per
replicate), own task vectors, own endpoint bounds (c1 = 2.3949991, c_P = 3.3040310, c_D = 3.4163672, t_0.975 = 1.9605641,
df 3954), own claim logic from `CLAIMS_DRAFT.md`.
- **Coordinator evaluation completed.** `report/DEV_REPORT_RESULTS.json`, `report/PRIMARY_ENDPOINTS.csv` and the private
  `ENDPOINT_DRAWS.npz` / `HOUSEHOLD_CONTRIBUTIONS.npz` were all written by `run()` (03:08:33-34Z) before it returned; the
  queue's FAILED status comes from the `__main__` print (`KeyError: 'reader_sensitive_keys'`) after the return (disclosed,
  `PROTOCOL_DEVIATIONS.md` item 6). Exactly one coordinator event exists in this campaign's log and in the lineage log.
- **Endpoints.** 130 / 130 endpoints in each scorecard: max |point difference| 0.0, max |SE difference| 2.2e-19, max |bound
  difference| 1.7e-18; 0 clause-pass mismatches; 216 / 216 stored draw vectors reproduced (max difference 5.2e-18; 88 bitwise,
  the rest differ only in summation order); per-service recovery AUCs and SEs (8 services x 2 seeds x 2 banks x 3 views) and all
  per-service utility points identical; `PRIMARY_ENDPOINTS.csv` (260 rows) equals the JSON.
- **Controls (CV).** 30 controls (5 readers x 3 views x 2 seeds): all inside the bands on ATTACK_VALIDATION and on DEV_REPORT.

| Claim | ENHANCED (decides) | ORIGINAL | Verified basis (ENHANCED unless stated) |
|---|---|---|---|
| E | DEV_POINT_ELIGIBLE | same | 24 utility, 4 guard, 8 recipient points pass; confidence unresolved 17 of 28 |
| R2 | RESERVE_ERODED | same | 10 / 12 income margins >= 1.7267108 x max(SE_S, SE_H_LJ); seed-1 accuracy loss 0.00707 (U0 = Ucal), margin 0.00293 = 0.830 SE, required 0.00610 |
| R3 | INCOME_CONFIDENCE_UNRESOLVED (4 of 12) | same | the 4 accuracy clauses |
| P | NOT_SUPERIOR_TO_ALL_ELIGIBLE [READER_SENSITIVE] | NOT_SUPERIOR_TO_ALL_ELIGIBLE | BEN_NEURAL_RAW s0 +0.01278 (L1 +0.00037), s1 +0.01233 (L1 -0.00148; ORIGINAL -0.00160); BEN_H_LJ +0.01414 (L1 +0.00916), +0.01659 (L1 +0.00565); NW_H_LJ income s1 +0.00606, U1 0.02169 (ORIGINAL -0.00700, U1 0.00134) |
| N | NON_WORSENING_UNRESOLVED | NON_WORSENING_ESTABLISHED | max point 0.00606 |
| M | TASK_ONLY_ATTRIBUTION_ONLY | same | BEN_TW +0.02848 (L1 +0.02199), +0.03158 (L1 +0.02091); all NW_TW pass; step 2 = P's failure |
| C | NOT_ESTABLISHED | same | BEN_CC +0.00203 (L1 -0.00258), -0.00260 (L1 -0.01344) |
| T | TARGET_NOT_ESTABLISHED | same | the four reference benefits 0.0123-0.0166, all below 0.020 |
| O | INHERITED_CONSTRAINTS_BIND | same | occupation output bitwise H_LJ's on both seeds; 6 of 16 below the R2 rule: sex0 guard s0 / s1, occupation NLL vs U0 s0 / s1, vs Ucal s0 / s1 |
| D | NO_DESCRIPTIVE_GAIN_OVER_B2TWIN | same | B2TWIN pair difference +0.00419 / +0.00151, marginal lower bounds -0.0046 / -0.0083 |
| Composite | NOT_COMPETITIVE | NOT_COMPETITIVE | E holds; P fails; R2 eroded |

READER_SENSITIVE: exactly one conjunction key differs between the banks, `NW_H_LJ|income|s1` (ENHANCED fails, ORIGINAL passes),
affecting P only, as stored. These are statements about the declared readers and auxiliary budget only.

## Check 9: planning feasibility (PASS on the numbers; documentation discrepancies V9-D1, V9-D2)

Code `ver_feas_check.py`, unit `VER_FEAS_CHECK_V1`; inputs `CONFIRMATION_FEASIBILITY_RESULTS.json` (sha `2020f1e2...`),
`CONFIRMATION_FEASIBILITY.md` (sha `2ebb9e5a...`, re-read after its 03:16Z rewrite), published DEV points and SEs and the private
endpoint draws (derived endpoints only). `ihs.feasibility` not imported.
- Binding clauses (analytic normal, SE x sqrt(3955 / 17510)): occupation NLL vs Ucal s0 0.397619 SE (0.836636 at 17,510); men's
  (sex0) guard s1 0.549741 (1.156719); s0 0.862423 (1.814639); income accuracy s1 0.868509 (1.827444): all exact.
- Weakest ENHANCED condition NW_H_LJ|income|s1: closed-form 80 %-power household count 113,758, below the stored joint D1 of
  128,599 (as it must be); analytic marginal power at 128,599 is 0.8523 vs the stored Monte Carlo 0.85505.
- Joint result: own Gaussian Monte Carlo (Cholesky, 40,000 draws) reproduces every stored 80 %-power count except two ORIGINAL
  cells one grid step lower; own rescaling of the 2,048 bootstrap replicates reproduces every count except three ENHANCED D1
  cells one grid step lower (116,082 vs 128,599), about the resolution of 2,048 replicates. Example D1 power at 17,510: stored
  0.0887 (ENHANCED) / 0.6977 (ORIGINAL); rescaling 0.0781 / 0.7061; Gaussian 0.0881 / 0.7047. S4 (reader-refit spread) and S6
  (assurance) were not recomputed.

## Check 10: labels and limitations (PARTIAL)

Read: `RESEARCH_DECISION.md` (03:18Z version), `START_HERE.md`, `QUICKSTART.md`, `MANUSCRIPT_CLAIMS_HANDOFF.md`,
`HANDOFF_DECISION.json`, `MECHANISM_CARDS.md`, `PROTOCOL_DEVIATIONS.md`, `CONFIRMATION_FEASIBILITY.md`; spot checks of
`COMPARISON_TABLE.csv` and `FRONTIER.csv`. Documents may still change; this is the state at about 03:25Z.
- **Supported by the numbers.** All 28 cells of the RESEARCH_DECISION DEV table (pair and income AUC, income Brier vs Ucal, income
  accuracy loss for 7 releases x 2 seeds), the claim table, the MODEL_SELECT figures (H3 0.6101 / 0.6136 at income Brier
  -0.0023 / -0.0033; P3 0.6072 / 0.6113; O1 men's guard 0.00166 / 0.00230 = 1.83 / 1.62 SE; OC_TCAL 0.00513 / 0.00509), the
  configuration counts (H 9, P 14, O 7, combinations 4; 34 of 36), the 14 lineage DEV_REPORT reads, the refit spread (seed-0
  pair AUC 0.594-0.604 under bootstrap-refitted readers), and the feasibility figures (D1 128,599, D2 291,759; retired 4.6 / 60
  million). RESEARCH_DECISION.md states that seeds are not replications, that conditional intervals do not cover the adaptive
  search, and makes no injectivity or information-equality inference.
- **Flagged** (details in the discrepancy list): V10-F1 tie-break explanation; V10-F2 "established income headroom"; V10-F3
  declared-readers qualifier; V10-F4 M / C wording and qualifiers; V10-F5 to F7 trivial; V10-F8 to F10 information.

## Discrepancies

No numerical discrepancy was found in any verified result, label or selection. Everything below is documentation, engineering
record or information.

| ID | Severity | Description |
|---|---|---|
| V9-D2 | MODERATE (documentation; no claim or decision change) | `CONFIRMATION_FEASIBILITY.md`, S4 reader-refit row and conclusion 1, quote about 733,000 households (ENHANCED). The JSON names ORIGINAL as governing for S4\|D1 and S4\|D2 at 1,355,573 households (limit 0.828). Under the document's own rule (the less favourable scorecard governs), the requirement is about 1.36 million. S4 itself was not recomputed by the verifier. |
| V9-D1 | LOW (documentation) | `CONFIRMATION_FEASIBILITY.md`, "true slack 0.002" row, gives D2 as about 157,800, which is ORIGINAL's 157,828. The JSON's governing ENHANCED value is 174,847, reproduced by both verifier methods. |
| V7-D1 | LOW (engineering record) | `ihs/deploy.py` changed after REPORTING_LOCK and the DEV read (commit `f615cbf53`, 03:15:07Z; +3 lines adding family H's Step-1 HGB files to the package dependency hashes). No lock covers it and no output changed (check 7 bitwise). Not recorded in `PROTOCOL_DEVIATIONS.md`. |
| V10-F1 | LOW (documentation) | `RESEARCH_DECISION.md:61-62` and `MECHANISM_CARDS.md:66`: "the occupation-repaired combination lost the tie-break (equal income ratio; more model fits)". CMB_P3JR_O1SEXBAL lost to H3 at the min income ratio step (2.677 < 2.862). Its equal ratio and larger fit count only rank it behind P3_JR_B60. CMB_P2JR_O1SEXBAL (0.61451) was outside the near-tie window (cutoff 0.61362). No label changes. |
| V10-F2 | LOW (wording) | "Established income headroom" (`RESEARCH_DECISION.md:22, 119, 128`; `START_HERE.md:27`; `HANDOFF_DECISION.json` status `INCOME_HEADROOM_ESTABLISHED_NOT_COMPETITIVE`, which is not a registered label). The registered labels are R2 RESERVE_ERODED and R3 INCOME_CONFIDENCE_UNRESOLVED (4 of 12). The phrase is supported only for the 8 income Brier / NLL clauses (family bounds at c_P below the caps; margins 3.77-4.98 and 5.75-7.05 SE in max(SE_S, SE_H_LJ) units), as development evidence on already-used records. It needs "for Brier and NLL" everywhere, as `START_HERE.md:19` and `RESEARCH_DECISION.md:98` already have. |
| V10-F3 | LOW (wording) | `RESEARCH_DECISION.md:16-19` and `START_HERE.md:21-22` state the reader results without "evidence against the declared readers and auxiliary budget only" (`MANUSCRIPT_CLAIMS_HANDOFF.md:14-15` has it). |
| V10-F4 | LOW (wording) | Claims M and C go beyond the registered sentence: `RESEARCH_DECISION.md:110-111` ("do the recovery work") and `MANUSCRIPT_CLAIMS_HANDOFF.md:18` ("The privacy work is done by..."). The mandatory M qualifiers are not printed beside the label: the S - TW and S - CC NLL / Brier differences, TW's own eligibility and reserve, and TW's pair benefit against the references, which is negative on DEV (TW pair 0.6557 / 0.6578 vs H_LJ 0.6414 / 0.6428). |
| V10-F5 | TRIVIAL | `RESEARCH_DECISION.md:94` "0.0143 / 0.0094 above the retired candidate": the seed-0 value is 0.014248, which rounds to 0.0142. |
| V10-F6 | TRIVIAL | `RESEARCH_DECISION.md:107` "a little recovery reduction there (0.0016 / 0.0033)": the MODEL_SELECT pair-proxy differences HC_TTREE_LJDIR_PB - H3 are 0.001563 / 0.003243, which round to 0.0016 / 0.0032. |
| V10-F7 | TRIVIAL | `MANUSCRIPT_CLAIMS_HANDOFF.md:10-11` says "household-clustered SEs". DEV_REPORT SEs are household-bootstrap SEs, and the 3.8 lower end uses max(SE_S, SE_H_LJ). |
| V10-F8 | INFO | Files cited as existing were absent at check time: `BACKUP_VERIFICATION.json`, `RESOURCE_ACCOUNTING.json`, `ACCESS_RECEIPT.json`, `HANDOFF.json`, `PUBLICATION_RECORD.json`, `INDEPENDENT_VERIFICATION.json` (`RESEARCH_DECISION.md:139-141`; `START_HERE.md:4, 44-46`). Closeout is still in progress. |
| V10-F9 | INFO | `HANDOFF_DECISION.json` (and `NOMINATION.json`) call CMB_P3JR_O1SEXBAL the "best diagnostic package". `NOMINATION_RULE.md` defines that term only for the case where no configuration qualifies. |
| V10-F10 | INFO | Family O's "information-preserving" and "bijection" wording (`MECHANISM_CARDS.md:49-51`, `MANUSCRIPT_CLAIMS_HANDOFF.md:22`) describes the exact map. It is consistent with equal distinct-vector counts against H_LJ on MODEL_SELECT (7,500 / 7,500; 7,498 / 7,498). No population information-equality inference follows. |
| V7-D2 | INFO | `FINALIST_LOCK.json` does not list the HGB model file. It is covered through the locked `STATE.json`, whose recorded hash matches on disk. |
| V8-I1 | INFO | Coordinator unit `IHS_REPORT_V1` is recorded FAILED by the queue because of a `KeyError` in the `__main__` print after `run()` returned. All outputs were complete, reproduced exactly by the replay, and backed by one coordinator event. Disclosed in `PROTOCOL_DEVIATIONS.md` item 6. |
| V1-I1 | INFO | Two MODEL_SELECT label reads precede the registration commit (`IHS_REFERENCES_S0_V1` / `S1_V1`, historical references only). Disclosed in `PROTOCOL_DEVIATIONS.md` item 2. Check 5 shows that those reference cells reproduce bitwise under the current evaluator. |

## Units and resources

| Unit | Status | Wall s | CPU s |
|---|---|---|---|
| VER_READERS_WINNERS_V1 | done | 1.0 | 0.8 |
| VER_REDEPLOY_V1 | done | 5.1 | 4.7 |
| VER_READERS_S0_V1 | done | 29.2 | 28.6 |
| VER_READERS_S1_V1 | done | 29.1 | 28.7 |
| VER_PROXY_CELLS_V1 | done | 125.0 | 124.2 |
| VER_DEV_DRYRUN_INVENTED_V1 | done | 11.1 | 10.8 |
| VER_FEAS_CHECK_V1 | done | 22.4 | 22.3 |
| VER_DEV_REPLAY_V1 | done (the one DEV_REPORT event) | 12.5 | 12.1 |
| VER_PROXY_CELLS_S1_V1 | done | 125.2 | 124.4 |
| VER_MS_CHECKS_V1 | done (verifier wire-hash definition corrected in V2) | 1.6 | 1.2 |
| VER_MS_CHECKS_V2 | done | 1.5 | 1.2 |
| VER_ADMISSION_V1 | done | 0.6 | 0.5 |
| VER_DEPLOY_MECHANISM_V1 | done | 2.2 | 1.9 |
| **Total (13 units)** | 13 done, 0 failed | 366.6 | 361.4 |

No unit failed. Verifier label reads, all logged with purposes:
- MODEL_SELECT: 4 reads (`VER_PROXY_CELLS_V1`, `VER_PROXY_CELLS_S1_V1`, `VER_MS_CHECKS_V1`, `VER_MS_CHECKS_V2`);
- ATTACK_FIT / ATTACK_VALIDATION SEX: 5 reads (`VER_READERS_*`);
- DEV_REPORT truth: once (`VER_DEV_REPLAY_V1`).

The sealed assessment was never opened. Small out-of-queue scripts (under 1 s each) read only aggregate JSON and CSV files. Agent
effort is charged with `ihs.jobqueue charge --label verifier`.

Private verifier outputs, aggregate only, are in `.private/pcrl_income_headroom_successor_v1/verifier/`: `ADMISSION_CHECK.json`,
`MS_CHECKS.json`, `PROXY_CELLS_S{0,1}.json`, `PROXY_SUMMARY.json`, `FEASIBILITY_CHECK.json`, `WINNERS_CHECK.json`,
`READERS_REFIT_S{0,1}.json`, `REDEPLOY_CHECK.json`, `MECHANISM_RECOMPUTE.json`, and `DEV_REPLAY/VERIFIER_DEV_REPLAY.json` with
`VERIFIER_ENDPOINT_DRAWS.npz`.
