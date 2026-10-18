# Freeze check (baseline and statistical reviewer)

**Scope.** I checked the four frozen files of commit `930235e9e`, with `NOMINATION.json`, against `NOMINATION_RULE.md` and
`CLAIMS_DRAFT.md` (registered at `74376912c`). The frozen files are FINALIST_LOCK.json, ANALYSIS_LOCK.json,
SERVICE_BANK_MANIFEST.json and READER_PLAN.json. No label or person-level data was read; only file hashes were computed,
including those of the two private bank files.

**Integrity: all match.**
- The four files, `NOMINATION.json`, `EXPERIMENT_LEDGER.csv`, `ihs/claims.py` and `ihs/report.py` are committed and
  unchanged on disk.
- FINALIST_LOCK's hashes of the three other locks, of NOMINATION.json and of the ledger match the files.
- The two bank files match `bank_files_sha256`.
- ANALYSIS_LOCK's `claims_code_sha256` and `report_code_sha256` match the committed code.
- READER_PLAN's five reader-code hashes match the current files.
- NOMINATION.json's `reserve_rule_sha256` is the registered `aca4d8e4...0accd70`.

**Verdict: PASS.** I found no material problem, so the DEV_REPORT read need not be held. The three notes at the end are
non-material.

## 1. Nominee and its record: PASS

- **The choice follows the rule exactly.** Six configurations qualify: CMB_P2JR_O1SEXBAL, CMB_P3JR_O1SEXBAL,
  H3_STRONG_LJDIR_PB, P2_JR_B40, P3_JR_B60 and PC_JSCALE_B60.
  - Best two-seed score: 0.6086170 (CMB_P3JR_O1SEXBAL). The near-tie window is <= 0.6136170.
  - Inside the window: CMB_P3JR_O1SEXBAL (+0), P3_JR_B60 (+0.0006106), PC_JSCALE_B60 (+0.0015022) and H3 (+0.0032380).
  - Outside: P2_JR_B40 (+0.0058662) and CMB_P2JR_O1SEXBAL (+0.0058957).
  - Largest `min_income_ratio`: H3, 2.8619483, against 2.6767809 (P3 and CMB_P3, which share the same income output) and
    2.2122005. So H3 is chosen; the fits and entry tie-breaks are not reached.
  - This is the rule's "robust reserve before point AUC" applied inside the window: the nominee scores 0.0032 above the best.
- **H3 record:**
  - kind `fit`, so not control-designated; label NOMINATED_FINALIST;
  - (a) true, and (b) RESERVE_MET on both seeds with the occupation side INHERITED;
  - c1 true (cell minima +0.0134 / +0.0091 vs H_LJ, +0.0179 / +0.0211 vs NEURAL_RAW); c2 true (maximum recipient cell
    difference 0.0); c3 true (minima +0.0064768 / +0.0011599);
  - not an alias; inputs recorded as the six raw fields; `record_mismatch` empty on both seeds;
  - flags: READER_UNSTABLE (both the configuration's and an eligible reference's winners vary across cells);
    UTILITY_CONFIDENCE_UNRESOLVED (8 of 28) is reported.
- **The other configurations get their labels in the rule's order.** For example:
  - HC_TTREE_LJDIR_PB: NO_GAIN_OVER_B2TWIN;
  - HC_STRONG and PC_TWIN_B40: UTILITY_HEADROOM_ONLY;
  - OC_BFIT: RESERVE_NOT_MET (RESERVE_ONE_SEED_ONLY; no EDGE qualifier because c1 fails);
  - OC_TCAL: POINT_INELIGIBLE;
  - the 8 one-seed configurations: NOT_CONSIDERED_ONE_SEED_ONLY.
  That makes 13 x 2 + 8 = 34 seed-specific configurations. NOMINATION.json carries the frontier, the marked references and
  the best diagnostic package.

## 2. TW and CC: PASS

- **TW = H:HC_STRONG, consistent with the draft's matched task-only twin.**
  - Its definition: the STRONG learner's own output (beta = 0) beside the JOINT_TREE occupation output. H's card declared it
    before the first H fit (CARD.md line 79) and named it again as H3's matched twin when H3 was decided (lines 206-207).
  - Same base, inputs and occupation output; only the recovery-targeting LJDIR step is removed.
  - Both seeds were evaluated with the frozen evaluator. Its nomination record is copied into FINALIST_LOCK
    (`control_nomination_records`): point-eligible, RESERVE_MET (3.2955), UTILITY_HEADROOM_ONLY.
  - The "ineligible twin" qualifier does not arise.
- **CC = H:HC_TTREE_LJDIR_PB, consistent with the draft's matched component control** ("any control that isolates one
  component of S").
  - It applies the identical step to the old T_TREE base: same D_J direction, same B1 values, same precision-aware budget
    rule. The rule chose the same beta on each seed (0.25 / 0.20), and the occupation output is the same inherited one.
    S - CC therefore isolates one component: the STRONG income learner.
  - It was not chosen after seeing results. The learner-isolation control is defined in the initial card (line 81), and the
    PB variant was named as H3's component control when H3 was decided, before its fit (lines 199, 207).
  - It is point-eligible and RESERVE_MET (2.4003) on both seeds, so a C result would not be an advantage over an infeasible
    alternative.
- **Consequences of declaring CC.**
  - n_claims becomes 3, which raises c1 from 2.2422567 to 2.3949991 for P and M as well. That can only make P and M harder,
    so it carries no favourable-selection risk for them.
  - Expect C to be hard to establish: on MODEL_SELECT, S's pair reads only 0.0016 / 0.0033 below CC's (card line 232).

## 3. Bank: PASS

- 8 releases per seed: NEURAL_RAW, H_LJ, SUCCESSOR, TW, CC, RETIRED_B3CTL, B2TWIN and PUB_IBSI. The references come first
  (L1), and every service is its own alias group, so there are no aliases.
- SUCCESSOR's occupation wire equals H_LJ's on all three roles, as do TW's and CC's. This confirms
  `occupation_output_inherited` = true. SUCCESSOR's income wire differs from every other service's.

## 4. Endpoint roster and critical values: PASS

- **130 endpoints.** Family P has 52 = 2 x (12 utility + 2 subgroup + 6 eligible-reference + 3 TW + 3 CC): E 28, P 12, M 6,
  C 6.
- **Orientation:** BEN|X has a = X, b = SUCCESSOR, so the benefit is AUC_pair(X) - AUC_pair(S), margin 0. NW|X has
  a = SUCCESSOR, b = X, margin 0.010.
- **Family D has 78:**
  - UTILITY|H_LJ (24) and SUBGROUP|H_LJ (4), the R2 and O floors;
  - DESC|LOSS|TW (8) and DESC|LOSS|CC (8), the M and C qualifiers;
  - DESC|TWBEN|{NEURAL_RAW, H_LJ} (4), which is AUC_pair(REF) - AUC_pair(TW);
  - 30 descriptive differences against RETIRED_B3CTL, B2TWIN and PUB_IBSI.
- **Inference settings:** n_claims 3; c1 2.3949991; c_P 3.3040310 (K_P 52); marginal 1.9605641; c_D 3.4163672 (descriptive
  only); alpha 0.025 one-sided; bootstrap 2026101705 with B = 2048; control bands planted >= 0.95, shuffled in [0.45, 0.55].

## 5. Declared flags: PASS

TW true, CC true, `occupation_inherited` true (verified from the wires), `control_designated` false (H3 is kind `fit`).
Claim O's `INHERITED_CONSTRAINTS_BIND` is therefore expected (`CLAIMS_DRAFT.md`, claim O).

## 6. Reader plan: PASS

- `.venv` interpreter with numpy 2.4.2, scipy 1.17.1 and scikit-learn 1.8.0: the validation lock's versions.
- Source 8 in source order and new 8 in the validation's order.
- ORIGINAL and ENHANCED selection: source-first order, ignore-one alternatives, exact ties to the earlier candidate, AV near
  tie below 0.010.
- Controls: OUT_HGB, ET5, HGB31, MLP6432 and CATI1, with NULL_SEED + s and the bands above.
- Variation and auxiliary refits: SUCCESSOR, H_LJ and NEURAL_RAW; model seeds + 1000 j; resamples 2026101805 + 10 s + j.
- Independent verifier roster: 12 fixed pair readers, plus the ENHANCED and ORIGINAL pair winners of the three services, on
  both seeds.
- The code hashes match.

## Non-material notes

1. `release_records` in the manifest lists SUCCESSOR and TW but not CC. CC's outputs are pinned by the manifest wire hashes
   and the bank file hash, so the evaluation is fully pinned. Recording CC's `release_sha256` would help the verifier's
   deployment check.
2. PROTOCOL_DEVIATIONS.md item 3 (the S1 / S2 safeguards) sits after item 5 (line 35). State when it was appended, so the
   append-only order is not read as a deletion.
3. H3's development margin over B2TWIN is thin on seed 1: minimum cell +0.0011599, mean benefit vs H_LJ +0.0103. Report it
   beside the final results; claim D compares S with B2TWIN using marginal intervals.
