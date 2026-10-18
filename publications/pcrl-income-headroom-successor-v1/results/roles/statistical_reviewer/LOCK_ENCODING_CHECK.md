# Lock-encoding check (baseline and statistical reviewer)

**What was checked.** Does the code encode the documents registered at `74376912c`? Those are `NOMINATION_RULE.md`,
`CLAIMS_DRAFT.md`, `UTILITY_RESERVE_RULE.md` / `RESERVE_RULE.json`, and the validation's `READER_BANK_LOCK.json`.

**Files checked, as on disk at 22:51 EDT:**

| File | Modified (EDT) |
|---|---|
| `ihs/nominate.py` | 22:37:21 |
| `ihs/claims.py` | 22:38:28 |
| `ihs/locks.py` | 22:39:54 |
| `ihs/combine.py` | 22:44:43 |
| `ihs/report.py` | 22:49:20 (the version that calls `ihs.claims`) |
| `ihs/final_readers.py` | 22:16:14 |

No data or label was read. I recomputed the critical values with scipy (t, df 3954): c1 = 2.2422567 (n = 2) and 2.3949991
(n = 3); c_P = 3.2694077 (K_P = 46) and 3.3040310 (K_P = 52); marginal 1.9605641. All match.

**Verdict.** The decision logic encodes the documents:
- nomination: labels and their order, the c1 / c2 / c3 cell tests, near-tie set, tie-breaks, alias test and reserve;
- claims: E, R2, R3, N, P, M, C, O and COMPOSITE;
- locks: K_P, n_claims and the critical values;
- final readers: ORIGINAL and ENHANCED winner selection.

I request 17 changes, listed below. Most are integrity checks or quantities the documents require to be reported but the
code omits. Four affect correctness:
- **C3**: the registered control condition on ATTACK_VALIDATION is not encoded.
- **C4 and L1**: an aliased bank slot would crash the report or rename a reference.
- **C1**: one label edge case in T.

None of the 17 changes alters a current dry-run label.

**Timing.** `ANALYSIS_LOCK` records the SHA-256 of `ihs/claims.py` and `ihs/report.py` at `locks freeze`.
- Every claims.py and report.py change must therefore land before the freeze.
- nominate.py changes must land before `NOMINATION.json` is final.

## 1. `ihs/nominate.py` against NOMINATION_RULE.md

**PASS**
- Label order: one seed only (55-57); then alias, (a), (b) with the EDGE qualifier when c1 holds, UTILITY_HEADROOM_ONLY,
  RECIPIENT_SCREEN_FAILED, NO_GAIN_OVER_B2TWIN, QUALIFYING (117-130). This is exactly the rule's order.
- (a): 24 clauses and 4 guards at the point, both seeds (72-73). (b): `two_seed_status` == RESERVE_MET (75, 121).
- c1: strict > 0 in every cell, both references, both seeds (78-82, 16 differences). c2: every income and occupation cell
  <= 0.010 (83-84, 16). c3: B2TWIN minus configuration > 0 in every cell, both seeds (94-98, 8).
- (d): 3.1237346 x own SE over the 24 clauses and 4 guards, reported only (23, 100-102).
- Score: the mean of the 8 `max_incl_single` cells (103). Near-tie set: score <= best + 0.005, inclusive (24, 140-141).
- Tie-breaks: largest `min_income_ratio`, then fewest `n_model_fits` summed over both seeds, then earliest seed-0 entry (142).
- Controls compete (41 keeps `kind == 'control'`); `CONTROL_DESIGNATED_FINALIST` is set (146-147). The alias test compares
  both wires on both seeds against every reference (67-70), so RETIRED_B3CTL is excluded. The grid version is enforced
  (63-64). The no-finalist outcome is handled (152).
- Flags: PAIR_NEAR_TIE (86), READER_UNSTABLE (range part, 87), RECIPIENT_EDGE (85, 115-116).

**CHANGE-REQUESTED**
- **N1, "first complete evaluation" (40-49, 60).**
  - The first ledger row per (configuration, seed) is taken even if it is a FAILED / STOPPED row.
  - The `EVAL_MS.json` read at line 60 is whatever is on disk now, and a re-evaluation overwrites it.
  - Change: count only complete evaluation rows (decision not FAILED / STOPPED, with `ms_point_pass_12` filled). Recompute
    `ledger.summarize(EVAL_MS, comparisons, reserve)` and require equality with the counted row; on any difference, stop.
- **N2, admissibility "computed from the six permitted raw fields" (52-70).** Not checked. Read `inputs_used` from
  RELEASE.json (fam_p 452, fam_o 377, fam_h 449, combine 38) and record it. Refuse a release that records anything else.
- **N3, reserve provenance (75).** Require `ms[s]['reserve']['rule_sha256']` == `R.load_rule()['_sha256']`
  (`aca4d8e4...0accd70`). Or recompute `R.screen(ms[s], s)` and compare `status` and `failed`.
- **N4, reference winners (87, 90-92).** The rule reports "the winning reader in each cell for the configuration and for r".
  Only the configuration's winners are computed. Add each eligible reference's per-cell winners, and say in READER_UNSTABLE
  which side changed (configuration or reference). The validation's failure came from the comparator side.
- **N5, twin and component-control cell differences (missing).** The rule's "Always reported" list includes "the same cell
  differences against the configuration's own matched task-only control and any matched component control". Add a declared
  configuration-to-twin map (for example P2/P3 to PC_TWIN_B40) and report `EV.benefits` per seed.
- **N6, no qualifying configuration (137-152).** Report the frontier (score against `min_income_ratio`, with NEURAL_RAW,
  H_LJ, B2TWIN and RETIRED_B3CTL marked) and the best diagnostic package: the lowest-score configuration with (a) and (b).
- **N7 (77).** `edge` starts at 0.0, so `max_recipient_cell_difference` is floored at 0. Start at -inf. The flag logic is
  unaffected.
- **Context, no change.** On your rounded figures, P3_JR_B60 (0.60925) leads P2_JR_B40 (0.6145) by 0.00525, just outside the
  0.005 near-tie set. The decision rests on unrounded scores, which lines 140-141 correctly use. Print the exact difference
  in NOMINATION.json.

## 2. `ihs/claims.py` and `ihs/report.py` against CLAIMS_DRAFT.md

**PASS**
- **E** (29-41): UG, SG, RG, then the label. Failed keys come in the order UG, SG, RG. Utility confidence over 28 endpoints
  at c_P (family bounds), and recipient pass count.
- **R2** (14, 43-58):
  - factor 1.7267108 (= 2 x sqrt(3955 / 5306) to 7 decimals); se_used = max(SE_S, SE_H_LJ) on DEV;
  - margin = cap - point, inclusive test;
  - labels RESERVE_REPLICATED / RESERVE_ERODED / CAP_EXCEEDED_AT_POINT.
- **R3** (59-62): family bounds against the caps; three labels. **R4** (52): S's own SE x sqrt(3955 / 17510).
- **N** (66-74): ESTABLISHED / UNRESOLVED / VIOLATED_AT_POINT, as drafted.
- **P** (75-91): CV ∧ E_POINT ∧ 4 BEN ∧ 8 NW. Failure order: CONTROL_FAILURE, UTILITY gate, SUBGROUP gate, RECIPIENT point
  gate, NOT_SUPERIOR. The condition vector is kept.
- **M** (102-114): step 1 against the twin (CV, E_POINT, 2 BEN, 4 NW), then P. The labels and NOT_TESTED_SEQUENCE_STOPPED
  match, and the control-designated flag is carried.
- **C** (115-119): conjunction, NOT_ESTABLISHED, or NOT_CLAIMED when no CC is declared.
- **O** (121-138):
  - inherited occupation: BIND if any of the 16 conditions misses 1.7267108 x max(SE_S, SE_H_LJ);
  - changed occupation: RELIEVED / IMPROVED_BUT_BINDING / WORSENED, with "worse" meaning below the inherited margin.
- **COMPOSITE** (140-146): COMPETITIVE_SUCCESSOR = E ∧ R2 ∧ P, else SUPERIOR_WITHOUT_RESERVE when only P holds; carries O
  and R3.
- **report.py** now:
  - evaluates `CL.evaluate(eps, CV_ok, lock['declared'])` for each scorecard (175) and `CL.reader_sensitive` (176);
  - decides on ENHANCED (177-178), from the frozen roster (144);
  - checks the critical values against the lock (61-66).
- **How the scorecards differ, as encoded.**
  - Only the AUC endpoints (BEN, NW, DESC|BEN) differ. ENHANCED uses the enhanced winners and ORIGINAL the source-slate
    winners refitted here (90-99). E_POINT can differ through RG.
  - Utility and guard endpoints are reader-free and identical (101-118).
  - CV is shared: the 5-recipe ENHANCED control set is used for both scorecards. That is stricter than the validation's
    ORIGINAL_CV (OUT_HGB only), and it cannot create a READER_SENSITIVE artifact. Acceptable; say so in the report.

**CHANGE-REQUESTED**
- **C1, T edge case (claims.py 93-98).** The draft defines TARGET_POINT_ONLY as "all 4 points > 0.020, **but some bound <=
  0.020**", and TARGET_NOT_ESTABLISHED as otherwise.
  - The code gives TARGET_POINT_ONLY whenever all 4 points exceed 0.020. That includes P failing for other reasons while all
    4 bounds exceed 0.020.
  - Use `all(point > 0.020) and any(L1 <= 0.020)` for TARGET_POINT_ONLY; everything else is TARGET_NOT_ESTABLISHED.
- **C2, READER_SENSITIVE (claims.py 150-159; report.py 176-178).** The draft says "disagree on **any key** of a conjunction
  ... with the key and **side**".
  - (i) For M and C, compare the step-1 condition values (BEN / NW of TW and CC, CV, E_POINT), not only the labels. M's
    label can stay the same while a key flips.
  - (ii) For each differing key, add the side, using the validation's paired reader-class change rule. Base it on the
    ENHANCED - ORIGINAL AUC change of S's view and of X's view, and on whether each winner changed.
  - (iii) Append `[READER_SENSITIVE]` to each affected ENHANCED label in `res['labels']`.
- **C3, CV on ATTACK_VALIDATION (report.py 120-131, test at 127).**
  - Only the DEV_REPORT bands are tested, although the AV values are recorded (130). The draft and READER_BANK_LOCK require
    the bands "on ATTACK_VALIDATION and DEV_REPORT".
  - Add the AV test to `CV_ok`. Also test the AV controls in `locks.reporting()` (182-191) before REPORTING_LOCK, logging an
    out-of-band AV control as a deviation (READER_BANK_LOCK `controls.no_retuning`).
- **C4, aliased slots (report.py 144-159).** `build_bank` drops exact aliases, so a slot such as TW or CC that aliases
  another service is absent from `bank`. `auc_pt[(a, ...)]` or `util[(svc, ...)]` then raises KeyError. Resolve each
  endpoint's `a`, `b` and `service` through `man['per_seed'][f'S{s}']['aliases']` (loaded at 52, unused), and mark
  IDENTICAL_RELEASE when the resolved names are equal.
- **C5, lock integrity (report.py about 50-52).**
  - Assert `sha(ihs/claims.py) == lock['claims_code_sha256']` and `sha(ihs/report.py) == lock['report_code_sha256']`.
  - Compare `res['locks_sha256']` (71) with `rlock['locks_sha256']` (locks.py 188).
- **C6, mandatory qualifiers and claim D.**
  - (i) C's qualifier, the S - CC income and occupation NLL and Brier differences, has no endpoint. Add `DESC|LOSS|CC|...`
    when a CC is declared (locks.py 128-131 has it only for TW).
  - (ii) M's qualifiers lack TW's pair benefit against REF. Add `DESC|BEN_TW|<REF>|s` = AUC_pair(REF) - AUC_pair(TW). Also
    copy TW's MODEL_SELECT eligibility and reserve record (from NOMINATION.json) into FINALIST_LOCK and the M record.
  - (iii) The draft says "marginal 95% intervals only" for D, but family-D endpoints get only c_D bounds (report.py 167).
    Add point ± 1.9605641 SE. Then compute `GAIN_OVER_B2TWIN_DESCRIPTIVE`: the marginal lower bound of `DESC|BEN|B2TWIN|s`
    > 0 on both seeds.
  - These are family-D additions, so K_P, c1 and c_P are unchanged.
- **Optional.** Link into N's record the AV near-tie flags of both winners and whether each winner changed between banks.
  They already exist per service in report.py 97-98 and READERS_SEED `same_as_original`.

## 3. `ihs/locks.py` (endpoint roster, analysis lock, bank, READER_PLAN)

**PASS**
- **Roster** (98-132). Family-P endpoints per seed: 12 utility, 2 subgroup, 3 per eligible reference (BEN plus 2 NW, so 6),
  and 3 for TW: 23 in all, plus 3 with a CC. K_P = 46 without a CC and 52 with one (139).
- **n_claims** is 2 or 3 (138). c1 = crit(G, 0.025 / n_claims) and c_P = crit(G, 0.05 / (2 K_P)) (145-146), with
  crit = t_{G-1} (amd.report 27-29). Values above.
- Bootstrap 2026101705 / B 2048 (29). Bands planted >= 0.95, shuffled [0.45, 0.55] (148). `declared` is set from the bank
  (159, 164).
- **Bank** (39-78): SUCCESSOR, TW, CC (if declared), NEURAL_RAW, H_LJ, RETIRED_B3CTL (source FINALIST), B2TWIN and PUB_IBSI.
  Exact aliases are removed over 3 roles x 2 tasks (57-64), and the bank refuses more than 8 releases (65-66).
- **READER_PLAN** (81-95): `.venv` versions numpy 2.4.2 / scipy 1.17.1 / sklearn 1.8.0; full NEW8 parameters; ORIGINAL and
  ENHANCED selection text; control recipes and permutation seed; variation and auxiliary refit roster; code hashes.

**CHANGE-REQUESTED**
- **L1, alias naming (42-64).** The services are inserted with SUCCESSOR, TW and CC before NEURAL_RAW and H_LJ, and the
  first occurrence is kept. If TW or CC were an exact alias of H_LJ, the bank would store H_LJ under the name `TW`, and every
  `H_LJ` endpoint (REF legs, the R2 and O floors) would break. Insert NEURAL_RAW and H_LJ first, keep the reference names
  canonical, and let C4 resolve the aliases.
- **L2, independent winner-refit roster (81-95).** PROTOCOL §10 requires "a predefined independent winner-refit roster
  covering the successor and the strongest references". READER_PLAN defines only the variation and auxiliary refits. Add,
  per seed, the validation's `verifier_refit_roster` pattern:
  - PAIR__{SUCCESSOR, NEURAL_RAW, H_LJ}__{ET5, HGB31, MLP6432, CATI1};
  - the ENHANCED and ORIGINAL pair winners of those three, resolved by rule after selection;
  - bitwise replay against the stored DEV_REPORT scores.
- **L3.** The roster additions of C6 (family D only).

## 4. `ihs/final_readers.py` against the validation READER_BANK_LOCK

**PASS**
- Source 8 in source order (41; scm_bench SLATE plus LOOKUP), new 8 in the validation's NEW8 order (38).
- Single-view winners: ORIGINAL over the source 8; ENHANCED over the source 8 then the new 8 (181-185).
- Pair candidates in source-first order (188-194): source 8 pair, source 8 IGNORE_occupation, source 8 IGNORE_income, then
  the same three groups for the new 8. ORIGINAL uses the source group only.
- `pick`: maximum oriented AV AUC, exact ties to the earlier candidate, AV_NEAR_TIE below 0.010 (ffv/readers 106-113).
- Orientation on ATTACK_VALIDATION only (128, 135, 146). Fit on ATTACK_FIT, score DEV_REPORT outputs.
- Exact-wire deduplication over the 3 roles (154-169). CATI design fitted on ATTACK_FIT outputs (143).
- Controls (199-226): OUT_HGB, ET5, HGB31, MLP6432 and CATI1 on the SUCCESSOR wires, x 3 views. Shuffled-SEX permutation
  seed NULL_SEED + s = 2026101602 + s; planted bit on AV and on DEV for both bit values.
- Refit roster (228-250): the ENHANCED pair winners of SUCCESSOR, H_LJ and NEURAL_RAW. Seed-dependent winners get
  seeds + 1000 j, j = 1..4; each winner also gets 4 household-bootstrap ATTACK_FIT resamples, seeds 2026101805 + 10 s + j.
- Versions and NEW8 parameters are asserted against READER_PLAN (80-85). Role sizes 8,001 / 3,000 / 5,943 (100-101).

**CHANGE-REQUESTED**
- **F1, near tie with the original winner (194-196).** The validation adopted O4: "AV gap to runner-up **and to the original
  winner** < 0.010". Only `same_as_original` is recorded. Add `gap_to_original_winner` (enhanced minus original AV AUC) and
  flag it below 0.010.
- **F2, CATI fitted state (143-151).** `desc` from `design_for` is discarded. The validation recorded the CATI state
  (categories, cuts, knots), and its REPORTING_LOCK checked it. Store `desc` for every CATI fit, marginal and pair, in
  READERS_SEED.json so the verifier can check it.

## 5. Question: `n_model_fits` of a coordinator combination (combine.py 40)

**Confirmed for the nomination tie-break, with two qualifications.**
- **Why the sum fits.** The tie-break stands for "the simpler mechanism". A combination deploys both sources' fitted
  components, so the sum of its sources' fits is the right count. Frozen references contribute 0 (combine.py 27 and 40),
  and nested combinations carry their sources' counts.
- **Qualification 1: count each distinct upstream fit once.** If both sources reuse the same fitted component (a shared
  base model or calibrator), the sum double-counts it.
  - Nothing is shared in CMB_P2JR_O1SEXBAL: 10 + 47 = 57 per seed.
  - The outcome you report is what the rule gives. Equal income outputs give an equal `min_income_ratio`, so fewest fits
    decides: 20 for P2_JR_B40 against 114 for CMB_P2JR_O1SEXBAL.
- **Qualification 2: accounting.** The ledger's `n_model_fits` column then carries upstream fits, so campaign totals must not
  sum it across rows. CMB rows, and revisions that reuse a base (H2 may reuse H1's base), would be double-counted.
  - Keep the sum for nomination; optionally store it as `n_model_fits_upstream`.
  - Record new fits = 0 for CMB rows in the resource accounting.

## Summary

| Item | Status |
|---|---|
| nominate.py: labels and order, c1/c2/c3, (a), (b), (d) 3.1237346, near-tie 0.005, tie-breaks, controls, alias | PASS |
| nominate.py N1-N7 (first complete evaluation, raw-field admissibility, reserve-rule hash, reference winners, twin/CC cells, frontier and diagnostic package, edge init) | CHANGE-REQUESTED |
| claims.py E, R2 (1.7267108, max SE), R3, R4, N, P and its failure order, M sequence, C, O, COMPOSITE | PASS |
| claims.py C1 (T edge), C2 (READER_SENSITIVE keys, side, qualifier) | CHANGE-REQUESTED |
| report.py use of claims.evaluate / reader_sensitive, ENHANCED decides, scorecard construction | PASS |
| report.py C3 (CV on AV), C4 (alias resolution), C5 (lock hashes), C6 (family-D marginal intervals, B2TWIN label) | CHANGE-REQUESTED |
| locks.py K_P 46/52, n_claims 2/3, c1, c_P, bootstrap, bands, bank size and dedup, READER_PLAN core | PASS |
| locks.py L1 (alias naming), L2 (independent refit roster), L3 (family-D additions) | CHANGE-REQUESTED |
| final_readers.py selection, order, ignore-one, orientation, controls, refit roster, version asserts | PASS |
| final_readers.py F1 (gap to original winner), F2 (CATI state) | CHANGE-REQUESTED |
| combine.py n_model_fits = sum of sources, for the tie-break | CONFIRMED, with dedup and accounting caveats |
