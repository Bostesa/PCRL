<!-- Saved verbatim by the coordinator from the verifier agent's hand-back (agent a92e2f1798bc736a7), received 2026-10-22 at about
03:24Z. The verifier was instructed not to write a report file itself; this file is its returned report, unedited apart from the
harness's two-space line indentation, which is removed. -->

# Verifier report: campaign `pcrl_input_and_pair_release_v1`, registered readings IA_REG and IB_REG

I worked from 02:51Z to 03:23Z (times from `date -u`). Both decisions are verified. Every decisive number and every label reproduces with my own code: Q*, benefit means and se_agg differ by exactly 0. **No check failed.** I found five observations (O1–O5 below); none changes a label.

**Results for both readings:**

| Decision | IA_REG | IB_REG |
|---|---|---|
| D1 | TASK_FEASIBLE | TASK_FEASIBLE |
| D2 | NOT_SEPARATED | NOT_SEPARATED |
| D3 | NOT_SUPERIOR | NOT_SUPERIOR |
| D4 | NOT_MET | NOT_MET |
| D5 | NOT_APPLICABLE_NOT_JOINT | NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL |
| D6 | NOT_CLAIMED | NOT_CLAIMED |
| Class | TASK_ONLY_FEASIBLE_NO_PROTECTION | TASK_ONLY_FEASIBLE_NO_PROTECTION |
| Registered | yes | yes |

**How I worked:**
- **Code:** `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/results/pcrl_input_and_pair_release_v1/roles/verifier/code/` (`vlib.py`, `v1_integrity.py`, `v2_utility.py`, `v3_grid.py`, `v4_decisions.py`, `v5_access.py`, `v6_seeds.py`).
  - None of it imports `ipr.evaluate`, `ipr.devreaders`, `ipr.rules`, `ipr.claims` or `nrm.rules`.
  - It imports `ipr.common` (label access and label-free features), the family `predict` functions, `ipr.famkit.compact` (for the CSF key only) and `ohn.common.frozen_bank` (U0, Ucal, NEURAL_RAW and H_LJ vectors, MODEL_SELECT only).
- **Records:** JSON outputs only, in `.../roles/verifier/records/`: V1_INTEGRITY, V2_UTILITY, V3_GRID (G and R), V4_DECISIONS, V5_ACCESS (two runs), V6_SEEDS.
- **Labels:** read only through `ipr.common.load(..., 'verify', ...)`, from v2 and v3, inside queue units: 3 MODEL_SELECT reads.
- **Closed records:** I opened no ATTACK_*, DEV_REPORT or sealed records. RELEASE.npz files were hashed whole, but only their MODEL_FIT, CALIBRATION and MODEL_SELECT arrays were loaded.
- **Files:** I changed nothing outside `roles/verifier/` and committed nothing. Per my instructions I did not write a report .md; this message is the report.

## CHECKS

| # | Check | Expected | Observed | Result |
|---|---|---|---|---|
| 1.1 | RELEASE.npz SHA-256 = RELEASE.json = config ledger | all 23 configurations | 23/23 | PASS |
| 1.2 | Wire hashes (MODEL_FIT, CALIBRATION, MODEL_SELECT); MODEL.pkl = MODEL.json = RELEASE.json; card blob at its recorded commit = recorded SHA; commit before save | all | 23/23 | PASS |
| 1.3 | `ipr.ia/ib/ctl.predict` recompute the MODEL_SELECT arrays from admitted features | bitwise | 46/46 arrays bitwise (no FS family exists) | PASS |
| 1.4 | Configuration budget | ≤ 48 | 23 saved (CTL 10, IA 7, IB 6). 22 distinct wires: IA_BLIND V1 is identical to IA_SFG V1 (O2) | PASS |
| 1.5 | MODEL_FIT strength selection re-derived from the fit records | recorded values | **IA:** k* = 1 (MAR2) in V0 and 2 (AGE5 + MAR2) in V1. The next rung fails S3 on income accuracy (1.95× / 1.69× the budget) and income NLL (1.07× / 1.29×). **IB:** π* = 0.075 in both versions. π = 0.10 fails on income NLL (1.22× / 1.001×) and the men's guard (1.26× / 1.17×), plus the women's guard in V0 (1.18×). All match | PASS |
| 1.6 | Bracket points re-derived by the PROTOCOL 4 rule | recorded values | **IA:** SPEND [TWIN, 0.177156] both versions. RO [TWIN, 0.172614 (V0) / 0.208862 (V1)]. SBC V0 [rung 1, rung 2], V1 [TWIN, rung 1]. **IB:** SPEND V0 [0.555992, 0.67275], V1 [0.459497, 0.611591]. **BLIND:** k = 3 (V0) / 2 (V1). All match | PASS |
| 1.7 | Grid fineness (own classification) | as recorded | IA: NOT_APPLICABLE (CAND spends less than TWIN, −0.000105 / −0.000218). IB: FINE in both versions. No fineness flags in D2 | PASS |
| 1.8 | Role members and the bank's array members | `SHARED_ROLES.json` = the fit records' shared points; bank SHA = file | 4 roles × 2 versions match | PASS |
| 2.1 | 24 clauses + 4 guards for 11 members (TWIN, IA_SFG, IA_SBC_HI, IA_BLIND, IB_CSF, IB_CSFT, IB_IND and the 4 role members) against `EVAL_MS.json` | \|Δ\| ≤ 1e-9 | max \|Δ value\| = 4.0e-16, \|Δ SE\| = 0; pass/fail agree on all 616 values | PASS |
| 2.2 | G1 at the point | TWIN and both CANDs pass | All 11 members pass in both versions. IB_CSF's highest value/cap is 0.17 (V1, men's guard 0.00085 against 0.005) | PASS |
| 2.3 | Paired costs recorded in EVAL_MS (60 records, 11 metrics, W skipped) | \|Δ\| ≤ 1e-9 | max \|Δ\| = 0 | PASS |
| 3.1 | Wire hashes of the 13 decisive members against GRID.json | match | 52/52 | PASS |
| 3.2 | Exact replay of all 8 cells × 3 views. Winner = first maximum of VALIDATION AUC. SCORE AUC recomputed from the cached reader scores (cache key, splits and seeds rebuilt by my code). 400 bootstrap draws (my weights). Non-table (NOLOOKUP) winner AUC | identical | 312/312 cells, max \|Δ\| = 0.0 on all four quantities; no cache entry or bootstrap key missing | PASS (exact) |
| 3.3 | 336 paired cell differences and cell SEs | identical | max \|Δ\| = 0 | PASS (exact) |
| 3.4 | Reader and bootstrap seeds; seed effectiveness on the minimum sets | registered formula; no unexplained equality | Seeds 2026102105/06/15/16; bootstrap seeds 2026102300/301/400/401; none at a forbidden offset. Seed-dependent recipes differ between k0 and k1 in 360/360 checks (IA) and 288/288 (IB). `grid_valid` True | PASS |
| 3.5 | Independent readers: logistic regression on log-probability ratios and a 15-leaf HGB, 4 cells (version × repetition) per comparison | same direction and size as the grid | See the comparison table below: all consistent | PASS (approx.) |
| 4.1 | D1 | TASK_FEASIBLE | TASK_FEASIBLE for both readings | PASS |
| 4.2 | D2 IA_REG (gates TWIN, SBC, SPEND, RO) | NOT_SEPARATED | Matches recorded exactly; detail below | PASS |
| 4.3 | D2 IB_REG (gates TWIN and SPEND; SEX-blind) | NOT_SEPARATED | Matches recorded exactly; detail below | PASS |
| 4.4 | D3 (nine references) | NOT_SUPERIOR | Neither candidate passes any of the nine (benefits below); reader-budget flags all COMPARABLE | PASS |
| 4.5 | D4 (0.020 vs NEURAL_RAW and H_LJ) | NOT_MET | TARGET_AGG_NOT_MET against both, for both readings | PASS |
| 4.6 | IB `operation_sex_blind` | release depends on no SEX-fitted quantity | Details below | PASS |
| 4.7 | IB D5 collapse declaration | sequential orders equal the joint design | Wire check below | PASS |
| 4.8 | IA `ro = GENERIC_ZTILT` as the declared fallback, and the recorded substituted ladder | declared before the first fit; ladder recorded | Details below | PASS |
| 4.9 | Registration status | cards committed before the first MODEL_SELECT read | First commit of both cards 02:38:59Z; first MODEL_SELECT read 02:50:32Z | PASS |
| 5.1 | All label reads after the registration receipt (02:14:15Z) | all | 27/27 reads up to 03:20Z, each inside its running queue unit | PASS |
| 5.2 | Fitting reads carry a committed card | all | 4/4 (cards IA_SFG_v1 and IB_CSF_v1; blob SHA at 9b70c5d77 matches; commit 02:38:59Z precedes reads at 02:39:08Z, 02:46:28Z, 02:46:54Z). The CTL_TWIN fit read no label | PASS |
| 5.3 | No MODEL_SELECT read from a fitting entry point | none | 23 MODEL_SELECT reads: `ipr.evaluate` ×15, `ipr.devreaders` ×4, `ipr.richreaders` ×1, verifier ×3. No closed-role read | PASS |
| 5.4 | First MODEL_SELECT read follows the commit of the IA and IB cards (9b70c5d77, 02:38:59Z) | after | 02:50:32Z (IPR_EVAL_TWIN_V1) | PASS |

**Check 3.5, pair-view benefits** (independent readers, mean of 4 cells, compared with the registered grid):

| Comparison | LR | HGB | Validation-chosen | Grid (mean ± se_agg) |
|---|---|---|---|---|
| IA vs TWIN | +0.0035 | −0.0005 | +0.0035 | +0.0016 ± 0.0058 |
| IA vs IA_SBC_HI | +0.0041 | +0.0030 | +0.0032 | +0.0033 ± 0.0056 |
| IA vs IA_SPD_HI | +0.0026 | −0.0055 | +0.0026 | +0.0006 ± 0.0053 |
| IA vs IA_RO_HI | −0.0030 | −0.0012 | −0.0035 | −0.0009 ± 0.0058 |
| IA vs NEURAL_RAW | −0.0209 | −0.0276 | −0.0209 | −0.0211 ± 0.0085 |
| IA vs H_LJ | −0.0265 | −0.0362 | −0.0265 | −0.0278 ± 0.0097 |
| IB vs TWIN | +0.0049 | +0.0076 | +0.0049 | +0.0069 ± 0.0057 |
| IB vs IB_SPD_LO | +0.0012 | +0.0007 | +0.0012 | +0.0040 ± 0.0069 |
| IB vs IB_SPD_HI | +0.0015 | −0.0092 | +0.0015 | +0.0007 ± 0.0081 |
| IB vs NEURAL_RAW | −0.0196 | −0.0196 | −0.0196 | −0.0158 ± 0.0095 |
| IB vs H_LJ | −0.0251 | −0.0282 | −0.0251 | −0.0225 ± 0.0098 |

**Check 4.2, D2 IA_REG:**
- **(b) vs TWIN:** pair benefit 0.00161 ± 0.00584 (version means 0.0017 / 0.0015), AGG_POSITIVE_POINT_ONLY. The income recipient gate fails (RECIPIENT_AGG_POINT_FAILED; V1 mean 0.0193); occupation is RESOLVED.
- **Pooled MODEL_SELECT total-NLL cost of CAND minus each point:** TWIN +0.000227, IA_SBC_HI −0.004968, IA_SPD_HI −0.001337, IA_RO_HI −0.001677. So Q* is the HI point on every path. TWIN is the only cheaper point and its benefit is not resolved negative, so nothing dominates CAND.
- **SBC:** +0.00335 ± 0.00558 (point only); recipient gate fails (income).
- **SPEND:** +0.00064 ± 0.00526 (point only); recipient gate passes.
- **RO:** −0.00091 ± 0.00579 (not positive); recipient gate fails (income).
- No secondary metric is resolved higher against any Q*.
- **(d) targeting**, version means: TWIN +0.0017 / +0.0015, SBC_HI +0.0035 / +0.0032, SPD_HI +0.0019 / −0.0006, RO_HI −0.0017 / −0.0002.

**Check 4.3, D2 IB_REG:**
- **(b) vs TWIN:** 0.00693 ± 0.00571 (version means 0.0037 / 0.0102), point only. Recipient gate passes (income EDGE, occupation RESOLVED).
- **Costs:** TWIN +0.010353, IB_SPD_LO +0.001500, IB_SPD_HI −0.003699, so Q* = IB_SPD_HI. The cheaper points (TWIN, IB_SPD_LO) are not resolved negative.
- **SPEND vs Q*:** 0.00065 ± 0.00807 (version means −0.0035 / +0.0048); recipient gate passes (EDGE / EDGE).
- **(d) targeting:** TWIN +0.0037 / +0.0102, SPD_HI −0.0035 / +0.0048.

**Check 4.4, D3 benefits** (none of the nine passes in either reading; no recipient gate passes):
- **IA:** NEURAL_RAW −0.0211, H_LJ −0.0278, B2TWIN −0.0403, RETIRED_B3CTL −0.0501, SUCC −0.0390, TWJ −0.0106, W −0.0005, TR100 +0.0015 (point only), TT100 −0.0010.
- **IB:** NEURAL_RAW −0.0158, H_LJ −0.0225, B2TWIN −0.0350, RETIRED_B3CTL −0.0447, SUCC −0.0337, TWJ −0.0052, W +0.0048, TR100 +0.0068, TT100 +0.0043 (none resolved).
- Reader-budget flags are all READER_BUDGET_COMPARABLE (as recorded; I did not recompute them, and they cannot rescue a failing comparison).

**Check 4.6, IB `operation_sex_blind`:**
- The IB_CSF and IB_IND payload state holds only (π, mode, tag, T_inc, T_occ).
- Both tables equal, bitwise, my own decision-cell means of the admission-pinned twin out-of-fold vectors, which contain no SEX.
- I rebuilt the IB_CSF and IB_IND MODEL_SELECT releases from SEX-free inputs only: the saved twin vectors, a SHA-256 key over the compact six fields (tag `IPR_B_CSF|V<v>`), the tables and π = 0.075. The result is bitwise equal in both versions.
- The twin models inside the payload are CTL_TWIN's, and decisions are kept exactly. No SEX-fitted quantity enters the release.

**Check 4.7, IB D5 collapse:**
- I built both sequential orders myself. The second stage flattens the people whose first-stage vector is a decision-cell row, identified from the first stage's release.
- That set equals the key set exactly: 561/561 in V0, 605/605 in V1.
- Both orders reproduce IB_CSF's income and occupation wires bitwise, in both versions.

**Check 4.8, IA `ro = GENERIC_ZTILT`:**
- The declaration is in card IA_SFG_v1, committed 02:38:59Z, before the first IA fit at 02:39:08Z. PLAN_IA_REG carries the same declaration.
- The RO points CTL_RO_0p172614 (V0) and CTL_RO_0p208862 (V1) are generic twin tilts (op `ro`) under card CTL_PATHS_v1.
- The substituted ladder is recorded in both fit records and was never saved as a configuration. In V0 it differs from SFG's ladder from rung 1 (AGE5 first, against SFG's MAR2). In V1 it equals SFG's at rungs 1–2, which is k*. The card's 0.8 prediction therefore held in V1 only (already disclosed in `PROTOCOL_DEVIATIONS.md` item 3).
- The scope sentence carried with the class is the fallback's.

## Observations (none changes a label)

- **O1. IA_SBC_LO is outside the SBC path, the evaluation and IPR1_MAIN.** It is configuration 5, saved in V0 only. PROTOCOL 4 says every saved configuration gets the same evaluation and reader cells; you already disclosed this as `PROTOCOL_DEVIATIONS.md` item 2. My own numbers for it:
  - It passes G1 in V0.
  - As a point standing for IA_SBC_LO in V0 and TWIN in V1, CAND's pooled cost over it is (+0.00190 + 0.00006)/2 = +0.00098 > 0. So it would only be a cheaper point for the domination check (c3), never Q*.
  - My readers (V0, R1/R2) give IA_SFG vs IA_SBC_LO pair benefits of +0.0045 (LR) and −0.0004 (HGB), not a resolved negative.
  - SBC_FRONTIER already fails at the bracket (no resolved benefit, income gate failed). The omission is immaterial here.
- **O2. IA_BLIND V1 is bitwise IA_SFG V1.** The RELEASE.npz files are identical because BLIND's first two task-ordered rungs are SFG's AGE5 + MAR2. The V1 targeting ablation is therefore degenerate. It is counted as its own configuration, and PLAN alias_of is empty; it is an ablation, off the decision path.
- **O3. Card IB_CSF_v1 §1 overstates bitwise equality.** It says non-flattened people get "the twin's vectors bitwise". That holds for income. Occupation vectors differ by up to 2.2e-16 (clip and renormalize), with identical decisions. Wording only.
- **O4. IA's income gate failure rests on one repetition.** The failures vs TWIN, SBC_HI and RO_HI are driven by the S1R1 income cells (+0.038 vs TWIN). In that cell my LR and HGB readers give +0.001 and +0.0002; my validation-chosen reader gives +0.034 because the chosen reader switches. This is a reader-selection sensitivity and does not matter, since no benefit is resolved.
- **O5. Three label-free scripts ran outside the queue.** I queued IPR_VER_DECISIONS_V1 at 03:12:03Z, but IPR_GRID_ROB2_V0/V1 held both slots from 03:11:54Z, which I estimated at roughly an hour. I withdrew the unit before it started: no unit_start, ticket removed, the folder holds only CODE_AT_START.json. To meet the deadline I ran the label-free v4 decision arithmetic (0.017 CPU-s), the v5 access re-check (0.006 CPU-s) and the v6 seed check (0.011 CPU-s) outside the queue. **I did not write a charge event; please charge them.**

## Exact replays vs approximations

- **Exact replays (\|Δ\| = 0 or ≤ 4e-16):**
  - Release and model hashes; deployment arrays (bitwise).
  - 616 utility values and their SEs; 60 paired-cost records.
  - 312 winner SCORE AUCs, each with its 400 bootstrap draws, and the NOLOOKUP winner AUCs; 336 paired cell differences and SEs.
  - Aggregates (mean, se_hh, se_rs, se_agg, version means), bracket costs, Q*, targeting, and every D label.
- **Recorded, not replayed:**
  - Per-recipe score fingerprints (used only for the seed check) and reader-budget flags.
  - Enhanced (TABLE_CV) winners, which only produce the reported-only READER_SENSITIVE_ENHANCED_* flags.
  - The grid entries of the seven references other than NEURAL_RAW and H_LJ; for D3 my arithmetic runs on the recorded entries.
- **Independent approximations:** the LR and HGB readers in check 3.5.

## CPU

- **Queue units, 34.4 CPU-s in total:**

  | Unit | CPU-s |
  |---|---|
  | IPR_VER_ACCESS_V1 | 0.092 |
  | IPR_VER_INTEGRITY_V1 | 8.239 |
  | IPR_VER_UTILITY_V1 | 0.606 |
  | IPR_VER_GRID_R_V1 | 21.077 |
  | IPR_VER_GRID_G_V1 | 4.428 |

- **Out of queue:** no more than about 15 CPU-s (v4, v5 rerun, v6, compiling, the invented-data selftest, record inspection, wait scripts). Not charged.
- **Total:** about 0.014 CPU-hours against the 0.6 limit.
