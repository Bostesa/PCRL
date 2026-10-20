# Independent verifier, interim reports at 16:24Z (saved verbatim with attribution)

Author: the independent verifier of campaign pcrl_new_release_mechanisms_v1. Two hand-backs returned to the coordinator shortly after
16:24Z on 2026-10-20 (the verifier's `date -u`), the second a fuller version of the first. Both are saved verbatim; nothing below either
line was edited.

Coordinator note (not part of the reports): the "16:23Z request" the verifier answers is the coordinator's message whose header said
16:23Z; that header time was ahead of the clock (the coordinator's clock read 16:21:52Z just after sending it). See PROTOCOL_DEVIATIONS.md
item 11.

---

## First hand-back

INDEPENDENT VERIFIER REPORT: pcrl_new_release_mechanisms_v1. Interim report, 2026-10-20 16:24Z (`date -u`). The harness required the hand-back at this time.

## Bottom line
- **All checks I replayed so far agree with the campaign records.** Accepted inputs were the source admission, the access log, releases, utility, the grid and claims. I used separate code and no sub-agents (0 spawned).
- **Your 16:23Z request on MAIN_V2: AGREEMENT.** Check V6B.CLAIMS_SELECTION.NRM1_MAIN__MAINV2 passes with 0 mismatches.
  - **Validity.** My own k0/k1 equalities come from the cached SCORE scores, not the extras fingerprints: 5,376 checks, 28 seed-dependent equalities.
  - **Which equalities.** All 28 are ET5 or ET25 on low-cardinality views:
    - B2TWIN income: 6 distinct vectors, 8 equalities;
    - RETIRED_B3CTL income: 6 distinct vectors, 8 equalities;
    - PRC_TWIN income: 16 distinct vectors, 8 equalities;
    - PRC_TWIN occupation: 64 distinct vectors, 4 equalities.
  - **Explanations.** Every equality meets A-SC1 (a), (b) and (c). No seed-invariant recipe was non-deterministic.
  - **Validity labels.** Every Family B member is valid. PRC_TWIN is invalid: 12 equalities fall on its own minimum set.
  - **Outcome.** G1–G7, the scores and NO_DEVELOPMENT_CANDIDATE are identical to MAIN_V2.
  - **Superseded MAIN.** Replaying it gives 91 mismatches (check …MAINV1SUPERSEDED, FAIL). All of them come from the deviation-10 bug: validity, G2–G6, the scores, and reserve qualification left False by the invalid-grid branch. This is expected, not a new discrepancy.
- **No blocking discrepancy.** Record-quality findings are listed below.

## Status counts (CHECKS.json, 147 checks; append-only hashes intact)
| Status | Numerical | Procedural |
|---|---|---|
| PASS | 91 | 82 |
| FAIL | 3 | 0 |
| ATTENTION | 3 | 11 |
| NOT_APPLICABLE | 50 | 54 |
| NOT_RUN | 0 | 0 |

## Every non-PASS check, with its cause

**FAIL (numerical), 3 checks.**
1. **V2.UTILITY.FB.FB_ISO_S2__FB1 and V2.UTILITY.FB.FB_POSTHOC_S2__FB2: genuine record inconsistency, not a verifier bug.**
   - Before 705966524, the per-clause point_pass, margin, normalized and margin_se_units of randomized evaluations described draw 0, not the draw average. So did utility point_pass_12 and guard_pass.
   - This produced 4 recorded per-clause point_pass flips.
   - The decision fields replay exactly: values within 1.4e-16, SEs within 4.3e-19, MC SEs within 2.1e-17, plus point summaries, reserve and paired costs.
   - The comparison rule is kept unchanged.
2. **V6B …MAINV1SUPERSEDED.** Expected; see the bottom line.

**ATTENTION (numerical), 3 checks.**
- **V2B.EVAL_DERIVED_FIELDS__V1 and __V2.** The ENG1-06 fix (705966524) left one per-clause field, `normalized`, at its draw-0 value in all 8 FB2 randomized version-records. It is not decision-relevant.
  - Suggested fix: set `c['normalized'] = value / cap` in `nrm.evaluate.utility_of`.
- **V5.REFIT.NRM1_MAIN.S0__V1.** My weaker own readers (LR, HGB, table) disagree in sign on one comparison in S0R1k0: FB2_SEX_S3 against H_LJ, own +0.0002 vs grid −0.0286. The other three comparisons agree in sign. V1 (S1R2k0) passes 4 of 4 with a maximum gap of 0.004.

**ATTENTION (procedural), 11 checks.**
- **Twin units read MODEL_FIT SEX** (FB_TWIN and FB2_BASE, both versions). The twins do not use it:
  - the FB2_BASE wires equal the HC_STRONG income wire and the A1_PO_16 occupation wire exactly;
  - see also the SEX-invariance experiment below.
- **CTL_REM saving units read no label** (4 checks). The reads were made in the plan units CTL_REM_PLAN_V0 and V1: CALIBRATION occupation only, under CTL_CARD_v2 (committed 15:53:54Z, before the 16:17Z reads). The controls are SEX-blind as declared.
- **Duplicate configuration number 21** (3 budget checks). The two PRC_TWIN_C128 saves at 15:34:27Z were concurrent and the counter was not atomic. Only 22 of 60 were used then. A save lock has since been added in nrm/release.py (locked at 1128afd67); later saves carry distinct numbers.

## Checks by task
**1. Source admission and role use: PASS.**
- **Admission replay (label-free):**
  - the ed7bc1ccd pin equals the remote tip;
  - 0f27e5937 is an ancestor of HEAD;
  - published results: 222 files (199 identical, 23 identical after the sanitiser); published code: 28 files (27 + 1);
  - 2,618 private copies equal the source worktree, and 201 mirror hashes match;
  - 22 reference wires and the interpreter versions match.
- **Receipt:** re-verified with my own git calls.
- **POLICY:** nrm.common's POLICY equals ACCESS_POLICY.json.
- **Access-log audit (V1–V3; 78 reads at 16:21Z): no finding.** It covers:
  - no read before the 15:14:43Z receipt;
  - no closed role read;
  - MODEL_SELECT labels only through nrm.evaluate, nrm.screen, nrm.devreaders and my own code;
  - every fit or calibrate read carries a card committed beforehand, whose bytes at that commit match;
  - unit intervals and row-set hashes;
  - code committed at launch;
  - new rule R10: the unit that reads labels runs the operation's own entry point, and no MODEL_SELECT read happens in a fit-stage unit.

**2. Training and calibration exposure: PASS.**
- Releases checked:
  - all 18 FB, the 10 FB2 and the 4 PRC twins: roles read match the logs, full pool 20,000 / 13,337;
  - the 4 CTL_REM releases: CALIBRATION 2,982.
- **Family B SEX-invariance experiment, both versions** (the family's own pure fit function, run in memory):
  - the none, minf:S2 and iso:S2 learners are bitwise invariant to a household permutation of SEX;
  - the positive control sex:S2 changes;
  - all four learner refits reproduce the saved .pt files bitwise.
- **FB2 revision order:** the proposal (15:31:16Z) and ruling (15:38:58Z) precede the card (15:45:14Z), which precedes the first read and save. The revision is disclosed in PROTOCOL_DEVIATIONS.

**3. Task and subgroup metrics: PASS** apart from the stale-field records above.
- **Coverage:** 11 references, plus 11 with wires including TR100 and TT100 (component wires equal), and every evaluated FB, FB2 and PRC release.
- **Agreement:** values within 3.6e-16, SEs within 8.7e-19, paired costs (quadrature MC rule) within 8.7e-19.
- **Governing reserve:** the MC-aware reserve_rules replays.
- Examples:

  | Release | Point eligibility | Reserve (min ratio, v0 / v1) |
  |---|---|---|
  | W | eligible | met, 3.511 / 2.862 |
  | FB2_BASE | eligible | met, 4.578 / 3.296 |
  | FB2_SEX_S3 | eligible | qualified |
  | FB2_SEX_S5 | eligible | not qualified (v1 1.532) |
  | FB2_MINF_S5 | fails v1 | — |
  | FB_TWIN | fails the v1 men's guard | — |

**4. Selection: PASS.**
- My rules are re-implemented from the RULES.md text.
- They agree with nrm.rules on 3,000 invented entries and 150 invented ledgers.
- The MAIN_V2 replay from the recorded sources agrees exactly.

**5. Transmitted outputs: PASS.**
- **Validity:** all arrays are valid float64 probability vectors (maximum row-sum deviation 6.7e-16), with no tied argmax.
- **Hashes:** the array hashes equal the RELEASE.json records.
- **Grid wires:** the wires the grid read equal the saved releases. Per-cell draws follow the instance map R1→0, R2→1 (+2/+4 offsets).
- **Finite cells:** PRC_TWIN has 16 / 64 values (636 / 616 linked, with 100 / 77 singleton linked values); C128 has 24 / 128.

**6. Recovery: PASS.**
- **Grid replay:** NRM1_MAIN, both seeds, from the saved reader scores. Cache keys, winners, SCORE AUCs, 336 × 400 bootstrap draws, paired differences and SEs all match with difference 0.0.
- **Lock:** the grid ran the locked code (1128afd67).
- **Own-reader refits:** described under ATTENTION above.

## Not yet done, pending the coordinator's next outputs
- the CTL_REM evaluation and its grid;
- PLAN_FINAL claims with the CTL path;
- the robustness grids;
- MEAN and FB2P;
- deployment from restored artifacts: no package or backup existed yet. Code is ready in v7_deploy.py / v7_child.py.

The same scripts rerun on those outputs: v2, v3, v4, v6b and v7.

## Process disclosures
- **Failed units (3):**
  - `VER_ACCESS_SELFTEST_V1 v1_access.py` and `VER_UTILITY_SELFTEST_V1 v2_utility.py` (0.015 CPU-s each): my zsh loop did not word-split the arguments. Rerun as V2.
  - VER_UTILITY_FB2_EVAL_V1 (0.54 CPU-s): a KeyError in my own summary code. It logged one verify read and wrote no check. Rerun as V2.
- **Commit messages:** four of my record commits rounded the minute up, up to 31 s ahead of the clock (42ea85eab, d8e8d05ba, 07ed52dad, 9ee8a53b4). They are not amended and are recorded in NOTES.
- **Invented-data selftests:** some selftest tolerances and fixtures were adjusted before first use. No real-data rule was changed.
- **CPU:** 37 queue units, 141.8 CPU-s, plus 40 CPU-s charged for out-of-queue work.

## Files
All under `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/results/pcrl_new_release_mechanisms_v1/roles/verifier/`; last commit 1c4b2cfec.
- `CHECKS.json`
- `records/` (V*_*.json, NOTES.jsonl, history/)
- `code/`: vlib, vref, v0–v7, v2b, v3r, v3x, v6b, v_summary

---

## Second hand-back

Independent verifier report: campaign pcrl_new_release_mechanisms_v1 (interim hand-back at 16:24Z, 2026-10-20)

Attribution: written by the independent verifier and returned as text.

**Scope and status.** I was asked to hand back now, so this is an interim report. Everything that existed by 16:24Z has been verified. The late-phase items (listed at the end) were not available yet. The code for them is written and committed.

I spawned no sub-agents (0).

## Bottom line

- **Source admission, receipt and access log:**
  - My own replays of source admission and the registration receipt pass.
  - The label-access audit of 76 reads finds nothing under rules R2–R10.
- **Utility replays:**
  - Every replayed utility number agrees with the records: values within 3.6e-16, SEs within 8.7e-19, MC SEs within 3.1e-17, paired costs within 8.7e-19.
  - This covers all references (including TR100/TT100), all 18 FB_*, all 10 FB2_* and the FA twin.
- **Main grid NRM1_MAIN:** reproduces exactly from the saved reader scores in both seeds (maximum difference 0.0).
- **MAIN_V2 claims and selection (your 16:23Z request): agreement.**
  - My own replay is V6B.CLAIMS_SELECTION.NRM1_MAIN__MAINV2: PASS, 0 mismatches.
  - It ran on the recorded sources, not your in-memory ledger.
  - Validity from my own cached k0/k1 equalities: 5,376 checks and 28 seed-dependent equalities.
    - All 28 are ET5/ET25 on low-cardinality views: B2TWIN income (6 distinct vectors), RETIRED_B3CTL income (6), PRC_TWIN income (16) and occupation (64).
    - All 28 satisfy A-SC1 (a), (b) and (c).
    - Every FB member's grid is valid. PRC_TWIN's is invalid: 12 equalities fall on its own minimum set.
  - G1–G7, scores and tiers are identical. FB2_SEX_S3's score is −0.0374, with G2 NOT_MET, G3 NOT_SEPARATED_FROM_TASK_SPENDING and G4 NOT_SUPERIOR.
  - The outcome is NO_DEVELOPMENT_CANDIDATE.
  - Replaying the superseded MAIN gives 91 mismatches (validity, G2–G6, scores, and reserve qualification left False). All of them come from the deviation-10 bug, as expected.
- **No blocking discrepancy found.**

## Check counts

CHECKS.json holds 147 checks, append-only (entry hashes intact).

| Status | Numerical | Procedural |
|---|---|---|
| PASS | 91 | 82 |
| FAIL | 3 | 0 |
| ATTENTION | 3 | 11 |
| NOT_APPLICABLE | 50 | 54 |
| NOT_RUN | 0 | 0 |

## Every check, by group (numerical / procedural)

**Self-tests on invented data**
- V0.VLIB.SELFTEST__V1, V1.ACCESS_AUDIT_SELFTEST__V2 and __V3, V2.UTILITY_SELFTEST__V2 and __V3, V5.REFIT_SELFTEST__V1, V6.SELECT_SELFTEST__V1: PASS / NA.
- V6.RULES_AGREEMENT_INVENTED__V1: PASS / NA. My rule implementation agrees with nrm.rules on 3,000 invented entries and 150 invented ledgers.

**Reader splits**
- V0.STRUCT.READER_SPLITS__V1: PASS / NA. R1–R4 and the screening halves rebuild exactly, including hashes, overlaps, seeds and recipes.

**Source admission and access log**
- V1.SOURCE_ADMISSION__V1: PASS / PASS. Remote pin, 222+28 published files, 2,618 private copies, 201 mirror hashes, 22 reference wires and interpreter versions all match.
- V1.RECEIPT__V1/V2/V3 and V1.ACCESS_POLICY_CODE__V1/V2/V3: NA / PASS.
- V1.ACCESS_LOG__V1 (6 reads), __V2 (37) and __V3 (76): NA / PASS, no finding.
- V1.ROW_SETS__V1/V2/V3: PASS / NA.

**Utility replays (V2)**
- V2.UTILITY.REF.*__V1 (11 references) and __V2W (11, including TR100/TT100 with component wires equal): PASS / PASS.
  - W occupation NLL excess vs U0 is −0.01614 / −0.01280, and men's Brier excess −0.00385 / −0.00229.
  - Reserve ratios equal the published labels, e.g. NEURAL_RAW 1.947/1.558, H_LJ 0.321/0.624, W 3.511/2.862, TR100 3.412/2.862, TT100 3.326/2.600.
- V2.UTILITY.FB.* (FB1, FB2 tags): PASS / PASS, except FB_ISO_S2__FB1 and FB_POSTHOC_S2__FB2, which are FAIL / PASS (finding 1).
- V2.UTILITY.FB.FB2_*__FB2E (5): PASS / PASS, including the governing MC-aware reserve_rules.
  - FB2_SEX_S3: qualified, 3.898 / 2.816.
  - FB2_SEX_S5: not qualified (v1 1.532).
  - FB2_MINF_S3: qualified, 3.894 / 3.175.
  - FB2_MINF_S5: fails occupation|Ucal|nll in v1.
  - FB2_BASE: 4.578 / 3.296.
- V2.UTILITY.FA.PRC_TWIN__FA1: PASS / PASS.
- V2B.EVAL_DERIVED_FIELDS__V1 and __V2: ATTENTION / NA (finding 1).

**Outputs and exposure (V3)**
- V3.OUTPUTS.* (FB 18, FB2 10, FA 4, CTL 4): all PASS / NA. Valid float64 vectors on all six roles, array hashes equal, and finite-cell cardinalities recorded (e.g. PRC_TWIN 16 / 64 / 636 linked values).
- V3.EXPOSURE.*: PASS, except the following ATTENTION items, which are resolved by evidence:
  - FB_TWIN and FB2_BASE: their units also read MODEL_FIT SEX.
  - CTL_REM: the saving unit had no reads; the plan units read CALIBRATION occupation only, under CTL_CARD_v2.
- V3X.FB_SEX_INVARIANCE.V0/V1__V1: PASS / PASS. The twin, MINF_S2 and ISO_S2 learners are bitwise invariant to a household permutation of SEX. sex:S2 changes. All refits reproduce the saved learners bitwise.
- V3R.REVISION.FB_FSC_v2__V1: PASS / PASS. The approvals at 15:31:16Z and 15:38:58Z precede the card at 15:45:14Z, which precedes the first read and save. The revision is disclosed, and the base equals the HC_STRONG + A1_PO_16 wires.
- V3.CONFIG_BUDGET__FB1: NA / PASS. __FA1: NA / PASS. __FA2, __FB2 and __CTL1: NA / ATTENTION (finding 2).

**Grid and reader refits (V4, V5)**
- V4.GRID.NRM1_MAIN.S0/S1__V1: PASS / PASS. All 336 bootstrap series, paired differences and SEs are identical. Member wires equal the saved releases, the instance map is right, and the code equals lock 1128afd67.
- V5.REFIT.NRM1_MAIN.S1__V1: PASS. 4 of 4 signs agree; maximum gap 0.0042.
- V5.REFIT.NRM1_MAIN.S0__V1: ATTENTION.
  - 3 of 4 signs agree.
  - H_LJ is a near tie under my weaker own readers: +0.0002 vs the grid's −0.0286, with FB2_SEX_S3 at 0.634 vs 0.665 in the grid.
  - This is not a grid failure.

**Claims and selection (V6B)**
- V6B.CLAIMS_SELECTION.NRM1_MAIN__MAINV2: PASS / NA.
- …__MAINV1SUPERSEDED: FAIL / NA (expected; deviation 10).

## Findings

1. **Randomized evaluation records carried draw-0 derived fields.**
   - Before 705966524, per-clause point_pass, margin, normalized and margin_se_units, and point_pass_12 / guard_pass, described draw 0 instead of the draw average.
   - This flipped point_pass on 4 clauses of FB_ISO_S2 and FB_POSTHOC_S2.
   - After the ENG1-06 fix only `normalized` is still draw 0 (8 FB2 version-records).
   - It is not decision-relevant: the point summaries, reserves and rules all use value/cap and agree.
2. **Concurrent saves gave a duplicate configuration_number.**
   - FA/PRC_TWIN_C128 V0 and V1 both got 21, because the budget counter was not atomic.
   - The distinct count is correct (36 of 60).
   - The later save lock (lock files 1128afd67) addresses this.
3. **My own process disclosures:**
   - Three of my queue units failed. Two had malformed ids from zsh word splitting (0.015 CPU-s each, rerun as V2). VER_UTILITY_FB2_EVAL_V1 hit a KeyError in my summary code; it was rerun as V2.
   - Four of my record-commit messages rounded the minute up, stating times up to 31 s ahead of the clock. These are disclosed in records/NOTES.jsonl and not amended.
   - I used about 150 CPU-s in total: 37 units plus 40 s of charges.

## Not run yet (code committed and ready)

- **Robustness grids (NRM1_ROBUST_*) and any CTL grid:** v4_grid.py.
- **CTL_REM, MEAN and FB2P evaluations:** v2_utility.py and v3_outputs.py.
- **PLAN_FINAL claims with the CTL path:** v6b_claims.py.
- **Deployment from the restored backup:** v7_deploy.py / v7_child.py.
  - It does its own manifest re-hash and extraction, checks bitwise parity against the stored, restored and live wires, and runs my own refusal probes.
  - Use `--interpreter` with the torch venv for FB packages.
  - No package or backup existed at hand-back.
- **Final access audit:** v1_access.py.

## Files

- `results/pcrl_new_release_mechanisms_v1/roles/verifier/CHECKS.json` (147 checks)
- `results/pcrl_new_release_mechanisms_v1/roles/verifier/records/` (V*_*.json, NOTES.jsonl, history/)
- `results/pcrl_new_release_mechanisms_v1/roles/verifier/code/` (vlib.py, vref.py, v0–v7 scripts, v_summary.py)

The last verifier commit is 1c4b2cfec.

---

## Third hand-back (same interim report)

The verifier returned the second hand-back's text a third time, identical except for one added sentence after "I spawned no sub-agents (0).":

"After this hand-back I only stopped my own background progress monitor; I did no further work."
