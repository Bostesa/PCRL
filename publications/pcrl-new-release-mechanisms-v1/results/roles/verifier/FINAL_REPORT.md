# Independent verifier, final report (saved verbatim with attribution)

Author: the independent verifier of campaign pcrl_new_release_mechanisms_v1 (separate numerical code under `roles/verifier/code/`).
Returned to the coordinator after its final pass (16:48-16:57Z on 2026-10-20, its own header). Saved verbatim; nothing below the line
was edited.

---

Independent verifier: final report, campaign pcrl_new_release_mechanisms_v1 (final pass 16:48–16:57Z, 2026-10-20)

Attribution: written by the independent verifier and returned as text.

**Method.** Every result below comes from my own code under `roles/verifier/code/`. All of it ran through the shared queue (stage `verify`, and `deployment_check` for the deployment run). Labels were read only through operation `verify`. I spawned no sub-agents.

## Bottom line

All decision-relevant records agree with my independent replay. There is no procedural FAIL and no blocking discrepancy.

`CHECKS.json` holds 253 append-only checks; the earlier entries' hashes are unchanged.

| Status | Numerical | Procedural |
|---|---|---|
| PASS | 153 | 129 |
| FAIL | 3 | 0 |
| ATTENTION | 4 | 20 |
| NOT_APPLICABLE | 93 | 104 |
| NOT_RUN | 0 | 0 |

## Agreement with your records

**claims/FINAL.** My replay rebuilt everything from the recorded sources and found 0 mismatches.
- Sources: PLAN_FINAL, NRM1_CTL, the evaluation records, the grid, and the cached k0/k1 scores.
- Validity: 5,376 k0/k1 checks with 28 explained ET5/ET25 equalities. PRC_TWIN is invalid on its own minimum set.
- Claims, G1–G7, scores and tiers all match. The outcome is NO_DEVELOPMENT_CANDIDATE.

**claims/FINAL_V2.** I replayed it with the wave-2 definitions of 735cde202 and found 0 mismatches. FB_SEX_S2's selection-record G2 now reads NOT_MET (ENG2-02), as you said. The summaries and the selection are identical to FINAL.

**Decisive records.** All six agree with my own replay of each rule:
- FB2P_RULE: NOT_RUN. Against the same-strength MINF, S5 is AGG_POSITIVE_POINT_ONLY at 0.00186 ± 0.00889, and S3 is POINT_ONLY at 0.00271 ± 0.00890.
- F1_MAIN_V2, F1_FINAL and F1_FINAL_V2: F1_HOLDS. On FINAL_V2, every declared path (MINF and CTL) at S3 and S5 is NOT_SEPARATED_FROM_TASK_SPENDING.
- MEAN_RULE_FINAL and MEAN_RULE_FINAL_V2: NOT_FITTED.

**TABLES section 4 (reserve frontier).**
- My own frontier agrees with both the pre-wave-2 and the post-wave-2 TABLES.json. The post-wave-2 version uses Monte Carlo pooled pair AUC: FB2_SEX_S3 is 0.6536 pooled against 0.6531 on the base instance.
- FB2_SEX_S3 is NO_RESERVE_FRONTIER_IMPROVEMENT. It is point-only against FB2_BASE, FB2_MINF_S3, W, TR100 and TT100.
- The qualified points are FB2_BASE, FB2_SEX_S3, FB2_MINF_S3, W, TR100 and TT100.

**custody/BACKUP_VERIFICATION.json.** My own re-hash of MANIFEST.sha256 equals all four recorded file hashes.

**Deployment from the backup alone.**
- I extracted code.tar and private.tar into my own empty directory. That is the code at tip ae6aa602c, which predates 735cde202. All imported modules were confirmed to come from the restored root.
- All six packages are bitwise equal to three references: the restored stored wire, the restored release, and the live evaluated release (DEV_REPORT, and MODEL_SELECT from features).
- This held through the API, my own CSV, and the underlying entry point. Decisions are the first argmax.
- Each package refused all 21 probes, including my own lowercase `sex`, non-finite input and `cell_id` probes. Only probability and decision are exported.
- This was a warm, same-device check. The extraction was deleted afterwards. B2 (off-device) is still pending.

**Grids.** These replay exactly from the saved reader scores (maximum difference 0.0 on SCORE AUCs, every bootstrap series, paired differences and SEs):
- NRM1_CTL, S0 and S1: 30 members.
- NRM1_ROBUST_REPS, S0 and S1_V2: 8 cells including R3/R4 at instances 6/7. The killed S1 is empty.
- NRM1_ROBUST_SEEDS, S0 and S1: k2/k3.

Member wires equal the saved releases, and every grid unit ran the locked code (lock 1128afd67).

**Evaluations.** All six EVAL_MS__CTL records replay: values within 3.1e-16, SEs within 8.7e-19, and the MC-aware reserve matches.

**Access and budget.**
- The final audit covered 97 reads: no read before the receipt, no closed role, the policy is respected, and every MODEL_SELECT read went through its proper entry point.
- Cards were committed before every fitting read. Unit intervals, row sets and launch code all check out.
- The nrm.readercustody amendment was committed at 16:34:09Z and first used at 16:34:18Z. That use was a single MODEL_SELECT SEX read in unit NRM_READER_CUSTODY_V1.
- The configuration budget stands at 36 of 60.

## Every non-PASS check and its cause

**Numerical FAIL (3)**
- V2.UTILITY.FB.FB_ISO_S2__FB1 and FB_POSTHOC_S2__FB2:
  - In records written before 705966524, randomized releases carried point_pass, margin and related fields from draw 0. That flipped 4 per-clause point_pass flags.
  - The decision fields agree with my replay: values within 1.4e-16, MC SEs within 2.1e-17.
- V6B…MAINV1SUPERSEDED: expected. The superseded MAIN run (deviation 10) gives 91 mismatches; MAIN_V2 agrees exactly.

**Numerical ATTENTION (4)**
- V2B.EVAL_DERIVED_FIELDS__V1, __V2 and __V3: the same draw-0 fields in old records. 999e216b0 fixed `normalized`, and the new CTL-tag records are clean.
- V5.REFIT.NRM1_MAIN.S0__V1: under my weaker own readers, H_LJ is a near tie with FB2_SEX_S3 (+0.0002, against −0.0286 in the grid). In S1R2k0 all four comparisons match in sign. This is not a grid failure.

**Procedural ATTENTION (20)**
- 16 exposure flags, all resolved:
  - Twin units (FB_TWIN, FB2_BASE) also read MODEL_FIT SEX. The FB learners (twin, MINF, ISO) are bitwise invariant to a household SEX permutation and reproduce the saved files. FB2_BASE equals the HC_STRONG + A1_PO_16 wires.
  - The CTL_REM saving units read nothing; the plan units read CALIBRATION occupation only.
- 4 configuration-budget flags: duplicate configuration_number 21 from concurrent saves. The count is unaffected, and the later save lock addresses it.

## My own disclosures

- Three of my units failed: two had malformed ids from zsh word splitting, and one hit a KeyError in my summary code. All were rerun.
- Four of my commit messages rounded the minute up, ahead of the clock. These are listed in records/NOTES.jsonl.

## Not covered

I did not re-score the reader-custody pickles or the RESTORE_CHECK's own reader pass. The off-device copy B2 is still pending.

**CPU.** 287.9 CPU-s in total: 49 verifier units (one is the three failed attempts above) plus 46 s of charged light work, about 0.08 CPU-h.

**Files.** `results/pcrl_new_release_mechanisms_v1/roles/verifier/` (CHECKS.json, records/, code/). The last commit is 631cd1597.
