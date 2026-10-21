# Independent verifier, second hand-back (saved verbatim with attribution)

Author: the independent verifier of campaign `pcrl_finite_routing_comparison_v1` (separate code under `roles/verifier/code/`).
Returned at about 00:59Z on 2026-10-21 (its own header: written 00:58:37Z). Saved verbatim by the coordinator (indentation removed).
Its last commit before this hand-back: 1f1065652.

---

**Independent verifier, second hand-back (written 00:58:37Z on 2026-10-21)**

**Bottom line.** Every item you assigned at 00:35Z replays with my own code and agrees with your records, with one exception: I skipped the P1 person-level numbers (spends, SEs, AUCs, F4 bootstrap SE) because they cannot be recomputed without a refit. There is no numerical FAIL.

CHECKS.json now holds 109 checks:

| | PASS | FAIL | ATTENTION | NOT_APPLICABLE | NOT_RUN |
|---|---|---|---|---|---|
| Numerical | 92 | 0 | 0 | 15 | 2 |
| Procedural | 61 | 3 | 4 | 41 | 0 |

All my units since the interim report were label-free. Deviation 9 (the gate refusing label reads) did not affect any of them: my last label read was at 00:25:31Z.

**1. V5d: R1 UNT bracket (FRC_VER_UNT_V2, 00:49:30Z) — PASS in both versions**
- The twin's nested means equal P1's, and the target equals P1's PRC_S2 target.
- I recomputed PRC_S2's out-of-fold re-routing rates relative to the twin with my own routing:
  - V0: income 0.04815, occupation 0.11685.
  - V1: income 0.04485, occupation 0.1025.
  - The f_I / f_O ratio equals the recorded one.
- The first-reaching rule over the registered f_O grid gives UNT@0.1 in V0 and UNT@0.3 in V1.
- The saved UNT_BR books carry the right f_O, f_I, ratio and tag, and the twin's codewords.

**2. V6: registered grid FRC1_MAIN and the ER1 decisions (FRC_VER_GRIDCLAIMS_V1, 00:49:34Z) — PASS**
- **Paired cells.** All 4,800 paired cell values and household-bootstrap SEs reproduce exactly (difference 0.0) from the cell AUCs and BOOT.npz.
- **ER1 grid validity reproduces (effective).** It was rebuilt from the score fingerprints and my own label-free distinct-input counts.
  - 4,032 k0/k1 checks.
  - 129 seed-dependent equalities, 85 of them on the minimum set; all 129 are explained by (a)+(b)+(c).
  - Seeds are the registered +700 values, none forbidden.
  - No seed-invariant recipe changed between seeds.
- **Both readings (MAIN_ER1, R1_ER1) reproduce in full.**

| Decision | Result |
|---|---|
| D1 | REPAIRED_TASK_ONLY_RELEASE |
| D2 | NO_REDUCTION_BEYOND_ANY_CONTROL |
| D3 | NOT_SUPERIOR |
| D4 | NOT_MET |
| Outcome | ENGINEERING_REPAIR |

- **D2 by comparison** (bracket costs are pooled total NLL, PRC_S2 minus the control):

| Comparison | Bracket (cost) | Pair benefit (se_agg) | Result |
|---|---|---|---|
| TWIN | — | 0.0070 (0.0040), POINT_ONLY | recipient gate passes; not passed |
| SHUF, registered path [SHUF_S2] | none; PRC_S2 spends +0.00127 more | — | CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT |
| SHUF, R1 path [SHUF_S2, UNT_BR] | UNT_BR (−0.00219) | 0.0031 (0.0076) | NOT_SEPARATED |
| HQ | HQ_S2 (−0.00265) | 0.0003 (0.0051) | NOT_SEPARATED |
| ORD | ORD_S2 (−0.00185) | 0.0053 (0.0059) | NOT_SEPARATED |

- **F4** passes in both versions. The plan's F4 values equal the SELECTION records.
- **Reader targeting.**
  - TWIN and ORD_S2 pass.
  - HQ_S2 fails: non-table benefit version means are +0.0013 / −0.0006.
  - UNT_BR fails: +0.0084 / −0.0022.
- The plan's nested LOOKUP benefits equal the AUC differences in the P1 and R1 selection records.
- **The nine references all fail.**
  - Six are AGG_NOT_POSITIVE, for example NEURAL_RAW −0.0177 and H_LJ −0.0233.
  - W (+0.002), TR100 (+0.011) and TT100 (+0.0065) are point-only.
  - Every recipient gate fails.
  - Six references are NOT_CREDITED; W, TR100 and TT100 are CREDITED.
  - Both D4 target labels are TARGET_AGG_NOT_MET.

**3. V6b: reader-budget contrasts (FRC_VER_BUDGET_V1) — PASS**
For both plans, every flag reproduces from my own label-free computation, along with each version/repetition's group-size ratio and unseen-share difference. There were 0 mismatches.

**4. V7: robustness grid FRC1_ROB and CLASSIFIER_STABILITY.json (FRC_VER_ROB_V2, 00:57:31Z) — PASS**
- All 2,592 paired cell values reproduce exactly.
- Everything matches the stability record:
  - PRC_S2 is ROBUSTNESS_NOT_CONFIRMED against TWIN, HQ_S2, SHUF_S2, UNT_BR, ORD_S2, NEURAL_RAW and H_LJ.
  - ABL_LOC and ABL_PAIR are AGG_NOT_POSITIVE vs PRC_S2 and AGG_POSITIVE_POINT_ONLY vs TWIN; their robustness is not confirmed.
  - My own splits give four distinct assignment hashes, with SCORE overlaps of 0.2975–0.3132.
- The refit is REFIT_NOT_GENUINELY_EFFECTIVE. The splits part passes; the strict eight-seed rule fails with 240 seed-dependent equalities on the minimum set.
- The first attempt, FRC_VER_ROB_V1 at 00:51:07Z, is NOT_RUN for the comparison only, because the record did not exist yet. Its paired check passed.

**5. Access audit V4 (FRC_VER_ACCESS_V4, 00:58:05Z)**
- 35 reads. The receipt check passes again: the registration files equal the receipt bytes.
- The receipt check now also lists the registration-file history after the research commit (deviation 9):
  - RULES.md and RULES.json changed in 6d6cd0829 at 00:32:13Z (not the receipt bytes).
  - They were restored in adbed977f at 00:53:08Z (receipt bytes).
  - No label read was logged in between; the next reads are at 00:53:34Z.
- The only finding is still R6 (deviation 7). R11 shows no differing entry package across 81 units (deviation 8, ATTENTION).
- Separately, from git and the ledger: the plans were committed at 00:10:28Z (70e84adf9). That is before the grid units started (00:10:39Z) and before both claims runs (00:25:48Z, and the ER1 runs at 00:32:21Z).

**The three procedural FAILs are one cause.** All are V1.ACCESS_LOG R6 (V2, V3, V4): the same append-only card amendment, deviation 7.

**My disclosures**
- At 00:42:31Z I stopped my own chained wait for V6, before it had submitted anything, and resubmitted V6 directly so it kept its queue position.
- FRC_VER_UNT_V1 was stopped while still waiting, before any unit_start; the rerun is FRC_VER_UNT_V2.
- Out-of-queue work is charged as upper estimates (15 CPU-s in total).
- CPU: 20 verifier units (one failed, disclosed earlier) used 34.39 CPU-s, plus the 15 CPU-s charged, for 49.39 CPU-s. The campaign total is 1.718 CPU-h at 00:58:37Z.

**Not covered**
- The readers' own SCORE AUCs and bootstrap draws. Only score fingerprints are saved, so I recomputed paired values from the saved AUCs and draws.
- Winner-recipe selection.
- nrm reader_stability.
- The FRC1_LC learning-curve grid, reader custody and PROBE_STRENGTH.
- P1 person-level numbers (as above).
- Deployment from the backup.

**Files.** Code: v1_access.py (now with R11 and the registration history), v5d_unt_selection.py, v6_grid_claims.py, v6b_reader_budget.py, v7_robustness.py. Records: V5D_UNT__V2, V6_GRID_CLAIMS__V1, V6B_READER_BUDGET__V1, V7_ROBUSTNESS__V1/V2, plus history/. Last commit 1f1065652.
