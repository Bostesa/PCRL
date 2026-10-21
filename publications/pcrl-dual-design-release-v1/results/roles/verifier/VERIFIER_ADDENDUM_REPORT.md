<!-- Saved verbatim by the coordinator after 19:45:54Z (the last clock read before the save) from the verifier agent's second hand-back (agent aace50fb026bc6188, addendum
requested at 19:33Z). The harness's two-space indentation of every line was removed; nothing else was changed. -->

# Verifier addendum report — `pcrl_dual_design_release_v1`, post-registration readings (A1–A8)

All of A1–A8 ran and everything passes; nothing is NOT_RUN. My own arithmetic reproduces every decision and robustness record exactly, and every post-registration grid bank replays exactly. Three minor record discrepancies are listed at the end; none changes a label. JSON records were written normally; please save this text as the addendum report.

**How it ran.** Same rules as the first report: my own code; nothing imported from ddr.evaluate, ddr.devreaders, ddr.rules or ddr.claims for recomputation (`ddr.deploy.release` is used only for A8, as you asked). Every numerical job went through the queue.

**Label reads** (MODEL_SELECT, operation `verify`, both logged):
- DDR_VER_ADD_LABEL_V1 at 19:37:53Z
- DDR_VER_ADD_LABEL_V2 at 19:43:4xZ

**Queue units** (all done, none failed, no code drift):

| Unit | CPU-s |
|---|---|
| DDR_VER_ADD_LABEL_V1 | 32.9 |
| DDR_VER_INTEGRITY_V3 | 42.1 |
| DDR_VER_ADD_LABEL_V2 | 3.6 |
| DDR_VER_ADD_FREE_V1 | 1.7 |
| DDR_VER_ACCESS_V3 | 0.4 |

Charged out-of-queue work: 15 s (dry runs, the v6 one with invented labels, plus record analysis).
- **Addendum CPU:** 95.6 s, or 0.027 CPU-h (limit 0.3).
- **Verifier total for the campaign:** 275 s, or 0.076 CPU-h.

Finished at 19:44Z.

**Exact replay vs approximation.** Every check here is an exact replay: same inputs, my own code.

## CHECKS

| ID | What | Expected | Observed | Tol | Status |
|---|---|---|---|---|---|
| A1.1 | RELEASE.npz = RELEASE.json = CONFIG_LEDGER hashes | 10 new; 48 total | 10/10 new; 48/48 total (DA 22, DB 26) | exact | PASS |
| A1.2 | Per role×task array hashes in RELEASE.json; MODEL.pkl = MODEL.json = RELEASE.json | 48/48 | 48/48 | exact | PASS |
| A1.3 | Card blob at the recorded commit = recorded card SHA, and card committed before the save | before each save | DA_STD_R1 (00eef8eb7, 19:13:36Z): saves 19:14:38–19:17:44Z. DB_PTC_R1_ABL (7def05c9c, 19:21:36Z): saves 19:22:11 / 19:22:14Z. DB_ZTILT_R1 (93da8ca06, 19:26:25Z): saves 19:27:05 / 19:27:11Z. All 48 configurations pass | exact | PASS |
| A1.4 | Deployment function (current code) recomputes the saved MODEL_SELECT arrays | bitwise | 48/48 bitwise | byte equality | PASS |
| A1.5 | Configuration budget | 48 ≤ 48 | 48 distinct, no duplicate rows | exact | PASS |
| A1.6 | DB_PTC_R1_OCC income = DA_TWIN income; occupation = DB_PTC_R1 occupation | bitwise | 24/24 arrays (six roles × two tasks × two versions) | byte equality | PASS |
| A2.1 | 12 clauses + 2 guards per version for 8 configurations (the 10 new configuration-versions plus DB_PTC_R1, DB_SBC_R1_LO, DB_SBC_R1_HI) vs EVAL_MS.json; paired costs, 11 metrics, value and SE | ≤ 1e-9 | Values max \|Δ\| 3.3e-16; SEs 0.0; paired costs 0.0 / 0.0; pass flags and release hashes agree | 1e-9 | PASS |
| A2.2 | G1 (24 + 4) | pass | All 8 pass. Closest to the caps: DB_ZTILT_R1 v1 guard sex0 = 0.004824 (0.96 of cap); DB_SBC_R1_HI v0 guard 0.002256; DB_PTC_R1 v1 guard 0.001625 | inclusive | PASS |
| A2.3 | Two-SE reserve, my implementation of the OHN1 rule (floor = H_LJ SEs; ratio = margin / max(own SE, floor); met iff ≥ 2) | recorded status and ratios; DB_PTC_R1 RESERVE_ONE_SEED_ONLY, v1 ratio 1.87 | All 8 statuses agree; min ratios within 1.7e-14. DB_PTC_R1: ONE_SEED_ONLY (2.664 / **1.866**). DB_PTC_R1_OCC: same (2.664 / 1.866). DA_IPW: MET (2.846 / 3.496). DA_IPW_SBC_LO: MET (4.437 / 3.903). DA_IPW_SBC_HI: MET (2.736 / 2.030). DB_SBC_R1_LO: MET (3.402 / 3.300). DB_SBC_R1_HI: NOT_MET (1.365 / 1.587). DB_ZTILT_R1: NOT_MET (0.862 / 0.089) | 1e-9 | PASS |
| A3–A6.replay | Exact replay of the post-registration banks: winners' SCORE AUCs, household bootstrap with registered seeds (including R3/R4: 2026102202/03 and 2026102302/03, reader seeds k0–k3), paired cells and SEs | ≤ 1e-12; no missing entries | 2976 member-view-cells and 10,656 paired cells, all max \|Δ\| 0.0. Per bank (cells / paired cells): DDR1_R1 192/168; DDR1_DA_R1 192/168; DDR1_ROB 1344/8736; DDR1_ROB_DB_R1 672/1152; DDR1_R1_BESIDE 168/144; DDR1_DA_R1_BESIDE 120/96; DDR1_R1_ZTILT 288/192. 0 missing cache entries, 0 winner violations, 0 wire mismatches | 1e-12 | PASS |
| A3.DB_R1 | D1–D4, all conjuncts, Q*, gates, dominated lists, targeting; adaptive tag and label prefixes | equal | All equal; numbers \|Δ\| 0.0. Tag ADAPTIVE_R1 (card first committed 18:15:30Z, after the first MODEL_SELECT read at 18:09:16Z); every label prefixed `ADAPTIVE_R1__`; D3 NOT_SUPERIOR, D4 NOT_MET; 432 minimum-set seed checks, 0 equalities | exact / 1e-12 | PASS |
| A3.DA_R1 | Same | equal | All equal; \|Δ\| 0.0. Tag ADAPTIVE_R1 (card first committed 19:13:36Z); D3 / D4 equal; 0 seed equalities | exact / 1e-12 | PASS |
| A4.DA_REG_D2R | Robustness and all-cell aggregates, labels, n_eff, qualifiers; refit effectiveness; ablations | equal | Family NOT_CONFIRMED; every comparator equal (\|Δ\| 0.0); refit genuinely effective; ablations equal | 1e-12 | PASS |
| A4.DB_REG_D2R | Same | equal | Family NOT_CONFIRMED; all equal; refit effective; ablations equal | 1e-12 | PASS |
| A4.DB_R1_D2R | Same (banks DDR1_R1 + DDR1_ROB_DB_R1) | equal | Family **ROBUSTNESS_CONFIRMED**; all equal; refit effective; ablations equal | 1e-12 | PASS |
| A4.refit | My split hashes; SCORE overlaps; seeds; seed-dependent equalities | effective | R1–R4 assignment hashes distinct (6ef86bdf…, ce81d294…, 3f0408d5…, 5ac89ae5…); SCORE household overlaps 0.281–0.309; reader seeds k0–k3 and bootstrap seeds R1–R4 equal the registered values; 576 checks per seed pair (k0/k1 and k2/k3), 0 equalities, for every minimum set | exact | PASS |
| A5 | Strongest feasible nonzero release (RULES 8; DB_PTC_R1 strength = α_I + α_O = 1.0 per version) | recorded selection | DA: DA_STD (x 0.017493) over DA_IPW (0.007201). DB: DB_PTC_R1 (0.023893) over DB_PTC (0.016371). x values \|Δ\| 0.0; all eligible (my G1 passes; nonzero strengths; no aliases) | 1e-12 | PASS |
| A6.1 | ZTILT_R1 spend matching: first grid pair, k = 0.5…3 along α_O/α_I = 1.857, whose nested spend reaches DB_PTC_R1's; saved pair = chosen | rule followed | k = 1.5 in both versions (k = 1 falls short: 0.00805 < 0.00896 in v0, 0.00653 < 0.00876 in v1). Chosen spend 0.018674 / 0.015177, i.e. **overshoot 2.08× / 1.73×**. Saved alpha_inc / alpha_occ equal the chosen pair | exact | PASS |
| A6.2 | DB_PTC_R1 vs DB_ZTILT_R1 on DDR1_R1_ZTILT vs BESIDE_DDR1_R1_ZTILT.json | equal | Pair-view label, mean, se_agg and version means equal (\|Δ\| 0.0). The record's single-view fields match my means of all 32 cells exactly (see discrepancy 1) | 1e-12 | PASS (with note) |
| A7 | Access audit over the whole log (63 reads at 19:43:53Z) | 0 problems | 0 problems. benchmark 25, fit 13, readers 22, verify 3. All after the receipt, inside running units, policy-conforming. Fitting reads carry committed family cards (ddr.da.std 3, ddr.db.ptc 2, ddr.db.ptc_r1 6, ddr.da.ipw 2, covering DA_STD_R1, DB_PTC_R1_ABL and DB_ZTILT_R1). None of the 50 MODEL_SELECT reads comes from a fitting module | exact | PASS |
| A8 | Deployment spot-check: packages DA_STD_V1 and DB_PTC_R1_V0, 25 admitted DEV_REPORT raw rows through `ddr.deploy.release`, compared with the saved RELEASE.npz DEV_REPORT rows | bitwise; probes refused | Both packages bitwise on both tasks; decisions equal; only probability and decision exported; SEX-field and extra-field probes refused (PermissionError) | byte equality | PASS |

## Decisive numbers (exact replays)

Pair benefit = comparator AUC − candidate AUC, mean ± se_agg.

**DB_R1 — ADAPTIVE_R1__PARTIAL_TWIN+SPEND_FRONTIER_ONLY**
- vs TWIN: +0.023893 ± 0.010119, AGG_POSITIVE (lower bound +0.0037). Recipient gate passes.
- Pooled total-NLL costs (CAND − point): TWIN +0.009874; SBC_R1_LO +0.005998; SBC_R1_HI −0.000801; SPD_LO +0.008049; SPD_HI −0.002532.
- SBC: Q* = DB_SBC_R1_HI, +0.026541 ± 0.013897 (lower −0.0013) → NOT_SEPARATED.
- SPEND: Q* = DB_SPD_HI, +0.024488 ± 0.012184 (lower +0.00012) → DOMINATES.
- Targeting passes against all three points.
- Flag POINT_FEASIBLE_NOT_RESERVE_QUALIFIED, consistent with A2.3.

**DA_R1 — ADAPTIVE_R1__NOT_SEPARATED**
- vs TWIN: +0.007201 ± 0.008728.
- SBC: Q* = DA_IPW_SBC_HI (costs TWIN +0.007757, LO +0.003935, HI −0.003784), +0.009522 ± 0.016419.
- SPEND: Q* = DA_SPD_HI, +0.003387 ± 0.017835.
- Targeting fails in version 1: −0.00102 vs DA_IPW_SBC_HI and −0.00712 vs DA_SPD_HI. The enhanced winner's numbers equal the non-table winner's, so the two READER_SENSITIVE_ENHANCED flags are consistent.

**D2R robustness cells** (mean ± se):

| Reading | TWIN | SBC Q* | SPEND Q* | Family |
|---|---|---|---|---|
| DA_STD | 0.017629 ± 0.010983 | 0.021935 ± 0.009949 (confirmed) | 0.010289 ± 0.012471 | NOT_CONFIRMED |
| DB_PTC | 0.012595 ± 0.006701 | 0.011930 ± 0.012155 | 0.009636 ± 0.009793 | NOT_CONFIRMED |
| DB_PTC_R1 | 0.021882 ± 0.010316 | 0.023073 ± 0.009111 | 0.018923 ± 0.009095 | **CONFIRMED** |

For DB_PTC_R1, all-cell TWIN is 0.023052 ± 0.007556. NEURAL_RAW and H_LJ are not confirmed for any reading.

**Ablations** (registered cells): DA_COMP −0.030413 vs release and −0.012920 vs twin; DA_STD_OCC −0.006782 / +0.010711; DB_PTC_OCC −0.010082 / +0.006289; DB_PTC_R1_OCC −0.014603 / +0.009290. None is robustness-confirmed.

**ZTILT beside comparisons:**
- DB_PTC_R1 vs DB_ZTILT_R1, registered cells: +0.012821 ± 0.008147, POINT_ONLY (lower −0.0035).
- Same, all 32 cells: +0.007507 ± 0.009339, POINT_ONLY (lower −0.0112).
- For contrast, DB_PTC_R1 vs the old DB_ZTILT (registered cells, DDR1_R1_BESIDE) was +0.024528 ± 0.007548, AGG_POSITIVE.
- So the spend-matched ZTILT_R1 removes the resolved separation. ZTILT_R1 spends about 1.7–2.1× the candidate's MODEL_FIT spend. That follows the card's first-reaching rule on its k grid. It means the comparison is at matched-or-higher control spend, not equal spend.

## Discrepancies and observations

1. **BESIDE_DDR1_R1_ZTILT.json mixes cell sets without saying so.** The pair-view fields (label, mean, se_agg, version means) are aggregates over the 8 registered cells. The single-view fields `income_cand_minus_comp` and `occupation_cand_minus_comp` are plain means over all 32 cells; they differ from the registered-cell means by 0.0013–0.0029. This is a documentation fix, not an arithmetic error.

2. **CAND_STRENGTH_DIFFERS_BY_VERSION is not recorded for DA_R1.** λ* is 1.0 in v0 and 0.75 in v1, but the flag that PROTOCOL 4 requires appears in no record, and PLAN_DA_R1.json has no `strength_differs_by_version` field (PLAN_DA_REG has it).

3. **D2R.json `d2r.release` is null** in all three D2R records. The plans' strongest-release stub lacks a `selected` key. Cosmetic: the release is DA_STD / DB_PTC / DB_PTC_R1.

4. **Which DB D2R is the family's D2R.** DB_REG_D2R is computed for the registered CAND DB_PTC, but STRONGEST.json selects DB_PTC_R1 for DB. Under RULES 8, DB's D2R is DB_R1_D2R (CONFIRMED). DB_REG_D2R should be labelled as extra and descriptive. Also, DB_R1 is an ADAPTIVE_R1 reading; its confirmed D2R never changes D2, which stays PARTIAL_TWIN+SPEND_FRONTIER_ONLY.

5. **Near-limit controls.** DB_ZTILT_R1 passes G1 with its v1 women's guard at 0.96 of the cap, and its reserve is NOT_MET (ratios 0.86 / 0.09). DB_SBC_R1_HI's reserve is also NOT_MET (1.37 / 1.59).

## Not checked

- MODEL_FIT computations themselves: the DA and DB R1 trigger conditions, the IPW and ablation fits, the DA_R1 selection (λ*, γ, brackets), the DB R1 split selection (0.35, 0.65), and ZTILT_R1's nested spends (I checked only the matching rule on the recorded spends).
- TABLE_CV and enhanced winners (only their equality with the replayed non-table winners).
- Data-rich readers, reader custody (including the failed DDR_READER_CUSTODY_R1_V1 unit), CAMPAIGN_SUMMARY.json, and the R2 trigger records.
- Deployment beyond the two packages spot-checked.

## Files

Under `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/results/pcrl_dual_design_release_v1/roles/verifier/`.

New code (`code/`):
- v6_label.py
- v7_addendum.py (reuses v4_decisions.py)

New records (`records/`):
- V1_INTEGRITY__DDR_VER_INTEGRITY_V3.json
- V6_ADD_LABEL__DDR_VER_ADD_LABEL_V1.json
- V6_ADD_LABEL__DDR_VER_ADD_LABEL_V2.json
- V7_ADD_FREE__DDR_VER_ADD_FREE_V1.json
- V5_ACCESS__DDR_VER_ACCESS_V3.json
- CHECKS_ADDENDUM__2026-10-21T194410Z.json (machine-readable table; its A6.2 entry is generic, the A6.2 row above has the field-level comparison)
