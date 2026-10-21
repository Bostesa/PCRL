<!-- Saved verbatim by the coordinator at 19:18Z from the verifier agent's hand-back (agent aace50fb026bc6188); the harness refused the
agent's own write of this file. The harness's two-space indentation of every line was removed; nothing else was changed. -->

# Independent verifier report — pcrl_dual_design_release_v1 (registered readings DA_REG and DB_REG)

The harness refused my report .md file; please save this text verbatim. The JSON records under `roles/verifier/records/` were written normally. All checks in the brief ran; nothing is NOT_RUN. Both registered decision records reproduce exactly. There is one FAIL, a procedural one (V2.5).

**Scope and how it ran.** I verified checks (1) to (5) of `VERIFIER_BRIEF.md` with my own code. The utility, AUC, bootstrap, aggregate and decision arithmetic do not import ddr.evaluate, ddr.devreaders, ddr.rules or ddr.claims. Imported only for data:
- ddr.common for label-free features and the one logged label read;
- ddr.da.predict / ddr.db.predict, which load only each configuration's model file;
- tsz.evaluate.refs_of for the frozen U0, Ucal, NEURAL_RAW and H_LJ vectors.

There was **one label read**: MODEL_SELECT income, occupation and SEX, operation `verify`, at 18:33:15Z, in unit DDR_VER_LABEL_V1 from `v2_label.py`. It is in the access log.

**Queue units** (all done, none failed, no code changed between registration and launch):
- DDR_VER_LABEL_V1: 28.6 CPU-s
- DDR_VER_INTEGRITY_V1: 35.0 CPU-s
- DDR_VER_ACCESS_V1: 0.3 CPU-s
- DDR_VER_DECISIONS_V1: 0.3 CPU-s
- DDR_VER_ACCESS_V2: 0.4 CPU-s
- DDR_VER_INTEGRITY_V2: 34.9 CPU-s

**Charged label-free work outside the queue:** dry runs (60 s; the dry run of `v2_label.py` used invented labels) and record analysis (20 s). **Total about 179 CPU-s, or 0.050 CPU-h**, one thread. I committed nothing.

**Exact replay vs approximation.** V2, V3.1–V3.7 and V4 are exact replays: same inputs, my own code, differences of 0 to about 1e-16. V3.8 is an independent approximation: my own LR and gradient-boosting readers on the R1/R2 FIT/VALIDATION/SCORE parts, 4 cells per comparison.

## CHECKS

| ID | What | Expected | Observed | Tol | Status |
|---|---|---|---|---|---|
| V1.1 | RELEASE.npz SHA-256 = RELEASE.json = CONFIG_LEDGER | 38/38 | 38/38 (32 registered + 6 DB R1) | exact | PASS |
| V1.2 | Per role×task array hashes in RELEASE.json = my hash | 38/38 | 38/38 (12 arrays each) | exact | PASS |
| V1.3 | MODEL.pkl SHA = MODEL.json = RELEASE.json model_sha256 | 38/38 | 38/38 | exact | PASS |
| V1.4 | Card blob at the recorded commit hashes to the recorded card SHA; card committed before the save | 38/38 | 38/38 (5f2fd02a9 for the 32 registered; ae967496a for R1) | exact | PASS |
| V1.5 | Deployment function on admitted MODEL_SELECT features reproduces the saved arrays | bitwise | 38/38 bitwise, both tasks; all six roles bitwise for DA_TWIN, DA_STD, DB_PTC in both versions. Re-run at 19:13Z with current code (after the ipw / ptc_r1 dispatch edits): still 38/38 bitwise | byte equality | PASS |
| V1.6 | Configuration budget | ≤ 48 | 38 (DA 16, DB 22) | exact | PASS |
| V1.7 | MODEL_FIT strength rule re-derived from recorded spends (estimate + 2 SE ≤ S3) | DA λ* 1.0/1.0; DB α* 0.35/0.35 | Same. DB fails at α = 0.5 on income accuracy, Brier and NLL in v0, and income Brier and NLL in v1 | exact | PASS |
| V1.8 | Control bracket points re-derived from recorded MODEL_FIT bracket grids | recorded LO/HI | DA SPEND (0.35, 0.7) in both versions. DA SBC v0 (0.5, 1.0) with PSEUDO2 γ=2; v1 (0.75, 1.0) with γ=1. DB SPEND (0.25, 0.6) in both. DB SBC v0 (0.134, 0.335) with PERMUTED_SEX; v1 (0.130, 0.326) with PSEUDO2 γ=1. All equal the saved strengths; no CONTROL_PATH_MAY_NOT_BRACKET | exact | PASS |
| V2.1 | 12 clauses + 2 guards per version for the 15 configurations with EVAL_MS.json | ≤ 1e-9 | max \|Δ\| 3.5e-16 | 1e-9 | PASS |
| V2.2 | Clause and guard SEs (clustered and linearized subgroup) | ≤ 1e-9 | 0.0 | 1e-9 | PASS |
| V2.3 | Paired costs: 11 metrics, value and SE, 170 config×version×comparator entries (all except W) | ≤ 1e-9 | 0.0 / 0.0; key sets equal | 1e-9 | PASS |
| V2.4 | G1 (24 + 4 at the point), point-pass flags | TWIN, candidates and control points pass | All 16 DA/DB configurations pass. Max value/cap: DA_STD 0.099 (v1 guard sex0 +0.000496); DB_PTC 0.0125 (v1 income U0 accuracy +0.000125); DA_TWIN −0.10. Flags agree with EVAL_MS. Public UTILITY_LEDGER.csv agrees to its 6-decimal rounding (≤ 5e-7) | inclusive | PASS |
| V2.5 | Every saved configuration has a utility record before the grid (PROTOCOL 4, equal evaluation effort) | 16/16 | DB_ZTILT missing at 18:33Z: it was not in DDR_EVAL_DB_V1 | exact | FAIL (procedural) |
| V2.6 | Late DB_ZTILT record (DDR_EVAL_DB_ZTILT_V1, 19:13:08Z) vs my recomputation | ≤ 1e-9 | 1.4e-16, SE 0, flags and release hash agree; G1 passes (max value/cap 0.25) | 1e-9 | PASS |
| V3.1 | Wire hashes of the 15 decisive members vs GRID.json members | 60/60 | 60/60 | exact | PASS |
| V3.2 | Reader-cache entries found under the registered key (seeds 2026102005/6 and 2026102015/6) | 360/360 | 360/360, so the registered reader seeds were used | exact | PASS |
| V3.3 | Recorded winner's VALIDATION AUC = maximum of its candidate set | 0 violations | 0 | exact | PASS |
| V3.4 | Exact replay of the winners' SCORE AUCs (own AUC) | ≤ 1e-12 | 0.0 on all 360 | 1e-12 | PASS |
| V3.5 | Household bootstrap (own weights, seed 2026102200 + 100v + rep) vs BOOT.npz | ≤ 1e-12 | 0.0 on all 360 × 400 draws | 1e-12 | PASS |
| V3.6 | Paired cell differences and cell SEs for the decisive comparisons (DA_STD vs 7, DB_PTC vs 9; 3 views; 8 cells) | ≤ 1e-12 | 0.0 / 0.0 on 384 cells. All 4368 paired cells in GRID.json are internally consistent with the stored AUCs and BOOT (0.0) | 1e-12 | PASS |
| V3.7 | Non-table (targeting) winner | consistent | Equals the main pair winner in 216/216 member-cells; its validation AUC is the maximum of the non-LOOKUP candidates. So the targeting inputs are covered by V3.4 | exact | PASS |
| V3.8 | Independent readers: sign of the pair benefit vs TWIN, SBC, SPEND, ZTILT, SHUF | same sign as the exact grid | 14/14 same sign for LR, HGB and my winner. Vs NEURAL_RAW and H_LJ the sign depends on the reader (table below) | sign | PASS (descriptive) |
| V4.DA.labels | D1, D2 and every conjunct (gates, Q*, dominated list, secondary metrics, targeting) | equal | All equal | exact | PASS |
| V4.DA.numbers | Benefit means and se_agg, pooled total-NLL costs, targeting version means | ≤ 1e-12 | 0.0 | 1e-12 | PASS |
| V4.DA.D3D4 | D3 (G4) and D4 level | equal | NOT_SUPERIOR (all 9 fail) / NOT_MET | exact | PASS |
| V4.DA.validity | Seed-dependent recipe equalities on the minimum set (RULES 9.3) and seed records | none unexplained; registered seeds | 432 checks, 0 equalities; seeds and bootstrap seeds equal the registered values; READER_SEEDS_EFFECTIVE | exact | PASS |
| V4.DB.labels / numbers / D3D4 / validity | Same four checks for DB_REG | equal; ≤ 1e-12 | All equal; 0.0; NOT_SUPERIOR / NOT_MET; 432 checks, 0 equalities, seeds match | as above | PASS |
| V5.1 | Every label read after the verified receipt (17:42:42Z) | all | 34/34; first read 18:06:27Z | exact | PASS |
| V5.2 | Each read inside its running queue unit; policy (role × field × operation) and authorized entry point, including the ddr.richreaders amendment | all | 0 problems (benchmark 20, fit 7, readers 6, verify 1) | exact | PASS |
| V5.3 | Fitting reads carry a committed, unchanged card of their family; card committed before the read | all | 7/7: four registered fits, two DB R1 fits, one DA R1 trigger (all `fit` on MODEL_FIT) | exact | PASS |
| V5.4 | No MODEL_SELECT read from a fitting entry point or module chain | none | 0 of 27 MODEL_SELECT reads | exact | PASS |
| V5.5 | First MODEL_SELECT read after the commit of both cards (5f2fd02a9, 18:06:14Z, both cards in that commit) | ≥ 18:06:14Z | 18:09:16Z (DDR_EVAL_DA_V1) | exact | PASS |

**Other checks that passed:**
- All 8 registration files are unchanged since the receipt.
- The rules_sha256 in DECISIONS.json equals the receipt's RULES.json hash.
- Both plans and the DDR1_MAIN bank were committed at 18:09:09Z, before the first MODEL_SELECT read and before any grid output (18:31:57Z).
- DEVGRID_CODE_LOCK.json was last committed at registration, and its hashes match devreaders.py, readerextra.py and release.py.

## Decisive numbers (exact replays)

Pair benefit = comparator AUC − candidate AUC, mean ± se_agg, with the two version means in brackets.

**DA_REG — D2 = PARTIAL_SBC_FRONTIER_ONLY; outcome TASK_ONLY_FEASIBLE_NO_PROTECTION**
- vs TWIN: 0.017493 ± 0.008973 [0.0160 / 0.0190]. AGG_POSITIVE_POINT_ONLY (lower bound −0.00045). Recipient gate passes.
- SBC path, pooled total-NLL cost (CAND − point): TWIN +0.00805, LO +0.00273, HI −0.00736, so Q* = DA_SBC_HI. Benefit 0.027287 ± 0.010428 → DOMINATES_CONTROL_FRONTIER; no secondary metric higher; no cheaper point dominates. Passes.
- SPEND path: TWIN +0.00805, LO +0.00391, HI −0.00740, so Q* = DA_SPD_HI. Benefit 0.013680 ± 0.017449 → NOT_SEPARATED.
- Targeting version means: TWIN 0.0160 / 0.0190; SBC_HI 0.0302 / 0.0243; SPD_HI 0.0194 / 0.0079. All pass.

**DB_REG — D2 = NOT_SEPARATED; outcome TASK_ONLY_FEASIBLE_NO_PROTECTION**
- vs TWIN: 0.016371 ± 0.010705 [0.0236 / 0.0091]. POINT_ONLY; recipient gate passes.
- SBC path: costs TWIN +0.00528, LO +0.00337, HI −0.00430, so Q* = DB_SBC_HI. Benefit 0.017289 ± 0.017055 → NOT_SEPARATED.
- SPEND path: costs TWIN +0.00528, LO +0.00345, HI −0.00713, so Q* = DB_SPD_HI. Benefit 0.016966 ± 0.009381 → NOT_SEPARATED.
- Targeting passes against all three points.

**Independent readers (approximation), pair view:** exact grid aggregate vs my LR / HGB / my winner.

| Comparison | Grid (exact) | My LR | My HGB | My winner |
|---|---|---|---|---|
| DA_STD vs TWIN | 0.0175 | 0.0256 | 0.0128 | 0.0207 |
| DA_STD vs SBC_HI | 0.0273 | 0.0262 | 0.0184 | 0.0278 |
| DA_STD vs SPD_HI | 0.0137 | 0.0197 | 0.0046 | 0.0148 |
| DB_PTC vs TWIN | 0.0164 | 0.0268 | 0.0149 | 0.0193 |
| DB_PTC vs SBC_HI | 0.0173 | 0.0220 | 0.0123 | 0.0182 |
| DB_PTC vs SPD_HI | 0.0170 | 0.0231 | 0.0040 | 0.0168 |
| DB_PTC vs ZTILT | 0.0170 | 0.0040 | 0.0074 | 0.0063 |
| DA_STD vs NEURAL_RAW | 0.0046 | 0.0066 | −0.0065 | — |
| DA_STD vs H_LJ | −0.0122 | 0.0059 | −0.0242 | — |
| DB_PTC vs NEURAL_RAW | 0.0035 | 0.0078 | −0.0044 | — |
| DB_PTC vs H_LJ | −0.0133 | 0.0071 | −0.0220 | — |

D3 and D4 fail with any of these readers, so the reader-dependent signs vs the references change no decision.

## Discrepancies and observations

1. **V2.5 FAIL, procedural.** DB_ZTILT was saved, is in DDR1_MAIN, and is reported beside D2, but it was not utility-evaluated before the grid. The evaluation came later (DDR_EVAL_DB_ZTILT_V1, 19:13Z, after the grid and claims) and matches my recomputation, which I computed at 18:33Z (G1 passes). No decision uses it. It should be disclosed as a deviation from PROTOCOL 4's equal evaluation effort.

2. **Interpretation caveat for DA's SBC pass** (my computation from the replayed cell AUCs, label-free).
   - Both DA SBC points are more recoverable than DA_TWIN. Pair AUC minus TWIN, version means: DA_SBC_LO +0.016 / +0.003, DA_SBC_HI +0.014 / +0.005. DA_COMP is also higher, +0.014 / +0.012.
   - So DA_STD's resolved benefit over Q* = DA_SBC_HI is partly a comparison against a control path that adds recoverable signal. DA_STD vs its own TWIN is not resolved.
   - The rules were applied correctly, and the partial outcome is not a claim. This caveat should be stated beside the reading.

3. **DB declared reading beside D2.** DB_PTC vs DB_ZTILT is 0.0170 ± 0.0121 (lower bound −0.0072): not resolved. Under the card's rule, that is the "PRC failure repeated" reading. At the point, ZTILT's pair AUC is about TWIN's (+0.001 / +0.000), and my own readers give a smaller benefit (+0.004 to +0.007).

4. **Code changes during grid runs** (benign).
   - The coordinator's grid units record code_changed_during_run:
     - MAIN units: ddr/db/ptc.py (a 3-line R1 dispatch), ptc_r1.py, plan.py, report.py, restore_check.py.
     - ROB2/ROB3 units: ddr/da/std.py (ipw dispatch and trigger_r1), ipw.py, plan.py, richreaders.py.
     - My verifier files, written while those units ran, also appear in these lists.
   - None of these is on the grid's code path; the locked grid files are unchanged. V1.5 confirms that all registered releases still redeploy bitwise with the current family modules.
   - ddr/richreaders.py was changed (commit 1c7064bd6, 18:48Z) after DDR_RICH_MAIN_V1 ran at 18:34Z. This is outside my checks.

5. **Stale documentation.** The docstring of ddr/da/std.py still describes the pre-amendment SPEND path (KL k-means coarsening) and the old PSEUDO_γ fallback. The code, the card and the fit records use DQ and PSEUDO2.

6. **Control constructions differ by version.**
   - DB's SBC attribute is PERMUTED_SEX in v0, where Cov(q, s) = 1.48e-4, just above the 1e-4 degeneracy threshold. Its profile (b) is not recorded in db/FIT_V0.json. In v1 the permuted column was degenerate (−1.33e-4), so PSEUDO2 γ=1 was used.
   - DA's SBC uses PSEUDO2 γ=2 in v0 and γ=1 in v1.
   - The PSEUDO2 independence values (|corr| 0.0046, 0.0040, 0.0022 ≤ 0.03) are as recorded; I did not recompute them.

7. **Revision status.** The DB_PTC_R1 card was first committed at 18:15:30Z, after the first MODEL_SELECT read at 18:09:16Z. Under RULES 7 / 10.1 it is ADAPTIVE, which matches the coordinator's ADAPTIVE_R1 tag.

8. **Exact aliases are by design.**
   - Income: DA_STD_OCC = DA_TWIN = DB_PTC_OCC.
   - Occupation: DA_STD = DA_STD_OCC and DB_PTC = DB_PTC_OCC.
   - R1 income: DB_PTC = DB_PTC_R1 and DB_SBC_LO = DB_SBC_R1_LO.
   - No control point aliases TWIN or CAND, so alias_of {} in the plans is correct.

**Not run (outside the brief):** recomputing the MODEL_FIT spends from MODEL_FIT labels (only the selection and bracket arithmetic on the recorded spends was checked); D2R and the ROB bank; the data-rich reader check; the R1 readings.

## Files

All under `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/results/pcrl_dual_design_release_v1/roles/verifier/`.

Code (`code/`):
- vlib.py
- v1_integrity.py
- v2_label.py
- v4_decisions.py
- v5_access.py

Records (`records/`):
- V1_INTEGRITY__DDR_VER_INTEGRITY_V1.json
- V1_INTEGRITY__DDR_VER_INTEGRITY_V2.json
- V2_LABEL__DDR_VER_LABEL_V1.json (aggregates only, no person-level arrays)
- V4_DECISIONS__DDR_VER_DECISIONS_V1.json
- V5_ACCESS__DDR_VER_ACCESS_V1.json
- V5_ACCESS__DDR_VER_ACCESS_V2.json
- CHECKS__2026-10-21T191435Z.json (the table above in machine-readable form)
