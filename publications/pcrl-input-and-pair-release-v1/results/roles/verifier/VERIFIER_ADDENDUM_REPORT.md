<!-- Saved verbatim by the coordinator from the verifier agent's hand-back (agent a92e2f1798bc736a7, resumed at 04:53:48Z with
roles/coordinator/VERIFIER_ADDENDUM_BRIEF.md), received 2026-10-22 shortly after 05:08Z. The verifier was instructed to return its report
rather than write a file; this file is its returned report, unedited apart from the harness's two-space line indentation, which is
removed. -->

# Verifier addendum: post-registration readings of `pcrl_input_and_pair_release_v1` (checks A1–A8)

I worked from 04:53:54Z to 05:08:14Z (times from `date -u`). **All eight checks pass, and every recorded label reproduces exactly.** The IA_R1 decisions, all three D2R readings, the strongest-release choice and the campaign summary match my own arithmetic. Every replayed value differs from the record by 0.0, except the utility values (at most 3.5e-16) and the reserve ratios (at most 3e-14). I have five observations (O1–O5); none changes a label.

**Results:**
- **IA_R1 (tag ADAPTIVE_R1):** TASK_FEASIBLE / NOT_SEPARATED / NOT_SUPERIOR / NOT_MET / NOT_APPLICABLE_NOT_JOINT / NOT_CLAIMED. Class TASK_ONLY_FEASIBLE_NO_PROTECTION.
- **D2R:** IA_REG, IB_REG and IA_R1 are all ROBUSTNESS_NOT_CONFIRMED, with refits genuinely effective.
- **Strongest releases:** IA_OSG for IA, IB_CSF for IB.
- **Campaign summary:** 3 D2 looks, 0 met.

**How I worked:**
- **New code:** five files in `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/results/pcrl_input_and_pair_release_v1/roles/verifier/code/`:
  - `a_lib.py`, written 04:58:51Z;
  - `a1_integrity.py`, 04:59:56Z;
  - `a3_utility.py`, 05:01:07Z;
  - `a4_grid.py`, 05:02:20Z;
  - `a5_decisions.py`, 05:04:25Z.
- **Code timing:** all five were written before my first unit, and no campaign unit ran from 04:51:47Z to 05:08Z. My units show no launch drift and an empty `code_changed_during_run`. A7 reused the unchanged `v5_access.py`.
- **Imports:** none of `ipr.evaluate`, `ipr.devreaders`, `ipr.rules`, `ipr.claims` or `nrm.rules`.
- **Labels:** read only via `ipr.common.load(..., 'verify', ...)` in a3 and a4, inside queue units.
- **Records:** JSON outputs only, in `.../roles/verifier/records/`: A1_INTEGRITY, A3_UTILITY, A4_GRID, A5_DECISIONS (each with suffix `__IPR_VER_A*_V1`) and V5_ACCESS__IPR_VER_A7_ACCESS_V1.
- **Files:** nothing written outside `roles/verifier/`; nothing committed.
- **Closed roles:** the ATTACK_* and DEV_REPORT income arrays (A1) were compared as raw stored `.npy` bytes by SHA-256, never decoded.

## CHECKS

| # | Check | Expected | Observed | Result |
|---|---|---|---|---|
| A1.1 | 16 new configurations (nos. 24–39, family IA, card IA_OSG_R1): RELEASE.npz = RELEASE.json = ledger; wire hashes (MODEL_FIT, CALIBRATION, MODEL_SELECT); MODEL.pkl = MODEL.json = RELEASE.json | all equal | 16/16 | PASS |
| A1.2 | Card blob SHA at 221a91f21 = recorded; card commit before every save and before the first fitting read with the card | ordered | Card committed 04:11:44Z; first fitting read 04:11:54Z; saves 04:15:03–04:20:52Z | PASS |
| A1.3 | Deployment (`ipr.ia/ib/ctl.predict`) recomputes the MODEL_SELECT arrays | bitwise | **39/39** configurations, so the 23 registered ones also deploy unchanged under the current code | PASS |
| A1.4 | Income output of every IA_OSG* configuration = CTL_TWIN's, in every role | bitwise | 16/16 × 6 roles | PASS |
| A1.5 | Payloads | as the card states | CAND, SBC and BLIND are `osg`; SPEND and RO points are `dq` / `ro` with `outputs = occupation`. Strengths: SPEND 0.313843/0.37975 (V0), 0.555992/0.740025 (V1); RO 0.336375/0.492487 (V0), 0.595909/0.793155 (V1). Budget 39 ≤ 48; no new exact alias | PASS |
| A2.1 | k* under S3 with the scan rule | recorded | **V0 k* = 2**: rung 3 fails the women's occupation Brier guard (1.13×). **V1 k* = 4**: rung 5 fails the men's guard 1.43×, women's 1.22×, occupation NLL 1.15×. Income components exactly 0 at every rung | PASS |
| A2.2 | Ladders re-derived from the recorded single-step statistics with my own greedy rule | recorded | OSG, OSG-BLIND, and the registered SFG and BLIND ladders all equal. Rung steps equal my composition. CAND steps: V0 {MAR1}; V1 {AGE10, WK3, MAR1} | PASS |
| A2.3 | SBC / SPEND / RO bracket points (PROTOCOL 4 rule) | recorded | **SBC:** rungs 3/4 (V0), 2/3 (V1). **SPEND:** 0.313843/0.37975, 0.555992/0.740025. **RO:** 0.336375/0.492487, 0.595909/0.793155. No CONTROL_PATH_MAY_NOT_BRACKET flag | PASS |
| A2.4 | BLIND rung | recorded | k = 6 in both versions: {AGE10, WK5, COW5, MAR1} | PASS |
| A2.5 | Consistency of the recomputed single-step statistics | EQUAL | Own max \|Δ\| over 17 steps against `FIT_V<v>.json`: dT_occ 0.0, dI 0.0 in both versions | PASS |
| A2.6 | Fineness classes | match the D2 flags | RO FINE/FINE; SBC COARSE_BOTH/COARSE_BELOW; SPEND COARSE_BELOW/FINE. This gives CONTROL_GRID_COARSE_SBC and CONTROL_GRID_COARSE_SPEND, as recorded | PASS |
| A3.1 | 24 clauses + 4 guards for 8 new members plus CTL_TWIN, IA_SFG, IB_CSF, against EVAL_MS | \|Δ\| ≤ 1e-9 | 308 values: max \|Δ value\| 3.5e-16, \|Δ SE\| 0; pass/fail agree | PASS |
| A3.2 | Paired costs (130 records, 11 metrics, W skipped) | ≤ 1e-9 | max \|Δ\| = 0 | PASS |
| A3.3 | G1 at the point | TWIN and CAND pass | TWIN and IA_OSG pass in both versions (IA_OSG V1 highest value/cap 0.527). **IA_OSG_RO_HI fails in V1** (O1) | PASS |
| A3.4 | Two-SE reserve: own OHN1 arithmetic with the H_LJ floor | EVAL flags and D1 statuses | Flags agree everywhere; ratios within 3e-14. IA_OSG: RESERVE_ONE_SEED_ONLY (V1 men's-guard ratio 1.394 < 2). TWIN: RESERVE_MET. Matches D1 | PASS |
| A4.1 | IPR1_R1 exact replay (11 members × 8 cells × 3 views = 264) | identical | Wires match; winners are the first maximum of VALIDATION AUC; SCORE AUC, 400 bootstrap draws and NOLOOKUP AUC all max \|Δ\| 0.0; 240 paired cells and SEs exact | PASS (exact) |
| A4.2 | IPR1_ROB_R1 (11 × 32 × 3 = 1056) and IPR1_ROB (13 × 32 × 3 = 1248) | identical | Same results; 960 + 1344 paired cells and SEs exact; no missing cache entry or bootstrap key | PASS (exact) |
| A4.3 | Seed effectiveness | no unexplained equality | Seed-dependent recipe fingerprints: IPR1_R1 minimum set 576 checks (k0/k1); robustness minimum sets 1440 (IA_REG), 864 (IB_REG), 1152 (IA_R1) checks (k0/k1 and k2/k3). 0 equal in every case. Seeds and bootstrap seeds follow the registered formula; none forbidden | PASS |
| A5.1 | IA_R1 D1–D6 and class | recorded | All equal (labels listed above) | PASS |
| A5.2 | Adaptive tag | ADAPTIVE_R1 | Card first committed 04:11:44Z, after the first MODEL_SELECT read (02:50:32Z), revision 1, so ADAPTIVE_R1. Every label is prefixed `ADAPTIVE_R1__`; `registered` = false | PASS |
| A5.3 | Every D2 conjunct | recorded | Points, pass/fail, benefit means and SEs, primary costs, secondary metrics, gates and targeting all identical (detail below) | PASS |
| A5.4 | Grid validity | effective | Own result effective; recorded READER_SEEDS_EFFECTIVE | PASS |
| A5.5 | Flags | recorded | Derived: CAND_STRENGTH_DIFFERS_BY_VERSION, CONTROL_GRID_COARSE_SBC, CONTROL_GRID_COARSE_SPEND, NO_BRACKET_SPEND, POINT_FEASIBLE_NOT_RESERVE_QUALIFIED, RO_GENERIC_ZTILT. These equal the recorded class flags minus READER_SENSITIVE_ENHANCED_IA_OSG_RO_HI, a reported-only flag from the enhanced winners, which I did not re-derive | PASS |
| A6.1 | Refit effectiveness, all three D2R readings | REFIT_GENUINELY_EFFECTIVE | Own R1–R4 split-assignment hashes are distinct. R3/R4 SCORE household overlaps with other repetitions are 0.295–0.319, inside [0, 1). Eight reader seeds and bootstrap seeds match the formula. No fingerprint equality | PASS |
| A6.2 | Per-comparator robustness labels (24 robustness cells and all 32 cells, merged) | recorded | Labels, robustness means, se_agg, n_eff, reader-seed qualifiers and all-cell labels all match (within 1e-12). Registered cells in the robustness banks are identical to the main banks (max \|Δ\| 0) | PASS |
| A6.3 | Strongest release | recorded | Registered-cell benefit vs TWIN: IA_SFG 0.0016147, IA_OSG 0.0017078, so IA_OSG is selected; IB_CSF 0.0069293. \|Δx\| = 0 | PASS |
| A7 | Access: 47 label reads up to 05:06Z | all conforming | 6 fit\|MODEL_FIT (4 registered fits; 2 R1 fits at 04:11:54/55Z, after the card commit at 04:11:44Z), 23 benchmark, 13 readers (`ipr.devreaders` 8, `ipr.richreaders` 2, `ipr.readercustody` 3), 5 verify. All after the receipt, inside running units, from policy entry points. No closed-role read; no MODEL_SELECT read with a fitting module on the call chain | PASS |
| A8 | Code drift: 46 units ended since 03:11Z | no unexplained change | `code_changed_during_run` is non-empty only for IPR_GRID_MAIN_V0 (my six first-report files) and IPR_GRID_ROB2_V0/V1 (`v6_seeds.py`). One launch drift: IPR_DEPLOY_STORE_CTL_TWIN_V0_V1 (`v6_seeds.py`). All are deviation 6 | PASS |

**A8, what changed in `ipr/`:** only between units, from 04:08:01Z to 04:11:54Z: the R1 commit 221a91f21 (11 files).
- `osg.py` is new.
- `sfg.py` gained a 3-line `osg` dispatch.
- `paths.py` gained the occupation-only restriction.
- `plan.py`, `report.py`, `richreaders.py` and `custody.py` changed.
- `common.py`, `admit.py` and `closeout.py` changed in comments only.

A1.3 shows these changes left every saved release reproducible bitwise.

**A5.3, IA_R1 D2 detail:**
- **(b) vs TWIN:** pair benefit +0.00171 ± 0.00616 (version means 0.0008 / 0.0026), point only. Income view ALIAS_ZERO, occupation RESOLVED, so the gate passes; the benefit is not resolved.
- **Pooled MODEL_SELECT total-NLL cost of CAND minus each point:** TWIN +0.008006; SBC_LO +0.002797, SBC_HI −0.006079; SPD_LO +0.004349, SPD_HI +0.001850; RO_LO +0.003047, RO_HI −0.000838.
- **Brackets:** Q* = SBC_HI and RO_HI. SPEND has no bracket (NO_BRACKET_SPEND).
- **SBC vs Q*:** +0.00195 ± 0.00578 (point only); gate passes.
- **RO vs Q*:** −0.00722 ± 0.00643 (not positive). Occupation gate fails at the point (POINT_FAILED); occupation|women_brier is resolved higher.
- **Domination:** none of the cheaper points has a resolved negative benefit.
- **(d) targeting**, version means: TWIN 0.0008 / 0.0026; SBC_HI 0.0031 / 0.0008; RO_HI −0.0046 / −0.0098.
- **D3:** fails all nine references:

  | Reference | Benefit |
  |---|---|
  | NEURAL_RAW | −0.0210 ± 0.0087 |
  | H_LJ | −0.0277 ± 0.0101 |
  | B2TWIN | −0.0402 |
  | RETIRED_B3CTL | −0.0500 |
  | SUCC | −0.0389 |
  | TWJ | −0.0105 |
  | W | −0.0004 |
  | TR100 | +0.0016 (point only) |
  | TT100 | −0.0009 |

**A6.2, robustness-cell benefits** (mean ± se_agg):

| Reading | TWIN | SBC Q* | SPEND Q* | RO Q* | NEURAL_RAW | H_LJ |
|---|---|---|---|---|---|---|
| IA_REG (IA_SFG) | +0.00183 ± 0.0055 | +0.00317 ± 0.00608 | +0.00181 ± 0.00443 | +0.00107 ± 0.00581 | −0.0185 | −0.0270 |
| IB_REG (IB_CSF) | +0.00613 ± 0.00558 | — | −0.00063 ± 0.00687 | — | −0.0142 | −0.0227 |
| IA_R1 (IA_OSG) | +0.00279 ± 0.00587 | −0.00101 ± 0.00806 | no bracket | −0.00567 ± 0.00658 | −0.0175 | −0.0261 |

## Observations (none changes a label)

- **O1. The RO path's bracket point fails G1.** IA_OSG_RO_HI is task-infeasible at the point in V1: men's occupation guard 0.00502 against 0.005 (value/cap 1.0046), reserve ratio −0.012. The rules gate only CAND and TWIN on G1, and RO_FRONTIER fails anyway. IA_OSG_SBC_HI misses the reserve in both versions (men's guard ratios 1.59 / 0.94). Both are descriptive.
- **O2. The occupation-only SPEND path does not bracket on MODEL_SELECT.** Its HI point bracketed CAND on MODEL_FIT (V0: 0.00407 ≥ 0.00236 + 2 SE). On MODEL_SELECT both points spend less than IA_OSG, so SPEND_FRONTIER fails as NO_BRACKET. Under the rules that is a limit of the path, not evidence of equality.
- **O3. The R1 SBC path is coarse.** V0 lower point spends 0.00005 against the 0.75 s threshold of 0.00177; V0 upper point 0.0121 against the 0.00585 ceiling. The resulting COARSE flags are reported only, never gating.
- **O4. The code drift is fully disclosed.** The only within-unit drift is the verifier's own files (deviation 6). The single `ipr/` change between units is the R1 commit described under A8.
- **O5. The superseded early D2R records were not checked.** They sit in `claims/superseded/` (deviation 9). I verified only the reruns, which are the readings in use.

## Exact replays vs approximations

- **Exact replays:**
  - Release, model and card hashes; deployment of all 39 configurations, bitwise.
  - Income equality across all 6 roles.
  - 308 utility values with SEs and the reserve; 130 paired-cost records.
  - 2,568 replayed cells: winner checks, SCORE AUC, 400 bootstrap draws each, NOLOOKUP AUC.
  - 2,544 paired cells and SEs.
  - Aggregates, Q*, gates, targeting, every D label, the D2R per-comparator statistics, and the strongest-release values.
- **Recorded, not replayed:**
  - The grid entries of the 7 non-target references in D3 (arithmetic only).
  - The enhanced (TABLE_CV) winners behind the READER_SENSITIVE_ENHANCED flag.
  - The reader-budget flags (all COMPARABLE), the beside records, the data-rich readers and the custody records.
- **Independent approximations:** none in this addendum.

## CPU

- **Queue units, 47.35 CPU-s in total:**

  | Unit | CPU-s |
  |---|---|
  | IPR_VER_A1_INTEGRITY_V1 | 12.135 |
  | IPR_VER_A3_UTILITY_V1 | 0.690 |
  | IPR_VER_A4_GRID_V1 | 34.131 |
  | IPR_VER_A5_DECISIONS_V1 | 0.305 |
  | IPR_VER_A7_ACCESS_V1 | 0.093 |

- **Out of queue:** no more than about 6 CPU-s (compiling, record inspection, the wait script). Not charged by me.
- **Verifier total so far:** about 0.03 CPU-hours (first report about 49 CPU-s; this addendum about 53 CPU-s).
