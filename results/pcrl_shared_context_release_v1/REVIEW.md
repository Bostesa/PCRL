# Shared-context release v1: independent review (phase 2)

- **Reviewer:** independent verifier, fresh context. I did not write any study code.
- **Written:** 2026-09-24, about 20:15Z, after the locked assessment.
- **Machine-readable record:** `INDEPENDENT_VERIFICATION.json`.
- **Scope of what I ran:** my own scripts on the study host (`/opt/pcrl/verify/recompute.py`, `inner_alias.py`), reading the outer `OUTER_CONTRIBUTIONS.npz`, the inner `INNER_AUDIT.json`, unit `PARAMS.npz`/`SELECTED.json`, and `laws.person_law` on inner rows.
- **Study code I did not import:** `assess.py`, the TAC/TDR inference code, and `selection.py`.
- **Aggregates only.** No person rows, ids or labels were copied off the host.

**Labels:** **[R]** = recomputed with my own code; **[S]** = only read from a study report.

## Verdict: CONFIRMED

The headline "no route passed" is supported. The numbers reproduce exactly and the custody chain holds. The interpretation needs the caveats in the "Challenges" section below.

## 1. Chain of custody [R]

- **Lock hash.** `SELECTION_LOCK.json` has sha256 `d9954841…cadbfc` in all three copies: local, host lockcheck and host lock_staging.
- **Lock commit.** Commit `e5d555a` (19:54:21Z) is an ancestor of the origin branch tip `7a6fa58`, and `git show e5d555a:…/SELECTION_LOCK.json` hashes to the same value.
- **Host timeline (mtimes and receipts), strictly ordered:** last work-unit file 19:48:40 → INNER_SELECTION and lock staged 19:53:39–40 → lock commit 19:54:21 → `OUTER_UNLOCK.json` 19:56:26 (`remote_verified` true, matching lock sha and commit) → originals restored 19:56:41–44 → outer score files 19:57:33–58 → assessment 19:58:06.
- **Where outer labels exist.** The only outer-label objects on the host are the three lockcheck `original_2018_restore/anchor_*/prepared.joblib` files. There are none under `/opt/pcrl/work`.
- **No work unit touched outer rows:** all 58 receipts/queue units list only nuisance_train, audit_fit, coefficient_split, inner_selection and inner_check; no argv mentions outer or original objects; the inner audits record `outer_pool_opened: false`; the sanitized anchors' attacker_validation pools carry no labels.
- **Assessment files.** `INFERENCE.json`, `ENDPOINT_TABLE.json` and `FULL_RESULTS.csv` are byte-identical locally and on the host.
- **Caveat.** Mtimes and receipts are evidence controlled by the host, not an external timestamping authority.

## 2. Endpoint recomputation [R]

**Method, all my own code:**
- Per-anchor ratios over all rows of the role: U is Σdiff/n; PWGTP is Σw·diff/Σw.
- The three anchors are averaged with equal weight.
- The bootstrap is 10,000 draws of one multinomial over the union of 2,968 outer households, shared across all anchors, roles and endpoints. I used my own RNG (seed 424242), not the study's.
- z = norm.isf(.05/(2·m)), with m counted from the lock's endpoint lists (40, 4 and 120).

| Family | n | Max \|Δ point\| | Max rel. ΔSE | Decisions agree | Passes (mine / reported) |
|---|---|---|---|---|---|
| primary | 40 | 4.8e-18 | 1.2% | 40/40 | 8 / 8 |
| capability | 4 | 3.1e-17 | 1.1% | 4/4 | 4 / 4 |
| secondary (descriptive) | 120 | 3.1e-17 | 2.1% | 119/120 | 33 / 32 |

- **Exact zeros.** The 8 exact-zero endpoints (zero point estimate and zero SE) are exactly the 8 flagged `exact_zero_same_route`.
- **Thresholds.** No primary or capability decision lies within 10% of its threshold.
- **The one secondary disagreement** is `U_nominee vs J, task, PWGTP`: my upper bound is −0.00308 (pass) and the reported one is −0.00292 (fail), against a threshold of −0.003.
  - It flips with the bootstrap RNG and sits within 3% of the threshold.
  - It is descriptive and not part of any pass.

## 3. Inner selection replay [R]

I rebuilt the §6 contrasts, screens and ranking keys from the three pinned `INNER_AUDIT.json` inner_check scores; their SHAs match the pins in INNER_SELECTION. The replay matches the reported selection exactly:

- **Screens.** None of the four NM units passes either screen, so both slots are **DIAGNOSTIC_ONLY**.
- **U nominee: NM1_U.** It ties NM4_U, which is an exact alias, and wins on id. Its key is (max task +.00163, mean +.00154, max recovery +.00180).
- **P nominee: NM4_P.** Its key is (max AB/SEX −.00935, mean −.01043, max task +.00698).
  - It fails the P screen on task (+.0070 U, +.0031 PWGTP, against a cap of +.001) and on AB/RAC1P recovery (+.0026 U, +.0035 PWGTP, against +.001).
- **ADV representatives.** ADV_B1 for U and ADV_B1_P for P. Every ADV unit is identical to D17, so both win their route's eligible rank-minimum only by id order.
- **Ranking keys.** Mine differ from the reported ones by at most 1.3e-17.
- **Global aliases.** My groups equal `alias_of` exactly.

## 4. Alias checks [R], using `laws.person_law` on 8.4–8.5K inner people per anchor

- **RD_PRIV, ADV_B1, ADV_B2, ADV_B1_P, ADV_B2_P and DET_SEL1 are byte-identical to D17 on all three anchors.**
- **NM1_U, NM4_U and T32_U are identical on all three anchors.**
  - On a0 this shared law is a one-hot **T32 kernel**. It differs from D17 for 71.6% of people, but its within-T32 total variation (TV) is 0.
  - On a1 and a2 it is D17 itself.
- **RD_TASK** equals D17 on a0 and a1, and differs for 0.29% of a2 people.
- **DET_SEL4** equals D17 on a1 and is a within-T32 deterministic selector on a0 and a2.
- **NM4_P (the assessed P nominee) varies within a T32 state only on a0** (mean TV 0.52). On a1 and a2 it is a stochastic T32 kernel (TV 0).
- **Outer rows [S].** `OUTER_AUDIT.outer_alias_confirmed` lists every alias, and there are no breaks on any anchor.

## 5. Capacity spot checks [R]

Source: `round_r*/PARAMS.npz` (eta and the non-D17 A mass), compared with `ENCODER_CAPACITY.json`.

- **U form.** NM1_U and NM4_U have eta = 0 and zero non-D17 A mass in every round (r00–r06) on every anchor.
  - Status is SELECTED_ROUND (round 1) on a0 and WITNESS_FALLBACK on a1 and a2.
  - This matches the report.
- **T32 units.** T32_U and T32_P have eta = 0 in every round, as designed.
- **P form.** NM1_P (all anchors) and NM4_P (a0, a1) have rounds with positive non-D17 mass. **NM4_P a2 has eta = 0 in all seven rounds.**
- **Selected rounds:** NM4_P a0 round 5 (eta .78), a1 round 3 (eta 0), a2 round 0 (eta 0); NM1_P a0 round 5, a2 round 3, a1 WITNESS_SELECTED.
- **Status agreement.** All 18 NM/T32 statuses match the report.
- **The report's `used` flag is round-level.** A unit counts as used if any round used within-T32 variation. So NM4_P a1 shows used = true although the law actually assessed does not vary within T32.
- **Controls:** RD_PRIV, ADV_B1 and ADV_B2 are WITNESS_SELECTED_BY_RULE on every anchor; the ADV_B*_P units are WITNESS_FALLBACK except a1 ADV_B2_P (WITNESS_SELECTED_BY_RULE); RD_TASK selected tau = 0.3, the top of its grid, on every anchor.

## 6. Discrepancies

- **D1 (minor).** PROTOCOL §7 registers 80 primary endpoints and z = norm.isf(.05/160) = 3.4205; the lock merged the exact-D17 comparators into 40 endpoints with z = 3.2272. Under the registered z the decisions are unchanged (8/40), but the write-up should disclose the deviation from the protocol text.
- **D2 (info).** The secondary J endpoint flips with the RNG (section 2).
- **D3 (wording).** RUN_STATUS says "P-form NM used within-T32 variation on 5/6 unit-anchors". That holds at the level of rounds. The assessed NM4_P varies within T32 only on a0.
- **D4 (wording).** RUN_STATUS says the ADV_B*_P units were "WITNESS_FALLBACK at least on a1/a2". In fact a1 ADV_B2_P is WITNESS_SELECTED_BY_RULE. Both outcomes end at D17.
- **D5 (wording).** The D17 alias class is labelled "ADV_B1" in the lock and outer maps, because the canonical name is chosen alphabetically. A reader can mistake D17 scores for ADV scores.

## 7. Challenges to interpretation

1. **The U slot has no non-trivial pass.**
   - All 8 of its passes are `exact_zero_same_route` (A/RAC1P and AB/RAC1P, two weightings, against D17 and against RD_TASK). That leaves 0 of 12 substantive clauses passing.
   - On a1 and a2 the nominee *is* D17. On a0 both releases chose the H-only route H/mlp_120.
2. **The U nominee's effect comes from a0 alone.**
   - The a1 and a2 contributions are exactly 0, so the three-anchor mean divides the a0 effect by 3.
   - Outer task is −0.00015 (U) and −0.00002 (PWGTP), against a target of −0.003.
   - On inner data it was *worse* than D17 (+0.0016).
3. **The P nominee demonstrably breaks its task guard.**
   - The lower bound exceeds +.001 against both D17 and RD_TASK in both weightings. Point estimates are +0.0078 (U) and +0.0076 (PWGTP).
   - Its AB/SEX gain (−0.0040 U, −0.0031 PWGTP) is a trade-off, not dominance.
   - On a1 the PWGTP AB/SEX estimate has the wrong sign (+0.0008), and A/RAC1P and AB/RAC1P recovery rise by +0.0014 to +0.0023.
4. **The strongest adverse secondary comparisons** (point estimates, descriptive): NM4_P vs DET_SEL4 task +0.0080/+0.0072 at equal AB/SEX (−0.0003/+0.0003), so the deterministic same-K selector gets the same recovery far more cheaply; NM4_P vs T32_P task +0.003 for AB/SEX of only −0.001; NM4_P vs Q_HIST task +0.0088/+0.0079.
5. **The comparators are largely vacuous.** RD_PRIV and every ADV unit collapsed to D17, so the primary comparison is really D17 plus a near-D17 RD_TASK. Any claim to "beat RD_PRIV/ADV" would be empty.
6. **The H-capability passes (4/4) are inherited.** They reflect information already in T0/D17, not the new mechanism.
7. **Wording:**
   - Do not call anything a privacy guarantee. The evidence is a finite audit slate, conditional on the fitted and selected objects, on repeatedly used 2018 development data.
   - Do not say NM was "used" for the U nominee: on a0 it is a T32 kernel with eta 0.
   - Do not say NM was "used" for NM4_P on a1 or a2.
   - Report the result as "no route passed; the P nominee trades task for AB/SEX and is dominated on task by DET_SEL4".
