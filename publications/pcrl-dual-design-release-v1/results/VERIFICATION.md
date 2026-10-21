# Verification (campaign `pcrl_dual_design_release_v1`)

An independent verifier agent checked the campaign's decisive numbers twice with its own code. For recomputation it imported none
of `ddr.evaluate`, `ddr.devreaders`, `ddr.rules` and `ddr.claims`:
- the registered readings, at 18:33-19:14Z: `roles/verifier/VERIFIER_REPORT.md`;
- the post-registration readings, at 19:37-19:44Z: `roles/verifier/VERIFIER_ADDENDUM_REPORT.md`.

Both reports are saved verbatim; the harness refused the agent's own report files. The verifier's code is in `roles/verifier/code/`
and its records in `roles/verifier/records/`. Every numerical job ran through the queue, with 3 logged MODEL_SELECT label reads
under operation `verify`. The verifier used 275 CPU-s (0.076 CPU-h) in all.

## Result

- **Registered readings (first report, checks V1-V5):** every check passes except one procedural FAIL. V2.5: DB_ZTILT had no utility
  record before the grid. This is deviation 5; its late record matches the verifier's recomputation, and no decision uses it.
- **Post-registration readings (addendum, checks A1-A8):** every check passes.
- **Replays.** All decision records (DA_REG, DB_REG, DB_R1, DA_R1), all three robustness (D2R) readings, the strongest-release
  selection and every grid bank replay exactly, with differences of 0 to 3.5e-16:
  - winners' SCORE AUCs;
  - household bootstrap with the registered seeds;
  - paired cells and SEs: 4,368 cells in the registered bank and 10,656 in the later banks.
- **The one approximation.** The verifier's own LR and gradient-boosting readers are an independent approximation, not a replay.
  They agree in sign with the grid for every comparison against the twin and the matched controls. Against NEURAL_RAW and H_LJ the
  sign depends on the reader; D3 and D4 fail under every reader used.

## What was checked

| Area | Checks | Outcome |
|---|---|---|
| Release integrity | All 48 configurations: release, array and model hashes; card committed before each save; the deployment function recomputes every saved MODEL_SELECT array bitwise; the ablation's outputs equal the twin's income and DB_PTC_R1's occupation bitwise | PASS |
| Utility | 12 clauses + 2 guards per version, the clause SEs and the paired costs (11 metrics) for every evaluated configuration; G1 at the point; the two-SE reserve (DB_PTC_R1 `RESERVE_ONE_SEED_ONLY`, version-1 ratio 1.866) | PASS (V2.5 procedural FAIL, deviation 5) |
| Selection and brackets | The MODEL_FIT strength rules and the control bracket points re-derived from the recorded spends; the ZTILT_R1 spend-matching rule (k = 1.5 chosen, overshoot 2.08x / 1.73x) | PASS |
| Reader grids | Wire hashes; registered reader seeds and cache keys; winner selection; exact SCORE AUCs; bootstrap; paired cells | PASS (exact) |
| Decisions | D1-D4 and every D2 conjunct of the four readings; adaptive tags from the card commit times; grid validity (0 seed-dependent equalities) | PASS (exact) |
| Robustness and ablations | The three D2R readings, refit effectiveness (distinct split hashes, effective seed pairs), the four ablations | PASS (exact) |
| Access | 63 label reads (at 19:43:53Z): all after the receipt, inside running units, policy-conforming; fitting reads carry committed cards; no MODEL_SELECT read from a fitting module; the first MODEL_SELECT read followed both registered cards | PASS |
| Deployment | Two packages spot-checked from DEV_REPORT raw rows (bitwise; only probability and decision exported; SEX and extra fields refused) | PASS |

## Discrepancies the verifier raised, and what was done

1. **DB_ZTILT's late utility record** (V2.5): disclosed as deviation 5.
2. **DA's SBC pass is weak evidence**: both DA SBC points leak more than the twin. This is stated beside the DA reading in
   `RESEARCH_DECISION.md`.
3. **The registered DB gain is not separated from ZTILT** (+0.0170, se 0.0121): stated as the card's declared reading.
4. **Code changed during grid runs**, outside the grid lock: deviation 6.
5. **Stale docstring** in `ddr/da/std.py`, which still described the pre-amendment SPEND path: corrected in the code docstring
   (commit 3ceffab5b).
6. **Control constructions differ by version** (DB's SBC target: permuted SEX in version 0, PSEUDO2 in version 1; DA's PSEUDO2
   gamma 2 / 1): reported in the cards' fit records and here.
7. **Beside records mixed cell sets** for their single-view fields: corrected, deviation 12(a).
8. **`CAND_STRENGTH_DIFFERS_BY_VERSION` not recorded for DA_R1** (lambda* 1.0 / 0.75): deviation 12(b).
9. **D2R `release` field null**: deviation 12(c).
10. **Which DB D2R is the family's**: DB_R1_D2R, under RULES.md 8; DB_REG_D2R is an extra reading. Deviation 12(d).
11. **Near-limit controls**: DB_ZTILT_R1 (version-1 men's occupation Brier guard at 0.96 of the cap; reserve 0.86 / 0.09). The
    addendum's discrepancy 5 says "women's" while its own check A2.2 and the record (group `sex0`, men) give the men's guard; the
    claims review found this. And
    DB_SBC_R1_HI (reserve
    1.37 / 1.59) pass G1 but miss the reserve.

## Not checked by the verifier

- The MODEL_FIT computations themselves. Only the selection and bracket arithmetic on the recorded spends was checked; the trigger
  conditions and the fits were not recomputed.
- The data-rich readers.
- The custody readers and the failed custody unit.
- The campaign summary and the R2 trigger records.
- Deployment beyond two spot-checked packages; all ten are covered by `deploy/DEPLOYMENT_CHECK.json` and by the restore check run
  from the backup (`custody/BACKUP_VERIFICATION.json`).
