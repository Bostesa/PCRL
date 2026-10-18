# Protection investigator report (privacy stage; saved by the coordinator)

**Disclosure.** The protection investigator (a separate agent) returned this report as the text of its final hand-back at 07:45Z;
the coordinator saved it here unchanged in substance. Design, revisions, registered predictions and the closing summary are in
`CARD.md` (committed by the coordinator before each revision's fits except as recorded in PROTOCOL_DEVIATIONS items 10 and 14).

This is exploratory development on previously used records (public-use ACS 2019 Florida PUMS, development roles). The MODEL_SELECT
results below are adaptive development evidence, not confirmation. All releases are H3-anchored: H3's income is frozen and only the
occupation output changes.

## 0. Bottom line

**Protection beyond controls holds.** A correction to the occupation output, trained against SEX critics on the linked pair (PAIR),
lowers pair recovery beyond two controls on both witnesses (B/B2_SINGLE, A/A1_PO_16) and both release seeds: beyond its matched
task-only continuation twin (CONT), and beyond a label-free compression code (COMP) at matched out-of-fold budget utilization. At B80,
all 8 grid cells are positive against both controls on both witnesses. Against CONT it is STABLE_POSITIVE at the largest saved scale
on each witness.

**Superiority over SUCC does not hold.** On seed 0 it reaches or beats SUCC (the inherited JOINT_TREE occupation beside H3) at better
occupation utility: P2_PAIR_B80 pair 0.5972 vs SUCC 0.5997, occupation NLL vs U0 −0.0025 vs +0.0025, men's guard +0.0023 vs +0.0044;
P2A_PAIR_B80 pair 0.5813, below all six required comparators in every seed-0 cell. But these points miss the seed-0 men's 2-SE
reserve, and on seed 1 the projection chose smaller scales. Over 8 cells the best points are SIGN_CHANGES against SUCC and
NOT_POSITIVE against RETIRED_B3CTL.

**The reserve-meeting points stay above SUCC.** Points that meet the whole-release reserve on both seeds (P2_PAIR_B50, P2A_PAIR_B50,
P2_LOC_B80) read 0.604–0.625 and are NOT_POSITIVE against SUCC.

**No configuration can be nominated.** Every PROT configuration fails RECIPIENT_POINT: H3's income view is +0.0309 vs H_LJ in every
case; the smallest occupation-view maximum is +0.0187 (P2_PAIR_M90), above the 0.010 gate. The best nomination score (minimum over
the six required comparators of the 8-cell mean pair benefit) is −0.0027 (P2A_PAIR_B80) and −0.0033 (P2_PAIR_M90), both bound by
RETIRED_B3CTL.

## 1. Design

**Operation.** The released occupation vector is `softmax(log p_W(x) + s·Δθ(x))`, with argmax as the decision: p_W is the witness's
own deployed `predict`; Δθ is a leaf-score correction on the frozen T_TREE occupation routing (800 trees × 16 leaves × 5; θ = 0 gives
the witness); s is the end-projection scale. Runtime input is X only.

**Arms.** CONT: CE only; critics updated but unused (λ = 0). LOC: + λ·risk(occupation wire). PAIR: + λ·[risk(occupation wire) +
risk(pair wire with H3's frozen income wire)] — a coalition-aware occupation correction; claim J (two-output joint design) stays
NOT_CLAIMABLE. COMP: label-free K-means code of the witness's cross-fitted MODEL_FIT vectors; K is the smallest ladder value whose
out-of-fold budget utilization is ≤ the matched PAIR configuration's. A temperature or shrinkage control was not used: both are
bijections and cannot lower an ideal reader's recovery.

**Training base.** The witness's household-cross-fitted MODEL_FIT output (CROSSFIT.npz); the deployed correction is applied to the
witness's deployed output.

**Critics and strength.** The leafcorr critic bank (linear, MLP32, MLP64 per view; 80 warm rounds; 5 rounds per update).
λ = ρ·median‖∇task‖ / median‖∇P‖ at the start, ρ = 1 (JOINT_TREE's strength). 20 epochs × 40 × 256, AdamW 0.01, weight decay 0.01.
One strength only, no sweep.

**Cross-fitting and projection.** The whole pipeline is fitted 6 times (5 OHN1_crossfit folds plus the full data). The deployed full
correction is scaled by the largest s whose out-of-fold MODEL_FIT costs meet the budget (men's and women's summed-Brier increase,
aggregate Brier, NLL and accuracy loss over the witness). Budget = f × the witness's MODEL_SELECT reserve slack (B50, B80) or 0.9 × its
point margin (M90, a frontier-only point). This is a MODEL_SELECT-guided design choice, recorded as `budget_provenance` in every
record.

**Guard-aware element.** P1: an equal-utilization per-SEX Brier Lagrangian in training. P2: none in training. P3: static per-SEX CE
weights 1/S_g, from the same logged slacks. The per-SEX constraints of the projection apply in all three.

## 2. Code and checks

`ohn/protect.py`: `selftest`, `dryrun`, `fit`, `predict`, `verify`, `parity`, `gridspec`. Self-test OHN_PROT_SELFTEST_V3–V6 pass (V6:
31 invented-data checks; V1 failed on a dictionary key, V2 on two over-strict check designs). Label-free dry run (OHN_PROT_DRYRUN_V1;
real features, invented labels and SEX; 131.7 CPU-s): about 25 CPU-s per training pipeline. Witness parity: `predict` under the .venv
interpreter is bitwise equal to the saved release on all six roles for A1_PO_16 (both seeds), B2_SINGLE (both seeds) and B2_STACK
seed 0. Every saved configuration was verified bitwise on all six roles inside its own fit; no later verify was run.

## 3. Sequence, revisions and registered predictions

**P1 — initial design, B2_SINGLE seed 0.** Every P1 configuration met the seed-0 reserve. PAIR beat its twin by +0.014 to +0.016 on
the grid but stayed about 0.02 above SUCC. Two defects showed in the training records: the Lagrangian oscillated (μ jumped 0↔5 epoch to
epoch in PAIR, LOC and even CONT, and the fold pipelines ended in opposite states), and the projection did not transfer (the
fold-average label-free change was 1.9× the deployed one). The card's claim that CONT keeps the guard inactive was false on real data.

**Revision 1 → P2 (06:47Z): the Lagrangian is removed.**

| Prediction | Outcome |
|---|---|
| (i) out-of-fold / MODEL_SELECT label-free change within 0.67–1.5 | CONFIRMED (0.94–0.96) |
| (ii) MODEL_SELECT men's increase within one floor SE of the projection | REFUTED on seed 0 (+0.0016 to +0.0029 on both witnesses; COMP showed the same excess); on seed 1 the error reversed sign (−0.0008 to −0.0012): a per-seed transfer error of about ±0.002 rather than a bias |
| (iii) P2_PAIR_B80 vs P2_CONT > +0.010 and vs COMP > +0.003 on seed 0 | CONFIRMED (+0.0378 / +0.0274) |
| (iv) the men's guard binds the projection | CONFIRMED |

**Revision 2 → P3 (07:15Z, amended 07:20Z, committed before any fit): static weights, both witnesses, seed 0.**

| Prediction | Outcome |
|---|---|
| (i) men's MODEL_SELECT increase lower by ≥ 0.0015 than P2's, with the reserve met | CONFIRMED on both witnesses (men improve: −0.0024 / −0.0025 vs P2's +0.0064 / +0.0071) |
| (ii) pair within 0.005 of P2_PAIR_B80 | REFUTED on both (+0.018 / +0.037) |
| (iii) women's guard at reserve | CONFIRMED |
| (iv) PAIR beats P2 CONT by > 0.020 | REFUTED on B2_SINGLE (+0.0195), CONFIRMED on A1_PO_16 (+0.0215) |

As registered, P3 got no seed 1 and no COMP twin.

**What each step taught.** P1: the oscillating guard protected the men's margin (P1_PAIR_B80 men's guard −0.0019, ratio 3.83) but
weakened the critic. P2: the critic is strong and its aggregate cost transfers, but the men's guard binds. P3: a fixed 2:1 weight flips
the binding constraint to women, at half the scale; the cost balance between SEX groups is controllable. The reserve-feasible optimum
lies between P2 and P3 and was not located: the two allowed revisions were used.

**Last step (not a revision):** the predeclared LOC arm on seed 1 for both witnesses.

## 4. Every configuration (MODEL_SELECT, H3 income)

Values vs U0; ratio = margin / se_used (frozen reserve rule). "Reserve" is the whole-release status over both seeds. Pair is the
grid's mean SCORE AUC per seed.

| Config | Seed | Scale or K | Men's guard (ratio) | Women's | Occ NLL | Occ Brier | Occ acc | Pair | Reserve (min ratio) |
|---|---|---|---|---|---|---|---|---|---|
| W B2_SINGLE | 0 | – | −0.00407 (5.00) | −0.00908 | −0.01469 | −0.00654 | −0.00875 | 0.6426 | MET (2.86) |
| W B2_SINGLE | 1 | – | −0.00250 (4.42) | −0.00818 | −0.01136 | −0.00530 | −0.00875 | 0.6392 | |
| P1_CONT | 0 | 1.0 | −0.00488 (5.45) | −0.00915 | −0.01592 | −0.00699 | −0.00837 | 0.6327 | seed-0 met |
| P1_LOC_B80 | 0 | 1.0 | −0.00012 (2.80) | −0.00959 | −0.00940 | −0.00480 | −0.00562 | 0.6240 | seed-0 met |
| P1_PAIR_B50 | 0 | 0.609 | −0.00280 (4.30) | −0.00851 | −0.01185 | −0.00562 | −0.00775 | 0.6233 | seed-0 met |
| P1_PAIR_B80 | 0 | 0.861 | −0.00194 (3.83) | −0.00773 | −0.00937 | −0.00480 | −0.00637 | 0.6190 | seed-0 met |
| P1_PAIR_M90 | 0 | 1.0 | −0.00139 (3.51) | −0.00716 | −0.00766 | −0.00424 | −0.00600 | 0.6170 | seed-0 met |
| P2_CONT | 0 | 1.0 | −0.00507 (5.55) | −0.00876 | −0.01576 | −0.00690 | −0.00912 | 0.6351 | MET |
| P2_CONT | 1 | 1.0 | −0.00285 (4.62) | −0.00922 | −0.01320 | −0.00600 | −0.01062 | 0.6382 | |
| P2_LOC_B80 | 0 | 0.952 | +0.00075 (2.20) | −0.00525 | −0.00260 | −0.00222 | −0.00375 | 0.6142 | MET (2.20) |
| P2_LOC_B80 | 1 | 0.809 | +0.00012 (2.76) | −0.00618 | −0.00438 | −0.00300 | −0.00525 | 0.6076 | |
| P2_PAIR_B50 | 0 | 0.638 | +0.00018 (2.62) | −0.00831 | −0.00743 | −0.00401 | −0.00675 | 0.6076 | MET (2.62) |
| P2_PAIR_B50 | 1 | 0.529 | −0.00123 (3.67) | −0.00697 | −0.00766 | −0.00407 | −0.00762 | 0.6248 | |
| P2_PAIR_B80 | 0 | 0.856 | +0.00230 (1.39) | −0.00705 | −0.00251 | −0.00232 | −0.00575 | 0.5972 | ONE_SEED_ONLY |
| P2_PAIR_B80 | 1 | 0.717 | −0.00039 (3.10) | −0.00596 | −0.00495 | −0.00314 | −0.00662 | 0.6123 | |
| P2_PAIR_M90 | 0 | 1.0 | +0.00385 (0.57) | −0.00592 | +0.00142 | −0.00098 | −0.00350 | 0.5919 | ONE_SEED_ONLY |
| P2_PAIR_M90 | 1 | 1.0 | +0.00121 (2.03) | −0.00380 | +0.00050 | −0.00127 | −0.00537 | 0.6025 | |
| P2_COMP_B80 | 0 | K32 | +0.00077 (2.30) | −0.00875 | −0.00443 | −0.00393 | −0.00737 | 0.6246 | ONE_SEED_ONLY |
| P2_COMP_B80 | 1 | K32 | +0.00158 (1.92) | −0.00763 | −0.00182 | −0.00297 | −0.01075 | 0.6208 | |
| P2_COMP_M90 | 0 | K20 | +0.00327 (0.90) | −0.00597 | +0.00313 | −0.00130 | −0.00262 | 0.6233 | NOT_MET |
| W A1_PO_16 | 0 | – | −0.00385 (4.88) | −0.00968 | −0.01614 | −0.00673 | −0.00987 | 0.6361 | MET (2.86) |
| W A1_PO_16 | 1 | – | −0.00229 (4.30) | −0.00870 | −0.01280 | −0.00546 | −0.00975 | 0.6391 | |
| P2A_CONT | 0 | 1.0 | −0.00434 (5.15) | −0.00932 | −0.01655 | −0.00680 | −0.01187 | 0.6395 | MET |
| P2A_CONT | 1 | 1.0 | −0.00234 (4.33) | −0.00972 | −0.01426 | −0.00599 | −0.00925 | 0.6332 | |
| P2A_LOC_B80 | 0 | 0.788 | +0.00232 (1.41) | −0.00783 | −0.00503 | −0.00270 | −0.00137 | 0.6108 | ONE_SEED_ONLY |
| P2A_LOC_B80 | 1 | 0.659 | +0.00104 (2.26) | −0.00828 | −0.00737 | −0.00357 | −0.00500 | 0.6236 | |
| P2A_PAIR_B50 | 0 | 0.699 | +0.00092 (2.15) | −0.00695 | −0.00616 | −0.00297 | −0.00462 | 0.6043 | MET (2.15) |
| P2A_PAIR_B50 | 1 | 0.546 | −0.00094 (3.43) | −0.00665 | −0.00802 | −0.00377 | −0.00862 | 0.6194 | |
| P2A_PAIR_B80 | 0 | 0.921 | +0.00329 (0.85) | −0.00476 | +0.00002 | −0.00069 | −0.00125 | 0.5813 | ONE_SEED_ONLY |
| P2A_PAIR_B80 | 1 | 0.719 | −0.00008 (2.82) | −0.00536 | −0.00504 | −0.00269 | −0.00762 | 0.6118 | |
| P2A_PAIR_M90 | 0 | 1.0 | +0.00421 (0.38) | −0.00381 | +0.00256 | +0.00025 | −0.00112 | 0.5766 | NOT_MET |
| P2A_COMP_B80 | 0 | K24 | +0.00053 (2.34) | −0.00600 | −0.00122 | −0.00270 | −0.00787 | 0.6240 | ONE_SEED_ONLY |
| P2A_COMP_B80 | 1 | K32 | +0.00172 (1.80) | −0.00537 | −0.00028 | −0.00179 | −0.00725 | 0.6215 | |
| P3_PAIR_B50 | 0 | 0.318 | −0.00587 (5.99) | −0.00523 | −0.01213 | −0.00555 | −0.01125 | 0.6304 | seed-0 met |
| P3_PAIR_B80 | 0 | 0.469 | −0.00642 (6.30) | −0.00278 | −0.00971 | −0.00462 | −0.01050 | 0.6156 | seed-0 met |
| P3A_PAIR_B50 | 0 | 0.289 | −0.00575 (5.93) | −0.00529 | −0.01313 | −0.00552 | −0.00912 | 0.6274 | seed-0 met |
| P3A_PAIR_B80 | 0 | 0.430 | −0.00635 (6.25) | −0.00244 | −0.01035 | −0.00442 | −0.00650 | 0.6180 | seed-0 met |

Own SEs of the men's guard are 0.0016–0.0021. The income conditions are H3's (min ratio 3.51 seed 0, 2.86 seed 1). Out-of-fold cost
differences over the witness carry household-clustered SEs of 0.0002–0.0007 (Brier terms), 0.0003–0.0013 (NLL), 0.0010–0.0017
(accuracy). The dominant uncertainty is the ±0.002 per-seed transfer of the men's cost to MODEL_SELECT. Paired MODEL_SELECT SEs of
differences between two configurations are not available from the scorer (it compares only against U0, Ucal and the frozen task-only
references).

## 5. Reader grid (two repetitions × two reader seeds per release seed; SCORE AUC; benefit = comparator − candidate)

References (seed 0 / seed 1): SUCC 0.5997 / 0.6020; RETIRED_B3CTL 0.5964 / 0.5914; B2TWIN 0.6096 / 0.6040; NEURAL_RAW 0.6191 /
0.6144; H_LJ 0.6217 / 0.6149; TWJ 0.6337 / 0.6277; H3+T_TREE 0.6363 / 0.6360.

| Candidate | vs CONT | vs COMP_B80 | vs SUCC | vs RETIRED_B3CTL |
|---|---|---|---|---|
| P2_PAIR_B80 | POSITIVE_NOT_STABLE +0.0319, all cells > 0 | POSITIVE_NOT_STABLE +0.0179, all > 0 | NOT_POSITIVE −0.0040 (seed 0 +0.0024 / seed 1 −0.0103) | NOT_POSITIVE |
| P2_PAIR_M90 | STABLE_POSITIVE +0.0395 | POSITIVE_NOT_STABLE +0.0255 | SIGN_CHANGES +0.0036 (+0.0077 / −0.0005) | NOT_POSITIVE −0.0033 |
| P2A_PAIR_B80 | STABLE_POSITIVE +0.0398 (+0.058 / +0.021) | POSITIVE_NOT_STABLE +0.0262 | SIGN_CHANGES +0.0042 (+0.018 / −0.010) | NOT_POSITIVE −0.0027 |
| P2_PAIR_B50 | SIGN_CHANGES +0.0205 | SIGN_CHANGES +0.0065 | NOT_POSITIVE −0.0154 | |
| P2A_PAIR_B50 | POSITIVE_NOT_STABLE +0.0245 | SIGN_CHANGES +0.0109 | NOT_POSITIVE −0.0110 | |
| P2_LOC_B80 | POSITIVE_NOT_STABLE +0.0258 | POSITIVE_NOT_STABLE +0.0118 | NOT_POSITIVE −0.0101 | |

P2_PAIR_M90 is also STABLE_POSITIVE vs TWJ and POSITIVE_NOT_STABLE vs H_LJ and NEURAL_RAW. The COMP codes are POSITIVE_NOT_STABLE vs
their witness (+0.018 / +0.015). CONT is SIGN_CHANGES vs its witness (+0.004 / +0.001).

PAIR vs LOC at B80: PAIR reads lower in all 8 cells on A1_PO_16 (by 0.008–0.036). On B2_SINGLE it reads lower in 6 of 8: all of seed
0, mixed on seed 1, where LOC's scale of 0.809 exceeded PAIR's 0.717.

Recipient points: the income view is +0.0309 for all, so every candidate fails. Occupation-view maximum vs the JOINT comparators:
P2_PAIR_M90 +0.0187; P2A_PAIR_B80 +0.0327 (seed 0 alone +0.0074).

P1 and P3 are seed 0 only. P1_PAIR_B80 / M90 vs P1_CONT: +0.0137 / +0.0157. P3_PAIR_B80 and P3A_PAIR_B80 vs P2 CONT: +0.0195 / +0.0215.

## 6. Frontier (pair recovery vs MODEL_SELECT occupation cost)

B2_SINGLE, seed 0: W 0.6426 (NLL −0.0147, men −0.0041) → CONT 0.6351 → COMP_B80 0.6246 (−0.0044, +0.0008) → LOC_B80 0.6142 → P2 B50
0.6076 (−0.0074, +0.0002) → P2 B80 0.5972 (−0.0025, +0.0023) → P2 M90 0.5919 (+0.0014, +0.0039); SUCC (reference) 0.5997 (+0.0025,
+0.0044; ratio 0.32). A1_PO_16, seed 0: P2A B50 0.6043 → B80 0.5813 (NLL +0.0000, men +0.0033) → M90 0.5766. Seed 1: P2_PAIR_M90 reads
0.6025 at NLL +0.0005 and men's guard +0.0012, meeting the seed-1 reserve (ratio 2.03), against SUCC 0.6020 at NLL +0.0029 and men's
guard +0.0039 (ratio 0.62).

On both seeds the stronger witness plus the pair critic reaches SUCC-level recovery at lower occupation and men's cost. A single
configuration does this on both seeds only outside the seed-0 reserve. P1 and P3 sit at 0.616–0.630 with large margins on the men's
guard.

## 7. What distinguishes protection from continuation and compression

- CONT barely changes recovery: with the identical pipeline and λ = 0, it moves pair recovery by only +0.004 / +0.001 vs the witness
  (SIGN_CHANGES). It slightly improves MODEL_SELECT NLL, while out of fold it overfits (+0.004 to +0.005 NLL). Its training critics
  find more SEX evidence by the end of training (final occupation risk 0.058–0.067, against 0.020–0.024 under PAIR).
- COMP reduces recovery only modestly at real cost: at matched out-of-fold utilization it reduces recovery by about 0.015–0.018 at a
  large MODEL_SELECT cost (B2: men +0.0048 / NLL +0.0103 seed 0, men +0.0041 / NLL +0.0095 seed 1), and the codes fail the seed-1
  reserve.
- PAIR at B80 beats both controls in all 8 cells on both witnesses. Its cost against COMP is higher on seed 0 and lower on seed 1. On
  seed 0, P2_PAIR_B80 dominates P2_COMP_M90 outright: cheaper (men +0.0064 vs +0.0073; NLL +0.0122 vs +0.0178) and 0.026 lower.
- Internal held-out MODEL_FIT readers agree: the LR reader's pair AUC moves from W 0.637 to PAIR B80 0.600, CONT 0.636, COMP 0.632 (B2
  seed 0).

## 8. Training records

| Arm | λ | Training-critic risk, occupation / pair (epoch 1 → final) |
|---|---|---|
| P2 PAIR B2 s0 | 0.89 | 0.038 / 0.048 → 0.022 / 0.040 |
| P2 PAIR B2 s1 | 0.74 | |
| P2A PAIR s0 | 0.95 | |
| P2A PAIR s1 | 0.78 | |
| LOC | 1.20–1.59 | |
| P3 | 0.98 / 1.06 | |
| P1 PAIR | 0.89 | 0.038 / 0.048 → 0.037 / 0.060 |

In-sample full-correction cost at the chosen scale (P2_PAIR_B80 s0): men +0.0038, NLL +0.0066. All curves, multipliers, out-of-fold
SEs and label-free transfer statistics are in each OCC.json.

## 9. Failures, deviations, refusals

- Failed units: OHN_PROT_SELFTEST_V1 and V2 (code and check fixes). OHN_PROT_P1_PAIR_S0_V1 and OHN_PROT_P1_CONT_S0_V1 were stopped
  after about 50 CPU-s, nothing saved, so the witness id could be recorded as a string; relaunched as _V2. Ledger rows 201–205 record
  these two stops and three decisions.
- Deviation 10: the first P2 unit started 51 s before the coordinator's commit; the card text preceded it.
- Deviation 14: protect.py was edited while four seed-1 P2 units ran; the `none` and COMP paths are unchanged.
- Record artifacts: COMP records carry a meaningless `guard: equal` from the CLI default; the P1 records predate the guard field.
- Refusals (sandbox, two commands; reissued as plain commands): "This session is isolated in the worktree ..., but this command is too
  complex to verify that it stays inside the worktree. Refusing to run it ..."; "... this command runs python with a program computed
  at runtime (the variable P) in a plain command, so what it runs cannot be shown not to be git. Refusing to run it ...".
- No data-access refusal occurred. No label of MODEL_SELECT, ATTACK or DEV_REPORT was read outside the scorer and grid.

## 10. Budget and CPU

PRIV24 23 of 24; CTRL12 10 of 12 (2 left unused, per the coordinator's rule). Queue units: 79 (75 done, 4 failed), 4,521.5 CPU-s (fit
2,824.9; readers 1,468.1; diagnostic 141.3; self-test 47.9; score 39.4). Out-of-queue charge PROTECTION_AGENT 180 CPU-s. Stage total
about 1.31 CPU-h. Physical fits: 17 complete pipelines × (6 corrections + 54 critics), plus 2 partial stopped pipelines; 58 K-means
fits; one projection scale per saved trained configuration. Reader fits 4,224; cache hits 1,344.

## 11. Files

`CARD.md` (through "Closing summary"); `ohn/protect.py`; `checks/PROT_SELFTEST.json`; `grid_specs/PROT_{P1,P2,P2A,P3,P3A,P2L,P2AL}_S*.json`;
private: `runs/PROT/<config>/S<s>/` (OCC.npz/json, PARAMS.json, THETA.npz or CODEBOOK.npz, VERIFY.json, EVAL_MS_H3.json),
`runs/PROT_TRAIN/<stem>/S<s>/`, `runs/GRID/{PROT_P1,PROT_P2,PROT_P2A,PROT_P3,PROT_P3A,PROT_P2L,PROT_P2AL}/`; ledger (35 PROT score rows;
rows 201–205).

Suggested next check: reader learning curves on P2A_PAIR_B80 and P2_PAIR_M90 with their CONT and COMP twins. (Coordinator's note: the
reader-robustness bank RR_LC_PROT_P2A already covers P2A_PAIR_B80 with P2A_CONT and P2A_COMP_B80 on both seeds; no learning curve was
run for P2_PAIR_M90.)
