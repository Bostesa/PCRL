# MMP data-free prototype: report

**Provenance.**
- Pre-registered in `PROTOTYPE_PREREGISTRATION.json`: expected outcomes and code hashes, committed in `998726d62` before any prototype run.
- Run once as queue unit `COORD_PROTOTYPE_MMP_V1` (52 CPU-seconds), producing `PROTOTYPE_RESULTS.json` (sha256 `0ea7c567…`).
- Code: `mechanism_design/prototype/`. The verifier's independent recomputation is in `../roles/math_verifier/PROTOTYPE_VERIFICATION.md`.

**What kind of evidence this is.**
- **Finite laws are exact.** Arithmetic uses fractions, and every one of the 41,209 calibrated partition pairs is enumerated per six-cell law.
- **Continuous laws are approximate.** Readers are fitted on finite samples, so their AUCs are estimates for that law and that slate, not population floors.
- **Possibility only.** Every law is constructed. A pass shows what is possible under that law, not a real-data advantage or novelty.

## Reference, budgets and comparators

- **Reference:** the exact SEX-blind posterior η, i.e. a perfect U0.
- **Budgets:**
  - cap: accuracy 0.010, NLL 0.010, summed Brier 0.005;
  - cap + headroom: 0.016, 0.020, 0.009, mirroring the ACS teacher headroom.
- **Strongest baseline analogue:** the exact global frontier, the minimum pair Bayes AUC over every calibrated partition pair feasible at the budget. It weakly dominates every deterministic release in Bayes terms, the neural control included.
- **Also computed:**
  - the local design, in which each recipient minimises its own leakage;
  - both greedy sequential orders, S12 and S21;
  - MMP's own partition, released calibrated.

## Finite laws (exact)

Pair columns are the Bayes AUC of the linked pair. Frontier values are global / local / S12 / S21 at the cap.

| Law | What it is | AUC*(S;X) = twin pair | MMP pair | MMP feasible? (cap; +headroom) | Frontier at cap | Verdict vs pre-registration |
|---|---|---|---|---|---|---|
| L_NEG | Tasks depend on X only through S (required disclosure) | 0.811 | 0.500, with no task information | no; no (occupation Brier +0.020) | 0.772 / 0.772 / 0.772 / 0.772 | as expected |
| L_PROXY | z is a pure SEX proxy; w is a task feature independent of S | 0.767 | **0.500** | **yes; yes** (occupation Brier 0.00375, no decision changes) | 0.500 at every design | as expected: genuine removal |
| L_PROXY_FLIP | as L_PROXY, but gendered gaps swap the occupation argmax | 0.767 | 0.500 | no; no (accuracy loss 0.028) | 0.650 | as expected (failure mode 2) |
| L_MIXED | h moves P(S) and the tasks; z is a pure proxy | 0.717 | **0.650 = AUC*(S;h)** | yes; yes | 0.650 at every design | as expected: residual through the task-relevant proxy |
| L_NOPROXY | every SEX-predictive field affects the tasks within SEX | 0.731 | 0.731 (no Bayes effect) | yes; yes | 0.703 global / 0.731 local / 0.703 S12 / 0.731 S21 | as expected: honest limit |
| L_XOR | income needs a, occupation needs b, and SEX follows a XOR b | 0.850 | 0.850 (singles 0.500) | yes; yes | 0.850 at every design | as expected: coalition required disclosure |

**Identities.** In every law and both tasks, the Brier identity holds exactly, and NLL excess = E KL(η‖m) ≤ I(S;X).

**w\*.** It differs from π̄ only in L_MIXED: 0.542 vs 0.500, lowering the occupation Brier price from 0.00234 to 0.00223.

**Subgroup, occupation within-SEX Brier increase.**
- L_MIXED: women +0.0046, men +0.0001. Women bear 1.97× the population price.
- L_NOPROXY: women +0.0027, men +0.0041.
- Every other law splits equally (π̄ = 1/2, symmetric).

**What the finite laws show.**
- MMP deletes SEX information exactly where it enters only through the mixture weights, i.e. along pure-proxy directions, and it pays exactly the predicted price.
- It does nothing where the tasks themselves require the SEX-revealing fields: L_NEG, L_XOR and L_NOPROXY.
- It can fail accuracy when gendered gaps change decisions: L_PROXY_FLIP.
- In L_NOPROXY the global frontier at the cap (0.703) is *lower* than MMP's 0.731. A partial merge of task-relevant cells beats MMP there, and so does S12. MMP is not frontier-optimal in that regime.

## Continuous laws (fitted ACS seven-reader slate; 3 replications)

Each replication uses 8,001 fit, 3,000 validation and 5,943 test rows, with 21 pair candidates. Figures are mean test AUCs; Bayes figures are Monte Carlo estimates with exact posteriors.

| Law | Arm | Income | Occupation | Pair | Bayes pair |
|---|---|---|---|---|---|
| L_CONT_PROXY (pure proxy Z present) | twin (η) | 0.666 | 0.765 | 0.779 | 0.791 |
| | **MMP** | 0.645 | 0.723 | **0.723** | 0.724 |
| | local linear eraser (same Brier price) | 0.666 | 0.765 | 0.779 | 0.791 |
| L_CONT_NOPROXY (no pure proxy) | twin (η) | 0.659 | 0.721 | 0.721 | 0.722 |
| | **MMP** | 0.650 | 0.721 | **0.721** | 0.722 |
| | local linear eraser (same Brier price) | 0.659 | 0.721 | 0.722 | 0.722 |

**MMP's utility price over η** (Monte Carlo):

| Law | Occupation summed Brier | Occupation NLL | Occupation accuracy | Within-SEX occupation Brier increase |
|---|---|---|---|---|
| L_CONT_PROXY | 0.0084 | 0.0128 | 0.0104 | men 0.0134, women 0.0035 |
| L_CONT_NOPROXY | 0.0050 | 0.0077 | 0.0063 | men 0.0115, women −0.0023 |

- The L_CONT_PROXY price is above the strict caps but within cap + headroom.
- In both continuous laws men bear more, reversing the constant-gap prediction (verifier M6: the gap is not constant here).

**What the continuous laws show.**
- With a pure proxy present, MMP's gain of 0.056 is the Bayes-level merge (0.791 → 0.724), and fitted readers reach it.
- **With no pure proxy, deleting Δ changed nothing a reader learns:** pair 0.7211 vs 0.7207. In these smooth, low-dimensional laws the readers are near-Bayes, and MMP's release is still injective, so its Bayes AUC equals the twin's.
- The matched-price local linear eraser had essentially no effect on any view. The "deletion beats masking" comparison is therefore **inconclusive**, not won: the masking arm did not mask.

## Pre-registered checks

24 of 27 held. Three failed:

| Check | Expected | Observed | Meaning |
|---|---|---|---|
| L_CONT_NOPROXY: fitted MMP below twin by ≥ 0.005 | yes | **no** (0.0004) | The **learnability leg** of MMP's rationale is not supported. Without exact merging, deleting the inferred-SEX term did not reduce what readers learn |
| L_CONT: women's occupation increase > men's | yes | **no** (the reverse) | The subgroup burden direction depends on the law. A pilot must report both groups |
| L_CONT: local eraser shows a masking signature | yes | **no** (no effect at all) | The masking comparison is inconclusive |

## Verdict

**MMP's rationale is partly contradicted by its own pre-registered prototype.**
- **Confirmed.** The merging leg holds: exact removal of pure-proxy SEX information at the exact predicted price, with exactly the expected limits (L_NEG, L_XOR, L_NOPROXY).
- **Contradicted.** The learnability leg fails, and it is the leg a real-data benefit would need. Fitted SEX-aware CatBoost releases are essentially injective, so they do not merge cells. The synthetic pilot rehearsal found 797–798 distinct release pairs for 799 distinct inputs.

Per the brief, **MMP is not described as ready for a real-data pilot.** The prepared pilot (`../PILOT_PROTOCOL_DRAFT.md`) is delivered with that status.

## Independent verification and corrections (added 2026-10-16 after the verifier's phase 2)

Source: `../roles/math_verifier/PROTOTYPE_VERIFICATION.md` and `.json`. The verifier used separate code and registered predictions P1–P7 before its runs.

1. **Reproduction.**
   - Every finite-law quantity reproduces exactly: maximum difference 0.0.
   - The continuous laws reproduce in direction and size: L_CONT_PROXY twin − MMP 0.054–0.063 (exact Bayes 0.788 → 0.722); L_CONT_NOPROXY 0.0004–0.0007.
   - **No bug was found in the coordinator's prototype.**
   - Local-design values are tie-sensitive in two laws: L_NEG at the cap ranges 0.772–0.811, and L_NOPROXY with headroom 0.712–0.731. The global minima are unaffected.
2. **Coordinator error: the masking comparator was mis-specified.**
   - At fraction t < 1, the matched-price local eraser is an exactly invertible affine map of each recipient's centred log-probabilities (reconstruction error ≤ 2e-15). Readers refitted on the defended release therefore recover the same information.
   - The arm could not mask, and check (c) carries no information about MMP. The companion check, "MMP's pair below the eraser's", passed trivially.
   - A non-invertible eraser (t = 1) does show the masking signature: the single recipient drops 0.19–0.23 against the pair's 0.003–0.049 for LR readers. But its price is 6–10× MMP's.
   - **Deletion versus masking at matched price remains untested.** A proper test needs a non-invertible, field-driven local correction.
3. **The subgroup expectation was mis-specified.**
   - The pre-registered "women's increase > men's" came from the constant-gap approximation (verifier M5).
   - The exact decomposition for any gap is: men E[(π̄² − π²)‖g‖² | S=0], women E[((1 − π̄)² − (1 − π)²)‖g‖² | S=1]. Its sign is set by corr(π, ‖g‖²), which is −0.35 / −0.45 for occupation in L_CONT.
   - **Against η, men's within-SEX occupation increase is 0.0130 and 0.0111: 2.2–2.6× the 0.005 guard,** while the population price is at most 0.0084.
4. **Check (a) is a genuine failure, of low power.** The L_CONT_NOPROXY release is injective in three fields, and the readers sit within 0.001 of Bayes. No release with the same Bayes AUC could have been learned materially less well.
5. **Decisive post-hoc probe P7** (registered by the verifier before running; post-hoc relative to this report's pre-registration).
   - The law: 10 Gaussian fields, every one task-relevant within SEX, and a 3-dimensional release.
   - **MMP's effect had no definite sign:** twin − MMP was +0.033, −0.018 and −0.019 in three random draws.
   - A strong reader trained on 1M rows reached MMP's exact Bayes value within 0.003, so the reversal is not a reader-capacity artefact.
   - The sign follows whether field-mediated and direct SEX effects align. When they are opposed, Δ masks field-mediated SEX signal in the twin, and deleting it **raises** recovery by about 0.02.
6. **The merging case fails the strict caps.** In L_CONT_PROXY the merge fails all three occupation clauses at the strict cap and passes only with headroom.

**Updated verdict.** MMP's genuine removal is confirmed only where it merges inputs: pure-proxy fields, or SEX directions outside the span of the within-SEX task effects. Without such structure its effect on coalition recovery has no definite sign, and it can increase recovery. Its utility price, and in particular the within-SEX burden, can exceed the guards.

**MMP is not ready for a real-data pilot.** The verifier agrees and states the reason more strongly than the coordinator did.

## Correction note (added 2026-10-16 in the correction pass; numbers, checks and verdict unchanged)

See `../CORRECTIONS.md` rows 8 and 10.

- **The verdict's sentence about trees.** "Fitted SEX-aware CatBoost releases are essentially injective, so they do not merge cells. The synthetic pilot rehearsal found 797–798 distinct release pairs for 799 distinct inputs." This is **unsupported**.
  - Those counts come from 800 invented MODEL_SELECT rows, compared bitwise. MMP had 1–2 collisions per seed and the twin none.
  - They describe that synthetic sample only. Population injectivity of any real model is unknown.
  - The failure of the learnability leg (check (a)) and the verifier's probe P7 do not depend on this sentence.
- **"The leg a real-data benefit would need"** is **conditional**. A real-data benefit would need either merging structure in the Florida fields (inputs with different SEX posteriors merged by the release) or a learnability effect. Which applies, if either, is unknown.
- **"With no pure proxy, deleting Δ changed nothing a reader learns"** is correct for L_CONT_NOPROXY, whose release is one-to-one in three fields by construction. It is not a general statement about laws without pure proxies (P7).
- **The "Updated verdict"** above already states removal through merging of SEX directions outside the span of the within-SEX task effects. That stands. Its next sentence, "Without such structure its effect … has no definite sign", should read "without pure-proxy fields": the probe's releases do merge through dimension reduction, and the sign was still indeterminate (`../CORRECTIONS.md` row 21).
