# Prototype verification (Phase 2): independent recomputation of the MMP prototype

This file is written by the math_verifier. The predictions in section 0 were written **before** I ran unit `V_PROTO_FINITE_V1` or unit `V_PROTO_CONT_V1`. At that point I had already read the coordinator's results.

## 0. Predictions, registered before my own runs

Some of these predictions are informed by the coordinator's published results, which I read first. They are therefore *reproduction* predictions, not blind ones. The genuinely new predictions concern items that the coordinator's run did not test: P4b and P6.

- **P1. Finite laws.** Every reported quantity reproduces to ≤ 1e-9: AUC\*(S;X), twin and MMP pair/single AUCs, MMP excess vs η, the identities, the frontier minima, subgroup increases and w\*. Local, S12 and S21 may be sensitive to ties; I will report the range over tied choices.
- **P2. L_CONT_PROXY.**
  - Bayes: twin ≈ 0.79, MMP ≈ 0.72.
  - Fitted pair (LR and HGB): MMP below the twin by about 0.05 ± 0.01.
- **P3. L_CONT_NOPROXY.**
  - Bayes: twin and MMP are equal.
  - Fitted: twin's readers within 0.003 of Bayes; MMP within ±0.003 of the twin for LR and for HGB.
  - Consequence: no release keeping the same Bayes AUC could sit ≥ 0.005 below a near-Bayes twin. The pre-registered threshold was therefore close to unreachable in this law.
- **P4a. Matched-price partial eraser (fraction t < 1).** It is an invertible affine map of each recipient's centred log-probabilities. Refit LR readers on the log wire should not change by more than ±0.002, for singles or the pair.
- **P4b. Full eraser (t = 1).** It costs more than MMP's price. It lowers the occupation-only LR AUC substantially, and lowers the pair less: a masking signature. HGB recovers part of the occupation-only loss.
- **P5. Population within-SEX occupation increases** (exact, by quadrature):
  - men > women in both continuous laws;
  - the women's increase is negative in L_CONT_NOPROXY.

  If so, the cause is the non-constant gap (verifier M6), not a bug.
- **P6. Exact decomposition.** The group increases equal E[(π̄² − π²)‖g‖² | S=0] and E[((1 − π̄)² − (1 − π)²)‖g‖² | S=1], for any non-constant g. ‖g(x)‖² should be negatively correlated with π(x) in L_CONT, which explains the reversal.

### 0b. A further prediction, P7, for a post-hoc probe (`V_PROTO_HD_V1`)

**Status.** P7 was registered before that probe ran, and after the finite and continuous recomputations. It is **post-hoc relative to the coordinator's pre-registration** and is reported as such.

**Motivation.** In L_NOPROXY and L_CONT_NOPROXY the release is (near-)injective in the fields: 3 fields map to 3 release dimensions. That is stronger than "no pure-proxy field". With more fields than release dimensions, MMP's release depends on x only through the span T of the within-SEX task effects. It therefore merges every SEX-predictive *direction* outside T, even though every *field* is task-relevant.

**Probe law.**
- K = 10 Gaussian fields, x | S ~ N(S·d, I), with AUC\*(S;X) ≈ 0.70.
- Every field has nonzero within-SEX effects on both tasks.
- The task logits use 3 random directions (income 1, occupation 2) plus the same direct SEX effects as L_CONT.
- Three random draws of the directions.

**P7.**
- MMP's Bayes pair AUC equals Φ(‖P_T d‖/√2) exactly, which lies below AUC\*(S;X).
- The twin's fitted pair AUC (LR and HGB, n_fit = 8,001) exceeds MMP's by ≥ 0.01 in each draw, with no pure-proxy field present.
- MMP's fitted AUC lies within 0.01 of its exact Bayes value.

**What follows from each outcome.**
- If P7 holds, check (a)'s failure reflects the injective law design, not a general property of MMP without pure proxies.
- If P7 fails, the coordinator's reading generalises to this wider class.

---

## 1. Finite laws (unit `V_PROTO_FINITE_V1`, exact Fractions, own toolkit)

**Every quantity reproduces the coordinator's numbers exactly** (maximum absolute difference 0.0 on every scalar, in all six laws). That covers:
- AUC\*(S;X), and the twin's and MMP's pair and single AUCs;
- MMP's excess over η on each clause;
- the within-SEX increases and w\*;
- MMP's feasibility at both budgets;
- the number of feasible partitions;
- the global frontier minima at both budgets.

The coordinator's local, S12 and S21 values lie inside my tie ranges. The Brier identity holds as an exact Fraction equality in every law and task. NLL excess equals E KL(η‖m) to ≤ 2e-16, and the I(S;X) bound holds.

| Law | AUC\*(S;X) | Twin pair | MMP pair (inc / occ) | MMP feasible (cap / +headroom) | MMP occupation excess (acc / NLL / Brier) | Global frontier (cap / +headroom) | Within-SEX occ. increase (men / women) |
|---|---|---|---|---|---|---|---|
| L_NEG | 0.8111 | 0.8111 | 0.500 (0.500 / 0.500) | no / no | 0.0158 / 0.0346 / 0.0200 | 0.7722 / 0.7333 | 0.0200 / 0.0200 |
| L_PROXY | 0.7667 | 0.7667 | **0.500** (0.500 / 0.500) | **yes / yes** | 0 / 0.0078 / 0.00375 | **0.500 / 0.500** | 0.00375 / 0.00375 |
| L_PROXY_FLIP | 0.7667 | 0.7667 | 0.500 (0.500 / 0.500) | no / no (acc 0.0283) | 0.0283 / 0.0134 / 0.0084 | 0.650 / 0.650 | 0.0084 / 0.0084 |
| L_MIXED | 0.7167 | 0.7167 | **0.650** (0.650 / 0.650) | **yes / yes** | 0 / 0.0040 / 0.00234 | **0.650 / 0.650** | 0.00007 / **0.00461** |
| L_NOPROXY | 0.7311 | 0.7311 | 0.7311 (0.681 / 0.731) | yes / yes | 0 / 0.0066 / 0.00338 | **0.7033 / 0.6699** | 0.00410 / 0.00274 |
| L_XOR | 0.8500 | 0.8500 | 0.850 (0.500 / 0.500) | yes / yes | 0 / 0.0035 / 0.00184 | 0.850 / 0.850 | 0.00184 / 0.00184 |

**Additional observations.** None of these is an error.

1. **Local designs are tie-sensitive in two laws.** Many partitions tie on own-AUC and Brier.
   - L_NEG at the cap: local pair in [0.772, 0.811].
   - L_NOPROXY with headroom: local pair in [0.712, 0.731].

   The coordinator's single values are valid members of those sets, and the global minima are unaffected. Report the ranges if local designs are quoted.
2. **L_NOPROXY: MMP is feasible but sits 0.028 above the global frontier at the cap, and 0.061 above it with headroom.** The greedy S12 design attains the frontier. This is the registered "honest limit". It was not in the contradiction list, which covered only L_PROXY and L_MIXED.
3. **Which group bears the subgroup price is law-dependent.**
   - In L_MIXED, women bear 1.97× the population occupation price, close to the 0.005 guard.
   - In L_NOPROXY, men bear 1.21×.
   - In laws where π and the gap are independent, the price is equal across groups.
4. **w\*** differs from π̄ only for L_MIXED occupation (0.542). There it lowers the Brier price by 4.7% (0.002234 vs 0.002344).

## 2. Continuous laws (unit `V_PROTO_CONT_V2`)

**Setup.**
- Own sampler, posterior, eraser and readers. Unit `V_PROTO_CONT_V1` failed on a key-name bug in my own script and was re-run unchanged otherwise.
- Population values are exact up to quadrature, on an 801-point h-grid × all levels. Fitted values are the mean of 3 replications, with 8,001 fit rows and a 50,000-row test set.

| | L_CONT_PROXY | L_CONT_NOPROXY |
|---|---|---|
| Bayes, twin = AUC\*(S;X) (exact) | 0.7878 (coordinator MC 0.7913) | 0.7216 (coordinator MC 0.7222) |
| Bayes, MMP (exact) | 0.7216 (coordinator MC 0.7241) | 0.7216 (equal to the twin) |
| Fitted pair, twin − MMP: LR / HGB / MLP | **0.056 / 0.063 / 0.054** (coordinator slate 0.056) | **0.0006 / 0.0007 / 0.0004** (coordinator 0.0004) |
| Twin's fitted pair vs Bayes (LR) | 0.775 vs 0.787 | 0.7225 vs 0.7233 (occupation-only LR 0.7222) |
| MMP occupation price (Brier / NLL / accuracy) | 0.00844 / 0.0128 / 0.0104: **fails all three cap clauses**; passes cap + headroom | 0.00498 / 0.0076 / 0.0062: passes the cap (Brier at 0.996 of the cap) |
| Within-SEX occupation increase, men / women (exact) | **0.0130 / 0.0033** | **0.0111 / −0.0020** |
| Constant-gap (M5) prediction, men / women | 0.0083 / 0.0100 | 0.0058 / 0.0055 |
| corr(π, ‖g‖²) under P(x), occupation | −0.35 | −0.45 |
| Matched-price eraser fractions t (income, occupation) | 0.110, 0.427 (coordinator 0.110, 0.427) | 0.082, 0.342 (coordinator 0.081, 0.340) |
| Matched eraser: largest change in any fitted AUC vs twin | ≤ 0.0008 | ≤ 0.0026 |
| Matched eraser: CLR inversion error | 1.8e-15 | 1.3e-15 |
| Full occupation eraser (t = 1): drop in occupation-only AUC / pair AUC, LR | 0.232 / 0.049 | 0.193 / 0.003 |
| Full occupation eraser: drop in occupation-only AUC / pair AUC, HGB | 0.093 / 0.049 | 0.076 / 0.000 |
| Full occupation eraser: occupation Brier / NLL / accuracy price | 0.051 / 0.078 / 0.063 (6× MMP) | 0.048 / 0.073 / 0.057 (10× MMP) |

**What this establishes.**
- **Direction and size reproduce.** In L_CONT_PROXY, MMP lowers both the Bayes AUC (by 0.066) and the fitted AUC (by 0.054–0.063). In L_CONT_NOPROXY nothing moves.
- **The within-SEX decomposition is exact, and it explains the subgroup reversal** (P6 confirmed).
  - For any x-dependent gap, the decomposition is: men's increase = E[(π̄² − π²)‖g‖² | S=0], and women's = E[((1 − π̄)² − (1 − π)²)‖g‖² | S=1].
  - It matches the direct computation to machine precision.
  - Here ‖g(x)‖² is largest where π(x) is small (male-typical x), so men bear most of the price.
- **The matched-price eraser is exactly invertible.** P4a confirmed; the proof is in §4(c).
- **A non-invertible eraser does show the masking signature** (P4b confirmed): the single recipient drops far more than the pair. But it does so only at 6–10× MMP's price, and is grossly infeasible.
- **At one-sixth of the full eraser's price,** MMP's pair in L_CONT_PROXY (LR 0.719, HGB 0.697) is still below the full eraser's (0.726, 0.712).

## 3. Post-hoc probe P7 (`V_PROTO_HD_V1`, `V_PROTO_HD_V2`, `V_PROTO_HD_ALIGN_V1`): registered before running, and **failed**

**The law.**
- K = 10 Gaussian fields; every field has nonzero within-SEX effects on both tasks, so there is no pure-proxy field.
- The release has 3 dimensions, fewer than the number of fields, so it is not injective.
- AUC\*(S;X) = 0.6996 exactly.
- MMP's Bayes value is exact: Φ(‖P_T d‖/√2).

| Draw | MMP Bayes (exact) | Twin − MMP fitted pair, LR / HGB (8k) | Twin − MMP, strong HGB (1M rows) | Strong reader's gap to MMP's exact Bayes | Field-mediated vs direct SEX effects on the 3 logits | MMP occupation Brier price |
|---|---|---|---|---|---|---|
| 0 | 0.6026 | **+0.032 / +0.033** | **+0.033** | 0.0004 | all aligned | 0.0036 |
| 1 | 0.6346 | **−0.018 / −0.020** | **−0.018** | −0.0007 | all anti-aligned | 0.0039 |
| 2 | 0.5900 | **−0.019 / −0.020** | **−0.019** | 0.0027 | 2 of 3 anti-aligned | 0.0038 |

**Reading.**
- **The sign of MMP's effect is indeterminate when no pure proxy is present.** The strong reader reaches MMP's exact Bayes value to within 0.003, yet the twin stays below MMP in draws 1 and 2. So the reversal is not a reader-capacity artefact.
- **The sign is governed by alignment, exactly as the diagnostic shows.**
  - When the field-mediated SEX shift of each output logit has the same sign as the direct SEX effect, Δ *reinforces* the SEX signal, and deleting it helps (draw 0).
  - When the signs are opposite, Δ *masks* field-mediated SEX signal in the twin, and deleting it *unmasks* that signal: MMP raises recovery by about 0.02 (draws 1 and 2).
  - This is the continuous, generic form of the cancellation counterexample M10 from Phase 1. It was not confined to a measure-zero coincidence: it appeared in 2 of 3 random draws.
- **Merging through dimension reduction is real.** When the signs align, MMP lowers recovery by 0.033 with no pure-proxy field, which goes beyond the "pure-proxy merging" leg.
- **P7 as registered is falsified.**

## 4. The coordinator's interpretation: do I agree, and was anything a bug?

**No bug was found in the coordinator's prototype code.** Every number reproduces independently. The three failed pre-registered checks have different causes.

**(a) L_CONT_NOPROXY: fitted MMP not below the twin by ≥ 0.005 (observed 0.0004; mine 0.0004–0.0007).**
- *Agree that it failed, and that it is not a bug.*
- I would describe it as a **low-power test** rather than a contradiction in the strong sense:
  - the release is injective in the three fields;
  - the twin's readers are within about 0.001 of Bayes;
  - the pair is about equal to occupation alone.
- So no release keeping the same Bayes AUC could have been learned materially less well. The law shows that *when the release is easy to invert, deleting Δ changes nothing*. It cannot speak to readers far below Bayes.
- The post-hoc probe (§3) shows the more important point: with more fields than release dimensions, MMP's effect has **no definite sign**. It can lower recovery (−0.033) or raise it (+0.02).
- **So the coordinator's decision not to call MMP pilot-ready is right. If anything, the reason is understated.** The risk is not only "no learnability gain": MMP can *increase* coalition recovery. Only a real-data premise gate (F1, read with its sign) can tell which case applies.

**(b) Women's occupation increase > men's in L_CONT (observed the reverse).**
- *Not a bug, and not a failure of the mechanism: a mis-specified expectation.*
- The registered expectation came from my Phase 1 constant-gap formula M5 ("women ≈ 1.26×"). I had flagged that formula as condition-dependent (M6), but I stated the 1.26× too prominently.
- **I correct that here.** The exact decomposition (§2) applies to any gap. The sign of corr(π, ‖g‖²), together with κ₃, decides which group bears the price. In L_CONT it is negative (−0.35 / −0.45), so men bear it.
- The substantive finding is a feasibility warning. Against η, men's within-SEX occupation increase is **0.0130 and 0.0111, i.e. 2.2–2.6× the 0.005 guard**, in both continuous laws, while the population price is ≤ 0.0084.
- Any future subgroup prediction should come from the exact decomposition, with the sign of corr(π, ‖g‖²) estimated, not from M5.

**(c) The local eraser's masking signature (the eraser had essentially no effect).**
- *A mis-specified comparator, not evidence about MMP.*
- **The proof.** At fraction t < 1, the LEACE-type eraser maps each recipient's centred log-probabilities X ↦ μ + (X − μ)(I − tPᵀ). P is idempotent, so the map has the exact inverse I + t/(1 − t)·Pᵀ: numerical reconstruction error is at most 2e-15.
- **The consequence.** A defense-aware reader refit on the erased release, in particular a linear reader on log-probabilities, recovers the same information. The single-recipient AUCs and the pair AUC therefore cannot drop: observed changes are ≤ 0.003.
- **A properly non-invertible eraser (t = 1) does show the masking signature,** with the single recipient dropping far more than the pair. It does so only at 6–10× MMP's price.
- **What remains untested.** The deletion-versus-masking claim (MMP §3.6) remains **untested at matched price**. Testing it needs a non-invertible, field-driven local correction: a function of x such as a LOCAL_TREE-style leaf correction, or a local-target counter-shift like the MIN arm of `V_RCCS_MAP2_V1`.
- The companion check "MMP's pair is below the eraser's" passed trivially, because the eraser was the twin.

**The merging leg.**
- "Pure-proxy removal at the exact price" is confirmed: the price equals the identity's value, and the removal reaches the frontier in L_PROXY and L_MIXED at the cap.
- Two qualifications:
  - In L_CONT_PROXY the merging case itself **fails all three occupation clauses at the cap** and passes only with headroom. It also exceeds the men's subgroup guard 2.6×.
  - "Merging" also happens through dimension reduction, not only through pure-proxy fields, and the opposite unmasking effect can outweigh it (§3).

## 5. Scorecard of my registered predictions

| Prediction | Outcome |
|---|---|
| P1 (finite laws reproduce) | **held** (exact) |
| P2 (L_CONT_PROXY: Bayes 0.79 / 0.72; fitted −0.05 ± 0.01) | **held** (0.788 / 0.722; −0.054 to −0.063) |
| P3 (L_CONT_NOPROXY within ±0.003) | **held** (+0.0004 to +0.0007; readers within 0.001 of Bayes) |
| P4a (matched eraser has no effect) | **held** (≤ 0.003; exactly invertible) |
| P4b (full eraser shows masking at higher price) | **held** (single drop 0.19–0.23 vs pair 0.003–0.049 for LR; 6–10× price) |
| P5 (men > women; women negative in NOPROXY) | **held** (0.0130 / 0.0033; 0.0111 / −0.0020) |
| P6 (exact decomposition; negative correlation) | **held** (decomposition exact; corr −0.35 / −0.45 for occupation, −0.42 / −0.54 for income) |
| P7 (twin − MMP ≥ 0.01 in each high-dimensional draw) | **failed** (+0.033, −0.018, −0.019; sign set by alignment) |

## 6. Units and files

**Units.**
- `V_PROTO_FINITE_V1`
- `V_PROTO_CONT_V1`: failed on my own key bug.
- `V_PROTO_CONT_V2`
- `V_PROTO_HD_V1`
- `V_PROTO_HD_V2`
- `V_PROTO_HD_ALIGN_V1`
- `V_PROTO_SUMMARY_V1`

Together they used about 72 CPU-seconds; the campaign total is 0.168 CPU-hours.

**Outputs.**
- `V_PROTO_FINITE_V1.json`
- `V_PROTO_CONT_V1.json`, written by unit V2
- `V_PROTO_HD_V1.json`, `V_PROTO_HD_V2.json` and `V_PROTO_HD_ALIGN_V1.json`
- **`PROTOTYPE_VERIFICATION.json`**: the key recomputed numbers and the check verdicts

**Code.** `mechanism_design_independent/v_proto_{finite,cont,hd,hd2,hd_align,summary}.py`. None of it imports the coordinator's prototype modules; their law definitions were re-entered by hand.
