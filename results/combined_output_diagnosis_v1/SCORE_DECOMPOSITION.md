# Score decomposition: what the released task output carries

**Status.** Development evidence on repeatedly used rows. Measured recovery by the registered attacker slate; this is
not an information quantity or a privacy guarantee. The full proofs and float-precision checks are in
`notes/review/SCORE_MATH.md` and `notes/review/score_math_checks.{py,out}`, which use synthetic arrays only.

## 1. The elementary identities

These are standard algebra, not a new method.

**Binary logits (l0, l1).**
- Margin d = l1 − l0 and offset c = (l0 + l1)/2. The map (l0, l1) ↔ (d, c) is linear and invertible (det = −1), with
  l0 = c − d/2 and l1 = c + d/2.
- softmax(l) = (σ(−d), σ(d)). So the probabilities, the argmax decision (1[d > 0]) and every proper task loss depend on
  d only. Subtracting c from both logits changes none of them.
- **Multiclass:** c = mean_k l_k and centred = l − c. Then l = centred + c, and softmax(l) = softmax(centred).

**What this does and does not show.**
- A deterministic function of a release cannot carry more information about S than the release itself.
- Removing c is an identifiable ablation, but it does **not** prove that c reveals S. Fitted-attacker AUC need not
  follow information orderings, in either direction.
- A head that is a fixed public function of protected features adds no population information beyond those features.
  It can still help a finite attacker use what is already there.

**Numerical checks on the stored heads** (`EXACTNESS.json`).
- The round trip (d, c) → (l0, l1) is bit-exact: the stored logits are float32 values upcast to float64.
- argmax = 1[d > 0] on every row.
- |softmax₁ − σ(d)| ≤ 1.7·10⁻¹⁶.
- No primary binary head reaches float64 saturation (p1 = 1 needs d ≥ 36.74). Rows with |d| ≥ 16.64, which would
  saturate in float32, are counted per head.
- The Adult education heads are heavily saturated on seeds 1 and 2. So their probability surface is not a mere
  re-parameterisation of the centred logits; this is noted wherever those cells appear.

## 2. The ablation

**Recipient and attackers.** Outputs only: the recipient gets the output and nothing else. The attackers are the
benchmark slate:
- GBT or MLP, selected on attacker_val log loss;
- refitted at attacker seeds 0, 1, 2;
- standardised on fit rows only.

**Banks.** These are validation-only and fit-free:
- **ignore-offset (IO) bank:** {centred = d, probability};
- **full bank (FB):** {full logits, (d, c), centred, probability}.

**Primary results.** Frozen-head stratum, mean over 3 encoder seeds. Two-sided Bonferroni intervals over the 30-endpoint
family, z = 3.144.

| Surface | Adult income / sex | HMDA underwriting / race |
|---|---|---|
| Full bank (FB) | 0.779 | 0.769 |
| Full logits (own slate) | 0.776 | 0.768 |
| (d, c) (own slate) | 0.778 | 0.769 |
| Ignore-offset bank (IO) | 0.708 | 0.702 |
| Centred = d | 0.708 | 0.701 |
| Probability | 0.705 | 0.702 |
| **Offset c alone** | **0.704** | **0.737** |
| Hard decision | 0.513 | 0.505 |
| True task label (diagnostic, not a release) | 0.601 | 0.522 |
| Constant | 0.500 | 0.500 |

**FC: removing the offset** (R(FB) − R(IO), target > 0.02).

| Cell | Δ | Bounds | Decision | Full-bank selection by seed |
|---|---|---|---|---|
| Adult | +0.072 | [0.062, 0.081] | **PASS** | dc, dc, full |
| HMDA | +0.067 | [0.060, 0.075] | **PASS** | full, dc, dc |

Both passes are **offset-attributable**: an offset-using candidate was selected on every seed.

**CH: releasing the decision instead of the centred scores** (R(IO) − R(hard), target > 0.02).

| Cell | Δ | Lower bound | Decision |
|---|---|---|---|
| Adult | +0.195 | 0.177 | **PASS** |
| HMDA | +0.196 | 0.185 | **PASS** |

**Group pairs.**
- **Adult sex (0, 1):** a declared alias of the macro result, and equal to it.
- **HMDA race, estimable pairs:**
  - FC passes for (0,1) +0.081, (0,2) +0.060 and (1,2) +0.060.
  - CH passes for (0,1) +0.220 and (0,2) +0.240.
  - CH is NOT_ESTABLISHED for (1,2): +0.023 with lower bound −0.005.

  On that pair (Black vs Asian applicants), the decision barely separates the two groups less than the centred scores
  do. A macro average would hide this.
- The 7 pairs involving race groups 3 and 4 are NOT_ESTIMABLE, as declared.

**Reading the results.**
- The common offset, a quantity that changes no probability, no decision and no task loss, is by itself about as
  sex/race-revealing to these attackers as the decision-relevant margin. Withholding it lowers measured recovery by
  0.07 AUC in both primary cells, with the probabilities left exactly unchanged.
- The probability vector and the centred logits recover equally (0.705 vs 0.708). The earlier full-vs-probability gap
  was the offset, not the probability conversion.

## 3. The refitted-head stratum (secondary)

HEAD__A is an LR head refitted on the untreated representation. Its logits are log-probabilities, so its offset is a
deterministic function of d. Full and centred therefore carry the same information by construction: FB = IO =
**0.774** (Adult) and 0.635 / 0.660 (HMDA).

The comparison that matters is with usefulness:

| | Frozen head | Refitted head |
|---|---|---|
| Adult accuracy gain over constant | 0.039 | 0.081 |
| Adult balanced accuracy | 0.57–0.60 | 0.66–0.76 |
| Adult recovery from the offset-free output | 0.708 | 0.774 |

**So removing the offset is not a general privacy lever.** A more useful head can leak more through the very part of
the output its decisions need. The offset result is specific to how the frozen PCRL heads were parameterised.

## 4. Replication over the 14 stored purpose/attribute pairs (S3, z = 3.180)

**FC (offset):**
- **PASS on 5 of 14:** Adult income × {sex, race}; HMDA underwriting × {race, ethnicity}; HMDA pricing × race (a
  5-class head).
- **NOT_ESTABLISHED on 9.**
  - All 6 Adult multiclass employment and education pairs sit at −0.017 to +0.001. The centred vector does as well as
    the full logits there, so it is a **counterexample** to a general offset effect.
  - HMDA pricing × sex, and HMDA fair_lending × {race, sex}: that head is constant (see below).
- **Interpretation:** the offset effect is a property of particular heads, not of logits in general.

**CH (decision vs centred scores):** PASS on 14/14, but the reasons differ:
- On HMDA fair_lending the frozen head predicts the majority class for every person on every seed. The "decision" is a
  constant, so its low recovery is trivial, and its task gain is exactly 0.
- On Adult education/income, CH is only +0.042.

**Usefulness of frozen heads (S3-U):**
- **PASS:** Adult income (+0.039), employment (+0.687), education (+0.632); HMDA pricing (+0.371).
- **NOT_ESTABLISHED:** HMDA underwriting (+0.0156, lower bound 0.0097, just below the 0.01 target); HMDA fair_lending
  (0.000: a constant head).

## 5. Canonical output utility

`odx/canonical.py::canonical_logits` returns the centred logits and the removed offset. It is tested in
`odx/tests/test_fixtures.py`, including a doctest example, for K = 2, 4, 6:
- softmax unchanged (≤ 1e-12);
- decisions unchanged.

**What it is.** A standard output transformation. **Measured effect:** −0.07 AUC for these frozen binary heads and this
attacker slate; no effect for the multiclass employment and education heads, or for a refitted LR head.

**What it is not.** It is not a privacy guarantee and not a new algorithm.
