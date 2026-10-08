# Utility contract — ccm (frozen at FEASIBILITY_LOCK; independently reviewed in MATH_REVIEW.md)

## 1. Reference, decision and conventions

**Reference.** The PRIMARY reference is **Ucal**: the frozen hcal calibrated continuous teacher. It is the admitted
U of seed k with hcal's H-GLOBAL-TEMP inverse temperature α_k per task, under the frozen log-input rule:
- for α ≠ 1, p' = max(p, 1e-12) renormalised, then softmax(α log p');
- for α = 1, p exactly.

α is never refitted. **U0** (the raw teacher) is a SECONDARY reference: a bound against Ucal is NOT a bound against
U0.

**Decision.** d(x) = numpy argmax of U0(x), first index on ties (the source tie rule). Ucal preserves it, and this is
checked on every row.

**Conventions.**
- Natural logarithm.
- Scoring log loss is −log clip(prob_y, 1e-12, 1).
- Multiclass Brier is Σ_k (q_k − 1{k = y})². There is no factor ½: this is the source per-row convention
  (dpc.utility.per_row).

## 2. The pointwise all-label guard G(d, b)

**Constants.** d = 0.005 and b = 0.0025. These are the only registered setting, inside hcal's 0.010 / 0.005
allowances, and they are never loosened in this sprint.

**Conditions.** For an input with reference p (positive probabilities; K classes) and released vector q in the simplex:
- **(NLL)** q_k ≥ exp(−d)·p_k for every k.
  - ⇒ log(p_y / q_y) ≤ d for every label y.
  - The clipped scoring loss obeys the same bound (MATH_REVIEW.md checks this): with clip c = 1e-12, if q_y < c the
    clipped q-loss is −log c, and p_y ≤ e^d·q_y < e^d·c gives an excess below d.
- **(Brier)** for every label y, Σ_k q_k² − Σ_k p_k² − 2(q_y − p_y) ≤ b.
- **(Class)** q has the strict argmax d(x).

**Scope.** These are utility statements relative to Ucal, pointwise for every input and every possible label. They are
not average or expected statements. They do not prove calibration, task correctness, attribute privacy, or usefulness
of an uninformative teacher.

**Randomised releases.** The conditions must hold for EVERY supported output. A mean guarantee is not a pointwise
guarantee, and the two are not interchangeable.

## 3. Bins, representatives and certificates

- A token is a bin B of inputs with one representative q released for all of them. B is admissible iff a single q
  satisfies G for every p ∈ B.
- **Necessary condition (NLL):** Σ_k max_{p∈B} p_k ≤ exp(d). Since Σ_k max(p_k, p'_k) = 1 + TV(p, p'), every admissible
  bin has total-variation diameter at most exp(d) − 1 ≈ 0.0050125.
- Pairwise feasibility is NOT sufficient for a whole bin. MATH_REVIEW.md gives a 3-member counterexample.
- **Certification.**
  - A representative is certified only by checking every member and every label in float64 with no tolerance.
  - Infeasibility is certified only through the closed-form necessary condition. A solver that fails to find q never
    proves infeasibility; it is recorded NOT_FOUND.

## 4. Deployment behaviour on every valid input

**Unseen inputs.** For an input whose Ucal vector is not admissible under any registered representative of its
decision class, the contract defines a disclosed fallback: release Ucal(x) itself with a fallback flag. This satisfies
G trivially (q = p). Refusal is not used.

**Privacy view.** Fallback flags, fallback vectors, token identities, representatives and all metadata belong to the
recipient's view and must be attacked. A fallback can undo privacy: it discloses p.

## 5. Teacher-expected guard G_exp(d, b) — SECONDARY DIAGNOSTIC ONLY

**Conditions.**
- KL(p‖q) ≤ d: the expected log-loss excess if y ~ p;
- Σ_k (q_k − p_k)² ≤ b: the expected Brier excess if y ~ p;
- strict class preservation.

**Scope.**
- G_exp is weaker than G. Its observed-label meaning depends on calibration of the reference.
- Pinsker gives TV ≤ √(d/2) ≈ 0.05 per member, so a bin's diameter can be about 0.1.
- It is computed only to inform the next study's contract. It is never the selected contract, never a pass, and never
  a prototype here.
