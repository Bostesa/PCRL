# Utility contract — ccm (frozen at FEASIBILITY_LOCK; independently reviewed in MATH_REVIEW.md)

## 1. Reference, decision and conventions

**Reference.** The PRIMARY reference is **Ucal**: the frozen hcal calibrated continuous teacher. It is the admitted
U of seed k with hcal's H-GLOBAL-TEMP inverse temperature α_k per task, under the frozen log-input rule:
- for α ≠ 1, p' = max(p, 1e-12) renormalised, then softmax(α log p');
- for α = 1, p exactly.

α is never refitted. **U0** (the raw teacher) is a SECONDARY reference: a bound against Ucal is NOT a bound against
U0.

**Decision.** d(x) = numpy argmax of U0(x), first index on ties (the source tie rule). Ucal preserves the first-index
numpy argmax, and this is checked on every row. It does NOT preserve strictness: tempering can turn U0 tops a few ulps
apart into an exact Ucal tie (MATH_REVIEW.md R3). Tied Ucal tops are counted label-free and handled in section 4.

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
- Pairwise feasibility is NOT sufficient for a whole bin when K ≥ 3. MATH_REVIEW.md gives 3-member counterexamples in
  exact rational arithmetic. For K = 2 every member's admissible set of q_1 is an interval, so (Helly in one dimension)
  pairwise or extreme-pair feasibility is sufficient (R4.7).
- For K ≥ 3 the set of inputs compatible with one output is not convex in p; membership is checked per input.
- **Certification.**
  - A representative is certified only by checking every member and every label with the canonical predicate of
    section 6.
  - Infeasibility is certified only through the closed-form necessary condition with the conservative margin of
    section 6. A solver that fails to find q never proves infeasibility; it is recorded NOT_FOUND.
- **Representative choice (registered, p-only).** An input is served by the FIRST representative, in the registered
  cover order, of its decision class that certifies it; otherwise it falls back. The rule reads only p, never a label
  or SEX. A release that chooses among admissible outputs by SEX could leak log 2 even when p carries nothing
  (MATH_REVIEW.md R6, R8.4).

## 4. Deployment behaviour on every valid input

**Unseen inputs.** For an input whose Ucal vector is not admissible under any registered representative of its
decision class, the contract defines a disclosed fallback with a fallback flag. Refusal is not used.
- If the Ucal top is strict, the fallback releases Ucal(x) itself, which satisfies G (q = p).
- Ucal(x) satisfies G iff its top is strict. If the top is tied, the fallback releases
  q = (1 − η)·Ucal(x) + η·e_d(x) with the fixed η = 1e-6. MATH_REVIEW.md R3.3 proves this admissible for every p whenever
  0 < η ≤ (√(1 + 8b) − 1)/4 = 0.0024876.
- The number of permitted rows with a tied Ucal top is reported label-free (expected 0).

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

## 6. Float64 semantics (registered after MATH_REVIEW.md B2, before FEASIBILITY_LOCK)

Algebraically equivalent float forms of G disagree at the boundary where NLL-tight representatives sit (the review
measured 100% vs 63% vs 59% passes for the multiplicative, log and ratio forms at q = fl(e^{−d}·p)). One form is
therefore named.

**Canonical predicate (certification).** It is `ccm.guard.check_release`, in float64:
- E = exp(−d), computed once.
- NLL: `q_k >= E * p_k` for every k.
- Brier: `sum(q*q) − sum(p*p) − 2*(q_y − p_y) <= b` for every y, with numpy sums along the class axis.
- Class: q_{d(x)} > q_k for every k ≠ d(x).
- Simplex: q finite, q ≥ 0, |Σq − 1| ≤ 1e-12.
- The independent verifier reimplements exactly this formula.

**Construction margin.** Representatives are built against tightened targets and checked at the true d and b. The
tightened targets are:
- NLL lower bound E·p·(1 + 1e-9);
- Brier target b − 1e-9;
- class gap 1e-9.

Every algebraically equivalent form (multiplicative, ratio, log, clipped; dot, per-row, fsum) then also passes at d
and b. This tightens the contract and never loosens it.

**Conservative closed forms.** Ucal rows sum to 1 only within 1e-12.
- A bin is declared INFEASIBLE_NLL only if Σ_k max p_k > e^d·(1 + 1e-12).
- The F2 infeasible-pair edges use the same test.
- The F3u neighbourhood counts a pair as compatible iff Σ_k max ≤ e^d·(1 + 1e-12).

F2 therefore stays a valid lower bound and F3u a valid upper bound.

**Log-form statements.** The log-form and clipped-score implications of section 2 hold in real arithmetic. In float64
they hold to about 3e-15.

## 7. What any admissible release can disclose (MATH_REVIEW.md R5, R6)

**Quantisation.** Every admissible output q pins each p_k to [e^d·q_k − (e^d − 1), e^d·q_k]. This is a quantisation of
the reference at about 0.005, or about 0.0025 where the Brier guard binds in the binary range.

**Bins needed versus capacity.** Covering a whole income decision class needs at least 147 bins, against a capacity of
8. Covering a uniform occupation class region needs about 5e10 bins, against 64.

**Smooth-law bound.** Suppose the release reads only p and independent randomness, and P(SEX | p) is L-Lipschitz in
total variation. Then any admissible release, deterministic or randomised, has SEX Bayes accuracy within L·(e^d − 1) of
releasing p itself.

**Non-smooth laws.** Without smoothness the bound fails: a law whose SEX alternates every 0.0005 in p can lose all its
information inside one admissible token. A plug-in estimate on finite rows has that form, so in-sample "protection"
from tiny bins is not evidence of protection on the population.
