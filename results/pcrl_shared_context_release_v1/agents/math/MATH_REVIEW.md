# MATH_REVIEW — shared-context release v1 (independent mathematical reviewer)

Date 2026-09-24. Scope: handoff §§4, 7, 8, 11, 14, 16, 21, 22 and the predecessor code at
`2ce5d171f` (`experiments/pcrl_adaptive_release_v1/{refinement,reference}.py`). Pure synthetic
laws only; no ACS rows, no cloud, nothing committed. Evidence: `reproduce_examples.py` (run from
worktree root; all assertions pass) and `reproduce_examples.out`; exact fixtures in
`tests/pcrl_shared_context_release_v1/fixtures/math_review_exact_laws.json`.

## 0. Findings that should change the design (ranked)

1. **Single shared eta + K=1 cannot say *where* to use a richer policy** (§2.4, exact counterexample C).
   With bank {D17, d_new} and K=1 the nested LP value equals `min(C(D17), C(d_new))`; the pointwise
   combination "D17 where it is better, d_new where it is better" (value 0 vs 1/5) is not in the class.
   Adding D17 as a column does **not** fix this. What fixes it: contexts aligned with where the policy
   helps, or D17-anchored (switched) policy columns. Consequence: at K=1 the mixture can only
   *time-share globally*; every within-T32 decision must already be inside a policy column.
2. **Vertex structure ⇒ the nested LP returns time-sharing, and interior eta randomizes everyone** (§5).
   With r tight cuts the optimum is a convex combination of ≤ r+1 deterministic releases; if
   0<eta<1 it is `(1−eta)·(deterministic T32 kernel) + eta·(compact policy)`, so every person on whom
   the two disagree is randomized. "Few binding constraints ⇒ almost deterministic" holds in parameter
   space and for eta∈{0,1} with hard contexts, **not** for interior eta or soft contexts.
3. **eta is not identified** (§2.5). Example A's private channel is emitted at eta=1 (A on the D17
   column) and at eta=3/4 (D17 carried in B) — identical q. Each T32-measurable column adds one exact
   alias direction (rank check: 533 → 532 → 531). Report within-T32 token-law variation and a
   T32-projection test, never eta.
4. **Frozen attackers are anti-conservative for mixtures; frozen decoders are conservative** (§3.3, S).
   Mixing D17 with its own token relabel (pure randomized response) at λ=0.1: frozen-attacker loss
   0.5836 vs refit (Bayes) 0.5731 — an overstatement of 0.0105 nats, 10× delta=.001. Any LP
   "privacy gain" must be re-scored with attackers refit on the candidate itself.
5. **Task sufficiency lemma** (§6.1): if Y ⟂ X | (T, H_A), no within-(T,H_A) variation can change
   *any* decoder's population task loss. Richer inputs can then help only privacy (Example A is exactly
   this: its T-projection keeps I(Y;Z)=0.08972 but loses privacy). In-sample task gains from
   within-cell variation are then pure overfitting. The utility route U needs X_A to carry task
   information beyond (T, H_A); the task-only richer policy is the preflight for that.
6. **Convexity map correction** (§2.3): attacker refits and the `min` over the bank do **not** break
   convexity (the constraint set `{q: min_{a∈F} L_a(q) ≥ c}` is an intersection of half-spaces for any
   fixed family F on fixed rows). **Decoder refits do** (utility route minimizes a concave function;
   privacy route's task guard becomes a union of half-spaces). Policy/context learning is non-convex.
7. **Rebasing can admit a previously infeasible candidate** (§3.2, P): adding a stronger attacker
   dropped rho 0.7584 → 0.5004 and made a candidate with L_a1=0.6380 feasible.
8. Persistence: Example A's exact privacy fails under two fresh draws (P(Z1=Z2=1|S)=1/5 vs 1/20).
   A persistent per-record draw is part of the mathematical contract, not an implementation detail.

## 1. Examples A and B (reproduced exactly from committed fixtures)

`refinement.fixture_b1/b2` at the pinned commit were imported and every field cross-checked against an
independent exact (Fraction) recomputation.

**Example A.** Masses (2/5,1/10,1/10,2/5), S=(0,1,0,1), P(Y=1|x)=(1/10,1/10,9/10,9/10), T∈{lo,hi}.
P(S=0)=P(S=1)=1/2; P(T=lo|S=0)=4/5, P(T=lo|S=1)=1/5 (det 3/5). For a T-only channel with rows q_lo,
q_hi over *any* alphabet, P(Z|S=0)−P(Z|S=1) = (3/5)(q_lo−q_hi), so exact privacy ⇔ q_lo=q_hi ⇒ Z ⟂ X
⇒ I(Y;Z)=0. Richer channel P(Z=1|x)=(0,0,1,1/4): P(Z=1|S=0)=P(Z=1|S=1)=1/5 exactly,
I(Y;Z)=0.08972125227 nats. All 15 set partitions (restricted-growth strings; this covers every
deterministic map for any alphabet ≥4) enumerated: exactly 2 private — (0,0,0,0) and {x0,x3|x1,x2}
— both with I(Y;Z)=0. Extras:
- Rassouli–Gündüz Prop. 1 (general observation W) certifies it: for W=X, v=(1,0,−1,0) ∈ Null(P_{S|X})
  with P_{Y|X}v=(4/5,−4/5)≠0 ⇒ feasible; for W=T, P_{S|T} is invertible ⇒ infeasible.
- Binary private channels form {q∈[0,1]^4: one linear equality}; 8 vertices; I(Y;Z) is convex in q so
  its max is at a vertex: the fixture channel, which has **exactly one fractional row**.
- It is a **K=1 compact mixture** ¼·D17-analog(0,0,1,1) + ¾·d_new(0,0,1,0), and **neither component is
  private** (P(Z=1|S) = (1/5,4/5) and (1/5,0)). Randomization is strictly necessary here.
- Fixture hygiene: `t_only_private_implies_row_equality` and `context_blind_private_implies_constant`
  are hard-coded constants in the fixture, not derived; both are re-derived in the script.

**Example B.** H,T fair independent; H=0: Y=T, S⟂; H=1: Y=0, S=T. Z = T·1{H=0}: I(Y;Z|H)=log2/2 =
0.34657359027997, I(S;Z|H)=0 exactly. Context-blind q(z|T): at H=1, P(Z|S=s,H=1)=q(·|T=s) ⇒ privacy
forces q(·|0)=q(·|1) ⇒ zero task information. Nested representation: hard contexts φ_k=1{H=k}, bank
{Z=T, Z=0}, eta=1, one-hot A — deterministic. Scope: both are finite-law possibility results
(§7 of the handoff), not ACS evidence or new theorems.

## 2. Nested parameterization

`q(z|x) = B[T(x),z] + Σ_k Σ_m φ_k(x) A[k,m] 1{d_m(x)=z}`, B,A ≥ 0, Σ_z B[t,z]=1−eta ∀t,
Σ_m A[k,m]=eta ∀k, φ_k ≥ 0, Σ_k φ_k(x)=1, d_m(x)∈{0..16}.

**P1 (row-stochastic).** q ≥ 0 termwise; Σ_z q = (1−eta) + Σ_k φ_k(x)·eta = 1. Needs, on every row
including deployment rows: Σ_k φ_k(x)=1 (no NaN/missing-feature fallthrough), every d_m emits exactly
one of the 17 tokens (a "withhold" action must be a token), and B rows of unsupported states pinned.
**P2 (affine, polytope).** q is linear in (B,A); eta enters only the constraints (eta is determined by
A, a linking variable). The feasible set 𝒫 is a polytope and equals the **free join**
conv(𝒫₀ ∪ 𝒫₁), 𝒫₀={eta=0}=all T32 kernels, 𝒫₁={eta=1}=compact model: any point with eta∈(0,1) is
(1−eta)(B/(1−eta),0,0)+eta(0,A/eta,1). The affine hulls are skew, so dim 𝒫 = dim 𝒫₀+dim 𝒫₁+1 and
vert 𝒫 = vert 𝒫₀ ∪ vert 𝒫₁ (deterministic T32 kernels ∪ one-hot-A policies). Exact check in script.
**P3 (T32 embedding).** eta=0 ⇒ A=0 and B is any row-stochastic 32×17 kernel. D17 is a 32×17 matrix
in `reference.calibrate_reference` (T32-measurable), so B=D17 is exact. **P4.** eta=1 ⇒ B=0, compact.
**P5 (linearity for frozen decoder/attacker).** With ℓ_iz = loss(y_i, dec(H_i,z)):
`L = Σ_{t,z} B[t,z]·c^B[t,z] + Σ_{k,m} A[k,m]·C[k,m]`, c^B[t,z]=Σ_{i:T_i=t} w_i ℓ_iz,
C[k,m]=Σ_i w_i φ_k(x_i) ℓ_{i,d_m(x_i)}; identically for each attacker on its view. Verified exactly
(Fractions) against the person-level expected-token sum. Note c^B for attackers must be aggregated
from **per-person** losses (H varies within T32), not from a state-level attacker table.

### 2.3 What breaks convexity (and what does not)

| Operation | Effect |
|---|---|
| Frozen decoder, frozen finite bank, frozen φ, d | LP (P5). |
| `rho = min_{a∈bank} L_a(D17)` | A constant; no effect. |
| Constraint `min_{a∈F} L_a(q) ≥ c` for a *fixed* family F on fixed rows | min of linear = concave ⇒ superlevel set convex (= ∩_a half-spaces). Attacker refit = separation oracle of a convex semi-infinite program (Kelley/constraint generation). **Does not break convexity**; it is only an inexact oracle when fits are approximate or on other rows. |
| Attacker *selected on validation rows* inside the constraint | Not a min on the same rows ⇒ no concavity; non-convex. |
| Decoder refit in the objective (utility route) | `min_dec L(q,dec)` is concave in q; minimizing it is non-convex (optimum at extreme points; cf. privacy funnel). |
| Decoder refit in the task guard (privacy route) | `{q: min_dec L ≤ c}` = union of half-spaces; non-convex. |
| Alternating LP ↔ decoder refit | Block-coordinate descent on a jointly non-convex problem. Monotone in the fitted objective **only if** the bank is fixed and each refit is an exact in-family minimizer on the same rows with the previous decoder in the family; rebasing, regularized/early-stopped fits, or different rows void monotonicity. |
| Policy learning, context learning, φ·A jointly | Costs depend non-linearly/combinatorially on d_m, φ_k; φ_k·A[k,m] is bilinear. Non-convex. |
| Per-state eta as `eta_t · A` | Bilinear (see 2.4). |

### 2.4 Is one shared eta a restriction? Yes — and it matters.

Affine consistency forces the A-mass received by x, Σ_k φ_k(x)Σ_m A[k,m], to equal 1−Σ_z B[T(x),z].
With contexts shared across states, eta must be constant on every connected component of the bipartite
state–context support graph; with contexts spanning all states it is global. Per-state eta is only
expressible bilinearly (`eta_t·A`) or by per-(t,k) A (32·K·M parameters; no pooling).

**Counterexample (exact, script §C).** States t1={x0,x1}, t2={x2,x3}, weights ¼, tokens {0,1,2},
frozen costs rows (0,½,½),(0,½,½),(⅖,0,1),(⅖,1,0). D17≡0 is the rowwise T-minimizer (t2 row costs
⅖ < ½). d_new=(1,2,1,2). C(D17)=1/5, C(d_new)=1/4. Nested K=1, bank {D17,d_new}: q = B[T] +
A_D17 δ_D17 + A_new δ_new, optimal B rows are D17 rows, so value = min_{A_new∈[0,1]}
(1−A_new)·C(D17)+A_new·C(d_new) = **1/5** (confirmed by vertex enumeration and HiGHS). Pointwise
target "D17 on t1, d_new on t2" has value **0**. K=2 contexts aligned with T: 0; K=2 contexts *crossing*
T ({x0,x2},{x1,x3}): 1/5; K=1 with the switched column (0,0,1,2): 0.

*Does D17-as-a-column fix it?* No (it is already in the bank above). It only lets A-mass reproduce
**one-hot D17** B-content per *context*; it cannot reproduce a stochastic B row, and cannot vary the
D17 share within a context. *Minimal fixes (pick ≥1, register before fitting):*
(a) make every non-D17 column **D17-anchored**: a cost-sensitive policy whose default is D17(x) and
which deviates only where its fitted cost improves (T(x) is a legal policy input); (b) include
such anchored policies at 2–3 fixed Lagrangian prices (Agarwal-style best responses), because the
LP can only choose *how much*, not *where*; (c) choose K>1 contexts so that the region where a
column helps is a union of contexts (context-specific benefit then is a real ablation). State in
METHOD.md that at K=1 the mixture adds only global time-sharing to its columns.

### 2.5 Aliasing and the correct nonalias metric

Alias directions (exact, script §N): for any **T32-measurable** column m (D17 or any function of T),
`dB[t,d_m(t)] = −1 ∀t, dA[k,m] = +1 ∀k, d eta = +1` leaves q unchanged. Others: two columns equal on a
context's support; global token relabels of a T32 kernel (still a T32 kernel). Hence "eta>0" or
"A-mass off D17" proves nothing. Example A: eta=1 and eta=3/4 emit the same q (script §V).

Recommended metric (on admitted rows, fixed tolerance set from replay parity before outcomes):
- q̄_t = Σ_{i∈t} w_i q(·|x_i)/W_t (state mean; itself a T32 kernel — the **T32 projection** Π q).
- V = Σ_t Σ_{i∈t} w_i TV(q(·|x_i), q̄_t) (weighted and unweighted), max_{i,j∈t} TV, histogram of
  per-state V_t, and **households** with TV(q_i, q̄_{T_i}) > tol.
- **Decision value of variation:** re-score Π q with the same frozen decoder and bank, and in the
  audit with refit probes/attackers. A candidate is a *new release* iff V>tol; it *uses* its extra
  capacity iff Π q is materially worse on the declared tradeoff. Example A: V=0.12, max TV 0.75, Π q =
  (0,0,⅖,⅖) keeps task information 0.08972 but is not private.
- Report token-law identity separately from all-input function identity (dataset vs deployment alias).

## 3. Finite-bank reference convention

### 3.1 Witnesses
rho_r = min_{a∈B_r} L_a(D17), constraint L_a(Q) ≥ rho_r − δ ∀a∈B_r. **(i)** B=D17, A=0, eta=0 ⇒ Q=D17
⇒ L_a(Q)=L_a(D17) ≥ rho_r ≥ rho_r−δ. **(ii)** B=0, A[k,D17]=1 ∀k, eta=1 ⇒ q(z|x)=Σ_kφ_k(x)1{D17(x)=z}
= 1{D17(x)=z}, also with soft φ (exact in script). Both require the D17 column to be the *same*
function (T32 replay, tie rule) on the *same* rows/weights/class order as rho.
**H-only attackers** have L_a(Q) ≡ L_a(D17) ≥ rho: they can never make the witness infeasible when
evaluated on the same rows; an H-only infeasibility therefore diagnoses a row/weight/denominator
mismatch, exactly as the handoff says.

### 3.2 Pitfalls found
1. **Rebasing relaxes** (script §P). Old bank {a1}: rho=0.7584; add a2 (stronger on D17): rho=0.5004.
   A candidate with L_a1=0.6380, L_a2=1.3322 is infeasible before and feasible after (a1 is deliberately
   poorly calibrated; the geometry, not realism, is the point). Feasible sets under
   successive banks are not nested. Save (old rho, new rho, bank hash, per-attack reference losses).
2. **Min-bank semantics are weak per attacker.** L_a(Q) ≥ rho−δ allows any individual attacker to gain
   up to L_a(D17)−rho+δ. The per-attacker reference L_a(Q) ≥ L_a(D17)−δ is stricter and still has the
   D17 witness; consider it as a registered sensitivity analysis.
3. **Float witness at δ=0.** With soft φ, Σ_kφ_k=1±1e−16; the witness can violate a δ=0 floor by
   rounding. `reference.calibrate_reference` uses 1e−10 witness tolerance and PRIMAL_TOL=1e−7 — keep
   an explicit tolerance and never register δ=0 exactly.
4. **Unseen-token pricing.** Frozen attackers/decoders must be defined for all 17 tokens × H. Tokens
   rare under D17 in an H region get arbitrary (possibly confidently wrong) attacker outputs; the LP
   will buy privacy there. Clip attacker probabilities (log loss unbounded otherwise); fit column-wise
   best-response attackers on every d_m release before solving; re-score on refit attackers.
5. **In-sample costs.** If C, c^B are computed on the rows used to fit the decoder, the LP exploits
   decoder memorization (task optimism). Cross-fit decoder/attacker costs vs coefficient rows.

### 3.3 Frozen vs refit (in-sample, fixed family, same rows)
- Any fixed decoder/attacker loss is ≥ its in-family refit loss on the same rows. So frozen **task**
  loss is an upper bound (conservative); frozen **attack** loss is an upper bound on the best attacker's
  loss (anti-conservative for privacy). Script §S: overstatement +0.0027/+0.0105/+0.0399 nats at
  λ=0.05/0.1/0.2.
- **Concavity certificate.** For a fixed family F, L*_F(q)=min_{a∈F}L_a(q) is concave, so for
  q=Σ_jλ_j q_j: L*_F(q) ≥ Σ_jλ_j L*_F(q_j). With the Carathéodory decomposition of §5 into ≤ r+1
  deterministic components, refitting attackers on each component gives a *conservative* in-sample
  lower bound on the mixture's attack loss. It is sufficient, not necessary: in Example A both
  components leak and the mixture is exactly private, so the final audit must refit on the mixture.

## 4. Degrees of freedom

| Block | Free parameters (given frozen φ, d) |
|---|---|
| B (32 states × 17 tokens, row sums 1−eta) | 32·16 = 512 |
| A (K × M, row sums eta) | K(M−1) |
| eta | 1 |
| **Polytope dimension** | 512 + K(M−1) + 1: K=1,M=6 → 518; K=4,M=6 → 533 |
| **Identifiable rank of (B,A,eta)→q** | minus one per T32-measurable column (generic φ, d): 517; 532; 531 with a second T32-measurable column (numerical rank on a 4000-row synthetic instance) |

The "additional" count is K(M−1)+1 in the interior but only K(M−1) identifiable when D17 is a column
(and ≤ that with more aliases). This is conditional on frozen policies and contexts. Separately report
the fitted complexity of each policy (tree leaves/splits or network weights, and the pricing inputs it
saw), of φ (boundaries/normalizers), and of the decoder/attacker families; and note that the LP
chooses among ~17^32 T32 vertices, so selection complexity on coefficient rows is not "21 parameters".

## 5. Deterministic releases, vertices, and when randomness can matter

**Emitted determinism (exact conditions).** q(·|x) is one-hot at z*(x) iff B[T(x),z]=0 ∀z≠z*(x) and
d_m(x)=z*(x) for every (k,m) with φ_k(x)A[k,m]>0.
- eta=1: hard φ + one-hot A ⇒ deterministic (sufficient); soft φ: deterministic iff at each x all
  positively weighted (context, column) pairs agree at x.
- eta=0: deterministic iff every visited B row is one-hot.
- **0<eta<1 (lemma):** B[T(x)] carries mass 1−eta>0 and is constant on the state, so a deterministic q
  forces B[t]=(1−eta)δ_{z_t} and q(·|x)=δ_{z_t} for all x∈t: the release is a deterministic **T32**
  function on those rows. So the only deterministic non-T32 members of the nested class are compact
  (eta=1) switched policies. The exact same-bank deterministic optimum is therefore
  min(best deterministic T32 kernel [MIP], best switched policy). For hard contexts the latter is
  **enumerable**: M^K = 6^4 = 1296 linear evaluations per cost vector — no MIP needed.

**Vertex counting (precise).** Let 𝒫 be the free-join polytope and add r linear inequalities (all
cuts incl. any task guard). A vertex v of 𝒫∩{cuts} lies in the relative interior of a face F of 𝒫
with dim F ≤ (number of cuts tight at v) ≤ r. Faces of 𝒫 are joins of faces G₀⊆𝒫₀, G₁⊆𝒫₁ (either may
be empty), dim = dim G₀ + dim G₁ + 1 (both nonempty). With excess e(B)=Σ_t(|supp B_t|−1),
e(A)=Σ_k(|supp A_k|−1):
- eta=0: e(B) ≤ r — at most r stochastic T32 states;
- eta=1: e(A) ≤ r — at most r contexts with fractional rows;
- 0<eta<1: e(B)+e(A) ≤ r−1; for r=1, q = (1−eta)·(deterministic T32 kernel) + eta·(switched policy).
By Carathéodory v is a mixture of ≤ dim F + 1 ≤ r+1 vertices of 𝒫 (deterministic releases for hard
contexts). This is the LP analogue of the few-classifier randomized solutions in reductions approaches;
I state it as a standard LP fact, not as a result attributed to Agarwal et al.

**What it predicts.** Randomization in the frozen LP can move the objective and each cut by at most
W_frac·range(ℓ), W_frac = weight of people in fractional blocks — small for a few of 32 T32 states,
**not** small for one of K=4 contexts (~25% of people) or for any interior eta (everyone whose two
components differ). Example A shows one fractional row can be decisive (0 → 0.0897 nats) under an
exact-equality constraint. Implementation: use a simplex/crossover solver so a vertex is returned, and
log e(B), e(A), eta, and W_frac for every solve; an interior-point solution without crossover can be
more randomized than necessary.

## 6. Additional provable facts the design should use

6.1 **Task sufficiency.** If Y ⟂ X_A | (T(x), H_A(x)) then for every decoder,
E ℓ(Y, dec(H_A,Z)) depends on q only through q̄(·|t,h) (cell average), so within-(T,H_A) variation
cannot change population task loss; only privacy can change. In-sample, such variation that lowers
task loss is fitting label noise. Example A instance: Π q has the same 0.08972 nats. (For the coalition
privacy side no such invariance holds.)
6.2 **Persistence.** One persistent draw per record; two fresh draws break Example A's exact privacy.
6.3 **Codebook.** Mixing a policy with a token-relabelled copy is randomized response: a genuine
garbling, mispriced by both frozen decoder (overstated cost) and frozen attackers (overstated
privacy). Align codebooks on fitting rows before solving (handoff §8), and treat relabels as aliases.

## 7. Prior work

| Ingredient | Closest prior source (verified what?) | Adaptation here | Ablation needed |
|---|---|---|---|
| Richer vs restricted observation for perfect privacy; nullspace feasibility; LP for finite alphabets | Rassouli & Gündüz, *On Perfect Privacy*, arXiv 1712.08500 — read: Prop. 1 (feasible iff dim Null(P_{X|W})\Null(P_{Y|W}) ≠ 0, general W), output-perturbation vs full-data-observation models, Thm 1 LP with \|U\| ≤ nul(P_{X\|Y})+1 (their X private, Y useful) | Example A = Prop. 1 with W=X vs W=T; ACS uses fitted attackers, not exact independence | Parent-restricted T32 control vs richer class |
| Randomized mixtures of a policy bank from cost-sensitive best responses under empirical constraints | Agarwal et al. 2018, *A Reductions Approach to Fair Classification*, ICML — read: randomized classifier Q over H, Lagrangian saddle point, exponentiated gradient, grid search, cost-sensitive oracle | Bank of 17-token policies; LP over mixing; contexts extend Agarwal's global Q (=K=1) | K=1 vs K=4; best deterministic switched policy; column-generation parity for controls |
| Learned randomized privatizer vs adversary | Tripathy, Wang, Ishwar, *Privacy-Preserving Adversarial Networks*, arXiv 1712.07008 — abstract read: adversarially trained randomized mechanisms, variational MI privacy, distortion constraint | Standard adversarial categorical 17-token comparator | Strong adversarial control with same inputs |
| Sequential releases with prior outputs and colluding recipients | Taylor, Vippathalla, Coon, *Adaptive Privacy of Sequential Data Releases Under Collusion*, arXiv 2601.21859 (v1 29 Jan 2026, v2 10 Jul 2026) — abstract read: MI privacy, distortion/MI utility, Blahut–Arimoto-style adaptive algorithm, optimal for distortion utility | Z appended to frozen H_A,H_B; coalition view (H_A,H_B,Z) | Coalition vs local-only constrained variants |
| Log-loss adversary ⇒ information leakage; privacy–utility mapping design | Calmon & Fawaz, *Privacy Against Statistical Inference*, Allerton 2012 (arXiv 1210.2123) — abstract read | Cross-entropy attackers as finite-bank log-loss proxies (no MI bound) | none (framing); do not import guarantees |
| Non-convexity of privacy–utility under log loss | Makhdoumi, Salamatian, Fawaz, Médard, *From the Information Bottleneck to the Privacy Funnel*, ITW 2014 (arXiv 1402.1774) — abstract read | §2.3: decoder refits make the problem concave-minimization | report frozen vs refit decoder gap |
| Time-sharing / LP vertex sparsity / Kelley cutting planes | Standard convex analysis and LP theory | §3.3, §5 | exact deterministic comparator |
| With/without direct access to private data | Zamani, Oechtering, Skoglund, arXiv 2212.12475 — **title seen in search only, not read** | possibly closest to "legal predicted risk vs true S" | — |

Not verified here: Erdogdu–Fawaz and Diaz et al. statements (defer to the pinned guarantee review).

## 8. Claims ledger

**(a) Established prior work.** Perfect-privacy feasibility via nullspaces and richer observation
(Rassouli–Gündüz); randomized solutions from cost-sensitive reductions (Agarwal et al.); adversarially
trained randomized privatizers (Tripathy et al.); sequential/collusion releases (Taylor et al.);
convexity of MI in the channel / concavity of conditional entropy; time-sharing; LP vertex sparsity;
cutting-plane methods; non-convexity of privacy funnels.

**(b) Implementation properties provable here (exact-law or algebraic; checked in the script).**
P1–P5; free-join structure and vertex set; D17 witnesses (both representations, soft φ); H-only cuts
never infeasible on consistent rows; alias directions and rank 518/533 → 517/532; emitted-determinism
conditions and the 0<eta<1 lemma; vertex/face counting and Carathéodory ≤ r+1; frozen-decoder
conservative / frozen-attacker anti-conservative (in-sample, in-family); concavity certificate;
shared-eta counterexample; task-sufficiency invariance; rebasing non-monotonicity; fresh-draw failure.
None of these is a population privacy guarantee.

**(c) Empirical — need this experiment.** Whether ACS richer policies change decisions within T32
(V>tol, households affected); whether X_A carries task information beyond (T,H_A) (task-only richer
policy vs D17 on checking rows); whether any LP gain survives refit and fresh auditors; whether K=4
beats K=1; whether randomization beats the best deterministic switched policy/strong deterministic
controls; whether gains exceed the §16 margins under household resampling on all three anchors.

## 9. Recommended registration changes

1. Bank columns: D17 plus **D17-anchored** cost-sensitive policies (task-only; local-risk; coalition-risk)
   at 2–3 fixed prices; drop unanchored columns or add their switched versions.
2. Nonalias gate = V, affected households, and the T32-projection decision value — never eta.
3. Log e(B), e(A), eta, W_frac per solve; require a vertex solution.
4. Deterministic comparator for hard contexts: exhaustive M^K switched-policy search (exact).
5. Every LP candidate re-scored with attackers refit on the candidate; report frozen vs refit gap;
   optionally the component-refit concavity lower bound.
6. Preflight for route U: task-only richer policy must show task gain over D17 on checking rows;
   otherwise record that richer inputs can only serve route P (§6.1).
7. Rebase ledger with rho history; δ>0 with explicit float tolerance; clip attacker probabilities;
   cross-fit costs; persistent per-record draws.
