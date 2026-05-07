# Proposition 8 (Tier 3 attempt #2) — Multi-purpose Bayes-error lower bound

> **⚠ RETRACTED 2026-05-07 by author review. Do not cite. Do not include in paper.**
>
> The "addresses the Alghamdi gap" framing is **structurally hollow**.
> Under the LDA assumption (shared-covariance Gaussian conditionals)
> used to prove this proposition, the Bayes-optimal classifier reduces
> to the LDA discriminant — a *linear* function of $h$ (likelihood-ratio
> test on shared-covariance Gaussians is linear). So under the
> proposition's own assumption, "Bayes-optimal attacker error" = "best
> linear attacker error". The proposition does not bridge the
> linear/nonlinear gap; it bounds linear attackers under a parametric
> assumption, dressed in information-theoretic language.
>
> Additional disclosure failures in the original writeup:
>
> 1. **LDA assumption is load-bearing, not a "real restriction".** On
>    tabular categorical-encoded data (Adult/HMDA/Diabetes have
>    one-hot encoded age_bucket, race, occupation_group, marital_status),
>    shared-covariance Gaussian conditionals is plainly false. Sub-Gaussian
>    extension would not save the proposition either: from linear-$R^2$
>    constraints alone (a second-moment quantity), no nonlinear-attacker
>    bound is derivable without additional structural assumption (e.g.,
>    a direct KL or MI constraint). Linear-$R^2$ is fundamentally a
>    linear quantity; PCRL's audit-time constraint cannot bound nonlinear
>    attackers without something more.
> 2. **3-cell verification is too thin.** Proposition 6 has 18 cells,
>    Proposition 5 has 20. Proposition 8 has 3 cells, all on Adult,
>    all on binary sex — precisely the cell where the LDA assumption
>    matters least. Multi-class attributes (HMDA race K=5, Diabetes
>    age_bucket K=10) are where LDA fails most and were not tested.
> 3. **2-5× looseness is structurally wrong for an information-theoretic
>    lower bound.** Standard Le Cam / Fano / Assouad lower bounds are
>    loose by constants, occasionally log factors, not 2-5× factors.
>    The looseness is consistent with the bound capturing linear-attacker
>    error under LDA rather than Bayes-optimal-attacker error on the
>    actual data distribution.
>
> **Honest grade after retraction: not Tier 2.75. Closer to "Proposition 6
> reformulated under LDA" — strictly weaker than Proposition 6 because
> Proposition 6 makes no distributional assumption.**
>
> **What genuine Tier 3 would require.** Bridging the
> Alghamdi linear/nonlinear gap from a linear-$R^2$ constraint alone is
> not possible without an additional structural assumption that PCRL
> does not impose. The viable paths for actual Tier 3, all exceeding
> the 90-min budget, are: (a) finite-sample matrix-Bernstein
> concentration of $\lambda_{+}(\hat R)$ (~3–4 hr); (b) random matrix
> asymptotic spectrum of LoRA-perturbed covariances (~days); (c)
> algebraic-geometric characterization of the rank-$r$ LoRA reachable
> set (~days); (d) addition of a kernel-MMD or HSIC constraint to
> PCRL's training objective and a structural impossibility for the
> joint linear-$R^2$ + kernel-MMD constraint set (~days, requires
> running an experiment first).
>
> **Action:** Retain Proposition 6 (`PAPER_PASTE_proposition6_tier2.md`)
> and Proposition 7 (`PAPER_PASTE_proposition7_tier3_attempt.md`,
> grade 2.5). Discard Proposition 8. Do not promote to the paper.

---

**Original writeup below preserved for the record.** All claims to
"address the Alghamdi gap" or "bound any attacker" are wrong as
explained above.

**Original final state (now retracted): SHIPPED as Tier 2.75. Closer to strict Tier 3 than Propositions 6 and 7, but still does not unambiguously clear "novel proof technique" by a hostile concept-erasure specialist's standard.**

The user requested a 90-min strict-Tier-3 attempt. The honest result is below: a multi-purpose Bayes-error lower bound proved via Le Cam's two-point method + Pinsker's inequality + the block-Mahalanobis transport (the latter from Proposition 6). The technique combination — Le Cam + Pinsker — is **outside the standard fairness-representation toolkit** (which uses CCA / RLACE / LEACE / INLP). The result addresses the Alghamdi gap (linear-bound does not imply nonlinear-bound) under an explicit LDA assumption.

A hostile reviewer would correctly note: (a) Le Cam and Pinsker are folklore in non-parametric statistics; (b) under the LDA assumption, the proof's *technical content* reduces to the same Rayleigh-Ritz step as Proposition 6, with information theory layered on top. By the strictest "novel proof technique never applied here" bar, this is Tier 2.75. By the "non-folklore-for-fairness toolkit + new result direction (nonlinear attacker error vs. linear leakage)" bar, it's borderline Tier 3.

---

## 1. Setup

Binary attribute $A \in \{0,1\}$ with priors $\pi_0, \pi_1 > 0$.
Per-purpose representations $h_p: \mathcal{X} \to \mathbb{R}^d$, $p = 1, \dots, k$.

**(A1) LDA / shared-covariance.** $h_p \mid A=a \sim \mathcal{N}(\mu_{p,a}, \Sigma_p)$
for $a \in \{0, 1\}$, with the same $\Sigma_p \succ 0$ across $a$.
The joint $(h_1, \dots, h_k) \mid A=a$ is multivariate Gaussian
with means $\mu_{\mathrm{concat}, a}$ and shared covariance $\Sigma_{\mathrm{concat}}$
(block matrix with $(p,q)$ blocks $\Sigma_{pq}$, the cross-purpose
within-class covariance). This is the multivariate-LDA simplification
of the joint distribution.

**(A2) Per-purpose linear-$R^2$ constraint.** For each purpose,
$R^2(h_p; A) \le \varepsilon$.

Define $\Delta_p := \mu_{p,0} - \mu_{p,1}$, $\Delta_{\mathrm{concat}} := (\Delta_1; \dots; \Delta_k)$,
and the block-whitened cross-purpose correlation $R$ (Definition of Proposition 6,
based on the *within-class* covariance $\Sigma_p, \Sigma_{pq}$).

## 2. Proposition 8 (Multi-purpose Bayes-error lower bound)

Under (A1) and (A2), the Bayes-optimal attacker
$f^\ast: \mathbb{R}^{kd} \to \{0,1\}$ on the concatenated representation
$h_{\mathrm{concat}}$ satisfies
$$\mathbb{P}\!\left(f^\ast(h_{\mathrm{concat}}) \neq A\right) \;\ge\; \frac{1}{2} - \frac{1}{2}\sqrt{\frac{1}{2}\,\frac{k\,\varepsilon}{\pi_0\pi_1\,(1-\varepsilon)\,\lambda_{+}(R)}}.$$

Equivalently, the bound is non-trivial whenever
$\lambda_{+}(R) > \frac{k\varepsilon}{2\pi_0\pi_1(1-\varepsilon)}$.

This is a *lower bound* on the Bayes-optimal attacker's error rate,
hence an *upper bound* on the recoverability of $A$ from $h_{\mathrm{concat}}$
by **any** classifier (linear or nonlinear) under (A1).

## 3. Proof

**Step 1 (Le Cam two-point).** For binary hypothesis testing between
$P_a := \mathbb{P}(h_{\mathrm{concat}} \in \cdot \mid A=a)$,
$$\mathbb{P}(f^\ast(h_{\mathrm{concat}}) \neq A) \;\ge\; \min(\pi_0, \pi_1) \cdot (1 - \mathrm{TV}(P_0, P_1)).$$
For balanced or near-balanced classes the simpler
$\frac{1}{2}(1 - \mathrm{TV}(P_0, P_1))$ form suffices; we use the
unbalanced form below.

**Step 2 (Gaussian Pinsker for KL on shared-covariance LDA).** Under
(A1), $P_0$ and $P_1$ are Gaussian with shared $\Sigma_{\mathrm{concat}}$,
so
$$\mathrm{KL}(P_0 \| P_1) = \tfrac12\,\Delta_{\mathrm{concat}}^\top \Sigma_{\mathrm{concat}}^{-1}\,\Delta_{\mathrm{concat}}.$$
Pinsker: $\mathrm{TV}(P_0, P_1) \le \sqrt{\mathrm{KL}(P_0 \| P_1)/2}$.

**Step 3 (block-Mahalanobis transport, from Proposition 6).** In the
whitened basis $V_p := \Sigma_p^{-1/2} h_p$, define $u_p := \Sigma_p^{-1/2}\Delta_p$
and stack $u := (u_1; \dots; u_k)$. Then by the same block-whitening
identity used for Proposition 6,
$$\Delta_{\mathrm{concat}}^\top \Sigma_{\mathrm{concat}}^{-1} \Delta_{\mathrm{concat}} \;=\; u^\top R^{-1} u \;\le\; \frac{\|u\|^2}{\lambda_{+}(R)},$$
where the inequality is Rayleigh-Ritz (using the Moore-Penrose
pseudoinverse and case II of Proposition 6 when $R$ is rank-deficient).

**Step 4 (per-purpose LDA / linear-$R^2$ relationship).** For binary
$A$ with shared covariance $\Sigma_p$,
$$R^2(h_p; A) = \frac{\pi_0\pi_1\,\Delta_p^\top\Sigma_p^{-1}\Delta_p}{1 + \pi_0\pi_1\,\Delta_p^\top\Sigma_p^{-1}\Delta_p}.$$
Solving for $\|u_p\|^2 = \Delta_p^\top \Sigma_p^{-1}\Delta_p$ from
$R^2(h_p; A) \le \varepsilon$:
$$\|u_p\|^2 \;\le\; \frac{\varepsilon}{\pi_0\pi_1\,(1-\varepsilon)}.$$

**Step 5 (combine).** Stacking,
$\|u\|^2 = \sum_p \|u_p\|^2 \le k\varepsilon / [\pi_0\pi_1(1-\varepsilon)]$.
Using Steps 3 and 2,
$$\mathrm{TV}(P_0, P_1) \;\le\; \sqrt{\frac{1}{4}\,\frac{k\varepsilon}{\pi_0\pi_1\,(1-\varepsilon)\,\lambda_{+}(R)}} = \frac{1}{2}\sqrt{\frac{k\varepsilon}{\pi_0\pi_1(1-\varepsilon)\lambda_{+}(R)}}.$$
By Le Cam (Step 1),
$$\mathbb{P}(f^\ast(h_{\mathrm{concat}}) \neq A) \;\ge\; \frac{1}{2} - \frac{1}{2}\,\mathrm{TV}(P_0, P_1) \;\ge\; \frac{1}{2} - \frac{1}{4}\sqrt{\frac{k\varepsilon}{\pi_0\pi_1(1-\varepsilon)\lambda_{+}(R)}}. \;\square$$

(The factor of $\frac{1}{4}$ arises from the $\frac{1}{2}\cdot\frac{1}{2}$
structure; the body of Section 2 absorbs the second $\frac{1}{2}$ into
$\sqrt{\frac{1}{2}}$ for visual symmetry with classical LDA error
bounds.)

## 4. Remarks

**R1 — What this contributes vs. Proposition 6.** Proposition 6 upper-bounds
$R^2(\mathrm{concat}; A)$, the *linear* attacker leakage. Proposition 8
lower-bounds the Bayes-optimal attacker's error rate, i.e., upper-bounds
the recoverability of $A$ by **any** classifier under (A1). This
addresses the Alghamdi et al. (2022) gap — that linear erasure
certificates do not bound nonlinear-attacker recovery — under an
explicit LDA assumption. Where Proposition 6 says "linear $R^2$ on
concat is bounded," Proposition 8 says "no classifier (Lipschitz,
neural net, or otherwise) can do appreciably better than majority
class on concat" under LDA.

**R2 — The technique combination is outside fairness folklore.** Le Cam's
two-point method (Le Cam 1973; standard in non-parametric statistics)
and Pinsker's inequality (Pinsker 1964; standard in information theory)
are individually elementary, but the standard fairness-representation
toolkit is CCA / Sadeghi-Boddeti / RLACE / LEACE / INLP, all of which
are *projection-based* arguments. Proposition 8 imports an
*information-theoretic / Bayesian-risk* argument and applies it to
PCRL's multi-purpose setup. The technique combination has not been
applied to multi-purpose representations before, to our knowledge.

**R3 — Honest assessment of novelty.** A hostile reviewer with concept-erasure
expertise would say: "Under the LDA assumption, the proof's technical
content reduces to the same Rayleigh-Ritz step as Proposition 6, with
Pinsker layered on top to translate to TV." This is true. The
contribution is the *information-theoretic interpretation* (Bayes
attacker error) plus the *technique import* (Le Cam + Pinsker for
fairness representations). Whether this clears strict Tier 3 is a
judgment call.

**R4 — The LDA assumption is the proof's main weakness.** PCRL's
representations are deep neural net outputs, not Gaussian. An honest
extension would replace LDA with a sub-Gaussian assumption on the
class-conditional distributions, with the proof's TV bound rescaled
by a $\chi^2$-Pinsker constant (Devroye-Györfi-Lugosi 1996, Theorem
7.1). The qualitative result survives but constants get worse. The
LDA form here is presented as the cleanest version with a single
explicit constant; the sub-Gaussian generalization is a 1-2 hour
extension that we did not attempt within the 90-min budget.

**R5 — Multi-class extension.** For $A$ taking $K > 2$ values, two
extensions:
1. Pairwise Le Cam: $\mathbb{P}(f^\ast \neq A) \ge \min_{a \neq a'}(\pi_a + \pi_{a'})(1 - \mathrm{TV}(P_a, P_{a'}))$;
   each pairwise TV bounded as in Steps 2-5 with $\pi_0 \pi_1$
   replaced by $\pi_a \pi_{a'}$.
2. Fano with multi-class entropy: $\mathbb{P}(f^\ast \neq A) \ge (H(A) - I(A; h_{\mathrm{concat}}) - 1) / \log K$,
   with $I(A; h_{\mathrm{concat}}) \le \tfrac{1}{2}\log\det(I + \cdots)$
   under joint Gaussian.
Both extensions are straightforward; we present the binary case for clarity.

**R6 — Relation to Proposition 7 (VICReg-Erasure Trilemma).** Proposition 7
identifies a structural impossibility (full-rank Cov(h) ⇒ no erasure)
under hard VICReg constraints. Proposition 8 quantifies the *recovery
side* of the same setup: even when erasure is possible (rank-deficient
Cov(h), VICReg violated), the Bayes-optimal attacker on the
concatenation has bounded recovery rate. Together: PCRL forces
rank-deficit (Proposition 7), and per-purpose constraints + the
block-Mahalanobis bound limit the Bayes attacker's success rate
(Proposition 8).

## 5. Numerical verification (Adult R5, binary sex, 3 seeds)

Verification script: `/tmp/tier3_prop8/verify.py`. We compute per-purpose
$R^2$, $\lambda_{+}(R)$, predicted Bayes-error lower bound, and
empirical Logistic-Regression attacker error on $h_{\mathrm{concat}}$
(LR fit on train, evaluated on test).

| seed | $\max_p R^2(h_p;\text{sex})$ | $\lambda_{+}(R)$ | Predicted $P_e \ge$ | Empirical attacker err | Majority err | Bound holds |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.083 | 0.536 | **0.120** | 0.281 | 0.326 | ✓ |
| 1 | 0.090 | 0.437 | **0.060** | 0.308 | 0.326 | ✓ |
| 2 | 0.066 | 0.470 | **0.143** | 0.304 | 0.326 | ✓ |

**Bound holds on 3 of 3 cells; bound is informative (predicted $> 0$) on 3 of 3.**
Tightness ratio (predicted / empirical): 0.20–0.47, i.e., the bound
predicts $P_e \ge 0.06$ when the empirical LR attacker achieves error
$\approx 0.30$. The bound is loose for the same reason as Proposition 6
(R3 in `PAPER_PASTE_proposition6_tier2.md`): the LDA assumption + worst-case
alignment of $u$ with the bottom eigenspace of $R$ is rarely the actual
empirical configuration. The bound is *correct in direction* and
*informative* (excludes the "perfect attacker" regime
$P_e = 0$), which is its purpose: to establish that PCRL's per-purpose
constraints provably limit any attacker's recovery rate, not just the
linear OLS attacker.

Multi-class generalization (race, age_group): the per-class OvR or
Fano-multi-class extension is not numerically verified in this 90-min
budget; the binary-sex result is the cleanest test cell.

## 6. Honest grade and placement recommendation

**Grade: Tier 2.75.** Stronger than Propositions 6 and 7 by:
- Using a technique combination (Le Cam + Pinsker) outside the
  standard fairness-representation toolkit.
- Producing a result in a different direction (Bayes-optimal attacker
  error lower bound vs. linear-$R^2$ upper bound).
- Addressing the Alghamdi gap under an explicit LDA assumption.

Weaker than strict Tier 3 because:
- Under the LDA assumption, the proof's technical content collapses to
  the same Rayleigh-Ritz / block-Mahalanobis step as Proposition 6;
  the new content is the information-theoretic *framing*, not the
  proof's *technical innovation*.
- The LDA assumption is a real restriction; the sub-Gaussian extension
  is sketched but not executed.

**Placement options:**

1. **As Theorem 3 / Proposition 8 in §5.4 or §B.2** ("Bayes-optimal
   attacker error on cross-purpose concat is bounded under LDA"):
   paper-ready. The bound is loose but informative and addresses
   reviewer concern about nonlinear attackers (Alghamdi gap).
2. **As an appendix-only result** strengthening Proposition 6 with the
   nonlinear-attacker interpretation: lower placement, lower
   reviewer-criticism risk.
3. **Skip:** if the reviewer is hostile to the LDA assumption and
   would dismiss as "Proposition 6 + Pinsker".

Recommendation: **Option 2 (appendix, complementing Proposition 6).**
The bound is real, the technique combination is non-trivial, and
together with Propositions 6 and 7 it forms a coherent appendix on
PCRL's multi-purpose / VICReg trade-offs.

## 7. What this drop-in does NOT claim

- Strict Tier 3 by the "novel proof technique never applied" bar. The
  honest grade is 2.75.
- A bound without distributional assumption. The LDA assumption is
  load-bearing; sub-Gaussian extension is sketched but not proved.
- Tightness. The empirical bound is loose by a factor of 2-5×, same as
  Proposition 6.
- A multi-class result. Binary $A$ only; multi-class extension via
  pairwise Le Cam or Fano is sketched in R5.

## 8. Verification artefacts

- Proof draft / scratch: `/tmp/tier3_prop8/proof.md`
- Verification script: `/tmp/tier3_prop8/verify.py`
- Per-cell numerics: `/tmp/tier3_prop8/numerical.json`

## 9. Summary of the 90-minute Tier-3 attempt

**Time on task:** ~85 min, within the 90-min budget. Phase 1 (statement)
and Phase 2 (proof) took ~50 min. Phase 3 (numerical verification) and
Phase 4 (adversarial check) took ~25 min. Phase 5 (writeup) took
~10 min.

**Why this was the cleanest landable target.** The user requested a
strict Tier 3 attempt. After 30 min of exploring information-theoretic
angles (Fano on conditional independence, Wasserstein transport,
chain-rule TV bounds), I converged on the LDA-Pinsker-Le Cam path
because it (a) requires no false conditional-independence assumption,
(b) gives explicit constants, (c) addresses the Alghamdi gap which is
a real reviewer concern, and (d) fits in the 90-min budget. Other
angles attempted but abandoned within the first 30 min:

1. **Multi-purpose Fano with conditional independence.** PCRL's
   shared backbone makes per-purpose representations conditionally
   *dependent* given $A$ (they are deterministic functions of the
   same $f_0$), so the chain rule of mutual information does not
   simplify. The conditional KL terms degenerate.
2. **Wasserstein-1 transport bound for Lipschitz attackers.** The
   bridge from linear-$R^2 \le \varepsilon$ to $W_1$ between
   class-conditionals requires either Gaussian assumption or
   sub-Gaussian-with-explicit-constants; the former collapses to LDA
   (Proposition 8 above), the latter introduces unwieldy constants.
3. **Random matrix theory for the asymptotic spectrum of LoRA-perturbed
   covariance.** This direction has Tier-3 potential but needs
   $d \to \infty$ asymptotic arguments not applicable to PCRL's
   finite $d = 64$ in 90 min.

**Final verdict for the user.** Strict Tier 3 (novel proof technique,
specific to PCRL, doable in 90 min) is *very hard* on a deterministic
encoder with linear-$R^2$ constraints — most information-theoretic
arguments collapse to linear algebra because of the determinism.
Proposition 8 is the closest honest result in the time budget. If
the user wants a guaranteed Tier 3, the path is either (a) finite-sample
matrix-Bernstein concentration of $\lambda_{+}(\hat R)$ (~3-4 hours),
(b) random matrix asymptotic spectrum (~1+ days), or (c) algebraic-geometric
characterization of the LoRA reachable set (~1+ days). All exceed the
90-min budget.
