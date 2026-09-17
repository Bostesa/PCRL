# Contribution assessment

What this study actually contributes, relative to the prior work verified in
[`NOVELTY_MATRIX.md`](NOVELTY_MATRIX.md). The test applied throughout is not whether a word appears
in earlier papers, but whether **prior formulations can already express the construction** and
whether **their algorithms already solve it**.

## 1. Established prior work — claim none of it

The spectral trace step (Ky Fan), random Fourier features inside a closed-form solver (K-TOpt),
multiple weighted sensitive penalties (OptNet-ARL), eigenvalue-sign rank selection (SARL Thm 3,
OptNet-ARL Thm 4.1, K-TOpt Cor 4.1), a conditional dependence penalty inside such a solver (U-FaTE),
the conditional-independence moment family (KCI Lemma 2(v), RCoT Eq. 26), cross-fitting, the fact
that linear-certified erasure fails against nonlinear attackers (LEACE's own scoping; Elazar &
Goldberg 2018), and the existence of a utility/non-task-leakage floor (Stadler et al. 2024).

## 2. Adaptation — real engineering, not a new principle

**Conditioning on continuous released service probabilities.** U-FaTE's conditional measure is
stratification over a discrete label, and its own text scopes it to "when Y is not a continuous
label". `H_A` is two continuous probability vectors, so stratification is unavailable; the mechanism
substitutes cross-fitted nuisance residuals interacted with a degree-2 basis in `H`. Verified to be
exactly KCI Lemma 2(v) restricted to `f ∈ {Z_l b_k(H)}`, `g′ ∈ {1{S=c}}`. This is a competent
substitution of a known estimator into a known solver. It is not a new optimisation principle, and
the manuscript should not present it as one.

**Utility as reconstruction of a residualised teacher.** A design choice that operationalises "beyond
what H already explains". Prior work in this line measures utility as dependence with a label.

**Everything else in the mechanism is inherited.** Nothing in §1 is re-derived here.

## 3. Potentially distinct problem and evaluation contribution

Three things were **not found** in the surveyed prior work, and the negative findings are specific
enough to be checkable:

1. **A release constrained to reproduce an already-published output bitwise while adding capability
   beside it.** LEACE and SPLINCE *edit* a representation the releaser owns. SARL, K-TOpt and U-FaTE
   design a representation from scratch. Sankar et al. design one sanitised database from scratch and
   have no notion of a prior commitment. The constraint here is structural — the channel is appended,
   never mixed — which is why exact preservation is provable rather than measured.
2. **Recipient-specific *and* coalition-specific penalties, with only one recipient receiving a
   channel.** The multi-consumer information-theoretic line is weaker prior art than it looks:
   Sankar et al.'s "multiple legitimate information consumers" all consume *one* common release and
   differ only in private side information; there is no per-recipient release and no collusion term.
   OptNet-ARL's multiple λ index multiple *attributes*, not multiple *recipients*.
3. **An evaluation design that keeps the distinctions that usually get collapsed.** Fresh and frozen
   attackers never pooled; absolute and incremental recovery reported side by side; negative selected
   increments retained unclipped and explained; every fixed ancestor attack scored in its own right;
   H's own attack routed onto each augmented wire as a control; a locked temporal transport to a
   sealed survey year with the multiplicity fixed in advance; a randomized-withholding control with
   its arithmetic verified to 2e-16.

Of these, (3) is the one this study *demonstrates* rather than merely *proposes*. (1) and (2) are
problem-formulation contributions; the paper can state that prior formulations do not express them,
having checked, without claiming priority.

## 4. What remains unverified

* Whether any of the three differences in §3 is genuinely absent from the literature. A bounded
  search cannot show this; eight targeted searches plus the full-text reads in §1 found nothing, and
  that is the whole of the evidence.
* The formal statements of Stadler et al. (2024) and TAPPFL, read only at abstract level.
* U-FaTE's supplementary material, which the paper says holds the per-criterion solutions.
* Whether conditioning on continuous released predictions *helps* relative to any alternative. The
  study answers this only against its own local controls, and the answer is: it lowers measured
  recovery, at a residence cost the data cannot bound below 0.001 nats.

## 5. The honest framing

This is **not** a competitive-method paper. F3 fails on all four sensitive endpoints under both
weightings, and the negative verdict survives average-utility matching in both directions. Framing it
as a method paper would require the baselines in §6, which have not been run.

The defensible framing is an **empirical study of the value and limits of coalition-conditioned
incremental releases**, with three deliverables: (i) a structural guarantee — exact preservation of a
published service — that costs nothing statistically; (ii) a locked, sealed-year demonstration that a
coalition-conditioning knob behaves as designed, together with an explicit statement of what the
interval evidence does *not* establish about its utility cost; (iii) a negative method result with its
cause narrowed but not isolated. That is a real contribution and it is smaller than a method claim.

## 6. The smallest set of external baselines a competitive-method claim would need

**Not implemented here, and not to be launched by Terminal B.** Specified so a future authorised run
can be fair.

| Baseline | Why it is required | Exact adaptation needed | Fairness conditions |
|---|---|---|---|
| **SARL** (linear + kernel) | It is the direct ancestor; the marginal arms M025/M1 are already its construction up to parametrisation. Without it, "our penalty helps" has no method-level control | Utility side: substitute the residualised teacher `R` for SARL's label target. Fit on the same 2018 representation-fit pool, same frozen PCA32 teacher, same random-feature map | Same width; report SARL's own Thm-3 width alongside a width-16 truncation; identical attacker families and selection rule |
| **K-TOpt** | The published combination of RFF + whitening + closed-form spectral solution. It is the strongest closed-form unconditional competitor | Same substitution; use its generalised eigenproblem and Cor 4.1 width, plus a width-16 variant | As above; report the approximate-normality assumption as a stated condition, not a silent one |
| **U-FaTE** | The only prior conditional-penalty spectral method. It is the control that isolates *what is conditioned on* | Its conditioning is a discrete stratum. Two arms are needed: (a) stratify on a discretised `H` (declare the binning in advance, since binning is a researcher degree of freedom), (b) the paper's own label-stratified form using the authorised A-task label | Declare the discretisation before any outcome; report both arms; identical attackers |
| **OptNet-ARL** | The non-closed-form member of the same family; separates "closed form" from "this family of objectives" | Deep encoder with closed-form ridge players, multiple λ for SEX and RAC1P | Same channel width, same training pool, same compute budget disclosed |
| **LEACE** (and **SPLINCE**) | They are the closed-form linear-erasure guarantee the audience will ask about | Apply to the concatenated `[H_A, Z]` wire, not to `H_A`, since `H_A` must be preserved bitwise. SPLINCE additionally preserves covariance with the residence label | State explicitly that their guarantee is against linear adversaries only, and score them against the same nonlinear attackers rather than against a linear probe |
| **A withholding-matched simple channel** | Already computed here, and it is cheap | Use the closed-form `p*` matching of `STATISTICAL_REANALYSIS.md` §10, declared prospectively next time rather than post-hoc | Report identifiability: `p*` outside [0,1] means the comparator cannot reach that utility at all |

Two design requirements that apply to the whole set: every baseline must be evaluated **at a declared
width**, because rank is the factor most likely to confound a leakage comparison; and each must be
scored by the *same* attacker families, selected on the *same* validation pool, with absolute and
incremental recovery reported separately.
