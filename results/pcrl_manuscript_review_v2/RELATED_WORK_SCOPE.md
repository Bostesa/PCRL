# RELATED_WORK_SCOPE — what we claim about prior work, and what a fair comparison would need

This extends `results/pcrl_evidence_review_v1/NOVELTY_MATRIX.md` and
`CONTRIBUTION_ASSESSMENT.md` (both at `0d8f4b67b6d4961dfa133289d0167c874d2f4794`), which remain the
primary records and are not superseded. Added here: the four-question separation the manuscript
uses, and two corrections to the external-baseline plan that would otherwise produce an unfair or
infeasible comparison.

## 1. Four questions, answered separately

Conflating these is how a bounded literature search turns into a novelty claim. The manuscript
answers them one at a time.

| Question | Answer | Basis |
|---|---|---|
| **(a) Did earlier work define an objective that can *express* a related setting?** | **Yes, several.** SARL, OptNet-ARL and K-TOpt all express "maximise a utility form minus λ times a dependence form under an orthogonality constraint", which is the shape of our original objective. U-FaTE expresses a conditional version of it. Information-theoretic privacy expresses the utility/leakage trade-off directly. | `NOVELTY_MATRIX.md` §1, full-text rows |
| **(b) Did it provide an *algorithm applicable* to this setting?** | **Yes, with adaptation.** SARL's and K-TOpt's solvers apply once the label target is replaced by the residualised teacher; U-FaTE's applies once a discretisation of *H* is declared. The adaptations are specified in §3 below. | `CONTRIBUTION_ASSESSMENT.md` §6 |
| **(c) Did it *evaluate* recipient-specific and combined-access policies with unchanged existing outputs?** | **We did not find one that did.** Eight targeted searches plus the full-text reads found no prior evaluation that constrains a previously published output to be reproduced bitwise while adding capability beside it, and none that penalises the joint view of two colluding recipients. Sankar et al.'s "multiple consumers" share **one** release and differ only in private side information. | `NOVELTY_MATRIX.md` §4, N5 |
| **(d) Did *our* study demonstrate an effect beyond its own chosen controls?** | **Yes, against L1 and L2 on one sealed year** (14 of 16 sensitive cells strictly better, 0 worse, 2 unresolved), and **no** against J, and **no** for the refinement against any of its intended comparisons. | `TRANSPORT_DECISION.json`, `DEVELOPMENT_2018.json` |

**The claim the manuscript makes is (c), in its weak form**: *we did not find a directly matching
evaluation*. It does not claim that prior formulations cannot express this setting — (a) says
several can — and it makes no priority claim. A bounded search cannot establish novelty.

## 2. Components that are prior work, listed so the paper cannot be read as claiming them

| Component | Prior art |
|---|---|
| The spectral trace step | Ky Fan (1949), applied as SARL and K-TOpt apply it |
| Random Fourier features inside a closed-form solver | K-TOpt |
| Multiple protected attributes via weighted penalties | OptNet-ARL, explicitly |
| Rank selection by eigenvalue sign | SARL Thm 3, OptNet-ARL Thm 4.1, K-TOpt Cor 4.1 — **three times**, and it changed nothing here |
| A conditional dependence penalty inside such a solver | U-FaTE (conditioned on a discrete label) |
| The conditional-independence moment form | KCI Lemma 2(v) / Daudin; RCoT Eq. 26 |
| Cross-fitting the nuisance | Chernozhukov et al. |
| The marginal arms M025 / M1 | SARL's construction, up to parametrisation |
| "Erasure certified against linear probes fails against nonlinear attackers" | LEACE scopes its own guarantee; Elazar & Goldberg (2018) |
| A task-utility / non-task-leakage floor | Stadler et al. (2024) |

**Reusing a known solver, a conditional-moment characterisation or a rank rule is attribution, not
innovation.** Appending *H* unchanged makes output preservation structurally simple; that is an
observation about concatenation and the manuscript does not inflate it into a theorem.

## 3. Corrections to the external-baseline plan

`CONTRIBUTION_ASSESSMENT.md` §6 specifies six baselines. Two rows would produce an unfair or
infeasible comparison as written.

### 3.1 LEACE and SPLINCE must act on the auxiliary channel only

**As written:** *"Apply to the concatenated `[H_A, Z]` wire, not to `H_A`, since `H_A` must be
preserved bitwise."*

**Problem.** An affine edit of the concatenated wire is a linear map on all of `[H_A, Z]`. Unless it
is block-constrained, it changes the `H_A` coordinates too, which is exactly the constraint it was
meant to respect. Applying LEACE to the concatenation is therefore not a way to preserve `H_A`.

**Correction.** If `H_A` is immutable, the erasure operator must be restricted to the `Z` block:
`[H_A, Z] → [H_A, W(Z)]`. Two consequences must then be stated with any result:

1. **The guarantee has that scope.** LEACE's linear-erasure guarantee, applied to `Z` alone, says
   no linear predictor of `S` from `W(Z)` beats the constant predictor. It says nothing about a
   linear predictor from the **joint** wire `[H_A, W(Z)]`, and nothing at all about the coalition
   wire `[H_A, W(Z), H_B]`. Since `H_A` already carries recoverable sex and race
   (0.0101 and 0.0165 nats at the measured floor), a whole-wire reading of the guarantee would be
   false in this setting.
2. **It remains a linear guarantee.** Both papers scope their claims to linear adversaries. They
   must be scored against the same MLP, boosted-tree and kernel families as every other arm, not
   against a linear probe.

### 3.2 No baseline may consume residence labels

**As written:** *"SPLINCE additionally preserves covariance with the residence label."*

**Problem.** Residence is the **transfer task**. It is excluded from every representation fit in
this study, and the residence gain over *H* is the utility axis on which all channels are compared.
A baseline that fits its representation using residence labels is being compared on a task it was
trained for, against arms that were not. That is not a fair comparison; it is an oracle.

**Correction.** In the main comparison, SPLINCE's target-covariance slot must be filled by an
**authorised A-task label** (income or employment), which the spectral arms' penalties also use, or
left unfilled. A residence-label variant may be reported only as an explicitly labelled
**privileged-label oracle**, outside the main table, with a one-line statement that it upper-bounds
what label access buys rather than measuring the method.

The same rule binds every other baseline: U-FaTE's stratification label, OptNet-ARL's target heads
and SARL's utility target must all come from the authorised set. The residualised-teacher
substitution specified in `CONTRIBUTION_ASSESSMENT.md` already satisfies this.

### 3.3 Two conditions that already apply, restated

* **Declared width.** Rank is the factor most likely to confound a leakage comparison — this
  study's own rank-8 arm reduced race recovery in 15 of 48 cells with no measurable residence cost.
  Every baseline is evaluated at a declared width, with its own rank rule's width reported beside a
  width-16 truncation.
* **Identical attack surface.** Same attacker families, same validation pool, same selection rule,
  absolute and incremental recovery reported separately, negative increments unclipped.

## 4. What the absence of baselines costs this paper

**No external published method was executed.** Consequently:

* The paper supports **no** competitiveness claim against the literature, in either direction.
* **J is an internal adversarially trained channel, not an external benchmark.** Where J beats our
  arms, that is evidence about J. It is not evidence that the spectral family is behind the state
  of the art, and it must not be reported as though it were.
* Equally, J's position is **not** an upper bound on attainable utility at a privacy constraint. A
  maximally leaky channel is not such a bound, and neither is any single measured point.
* A finite collection of failed variants — two original arms, twelve refined ones — rules out no
  method family. Spectral, kernel, nonlinear and fixed-service approaches all remain open here.

## 5. The private AAAI manuscript

Only the supplement of the user's AAAI manuscript was supplied in chat; the main paper is private
and **is not available locally in this repository**. No comparison in this package claims to have
reviewed it. The documented difference remains scoped to what was actually seen: a noisy
end-to-end removal/certificate audit, versus incremental releases with fixed services and
individual and combined recipient access. Both contain methods, and no priority claim is made in
either direction. Any private overlap notes stay untracked and unpushed; nothing of that manuscript
is needed to complete this public revision, and none of it is included here.
