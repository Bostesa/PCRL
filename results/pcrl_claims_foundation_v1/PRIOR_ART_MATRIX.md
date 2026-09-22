# Prior-art matrix

Primary-source verification is in [PRIOR_ART_NOTES.md](PRIOR_ART_NOTES.md). For every source it records the
PDF version opened and the section or theorem read, and it marks anything seen only as an abstract as
UNVERIFIED. Terminal 3 re-checked the load-bearing items directly at arXiv:
- Taylor, Vippathalla, Coon, *Adaptive Privacy of Sequential Data Releases Under Collusion*,
  arXiv:2601.21859, v1 2026-01-29 / v2 2026-07-10: MI leakage, collusion between parties, sequential
  releases, optimal for expected-distortion utility.
- Rassouli & Gündüz, *On Perfect Privacy*, IEEE JSAIT, arXiv:1712.08500: utility is measured by I(Y;U).

The LEACE theorem numbers are those in the notes (arXiv:2306.03819).

One correction to the notes. Their (b) row says the channel input is "a label-free code". That is true
only of the precursor studies (cd895e4, e3415b9). The latest release's T0 is a residence-supervised
residual-score code (EARLY_FINDINGS E1).

## Matrix

Columns: problem | prior outputs / side information | purpose-specific utility | recipient views |
immutable services | stochastic mechanism | optimisation assumptions | guarantees | empirical scope.

| Our item | Closest prior (exact locus) | Problem | Side info / prior outputs | Purpose-specific utility | Recipient views | Immutable service | Stochastic mechanism | Optimisation | Guarantee | Empirical scope | Substantive remaining difference |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) per-purpose permitted tasks and forbidden attributes | Creager et al. ICML 2019 (FFVAE §4); Stadler et al. 2024 (Def. 2, least privilege ↔ purpose limitation) | one encoder, sensitive subsets chosen per task | none | yes (per task) | one consumer per task | no | no | SGD (VAE) | none formal | tabular + image | conflicting permit/forbid sets across recipients, plus a coalition-level prohibition; formulation, not algorithm |
| (a′) original encoder line: per-purpose LoRA + LEACE warm start + proxy-Lagrangian R² bound | LEACE (Belrose et al. 2023, Thms 3.1, 3.4, 4.1–4.3); INLP; LAFTR; per-purpose LEACE baseline | per-purpose linear erasure | none | yes | per purpose | no (the releaser owns the representation) | no | nonconvex SGD with duals; selection tuned on the same grid | in-sample ridge R² ≤ τ on test rows; no nonlinear guarantee | Adult, HMDA, Diabetes (+Folktables) | parameter sharing across purposes; the restricted arm **ties** per-purpose LEACE in the controlled conflict; no LoRA-vs-affine-adapter comparison executed; backbone is a frozen random MLP (E5) |
| (b) a channel appended beside frozen H; increments measured over the H-only view | Erdogdu & Fawaz ISIT 2015 (§III, Thm III.3); Taylor et al. 2026 (§§II–IV); Sankar et al. 2013 (Thm 3); Issa et al. (Thm 6); Lopuhaä-Zwakenberg SRLIP (Def. 3) | new release given fixed prior releases; leakage conditional on side information | the releaser's own earlier releases (Erdogdu, Taylor); recipient side information (Sankar) | distortion / MI | single or sequential parties | earlier releases fixed | yes | convex (finite alphabets) | MI budgets in the model | synthetic / small | side information is a third party's frozen ML prediction vectors; audit by independently fitted attackers relative to an H-only family (methodology, not a formal result) |
| (c) per-recipient and coalition accounting | Taylor et al. 2026 (individual ε_k and collusion δ_k MI constraints); Li et al. TKDE 2012 (collusion-robust multi-level copies); SRLIP (all side-channel subsets) | multi-party, collusion-aware release | pooled previous releases | distortion / MI | per party + collusion | previous releases fixed | yes | convex for distortion (Blahut–Arimoto style) | MI constraints in the model | theory + small numerics | the coalition view includes H_B, which the mechanism never produced; constraints conditional per view. Only A receives a channel. The coalition-constrained nominee earned no attribution, and the selected release is local |
| (d) one-hot aggregate vs per-class; DA audit; exact-zero composition | variance-weighted multi-output R² (definition; e.g. scikit-learn `variance_weighted`); canonical correlation (Hotelling 1936, not opened); LEACE Thm 3.4 | audit statistics | — | — | — | — | — | closed form | algebraic identity; exact-zero guardedness | — | an empirical cautionary example (HMDA 0.027 vs 0.288, rarest class); the identity is not new |
| (d′) approximate composition bound (original P6) | Rayleigh–Ritz; standard linear algebra | bound concatenated linear R² | — | — | multiple views | — | — | — | R² ≤ Σ R²_p/λ_min(R) | Adult Table 3 (historical) | elementary lemma; tight example F11 |
| (e) finite stochastic release Q with CMI budgets, convex for a fixed table | Calmon & Fawaz 2012 (Thms 1–2); Salamatian et al. JSTSP 2015; Erdogdu & Fawaz 2015 (Thm III.3); Taylor et al. 2026; Calmon et al. NeurIPS 2017 (Prop. 1: randomised finite preprocessing, a different constraint type); Rassouli & Gündüz (zero-leakage nullspace criterion; LP for **MI** utility, Thm 1) | leakage–distortion channel design | fixed priors / releases | distortion, linear in Q | one or more | yes (Erdogdu, Taylor) | yes | convex | MI in the fitted model | synthetic | conditioning views are recipients' coarsened service outputs (2 or 4 cells); the cost is a frozen residence decoder's expected cross-entropy; I(S;Z\|C) = I(S;Z,C) − I(S;C), so this is a total-leakage budget with a shifted constant. **No new optimisation result.** |
| randomisation can dominate determinism (fixture) | Rassouli & Gündüz (perfect privacy via randomised mappings) | — | — | — | — | — | yes | exact | exact independence | fixture only | a design example; the ACS D17 control shows no demonstrated advantage |

## Mischaracterisations to fix in the manuscript

These are verified in the notes, Part 2. They are listed against main.tex @ 30a6fd19e.

- **M1** :176-177, :203-204. Rassouli–Gündüz's LP is for mutual-information utility and unconditional
  perfect privacy. Linear-cost utilities appear only in their Remark 2, without proof.
- **M2** :181. The convexity facts are standard, or due to Calmon & Fawaz 2012. The nullspace criterion is
  Rassouli–Gündüz's, and it is unconditional.
- **M3** :170-173. In U-FaTE, "conditional" means conditioning on the target label (for separation-type
  fairness). It has no eigenvalue-sign rank selection.
- **M4** :179-180. Liao et al. design no release, and Sankar et al. release to one side-informed user. The
  nearest multi-recipient prior art is Taylor et al. 2026.
- **M5.** Add calmon2012privacy, salamatian2015managing, erdogdu2015continual, taylor2026collusion, and
  calmon2017optimized (as the fairness analogue).
- **M6** :150. "Squared loss only" understates LEACE Thm 3.1, which covers any loss convex in the affine
  prediction. The accuracy counterexample stands.
- **M7** :73-75 and NOVELTY_AND_SCOPE item 3. The identity is not original. The contribution is the audit
  observation and the example.

BibTeX for the missing entries is in PRIOR_ART_NOTES.md, Part 3. Proposed `references.bib` additions are in
`tex/references_additions.bib`.

## Defensible novelty paragraph (paper-ready)

> The ingredients are established: finite-alphabet leakage–distortion programmes are convex for a fixed
> joint law [Calmon & Fawaz 2012; Salamatian et al. 2015], new releases have been designed beside fixed
> earlier releases with incremental-leakage budgets [Erdogdu & Fawaz 2015], and per-party and collusion
> mutual-information constraints on sequential releases were studied recently [Taylor et al. 2026]; zero
> linear cross-covariance and its exact composition are linear-guardedness facts [Belrose et al. 2023]. What
> this paper adds is a specific release setting and an empirical study of it: the side information is a
> *third party's already-published prediction vectors*, which the releaser cannot change and which a
> coalition partner holds; permitted and forbidden uses differ by recipient; every disclosure number is an
> independently fitted attacker's gain over an attacker restricted to the published outputs, with absolute
> recovery reported beside it. Within that setting we evaluate one instance of the established programme
> against the strongest channel we had and against matched deterministic releases. We claim no new
> optimisation result, and on our development data the stochastic machinery has not been shown to add
> anything over a matched deterministic release.

**Plain-language version.** A company already sends some predictions to partners and cannot take them
back. We asked how much extra useful information it can send one partner without letting that partner,
alone or together with a second partner, learn much more about people's sex or race than they already
could. The mathematical tools for designing such a release already exist. Our contribution is to set the
question up for this situation, to measure the answer honestly, and to report that a simple
non-randomised release did about as well as the optimised one.

**What would be needed for an algorithmic novelty claim.** One of the following is missing: (i) a proof
that the conditional-on-a-third-party-view programme differs in kind from a shifted total-leakage budget
(it does not; see the chain-rule identity); (ii) a jointly optimised set of channels for several
recipients (not implemented: only A receives a channel); or (iii) an empirical win of Q over the matched
deterministic control D17, which the registered 2016 comparison (Terminal 1) may or may not supply.
