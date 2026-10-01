# Capability table: multi-recipient and combined-release support (methods role, 2026-10-01)

Row IDs match `method_catalog.csv`. Evidence, citations and source paths are in the catalog's `sources`
and `existing_results_*` columns.

## Terms

* **Recipient**: a party that receives one release and has a declared sensitive set D_k.
* **Coalition**: a set of recipients who pool their releases.
* **Native**: the authors' method or paper supports the feature as published.
* **Per-recipient adaptation**: a wrapper that leaves the algorithm unchanged, applied once per recipient.
* **Sequential collusion-aware design**: each release is designed with the earlier releases in view, under a cumulative (coalition) constraint.
* **Joint design**: all recipients' releases are produced by one optimisation, or by one shared release.
* **Evaluation of combinations (existing)**: whether either repository has already measured attacks on combined releases.
* **Guarantee covering combinations**: whether a *stated* guarantee extends to a combined release. A row marked "derived" is our own one-line consequence, not the authors' claim.

**Cell values.** "not available" means the feature does not exist. It is not a numeric zero and must never be scored as 0 in an aggregate. "not evaluated" means nobody has run that evaluation. "not yet verified" means we could not establish the fact.

## Two facts that shape the table

These are mathematics, not run results. Both are checked by the identity Cov([A;B],Z) = [Cov(A,Z); Cov(B,Z)].

1. **Exact linear guardedness survives concatenation, for an attribute erased in every release.** Suppose every release a coalition holds has Cov(r_k(X), S) = 0 on the fitting sample. Then the stacked release also has zero cross-covariance with S. The same holds for chained LEACE: any later linear map keeps an earlier zero cross-covariance at zero.
   * Approximate criteria do **not** compose this way. PCRL's R² ≤ τ is one example: two releases each at R² = 0.04 can reach a concatenated R² above τ. PCRL's cross-purpose attack flags 26/33 for exactly this reason (origin/main README.md:25).
   * Nonlinear functions of the stacked release are not covered.
2. **An attribute declared for only some coalition members is not protected against the coalition** by any per-recipient method. If S is in D_1 but not in D_2, recipient 2's release may carry S, and the coalition reads it from there. Only a joint or collusion-aware design can address this. The relevant joint designs here are a union erasure (BAS-02, OWN-02), a coalition constraint (OWN-03, OWN-09 with AB) or a cumulative constraint (REL-03).

## Table

| ID | Method | Native purpose/coalition support | Runnable per-recipient adaptation | Sequential collusion-aware design | Joint design | Evaluation of combinations (existing) | Guarantee covering combinations |
|---|---|---|---|---|---|---|---|
| LIN-01 | INLP | not available (one label per run) | yes: one INLP per recipient over D_k (product label, or one attribute at a time) | not available | yes: INLP on the union of D_k | not evaluated | none stated (INLP has no formal guarantee) |
| LIN-02 | LEACE | multi-attribute only: a concatenated z erases several attributes jointly; no recipient notion | yes: BAS-01 (one LEACE per recipient). Always runnable; never "impossible" | not available | yes: BAS-02 (LEACE on the union) | ACS: leace_A0 scored on A, B and AB coalition endpoints (invariant-baselines@73903b7). Tabular: not evaluated as a per-recipient baseline | none stated by authors. Derived: linear, in-sample, for attributes erased in every member's release (fact 1) |
| LIN-03 | RLACE | not available | yes, one game per recipient (rank k must be chosen) | not available | yes: game on the union label | not evaluated (never actually run; PCRL's "R-LACE" is an INLP-style loop) | none stated |
| LIN-04 | Kernel RLACE | not available | needs a multiclass extension | not available | not available | not evaluated | none stated |
| LIN-05 | SAL / kSAL | multi-column Z only | yes (same as LEACE) | not available | yes: union Z | not evaluated | none stated. Derived (linear part): same as LEACE |
| LIN-06 | MiMiC | not available | needs z at deploy, which changes the access model | not available | not available | not evaluated | none stated |
| LIN-07 | SPLINCE | one (z, y) pair. Per-recipient by construction, if the recipient's task y is used | yes: SPLINCE(D_k, y_k) per recipient | not available | partial: stacked y plus union z, which can break the subspace-intersection condition (PCRL fell back to LEACE in 16/60 cells) | not evaluated for combinations. Note that SPLINCE preserves Cov(X, y_k), so a coalition gets every member's task signal | none stated. Derived: linear, in-sample, for attributes erased in every member's release. Caveat: a preserved task signal that is coupled to S can leak S through outputs |
| LIN-08 | QLEACE / orth variant | not available | QLEACE needs z at deploy. The orth variant works like LEACE | not available | orth: union | not evaluated | none stated |
| LIN-09, LIN-10, ADV-04 | theory / analysis / unverified | not applicable | not applicable | not applicable | not applicable | not applicable | not applicable |
| ADV-01 | Elazar & Goldberg adversarial removal | not available | via ADV-02 | not available | not available | not evaluated | none |
| ADV-02 | DANN-style adversarial censoring | not available (several adversaries are possible) | yes: one encoder per recipient, with adversaries over D_k | not available | yes: one encoder with union adversaries, or one encoder per recipient trained against a coalition adversary (our design, not published) | ACS: direct-adversarial C1 policy evaluated AB coalition endpoints (106de9a); no improvement | none |
| ADV-03 | Xie et al. controllable invariance | not available (s is an encoder input) | requires adaptation | not available | not available | not evaluated | none |
| AFR-01/02 | LAFTR (official) / LAFTR-style | single binary attribute; transfer evaluated empirically only | binary cells only (official). The reimplementation supports multiclass | not available | union adversaries (reimplementation) | not evaluated | none stated. The DP/EO bound holds only at the optimal adversary and for a single classifier |
| AFR-03 | Adversarial Forgetting | **sketched, not evaluated**: one forget-gate plus discriminator per (y_j, s_j) over a shared z | requires a reimplementation (no code) | not available | shared z with per-task masks (sketch) | not evaluated | none |
| AFR-04 | CFair | not available | binary cells only | not available | not available | not evaluated | none for combinations |
| VIN-01 | VFAE | not available (s is an input at deploy) | yes: MMD over D_k groups (product groups is an extension) | not available | union groups | not evaluated | none |
| VIN-02/03 | CVIB / FCRL | c may be a vector; no recipients | yes, per recipient | not available | union c | not evaluated | FCRL: I(z;c) bounds DP for any algorithm on z. For a coalition holding several z_k, I(z_1, z_2; c) is not controlled by the separate bounds (chain rule). Not covered |
| VIN-04 | FNF | not available (binary A, and A is needed to encode) | binary cells only | not available | not available | not evaluated | per-release Delta only. Not covered |
| VIN-05 | FARE | not available | yes: one tree per recipient (multivalued s via App. D.1) | not available | product s | not evaluated | per-release DP certificate only. Not covered: two finite-support releases jointly define a finer partition whose DP distance is not certified |
| VIN-06 | Efficient Fair PCA | multi-attribute variant in paper | binary only in the repo API | not available | union groups (paper variant) | not evaluated | none stated. Derived: linear means only |
| VIN-07, VIN-08 | Samadi / Olfat fair PCA | not applicable / not available | not meaningful / binary | not available | not available | not evaluated | not applicable / per-family only |
| VIN-09 | Obliviator | not available | yes: one run per attribute per recipient | not available | product label | not evaluated | none (empirical) |
| VIN-10 | DPNR | not available | not applicable (record-level LDP) | DP composition would apply | not applicable | not evaluated | DP composition bounds record privacy across releases. Says nothing about a declared attribute |
| VIN-11 | Gitiaux & Rangwala | not available | requires reimplementation | not available | not available | not evaluated | none |
| VIN-12 | FairNVT | not available | requires reimplementation | not available | not available | not evaluated | none |
| FLX-01 | FFVAE | **native** release of subsets of binary attribute dimensions b at test time (compositional subgroups). No recipient or coalition model | yes: recipient k receives [z, b_allowed(k)] | not available | yes: one shared latent | not evaluated | none (empirical subgroup DP) |
| FLX-02 | L-MIFR | several constraint types on one representation | yes, with u as an input | not available | not available | not evaluated | none for combinations |
| REL-01 | Privacy funnel | single analyst | finite alphabet only | not available | not available | not applicable | not applicable |
| REL-02 | Liu & Wang multi-task privatization | multiple tasks, one release | finite alphabet only | not available | yes: one release for many tasks | not applicable | covers the single shared release |
| REL-03 | Taylor/Vippathalla/Coon | **native** sequential multi-recipient design with per-recipient I(R̂_k;X) ≤ ε_k and cumulative I(R̂_1..R̂_k;X) ≤ δ_k | finite alphabet with known pmf; adapting it to a declared attribute and a plug-in pmf changes the claim | **native**, but the coalition is fixed as "all releases so far", not arbitrary subsets | yes (sequential) | in-paper only (UCI Adult, discretised) | yes, in the fitted finite model, for the cumulative coalition. Arbitrary subsets are covered only through MI monotonicity (subset ≤ cumulative). The protected object is the whole database X, not a declared attribute |
| REL-04 | OptNet-ARL / SARL | not yet verified | PCRL reimplementation | not available | not yet verified | ACS AB endpoints (invariant-baselines) | none |
| EVL-01/02 | Song & Shmatikov / Stadler | not applicable | not applicable | not applicable | not applicable | not applicable | Stadler: an impossibility result that applies to every row |
| BAS-01 | per-recipient LEACE | runnable per-recipient adaptation | **yes** (this row is that adaptation) | not available | no | not evaluated (recommended for the first benchmark) | derived only: fact 1 for attributes in every member's D_k. Fact 2 failure otherwise |
| BAS-02 | joint union LEACE | joint design (one shared release) | not applicable | not applicable | **yes** | ACS leace_A0 (AB endpoints) | derived: linear, in-sample, against the union, for any coalition (all members hold the same release) |
| BAS-03 | anchors | not applicable | not applicable | not applicable | not applicable | not applicable | constant/withhold: trivially yes |
| OWN-01 | PCRL v2 | **native** per-purpose (per-recipient) adapters on a shared backbone | native | not available | shared backbone. Per-purpose erasure only over each purpose's own D_k | cross-purpose concatenation attack flags 26/33 (README.md:25,102); Dominant-Axis audit | no. The per-purpose linear R² ≤ τ does not compose (fact 1 caveat) |
| OWN-02 | PCRL erase layer | native per-purpose adapters behind a joint union erasure | native | not available | **yes**: frozen joint LEACE over the union | inherits the concatenation evaluation in OWN-03 | derived: linear and in-sample at the erase-layer output. The adapters are trained afterwards under R² duals, so exactness is not maintained |
| OWN-03 | erase layer plus cross-purpose constraint | **native full-coalition** constraint (dual on linear R² of h_concat ≤ 0.10) | native | not available | **yes** | per-pair 60/60, h_concat 27/27, attack flags 22→8/33 (cross-purpose-rebuttal@d392112) | full coalition, linear R² ≤ 0.10 only. Sub-coalitions are implied by in-sample R² monotonicity. Not nonlinear |
| OWN-04 | erase-layer controls | not applicable | not applicable | not applicable | not applicable | not applicable | not applicable |
| OWN-05 | PCRL plus LAFTR hybrid | native per-purpose | native | not available | shared backbone | not evaluated | no |
| OWN-06 | isotropic noise channel | not available (attribute-agnostic) | yes: per-recipient σ | not available. Independent draws across recipients enable an averaging attack | one shared draw | averaging attack studied for isolate-then-noise (averaging_attack.json), not for multi-recipient release | none. Data-processing inequality: a shared single draw is no worse for a coalition than for one member. Independent draws are not covered |
| OWN-07/08 | isolate-then-noise; DP channel / hybrids | not available | yes | DP composition (OWN-08 only) | not available | not evaluated | OWN-08: DP composition on input-record privacy only |
| OWN-09 | ACS task-directed release Q | recipient-specific release (A) with an optional AB coalition CMI constraint. The selected nominee was local-only | native | not available (non-sequential) | **yes**: AB constraint variants | AB coalition endpoints on 2018/2017/2016 | in the fitted finite model on a coarse partition C, for the AB constraint variants only. Not population-level. Not for the nominee |
| OWN-10 | ACS comparators / channel family | Track E is **coalition-conditioned** (projection depends on the coalition) | native | not available | yes | AB endpoints throughout; all closed negative | none |

## Reading the table for benchmark design

* Separate per-recipient LEACE (BAS-01) and joint union LEACE (BAS-02) are runnable today for **every** tabular cell. Every claim about "multi-recipient methods" should be compared against them before anything else.
* Only five rows have native multi-recipient or subset structure, and they are of different kinds:
  * learned, no coalition model: FFVAE (binary only, no official code) and Adversarial Forgetting (sketch only, no code);
  * information-theoretic, finite alphabet: REL-03 Taylor et al.;
  * project-internal: OWN-01/02/03 and OWN-09/10.
* No published learned-representation method states a guarantee covering combinations of releases. Every "covering" entry above is derived (linear, in-sample) or project-internal.
* Columns that read "not available" must be reported as "not available" in any results table. They must not be scored as 0 or as a failure.
