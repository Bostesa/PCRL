# Prior-work delta review: joint complete-view (JCV) mechanism

**Role:** read-only mathematics / prior-work reviewer. **Date:** 2026-10-03. **Scope:** the runner's working
specification (two recipients on Adult: R1 `income`, R2 `occupation_group`; SEX forbidden to both; separate
64-64 MLP encoders; per-purpose official LEACE refitted at epoch boundaries; view `v_i = [r_i, centred h_i(r_i)]`;
critic-bank surrogate `R_v`; arms L, J, JP, S12/S21, U, E, F/F0). Nothing was run except text extraction and the
small checks in `THEORY_REVIEW.md`. Bib keys refer to `references_jcv.bib`.

**Markers.** [V] = checked today in the primary text (PMLR/NeurIPS/ACL/CVF PDF, or the arXiv version in the bib);
[abs] = abstract or landing page only; [meta] = publisher metadata only.

## 0. Bottom line

1. **The projection step is GEM's QP** (`lopezpaz2017gem` §3, Eq. 8) with the roles relabelled: the proposed
   direction is the privacy gradient `p`, the protected losses are the task guards, and only ε-active guards are
   constrained (a Rosen-style active set, `rosen1960gradient`). With one active guard it is exactly the A-GEM closed
   form (`chaudhry2019agem` §4, Eq. 11) and PCGrad's pairwise projection (`yu2020pcgrad` Alg. 1, line 7). Because
   `g1, h1` and `g2, h2` share no parameters, `a_1·a_2 = 0`, and the exact two-guard projection *is* two independent
   A-GEM projections (verified numerically; `THEORY_REVIEW.md` §7). The method card should call it an adaptation of GEM.
2. **The surrogate is Song et al.'s adversarial constraint `C2`** (`song2019controllable` §2.3, Eq. 6-9) with the
   empirical marginal as `p(u)`, maximised over a small critic bank and divided by `H_fit(S)`.
3. **JP is the fixed-multiplier Lagrangian** that Song et al. (Eq. 11) show LAFTR and earlier methods optimise.
4. **Per-purpose LEACE inside training** is LEACE §7's suggested next step, and PCRL already ran it (erase-layer
   pilot, joint-LEACE warm start). The prior in-house outcome was structural linear compliance with *higher*
   nonlinear recovery (48/60 cells).
5. **A coalition term on concatenated recipient releases** is already in PCRL (linear R² on `h_concat`,
   τ = 0.10). The new elements are a nonlinear critic coalition term, complete views and the projected update. In a
   bounded search, no learned-representation method was found that trains several recipient encoders against a
   coalition critic. Taylor et al. 2026 is the only located collusion-aware *release mechanism*, and it is a different
   problem (finite alphabets, known pmf, MI on the whole record).

## 1. LAFTR (`madras2018laftr`) [V]

- **Objective.** One encoder `f(X[,A])`, classifier `g`, decoder `k` and adversary `h`, trained by
  `min_{f,g,k} max_h E[α L_C + β L_Dec + γ L_Adv]` (§4, Eq. 1-2) with alternating descent/ascent (§4.2). `L_Adv` is a
  group-normalised L1 objective matched to DP, EO or EOpp (Eq. 3-4).
- **Theory.** The optimal adversary's objective upper-bounds `Δ_DP(g)` for any binary `g` on `Z` (§5.1, Eq. 5-8).
- **Recipient model.** One data owner and one or more downstream vendors who all receive the *same* `Z`
  (§4.3 "adversarial vendor"). There are no coalitions, no per-recipient releases, and the encoder may see `A`.
- **Evaluation.** Freeze `f`, then fit an unconstrained MLP on fresh data and report its accuracy and fairness gap
  (Alg. 1). Adult income/sex (§6.1) and Health transfer over 10 PCG tasks (§6.2).
- **Delta.** JCV trains two encoders, one per purpose, plus a coalition critic on `[v1, v2]`. It uses CE critics with
  a constant prior instead of the DP-matched L1 adversary, places LEACE in the forward path, and uses a guarded
  projection instead of the weighted min-max. **JP plus a pair critic is LAFTR's objective with one extra adversary.**
  LAFTR's DP theorem does not transfer: JCV's critic bank is finite, and LAFTR measures fairness gaps, not
  attribute-recovery AUC.

## 2. Song et al., "Learning Controllable Fair Representations" (`song2019controllable`) [V]

- **Objective.** `max I(x; z|u)` s.t. `I(z; u) < ε` (§2, Eq. 1). The practical form is `min L_r` s.t.
  `C1 = E KL(q(z|x,u) || p(z)) < ε1` and `C2 = E[log p_ψ(u|z) - log p(u)] < ε2` (§2.4, Eq. 10).
  `I(z; u) ≤ C2 + ℓ` only when the adversary's KL gap is at most `ℓ` (Corollary 4, Eq. 9); `C2` is "an upper bound
  only in the case of an optimal adversary" (§2.4).
- **Optimisation (L-MIFR).** The dual `max_{λ≥0} min_{θ,φ} max_ψ L_r + λᵀ(C - ε)` (§4, Eq. 12): descent on encoder
  and decoder, ascent on the adversary, gradient *ascent on λ*. Strong duality is proved only in distribution space
  (Thm 5); feasibility in parameter space is empirical. Existing methods, including LAFTR, are the same Lagrangian
  with *fixed* multipliers (§3, Eq. 11).
- **Recipients and evaluation.** One representation for unknown vendors, no labels in training; German, Adult
  (gender), Health (18 groups); MI estimated on test data by KDE (§5.1).
- **Delta: constrained formulation vs per-step projection.**
  1. **Roles inverted.** Song constrains leakage and maximises expressiveness. JCV *penalises* leakage with a fixed
     β and *constrains* task loss (budget = warm-start + 0.01 nats).
  2. **Enforcement.**
     - Song learns a multiplier per constraint, so constraint pressure accumulates over steps, and feasibility holds
       only at a dual equilibrium.
     - JCV has no multipliers: each step removes the component of `p` that would raise an ε-active task loss to
       first order, and backtracks if a guard budget is crossed.
     - The projection is memoryless and myopic: it guarantees only one-step, first-order non-increase on the guard
       subset (`THEORY_REVIEW.md` §7), never convergence to a constrained optimum.
  3. **Surrogate.** `R_v · H_fit(S) = H_fit(S) - min_a CE_a`, which is Song's `C2` with `p(u)` = the empirical marginal,
     maximised over a bank. It is established, not new.

## 3. Official LEACE (`belrose2023leace`) [V]

- **Theory.**
  - Linear guardedness ⇔ equal class centroids ⇔ zero cross-covariance with one-hot labels ⇔ statistical parity of
    every linear classifier (§3, Thm 3.4).
  - An affine `r(x) = Px + b` guards `Z` iff `ker P ⊇ colsp Σ_XZ` (Thm 4.1).
  - LEACE is the least-squares-optimal such map, `r(x) = x - W⁺ P_{WΣ_XZ} W (x - E[X])` (Thm 4.2-4.3).
  - All statements are about the distribution whose moments are used, i.e. the empirical moments when fitted on data.
- **Scope.**
  - Evaluation uses linear probes.
  - §6 "concept scrubbing" fits LEACE layer by layer on a frozen network.
  - §7: incorporating scrubbing into training is future work, and "it remains to be seen if gradient-based
    optimizers will be able to 'circumvent' such constraints by encoding protected attributes in completely
    nonlinear ways". General nonlinear erasure is conjectured intractable.
  - Footnote 5 (earlier review): softmax outputs of a head can leak.
- **Delta.** JCV runs LEACE *inside* training, refitted at epoch boundaries and frozen between them, one map per
  purpose. This is exactly the §7 scenario. The guarantee is exact only on the defense-fit rows at refit time
  (`THEORY_REVIEW.md` §1, §10a). The coalition critic is the attempt to counter the circumvention §7 predicts.
  PCRL's erase-layer pilot observed that circumvention (§8 below).

## 4. FARE (`jovanovic2023fare`) [V]

- **Objective.** A restricted encoder with `k` cells: a fair decision tree split by
  `FairGini = (1-γ) Gini_y + γ (0.5 - Gini_s)` (§6). Binary `s` and `y` (§3).
- **Guarantee.** A practical certificate `T` such that `sup_g Δ_DP(g) ≤ T` with probability `1-ε` over all
  downstream classifiers (Def. 4.1). It is built from Clopper-Pearson bounds (Lemmas 5.1-5.3) on a held-out `D_val`
  that is not used to train `f` (§6).
- **Recipient model.** One encoder for any consumer; transfer is flagged as weak (§8).
- **Evaluation.** Health, ACSIncome-CA/US; downstream 1-hidden-layer net (§7); k-means restricted encoders (App. C).
- **Delta.** FARE has no coalition notion, but the pair `(f1, f2)` of two FARE trees is itself a restricted encoder
  with at most `k1·k2` cells, so FARE's own certificate procedure applies to the coalition unchanged (looser). Since
  `AUC - ½ ≤ TV`, this gives `AUC ≤ ½ + T` for *any* coalition attacker (`THEORY_REVIEW.md` §11). The F arm can be
  certified at the coalition; JCV's neural arms cannot.

## 5. Taylor, Vippathalla and Coon, arXiv 2601.21859v2 (`taylor2026collusion`) [V]

- **Objective (per request k).** `min_{p(r̂_k | r̂^{k-1}, x)} -U(R̂_k, R_k)` subject to (Eq. 1; Eq. 2 in generic form):
  - `I(R̂_k; X) ≤ ε_k`;
  - `I(R̂_k, R̂^{k-1}; X) ≤ δ_k`.

  Utility is `-E d(R̂, R)` (convex problem) or `I(R̂; R)` (nonconvex) (§II). **The protected quantity is the whole
  database record `X`**, not a sensitive attribute. In the Adult experiment, `X = (education, income, age)`
  discretised to `|X| = 32`, and the requests are attributes of `X` (§VII-A).
- **Online, not offline.** "As she does not have knowledge of future requests at the time of a data release, the
  data handler cannot jointly optimise all releases" (§II). Each release is chosen given the realised previous
  releases.
- **Finite alphabets, known pmf.** Optimisation is over conditional pmfs on finite alphabets with a known joint
  `p(r, z, x)` (§II; an empirical pmf in §VII-A); cost `O(T |R̂||Z||X| N_j)` per release grows with history (§VI);
  releases are randomised channels.
- **Collusion budget.** Worst case: one actor holds *all* releases to date; `ε_k ≤ δ_k`, `δ_k` non-decreasing
  (§I-II). Subset or n-of-m collusion appears only as an extension (§IX).
- **"Optimal".**
  - The per-request problem is solved for fixed history by a Blahut-Arimoto-style algorithm at fixed `(μ1, μ2)`
    (§IV, Prop. 1, Thm 1). It converges to the global minimum for expected distortion, a convex problem.
  - It is then mapped to the target `(ε, δ)` by grid, bisection and time-sharing (Alg. 3, §VI).
  - Under MI utility it is only locally optimal (§IV-E, §VI).
  - **Optimality is per step and greedy over the sequence. It is not optimality of the whole sequence.**
- **Why S12/S21 is not their algorithm.**
  - S12 trains deterministic neural encoders by SGD against finite critics, protects a single attribute `S` rather
    than `X`, and uses task cross-entropy rather than distortion or MI. It has no multiplier search and no optimality
    statement.
  - Because `r_1` is a deterministic function of `X`, "conditioning on the previous release" is vacuous at the
    channel level (`p(r̂_2 | r̂_1, x) = p(r̂_2 | x)`). S12 uses only the *fixed map* `g1` through the coalition critic.
  - At most, S12 is a neural heuristic *inspired by* their online, collusion-constrained formulation, with `S` in
    place of `X` (a privacy-funnel variant, `makhdoumi2014funnel`). Their guarantees do not transfer.
- **Why an offline joint design's advantage is not algorithm-only.**
  - Any sequential policy is a particular joint policy, so the offline feasible set contains the online one (same
    constraints). Even with exact solvers, the offline optimum is weakly better.
  - That gap is the *value of knowing future requests*, which their online handler is denied by assumption (§II).
  - J vs S12 therefore conflates foreknowledge of purpose 2 with the optimiser. **Only J vs L holds the
    information fixed** (both arms know both purposes and train both encoders together) and isolates the coalition
    term.

## 6. Gradient projection and constrained multi-objective updates

| Method | Update | Relation to `Proj(p)` |
|---|---|---|
| GEM (`lopezpaz2017gem` §3, Eq. 6-11) [V] | `min_g̃ ½‖g − g̃‖²` s.t. `⟨g̃, g_k⟩ ≥ 0` for all past tasks; dual QP over `t-1` variables, `g̃ = Gᵀv* + g` | **Same QP.** With update `θ ← θ - α g̃` and `d = -g̃`, `⟨g̃, g_k⟩ ≥ 0 ⇔ g_k·d ≤ 0`. JCV: proposal = privacy gradient, protected = ε-active task guards. |
| A-GEM (`chaudhry2019agem` §4, Eq. 10-11) [V] | One constraint on an averaged reference gradient; `g̃ = g - (gᵀg_ref / g_refᵀg_ref) g_ref` if violated | Identical to JCV with one active guard; with block-disjoint guards, exact JCV = A-GEM per block. |
| PCGrad (`yu2020pcgrad` §2.3, Alg. 1) [V] | For each task, sequentially (random order) remove the component along any conflicting other-task gradient; sum all | Pairwise step = the one-constraint projection; but PCGrad modifies *every* task gradient symmetrically, and a one-pass sequential projection is not the exact two-constraint projection (they differed in 553/2000 random draws with non-orthogonal `a_1, a_2`). |
| MGDA (`sener2018mgda` §3, Eq. 3; `desideri2012mgda`) [V] | Min-norm point of the convex hull of task gradients (common descent direction), Frank-Wolfe | Different: no preferred direction `p`, no budgets. |
| Rosen gradient projection (`rosen1960gradient`) [meta] | Project the gradient onto the active constraints' tangent set | Classical origin of active-set projected directions; the ε-band "active" rule is a standard active-set strategy. |
| Zhang, Lemoine, Mitchell (`zhang2018mitigating` §3, Eq. 1) [V] | `∇L_P - proj_{∇L_A} ∇L_P - α∇L_A`, always applied | Fairness-specific, **reverse roles**: the *task* gradient is projected off the *adversary* gradient (equality projection). |
| PGU unlearning (`hoang2024pgu`) [V] | Forget-loss gradient projected orthogonal to the retain set's core gradient space | Closest "remove information without harming retained loss" analogue; equality projection onto a subspace, not ε-active inequality guards. |

**Verdict on the step.** `min ‖d − p‖² s.t. a_j·d ≤ 0, j ∈ A` is GEM's Eq. 8 (sign-flipped). With `|A| = 1` it is the
A-GEM rule. It is established. Two things are specific, and neither is a new algorithm:
- the active set is an ε-band around a fixed budget, rather than GEM's "all past tasks";
- the task term `-∇(L1+L2)` is added *unprojected*. This is safe to first order only because the guards are
  block-disjoint (`THEORY_REVIEW.md` §7).

No fairness or privacy method was located that projects the *privacy* gradient against *active task-loss* guards.
This is a bounded search: general web-search queries plus the papers above.

## 7. Multi-recipient / collusion-aware learned representations (bounded search)

Queries covered collusion-robust representation learning, multi-party privacy funnels, colluding analysts and
multi-task privacy-preserving representations. **No learned-representation method was found that jointly trains
several recipient-specific encoders against a coalition adversary.** The closest are:
1. **Taylor et al. 2026 (§5).** The only collusion-aware release *mechanism*. It is information-theoretic, online,
   uses a finite alphabet, and protects all of `X`.
2. **Dwork & Ilvento 2019** (`dwork2019composition`) [abs]. Classifiers fair in isolation need not compose into fair
   systems (constructions mostly for individual fairness): the closest *fairness-side* precedent for "per-recipient
   guarantees do not compose", with no learned representations and no attribute recovery.

Also relevant (prior review, `CLOSEST_PRIOR_ANALYSES.md`):
- Ganta et al. 2008: composition attacks.
- Tian et al. 2025: two prediction releases of one model raise attribute inference.
- Elazar & Goldberg 2018 §5.2: an *ensemble* of k adversaries with summed losses (JCV's bank takes the min CE).
  Their "fused encoders" compose parts of *one* encoder, not a coalition.

## 8. Original PCRL cross-purpose constraint and erased variants (in-house precedent)

From the repository:
- `REBUTTAL_WORK_RECONCILIATION.md` §1b-1c;
- `BOTH_REPOS_EVIDENCE_MAP.csv` EM-029..042 and EM-057..068;
- `CLOSEST_PRIOR_ANALYSES.md` §3 item 7;
- the trainer config comment on `cross_purpose_attrs`.

- **What PCRL did.**
  - Purpose-specific LoRA adapters on a frozen random backbone.
  - Per (purpose, disallowed attribute) linear `R²_onehot ≤ 0.05` constraints, via proxy-Lagrangian duals
    (`cotter2019proxy`) with best-iterate selection.
  - An opt-in cross-purpose constraint `R²_lin(h_concat, A) ≤ τ_cross = 0.10`, with one dual per attribute.
- **What it found.**
  - **Erase layer** (frozen union-LEACE upstream of every trainable map): strict linear compliance 60/60, but
    structural; the nonlinear auditor delta rose on 48/60 cells (delta ≥ 0.02: 27 → 44); adjusted passes 32 → 16.
  - **Cross-purpose rebuttal model:** INCR concat flags 22 → 8, but absolute nonlinear leakage rose on 20/22 cells;
    the union eraser also removed attributes a purpose is *allowed* to use (an access-policy change); both criteria
    are post hoc.
- **How JCV differs.**
  1. **Coalition term.** A nonlinear critic bank on `[v1, v2]` replaces linear R² on `h_concat`. With per-purpose
     LEACE fitted on common rows, the concatenation is already linearly guarded (`THEORY_REVIEW.md` §1). The old
     linear coalition constraint would be vacuous here, so J can only act on nonlinear dependence.
  2. **Release surface.** The views include centred logits. This adds no Bayes information (§4 there).
  3. **Erasure.** Per-purpose maps against SEX only. Since SEX is forbidden to both recipients, union and
     per-purpose erasure coincide for SEX, and the access-policy objection does not arise for the primary attribute.
  4. **Encoders.** Trainable separate MLP encoders, not adapters on a random backbone.
  5. **Optimiser.** GEM-type projection with task budgets, instead of proxy-Lagrangian duals on leakage.
- **What carries over as a warning.** Training through a linear eraser invited nonlinear re-encoding: prior PCRL
  shows it, and LEACE §7 predicts it. In-training adversaries are beaten by fresh attackers
  (`elazar2018adversarial` Table 3; `song2020overlearning`). J's critics are a finite family, so the audit's fresh
  attackers can exceed them (`THEORY_REVIEW.md` §10c).

## 9. Ingredient table

| Ingredient | Established source | Specific to this integration |
|---|---|---|
| Separate per-purpose encoders trained together | PCRL purpose-specific encoders (in-house); multi-task learning | Plain MLPs instead of LoRA on a random backbone; nothing algorithmic |
| CE critics on a representation, encoder ascends their loss | Edwards & Storkey 2016 (`edwards2016censoring`); LAFTR §4; Song `C2` | None |
| Surrogate `1 - min_a CE_a / H_fit(S)` | Song §2.3 `C2` with empirical `p(u)`; Barber-Agakov bound (`barber2003im`); uncertainty-coefficient normalisation | Max over a two-MLP + constant bank (Elazar & Goldberg sum an ensemble instead) |
| Coalition critic on concatenated releases | PCRL cross-purpose constraint (linear, in-house); Taylor `I(R̂_k, R̂^{k-1}; X)`; Ganta 2008 | Nonlinear critics on complete views of two learned recipients |
| Per-purpose LEACE in the forward path, periodic refit | LEACE Thm 4.2-4.3, §6 scrubbing, §7 "incorporate into training"; PCRL erase layer / joint LEACE warm start | Refit schedule per purpose at epoch boundaries |
| Release `[r, centred logits]` | Softmax shift invariance (prior review); log-linear guardedness | Release-surface accounting only; no Bayes information added |
| `min ‖d−p‖²` s.t. active `a_j·d ≤ 0` | GEM Eq. 8; A-GEM Eq. 11; PCGrad; Rosen 1960 | Privacy gradient as proposal, ε-band task budgets as guards; reduces to per-encoder A-GEM |
| Backtracking acceptance on budgets | Standard line search with feasibility check | Applied to fitting-role guard losses only |
| Fixed-weight penalty control (JP) | Song Eq. 11; LAFTR Eq. 2 | None |
| Sequential controls (S12/S21) | Taylor's online setting (problem, not algorithm) | Neural heuristic; not their method |
| FARE and γ = 0 twin (F/F0) | FARE §6; in-house compression control | FARE's certificate extends to the coalition as a `k1·k2`-cell encoder |
| Fresh nonlinear audit on views and coalition | Elazar & Goldberg 2018; Moyer 2018; Song & Shmatikov 2020 | Coalition audit with ignore-other-view candidates (`THEORY_REVIEW.md` §3) |

## 10. Verdict on algorithmic novelty (kept separate from any empirical result)

**Every component is established.**
- The update is GEM's projection (exactly A-GEM per encoder, given separate encoders) applied to an adversarial-MI
  surrogate that is Song et al.'s `C2`.
- The erasure is official LEACE run inside training, as LEACE §7 proposes and as PCRL already did.
- The coalition term is PCRL's own cross-purpose idea, moved from a linear R² check to nonlinear critics on complete
  views.

**What is specific is the integration**: a two-recipient, collusion-aware objective on complete views, with
per-purpose linear erasure and a guard-projected update, evaluated by a J-vs-L contrast at equal information. Call it
"an adaptation of gradient-episodic-memory projection and adversarial-MI penalties to a coalition-aware
multi-recipient release". Claim no new optimiser, surrogate or guarantee, and no priority on collusion (Taylor et al.
2026 and PCRL's cross-purpose constraint precede it).

A J > L coalition-AUC result, if obtained, would be an empirical finding about this integration. It would not
upgrade the algorithmic novelty, and it would need the confound controls listed in `THEORY_REVIEW.md` §5-§6.
