# Proposal comparison (phase 2): SCM and SCAR against the closest prior work

Role: closest-work and baseline reviewer, PCRL structural-mechanism campaign. Written 2026-10-16.

**Inputs (read, not edited).**
- `roles/designer_1/PROPOSAL.md` (SCM), sha256 `6bc0617b286dccd2e7b428c51b15c6399696cacf5737415188d6db46f0aa4002`.
- `roles/designer_2/PROPOSAL.md` (SCAR), sha256 `74b420674660468e4cbf6373b0353611b44da79721fd9abf00eb1591753a2437`.
- Both hashes were recomputed today.

**Basis.** Row IDs (F1, P1, P3, …) refer to `FIELD_MATRIX.md` in this folder. No new sources were opened for this phase.

**Rules.** No priority claim is made. "Not found" means not found in the sources listed in `FIELD_MATRIX.md` §6. **[D]** marks the reviewer's elementary derivation.

---

## 0. The shared skeleton

The two proposals are the same family and differ in their components. Both:
1. fit task-only CatBoost teachers on MODEL_FIT with the T_TREE registry;
2. quantise the 7-dimensional teacher-posterior space into fine cells (SCM: 512 by KL-Bregman k-means on out-of-fold posteriors; SCAR: 8,192 by Euclidean k-means on in-sample posteriors);
3. merge the fine cells greedily, with exact plug-in merge costs, using a cross-fitted SEX propensity π̂ in training only;
4. release, from **one shared coarse cell C**:
   - to the occupation recipient, the cell's pooled teacher posterior;
   - to the income recipient, the pooled posterior of a fixed, SEX-blind, equal-mass income group or bin of cells (32 groups in SCM, 64 bins in SCAR).

The deployed map reads the six fields only. Both designers state that no algorithmic novelty is claimed.

**Consequences of this skeleton** [D, as both proposals themselves state].
- **The pair adds nothing beyond the occupation release.** The occupation recipient's vector identifies C, and R1 is a function of C. For a Bayes reader, the coalition view therefore equals the occupation recipient's view.
- **The coalition target becomes a single-release target on C.** Protecting the pair means coarsening the partition the occupation recipient already sees.
- **This is the deterministic analogue of an existing criterion.** "Colluders do no better than the best single copy" is Li et al.'s criterion (S1), achieved there with correlated noise. That one representation serves several consumers is FARE's transfer use, and FARE's certificate scope covers any classifier of the shared cells (F1). None of this is new as a collusion mechanism.

---

## 1. SCM (designer 1): Shared-Cell Merging

### 1.1 Is it equivalent to an existing published method?

**No, not component for component.** Its closest method is the **privacy funnel, Alg. 1** (P1). Other components come from agglomerative IB (P2, which is SCM's twin), Kamiran et al.'s greedy ratio rule (K1) and FARE's shared-representation scope (F1).

| Component | SCM | Closest published | Match |
|---|---|---|---|
| Objective | min L(C) = Σ_c (n_c/N)(π_c − π̄)², the between-cell variance of π̂ (a Bayes SEX-Brier advantage), s.t. per-metric budgets Δ_k ≤ B_k | P1: min I(S; C) s.t. I(X; C) ≥ R | **Same structure**: minimise leakage subject to a utility constraint. The leakage functional is quadratic (a Ward cost on π) rather than logarithmic; both are Bregman merge costs. The utility side differs: several task-metric budgets instead of one input-information constraint |
| Partition construction | KL-Bregman k-means of out-of-fold teacher posteriors (512 cells), then merges | P1, P2: singletons of a known finite alphabet, then merges | The start differs: a quantiser in teacher-posterior space. This is the bridge to unseen inputs |
| Optimisation | All pairs; merge the admissible pair maximising G / Σ_k Δ_k/R_k; stop when none is admissible | P1: merge the pair with the largest privacy gain among pairs keeping the constraint; stop when none. K1 Alg. 1: greedy ratio Δdisc/Δacc under a budget | **Same algorithmic type.** A ratio priority over several budgets generalises the single-budget greedy rules |
| Task heads | Pooled out-of-fold teacher posteriors per cell (occupation) and per income group (income) | P2: merged conditionals p(y \| z); FARE as run: logistic head on one-hot cells (F2) | The same object as AIB's merged conditional under a plug-in law |
| SEX access | Cross-fitted π̂ in training only; deployment SEX-blind | P1: known P_{S,X}, map reads X only; F1: S in training only | Same access pattern |
| Release | R2 resolves C; R1 is a coarsening of C | P1: the merged symbol | The pair carries the same information as the merged symbol |
| Guarantee | Bayes data processing: AUC\*(S; V) ≤ AUC\*(S; C). No finite-sample certificate | P1 Thm 1 (MI, known law); F1 certificate (DP/TV distance on cells, finite-sample) | Weaker than FARE's. Same scope logic |

### 1.2 Substantive algorithmic difference?

**Rule check.** FIELD_MATRIX §4.3 says that "a privacy funnel or IBSI run on empirical cells with a different trade-off parameter" is not distinct.
- SCM is **more than a re-parameterisation**: the constraint set, the leakage functional and the input map all differ from the published funnel.
- Each difference is an established device or an application-specific choice:

| Candidate difference (FIELD_MATRIX §4.3) | In SCM? | Classification |
|---|---|---|
| Merge under per-task hard caps | Yes: budgets per task and metric (NLL, Brier, accuracy) plus the within-SEX occupation guard, all relative to C0, with a remaining-budget ratio priority | **Engineering variant.** The funnel has one hard constraint and RELAB one budget. The merge identities are standard (Ward for Brier; KL/JS for NLL, as in P1 Prop. 1 and P2). The accuracy cost is direct bookkeeping. Plugging the benchmark's clauses in as constraints is application-specific |
| Shared two-task partition with a coalition criterion | Yes | **Combination.** The coalition criterion reduces to the SEX information of C, which the occupation recipient already sees (§0). The bound follows from data processing, within FARE's scope (F1); its goal is Li et al.'s best-single-copy criterion (S1) |
| Output-space ties learned jointly with the partition | Only for income, via 32 fixed equal-mass groups ordered by mean income posterior, without SEX. Occupation vectors stay distinct across cells | **Not realised as an ingredient.** The income ties are a fixed post-hoc output quantiser, of the rounding-confidences or binning type (FIELD_MATRIX §4.1) |
| Generalisation to unseen inputs via teacher posteriors | Yes: averaged fold teachers, then the nearest KL centroid | **Engineering variant.** Vector quantisation of a learned embedding with established components (Bregman k-means). It is what makes a funnel deployable on these six fields, and it is also what places SCM in the output-channel lanes (§3) |

**Does any difference plausibly change what is released?**
- **Against FARE as run: yes.** SCM uses one shared partition instead of two per-task trees, posterior-space cells that can track the posterior, and pooled posteriors instead of a logistic head. Together these plausibly decide utility feasibility, which FARE failed (Z1 failed 12/12). That is a testable engineering difference (G0), not a new ingredient.
- **Against the privacy funnel: the released object is the same type** (a merged cell and its conditional). Only the partition selected changes.

### 1.3 Overlap with closed lanes

See §3, which applies to both proposals.

### 1.4 Objections (prior-work and baseline side)

**SERIOUS**
- **SCM-S1. Closed-lane overlap in substance (§3).** The privacy step acts only on a task-only teacher's outputs, and refitting that teacher does not change this. The closed lanes' common failure mode is that fitting-pool checks did not hold on held-out data (`CLOSED_LANES.md` L57). It applies directly to SCM's MODEL_FIT plug-in budgets. Cross-fitting mitigates it for the teacher posteriors but not for the merge selection on noisy π_c (the designer's W3). G0 and G2 must run before any reader, as SCM already registers.
- **SCM-S2. Attribution baseline missing.** Without the **published-objective arm** in §4 (an IBSI or funnel merge on the same fine cells, without per-task caps), a recipe benefit cannot be attributed to SCM's modifications rather than to running the published objective. The twin (AIB-like, task-only) isolates SEX-awareness, not SCM's departures from the published funnel or IBSI. This matters for claim 7 and for the paper's framing, not for eligibility.
- **SCM-S3. Framing of the coalition result.** "Both outputs are functions of the same final cell" must be presented as a data-processing consequence of a shared representation, with Li et al. (S1) and FARE (F1) cited. It is not a collusion-resistance mechanism. The pair endpoint and the occupation-recipient endpoint coincide for Bayes readers, so recipient-level and pair-level claims are not independent evidence.

**MINOR**
- "Agglomerative IB … which is the twin" is approximate. AIB merges by the loss in I(Z; Y) (NLL) alone; SCM's twin merges by a multi-metric, budget-normalised cost. Say "AIB-like".
- The dominance argument in §1 ("partition + cell posteriors weakly dominates every deterministic release") holds for the **pair** view. Replacing f₁ by P(Y₁ \| C) can make the income recipient alone **more** informative, because it may now resolve more of C. Recipient worsening is an endpoint, so the statement should be scoped to AUC\*(S; V).
- Citations not verified by me today: Banerjee et al., JMLR 2005, and Ward linkage. Verified today: Makhdoumi et al. (P1), Slonim and Tishby (P2), Chierichetti et al. (P5); SCM's descriptions of those three are accurate.
- No certificate is claimed, which is consistent with the release. A FARE-style certificate on C at 512 cells would almost certainly be vacuous at these role sizes [D].

### 1.5 Verdict line for SCM

See §6.

---

## 2. SCAR (designer 2): shared-cell agglomerative release

### 2.1 Is it equivalent to an existing published method?

**No, not component for component.** Its closest method is **IB with side information (IBSI)** (P3), with Y⁺ := the two task labels under the teacher plug-in law (SCAR sums the per-task terms) and Y⁻ := SEX under the propensity plug-in law, solved by the greedy agglomerative hard clustering that IBSI itself lists (AIB, P2). Its leakage merge cost n_c KL_B(π̄_c ‖ π̄_m) + n_c′ KL_B(π̄_c′ ‖ π̄_m) is the privacy funnel's Prop. 1 cost (P1) [D].

| Component | SCAR | Closest published | Match |
|---|---|---|---|
| Objective | min T(M) + λ·Î(S; C_M) at K cells, with per-clause caps. T is the cap-normalised NLL + Brier + accuracy excess for two tasks | P3: ℒ = I(X; T) − β[I(T; Y⁺) − γI(T; Y⁻)]. In the hard, fixed-cardinality form, merges trade the loss in I(T; Y⁺) against the loss in I(T; Y⁻) | **Same Lagrangian form.** The NLL part of T is the loss in I(C; Y) under the plug-in law [D]. SCAR adds Brier and accuracy terms and hard caps |
| Partition construction | Euclidean k-means on in-sample teacher posteriors (8,192 cells), then agglomeration to K | P2, P3: finite alphabet | Bridge to unseen inputs, as in SCM |
| Optimisation | Greedy with a lazy heap on a 16-nearest-neighbour graph plus its minimum spanning tree; merge the admissible pair with the smallest ΔT − λΔI/N | P2: greedy over all pairs; P3 lists greedy agglomerative clustering | **Same type.** The candidate graph is a computational restriction |
| Task heads | Pooled teacher posterior per cell; 64 equal-mass income bins | P2 merged conditionals | Same object |
| SEX access | 2-fold cross-fitted π̂ in training only | P3: known p(x, y⁻) | Same pattern |
| Release | Pair ≡ C; R1 binned | P3: cluster | Same information |
| Guarantee | None claimed beyond Bayes data processing | None | Same |

### 2.2 Substantive algorithmic difference?

**Rule check (FIELD_MATRIX §4.3).**
- SCAR is the closest of the two to "IBSI run on empirical cells with a different trade-off parameter". Its leakage term and its Lagrangian form are IBSI's, and λ plays γ's role.
- What goes beyond a re-parameterisation is the richer task cost (Brier and accuracy alongside NLL, normalised by the caps), the per-clause hard caps, and the teacher-posterior input map. Each is an established device or an application-specific choice:

| Candidate difference | In SCAR? | Classification |
|---|---|---|
| Per-task hard caps | Yes: six cumulative caps including the fine quantisation, plus a cap-normalised cost | **Engineering variant.** A constrained version of a Lagrangian greedy merge with standard merge identities |
| Shared two-task partition with a coalition criterion | Yes | **Combination** (as for SCM, §0) |
| Output-space ties | Only for income: 64 fixed SEX-blind equal-mass bins. The proposal notes that "distinct cells with near-equal vectors stay distinguishable" | **Not realised as an ingredient** |
| Generalisation via teacher posteriors | Yes: the single refit teacher, then the nearest Euclidean centroid | **Engineering variant** |

**Does any difference plausibly change what is released?**
- **Against FARE as run: yes**, through feasibility, as for SCM.
- **Against IBSI or the funnel: no change of object.** The partition selected changes.

### 2.3 Overlap with closed lanes

See §3. SCAR's own §10 asks the reviewer to judge the latent-code overlap; that judgement is in §3.

### 2.4 Objections (prior-work and baseline side)

**SERIOUS**
- **SCAR-S1. Closed-lane overlap in substance (§3).** The same point as SCM-S1, with one aggravating detail: the fine cells, the plug-in costs and the pooled heads use the teacher's **in-sample** MODEL_FIT posteriors. That is the fitting-pool optimism the closed lanes measured, and SCM avoids it with out-of-fold posteriors. With about 20,000 / 8,192 ≈ 2.4 MODEL_FIT people per fine cell, early merge decisions rest on very small cells.
- **SCAR-S2. Attribution baseline missing.** SCAR differs from published IBSI mainly by its caps and extra metrics. The published-objective arm in §4 is therefore the minimum needed before any benefit can be credited to SCAR rather than to IBSI.
- **SCAR-S3. Framing of the coalition result.** The same as SCM-S3. SCAR's statement that "linking the recipients cannot restore it" is a data-processing fact of a shared partition, within FARE's scope, and its goal is Li et al.'s criterion.

**MINOR**
- **FARE is described by the repository's run, not the published method.** "FARE grows one fair CART-style tree per task: at most 100 cells, … a logistic head" is how this repository ran FARE. The published method grows one tree, releases cell medians, leaves heads to consumers, and allows k̄ up to 200 (F1).
- **Two arguments rest on seed-0 numbers.** The motivation from JOINT_TREE and LOCAL_TREE and the back-of-envelope in §4 are fitted-reader results on DEV_REPORT, seed 0. They are fine as motivation, not as evidence for the family.
- **One citation not verified by me today.** Banerjee et al., JMLR 2005. Verified today: IBSI (P3), AIB (P2), and the privacy funnel (P1, including Prop. 1, whose cost matches SCAR's ΔI form). SCAR's IBSI attribution ("utility is … task-posterior resolution") is accurate.

### 2.5 Verdict line for SCAR

See §6.

---

## 3. Closed lanes: does refitting the teachers escape the frozen-output lanes? (both proposals)

**Short answer: it escapes the letter of the closed lanes but not their substance** [D, with `CLOSED_LANES.md` L27–40 as the evidence].

**Why not the substance.**
- In both proposals the release is h(q(x)): q is a task-only teacher trained without SEX, and h is a SEX-aware, many-to-one map of q's 7-dimensional output (quantiser, merge table, pooled heads).
- Refitting q with the T_TREE registry reproduces the frozen teacher's function class (SCAR uses T_TREE's settings outright; SCM cross-fits them). The privacy action is still a post-hoc map on a task-only model's outputs.
- Data processing then bounds both proposals by the teacher: AUC\*(S; V) ≤ AUC\*(S; q(X)), and each released task vector is a cell average of q.
- No input distinction is merged unless the teacher has already mapped it to a nearby posterior. "Which input distinctions to merge" is learned only through q's level sets, which is the campaign hypothesis in a weak form.

**Lane by lane.**
- **Frozen-code rescue** (decoders, calibrators, mixtures on frozen outputs). The direction differs: the closed lane rescued the utility of privacy-trained codes, whereas here privacy is applied to task-only outputs. The operator is related, though: each pooled head is a cell average of teacher posteriors, i.e. a binning of the teacher's outputs. Partial overlap.
- **Frozen-bank channels** (re-routing a frozen bank's codes, best gain 0.006; linear-programming change kernels, 0.0020; complete-view policies, 0.0024; replacement coder on ACS 2018 with quantisation cost +0.021). **This is the strongest overlap.** SCM says so itself: "SCM *is* a deterministic channel on its own teacher posteriors."
  - What differs: deterministic hard merges with pooled, calibrated heads instead of stochastic kernels or re-routing; a refit teacher instead of the historical bank; finer resolution (512 or 8,192 cells); one cell shared by both recipients; per-clause plug-in budgets.
  - These are construction differences. The lane's measured result, a privacy gain of 0.002–0.006 from channels on fixed outputs, is the direct prior for the G2 gates. Both designers' priors (P(G2 \| G0) ≈ 0.25–0.3) are consistent with it.
- **Latent codes** (discrete codes with heads; the finest decision-preserving code cost +0.020 nats occupation log loss; 10-bit codes +0.032). These overlap as finite codes with heads.
  - What differs: the codes are not trained neural bottlenecks but a quantiser in posterior space, with Bregman centroids that match the NLL (SCM) or Brier (SCAR) cost.
  - The lane's bottleneck, the log-loss cost of the code, is exactly what G0 tests on ACS. A passing G0 would be new evidence; a failing G0 would repeat the lane.

**Consequence for the coordinator.**
- Whether "a SEX-aware hard quantiser of a refit task-only teacher's posteriors" counts as a structural change under `DESIGNER_BRIEF.md` is the coordinator's ruling.
- From the prior-work side, the family is best described as **an empirical, deployable privacy funnel or IBSI on task-only posterior cells, inside the output-channel family**. It is not a learned input representation.

---

## 4. Which faithful published baseline should be run if the family is piloted

1. **Required as a descriptive attribution arm (cheap: same fine cells, same statistics, minutes per seed).** **PUB_IBSI**: the published IBSI objective solved by AIB-style greedy merging on the candidate's own fine cells.
   - Leakage term: the plug-in I(C; S) (the P1 Prop. 1 cost).
   - Task term: the plug-in I(C; Y₁) + I(C; Y₂), i.e. NLL only, with no Brier or accuracy terms and **no per-clause caps**. The sum is used because only per-task teacher posteriors exist.
   - Stopped at the candidate's K, with γ over three preset values chosen by the frozen MODEL_SELECT rule.
   - For SCAR this is SCAR minus its caps and extra metrics. For SCM it additionally swaps the variance leakage for MI and the budgets for a Lagrangian.
   - It isolates the proposals' engineering departures.
2. **Optional: the literal privacy funnel, Alg. 1 (P1).** It solves min I(S; C) s.t. I(C0; C) ≥ R, with R set so the path ends at K. Its utility is task-blind, so it will likely be utility-infeasible, but it is the published funnel as written.
3. **FARE.** The existing F1–F3 and Z1–Z3 outputs **suffice as the published raw-space partition learner already run on this benchmark**. They are utility-ineligible, so they enter claim 4 descriptively only, and need not be re-run for eligibility.
   - They do **not** test the shared-cell configuration both proposals rely on.
   - If the coordinator wants that attribution, a faithful shared-cell FARE is an optional descriptive control. It follows the paper's transfer use (one official tree grown on one task label, k̄ ≤ 200, both recipients' heads on the same cells).
   - Given Z1, that control is expected to be infeasible on utility as well; this is an expectation, not a measurement.
4. **Already present in both proposals.** The task-only twin (AIB-like merging) is the matched recipe-effect control. NEURAL_RAW ρ1 E2 is the mandatory competitive comparator. Neither replaces the PUB_IBSI arm for attribution.

---

## 5. Which variant is the more defensible representative of the family

**SCM, narrowly.** Neither contains a new ingredient, so the choice turns on estimation hygiene and closeness to the binding clauses, not on novelty.

**For SCM:**
- out-of-fold teacher posteriors for both the cell table and the plug-in costs, which avoids the in-sample optimism that sank the closed lanes' fitting-pool checks;
- the within-SEX occupation guard is a budget, not just a report; the subgroup guard is the clause on which MMP failed;
- KL-Bregman cells whose distortion equals the plug-in NLL cost;
- all-pairs exact greedy search, with no graph restriction;
- about 39 people per initial cell, against about 2.4 for SCAR.

**For SCAR:**
- its MI leakage maps one to one onto the published objective (IBSI and P1), which makes attribution cleaner;
- the NR_PROJECTED diagnostic averages NEURAL_RAW's pair within SCAR's cells. It answers `CLOSED_LANES.md`'s demand to "name information the neural control demonstrably retains", and nothing in SCM does that;
- a permuted-π̂ arm and lookup-reader proxies serve as gates (G3, G4);
- finer initial cells lower the G0 risk.

**Suggestion (the coordinator's call).** If the family is piloted, use SCM as the representative, graft SCAR's NR_PROJECTED diagnostic and its permuted-SEX and lookup gates onto it, and add the PUB_IBSI arm of §4.

---

## 6. Verdicts

- **SCM (designer 1):** ENGINEERING_COMBINATION_NO_NEW_INGREDIENT. It is a privacy-funnel-type constrained greedy merge (P1), with a quadratic leakage functional, several per-task budgets and a ratio priority, run on teacher-posterior cells shared by both recipients. Each part is established, or is an application-specific engineering choice.
- **SCAR (designer 2):** ENGINEERING_COMBINATION_NO_NEW_INGREDIENT. It is an agglomerative IBSI-type Lagrangian merge (P3, with P1's leakage cost), with cap-normalised Brier, NLL and accuracy costs and per-clause hard caps, run on teacher-posterior cells shared by both recipients. It is closest to "IBSI on empirical cells", and goes beyond that only by engineering.
