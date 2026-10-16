# Mathematical review: the injectivity claim and what depends on it

**Role and limits.** Mathematical reviewer, correction pass v1, 2026-10-16. This is the only file I wrote. I edited no other file, committed nothing and ran no numerical job. Every number below is quoted from the record, except two pieces of hand arithmetic, which are labelled.

**Access.** Markdown and JSON reports and Python source code only. I opened no array or model file, nothing under any `.private/` directory, no raw ACS file and no real-person data.

**Citations.**
- `R/` means `results/pcrl_mechanism_design_v1/`.
- Report line numbers are those of the committed campaign state (`b2958c76d`). The coordinator was editing several of these files in the working tree during this review, so I cite the committed lines.
- Code line numbers refer to this worktree. The `acs_replication` modules cited were last changed in `02e003405`; I did not check them against the code that produced the published comparator outputs beyond that history.
- Checks A–E are the laws in `R/corrections/CHECKS_PREREGISTRATION.md`. The coordinator's run of them (`R/corrections/EXACT_CHECKS.json`, uncommitted when I read it) reports 28 of 28 passing. I do not treat that run as independent verification.

## 0. Findings in brief

1. **No valid proof exists that any evaluated ACS release is injective, at any level.**
   - The 05:36Z premise is false as stated. The tree releases (T_TREE, TASK_REFIT, LOCAL_TREE, JOINT_TREE, the hybrids, FAGTB, and MMP and its twin) are piecewise constant, and FARE has finitely many outputs.
   - Neither continuity nor determinism implies injectivity.
   - The only collision counts in the record are on invented data, and they show collisions.
2. **Every evaluated model reads the same pooled 145-column encoding E.** The valid general statement is AUC\*(S;V) ≤ AUC\*(S;E(X)) ≤ AUC\*(S;X_raw). Equality at either step is UNKNOWN.
3. **Equality holds if and only if the release preserves the SEX posterior** (proved in §1(d); the converse holds). So a Bayes-level reduction requires giving the same output to inputs with different posteriors. Merging is necessary for this, but not sufficient.
4. **The downstream assertions (i)–(v) are unsupported**, and (ii) and (iii) are false as universal statements: check B is a deterministic release under the current output contract that removes all SEX information at zero utility cost.
   - The opposite claims are not established either.
   - Whether any ACS release has AUC\* below AUC\*(S;E(X)), or below AUC\*(S;X_raw), is UNKNOWN.
5. **MMP.**
   - Its price identities are exact under a known law with positivity. For fitted real models the cost is not known in advance.
   - "If male / if female" should read "the fitted model evaluated with its SEX input set to each value".
   - Removal is not confined to pure-proxy fields, and MMP can raise recovery.
6. **RCCS.**
   - The rejection "as presented" stands, on S5.
   - "Only hides from linear and quadratic attackers" is contradicted by the verifier's fitted HGB and MLP results in a smooth law.
   - "Never lowers optimal recovery" holds only under a condition.
   - The mechanism is unresolved pending the verifier's repairs. No method win follows.
7. **The final status still follows under the campaign's rule 4**, from synthetic-law evidence that does not use the injectivity claim. No universal impossibility theorem exists in the record.

## 1. Correct statements

**Setting.**
- S ∈ {0, 1}, with p = P(S = 1) ∈ (0, 1).
- X is an input: the six raw fields X_raw, or the encoded input E(X_raw), as stated in each place.
- R is the released view: one recipient's output or the linked pair. It is generated from X by a measurable map, or by a kernel whose randomness is independent of S given X. This is the Markov chain S → X → R.
- All variables live on standard Borel spaces.
- π(x) = P(S=1 | X=x) and π_R(r) = P(S=1 | R=r).

**Optimal ROC and AUC\*.** For a view Z, the optimal ROC is the Neyman–Pearson power function:
- β\*_Z(α) = sup { P(φ=1 | S=1) : φ is a randomised test of Z with P(φ=1 | S=0) ≤ α }.
- It is concave and nondecreasing.
- AUC\*(S;Z) = ∫₀¹ β\*_Z(α) dα = P(π_Z(Z₁) > π_Z(Z₀)) + ½ P(π_Z(Z₁) = π_Z(Z₀)), where Z_s is drawn independently from the law of Z given S = s.

This is the likelihood-ratio ranking with ties counted ½, the definition in `CHECKS_PREREGISTRATION.md`:8–10.

### (a) Data processing gives a bound, not an equality

**Statement.** Under S → X → R, β\*_R(α) ≤ β\*_X(α) for every α. Hence AUC\*(S;R) ≤ AUC\*(S;X).

**Proof.** For a test φ of R, ψ(x) = E[φ(R) | X=x] is a randomised test of X. Because R ⊥ S | X, E[ψ(X) | S=s] = P(φ(R)=1 | S=s) for s = 0, 1, so ψ has the same size and power as φ. Take suprema, then integrate. ∎

**Remarks.**
- This is an inequality only. It never gives equality by itself, and it says nothing about fitted readers.
- It holds for deterministic releases and for randomised ones whose randomness is independent of S given X. It can fail for a mechanism that reads SEX at release time.
- Applied repeatedly: AUC\*(S;R_i) ≤ AUC\*(S;V) ≤ AUC\*(S;E(X)) ≤ AUC\*(S;X_raw).

### (b) Reconstruction gives equality

**Statement.** If a measurable g satisfies g(R) = X almost surely, then AUC\*(S;R) = AUC\*(S;X).

**Proof.** X is then a function of R, so (a) applies in both directions. ∎

**Which law.** "Almost surely" refers to the population law of (X, R), the law that defines AUC\*. g may fail only on a set of population probability zero. For a discrete law, g must recover every input value of positive probability. It does not refer to a finite sample.

**Injectivity on a sample is neither necessary nor sufficient.**
- *Not sufficient.* A map can show no collision on the sample and still be 2-to-1 on the support, losing optimal AUC. In check C1 the release is 3-to-3 on the observed inputs, yet AUC\* falls to 1091/1782 from 1461/1782. In check C2, a map that merges every cell in pairs shows no sample collision with probability about exp(−n²/(4M)).
- *Not necessary.* See (c).

**Injectivity on a declared domain is not necessary either.** It is sufficient only when the domain contains the population support of the same variable. (An injective Borel map on a standard Borel space has a Borel inverse on its image.) The equality it gives is then with AUC\* of that variable: injectivity on the encoded domain gives AUC\*(S;R) = AUC\*(S;E(X)), not AUC\*(S;X_raw).

### (c) Sufficiency without reconstruction

**Statement.** If π(X) = h(R) almost surely for some measurable h, then AUC\*(S;R) = AUC\*(S;X).

**Proof.** π_R(R) = E[π(X) | R] (because S ⊥ R | X), which equals h(R) = π(X) a.s. The optimal scores of R and of X are then the same random variable, so their AUCs agree. ∎

**Example.** Check D: a 2-to-1 release that merges cells with equal posteriors keeps AUC\* = 4/5.

### (d) The converse holds

**Statement.** Under S → X → R, AUC\*(S;R) = AUC\*(S;X) implies π(X) = π_R(R) almost surely. That is, the posterior is a.s. a function of R.

**Proof.**
1. *Equal areas force equal curves.* β\*_R ≤ β\*_X pointwise by (a). Both are concave, hence continuous on (0, 1). A nonnegative continuous difference with zero integral vanishes.
2. *The optimal ROC and p determine the law of the posterior.*
   - Let G_s be the law of π_Z(Z) given S = s. Calibration gives dG₁/dG₀(t) = ((1−p)/p) · t/(1−t) on [0, 1).
   - The slope of β\*_Z at false-positive rate α is this ratio at t(α), the upper α-quantile of G₀. Since t ↦ t/(1−t) is strictly increasing, the curve determines G₀, and then G₁ on [0, 1). G₁'s mass at 1 is β\*_Z(0⁺).
   - Hence π_R(R) and π(X) have the same law.
3. *Variance step.* π_R(R) = E[π(X) | R], so E[π(X)²] = E[π_R(R)²] + E[Var(π(X) | R)]. Equal laws give E[π(X)²] = E[π_R(R)²]. Hence Var(π(X) | R) = 0 a.s., i.e. π(X) = π_R(R) a.s. ∎

**Conditions.** Binary S with 0 < p < 1; the Markov chain S → X → R; standard Borel spaces. No moment condition is needed, because π is bounded. This is the two-hypothesis case of the classical comparison of experiments; I rely only on the self-contained argument above.

**Consequences.**
- With (c): **equality holds if and only if the release preserves the SEX posterior a.s.**
  - For a deterministic R = f(X) with a discrete law, this means π is constant on the populated part of every fibre f⁻¹(r).
  - Check E verifies this on all 52 partitions of a five-cell law: exactly two keep AUC\*, and one of those two is not injective.
- A Bayes-level reduction, AUC\*(S;V) < AUC\*(S;X), requires the release to give one output to inputs with different posteriors.
  - So merging is necessary.
  - Merging cells with equal posteriors removes nothing (check D), so merging is not sufficient.

**Quantitative corollary (deterministic R = f(X)).**
- Let M be any set of inputs with π_R(f(x)) = π(x) for every x ∉ M; for example, the inputs whose populated fibre contains another input with a different posterior.
- Then 0 ≤ AUC\*(S;X) − AUC\*(S;R) ≤ P(X ∈ M | S=1) + P(X ∈ M | S=0).
- *Proof.* The two AUC integrands agree unless X₁ ∈ M or X₀ ∈ M, and each lies in [0, 1]. ∎

So "nearly injective" has a precise meaning that would give near-equality: merged fibres of small **population** mass. A sample collision count does not measure that mass (§2, finite-sample row).

### (e) Determinism, continuity and real-valued outputs do not imply injectivity

1. **Piecewise-constant maps.**
   - An oblivious-tree ensemble depends on x only through its split outcomes (`acs_replication/trees.py`:85–99).
   - It is therefore constant on every cell of the product partition: between consecutive borders of a numeric column it uses, and across one-hot levels that no tree splits on.
   - It is deterministic and real-valued but not continuous.
   - It is injective on a support only if some split separates every pair of populated cells and no two cells' leaf sums coincide.
2. **Fewer output dimensions.**
   - Suppose X has a density on an open set in ℝᵏ and f: ℝᵏ → ℝᵐ is continuous with m < k. Then f is injective on no open set (invariance of domain).
   - If f is also piecewise affine, as a ReLU network is, each affine piece has a nontrivial kernel, so f is not a.s. injective.
   - This is the situation in probe P7: 10 fields and 3 release dimensions (`PROTOTYPE_VERIFICATION.md`:121).
   - On a discrete support, as in ACS, dimension counting proves nothing in either direction. A continuous map into ℝ can be injective on any finite set, and a map with many outputs can still merge cells. Injectivity is then a combinatorial property of the fitted map on the populated cells.
3. **Pooled encodings.** A function of E(X) cannot be more injective in X than E is. The ACS encoder sends every level seen fewer than 10 times in MODEL_FIT, and every unseen level, to one OTHER column per field (§2).
4. **Serialised precision.**
   - A computed release takes values in a finite set.
   - Rounding can merge distinct real values and cannot separate equal ones. Injectivity of the real-valued idealisation therefore neither implies nor is implied by injectivity of the transmitted values.
   - The optimal attacker sees the transmitted values.
5. **"Almost surely" has no content here.** For a fixed fitted map and a fixed law, injectivity on the support is a yes-or-no fact. A genericity argument over parameters does not rescue the claim:
   - trained parameters are a deterministic function of data and seeds, not a draw from a density;
   - for ReLU networks, two inputs that switch off every first-layer unit collide on an open, hence positive-measure, set of parameters;
   - and the argument does not apply to trees at all.
6. **A non-injective map can keep all the information.** Merging cells with equal posteriors loses nothing (checks D and E). So a merge count, or its absence, says nothing about protection on its own.

## 2. Levels for the actual ACS evaluation (code and provenance only)

**What every evaluated model sees.**
- One encoder is fitted on MODEL_FIT people (`acs_replication/extract.py`:202) and applied to every role (`extract.py`:210–211). Every fitting and output path loads that encoded matrix (`data.py`:56–71).
- The encoder:
  - z-scores AGEP and WKHP with MODEL_FIT moments;
  - one-hot encodes COW, SCHL, MAR and POBP, keeping levels seen at least 10 times in MODEL_FIT;
  - sends every other level, including levels unseen in MODEL_FIT, to one OTHER column per field;
  - returns float32 (`extract.py`:109–133; `schema.py`:59–60; deployment applies the same rule in `deploy.py`:29–41, 69).
- The models that read it:
  - **U0 and Ucal:** `stage_a.py`:40–43, 53–62.
  - **T_TREE:** `stage_a.py`:66–74 and 85.
  - **TASK_REFIT, LOCAL_TREE and JOINT_TREE:** leaf-score corrections on T_TREE's routing (`stage_b.py`:1–9, 43–53; `trees.py`:11–14, 92–99). They are functions of T_TREE's leaf indices, so they are no finer than T_TREE.
  - **NEURAL_RAW:** `neural_raw_worker.py`:45–53.
  - **FAGTB:** a sum of regression trees (`fagtb.py`:12–23; `fagtb_worker.py`:43–45).
  - **FARE:** a head applied to a one-hot cell index, so at most n_cells distinct outputs (`fare_worker.py`:63–69).
  - **The pilot's MMP and twin arms:** `mechanism_design/pilot/pipeline.py`:51–83, through `sources.py`:110–112.
  - **The hybrids:** compositions of the above.

| Level | Established (code, provenance) | UNKNOWN | Implication for AUC\* |
|---|---|---|---|
| Population law | AUC\* is defined "under the population law" (`PROBLEM_SPECIFICATION.md`:39) | Whether this is the 74,091-person eligible population or a superpopulation; its support and π | Every "almost surely" is relative to this law. Under the finite-population law, any input value held by a single person has posterior 0 or 1, which inflates AUC\*(S;X); that is not the intended target. The record has no estimate of AUC\*(S;X) at any level: the verifier's 0.68–0.72 is a binormal reading used for pricing (`MATH_ASSESSMENT.md`:102) |
| Six raw fields X_raw | Discrete, not continuous: AGEP is an integer in 25–64, WKHP an integer in 1–99, and four fields are categorical strings (`schema.py`:37, 39, 43, 55) | AUC\*(S;X_raw) | Upper bound for every release: AUC\*(S;V) ≤ AUC\*(S;X_raw) |
| Encoded input E(X_raw): 145 float32 columns | The same encoder for every model (above). z-scoring is injective on the integer grids: a strictly increasing affine map whose grid spacing, 1/sd, far exceeds float32 resolution. Kept levels: COW 8, MAR 5, POBP 105, SCHL 21 (source `results/pcrl_acs_protected_tree_replication_v1/EXTRACTION_SUMMARY.json`). SCHL codes 02, 03 and 05 are not among the kept levels | Whether two or more non-kept levels of some field occur in the population (plausible for POBP; not verified, since no level counts were read). Whether pooled levels share SEX posteriors given the other fields | AUC\*(S;E(X)) ≤ AUC\*(S;X_raw), with equality iff the pooling preserves π (§1(d)). The loss is at most P(pooled with a different posterior │ S=1) + P(pooled with a different posterior │ S=0). **If pooling merges levels of positive probability, no evaluated model is injective in the raw fields, whatever the model does** |
| Single-recipient output R_i | A function of E(X). The tree arms and FAGTB are piecewise constant. FARE has finitely many outputs. U0, Ucal and NEURAL_RAW are ReLU networks with a 16-unit layer (`neural.py`:21–24). The income output is one-dimensional | Injectivity of any R_i on the populated encoded cells | AUC\*(S;R_i) ≤ AUC\*(S;V) ≤ AUC\*(S;E(X)) |
| Linked pair V | V's fibres are the intersections of R₁'s and R₂'s fibres, so the pair can separate cells that each recipient merges. A tree arm's pair is a function of the two teachers' leaf indices. MMP depends on x only through the SEX-aware teachers' split outcomes; the twin also depends on the propensity models' splits (`pipeline.py`:73–83) | For every arm: the injectivity of V, and AUC\*(S;V) | AUC\*(S;V) = AUC\*(S;E(X)) iff π_E is a.s. a function of V. Not checked for any arm |
| Finite observed sample | The only collision counts in the record come from the synthetic rehearsal: invented data (banner), an 800-row MODEL_SELECT stand-in, at most 60 trees instead of the registry's 800 (`PILOT_REHEARSAL_RESULTS.json` role_sizes and max_trees; `trees.py`:26). Comparison is bitwise, on float32 input rows and float64 release rows (`pipeline.py`:100–104). MMP had 797 and 798 distinct release pairs for 799 distinct inputs, i.e. at least two and one collisions; the twin had 799. No real-person array was read (`START_HERE.md`:13). The protocol calls the rehearsal numbers "meaningless as evidence" (`PILOT_PROTOCOL_DRAFT.md`:122) | Any count on real data | A sample shows a merge only if two members of a merged fibre are both drawn (checks C1, C2). *Illustrative arithmetic, hypothetical law:* if half the input mass lay in fibres of two cells of probability 10⁻⁵ each, the expected number of collisions among 800 draws would be C(800,2) × 25,000 × 2×10⁻¹⁰ ≈ 1.6, which is what was observed. Counts like these are compatible with heavy merging. And the MMP release was not injective even on this sample |
| Declared domain vs population support | The declared encoded domain is the AGEP grid × the WKHP grid × (kept levels + OTHER) for each categorical field. The population support is a subset | Which domain points are populated | Injectivity on a domain containing the support would be sufficient for the encoded variable. It has not been checked. It is not necessary: merges among unpopulated points, or among cells with equal posteriors, are harmless |
| Mathematical reals vs serialised precision | Inputs are float32 (`extract.py`:133; `deploy.py`:41). Tree routing compares float32 values with float32 borders, then sums float64 leaf values and applies a float64 softmax (`trees.py`:5–9, 83–99, 105–114). Network forward passes run in float32 and are cast to float64 afterwards (`neural.py`:72, 86; `neural_raw.py`:49, 57) | Whether precision-induced collisions occur on the support (not checked) | AUC\* must be taken for the transmitted values. Injectivity arguments about real numbers do not transfer |

**Is actual-model population injectivity, raw-field or encoded, established by any valid proof? No.** The record contains only the following:
- **D3** (`DEMONSTRATIONS.json`:47–64; `mechanism_design/prototype/demonstrations.py`:152–157).
  - A constructed 12-cell law with an injective release. It illustrates (b) and says nothing about ACS.
  - Its logistic reader stayed far below Bayes when n_aux rose 25-fold (0.566 → 0.570), because the logistic class cannot represent the non-monotone optimal ranking. That is a reader-class gap, not a sample-size gap.
- **The rehearsal counts above.** They are synthetic and small, used at most 60 trees, and show collisions for MMP.
- **Constructed laws whose release is injective by design:** L_NOPROXY, L_CONT_NOPROXY, designer 2's INJ arm and the verifier's RCCS law.
- **Explicit cautions that it is unverified.** The adversary reviewer marked T_TREE's injectivity UNVERIFIED (`PROPOSAL_THREAT_ASSESSMENT.md`:69–71). `HANDOFF.json`:67 lists "whether real ACS releases are injective" as not verified.

## 3. Claim-by-claim classification

**Classes.** correct; conditional; unsupported; false as stated; unresolved. "Replacement" is the wording I recommend.

### A. The injectivity claim and its direct consequences

| # | Where | Claim (short quote) | Class | Replacement | Evidence |
|---|---|---|---|---|---|
| A1 | `DECISION_LOG.md`:93 (05:36Z, coordinator's reading) | "Every evaluated ACS release is continuous and deterministic, hence almost surely injective on the populated inputs." | false as stated | "Every evaluated ACS release is a deterministic function of the pooled encoding of the six fields, and the tree arms and FARE are piecewise constant. Whether any of them is injective on the populated inputs is unknown." | §1(e); §2 |
| A2 | same | "Its population Bayes recovery is therefore AUC\*(S; X) whatever the defense." | unsupported | "Its population Bayes recovery is at most AUC\*(S; E(X)) ≤ AUC\*(S; X). It equals that bound exactly when the release preserves P(SEX \| input), which has not been checked for any defense." | §1(a), (d) |
| A3 | same | "The measured differences between defenses (0.60–0.66) are differences in what a fixed slate of readers learns from 8,001 labelled releases." | unsupported (as an exclusive attribution) | "The measured values (0.60–0.66) are what a fixed reader slate achieves with 8,001 labelled releases. How much of each difference reflects optimal recovery, and how much learnability, is unknown." | decomposition after group B |
| A4 | same | "Both designers state this limit themselves (designer 1 §3.5; designer 2 §3.5)." | correct, as a report | Add: "Designer 1 hedges it ('essentially', '≈'); designer 2 asserts it for the comparators without evidence." | E9, E12 |
| A5 | `DECISION_LOG.md`:143 (06:12Z) | "…wherever fitted teachers do not merge cells, which is the realistic case" | unsupported | "…wherever fitted teachers do not merge cells with different SEX posteriors. Whether that holds on the Florida data is unknown." | `HANDOFF.json`:67 |
| A6 | `DECISION_LOG.md`:145 (06:12Z) | "showed fitted SEX-aware CatBoost releases to be essentially injective: 797–798 release pairs for 799 distinct inputs on stand-ins" | false as stated | "On an 800-row synthetic stand-in with at most 60 trees, the MMP release had 797 and 798 distinct release pairs for 799 distinct inputs, so at least one collision per seed; the twin had 799. A sample count on invented data cannot certify that any real release is injective." | §2, finite-sample row; checks C1, C2 |
| A7 | `START_HERE.md`:46 (item 8) | "every evaluated release is essentially injective" | unsupported | "Whether any evaluated release is injective is unknown. All of them read an encoding that pools rare category levels." | §2 |
| A8 | `START_HERE.md`:47 | "…recovers as much as it could from the raw fields: AUC\*(S; V) = AUC\*(S; X). … D3 shows this." | unsupported; D3 misattributed | "The optimal coalition reader recovers at most AUC\*(S; X), and exactly that only if the release preserves P(SEX \| input). D3 illustrates this implication on a constructed 12-cell law; it does not show that ACS releases are injective." | §1(b), (d); §2 |
| A9 | `START_HERE.md`:48 | "All measured differences between defenses … are differences in what a fixed slate of readers learns…" | unsupported | As A3 | As A3 |
| A10 | `START_HERE.md`:49 | "Reader-independent protection requires merging input cells, which the closed lanes found too costly at the caps." | conditional (first clause); unsupported (second) | "Lowering the optimal reader's recovery requires giving the same output to inputs with different SEX posteriors; this is necessary, not sufficient. The closed lanes found the specific codes they tried too costly at the caps, mostly on Adult. That does not rule out other merges." | §1(d); `CLOSED_LANES.md`:27–31 |
| A11 | `ADVISOR_BRIEF.md`:24 (§2.1) | "Measured protection is learnability. (The verifier's probes confirm this independently.)" | false as stated | "Measured protection is fitted-reader recovery. How it splits into removed information and learnability is unknown. The verifier's probes concern synthetic laws, and the one probe with a non-injective release (P7) found Bayes-level changes of both signs." | `PROTOTYPE_VERIFICATION.md`:117–138 |
| A12 | `ADVISOR_BRIEF.md`:24 | "Every release evaluated so far is a deterministic continuous function of the six fields, and essentially one-to-one." | false as stated | "Every release evaluated so far is a deterministic function of a pooled encoding of the six fields, and most are piecewise constant. Whether any is one-to-one on the population is unknown." | §2 |
| A13 | `ADVISOR_BRIEF.md`:24 | "A population-optimal attacker on the linked pair therefore recovers exactly what it could from the raw fields. This holds for the trees, the hybrids, MMP and the neural control." | unsupported | "A population-optimal attacker on the linked pair recovers at most what it could from the raw fields, and exactly that only if the pair preserves P(SEX \| fields). This is not established for any of these releases." | §1; §2 |
| A14 | `ADVISOR_BRIEF.md`:25 | "Every measured difference, including the 0.020 targets, is about how much a fixed set of seven attackers learns from about 8,000 labelled examples." | conditional (true of what is measured; unsupported if read as "only learnability") | "Every measured difference, including the 0.020 targets, is a difference in what a fixed set of seven readers achieved with about 8,000 labelled releases." | As A3 |
| A15 | `ADVISOR_BRIEF.md`:26 | "Protection that does not depend on the attacker requires merging inputs. Earlier campaigns found that too costly…" | conditional; unsupported | As A10 | As A10 |
| A16 | `ADVISOR_BRIEF.md`:56 (§4) | "Under the current contract no deterministic release can beat the raw-field leakage against an optimal attacker, so the 0.020 margin over the neural control is a contest in learnability." | false as stated (first clause); unsupported (second) | "Under the current contract, a deterministic release lowers the optimal attacker's recovery below the raw-field level only by giving the same output to inputs with different SEX posteriors. That is possible in principle: an exact constructed example does it at no utility cost. It has been neither shown nor ruled out at the caps on Florida. The 0.020 margin is a contest in fitted-reader recovery at a fixed attacker budget." | Check B; §1(d) |
| A17 | `ADVISOR_BRIEF.md`:61 (§4, option 2) | "a contract change that makes attacker-independent protection possible: coarsened or randomised outputs … Earlier coarsening attempts cost too much occupation confidence, so this means accepting looser utility or a different utility definition" | false as stated (coarsening needs no contract change); unsupported (the inference from earlier attempts) | "Should the program target attacker-independent protection? Within the current contract this means many-to-one probability releases. Randomised releases are a declared contract change that may widen the trade-offs. The coarsenings tried so far failed for the codes tried; whether any merge meets the caps on Florida is open." | `PROBLEM_SPECIFICATION.md`:19–21, 127; source `SCIENCE_LOCK.json` `release_contract`; check B |
| A18 | `METHOD_SPECIFICATION.md`:42 (§3) | "With fitted continuous teachers, m̂ is essentially injective on the populated inputs, so its Bayes recovery equals AUC\*(S; X), exactly as for the neural control and the trees. Any real-data benefit is therefore a reduction in what readers … can learn" | false as stated (the premise); unsupported (the conclusions) | "The fitted teachers are piecewise constant, so m̂ is constant on the cells of the SEX-aware trees' splits. Whether it gives the same output to populated cells with different SEX posteriors is unknown. A real-data effect could therefore have a Bayes-level component, a learnability component, or both." | §1(e)1; `pipeline.py`:73–83 |
| A19 | `METHOD_SPECIFICATION.md`:143 (§9) | "Fitted SEX-aware trees are injective in practice, giving about 1.0 release pair per distinct input in the synthetic rehearsal. A real-data benefit would therefore have needed exactly this leg." | false as stated; unsupported | "In the synthetic rehearsal (invented data, at most 60 trees, 800 rows), the MMP release had 797–798 distinct release pairs for 799 distinct inputs. That cannot show whether real releases merge cells, so the real-data relevance of the merging leg is unknown." The preceding sentence ("with readers near Bayes and an injective release…") is correct for that law. | §2, finite-sample row |
| A20 | `METHOD_SPECIFICATION.md`:154 (§9 verdict) | "The rationale is contradicted where it matters for real data." | unsupported | "The rationale's learnability prediction failed in the pre-registered synthetic law, and the verifier's probe showed that MMP's effect can have either sign. Whether the Florida data have the merging structure in which MMP removes information is unknown." | §6 |
| A21 | `PROTOTYPE_REPORT.md`:95 (verdict) | "Fitted SEX-aware CatBoost releases are essentially injective, so they do not merge cells. The synthetic pilot rehearsal found 797–798 distinct release pairs for 799 distinct inputs." | false as stated (the counts cited are merges) | As A6 | §2, finite-sample row |
| A22 | `PILOT_PROTOCOL_DRAFT.md`:5, 79 | "Fitted SEX-aware trees do not merge cells." / "If MMP's releases are injective, as the synthetic rehearsal suggests…" | false as stated; unsupported | As A6. A working-tree draft already corrects the D0 row. | §2, finite-sample row |
| A23 | `PROTOTYPE_REPORT.md`:78, 117 | L_CONT_NOPROXY: "MMP's release is still injective, so its Bayes AUC equals the twin's" | correct (for that constructed law) | Keep, scoped to the law | Exact value 0.7216 for both (`PROTOTYPE_VERIFICATION.md`:92–93) |

### B. The five downstream assertions

| # | Claim | Class | Replacement | Evidence |
|---|---|---|---|---|
| (i) | All measured differences between defenses are only learnability differences (A3, A9, A11, A14) | unsupported | "Measured differences combine an unknown difference in optimal recovery with an unknown difference in how close the fixed slate gets to it." | Decomposition below |
| (ii) | No deterministic release under the current contract can reduce raw-field leakage (`ADVISOR_BRIEF.md`:56) | false as stated | "A deterministic release under the current contract can reduce optimal recovery below the raw-field level. Whether one can do so within the 24 caps on Florida is unknown." | Check B, a constructed law: 6 inputs map to 3 release pairs, the release equals the full-input Bayes predictor, and AUC\* falls from 1 to 1/2 |
| (iii) | A contract change is necessary for information removal (`ADVISOR_BRIEF.md`:61; `START_HERE.md`:49) | false as stated | "Information removal is possible inside the current contract. A contract change, such as randomisation, may widen the achievable trade-offs but is not necessary." | Check B; §1(d) |
| (iv) | All utility-feasible information removal has been ruled out (implied by `ADVISOR_BRIEF.md`:26, 56, 61 and `START_HERE.md`:49) | unsupported | "No utility-feasible information removal has been found for Florida, and none has been ruled out." | No theorem in the record; exact frontiers were computed only in constructed laws; `PROBLEM_SPECIFICATION.md`:142, 151 deny such a theorem |
| (v) | Earlier failed coarsening studies establish a universal privacy-utility limit (`ADVISOR_BRIEF.md`:26, 61; `START_HERE.md`:49) | unsupported | "Earlier coarsening studies failed for the specific codes, capacities and decoders tried, mostly on Adult and measured with fitted readers. They rule out those recipes, not every merge." | `CLOSED_LANES.md`:27–39; `PROBLEM_SPECIFICATION.md`:142 |

**Decomposition behind A3 and (i).** For defense D:
- Let Â_D be the measured DEV_REPORT recovery, g_D the reader selected on ATTACK_VALIDATION, and A(g_D) that reader's population AUC.
- Then Â_D = AUC\*(S;V_D) − [AUC\*(S;V_D) − A(g_D)] + [Â_D − A(g_D)].
  - The middle bracket is ≥ 0, because no reader beats the optimum. It mixes a reader-class gap and a finite-sample gap; D3 shows the reader-class part can dominate.
  - The last bracket is DEV_REPORT sampling error.
- A difference between defenses splits the same way. Assertion (i) sets the difference of the first terms to zero for every pair of defenses. Nothing in the record measures it.

**Fixed-reader results are unaffected by this correction.** They remain valid empirical evidence about the declared attacks: what the specified seven-reader slate, refitted on ATTACK_FIT and selected on ATTACK_VALIDATION, achieves on DEV_REPORT. They are lower estimates of optimal recovery for that view (`PROBLEM_SPECIFICATION.md`:38, 150). This holds whether or not any release is injective. Equally, independently of injectivity, no fixed-reader result establishes protection against arbitrary attackers.

### C. MMP summaries (details in §4)

| # | Where | Claim | Class | Replacement |
|---|---|---|---|---|
| C1 | `ADVISOR_BRIEF.md`:10 | "each model's 'if male' and 'if female' predictions are averaged" | false as stated (causal reading) | §4(c) |
| C2 | `ADVISOR_BRIEF.md`:10 | "Its utility cost is known exactly in advance." | false as stated for fitted real models | §4(b) |
| C3 | `START_HERE.md`:36 ("its price is exact"); `SELECTION_DECISION.md`:51 ("prices the deletion exactly"); `METHOD_SPECIFICATION.md`:158 ("an exactly known price") | MMP's price is exact | conditional (a population identity, given known conditionals and positivity) | "Its population price has an exact formula; the cost of a fitted release must be measured." |
| C4 | `ADVISOR_BRIEF.md`:28 | "Marginalising over SEX removes information only along pure-proxy directions." | conditional (incomplete) | §4(d) |
| C5 | `METHOD_SPECIFICATION.md`:38–39 | "only along directions where q_i^s(x) is constant … There, cells merge and AUC\*(S; V) falls to the SEX information of the task-relevant fields" | conditional | "With exact components, MMP loses SEX information relative to the inputs only where its output coincides on cells with different posteriors. That includes directions on which every within-SEX task posterior is constant, and coincidences of the mixture itself. The residual equals AUC\* of the task-relevant fields only if the release is injective in them." |
| C6 | `METHOD_SPECIFICATION.md`:27; designer 1 `PROPOSAL.md`:42 | "the counterfactual law P̃(x, s, y) = P(x)P(s)P(y \| x, s)" | false as stated (a causal term misapplied) | "the reweighted law P̃ = P(x)P(s)P(y \| x, s), in which the fields and SEX are independent" |
| C7 | `START_HERE.md`:39; `ADVISOR_BRIEF.md`:13 | "MMP removes pure-proxy SEX information exactly, at the predicted price" / "…or SEX directions that the tasks do not use" | correct (within the tested laws) | Keep, scoped to the tested laws (`PROTOTYPE_REPORT.md`:29–38; `PROTOTYPE_VERIFICATION.md`:61–68) |

### D. RCCS summaries (details in §5)

| # | Where | Claim | Class | Replacement |
|---|---|---|---|---|
| D1 | `ADVISOR_BRIEF.md`:29 | "hides it from linear and quadratic attackers" | conditional: describes the objective, not the tested effect | §5 |
| D2 | `ADVISOR_BRIEF.md`:29 | "It can fail for other attackers when the inputs are categorical" | correct as "can", for the optimal reader in a constructed law | Add: "untested for fitted readers on discrete fields" |
| D3 | `ADVISOR_BRIEF.md`:29 | "it never lowers the population-optimal attacker's recovery" | conditional | "In every tested law its optimal-reader recovery rose. It cannot fall below the base release's whenever the base pair can be recovered from the RCCS pair." |
| D4 | Coordinator's final message, as quoted in the correction brief | "its protection only works against the linear and quadratic attackers it optimises against" | false as stated for the law tested; unresolved elsewhere | §5 |
| D5 | `START_HERE.md`:37; `SELECTION_DECISION.md`:35 | "Its protection is defined against its own linear and quadratic critic class" | correct, as a description of its objective | Keep, and add that its tested effect was not confined to that class |
| D6 | `DECISION_LOG.md`:161 | "RCCS's synthetic success is a smooth-law learnability result. Its Bayes effect is ≥ 0." | correct (first sentence); conditional (second) | Add the recoverability condition from D3 |

### E. Historical role documents (classified only)

| # | Where | Claim | Class |
|---|---|---|---|
| E1 | `MATH_ASSESSMENT.md`:72 (1.3) | Exact components: "Those cells merge, and AUC\*(S;V) falls to AUC\*(S; X_task-relevant)" | conditional: the equality needs injectivity in the task-relevant fields |
| E2 | `MATH_ASSESSMENT.md`:73 | "A fitted CatBoost m̂ is constant only along columns no tree splits on." | false as stated: it is constant on every cell of the split partition (`trees.py`:85–99) |
| E3 | `MATH_ASSESSMENT.md`:73 | "AGEP and WKHP are continuous, and with up to 800 trees most columns will be used." | false as stated: both are integers (`schema.py`:37, 39). Unsupported: split usage was never measured, and the tree count is set by early stopping (`stage_a.py`:69–73) |
| E4 | `MATH_ASSESSMENT.md`:73–75 | "So m̂ is essentially injective … AUC\*(S;V_MMP) ≈ AUC\*(S;X). … The real-data effect MMP can deliver is therefore a fitted-reader learnability effect." | unsupported (§2) |
| E5 | `MATH_ASSESSMENT.md`:189 (R7; verifier's verdict "TRUE but incomplete") | "The release is injective, so AUC\*(S;V) = AUC\*(S;X), 'equally true of NEURAL_RAW and the trees'" | conditional. The implication is correct, and the premise holds in the constructed laws. "Equally true of NEURAL_RAW and the trees" and "generically it equals AUC\*(S;X)" are unsupported. The "≥ base" step is correct when the base pair is recoverable from the RCCS pair |
| E6 | `MATH_ASSESSMENT.md`:305, 320 | In the verifier's Gaussian law the RCCS release is injective, so its Bayes AUC is AUC\*(S;X) | conditional on injectivity in that law, which is asserted there |
| E7 | `PROTOTYPE_VERIFICATION.md`:30 | "With more fields than release dimensions, MMP's release depends on x only through the span T … It therefore merges every SEX-predictive direction outside T" | correct for the linear-index law family used |
| E8 | `PROTOTYPE_VERIFICATION.md`:121, 147, 150 | "3 dimensions, fewer than the number of fields, so it is not injective"; "injective in the three fields"; "when the release is easy to invert, deleting Δ changes nothing. It cannot speak to readers far below Bayes" | correct (§1(e)2) |
| E9 | Designer 1 `PROPOSAL.md`:75 (§3.5) | "A generic smooth (or very finely partitioned) deterministic release is essentially injective on the populated support, so AUC\*(S; V) ≈ AUC\*(S; X) for NEURAL_RAW, the trees and MMP. Fitted-recovery differences are differences in how learnable the signal is. MMP's population-level effect is limited to merged directions." | The first three statements are unsupported (§1(e)). The last is correct relative to X but incomplete relative to the twin (M10) |
| E10 | Designer 1 `PROPOSAL.md`:70 (§3.4) | "Those cells genuinely merge, so Bayes recovery falls along those directions" | conditional: recovery falls only if merged cells differ in posterior, and merging was never measured (`PROPOSAL_THREAT_ASSESSMENT.md`:29–33) |
| E11 | Designer 1 `PROPOSAL.md`:205 | "A smooth injective analogue would show no Bayes reduction at all." | correct, given injectivity |
| E12 | Designer 2 `PROPOSAL.md`:52 (§3.5) | "the release is injective on the input support, so its population Bayes AUC\*(S; V) equals AUC\*(S; X). That is equally true of NEURAL_RAW and the trees. … All claims are fitted-reader claims." | conditional. The implication is correct and holds for INJ in the constructed laws. It is unsupported for an ACS deployment, which is built from T_TREE and CatBoost propensity models, both piecewise constant, and for "NEURAL_RAW and the trees". The scope sentence is correct |
| E13 | Designer 2 `PROPOSAL.md`:44 | "any invertible post-processing leaves the Bayes information unchanged" | correct (§1(b)) |
| E14 | Designer 1 `CROSS_CRITIQUE.md`:13 | "the deployed release, which is injective and an explicit function of π̂" | conditional: deployment injectivity is unverified. The cancellation results in lines 16–29 are correct as exact optimal-reader computations in a constructed one-coordinate law |
| E15 | `PROPOSAL_THREAT_ASSESSMENT.md`:12–14, 29–33, 55–58, 69–71, 80–83 | The yardstick; "an aspiration until measured"; INJ raises Bayes recovery from 0.615 to 0.644; "Whether T_TREE is injective on the populated support is UNVERIFIED"; the merge diagnostic | correct. The yardstick needs "a.s. on the population support". The per-input count is descriptive and cannot certify injectivity |
| E16 | Prior-work reviewer `PROPOSAL_COMPARISON.md`:96 | "AUC\*(S;V) ≈ AUC\*(S;X) for injective releases" | correct; it is an equality for a.s.-injective releases |

### F. Problem specification

| # | Where | Claim | Class | Replacement or note |
|---|---|---|---|---|
| F1 | `PROBLEM_SPECIFICATION.md`:41 (§3) | "For a deterministic mechanism, AUC\*(S; V) ≤ AUC\*(S; X), by data processing" | correct | It also holds for randomised mechanisms whose randomness is independent of S given X. It is a bound, which the later equality claims ignored |
| F2 | `PROBLEM_SPECIFICATION.md`:38, 150 | Fitted recovery is "a finite-sample lower estimate … not a population floor" | correct | — |
| F3 | `PROBLEM_SPECIFICATION.md`:116–125 (§7) vs :19–21 (§2) | §7 lists "coarsened outputs" as a contract change, while §2 defines the contract as deterministic maps into the simplex | false as stated (internally inconsistent) | "Releasing something other than a probability vector and its argmax is a contract change: a cell label, a rounded or set-valued output, or an abstention symbol. A deterministic probability release that gives the same vector to many inputs is not a contract change; §2 already allows it, and FARE and the tree arms are such releases." The source `release_contract` sets no precision or injectivity requirement |
| F4 | `PROBLEM_SPECIFICATION.md`:142, 151 | "a trade-off observed in this fitted bank, not a universal impossibility"; "No theorem here bounds it for this data." | correct | These two lines contradict assertions (ii)–(v) |

## 4. MMP scope

### (a) Population identities and their assumptions

**Assumptions.**
- **A1. One law.** A joint population law P of (X, S, Y), with Y ∈ {1, …, K}. Every expectation below is under this P, the same law that defines π, q^s and η.
- **A2. Positivity.** 0 < π(X) < 1 almost surely, so q⁰ and q¹ are defined almost everywhere. Where positivity fails, m depends on an arbitrary convention, which in practice means extrapolation.
- **A3. Fixed weight.** w ∈ (0, 1) is fixed, not tuned on evaluation data.
- **A4. Metrics.** Summed Brier B(f) = E Σ_k (f_k(X) − 1[Y=k])². NLL N(f) = E[−log f_Y(X)], without the 1e−12 floor that the code applies (`data.py`:98). The identities carry over to the floored metric when every relevant probability is at least 1e−12.

**Identities.**
1. η = (1 − π) q⁰ + π q¹, by total probability (A1, A2).
2. For any measurable f of X: B(f) − B(η) = E‖f − η‖². The cross term vanishes because E[e_Y | X] = η.
3. For any measurable f of X with f_y > 0 wherever η_y > 0: N(f) − N(η) = E KL(η ‖ f). Otherwise the difference is +∞.
4. η − m = (π − w)(q¹ − q⁰). Hence B(m) − B(η) = E[(π − w)² ‖q¹ − q⁰‖²].
5. N(m) − N(η) = E KL(η ‖ m) ≤ E KL(Bern π ‖ Bern w) = I(S; X) + KL(Bern π̄ ‖ Bern w).
   - At w = π̄ the bound is I(S; X).
   - The inequality is data processing for KL through the kernel s ↦ q^s(· | x).
   - The excess is finite because m_y = 0 forces q⁰_y = q¹_y = 0, hence η_y = 0. This needs 0 < w < 1.

**On a finite evaluation sample.** The empirical clause differences equal these expectations plus sampling terms. For Brier, the empirical difference is E_n‖f − η‖² + 2E_n⟨f − η, η − e_Y⟩; the second term has mean zero but is not zero.

### (b) "Utility cost known exactly in advance": correction for future fitted real models

**The identities need the true conditionals, which are unknown.** A deployed MMP:
- uses fitted q̂(x, s) (`pipeline.py`:73–83);
- extrapolates them into sex-atypical cells, where one SEX is rare in MODEL_FIT;
- is scored after the 1e−12 floor;
- and is judged against fitted references: U0, and Ucal with its fitted temperature.

**The binding comparison is with r ∈ {U0, Ucal}, not with η.**
- For any functions f, r of X: Brier(f) − Brier(r) = E‖f − η‖² − E‖r − η‖², and NLL(f) − NLL(r) = E KL(η‖f) − E KL(η‖r) when both are finite.
- For f = m̂:
  E‖m̂ − η‖² − E‖r − η‖² = E[(π − π̄)² ‖q¹ − q⁰‖²] + E‖m̂ − m‖² + 2E⟨m̂ − m, m − η⟩ − E‖r − η‖².
  - The first term is the population price.
  - The second and third are estimation terms; the third can have either sign.
  - The last is the reference's own excess.
  - All four are unknown before fitting.

**Also not covered by the identities.**
- Accuracy has no identity.
- The within-SEX increases have exact decompositions, but in unknown quantities (`PROTOTYPE_VERIFICATION.md`:110).
- Finite-sample uncertainty in the clauses is a further, separate layer.

`METHOD_SPECIFICATION.md`:58–62 already gives the clause decomposition and calls real-data feasibility open. "Known exactly in advance" contradicts the campaign's own specification.

**Replacement.** "Under a known law, MMP's Brier price relative to the SEX-blind posterior is exactly E[(π − π̄)²‖q¹ − q⁰‖²]. For fitted models on real data, its cost against U0 and Ucal is not known in advance. It includes estimation and extrapolation error and the references' own excess, and it must be measured on held-out data."

### (c) "If male / if female" predictions

**What is computed.** m̂(x) = (1 − π̄) q̂(x, 0) + π̄ q̂(x, 1). Here q̂ is a fitted model of P(Y | X = x, S = s), and q̂(x, s) is that model evaluated with its SEX input slot set to s (`pipeline.py`:47–48, 79–80).

**Replacement.** "Task models are fitted with SEX as an extra input. At release, each fitted model is evaluated twice per person, with its SEX input set to each value, and the two outputs are averaged with fixed weights. These are evaluations of a fitted conditional model, not predictions of what would happen to the person under a different sex."

**What a causal or individual-counterfactual reading would also need.**
- **A causal graph** relating SEX, the six fields and the tasks. Several fields (hours, schooling, marital status, class of worker) are plausibly affected by SEX, so conditioning on them blocks part of any effect of SEX.
- **Identification assumptions.** For an interventional quantity: no unmeasured confounding. For an individual counterfactual: a structural causal model and cross-world assumptions.
- **Positivity:** 0 < π(x) < 1 at the relevant x.

**The adjustment-formula resemblance.** Σ_s P(y | x, s) P(s) is Pearl's back-door adjustment for an intervention on the *fields* X, with adjustment set {SEX}.
- It equals P(y | do(X = x)) only if SEX satisfies the back-door criterion for (X, Y): no other open back-door path, and SEX not a descendant of X.
- The record assumes this nowhere.
- Even when it holds, it is an interventional reading about setting the fields, not a counterfactual about the person's sex.

### (d) "Removal only through pure-proxy fields" versus the verifier's finding

**The claim and the finding.**
- `ADVISOR_BRIEF.md`:28 and `START_HERE.md`:39 present removal as a pure-proxy phenomenon.
- `PROTOTYPE_VERIFICATION.md`:117–138 studies a 10-field law in which every field is task-relevant. There, MMP's 3-dimensional release merges every SEX-predictive direction outside the span T of the within-SEX task effects. This lowered Bayes recovery when field-mediated and direct SEX effects were aligned. When they were opposed, deleting Δ unmasked field-mediated signal and raised recovery.

**Reconciled statement (exact components).**
- Relative to the inputs, MMP loses SEX information exactly where its output coincides on cells with different posteriors (§1(d)). That happens in two ways:
  1. along any input direction on which every within-SEX task posterior is constant: a pure-proxy field is one such direction, and a combination of task-relevant fields outside T is another;
  2. wherever the mixture itself coincides for different pairs of components.
- Relative to the SEX-blind posterior η that it replaces, MMP's fibres are in general neither finer nor coarser.
  - It can merge cells that η separates, which is removal.
  - It can separate cells that η merges or masks, which raises recovery (M10; P7 draws 1 and 2).
  - Its effect on optimal recovery therefore has no fixed sign.

**Wording.** "Pure-proxy fields" is too narrow. "Pure-proxy directions" is correct only with the general meaning above, and only relative to the inputs. For fitted models, which directions are merged is a property of the fitted trees, and that is unknown for ACS.

### (e) Observations to keep, without generalising them

All of these are constructed laws with known conditionals.

| Effect of MMP against its twin | Law and value | Kind |
|---|---|---|
| Lowered recovery | L_PROXY: pair 0.767 → 0.500 (`PROTOTYPE_REPORT.md`:32) | Exact Bayes, finite law |
| Lowered recovery | L_MIXED: 0.717 → 0.650 = AUC\*(S;h) (`PROTOTYPE_REPORT.md`:34) | Exact Bayes, finite law |
| Lowered recovery | Designer 1's laws: R, 0.678 → 0.607; F with headroom, 0.652 → 0.500 (designer 1 `PROPOSAL.md`:156, 211) | Exact Bayes, finite laws |
| Lowered recovery | L_CONT_PROXY: Bayes 0.788 → 0.722 by exact quadrature (`PROTOTYPE_VERIFICATION.md`:92–93); fitted pair 0.779 → 0.723 (`PROTOTYPE_REPORT.md`:59–60); verifier's LR / HGB / MLP gaps 0.056 / 0.063 / 0.054 | Bayes, and fitted |
| Lowered recovery | P7 draw 0: twin − MMP +0.032 / +0.033 (LR / HGB, 8k rows); +0.033 for a 1M-row reader within 0.0004 of MMP's exact Bayes value | Fitted, near-Bayes |
| No effect | L_NOPROXY: 0.731 = 0.731; L_XOR: pair 0.850 unchanged, singles 0.500 (`PROTOTYPE_REPORT.md`:35–36) | Exact Bayes |
| No effect | L_CONT_NOPROXY: Bayes 0.7216 for both; fitted 0.7211 vs 0.7207 | Bayes, and fitted |
| Raised recovery | M10: pair Bayes 0.5 → 0.8 (`MATH_ASSESSMENT.md`:67) | Exact Bayes, constructed cancellation |
| Raised recovery | P7 draws 1 and 2: twin − MMP −0.018 / −0.020 and −0.019 / −0.020 (LR / HGB, 8k rows); −0.018 and −0.019 at 1M rows | Fitted, near-Bayes |
| Utility failures | L_NEG infeasible; L_PROXY_FLIP fails accuracy; L_CONT_PROXY fails all three strict occupation caps; men's within-SEX increase reaches 2.2–2.6× the guard (`PROTOTYPE_VERIFICATION.md`:96–97, 158) | Exact or quadrature |

The sign of the effect followed the alignment of field-mediated and direct SEX effects (P7). None of these laws is calibrated to ACS.

## 5. RCCS reconciliation (interpretation only)

| Question | What the record shows | Kind |
|---|---|---|
| Which attacker class does its objective directly control? | Q(A) = R²_cv(S ~ φ1, φ2(f2^A), q) − R²_cv(S ~ φ1) + γ‖A‖²: a closed-form ridge criterion on linear and quadratic features of the wires (`MATH_ASSESSMENT.md`:171). It controls only the linear and quadratic partial criterion | Definition |
| Its population-optimal effect in the specified laws | Recovery rose: designer 2's own finite law, 0.615 → 0.644; the verifier's Gaussian laws, +0.021 to +0.053 (`MATH_ASSESSMENT.md`:189, 307–312, 320). General statement: by (a), it cannot fall below the base release's whenever the base pair can be recovered from the RCCS pair. That is a sufficient condition, not a theorem for every law | Exact or known-law Bayes |
| Fitted linear, quadratic, HGB and MLP readers | Every fitted reader fell by 0.019–0.045, within the binding clauses (B = 0.005 inside the cap; B = 0.009 inside cap plus headroom). HGB, at 8k and 100k training rows, and MLP, at 8k, fell at least as much as the criterion-class readers. This is one smooth law family (two settings, two budgets) with an oracle propensity and a criterion fitted on 30,000 rows: RCCS's best case (`MATH_ASSESSMENT.md`:295–323) | Fitted, synthetic |
| Benefit over the matched local-targeting comparator (MIN) | Linear readers tied (\|Δ\| ≤ 0.0012). Pair readers: HGB favoured RCCS by 0.012–0.022 and MLP by 0.008–0.011. The predicted occupation-only signature was absent, and falsifier 1 (OUT_LR) would reject the mechanism in this law (`MATH_ASSESSMENT.md`:325–342) | Fitted, synthetic |
| Smooth versus discrete fields, and real data | Designer 1's exact one-coordinate construction zeroes the linear and quadratic criteria. With a Gaussian driver, cancellation is complete. With lumpy categorical drivers, the optimal reader keeps 21% to more than 100% of the cancelled above-chance signal (designer 1 `CROSS_CRITIQUE.md`:16–29). This is an optimal-reader result in a constructed law. The literal map has had no fitted-reader test on discrete fields (`MATH_ASSESSMENT.md`:200–203) and no real-data test | Exact optimal reader; untested otherwise |
| Adverse evidence (all retained) | S1: inconsistent supporting accounting; under consistent accounting the MIN–RCCS gap shrinks from 0.009–0.110 to 0.002–0.031, and the LOCAL replay fails (0.591–0.609 against 0.657) (`MATH_ASSESSMENT.md`:246–260, 348). S2: model-dependent readings of published AUCs (`MATH_ASSESSMENT.md`:207–238). S3: the original finite "feasible case" fails the binding clauses; COND equals TASK at both budgets, and the first reduction costs 3.75 pp of accuracy (`MATH_ASSESSMENT.md`:272–275). S4: the falsifiers are not decisive in either direction (`MATH_ASSESSMENT.md`:286–291). Population recovery rose in every tested law. Designer 1's categorical-driver result. No real-data validation | — |

**Conclusion.**
- **The rejection as presented is supported.** S5 fails as presented: the proposal's own feasible case violates the binding clauses, and its size estimate rests on an inconsistent model. This depends neither on the injectivity claim nor on S6.
- **The summary characterisation is not supported.** "Only hides information from linear and quadratic attackers" and "only works against the linear and quadratic attackers it optimises against" are contradicted in the one smooth law tested, where fitted nonlinear readers fell at least as much.
  - The S6 "fail" rests on the coordinator's reading that the rationale depends on the reader class. The verifier rated that dependence MINOR (`MATH_ASSESSMENT.md`:352).
- **Corrected scope.**
  - The objective controls linear and quadratic critics.
  - In one smooth synthetic law, all fitted readers fell while optimal recovery rose.
  - With lumpy categorical drivers, an optimal reader recovers much of the cancelled signal.
- **A repaired RCCS is unresolved.** The verifier's repairs (i)–(iv) (`MATH_ASSESSMENT.md`:357–361) have not been done. Because its optimal-reader recovery rose in every tested law, any benefit a repaired version shows would be against the declared readers only.
- **No method win follows.**

## 6. Status derivation

**Rule 4** (`roles/coordinator/SELECTION_CRITERIA.md`:28): "If the prototype contradicts the mechanism's rationale, that result is reported and the candidate is not described as ready for a real-data pilot."

**What the rule-4 outcome rests on.** None of the following uses the injectivity of any ACS release:
1. **Pre-registered check (a) failed.**
   - In L_CONT_NOPROXY, twin − MMP was 0.0004 against a registered ≥ 0.005. This check was registered in advance as a contradiction of the rationale (`DECISION_LOG.md`:137; `PROTOTYPE_REPORT.md`:87).
   - It is a fitted-reader result in a constructed law, and it has low power (`PROTOTYPE_VERIFICATION.md`:144–152).
2. **The verifier's registered probe P7 failed its own prediction.**
   - Without pure-proxy fields, the sign of MMP's effect was indeterminate (+0.033, −0.018, −0.019), at near-Bayes accuracy (`PROTOTYPE_VERIFICATION.md`:117–138).
   - It was registered before it ran, but it is post-hoc relative to the prototype pre-registration, and the law is constructed.
3. **Utility findings in constructed laws.** The L_CONT_PROXY merge fails the strict caps, and the within-SEX burden reached 2.2–2.6× the guard.

**What changes when the injectivity claim is removed.**
- The argument that the confirmed merging leg cannot matter on real data is withdrawn (`PROTOTYPE_REPORT.md`:95; `METHOD_SPECIFICATION.md`:143).
- Whether the Florida data have the merging structure in which MMP removes information is UNKNOWN (`HANDOFF.json`:67). So is the sign of its effect there.
- The rule-4 outcome, "not described as ready for a real-data pilot", **still follows**, from check (a) as pre-registered, and P7 reinforces it. But the stated reason must change. It is not "real releases are injective". It is "the registered learnability prediction failed, the probe showed MMP's effect can go either way, and the real-data structure is unknown".

**A note on the label.** Rule 3 defines `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED` for the case where neither proposal is eligible. The campaign applied it, with the qualifier "for a real-data pilot", to a rule-4 outcome. That is consistent in substance. The literal outcome is: MMP selected but not described as pilot-ready (rule 4); RCCS rejected as presented (S5).

**Four things that must be kept apart.**
1. **No competitive method established: correct.** Nothing was compared with NEURAL_RAW on data, and there were no real-data fits.
2. **Candidates declined under this campaign's criteria: correct.** MMP under rule 4, on pre-registered check (a); RCCS under S5 as presented.
3. **Specific hypotheses contradicted under specified synthetic laws: correct.**
   - The ≥ 0.005 learnability prediction in L_CONT_NOPROXY.
   - P7's "≥ 0.01 in each draw".
   - The "women bear more" subgroup expectation, which was mis-specified.
   - RCCS's "feasible case".
4. **A universal impossibility claim: none is valid.**
   - No theorem in the record shows that no deterministic release under the current contract can reduce optimal recovery at the caps, or that no mechanism can beat NEURAL_RAW.
   - Check B shows the general claim false in a constructed law, and `PROBLEM_SPECIFICATION.md`:142, 151 explicitly deny such a theorem for this data.
   - The status must not be read as an impossibility result.

## 7. Recommended replacement sentences for the advisor-facing summaries

1. "Every evaluated ACS release is a deterministic function of the same pooled encoding of the six fields. None has been shown to be one-to-one on the population, and none has been shown not to be."
2. "The optimal attacker can never recover more SEX information from a release than from the inputs it is computed from. It recovers exactly as much if and only if the release preserves P(SEX | inputs)."
3. "Our recovery figures are what a fixed seven-reader slate achieves with about 8,000 labelled releases. How much of any difference between defenses reflects removed information, and how much reflects learnability, is unknown."
4. "A deterministic probability release within the current contract can lower optimal recovery, but only by giving the same output to inputs with different SEX posteriors; an exact constructed example does this at no utility cost. Whether any such release meets the 24 caps on Florida is open. The coarsening recipes tried so far failed for the codes tried."
5. "Under the population law, MMP's Brier price has an exact formula, E[(π − π̄)²‖q¹ − q⁰‖²]. For fitted models on real data, its cost against U0 and Ucal is not known in advance and must be measured."
6. "MMP evaluates a fitted model of P(task | fields, SEX) with its SEX input set to each value, and averages the two outputs with fixed weights. These are model evaluations, not counterfactual predictions for the person."
7. "MMP lowers optimal recovery only where its output merges inputs with different SEX posteriors: pure-proxy fields are one case, and directions unused by the within-SEX task predictions are another. It can also raise recovery relative to the SEX-blind prediction."
8. "RCCS's objective controls only linear and quadratic critics. In one smooth synthetic law, every fitted reader, trees and neural networks included, fell 0.019–0.045 while optimal recovery rose. With lumpy categorical drivers, an optimal reader recovers much of the cancelled signal. RCCS stays rejected as presented, because its feasibility case failed the binding clauses."
9. "Final status: neither candidate is ready for a real-data pilot under this campaign's criteria. This is not an impossibility result."

**Caution on opposite-direction wording.** A working-tree draft of `ADVISOR_BRIEF.md` that I saw during this review says that every model reads an encoding that pools rare levels, "so none is one-to-one in the raw fields". That holds only if at least two pooled levels of some field occur in the population, which is plausible for POBP but not verified here. Suggested wording: "…so none is one-to-one in the raw fields if, as is likely for POBP, two or more pooled levels occur in the population (not verified)."
