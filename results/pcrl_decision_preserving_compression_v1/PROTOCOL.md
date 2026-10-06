# Protocol: decision-preserving joint compression

Branch `research/pcrl-decision-preserving-compression-v1`, created from evidence commit `925e0fddfcb666116c6179575339728a324ed78e` (`research/pcrl-online-strength-frontier-v1`). Package `dpc/`.

## Registered exposure statement

"The design is motivated by previously opened Adult development results. Every assessment row has been used historically. This is an exploratory, locked benchmark comparison. Its nominal intervals condition on the fitted artifacts and do not correct for the adaptive research history. It is not fresh confirmation or a prospective population guarantee."

## 1. Question

Can each recipient keep its predictions and most of its useful confidence while the released scores reveal less SEX, especially when the two recipients pool their releases?

The mechanism is a public, class-preserving finite code learned from the frozen teachers' task probabilities. It changes what is released. It does not continue the closed penalty-strength, critic-refresh or feedback-controller lines. No encoder is retrained, and no randomisation, critic schedule or gradient normalisation is added.

## 2. Data, roles, teachers (`ROLE_MANIFEST.json`, `ADMISSION.json`)

The input is `adult_jcv.npz` (sha256 `e0d9e54a…2f12`). The pinned OSF roles are used unchanged (`osf.data` via `dpc/data.py`):

| Role | Rows | Use |
|---|---:|---|
| OSF_DEFENSE_FIT | 15,434 | Fine partitions, prototypes, compression policies and their privacy objectives |
| HEAD_VALIDATION | 1,500 | Historical head selection only; no new head is fitted or selected |
| AUDIT_FIT | 6,065 | Attacker fitting |
| INNER_SELECTION | 2,235 | Attacker, configuration and nominee selection |
| OSF_DEVELOPMENT_ASSESSMENT | 13,936 (13,929 exact-record groups) | Four admitted historical pools; labels masked until the pushed EVALUATION_LOCK |

**Inputs.** 83 permitted columns in the pinned order. Proxies such as relationship are kept. The tasks are binary income and six-class occupation_group.

**Teachers.** U and RAW-J β 0.3 at epoch 40, seeds 0–2 (osf units `rel__s{k}__U` and `rel__s{k}__RAW-J_b0.3`). Encoders and deployed StandardScaler+LogisticRegression heads stay frozen.
- Per person and purpose: teacher probability vector p_i, decision d_i = argmax p_i (numpy first-index tie rule) and centred scores.
- These are rebuilt by an independent forward pass and checked against the saved releases.
- Ground-truth task labels never define a code.

**References.** Official LEACE (E), FARE (F) and its no-fairness twin (F0), admitted from the OSF manifest with fit-role provenance. Their score-only releases are scored under the same useful-output metrics, and their full-feature views remain descriptive.

## 3. Release contract

For purpose i a public map C_i = g_i(p_i) gives:
- a token (categorical ID, audited as its full identity: one-hot and exact tuple);
- the token's public decoded probability vector;
- the decision.

The coalition receives both complete interfaces, aligned. No latent feature, continuous teacher score, logit, fine-cell ID or assignment distance accompanies the protected release.

g_i reads only its own teacher's p_i and d_i. It never reads SEX, a true label, the other recipient's score or the role.

**Class preservation.** States merge only within the same teacher-predicted class, and every decoded prototype has that class as its strict argmax. So the released decision equals the teacher's decision for every person, pointwise and independent of the data distribution (fixtures cover ties, underflow and unseen classes). Accuracy, confusion matrices and recalls are therefore unchanged relative to the teacher. Confidence quality is not guaranteed: log loss and Brier are measured.

## 4. Fine partitions, prototypes, objectives (`dpc/partition.py`, `dpc/compress.py`)

**Fine partitions.**
- Per teacher, seed, recipient and predicted class, a task-only KL/Bregman k-means runs on OSF_DEFENSE_FIT probability vectors with at most 16 cells.
- Initialisation is deterministic; there are 20 rounds and a recorded convergence rule.
- A predicted class absent from the fitting rows gets a reserved fallback cell and token with prototype smooth(uniform, class).

**Smoothing.**
- Prototypes use (mean + ε·1 + ε·e_d)/(1 + (K+1)ε) with ε = 1e-12, identically for every method.
- KL uses 0·log 0 = 0, float64 accumulators, and log-loss clipping at 1e-12 (the pinned convention).

**Deployment.** A row is assigned to the nearest fitting fine cell of its predicted class under the fixed KL rule. Assessment rows never update anything.

**Objectives.**
- D_i is the fitting-row mean KL(p_i‖decoded p_i), a teacher-distortion surrogate.
- I_1, I_2 and I_12 are plug-in mutual informations (natural log) from the exact OSF_DEFENSE_FIT contingency tables, with no smoothing of the joint alphabet.
- F_task = D_1 + D_2.
- F_local = D_1 + D_2 + λ(I_1 + I_2)/2.
- F_joint = D_1 + D_2 + λ((I_1 + I_2)/2 + I_12).
- These are training criteria only. Independent attack AUC governs every claim.

**Structural statements (verified, not claimed as new).** The decision is a function of the token. The codes are post-processing, so I(S;C_i | M) ≤ I(S;p_i | M) and likewise for the pair (data processing). I(S;C_i) = I(S;d_i) + I(S;C_i | d_i). None of these is a secrecy or training-data privacy bound.

## 5. Families and bank (`FIT_MANIFEST.json`)

Per teacher (U, RAW-J β 0.3), seed (0, 1, 2), cap m coarse cells per predicted class (m ∈ {2, 4, 8}) and λ ∈ {0.1, 1, 10}:

| Family | Rule |
|---|---|
| FINE-TASK | Greedy same-class agglomeration of fine cells minimising the F_task increment, then task-only refinement |
| DIRECT-TASK | Row-level KL k-means with m prototypes per predicted class (no sensitive labels; a stronger compression control, not contained in the joint learner) |
| LOCAL | Greedy and refinement on D_i + λ I_i/2, each recipient independently |
| SEQ-12 / SEQ-21 | Optimise the first recipient alone (D + 1.5 λ I: F_joint with the other recipient constant), freeze it, then optimise the second under F_joint. Never revise the first. |
| JOINT | Greedy F_joint over merges on either recipient, then refinement. Refinement also starts from the FINE-TASK, LOCAL, SEQ-12 and SEQ-21 solutions; the lowest fitting F_joint is kept (dominance over the witnesses verified). No selection data is used. |
| CLASS (m = 1) | One token per predicted class; a diagnostic decision-only release |

**Search rules.**
- **Greedy:** evaluate all eligible same-class merges, choose the smallest increment with a fixed lexicographic tie, and stop at the cap.
- **Refinement:** single fine-cell moves within a class without emptying a cell. Exact deltas; strict decrease beyond tol 1e-12; at most 5 sweeps per recipient (the same allowance for local and sequential methods). Work and convergence are recorded.

**Nominal bank.** 2 teachers × 3 seeds × (3 rates × 2 task-only families + 3 rates × 3 λ × 4 privacy families) = **252 mapping-pair units**, plus 6 CLASS units, the continuous source audits and the reference audits.

**Aliases.** Exact aliases keep their manifest slots and get alias receipts.

**Reduced bank.** Allowed only before real fits, on synthetic timing: m ∈ {4, 8}, λ ∈ {0.1, 1}, every family, all seeds and both teachers. The choice and nominal counts are recorded in TRAINING_LOCK.

## 6. Attacks (`dpc/audit.py`)

**Fitting and selection.** Attackers are fresh: fitted on AUDIT_FIT and selected on INNER_SELECTION.

**Slate.** Every primary contract uses the same source FINAL slate: LR ×5, MLP ×4, HGB ×4, DA_LR and DA_MLP, at pinned grids and seeds. Finite codes add cell-conditional readers (smoothing {0.5, 1, 5}) on the exact token identity.

**Pair attacks.**
- The pair reader uses the token tuple.
- Unseen local tokens fall back to the AUDIT_FIT SEX prior.
- Unseen pairs use a fallback (local readers or prior) chosen on INNER_SELECTION before scoring.
- Coverage is published.
- The coalition bank includes both ignore-recipient banks.

**Selection rules.** AUC-selected and CE-selected attackers stay separate. Orientation is fixed; nothing is flipped or clamped on scored rows.

**Composed source readers.** Because the maps are public, the continuous source-score attacker's inner candidate bank includes the code readers of every fitted policy of the same teacher and seed. These are fixed before C_global, T* or the nominees are selected. The final source audits include the full-slate code readers of the locked selected policies.

**Controls.** Real-data shuffled-label nulls and planted code leaks run before EVALUATION_LOCK. They include a within-predicted-class confidence token carrying a planted SEX bit and a decoder-collision token.

## 7. Useful-score contract and inner eligibility (`dpc/utility.py`)

**Anchor.** U's deployed continuous output, unchanged.

**Metrics.** True-label log loss, multiclass Brier (sum of squares), accuracy, gain over the fitting-prior constant, balanced accuracy, class recalls and ECE.

**Inner task eligibility.** All of the following on INNER_SELECTION, on EACH seed, for both tasks:
1. accuracy ≥ U − 0.01;
2. log loss ≤ U + 0.01 nats;
3. Brier ≤ U + 0.005;
4. gain over the constant ≥ 80% of U's gain;
5. gain ≥ 0.03.

Every policy must also pass pointwise class preservation against its own teacher. RAW-J-derived codes preserve RAW-J's decisions and must still meet these U-relative gates. These allowances are prospective for this new study and are never relaxed.

**Global configurations.** One configuration (teacher, family, m, λ) across all seeds.

**Tie rule.** Lower mean inner pair AUC, then lower mean task log loss, then fewer token states, then configuration ID.

## 8. Controls and nominees (`dpc/select.py`)

**C_match(teacher, m).** The lowest-mean-inner-pair-AUC task-eligible nonjoint control at that teacher and rate (FINE-TASK, DIRECT-TASK, LOCAL, SEQ-12, SEQ-21, across λ).

**C_global.** The strongest task-eligible nonjoint control across both teachers and all rates, plus eligible CLASS outputs, continuous source-score outputs (with composed readers) and the official reference score releases. JOINT is excluded.

**J\*.** Among globally configured, task-eligible JOINT policies whose local inner AUCs are ≤ C_match + 0.005 and ≤ C_global + 0.005 on every seed and recipient, the one chosen by the tie rule. Its C_match is recorded.

**T\*.** The strongest task-eligible privacy-untrained release: FINE-TASK, DIRECT-TASK, CLASS and continuous sources, across both teachers and rates.

**P\*.** Among task-eligible LOCAL, SEQ-12, SEQ-21 and JOINT policies whose local inner AUCs are ≤ T* + 0.005 on every seed, the one chosen by the tie rule.

**Missing nominees.** No eligible nominee gives NO_FEASIBLE_NOMINEE. A deterministic minimum-shortfall fallback is scored as DESCRIPTIVE_ONLY and cannot pass.

**Missing comparators.** A missing comparator invalidates the dependent claim.

## 9. Primary claims (`dpc/family.py`, `PRIMARY_FAMILY.json`)

| Claim | Nominee vs comparator |
|---|---|
| A | J* vs C_match |
| B | J* vs C_global |
| C | P* vs T* |

**Clauses (11 per claim):**
1. AUC_pair(C) − AUC_pair(N): lower bound > 0.02.
2–3. AUC_vi(N) − AUC_vi(C): upper bound < 0.01.
4–5. Acc(N) − Acc(U): lower bound > −0.01.
6–7. LogLoss(N) − LogLoss(U): upper bound < 0.01.
8–9. Brier(N) − Brier(U): upper bound < 0.005.
10–11. Acc(N) − 0.8 Acc(U) − 0.2 Acc(const): lower bound > 0.

**Inference.**
- 33 slots are always kept.
- z = NormalDist().inv_cdf(1 − 0.05/66) = 3.1717657833516224.
- B = 1,999 paired exact-record-group bootstrap replicates, seed 20261007, the same draws for all arms and seeds.
- Seeds are averaged by the source convention.
- A nonfinite replicate in a primary statistic makes that slot INVALID; it is never dropped.

**Labels.**
- JOINT_DEVELOPMENT_CRITERION_MET: A and B pass with a valid J*.
- PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET: C passes with a valid P*; the winning family is stated.
- EXPERIMENTAL_NO_ADVANTAGE: neither, with complete validity.
- INCOMPLETE_OR_INVALID: missing validity or coverage.

## 10. Locks and execution

| Lock | Contents | Governs |
|---|---|---|
| ENGINEERING_LOCK | Protocol, exposure statement, roles, pins, method code checked on synthetic data, budget | Admission |
| TRAINING_LOCK | Bank from synthetic timing, nominal counts, predictions (committed at `325cdbb`), pre-fit repairs | Fine partitions and every policy fit, including references |
| SELECTION_AND_AUDIT_LOCK | Attackers, metrics, gates, selection, family, inference, assessment code | Inner audits, controls, selection |
| EVALUATION_LOCK | Exact policy, prototype, teacher, attacker, role and code hashes | The single assessment opening |

**Push rule.** Every lock is pushed and verified on origin before its stage (`dpc.lock.verify_lock`). Agents never start a label-based stage; the lead runs every real-data stage.

**Repairs.** Engineering defects affecting a locked stage get quarantine, a dated amendment pushed before rerunning, and a rerun of the affected units only. Margins, roles, objectives and selection rules are never changed to repair a scientific failure.

**Assessment scope.** The locked nominees and comparators, the continuous source outputs (complete, scores, probabilities and decisions), the CLASS controls, the task-only rate curve (m = 1, 2, 4, 8), the eligible privacy policies and the nearest ineligible alternatives (fixed shortfall rule). All are scored once, in one locked opening.

**Ceilings.**
- 10 h elapsed (start 23:37:49Z; watchdog at 09:37Z stops only dpc processes).
- 20 CPU-h, two heavy workers, 8 GiB, ≥ 5 GiB free, $0.
- The final 2 h and 4 CPU-h are reserved for verification, reports, backup and closeout.

## 11. Prior art and claims

Agglomerative and multivariate information bottleneck, privacy funnels, output transformation, data processing and multi-recipient releases are established (`PRIOR_ART_AND_BASELINE_GAPS.md`). The SEQ arms are matched deterministic adaptations, not Taylor et al.'s algorithm. PURIFIER targets membership inference. Any development win is an exploratory benchmark result under this contract, not a population guarantee or a novelty claim.
