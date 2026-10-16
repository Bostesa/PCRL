# Problem specification: the ACS two-recipient output service

This fixes the problem before any solution is proposed. Every definition below was taken from the source study's code and locks: `acs_replication/data.py`, `results/pcrl_acs_protected_tree_replication_v1/SCIENCE_LOCK.json` and `POPULATION_AND_SCHEMA.md`, and the frozen-hybrid locks. A proposal that changes any of it must say so explicitly, as a contract change (section 7).

## 1. Variables

- **Population.** Florida ACS 2019 1-Year PUMS person records: housing-unit residents aged 25–64, civilian employed, WKHP 1–99, with a mapped civilian occupation and valid income, SEX, COW, SCHL, MAR and POBP. There are 74,091 eligible people in 49,375 households. The DEVELOPMENT roles are listed in section 5; 20% of households form SEALED_ASSESSMENT, which stays closed.
- **Inputs X.** Six permitted raw fields: AGEP, WKHP, COW, SCHL, MAR, POBP.
  - The model encoding has width 145: AGEP and WKHP are z-scored with MODEL_FIT moments; COW, SCHL, MAR and POBP are one-hot, with levels seen fewer than 10 times in MODEL_FIT pooled into an OTHER column.
  - Not inputs: SEX, every race variable, OCCP/SOCP, industry, income amounts, ESR, RELP, identifiers, weights and ADJINC.
- **Declared protected attribute S.** SEX, binary (`sex0` = male, SEX 1; `sex1` = female, SEX 2).
  - Protection is claimed for S only. Excluding other variables from the inputs does not make a protection claim about them, and nothing here claims to protect "every other attribute".
- **Tasks.**
  - Y1: income, 1[PINCP × ADJINC > 50,000 × 10^6], with K1 = 2.
  - Y2: occupation, one of five official broad civilian groups, with K2 = 5.

## 2. Releases, downstream predictions and the coalition view

- **The mechanism.** A release mechanism is a pair of maps f_i : X → Δ^{K_i}, for i = 1, 2.
  - In the current primary contract both maps are deterministic functions of the person's own six fields.
  - Recipient i receives R_i = (f_i(X), d_i(X)), where d_i = argmax_k f_i(X)_k: its own probability vector and its decision.
- **Downstream predictions.**
  - Recipient i's probabilistic prediction of Y_i is f_i(X), which the NLL and Brier clauses score.
  - Its decision is d_i(X), which the accuracy clause scores.
- **Coalition view.** V = (R_1, R_2) for the same person, perfectly linked. Linkage is an assumption of the threat model, not a measured capability.
- **Not released:** leaf ids, representations, teacher outputs, labels, raw features, identifiers, weights and flags. This is the source `release_contract`.

## 3. Adversary and recovery metric

The adversary reviewer's `ADVERSARY_AND_MOTIVATION.md` gives the full specification and its justification. The operational definition is:

- **Readers.** A reader is a scoring function g of a view: R_1 alone, R_2 alone, or V.
  - It is fitted on (view, S) pairs from ATTACK_FIT, computed under the same mechanism. This is a defense-aware refit to the actual defended releases.
  - It is selected and oriented on ATTACK_VALIDATION, and scored by AUC for S on DEV_REPORT.
  - The slate has 7 readers per view: OUT_LR, OUT_MLP and OUT_HGB on probabilities plus decision; LOG_LR and LOG_MLP on log-probabilities plus the one-hot decision; DEC_LR on the one-hot decision; CONF_HGB on the maximum probability.
  - The coalition bank also contains readers that ignore either recipient.
- **Recovery** is the DEV_REPORT AUC of the reader selected on ATTACK_VALIDATION.
  - It is a finite-sample lower estimate of what a stronger attacker could achieve. It is not a population floor, a certificate or a formal privacy guarantee.
- **Population target.** The coalition's Bayes recovery is AUC*(S; V), the AUC of the posterior P(S = female | V) under the population law.
  - It upper-bounds every reader's population AUC on V.
  - For a deterministic mechanism, AUC*(S; V) ≤ AUC*(S; X), by data processing for the Bayes-optimal ranking.

## 4. Utility requirements (unchanged; recovered from code)

These definitions come from `acs_replication/data.py::task_vectors/summary/clauses`. For person j, task i, released vector p = f_i(x_j) and label y_j:

- accuracy_j = 1[argmax p = y_j];
- nll_j = −log max(p_{y_j}, 1e−12);
- brier_j = Σ_k (p_k − 1[y_j = k])². This is the **summed** Brier over all K_i columns, so for the binary income task it equals twice the usual one-column Brier.

Each metric is averaged without weights over the evaluated role. For each task i, each reference r ∈ {U0, Ucal} and each seed s, three clauses must hold, with inclusive caps:

| clause | definition | cap |
|---|---|---|
| accuracy loss | acc(r) − acc(f_i) | ≤ 0.010 |
| NLL excess | nll(f_i) − nll(r) | ≤ 0.010 nats |
| summed Brier excess | brier(f_i) − brier(r) | ≤ 0.005 |

That gives 2 tasks × 2 references × 3 metrics × 2 seeds = **24 clauses**.

- **References.**
  - **U0** is the validated per-task neural network (d-64-64-16-K, Recipe A on MODEL_FIT).
  - **Ucal** is U0 with one global temperature per task, fitted on CALIBRATION by minimising mean NLL (bounded scalar search on [0.05, 20]).
- **Where the clauses are applied.**
  - MODEL_SELECT feasibility: all 24 clauses at the point. This is the selection rule.
  - DEV_REPORT point gate: all 24 clauses at the point. This gate has no error control.
  - Confidence: one-sided α/3 clause bounds and simultaneous family intervals. This is reported as a separate field.
- **Subgroup guard.** Occupation summed-Brier excess vs U0, computed within each SEX group, must be ≤ 0.005.

These caps, references and definitions are binding. No proposal may lower a threshold, drop a confidence requirement or redefine a comparator.

## 5. Training and selection access

| Role | Size (people) | Who may use it, and for what |
|---|---|---|
| MODEL_FIT | 20,000 | Every method: features, both task labels and SEX for fitting. Protection objectives may use SEX here only |
| CALIBRATION | 2,982 | Ucal temperatures only |
| MODEL_SELECT | 8,001 | Selection under the frozen rule: 24-clause point feasibility first, then the mean over seeds of the coalition AUC of a fixed OUT_LR reader trained on MODEL_FIT |
| ATTACK_FIT / ATTACK_VALIDATION | 8,001 / 3,000 | Readers only. Never used to fit or select a defense |
| DEV_REPORT | 5,943 (3,955 households) | Already opened, and re-read several times since. Any future use is exploratory |
| SEALED_ASSESSMENT | 14,693 | Closed. Disclosure D2 applies: the source extractor parsed sealed records in memory before roles were assigned, so it is not "never accessed" |

**Deployment** must take only the six permitted raw fields. It must not require SEX or any other prohibited input.

## 6. Comparators and claims (kept separate)

**Comparators**, all already fitted for two paired seeds:

| Comparator | What it is |
|---|---|
| TASK_REFIT | The task-only twin of the leaf-score recipe: the same loop with privacy terms disabled, 20 stages |
| LOCAL_TREE | Adversarial leaf-score correction of the frozen CatBoost teacher against each recipient's own critics (ρ = 1, E20) |
| JOINT_TREE | The same correction plus the pair term. A diagnostic: it failed income utility |
| NEURAL_RAW ρ1 E2 | The eligible neural adversarial control. It starts from U0 and uses income, occupation and pair critics. It is the only feasible point of a 12-point grid |
| T_TREE | The CatBoost teacher (depth 4, at most 800 trees, early stopped) |
| U0 / Ucal | The utility references |
| FAGTB and official FARE | Published baselines. Both were ineligible here, because no configuration was utility-feasible |

**Claims and status fields.** Each future evaluation reports these separately. None may be inferred from another.

1. **Point feasibility.** All 24 utility clauses hold at the point, on MODEL_SELECT and on DEV_REPORT. This is deterministic, with no error control.
2. **Utility confidence.** Clause bounds and simultaneous intervals: PASS / UNRESOLVED / ESTABLISHED_VIOLATION.
3. **Recovery reduction (recipe benefit).** Coalition benefit vs the candidate's own task-only twin, with a one-sided α/3 lower bound > 0.020 on both seeds. Each recipient's worsening upper bound must also be ≤ 0.010.
4. **Competitive method.** Coalition benefit vs every eligible comparator, at minimum NEURAL_RAW, with lower bound > 0 on both seeds. The nested material superiority margin is 0.020. The candidate must meet the same utility requirements as the comparators.
5. **Joint component** (only if a proposal claims joint design matters). The candidate is compared with its strongest local twin and with both sequential orders, under equal access and budget. This claim is separate from claim 4: a method can be useful without a joint component, and a joint component can be real without the method being competitive.
6. **Subgroup suitability.** Within-SEX occupation Brier guards; PASS, HARM and UNRESOLVED labels come from simultaneous intervals.
7. **Algorithmic novelty.** Judged against prior work only. Numerical success is not evidence of novelty, and a constructed example is not either.

**Planning arithmetic.** This is descriptive and does not lower any threshold.
- On the existing DEV_REPORT, the paired coalition-difference standard errors were 0.0039–0.0051 AUC. The source `PRIMARY_ENDPOINTS.csv` has them: 0.0045 and 0.0039 vs TASK_REFIT, and 0.0051 and 0.0050 vs NEURAL_RAW.
- The one-sided α/3 critical value is 2.395.
- So at DEV_REPORT size, establishing benefit > 0.020 over NEURAL_RAW needs a point benefit of about 0.032. That means coalition AUC near 0.608 on seed 0 and 0.606 on seed 1, against NEURAL_RAW's 0.640 / 0.638.
- Plain superiority needs a point benefit of about 0.012, i.e. coalition AUC near 0.628 / 0.626.
- These are the values a candidate would have to reach. They are not new thresholds.

## 7. Contract changes

A proposal may change the contract only by declaring the change and evaluating every comparator under the same changed contract. Examples:
- a feature or representation service instead of probabilities;
- extra metadata;
- released or shared random seeds;
- state shared between recipients;
- per-query server interaction;
- abstention symbols;
- coarsened outputs.

Stochastic releases count as a contract change. They are scored by expected NLL and Brier of the randomised output, under the same caps.

## 8. Facts any proposal must be consistent with

These are taken from published aggregates, as DEV_REPORT AUC on seed 0 / seed 1; `EVIDENCE_PACK.md` lists the sources.

1. **Local protection lowers each recipient's own recovery much more than the coalition's.**
   - LOCAL_TREE recipients: income 0.558 / 0.549, occupation 0.582 / 0.584. Linked pair: 0.657 / 0.649.
   - TASK_REFIT: 0.572 / 0.567, 0.650 / 0.645, pair 0.664 / 0.666.
2. **Coalition-aware protection lowers the pair further, but in this fitted bank it hit the income utility clauses.**
   - JOINT_TREE pair 0.603 / 0.606, failing 2–4 income clauses (summed Brier, and accuracy on seed 1).
   - NEURAL_RAW's only feasible grid point was its weakest, ρ1 E2. Every stronger point failed, mostly on income summed Brier; ρ1 E5 failed only seed-1 income Brier, at 1.06× the cap.
3. **The eligible neural control** has pair 0.640 / 0.638, with recipients at income 0.546 / 0.551 and occupation 0.622 / 0.623. Its pair exceeds its best single recipient by much less than LOCAL_TREE's does.
4. **The fixed hybrid H_LJ** (LOCAL income + JOINT occupation) has pair 0.641 / 0.643. Point utility passes.
   - Utility-ineligible combinations that use JOINT income reach 0.625 (H_JL) and 0.603 / 0.606 (H_JJ).
   - This is a trade-off observed in this fitted bank, not a universal impossibility.
5. **The pair-versus-single differences** are fitted-reader differences. They motivate studying combined outputs. They are not an information-theoretic synergy theorem.

## 9. Mathematical facts every proposal must respect

1. **Joint versus sequential.** With deterministic maps and no shared randomness, any pair (f_1, f_2) chosen jointly can be produced by a sequential designer with full look-ahead: choose f_1 knowing which f_2 will follow. A claimed joint benefit can therefore only concern constrained or greedy algorithms, computation, or optimisation quality. Shared randomness enlarges the achievable set of joint laws for randomised releases, and that is a contract question (section 7).
2. **Decision-preserving releases.** If d_1 and d_2 are fixed to the reference decisions, then AUC*(S; V) ≥ AUC*(S; d_1(X), d_2(X)), because the Bayes attacker may ignore the probabilities. A method allowed to change decisions is not bound by the old decision-only reference.
3. **True labels are not a disclosure floor.** The true labels (Y_1, Y_2) are not released, so AUC*(S; Y_1, Y_2) is neither a lower nor an upper bound on AUC*(S; V).
4. **Fitted AUC is not a population floor.** A fitted reader's AUC is a lower estimate of AUC*(S; V), not an exact population privacy floor.
5. **Required task disclosure.** If every utility-feasible f_i must approximately reproduce a task posterior whose level sets separate S, some recovery is unavoidable. How much is an empirical question about the population law. No theorem here bounds it for this data.
