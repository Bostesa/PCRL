# Decision: SCM engineering benchmark

**What this is.** An engineering benchmark of an existing-method recipe, Shared-Cell Merging (SCM). It makes no novelty claim. Everything here is exploratory development evidence on repeatedly used Florida ACS 2019 people. The sealed assessment stayed closed. DEV_REPORT truth was re-read twice, each time after the committed reporting lock and logged in both access logs:
- once for this report (`DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP`, 03:20:19Z);
- once by the independent verifier (`DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER`, 03:29:39Z; `roles/verifier/VERIFICATION.md`, check 8).

The verifier's temporary crash copy of the re-read arrays, kept outside the repository, was deleted at 03:30Z.

**Outcome under the frozen rules.**

| Claim | Label |
|---|---|
| A, recipe vs matched twin | `RECIPE_EFFECT_NOT_ESTABLISHED` |
| Attribution | `NOT_TESTED_RECIPE_NOT_ESTABLISHED` |
| B, competitive | `NOMINEE_FAILS_DEV_UTILITY_GATE` |
| C, subgroup | `SUBGROUP_UNRESOLVED` |
| D, joint component | not claimed |
| E, novelty | not claimed |
| Pipeline controls | valid |

Every linked-pair recovery comparison favoured the nominee at the point. These comparisons are descriptive, because the nominee is not DEV-utility-eligible. Per recipient they did not all favour it:
- SCM's income recipient read higher than NEURAL_RAW's (+0.018 / +0.015) and H_LJ's (+0.006 / +0.017);
- its occupation recipient read higher than H_LJ's (+0.041 / +0.030).

Claims A and B fail, and attribution is not tested, because the nominee **misses one DEV_REPORT utility point clause**: occupation NLL excess against Ucal on seed 0 is 0.0106, against a 0.010 cap. Claim C is unresolved for its own reason: men's guard clause bounds (0.0111 / 0.0127) exceed 0.005.

## 1. What was fitted and measured

**The utility bank.** Each seed has 31 releases from one teacher cache (five household-fold CatBoost task teachers, out-of-fold on MODEL_FIT) and one fine partition (512 occupied KL-Bregman cells):
- C0, the unmerged partition;
- SCM at β = 0.5, 1.0 and 1.5;
- a task-only twin for each SCM setting, matched to its final cell count;
- PUB_IBSI at three γ values for each setting, at the matched count;
- five shuffled-signal controls for each setting.

Every matched arm reached its target count on both seeds (`FIT_SUMMARY.json`). Seed 0's k-means stopped at the 50-iteration cap without converging; seed 1 converged at 33. Both are within the protocol.

**Selection, on MODEL_SELECT only.**
- β = 0.5 and 1.0 pass all 24 held-out point clauses (12 per seed). β = 1.5 fails occupation NLL (0.0106–0.0118).
- Nominee: **SCM β = 1.0**, with 261 / 266 cells and 32 income groups. Its cross-fitted pair proxy is 0.6105 / 0.6085, against 0.6461 / 0.6457 for its twin.
- The registered prediction (twin minus SCM ≥ 0.020 on the proxy) held: 0.036 / 0.037.
- Comparator nominees: IBSI at γ = 1.0 and shuffled control r2, both eligible.

**The final bank, per seed:** SCM, TWIN, IBSI, SHUF, C0, the frozen NEURAL_RAW ρ1 E2 and the frozen H_LJ.
- Fresh readers: 8 per view (the inherited 7 plus the frozen lookup reader), 24 candidates for the pair, fitted on ATTACK_FIT (8,001), selected and oriented on ATTACK_VALIDATION (3,000).
- The frozen comparators' reader scores replay the frozen-hybrids study bitwise (14 of 14 per seed).
- Shuffled-label and planted-signal controls pass.

## 2. Did bounded compression preserve usefulness?

**On MODEL_SELECT, yes, at the point,** for the unmerged C0 and for SCM at β ≤ 1.0. C0 alone is already within every cap.

**On DEV_REPORT the nominee is at the boundary.**
- 23 of 24 point clauses pass. Occupation NLL excess is 0.0089 / 0.0080 against U0 and 0.0106 / 0.0082 against Ucal (cap 0.010).
- Confidence status for all 24 clauses is `UNRESOLVED`: no clause's family-wise (K = 42) upper bound is below its cap, and no family-wise lower bound is above it. Confidence status is defined on that family interval. At the one-sided α/3 clause level, six upper bounds are below their caps: income NLL against U0 (0.0081 / 0.0087), and occupation accuracy loss against U0 and against Ucal (0.0085 / 0.0093).
- Every comparator in the bank, TWIN, IBSI, SHUF and C0 included, passes all 24 DEV point clauses.

**The subgroup cost falls on men.** Men's within-SEX occupation Brier excess against U0 is 0.0062 / 0.0078 on DEV, above the 0.005 guard at the point, with status unresolved. Women's is −0.0015 / −0.0048.

## 3. Did SEX-guided merging help beyond ordinary compression and the published objective?

Descriptively, SCM read lower than every comparator in this section. Only against the matched twin is the margin beyond the 0.020 recipe target at the clause bound. Against the published IBSI objective it is 0.014 / 0.014, with family lower bounds of 0.003 / 0.001.

Formally nothing is established: claim A failed the DEV utility gate, so attribution was not tested. The comparisons are matched on cell count, not on utility. SCM spent more plug-in occupation NLL than every comparator, and it is the only release here that fails a DEV point clause. Pair AUC on DEV_REPORT, seed 0 / seed 1:

| Release | Pair AUC | SCM's benefit (comparator − SCM) | Status |
|---|---|---|---|
| SCM β = 1.0 | 0.6175 / 0.6219 | — | — |
| Matched task-only twin | 0.6584 / 0.6595 | 0.041 / 0.038 | clause bounds 0.029 / 0.025, above 0.020; family intervals above 0.020 |
| C0 | 0.6620 / 0.6626 | 0.044 / 0.041 | secondary family lower bounds 0.029 / 0.026 (descriptive) |
| PUB_IBSI γ = 1 | 0.6317 / 0.6358 | 0.014 / 0.014 | family lower bounds 0.003 / 0.001 |
| Selected shuffled control (r2) | 0.6560 / 0.6577 | 0.038 / 0.036 | family lower bounds 0.022 / 0.020 (descriptive) |

- **Task-only merging alone changed little.** The matched twin, which uses SEX only through the within-SEX guard, is 0.0035 / 0.0030 below C0 (unresolved).
- **SCM against the twin.** At the same cell count, SCM is 0.041 / 0.038 below the twin, but it also spent far more utility. Its plug-in occupation NLL was 0.0100 nats against the twin's 0.0027 / 0.0026, and it fails a DEV point clause that the twin passes. The record therefore does not separate the effect of the SEX-guided merge order from that of the extra utility spent.
- **Against merging with permuted SEX rates** (the shuffled control, which keeps the true-label guard and spent 0.0039 / 0.0038 nats), SCM read lower by 0.038 / 0.036.
- **Against the published IBSI objective** at the same capacity (0.0064 / 0.0066 nats), SCM read lower by 0.014 / 0.014.
- All of these are descriptive. Attribution is tested only when claim A is established. A failed the utility gate, so none of these is an attribution result.

## 4. Did it beat the neural comparator?

**Not as a claim, because the nominee is utility-ineligible on DEV_REPORT.** The frozen report code gates claim B on the DEV 24-point utility gate, following the source convention. The protocol text named that gate only for claim A (verifier D2; `PROTOCOL_POSTRUN_CLARIFICATIONS.md` item 1). On the competitive clauses alone, all four lower bounds are above 0: 0.0084 and 0.0024 against NEURAL_RAW, and 0.0102 and 0.0074 against H_LJ. The nested 0.020 superiority is not established on either comparator, and the seed-1 family interval against NEURAL_RAW includes 0. The gate fails on a single point clause (0.0106 against 0.010), whose confidence status is UNRESOLVED (family interval [−0.0000, 0.0212]).

**Descriptively:**
- **Pair recovery.** SCM's pair AUC is lower than NEURAL_RAW's by 0.0225 / 0.0159; the clause bounds are above 0 on both seeds, and the seed-1 family interval includes 0. It is lower than H_LJ's by 0.024 / 0.021. Material superiority (0.020 at the clause bound) is not reached on either comparator.
- **Per recipient, against NEURAL_RAW.** SCM's occupation recipient reads lower (−0.021 / −0.010). The secondary family interval is below 0 on seed 0 and unresolved on seed 1. Its income recipient reads higher (+0.018 / +0.015, unresolved).
- **Per recipient, against H_LJ.** SCM reads higher on both recipients. Income is 0.564 / 0.565 against 0.558 / 0.549, and occupation is 0.601 / 0.613 against 0.560 / 0.583. These are descriptive and not endpoints.

## 5. The linked pair, stated carefully

**What the outputs show.** On each of the four recorded roles (MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT) and both seeds, SCM's occupation output takes as many distinct float64 vectors as there are occupied cells: 260 / 265 on ATTACK_FIT and 253 / 259 on DEV_REPORT. Its income output, with 32 values, is a function of the cell. So for these people the occupation output identifies the shared cell. **For a Bayes-optimal reader, the linked pair adds nothing beyond the occupation output for these people.** This rests on the recorded counts (verifier check 6). Ties among stored cells never occupied on these roles were not checked. That follows from the shared partition; it is not a new collusion result.

**What fitted readers did.** They still differ: the pair reader reached 0.6175 against 0.6010 for the occupation-only reader on seed 0, and 0.6219 against 0.6131 on seed 1.

**What this is not.** The fitted lookup table is not a population Bayes oracle, and finite-cell merging does not certify protection against arbitrary readers. The lookup reader won one view, the seed-1 twin's income recipient. Every pair winner came from the inherited slate, so the 7-reader slate gives the same pair figures.

## 6. The algorithmic bottleneck the evidence identifies

**Occupation log-loss is the binding held-out utility clause at these caps.** In the fit, every SCM setting stopped with two plug-in budgets exhausted, occupation NLL and grouped income Brier, so the fit record alone does not say which one blocked further merges. On held-out people, only occupation NLL breached a cap.
- At β = 1.0, SCM spent its whole plug-in occupation-NLL budget (0.0100 nats relative to C0) and its whole grouped income-Brier budget (0.005). It spent much less occupation Brier (0.0034) and within-SEX guard (0.0031–0.0035).
- The twin reached the same cell count for 0.0027 / 0.0026 nats. At the same count, IBSI spent 0.0064 / 0.0066 and the shuffled control 0.0039 / 0.0038.
- The merges that remove SEX signal pool cells whose occupation probabilities differ, and that costs occupation log-loss. The plug-in cost of a merge is the mass-weighted KL divergence of the pooled cells' occupation vectors from their merged mean: lost resolution, not lost calibration under the fitted law.
- On held-out people the occupation-NLL increase relative to C0 was 0.0106 / 0.0111 on MODEL_SELECT and 0.0134 / 0.0092 on DEV_REPORT, against the plug-in 0.0100. The plug-in cost understated the held-out cost on MODEL_SELECT and on DEV_REPORT seed 0, and overstated it on DEV_REPORT seed 1. It is not a bound on the held-out cost.
- C0's occupation-NLL headroom below the references absorbed most of the increase only on MODEL_SELECT seed 0 (0.0071 / 0.0062 nats against U0 / Ucal; 67% / 59%). Elsewhere it absorbed a minority: 40% on MODEL_SELECT seed 1, 34% / 21% on DEV_REPORT seed 0 and 14% / 11% on DEV_REPORT seed 1.
- β = 1.5, which would merge further, already fails occupation NLL on MODEL_SELECT.

**Where the cost lands.** The measured within-SEX cost concentrates on men. That cost is occupation Brier; NLL by SEX was not measured. Men's excess against U0 (0.0062 / 0.0078) is the largest of the seven final-bank releases on DEV_REPORT, on both seeds. Women's is negative (−0.0015 / −0.0048).

## 7. What remains unresolved or ineligible

- The nominee fails one DEV point clause by 0.0006, so claims A and B are not established and attribution was not tested.
- All 24 confidence statuses are unresolved, and the subgroup status is unresolved.
- The seed-1 family interval against NEURAL_RAW includes 0.
- Only one recipe and one capacity were benchmarked: 512 fine cells, 32 income groups and three fixed β values.

**Not tested:**
- other capacities, teachers or budgets;
- attackers with other access (target inputs, parameters, queries);
- auxiliary samples of other sizes;
- imperfect linkage;
- any other population.

**Scope.** The failure is scoped to this fixed recipe and capacity: the selected SCM setting reduced measured recovery but did not stay inside the DEV_REPORT utility caps. It does not show that shared partitions fail. Equally, the favourable recovery differences are measured against the declared readers and are not evidence of a competitive method.
