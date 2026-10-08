# Team A method specification — one sampled confidence token

**Status:** preregistration draft for S1 review. No new Adult fit is authorized by this file alone. The campaign-wide lock, S2's safe data path and shared endpoint rules govern implementation.

## 1. Input, bank and release

For each pinned teacher seed and recipient r, the provider computes the frozen U0 score vector from the permitted 83-column input. It computes Ucal with the fixed hcal H-GLOBAL-TEMP rule for the utility reference only. The **input bin** `T_r` is the inherited qpc fine partition assignment of that recipient's **U0 score**, with at most 32 bins per predicted class for income and 128 per predicted class for six-class occupation. These fine partitions use score-only KL/Bregman k-means on OSF_DEFENSE_FIT; source IDs and fingerprints must be admitted before use. No task label or SEX is an inference input. The input bin is private and is **never** transmitted.

The fixed **output bank** for each seed and recipient is the distinct token identities and decoded probability vectors of hcal's task-only `U|FINE-TASK|i8o64|H-GLOBAL-TEMP` release, at most 8 income and 64 occupation tokens **per predicted class**. It is selected before new fits, with no outcome-based addition. Distinct IDs stay distinct even when their vectors are numerically equal. A strict-argmax output for **every possible** predicted class is a deployment gate; if the inherited bank lacks one, the proposed no-fallback channel is blocked until a bank extension is prospectively locked, and the existing outcome cannot be repaired by silently adding a code. Any source fallback or empty-class reserve must be disclosed and shared identically by controls.

`Q_r(z|t)` is a row-stochastic matrix. A row supports only bank IDs whose predicted class equals the class of input bin `t`. The two channel matrices are trained jointly from the aligned fitting law but are **locally executed**:

```
provider -> frozen teacher r -> U0 score r -> fine input bin T_r
         -> independent private RNG draw Z_r ~ Q_r(. | T_r)
         -> recipient r receives {Z_r, bank_prob_r[Z_r], decision_r, public metadata}
```

The sampler for r cannot read the other score, other bin, other sampled output, true task labels, SEX or row ID. There is no shared per-person random seed and no post-draw cross-party coordination. The modeled service produces **one release per person**, with no repeated-query API in this pilot. Protection under repeated independent draws is outside this contract; such a service would require a separately locked aggregate attack. The public channel matrices, bank, code and weights are assumed known to a defense-aware attacker; their availability does not reveal an individual's untransmitted input bin.

Each recipient's attack view includes the sampled bank **ID**, probability vector, decision, version/model IDs, source/bank hashes, any flag, code length or fallback value, and all other transmitted metadata. `fallback=false` is a constant under this design; no raw-score fallback is allowed. The coalition sees both aligned recipient views and may ignore either. A recipient gets no other party's per-person output unless it joins the coalition, and neither view includes the provider's private 83-column input, row identity or protected labels. The public model and code do not themselves reveal a private per-person provider input. An attack on only the probability vector is incomplete if ID or metadata is transmitted. No input fine ID, `Q[t,:]`, averaged probability, clean U0/Ucal score or clean head output is sent. The exact sampled probability is what utility scoring uses. Duplicate output vectors do **not** justify suppressing their different IDs from attacks.

This is the **SCORE** disclosure track only. It makes no feature-sharing claim.

## 2. Fitted finite-law optimization

On OSF_DEFENSE_FIT only, construct the aligned empirical count law of binary SEX and `(T_1,T_2)`, plus task-label counts per input bin. The old assessment and HEAD_VALIDATION labels must never be materialized by the loader. Candidate fitting may use SEX and both task labels from the allowed defense-fit role; no such labels enter deployment. S2 must provide a path whose input artifacts contain only allowed rows and labels.

For each view `v in {1,2,12}`, let `G_v(Q1,Q2)` be the exact Bayes SEX classification accuracy under this fitted finite law, using respectively `Z1`, `Z2` and `(Z1,Z2)`. For binary SEX, `G_v = [1 + sum_z |P(S=1,z)-P(S=0,z)|]/2`. The local and pair output laws follow `Q1(z1|t1) Q2(z2|t2)` conditional on the two input bins. `F_v` is the fitted class-only decision floor for that same view. `R_v` is the fitted Bayes accuracy when the corresponding private input bin(s) are fully observed; `F_v <= R_v` by inclusion. If a denominator is zero, use an exact floor constraint rather than dividing by zero.

Fit **one** protected stochastic design by minimizing `a >= 0` subject to **all three** constraints

`G_v <= F_v + a (R_v - F_v)` for `v = 1,2,12`.

This minimax normalization prevents a coalition improvement bought by shifting leakage to one recipient. It is a finite-law Bayes-accuracy objective, not a bound on test AUC, mutual information, other sensitive attributes or the population. The class-only floor is structural for this fitted view; confidence utility of class-only must be measured separately and it is never assumed eligible.

For each task r and each frozen reference `b in {U0,Ucal}`, constrain fitted **mean observed-label** NLL excess at or below 0.010 nats and mean multiclass Brier excess at or below 0.005. Fixed bank vectors make both expected costs linear in each Q. Use natural log, probability clipping at `1e-12`, multiclass Brier `sum_k(q_k-1{k=y})^2`, and the campaign's exact boundary convention. These are fitted-law expectations under one sampled output; they are **not** a held-out or per-person guarantee. Structural class preservation fixes the released decision to the teacher's class for this score-only method; check tie semantics and strict bank argmax exactly.

The joint problem is nonconvex because `P(z1,z2)` contains products of Q1 and Q2. With one matrix fixed, the three Bayes-accuracy epigraphs and all mean costs form an LP. Use SciPy HiGHS `linprog` for block updates. The two starts proposed for the campaign lock are the matched-bank TASK-DET pair and TASK-RD pair, both fitted before the protected channel and both independently checked against the mean caps; if one is infeasible, record that failure and do not improvise a replacement. Fix the sweep order, at most five full alternating sweeps per start, and a total per-arm CPU ceiling after S3's smoke receipt. Every control using this search gets the same starts/sweep rule or a matched total compute ceiling. Verify primal residuals independently. Each block update has a **predeclared lexicographic tie break**: first minimize primary `a`; then hold `a <= a_opt + 1e-9` with all original constraints and minimize `G_1 + G_2`. This permits movement along a flat pair-recovery face without changing the primary criterion. Apply the same tie break to matched deterministic and sequential block updates where applicable, and log both stages and residuals. Report each start, final objective and any unresolved local optimum. Do not call a stationary point a global optimum. A deterministic ablation restricts every Q row to one-hot and uses a matched-budget MILP/block search; tiny laws can be exhaustively enumerated where practical. Solver failures are recorded, not silently replaced by a different objective. A fitted-law objective advantage is not a fresh-attack advantage.

## 3. Matched comparisons and actual releases

All new arms use the same score input bins, output bank, class support, calibration, mean utility constraints, two paired seeds, fitting rows, deployment interface, attacker slate, restarts and declared compute ceiling unless their scientific role makes a restriction explicit. Required new arms:

1. **JOINT-STOCH** — the protected Q1/Q2 above.
2. **JOINT-DET** — one-hot structural ablation with identical objective and cap.
3. **TASK-RD** — a task-only stochastic rate-distortion control independently minimizing empirical `I(T_r;Z_r)` under identical utility caps. This is privacy-label blind and is distinct from merely maximizing `H(Z|T)`, which need not minimize rate. Preserve a strong task-only deterministic map as a separate control.
4. **LOCAL-STOCH** — separately protected local channels, attacked jointly after independent design.
5. **SEQ-12 and SEQ-21** — first local channel fixed, second optimized against the resulting coalition view, with the same local and mean-utility constraints. These are local non-adaptive adaptations, not Taylor et al.'s full-database adaptive algorithm.
6. Historical continuous U0/Ucal, CLASS, hcal task-only FINE-TASK and strongest compatible qpc/lra/hcal privacy controls, only as exact immutable aliases when their source, split, seed, input, bank/decoder, output view and budget identities match. Otherwise rerun or label them historical context. Direct comparisons with older score releases use their actual complete released views and equal fresh attacker capability.

The evaluator must score actual sampled IDs/vectors under a frozen draw plan, account for mechanism variance and paired people across seeds, and include metadata/fallback readers, linear, nonlinear and ignore-recipient coalition readers. An expectation over Q may be reported as a fitted or analytic diagnostic; it cannot stand in for a sampled-recipient evaluation. Fresh attackers fit only on ATTACK_FIT_NEW and report on INNER_SELECTION. No old assessment person-level arrays or labels may be opened.

The class-only release supplies a decision privacy floor but may fail useful confidence. U0 and Ucal reference identities are both frozen and both conjunctions are reported. A complete common pass requires all applicable utility clauses against both. Accuracy and class recall are still reported, even though this channel structurally preserves the frozen decision. Promotion and simultaneous intervals follow S1's lock; point feasibility is only a promotion screen. Every seed and every task/view is reported, with no best-seed or best-draw selection.

## 4. Exact toy laws, checks and kill conditions

S1's `results/pcrl_method_portfolio_v1/shared/SYNTHETIC_LAWS.json` is the **authoritative frozen** five-law set, L1–L5. Expand these laws by independent enumeration and verify normalization and conditional independence. No separate Team A toy-law draft is part of the experiment. The expected assertions are:

- NULL: fitted input-bin SEX Bayes accuracy equals the class-only 1/2 floor; no privacy improvement can be below it.
- REMOVABLE: shared law L2 permits both recipients to collapse the explicit S-dependent confidence detail to the more confident `.9` true-class bank output for each decision class. This changes some bank choices while preserving task labels and decisions, hiding the explicit S clue and improving mean NLL/Brier in this law.
- COMPLEMENTARY: each input alone has SEX accuracy 1/2 but the pair determines S; any local-only privacy check misses that clue.
- NONCOMPLEMENTARY: the pair carries no more Bayes SEX accuracy than recipient 1; an ignore-recipient reader reproduces the local risk.
- CONFIDENCE: shared law L5's variable and constant probabilities all predict class 1, yet the input-conditioned probabilities have strictly better expected NLL and Brier. The class-only floor therefore need not be utility eligible.

The toy laws test capability and correctness, not prevalence in Adult. Exact enumeration is an oracle only at the declared tiny size; Adult alternating LP and deterministic heuristic are not exact global oracles. If the channel is algebraically the old ACS Q with only renamed variables, report it as transfer/ablation rather than method novelty. If an Adult score channel ties the strongest deterministic or sequential arm under complete-view attacks, it is not a competitive stochastic or joint result.
