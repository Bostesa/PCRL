# HANDBACK 1: mechanism investigator, RFA1 (reader-fit-aware release)

Campaign `pcrl_reader_fit_aware_release_v1`, code package `rfa/`. Worked 2026-10-25, 21:05Z to 21:34Z (`date -u`). This version
includes the coordinator's interface request, the occupation-only design decision (views and episode sizes), and the challenger's
required changes 1 to 4.

Rules followed:
- Everything is label-free. Nothing under `.private/` or `<worktrees>/` was opened, and no data, label,
  array or model file was read.
- Every number below comes from invented data.
- All computation ran on one thread.
- No git commands were run.

Total CPU used: about 25 CPU-seconds (6 selftest runs about 1.8 s each with start-up; 3 benchmark runs about 2.5 s each; scratch
scans about 5 s).

Files written:
- `rfa/fam/rfit.py`: the module (826 lines, sha256 prefix 04b22c664caa8c55).
- `results/pcrl_reader_fit_aware_release_v1/roles/investigator/RFIT_SELFTEST.json`: full selftest output (`all_pass: true`, 19
  checks, 1.0 CPU-s).
- `results/pcrl_reader_fit_aware_release_v1/roles/investigator/RFIT_BENCH.json`: one-thread cost per step in the occupation-only
  loop pattern.
- This file.

Two scratch scripts sit outside the repo, in the session scratchpad: a ridge / n scan and an occupation-only call check. Their numbers
are quoted below, and the selftest re-runs the key cases.

## 0. Summary up front

- The operation is the standard implicit (bilevel / Stackelberg) hypergradient. The outer objective is the held-out recovery
  surrogate on E. The inner problem is a ridge-logistic reader fitted on A. The total derivative is the direct path plus an
  implicit-function term. The implementation is exact:
  - over all 49 parameters of a tiny release, finite differences of the complete objective agree with FULL to a relative error of
    2.0e-10;
  - on a float64 U0-size network pair, along 3 random directions, the relative errors are 1.3e-10 to 1.2e-9;
  - in the loop's own pattern (registered views, income logits constant), the relative error is 1.9e-10 over the 35 occupation
    parameters;
  - STOPFIT is off by 48% in the same check, so the check does tell the two apart.
- **FB_PFD's gradient is a stopped-fit gradient.** Its criterion is an in-sample best response: D^2 = max_w of the Rayleigh quotient on
  the same 2,048 rows. By the envelope theorem, autograd through its closed-form solve equals the gradient with w = S^-1 Delta
  detached. I checked this on FB_PFD's own imported code: the relative difference is 1.4e-14. The challenger reached the same
  conclusion. OptNet-ARL has the same in-sample structure. The fit-path term is non-zero only because E differs from A and the outer
  criterion (an AUC) differs from the inner loss (BCE + ridge). With E = A and outer = -L_A, the fit-path term vanishes (ratio
  2.7e-13).
- On invented data the fit-path term is material at campaign sizes (2048 / 2048):
  - per term, ||fit|| / ||direct|| is 0.31 to 0.70, and cos(FULL, STOPFIT) is 0.77 to 0.96;
  - income view with the SIMPLE reader: exactly 0, a structural result;
  - all terms together: ratio 0.45, cosine 0.90.
- **The fit-path term mostly opposes the direct path.** cos(fit, direct) is -0.24 to -0.95 on 5 of 6 terms, and +0.04 on
  occupation/RFF. So FULL is usually a shorter, rotated version of STOPFIT. It holds back part of the push to fool the current
  reader, the part that a re-fitted reader would undo. Whether this lowers recovery by the campaign's independent readers is
  **untested**. I do not assume it works.
- **Risk to adjudicate: ridge bias.** Part of the fit path does not vanish with more data. On equal-covariance Gaussian occupation
  logits at n = 32,768 the ratio is 0.32 at the registered ridge 1e-2, against 0.03 at ridge 1e-4. So FULL can partly optimize
  against this reader's regularization. Nearly collinear pair coordinates make it worse: at ridge 1e-4 the ratio is still 0.56 at
  n = 32,768, because H^-1 amplifies weak directions. At n_A = 4096 the finite-sample part shrinks (about 1/sqrt n_A) while the
  ridge-bias part does not.
- **Cost is small.** The occupation-only loop pattern with registered views, diagnostics on, one thread, at the recommended
  n_A = 4096 / n_E = 2400:

  | Arm | CPU per step |
  |---|---|
  | FULL | 0.034 s |
  | STOPFIT | 0.021 s |
  | ALTCRIT (including its lag-diagnostic exact solve) | 0.028 s |
  | TASK | 0.001 s |

  The three reader arms total about 6.6 CPU-minutes for 2 versions x 6 units x 400 steps.

## 1. The module and the interface (`rfa/fam/rfit.py`)

### `RECIPE_RF`: every frozen value

- **Episode.** Top-level `n_A` = 4096 and `n_E` = 2400, plus an `episode` block with the rationale.
- **Views.** `views` = ['occupation', 'pair'], configurable. The income-view term is skipped. The pair view keeps START's constant
  income coordinate.
- **Moments.** `moments_convention`: mu_A and sd_A are detached constants of each call, so the FULL fit path runs through w* only.
  STOPFIT and ALTCRIT share this convention, as FB_PFD's `poly2` does.
- **Readers.** Families, features, the RFF seed rule, ridge, Newton settings and guards (section 6).
- **Surrogate.** Its definition and scale rule (section 6).
- **ALTCRIT.** Inner method, rounds, learning rate, initialization and call order.

### `EpisodeObjective(mode, seed=0, recipe=RECIPE_RF)`, with mode one of `FULL`, `STOPFIT`, `ALTCRIT`

**`objective(zA_inc, zA_occ, sA, zE_inc, zE_occ, sE, moments=None, diagnostics=True)`** returns `(S, info)`, with S = S_dir + S_fit.
S is a float64 scalar to DECREASE. Its autograd gradient is the mode's update direction.

**`objective_parts(...)`** (challenger 1) returns `(S_dir, S_fit, info)`.
- S_dir = sum_v mean_f R(phi_E(theta); w constant). It is exactly STOPFIT's objective at this call's w*. Checked: max abs difference
  of grad(S_dir) from the STOPFIT gradient is 0.0 on the same episode.
- S_fit = sum_v mean_f [-v . grad_w L_A(w*, theta)], with v = H^-1 dR/dw detached. It carries the fit path through the A outputs and
  its value is about 1e-15. Checked: grad(S_fit) equals FULL minus STOPFIT to 3.5e-18.
- For STOPFIT and ALTCRIT, S_fit = 0 * S_dir: still connected to the graph, so its gradient is a zero tensor rather than an error.

**`info`** holds plain floats:
- `S`, `S_dir`, `S_fit`, and `P` (= S_dir);
- per term `<view>/<family>`:
  - `R` and `auc_binormal_E`;
  - `orientation_sign_E` (the sign of dmu of the held-out score);
  - Mann-Whitney monitors `auc_exact_E` and `auc_exact_A`;
  - FULL / STOPFIT: `newton_iterations`, `newton_max_abs_grad`, `cond_H`;
  - FULL: `norm_grad_w_R`, `norm_v`, `fit_scalar_value`;
  - ALTCRIT (challenger 2): `critic_rounds`, `critic_max_abs_grad_A` = max |grad_w L_A(w_alt)|, and `critic_rel_lag` =
    ||w_alt - w*_A|| / ||w*_A||. Here w*_A is an exact Newton solve on the same A rows, a diagnostic only, never in the gradient;
- ALTCRIT summaries: `critic_rel_lag_median` and `critic_max_abs_grad_A_max`;
- FULL summary: `output_grad_norms` = {`direct_wrt_E_outputs`, `fit_wrt_A_outputs`, `fit_share`}, the output-space proxy for the
  fit-path share;
- `refused` is always False, because refusals raise.

**Failures** raise `ReaderRefused`, a subclass of `FloatingPointError`:
- any non-finite logit, feature, surrogate or fit-path value;
- A moments with sd < 1e-8;
- fewer than 16 rows of a SEX class in A;
- Newton not converged to 1e-10;
- cond(H) > 1e8;
- a degenerate held-out score variance.

Logits that are not float64 raise `TypeError`. FULL without the A-row graph raises `ValueError`. The `seed` is recorded only: the RFF
frequencies are frozen and identical in every version and fold.

### `monitor_fisher2(zA_inc, zA_occ, sA, zE_inc, zE_occ, sE, views=VIEWS, ridge=1e-3)` (challenger 3)

A diagnostic only, never in any gradient. For each of income, occupation and pair, it builds FB_PFD's reader:
- the pooled within-SEX covariance discriminant, w = (S + 1e-3 tr(S)/p I)^-1 Delta;
- on degree-2 polynomial features of the logit differences, standardized by the A-row moments (no soft bound);
- fitted on A and scored on E.

It returns the exact Mann-Whitney AUC on E, oriented by A, as `{view: auc}`, plus `sd_A` = {income: [1 SD], occupation: [4 SDs]}, the
per-coordinate population SD of the logit differences on A. The loop compares `sd_A` with START's to flag a shrink of more than 10x.
On invented data it returns income 0.512, occupation 0.823, pair 0.837.

### `selftest()`

Run it as `<repository>/.venv/bin/python -m rfa.fam.rfit selftest [--out PATH]`. It returns `{check: bool}` plus
`all_pass`, `numbers` and `cpu_s`, and exits 1 on any failure. Current result: 19 checks, all pass.

### Operation, per episode and per view v

1. **Features.**
   - Logit differences: u = z1 - z0 (income); u_k = z_k - z0, k = 1..4 (occupation); the pair is both.
   - Standardize: s = 6 tanh(((u - mean_A) / sd_A) / 6). The soft bound is smooth and close to the identity for |s| < 3. It caps the
     leverage of outlier logits, because the readers see unbounded logit differences while the wire carries bounded probabilities.
2. **Reader families.**
   - SIMPLE: [s, 1].
   - RFF: [s, sqrt2 cos(s Omega_v + b_v) (64), 1], with Omega_v ~ N(0, I / d_v) and b_v ~ U(0, 2pi), from a frozen seed
     `seed_for("RFA1-rff", view)`.
   - Why RFF and not ReLU random features: they are smooth, so the mixed derivative in the implicit-function theorem exists
     everywhere. ReLU features make the fit path piecewise and the finite-difference checks sensitive to kinks. The cosine features
     are bounded, which helps conditioning. And this is the standard random-feature approximation of a Gaussian-kernel logistic
     reader.
3. **Inner problem.**
   - L_A = mean BCE + 0.01/2 ||w_(-intercept)||^2.
   - Solved by Newton with Armijo backtracking in float64, detached. Warm start from the previous call (the minimizer is unique).
     Iterate to max|grad| <= 1e-13, or until it stalls below 1e-10. Iteration cap 60.
   - Measured: 5 iterations warm, at most 7 cold.
4. **Surrogate.**
   - R = (AUC_b - 1/2)^2, where AUC_b = Phi(dmu / sqrt(var_1 + var_0)) of the held-out score t_E = phi_E w*.
   - The variances are not detached and there is no temperature, so AUC_b is exactly invariant to scale and shift (checked to 1e-14).
   - R is symmetric, so an inverted score is not rewarded (checked).
   - S = sum over views of the mean over families.
5. **Modes.**
   - FULL: direct path + fit path.
   - STOPFIT: direct path only.
   - ALTCRIT: direct path through a persistent critic per (view, family). The critic is initialized by the exact fit on its first
     call's A rows, then gets 10 full-batch Adam steps at lr 3e-3 on each new episode's A rows, applied before the release gradient
     on that episode's E rows.
6. **When ALTCRIT is an alias of STOPFIT.** It is an exact alias if and only if the critic is re-solved exactly on the current A rows
   at the current release. With `altcrit_inner='newton'` and cold starts, the gradients are bitwise identical (max abs difference 0.0;
   relative lag 0.0). A first-order inner loop run to convergence is an alias up to its tolerance. The registered 10 Adam rounds lag:
   - on invented data, relative lag median 0.67, critic max|grad| 0.033, cosine 0.79 against STOPFIT;
   - so it is a distinct control, not a NEAR_ALIAS under the challenger's 1% rule;
   - the real-data lag decides, and is logged.

   The first ALTCRIT step equals STOPFIT's first step. STOPFIT's warm start makes a campaign alias exact only to about 1e-13; set
   `newton_warm_start=False` for a bitwise one.

## 2. Synthetic checks: invented data, deterministic, float64

| Check | Setting | Result |
|---|---|---|
| Complete-derivative finite differences, all parameters | tiny linear release, 49 parameters, n_A = n_E = 256, 3 views x 2 families, eps 1e-5, A moments frozen at theta0 | FULL vs FD: relative error 2.0e-10. STOPFIT vs FD: 0.48. PASS |
| Same, network scale | float64 U0-size pair (145-64-64-16-K), 3 random unit directions | relative errors 1.3e-10, 5.0e-10, 1.2e-9. PASS |
| Same, loop pattern | registered views (occupation, pair), income logits constant, FD over the 35 occupation parameters | relative error 1.9e-10; the income-view term is absent; zero gradient on income parameters. PASS |
| Optimization sign | step -eta grad(FULL) | eta 1e-3: P falls by 3.425e-5 against a first-order prediction of 3.423e-5. eta 1e-2: falls by 3.44e-4. PASS |
| Envelope identity | E = A, outer = -L_A | fit / direct = 2.7e-13 |
| Envelope fails out of sample | E != A, outer = -L_E | ratio 0.64. PASS |
| FB_PFD is a stopped fit | imported `sar.fb.pfd`, pair view: autograd through the solve vs the Rayleigh quotient at detached w = S^-1 Delta | relative difference 1.4e-14. PASS |
| `objective_parts` | same episode as STOPFIT | grad(S_dir) - grad(STOPFIT) = 0.0; grad(S_fit) - (FULL - STOPFIT) = 3.5e-18; S_fit value -2.8e-15. PASS |
| Exact zero | income view, SIMPLE reader | ratio 5e-16. PASS |
| Near coincidence | Gaussian occupation logits, SIMPLE, n = 32,768, ridge 1e-4 | ratio 0.031, cosine 0.9995 |
| Same data, registered ridge 1e-2 | as above | ratio 0.32, cosine 0.95. Ridge bias |
| Nearly collinear pair | ridge 1e-4, n = 32,768 | ratio 0.56 |
| Material difference | n_A = 96, RFF (70 parameters), pair view, weak plus variance signal | ratio 0.92, cos(FULL, STOPFIT) 0.51, cos(fit, direct) -0.61. PASS |
| Campaign size, invented | 2048 / 2048, all 6 terms | ratio 0.45, cosine 0.90 |
| ALTCRIT | as in section 1, item 6 | Newton: bitwise alias with lag 0. Adam: lagging, not an alias. Lag diagnostics present. PASS |
| Fisher-2 monitor | invented data | runs, oriented (all AUC > 1/2), plain floats. PASS |
| Guards | as in section 1 | one-class A, a NaN logit and a constant view are refused; the inverted score is not rewarded; scale and shift invariance holds; `info` is plain floats. PASS |

The 2048-row campaign-size terms, one by one:

| Term | ratio | cos(FULL, STOPFIT) | cos(fit, direct) |
|---|---|---|---|
| income/simple | 0 | 1.000 | (undefined; the fit term is exactly 0) |
| income/rff | 0.34 | 0.94 | -0.38 |
| occupation/simple | 0.41 | 0.91 | -0.35 |
| occupation/rff | 0.31 | 0.96 | +0.04 |
| pair/simple | 0.70 | 0.77 | -0.26 |
| pair/rff | 0.45 | 0.90 | -0.30 |

**When the ingredient matters.** The size of the fit path is ||(d grad_w L_A / d theta)^T H^-1 grad_w R(w*)||. It vanishes only when
the reader's training criterion is aligned with its held-out criterion at w*:
- a well-specified family;
- negligible ridge bias;
- large n_A;
- good conditioning.

It is large when:
- n_A is small relative to the number of features;
- the reader is misspecified;
- the ridge bends the direction;
- the coordinates are nearly collinear.

Ridge scan, scratch:
- simple/pair: the ratio stays at 0.43 to 0.68 up to n = 32,768 for every ridge from 1e-2 to 1e-6;
- RFF/pair at n = 2048: the ratio grows as the ridge shrinks: 0.49 at 1e-2, 0.76 at 1e-3, 0.92 at 1e-4, 1.13 at 1e-6.

## 3. CPU cost per release step: one thread, this Mac, torch 2.10 (`RFIT_BENCH.json`, v2)

Setup: the coordinator's loop pattern. The occupation network (145-64-64-16-5, float32) is trained, START's income logits are
constants, the views are the registered ones, the A and E rows are forwarded with their graph in every mode, diagnostics are on, and
there is a 256-row task batch with Adam and clip 5. Random weights and invented X. Mean of 4 timed steps after one warm-up:

| n_A / n_E | FULL | STOPFIT | ALTCRIT (+ lag-diagnostic solve) | TASK |
|---|---|---|---|---|
| 2048 / 2048 | 0.019 s | 0.013 s | 0.019 s | 0.001 s |
| **4096 / 2400 (recommended)** | **0.034 s** | **0.021 s** | **0.028 s** | 0.001 s |
| 4096 / 4096 | 0.043 s | 0.026 s | 0.035 s | 0.001 s |

Campaign estimate at 4096 / 2400, for 2 versions x 6 units x 400 steps:

| Item | Estimate |
|---|---|
| FULL | about 160 s |
| STOPFIT | about 100 s |
| ALTCRIT | about 135 s |
| START gate (64 episodes per unit, FULL parts + 2 backward passes) | about 40 s |
| Calibration | a few seconds |
| Reader work in total | about 7 CPU-minutes |

That is far inside 1.5 CPU-hours. START construction per fold (U0 recipe plus RAW-J to E2), the RAWJ ALT path and the out-of-fold
forward passes will dominate; I did not measure those here. The per-step working set is a few tens of MB.

## 4. Computational graph and the exact differences

```
theta (occupation network) --f_theta--> z_occ; z_inc = START constants (no graph); centred float64 logits on A, E, task rows
A rows: z_A -> u_A -> s_A = 6tanh(((u_A - mu_A)/sd_A)/6) [mu_A, sd_A detached] -> Phi_A(theta)
          Phi_A.detach() --Newton (no autograd edge)--> w*, H(w*)
          FULL only: g_A(theta) = Phi_A^T (sigmoid(Phi_A w*) - y_A)/n_A + kappa w*      [w* constant; value 0 at the solve]
E rows: z_E -> u_E -> s_E (A's constants) -> Phi_E(theta) -> t_E = Phi_E w* -> AUC_b -> R = (AUC_b - 1/2)^2
S_dir = sum_v mean_f R(Phi_E(theta); w*)                       (= STOPFIT's S)
S_fit = sum_v mean_f [ -v^T g_A(theta) ],  v = H^-1 dR/dw (constant)
FULL:    dS/dtheta = dS_dir/dtheta (E outputs) + dS_fit/dtheta = (dw*/dtheta)^T dR/dw (A outputs; IFT: dw*/dtheta = -H^-1 d g_A/dtheta)
STOPFIT: dS_dir/dtheta only (the A outputs get no gradient)
ALTCRIT: w_t = Adam^10(w_{t-1}; L_{A_t}(., theta_t) constant);  S = sum_v mean_f R(Phi_E(theta); w_t)  -> E outputs only
```

Only FULL's S_fit passes a derivative through the reader fit, and only through the A-row outputs. Neither the moments nor the ridge
pass a derivative; they are constants, as in FB_PFD.

**(i) NEURAL_RAW's online critics.**
- Nine persistent critics (linear / mlp32 / mlp64 x 3 views) read the wire (probabilities and decision) through a per-epoch
  DiagonalReference.
- Adam lr 3e-3, 10 critic rounds per release step on fresh 256-row batches.
- The release gradient is d/dtheta of risk = 1 - min(CE, H)/H on a 256-row privacy batch with the critics fixed: a direct path only,
  with a min-over-bank, a clamp at the entropy, and a per-recipient lambda.
- ALTCRIT is the closest RFA analogue (persistent, lagging, direct path only), but on logit-difference features with ridge-logistic
  families and the symmetric AUC surrogate.
- FULL replaces the lagging critic with an exact per-episode refit on household-separated A rows, evaluates on disjoint E rows, and
  adds S_fit.

**(ii) FB_PFD and FB_PFO.**
- A = E = the same 2,048-row batch, and the reader is the exact best response of the very criterion it is scored by.
- By the envelope theorem the gradient equals the stopped-fit gradient (verified to 1.4e-14). FB_PFD is an in-sample STOPFIT, an
  in-sample FULL with S_fit identically 0. It contains no generalization term.
- RFA differs in three ways: held-out E; an outer criterion (AUC_b) different from the inner loss; and different readers (linear +
  RFF logistic against a Gaussian discriminant on degree-2 polynomials).
- `monitor_fisher2` re-creates FB_PFD's reader held-out as a diagnostic.

**(iii) Plain continuation of NEURAL_RAW (TASK / CONT, lambda = 0).** The task cross-entropy only, with no SEX. The reader arms differ
from it only by + lambda S, with lambda shared.

## 5. Prior work: methods fetched, plus the closest named methods. Novelty is not claimed.

### Fetched with WebFetch (arXiv abstract page plus the ar5iv HTML method sections)

**OptNet-ARL**: Sadeghi, Wang, Boddeti, "Adversarial Representation Learning With Closed-Form Solvers", arXiv 2109.05535 (2021).
- The target and adversary are kernel ridge regressors. Their minimum MSE has a closed form,
  J_s(Z) = (1/n)||S~||_F^2 - (1/n)||P_{M_s}[S~^T; 0]||_F^2, computed on the same batch's embeddings and Gram matrix.
- The encoder solves min (1-lambda) J_y - lambda J_s, backpropagating through the closed form with a projection-norm derivative. It is
  called bilevel, but the envelope theorem and implicit differentiation are not named.
- Held-out data is used only at evaluation.
- Coincides with RFA: an exactly solved convex adversary differentiated through its solution.
- Differs from RFA: in-sample at its own optimum, so its solver derivative equals the stopped-fit derivative, as in FB_PFD. That is my
  inference from the stated objective; I did not run their code.

**RecUP-FL**: Cui, Meerza, Li, Liu, Zhang, Liu, arXiv 2304.05135 (2023; DOI 10.1145/3579856.3582819).
- Clients add a perturbation to shared gradients to fool attribute-inference classifiers.
- A zoo of 20 pre-trained substitute attackers is sampled 5 at a time. Iterative FGSM runs on 4 of them ("meta-train") plus one step
  on the held-out 5th ("meta-test"), repeated 10 times.
- The gradient passes only through FIXED pre-trained attackers' predictions; there is no differentiation through attacker training.
  "Held-out" means held-out MODELS (transfer across architectures).
- Coincides with RFA: it targets attacker transfer and generalization.
- Differs from RFA: RFA refits per episode on auxiliary rows, holds out ROWS, and differentiates through the fit.

### Closest published methods (from my knowledge; not fetched)

- **Implicit-differentiation bilevel hypergradients.** Bengio 2000; Pedregosa 2016 (HOAG); Lorraine, Vicol and Duvenaud 2020.
  Differentiable closed-form or convex heads: Bertinetto et al. 2019 (R2-D2); Lee et al. 2019 (MetaOptNet); Rajeswaran et al. 2019
  (iMAML). Our dS/dtheta is exactly this hypergradient (validation = E, training = A). Here theta parametrizes the features of both A
  and E.
- **Training-set poisoning through a learner's fit.** Biggio et al. 2012; Mei and Zhu 2015; Munoz-Gonzalez et al. 2017; Koh and Liang
  2017; Koh, Steinhardt and Liang 2022. Structurally the release poisons the auxiliary reader's training data (the A outputs) against
  its held-out recovery. The difference: the same map also generates the evaluation data.
- **Stackelberg / leader-follower gradients.** Fiez, Chasnov and Ratliff 2020: the leader's total derivative through the follower's
  best response via the IFT, the same formula. LOLA (Foerster et al. 2018): one opponent step. Unrolled GANs (Metz et al. 2017): k
  unrolled discriminator steps. RFA uses the exact convex optimum plus the IFT, with the best response fitted on different rows than
  the leader's loss.
- **Alternating-adversary representation learning, the ALTCRIT-type baseline.** Edwards and Storkey 2016; Elazar and Goldberg 2018
  (post-hoc attackers recover what an alternating adversary removed); Feutry et al. 2018.

Net: the operation coincides with the implicit-gradient bilevel / Stackelberg hypergradient, applied with a held-out row split and a
release map that generates both the training and the evaluation features. I claim no novelty.

## 6. Frozen recipe values (`RECIPE_RF` as committed) and the label-free provenance

| Item | Value | Provenance and reason |
|---|---|---|
| Episode | one per step, household-separated, from the unit's training rows; n_A = 4096, n_E = 2400 (coordinator sampler: whole households, slightly above) | Coordinator's declared recipient budget (FIT about 4,000, MODEL_SELECT FIT parts 0-4; SCORE about 2,400), so the fit path describes a reader with that much auxiliary data. Counts only, no labels or outcomes |
| Views | occupation, pair (pair keeps START's constant income coordinate) | Coordinator's occupation-only design |
| Families | SIMPLE [s, 1]; RFF64 | Width 64 is the top of the brief's 32-64 range; reasons for RFF in section 1 |
| Features | soft bound 6; detached A-row standardization; RFF seed `seed_for('RFA1-rff', view)`; bandwidth 1 / sqrt(d_v) | Set a priori |
| Ridge | 1e-2 (mean-loss form), intercept unpenalized | Written in the first draft at 21:15Z, before any computation, as a conventional moderate ridge for standardized features that keeps H >= 1e-2 I (cond about 1e3). Not tuned: the later invented-data scan only described the bias / variance trade-off and changed nothing |
| Newton | target 1e-13, acceptance 1e-10, at most 60 iterations, warm start | Set a priori |
| Guards | cond_max 1e8; at least 16 rows per SEX class in A; min_sd 1e-8 | Set a priori |
| Surrogate | S = sum_v mean_f (AUC_b - 1/2)^2; binormal, non-detached within-class variances, no temperature | Set a priori |
| ALTCRIT | Adam, 10 rounds, lr 3e-3 (NEURAL_RAW's critic schedule), full batch on A_t, exact-fit initialization | NEURAL_RAW's critic values |

No real data, label or model file was ever opened in this role.

Technical objections to n_A = 4096: none.
- The episode uses about 6.5K of a roughly 16K-row fold pool (41%).
- Conditioning improves with n.
- The cost is 0.034 s per FULL step.
- The one interpretive consequence: the finite-sample part of the fit path shrinks with n_A while the ridge-bias part does not, so log
  the per-term `norm_v` and the FULL `fit_share`.

**Monitors** (registered, descriptive):
- Per logged step, `auc_exact_E` against `auc_binormal_E` per term. Flag `BINORMAL_EXPLOIT_SUSPECT` if, over an epoch's logs, the
  pair-term AUC_b falls by more than 0.02 more than the exact AUC does.
- `monitor_fisher2` held-out AUCs and `sd_A` (a shrink of more than 10x against START).
- The ALTCRIT `critic_rel_lag_median`.
- The coordinator's parameter-space shares from `objective_parts`.

**lambda.** I agree with the coordinator's rule: rho = 1 x median ||grad CE_occ|| / median ||grad S_FULL|| over 16 fixed draws at
START, shared unchanged by FULL, STOPFIT and ALTCRIT. On invented data ||FULL|| is about 0.85 to 1.0 x ||STOPFIT||, which is
immaterial once lambda is shared.

**Optimizer and path.** Adam 1e-4, clip 5, 256-row task batches (THP1task), 40 steps per epoch, 400 steps.
- Checkpoints: E0..E10 as the coordinator plans. Optionally every 10 steps, following FB_PFO's lesson that FB_PFD's 40-step first
  checkpoint was too coarse when utility bound early; this costs only out-of-fold forward passes.
- Selection: the MODEL_FIT out-of-fold budget rule.

**Arms per version.** FULL (CAND); STOPFIT (ingredient ablation); ALTCRIT (ordinary alternating-critic control); TASK (CONT,
lambda = 0); RAWJ ALT (NEURAL_RAW's own recipe); plus the DQ CTRL if kept.

## 7. Caveats: not done, and open risks

- Every magnitude here comes from invented data: the fit-path ratios, the mostly negative cos(fit, direct), the lags, and the costs
  with random weights. Real-logit geometry decides the actual share. The challenger's START gate (MECHANISM_INERT) is the right first
  test.
- The ridge-bias component means FULL may partly optimize against this reader's regularization. Evaluate with readers that are not
  ridge-1e-2 logistic. The campaign's independent readers do this; `monitor_fisher2` adds FB_PFD's reader held-out.
- The finite-difference check proves the per-episode derivative exact, not its variance across episodes. S_fit is the derivative of a
  held-out estimate and is itself noisy, which is why the coordinator uses cross-group estimates.
- The detached moments and the tanh soft bound leave the readers slightly non-affine-invariant: a temperature move changes R a little.
  Moments are recomputed every call, so this is not cumulatively exploitable, but it is not exactly zero.
- Refusals raise, and the loop stops on them. No refusal occurred in any invented run.
