# Shared-context release v1: design specification (pre-fit, coordinator)

Status: draft for implementation, written 2026-09-24 ~17:50Z before any new release candidate was fitted and before any new outcome was seen. The binding version is whatever `PROTOCOL.md` + `RUN_MANIFEST.json` lock before fitting; this file tells implementers what to build.

Python: `/Users/nathansamson/PCRL/.venv/bin/python` locally. Scientific units run on one Linux x86 AWS host (same-host references); the Mac runs only unit tests and tiny smoke runs.

## 0. Reuse contract

Base everything on the predecessor package `experiments/pcrl_adaptive_release_v1` (AR) at 2ce5d171 and its dependencies `experiments/pcrl_task_aligned_cuts_v1` (TAC) and `experiments/pcrl_task_directed_release_v1` (TDR). New code lives in `experiments/pcrl_shared_context_release_v1/` (SC). Import AR/TAC/TDR functions; copy-and-modify only where a function hard-codes one row per T32 state. Do not edit AR/TAC/TDR files. See `agents/pipeline_fit/PIPELINE_FIT_MAP.md` and `agents/pipeline_audit/AUDIT_SELECTION_MAP.md`.

Data and roles: exactly the predecessor household roles (`AR/roles.py`, SHA256 of `pcrl_adaptive_release_v1|household_id`): nuisance_train 20%, audit_fit 20%, coefficient_split 25%, inner_selection 10%, inner_check 10%, outer 15%. The outer role stays behind `AR/outer_access` and is opened once by the coordinator after the lock. Legal deployable per-person inputs: `x` (X_A, 32), `ha` (H_A, 4), stored T0 code, stored teacher posterior `p`, stored residual `r`, stored historical `risk` (11), and frozen predictions computed only from those. Never `hb`, labels, ids, household, role, loss.

## 1. Channel: nested contextual policy mixture

For person x with old state t=T0(x), hard context k(x) in {0..K-1}, and frozen deterministic policies d_0..d_{M-1}: x -> {0..16}

    q(z|x) = B[t,z] + sum_m A[k(x),m] 1{d_m(x)=z}
    B >= 0, A >= 0, 0 <= eta <= 1, sum_z B[t,z] = 1-eta for all t, sum_m A[k,m] = eta for all k.

Hard contexts only in v1 (phi one-hot), so the matched deterministic subproblem is well defined.
d_0 = D17 exactly (the historical 32->17 one-hot map, per anchor).
eta=0, A=0, B=D17 is the exact reference witness; eta=0 embeds every T32 kernel (including the parent-restricted T32 control).

Per-person law is computed privately; the wire is one token. Deployment: `SharedContextRelease` object holding B, A, eta, frozen policies and context rule, emitting one keyed-persistent token (reuse the TAC/AR keyed HMAC session pattern).

### Coefficients (frozen decoder d, frozen attack a, weighting v in {U,W})

    C_B[t,z]  = sum_{i: T(i)=t} w_i^v loss_Y(y_i, d(H_Ai, z))                         (existing T32 block)
    C_A[k,m]  = sum_{i: k(i)=k} w_i^v loss_Y(y_i, d(H_Ai, d_m(x_i)))                  (new block)
    L_a = sum_{t,z} B[t,z] Acut_B[t,z] + sum_{k,m} A[k,m] Acut_A[k,m]                 (same for every cut)

with w_i^U=1/n_valid, w_i^W=PWGTP_i/sum PWGTP on the stated valid rows; per-person loss rows come from the existing per-person loss functions (`fit_b._attack_loss_rows`, task-loss rows) so the A block is just a (k,m)-grouped gather of per-person token-loss rows at token d_m(x_i).

### LP (utility form, "U")
min 0.5 C_U + 0.5 C_W over (B, A, eta) subject to row/eta equalities, nonnegativity, and every retained cut L_a >= rho_{r,v} - delta, rho = min over bank of L_a(D17) on the same coefficient rows (reuse `AR/reference.calibrate_reference`; generalise its witness check to the new variable vector). delta = 0.001.

### LP (privacy-first form, "P")
Same variables; maximise tau subject to AB/SEX cuts L_a >= rho + tau (both weightings), all other cuts >= rho - delta, task C_v <= C_v(D17) + 0.001 for v in {U,W} (mirror `AR/privacy_first.py`).

Validation for every solve: shapes, finite, nonnegative, per-person row sums = 1 within 1e-9, witness replay (D17 feasible), phase I, primal residuals, dual bound record.

### Alternation (same as Branch A)
Round 0: fit the frozen decoder on the predecessor coverage law (current + D17 + uniform)/3 using nuisance_train, selected on inner_selection; fit the attack bank on audit_fit (validated on inner_selection), cut coefficients on coefficient_split. Then for r=1..6: solve LP; refit decoder on the current per-person law (plus D17 and uniform coverage, as AR); fit best-response attacks on the current law; add to bank; rebase all cuts; re-solve; save checkpoint. Final round = AR rule (lowest inner_selection task loss among rounds feasible on the final bank). Policies and contexts are frozen at round 0 and never refit.

## 2. Policy bank (M=5), frozen before any coefficient-role LP

All policies are deterministic maps from legal inputs to the SAME 17-token codebook, trained as cost-sensitive classifiers against the round-0 frozen decoder and round-0 attack bank, using labels only on nuisance_train rows. Features for the policy learner: [X_A, H_A, logit(p), r, risk(11)] standardized (fit on nuisance_train).

Cost-regression oracle: for per-person, per-token priced cost g_i(z) = u_i(z) - sum_j lambda_j a_ij(z) (u = task CE under frozen decoder, a_ij = CE loss of cut j's attacker at token z, prices lambda from the round-0 T32 LP duals via `fit_b.fixed_bank_dual_prices`), fit a regularized multi-output regressor g_hat(z|x) on nuisance_train (one sklearn HistGradientBoostingRegressor per token, max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=40, l2_regularization=1.0, early_stopping with 20% internal household-grouped validation, fixed seed; PWGTP-mean-1 sample weights). Policy d(x) = argmin_z g_hat(z|x), ties to the lowest token id. Tokens never used by D17 are allowed.

- d_0: D17 exact.
- d_1: task-only (lambda = 0).
- d_2: local-priced (lambda restricted to A-view cuts: A/SEX, A/RAC1P).
- d_3: coalition-priced (lambda restricted to AB-view cuts).
- d_4: all-cut priced with lambda doubled (stronger protection).

If the round-0 T32 LP has all-zero duals for a group, use the registered fallback lambda_j = 1/(number of cuts in that role group) for that group and record it.

Record for each policy on nuisance_train and coefficient_split (no outer): fraction of people whose token differs from D17, within-T32 disagreement distribution, affected unique households, token histogram. Remove exact aliases (identical token vectors on coefficient_split) and record them in the alias ledger.

## 3. Contexts

K in {1, 4}; K=4 is a hard 2x2 split at PWGTP-weighted medians (fitted on nuisance_train) of (i) the stored residual r and (ii) the frozen local SEX risk max-probability from `AR/nuisance.fit_frozen_nuisance` (nuisance_train). Registered fallback: if any K=4 context has < 300 unique coefficient households or household-weight ESS < 150 on any anchor, use K=2 (split on SEX risk only). Record the census before fitting.

## 4. Panel (per anchor 0,1,2)

| id | description |
|---|---|
| NM1_U, NM1_P | nested mixture K=1, utility / privacy-first LP |
| NM4_U, NM4_P | nested mixture K=4 (or K=2 fallback) |
| T32_U, T32_P | parent-restricted control: same pipeline with eta fixed to 0 (A=0) |
| RD_TASK | richer deterministic task-only policy: d_1, then 3 Lloyd rounds (refit decoder on the deterministic law with the same coverage rule; recompute task cost regression; re-derive policy); inner_selection picks the round |
| RD_PRIV | richer deterministic privacy-trained policy: Lagrangian cost-sensitive policy with a single scalar multiplier mu on the round-0 dual-price direction; mu chosen by bisection (<= 12 steps) as the smallest mu whose deterministic policy satisfies every current-bank cut on coefficient_split; then 3 rounds of attacker refit (audit_fit) + rebase + re-bisection; inner_selection picks the round (lowest task loss among bank-feasible rounds) |
| ADV_B1, ADV_B2 | PPAN-style adversarial categorical policy: MLP x->softmax(17) (legal features as in section 2), jointly trained receiver decoder MLP (H_A, onehot z) and four adversary MLPs (A/SEX, A/RAC1P on (H_A,z); AB/SEX, AB/RAC1P on (H_A,H_B,z)) with EXACT expectation over the 17 tokens (no sampling); loss = task CE - beta * mean adversary CE; alternating Adam (5 adversary steps per encoder step), trained on nuisance_train ∪ coefficient_split, adversaries on audit_fit; two betas {0.5, 2.0}; epoch checkpoint chosen on inner_selection by the same bank-feasibility rule. The released object is the per-person softmax law. |
| D17 | exact historical reference |
| Q_HIST | historical task-directed Q (32x17 stochastic), the pre-registered strongest 17-token external reference |
| J, H_ONLY | continuity references evaluated by the existing audit path |

Nomination-eligible family: NM (NM1_*, NM4_*). Independent comparators for primary routes: D17, RD_TASK, RD_PRIV, ADV (the inner-selected ADV_B*). T32_*, NM1 vs NM4, Q_HIST, J, H_ONLY are reported in a secondary family. A control that wins is reported as a baseline win.

## 5. Audit, selection, assessment

Reuse the AR independent audit slate unchanged (`AR/evaluate.audit_panel`, `AR/audit.py`), adding a per-person release spec kind that recomputes q(z|x) from pinned objects (B, A, eta, policies, context rule, ADV network) using legal inputs only. Positive controls: AR `positive_control.py` revealing channels for AB/SEX and AB/RAC1P on all three anchors, run before the lock. Inner selection, route margins, endpoint generation, Bonferroni families, paired household bootstrap (10,000 common resamples across overlapping anchors): reuse AR selection/inference code with the new comparator lists; the endpoint list is generated by code, counted and hashed before the lock.

## Amendment M1 (2026-09-24 ~18:35Z, pre-fit, from the independent math review `agents/math/MATH_REVIEW.md`; no outcome seen)

1. **D17-anchored (switched) policy columns.** A shared eta with a global mixture cannot choose *where* a richer policy is used (exact counterexample in MATH_REVIEW). Every non-D17 policy is therefore switched: `d_m(x) = argmin_z g_hat_m(z|x)` only if `g_hat_m(D17(x)|x) - min_z g_hat_m(z|x) > tau` with fixed `tau = 0.002` nats, otherwise `D17(x)`; ties go to the D17 token (not the lowest id). tau is fixed, not tuned.
2. **Cross-fitted policy costs.** The round-0 decoder is fitted on nuisance_train, so per-person task costs used to train policies on nuisance_train must come from 2-fold household-grouped cross-fitted decoders (same slate and selection rule, each fold's decoder scores the other fold). Attack costs on nuisance_train are already out-of-sample (attackers fit on audit_fit). The LP's coefficient_split costs use the full round-0 decoder as before.
3. **Exact same-context deterministic selector (new control DET_SEL4, and DET_SEL1).** With hard contexts, any deterministic release in the class with 0<eta<1 is a T32 function, so the exact deterministic comparator is an exhaustive choice of one policy per context: M^K = 5^4 = 625 (K=4) or 5 (K=1) assignments. Evaluate every assignment on coefficient_split under the NM final-round decoder and final bank of the same K, and pick the lowest 0.5U+0.5W task loss among bank-feasible assignments (tie: fewer non-D17 contexts, then lexicographic). It is audited like every other release. It is the randomization-attribution control for NM4/NM1; RD_PRIV remains the independently trained deterministic comparator.
4. **Interpretation rules registered.** "eta>0" is not evidence of a new release (eta is not identified; each T32-only column adds an alias direction). The nonalias gate is the weighted within-T32 total-variation spread of the fitted per-person law, the maximum, and affected unique households, plus the within-T32-averaged re-score (value of the variation). A fixed-bank LP privacy gain is not a privacy claim; only refit attackers (alternation bank and the independent audit) count. Utility route U requires X_A to carry task information beyond (T0,H_A); the RD_TASK inner task difference vs D17 is reported as that preflight.
5. **Persistent tokens.** One persistent keyed token per record; fresh draws are a different contract (Example A breaks under two draws).
