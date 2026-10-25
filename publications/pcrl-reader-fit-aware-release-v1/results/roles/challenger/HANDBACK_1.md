# Challenger hand-back 1: RFA1 reader-fit-aware release

Independent challenger, 2026-10-25. Label-free. Only invented data was used. No real data, labels, private arrays, model files,
`.private/` or `.worktrees/` paths were opened. No git state was changed.
- Files: `roles/challenger/fitpath_check.py` (check code) and `roles/challenger/fitpath_check.json` (its output).
- CPU: about 46 CPU-s (two runs measured with `/usr/bin/time`: 1.2 s and 43.5 s), one thread.
- I formed this view before reading `rfa/fam/rfit.py` and `fit.py`. They were read afterwards, and section 6 is checked against them.

## Verdict: PROCEED, with required changes (section 6)

- **Not equivalent to any closed recipe.** The operation is not mathematically equivalent to NEURAL_RAW's online critics, to SAR1's
  FB_PFD / FB_PFO, or to plain continuation. The fit path is real, and it is not swamped by episode noise over the planned path.
- **Its systematic part can only be one of two things:**
  - steering of the inner family's own fitting bias; or
  - a finite-sample learnability term tied to the inner sample size n_A.

  Neither term targets the population information that the fresh 16-recipe bank measures.
- **Equivalence in the limit.** If the inner reader is AUC-consistent and n_A is large, CAND, STOPFIT and an in-sample best response
  (the FB class) all have the same limiting gradient.
- **Prior.** CAND ties STOPFIT on fresh readers at matched cost. The registered comparison must be built to detect exactly the two
  terms above (section 4).

## 1. Equivalence audit

**Math.** Notation:
- theta are the release parameters.
- The inner fit is w*(theta) = argmin_w L_A(w; theta), with L_A = mean BCE of SEX + kappa/2 ||w||^2. Its Hessian is H.
- The outer objective is J(w; theta) = (AUC_E(w; theta) - 1/2)^2.

The total derivative splits into two paths:

    dJ/dtheta = d_theta J |_(w fixed)                    (direct path, through E outputs)
              - grad_w J' H^-1 d_theta grad_w L_A        (fit path, through A outputs; implicit-function theorem)

**Envelope.** If the outer objective is the optimal value of the inner criterion on the same rows, then grad_w J(w*) = 0. The fit
path is then identically 0, and autograd through the solve equals the stopped-fit gradient.

**(b) FB_PFD / FB_PFO: an in-sample stopped-fit gradient, exactly.**
- FB_PFD's statistic is D^2 = Delta'(S + eps I)^-1 Delta = max_w [2 w'Delta - w'(S + eps I) w].
- `sar/fb/pfd.py:174, 189` detaches eps and the poly2 moments.
- So FB_PFD's gradient is exactly a stopped-fit gradient of an in-sample best response. Check E1: full autograd versus the gradient
  at a frozen w* differ by a relative 2.0e-15. The investigator's selftest `fb_pfd_gradient_is_stopped_fit` agrees.
- FB_PFO uses the same functions, with income as a constant input.

**(a) NEURAL_RAW (`acs_replication/leafcorr.py` Critics, `neural_raw.py`).**
- Persistent, lagging critics: 80 warm rounds, then 5 Adam rounds per update.
- They are evaluated on a fresh privacy batch with critic weights frozen (`requires_grad_(False)`): direct path only.
- The risk is a one-sided CE risk with a min(CE, H) entropy clamp.
- This is the alternating-critic class. It has no fit path.

**(c) Plain continuation.** CAND at lambda = 0 is TASK. CAND is not continuation.

**The genuine difference: outer objective != inner objective, combined with the fit path.**
- In RFA the outer and inner objectives differ in three ways: held-out rows (E != A), a different loss (AUC versus BCE), and a ridge
  term inside the fit only.
- Held-out evaluation is not the only source of a fit path. An in-sample AUC of a BCE-fitted reader also has one: E1 gives a
  per-step share of 0.26.

**Decomposition of the expected fit path.** E[g_fit] has two terms:

- **(i) Population steering.** The term is grad_w J_pop(w_bar)' dw_bar/dtheta. It is non-zero whenever the BCE + ridge population
  reader is not AUC-optimal, through misspecified features or ridge. It makes the inner fitting procedure land on a worse direction
  rather than removing information.
- **(ii) Finite-sample learnability.** This is the derivative of the inner reader's expected estimation loss at n_A, of order p/n_A.
  It pushes toward outputs that are harder to learn.

How the three objectives compare against the population target grad max_w AUC_pop (to which they all converge):

| Objective | Extra term beyond grad max_w AUC_pop |
|---|---|
| In-sample best response (FB) | +grad of optimism. Pushes toward coarser or lower-dimensional outputs: EDG1's bias in gradient form. |
| STOPFIT (held-out) | About neutral, with O(p/n) curvature terms. |
| CAND | -grad of estimation loss (learnability), plus steering. |

**When the fit path vanishes:**
- **Exactly:** the envelope case. This covers FB_PFD and OptNet-ARL, and PCRL's in-batch ridge R² up to the 1e-4 ridge.
- **In expectation:** only when n_A is large relative to p and the inner reader is AUC-consistent (well-specified, weak ridge).
  - Measured, p = 5: 0.092 at n_A = 2,048 and 0.052 at n_A = 8,192.
  - With random features (misspecified), a floor of about 0.2 remains at n_A = 8,192.
- **Strong ridge does not remove it.** AUC is scale-invariant, so grad_w AUC is perpendicular to w and scales like 1/|w|, while
  dw*/dtheta scales like 1/kappa. The product tends to the mean-difference reader's steering term.
  - Measured, well-specified case: 0.28 at kappa = 1 versus 0.09 at kappa = 1e-4.
- **E and A from the same law** is necessary for the "learnability only" reading, because no A-E shift can be created. It does not
  make the term vanish by itself.
- **Per step, it never vanishes** at a fixed n_E. Noise from E in grad_w J is O(sqrt(p/n_E)) whatever n_A is.

**Numbers.**

Setup: 6 invented inputs; ReLU nets 6-16-K with K = 2 and 5; 5 pair logit differences. The reader is ridge logistic on
standardized u plus m random-ReLU features (p = 5, 21, 69). Sizes n_A = n_E = 2,048 except where noted. The surrogate is a smooth
AUC (tau 0.2) on std-normalized held-out scores. Each configuration uses 240 episodes, with unbiased cross-group mean norms and
jackknife SEs over 8 groups.

Check E2 compares the IFT term with a central finite difference of the re-solved objective. They agree to 0.5% (kappa 1e-2) and 0.13%
(kappa 1). Direct only would have been 0.00050 against 0.00192: one episode's fit path can dominate.

Column key:
- **step fit/dir:** per-step median of |g_fit| / |g_dir|.
- **step cos:** per-step mean cosine between g_fit and g_dir.
- **mean ratio:** |E g_fit| / |E g_dir|, with SE in parentheses.
- **orth:** the part of E g_fit orthogonal to E g_dir, as a share of |E g_dir|.
- **cos(E tot, E dir):** cosine between the expected total and expected direct updates.
- **T2s:** steps until the systematic fit-path drift reaches 2 sigma of the full-trace per-step noise.

| law | n_A | p | kappa | AUC_E | step fit/dir | step cos | mean ratio (SE) | orth (SE) | cos(E tot, E dir) | noise dir -> tot | T2s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nonlin | 2048 | 5 | 1e-4 | .550 | .56 | -.11 | .165 (.017) | .097 (.026) | .996 | .0266 -> .0286 | 76 |
| nonlin | 2048 | 5 | 1e-2 | .550 | .62 | -.05 | .225 (.016) | .113 (.026) | .996 | .0248 -> .0277 | 50 |
| nonlin | 2048 | 5 | 1 | .541 | .23 | -.19 | .173 (.012) | .162 (.008) | .985 | .0101 -> .0111 | 41 |
| nonlin | 2048 | 21 | 1e-4 | .563 | .76 | -.17 | .353 (.012) | .294 (.015) | .971 | .0438 -> .0466 | 43 |
| nonlin | 2048 | 21 | 1e-2 | .561 | .72 | -.14 | .287 (.012) | .212 (.017) | .985 | .0317 -> .0355 | 42 |
| nonlin | 2048 | 21 | 1 | .542 | .31 | -.17 | .202 (.013) | .198 (.012) | .979 | .0118 -> .0137 | 41 |
| nonlin | 2048 | 69 | 1e-4 | .575 | .84 | -.28 | .443 (.017) | .414 (.019) | .941 | .0664 -> .0704 | 57 |
| nonlin | 2048 | 69 | 1e-2 | .567 | .73 | -.20 | .268 (.016) | .182 (.019) | .989 | .0392 -> .0396 | 49 |
| nonlin | 2048 | 69 | 1 | .542 | .47 | -.17 | .225 (.009) | .224 (.011) | .977 | .0138 -> .0182 | 65 |
| nonlin | 512 | 69 | 1e-2 | .550 | .81 | -.24 | .573 (.046) | .316 (.024) | .978 | .0394 -> .0374 | 24 |
| nonlin | 8192 | 69 | 1e-2 | .578 | .69 | -.17 | .228 (.015) | .221 (.014) | .979 | .0336 -> .0430 | 52 |
| well-specified | 2048 | 5 | 1e-4 | .618 | .54 | -.44 | .092 (.010) | .088 (.013) | .996 | .0219 -> .0104 | 55 |
| well-specified | 2048 | 5 | 1 | .591 | .29 | -.25 | .282 (.006) | .273 (.005) | .959 | .0075 -> .0076 | 6 |
| well-specified | 8192 | 5 | 1e-4 | .620 | .47 | -.38 | .052 (.008) | .051 (.009) | .999 | .0186 -> .0110 | 167 |

"Well-specified" means linear nets, a Gaussian mean shift and linear features.

**Reading the table:**
- **Systematic share.** The fit path's systematic share is 5-57% of the direct mean, and its orthogonal share is 5-41%.
- **Direction.** The expected total update stays within 3-20 degrees of the stopped-fit direction (cos 0.94-0.999).
- **At the investigator's settings** (kappa 1e-2; p about 6 and 70) the measured values are:
  - mean share 0.23-0.27;
  - orthogonal share 0.11-0.18;
  - a 5-9 degree turn.
- **Not swamped by noise.** Measured against the conservative full-trace noise, the drift needs about 6-170 steps (typically about
  50), against the 400-step path. It is near the noise only for the large-n_A, well-specified case.
- **Per-step behaviour.** Per step, the fit path is large (0.2-0.8 of the direct norm) but mostly noise, and it is anti-correlated
  with the direct path. It cancels the "chase the current reader" moves that a refit would undo. In the well-specified case it halves
  the per-step noise.
- **Consequence for logging.** A per-step `fit_share` therefore overstates the fit path's systematic role (see change C2).

**Closest closed relative.** STOPFIT is the held-out analogue of FB_PFO_R1. They share:
- START, the occupation-only loop, the frozen income and the optimizer;
- the 10-step path, the budgets and the bracket rule.

They differ in the reader family, the surrogate, and held-out versus in-sample evaluation. SAR1 closed FB as not separated from
NEURAL_RAW's own recipe continued.

## 2. Repository search: nothing differentiates through a reader fit

- **No higher-order differentiation anywhere outside `results/`.** There is no `create_graph`, `torch.func` or functorch, hvp, jvp,
  vjp, unrolling, or implicit/IFT code. The only Hessian is a numpy Fisher = Hessian check at `nrm/fb/channel.py:332`.
- **Differentiable solves inside training are all in-sample best responses** (envelope, so stopped fit):
  - `sar/fb/pfd.py:180-195`, reused by `sar/fb/pfo.py`.
  - `pcrl/training/losses.py:476-560` (VerificationRegularizer, in-batch ridge R²; envelope up to the 1e-4 ridge).
  - `scripts/run_pcrl_bios_layer12.py:189`.
- **Critics with frozen weights during the release gradient** (direct path only):
  - `acs_replication/leafcorr.py` Critics, with `neural_raw.py`.
  - `rgj/train.py` (refreshed critics: bounded refit plus CRITIC_VAL selection).
  - `smf/train.py:175, 391`.
  - `osf/train.py:187, 362, 433`.
  - `jcv/train.py`.
- **Held-out objectives with no gradient path:**
  - `pcrl/language/dual_controllers.py` (HeldoutR2Controller): a held-out R² for the dual variable only.
  - `drf/solver.py` and `drf/finite_law.py`: exact in-sample AUC* best response on finite outputs, solved by discrete search.
- **Numpy solves off any release gradient:** `nrm`, `ihs`, `ipr`, `jcv/project.py` and the guard modules.

`rfa/fam/rfit.py` is the first code in the repository with a fit path.

## 3. Prior work

The first two papers' method sections were read on arXiv/ar5iv, and the NTGA method from the PMLR PDF.

- **OptNet-ARL** (Sadeghi, Wang, Boddeti, arXiv:2109.05535). Read from the method sections.
  - Both the adversary and the target are kernel ridge regressors solved in closed form on one minibatch.
  - The encoder minimizes the projection norms (1 - lambda)||P_Ms [S';0]||^2 - lambda||P_My [Y';0]||^2. These are the adversary's
    minimum regularized MSE on the same rows.
  - The gradient flows through the projection (Golub-Pereyra).
  - Held-out adversaries appear only at final evaluation.
  - **Relation:** an in-sample best response, so the envelope applies and it is a stopped fit. It is the published analogue of FB,
    not of the fit path.
- **RecUP-FL** (Cui et al., arXiv:2304.05135). Read from the method sections.
  - FGSM-style sign perturbations of shared FL gradients against a zoo of 20 fixed pre-trained substitute attackers.
  - Q = 5 attackers are sampled per iteration: "meta-train" uses Q-1 of them and "meta-test" takes one step on the last.
  - Attacker weights are never updated, so only the direct path exists. "Meta-learning" is an analogy.
  - It is evaluated against one unseen NN attacker. There is no held-out selection.
  - **Relation:** a fixed-reader ensemble heuristic.
- **Methods that do differentiate a held-out loss through a learner's fit:**
  - **NTGA** (Yuan and Wu, ICML 2021, PMLR 139:12230). Method read from the PDF.
    - A closed-form NTK-GP mean is fitted on the perturbed training set and its loss evaluated on a separate validation set.
    - Sign-gradient ascent runs through (K^nn)^-1 and e^{-eta K t}.
    - **Key difference:** NTGA perturbs the training data only, so its lever is a train/validation shift. In RFA one network
      produces both A and E, so no shift can be created, and only steering and learnability remain.
  - **R2-D2 / LR-D2** (Bertinetto et al., ICLR 2019) and **MetaOptNet** (Lee et al., CVPR 2019). From prior knowledge; not fetched.
    - A ridge, logistic (Newton) or SVM head is fitted on the support set, with the loss on the query set.
    - Gradients pass through the solve (closed form or KKT/IFT).
    - **Relation:** RFA's FULL with the sign flipped.
  - **IFT hypergradients** (Bengio 2000; Pedregosa 2016; Lorraine et al. 2020; iMAML 2019): exactly RFA's fit path, with the release
    as the hyperparameter.
  - **Unrolled GANs** (Metz et al. 2017): the extra term vanishes as the discriminator converges on the same objective, which is the
    envelope.
  - **Learnability attacks:** Unlearnable Examples (Huang et al. 2021), MetaPoison (2020), back-gradient poisoning
    (Munoz-Gonzalez et al. 2017). Their protection is known to be tied to the surrogate learner and its regime (Fu et al. 2022).
  - **Elazar and Goldberg (2018):** fresh probes recover attributes that adversarial training had "removed".

None of these establishes novelty or suitability for this contract.

## 4. Mechanism-specific falsifiers to register before the first fit

- **F0, mechanism gate at START** (MODEL_FIT only; before any arm is fitted).
  - Measure over at least 64 START episodes, using unbiased cross-group estimates:
    - the systematic fit-path share |E g_fit| / |E g_dir|;
    - the orthogonal share;
    - cos(E g_tot, E g_dir).
  - If the orthogonal share is below 0.05 and the cosine is above 0.995, the result is MECHANISM_INERT. FULL then equals STOPFIT up
    to strength, and no fit-path claim is possible.
- **F1, logged mechanism.** Record the same three statistics per 40-step epoch along FULL. If the orthogonal share drops below 0.05
  before S_k*, the fit path is inert at the released point.
- **F2, primary test.** Compare FULL with STOPFIT at matched MODEL_FIT task spend (bracketed), never at matched step or lambda.
  - Readers: fresh-bank pair AUC (maximum over 16 recipes, VALIDATION-oriented, FIT fraction 1.0).
  - The fit path is credited only if FULL is below STOPFIT by the material margin, and the difference is resolved.
  - A tie means the fit path changed nothing that fresh readers see.
- **F3, inner-family overfitting (steering).** At matched spend, the FULL-minus-STOPFIT gain on the inner families (SIMPLE and RFF,
  refitted on fresh held-out rows at n_A) is at least twice the gain on the fresh bank. Also triggered if HGB15, ET25 or LOOKUP gain
  less than half of what LOG_LR gains (SAR1's R1 "best-response evasion" trigger).
- **F4, learnability only.** The FULL-minus-STOPFIT benefit shrinks monotonically over FIT fractions 0.25, 0.5 and 1.0
  (about 1,000 to 4,000 rows), toward 0, and is about 0 for readers trained on the full 20,000 MODEL_FIT rows. Optionally also refit
  the inner reader at 4 x n_A.
- **F5, inversion.** Any of the following is inversion, not protection:
  - the inner held-out E orientation is negative on a majority of steps;
  - the fresh unoriented AUC falls below 0.5;
  - the oriented AUC stays unchanged.
- **F6, surrogate hacking.** Any of the following counts:
  - AUC_b falls but the exact Mann-Whitney AUC of the same reader does not;
  - the signal moves into higher moments: a held-out degree-2 (FB Fisher) monitor or HGB rises while AUC_b falls;
  - a coordinate's sd_A shrinks more than 10x from START.

## 5. Risks

- **Weak inner reader.** SIMPLE and RFF-64 at n_A = 2,048 are weaker than the bank: 16 recipes at about 4,000 FIT rows, with
  VALIDATION selection, including HGB, ET, MLP, LOOKUP and decision/CATI readers. The fit path protects only against what this
  family learns at n_A.
- **Signal pushed out of SIMPLE's view.** SIMPLE together with the moment-based AUC_b is blind to signal in variances or
  interactions. A release can push SEX there; my "nonlin" law is exactly that case. RFF only partly guards against it.
- **Decision cells not differentiated.** Decision-cell leakage is never differentiated, because the argmax is detached.
- **Cost and strength mismatch.** At a common lambda, the arms run at different effective speeds:
  - The parallel component of E g_fit rescales the effective strength by about 0.9-1.2 times.
  - Adam's noise normalization differs between the arms: in my check FULL's per-step noise was halved (well-specified) or 0-30%
    higher (nonlin).

  Only a comparison at matched task spend is valid.
- **Calibration bias.** Calibrating lambda per arm on the median per-step norm would bias against FULL, because its per-step norm
  includes the noisy fit path.
- **Alias.** With a converged inner solve on the same A, an alternating critic re-solved from any warm start is an exact alias of
  STOPFIT, because the ridge-logistic loss is strictly convex. ALTCRIT with 'newton' is that alias, as `rfit.py` already notes.
  ALTCRIT with 10 full-batch Adam steps from a warm start may be nearly converged, which makes it a near-alias.
- **Inherited cap.** The occupation-only, frozen-income design inherits FB_PFO's cap: the frozen income can carry the pair's signal.

## 6. Required changes

Section 1 rules out equivalence; these changes are needed so the comparison can detect the two terms. Checked against `rfa/fam/rfit.py`
and `fit.py`.

Already satisfied:
- a common lambda;
- the binormal AUC_b, invariant to score scale and shift;
- logging of the exact AUC;
- Newton and condition-number guards;
- refusal is never a reward;
- an explicit ALTCRIT alias note;
- finite-difference checks.

| # | Change |
|---|---|
| C1 | Add the F0 gate and register F1-F6 in the card. |
| C2 | Log the systematic shares per epoch: running group sums of g_dir and g_fit in parameter space, 8 groups. Do not rely only on the per-step output-space `fit_share`. |
| C3 | Log ALTCRIT's lag per step: max \|grad L_A(w_alt)\| and \|\|w_alt - w*_A\|\| / \|\|w*_A\|\|. If the median relative lag is below 1%, declare a NEAR_ALIAS and do not count ALTCRIT as an independent control. |
| C4 | Declare that the A-moment path is stopped. mu_A and sd_A are detached, so FULL's fit path is partial (w* only). The stopped-moment convention is shared with STOPFIT; record it in the card. |
| C5 | Add guard readouts at matched spend: per recipe (HGB15, ET25, LOOKUP, DEC_LR, CATI), a held-out degree-2 Fisher monitor, and a per-coordinate sd_A ratio to START. |
| C6 | All FULL-versus-STOPFIT, FULL-versus-ALT and FULL-versus-ALTCRIT claims at matched spend only, with the bracket rule unchanged. Optional: an in-sample STOPFIT (E = A) to separate the held-out effect from the family change; otherwise cite FB_PFO_R1 as the closed in-sample relative. |
| C7 | Choose kappa and the families label-free (or on START MODEL_FIT only), never on outcomes. Note that kappa = 1e-2 and the RFF width set the steering share. |

**Challenger priors (unregistered):**

| Outcome | P |
|---|---|
| F0 passes (not inert) | 0.80 |
| FULL vs STOPFIT at matched spend: tie | 0.75 |
| FULL vs STOPFIT: FULL resolved better | 0.10 |
| FULL vs STOPFIT: FULL worse | 0.15 |
| F3 or F4 triggers | 0.50 |
| FULL resolved better than ALT (RAW-J continued) | 0.10 |
