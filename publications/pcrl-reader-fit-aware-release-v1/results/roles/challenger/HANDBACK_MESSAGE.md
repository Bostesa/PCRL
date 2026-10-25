# Independent challenger, final message (verbatim; delivered to the coordinator at about 21:27Z)

The challenger also wrote `HANDBACK_1.md` in this folder itself (288 lines, 17:25 local file time); the delivered message follows
verbatim (harness indentation removed).

---

# Challenger hand-back 1: RFA1 reader-fit-aware release (`pcrl_reader_fit_aware_release_v1`)

**Verdict: PROCEED, with the changes in section 6.**
- **Not equivalent to any closed recipe.** The operation is not mathematically equivalent to NEURAL_RAW's online critics, to SAR1's FB_PFD / FB_PFO, or to plain continuation. The fit path is real and is not swamped by episode noise over a 400-step path.
- **What the fit path can add is limited to two things:**
  - steering, which exploits the inner reader family's own fitting bias; or
  - a learnability term, which makes the inner reader harder to estimate at n_A.

  Neither targets the population information that the fresh 16-recipe bank measures.
- **Equivalence in the limit.** If the inner reader is AUC-consistent and n_A is large, the candidate (FULL), STOPFIT and an in-sample best response (the FB class) all have the same limiting gradient.
- **Prior.** FULL ties STOPFIT on fresh readers at matched cost.

**Housekeeping**
- **Files:** `results/pcrl_reader_fit_aware_release_v1/roles/challenger/` holds `HANDBACK_1.md` (this report), `fitpath_check.py` (the check) and `fitpath_check.json` (its output).
- **CPU:** about 46 CPU-s, one thread (`/usr/bin/time`: 1.2 s and 43.5 s for the two runs).
- **Rules kept:** invented data only. No real data, labels, private arrays, model files, `.private/` or `.worktrees/` paths. No git state changed.
- **Order of work:** I formed this view before reading `rfa/fam/rfit.py` and `fit.py`. I read them afterwards; section 6 is checked against them.

## 1. Equivalence audit

**Math.** With w* = argmin_w L_A(w; θ), L_A = mean BCE + κ/2‖w‖², Hessian H, and outer objective J = (AUC_E(w; θ) − 1/2)²:

dJ/dθ = ∂θJ|_(w fixed) [direct path, through E] − ∇wJᵀ H⁻¹ ∂θ∇wL_A [fit path, through A]

**Envelope theorem.** If the outer objective is the optimal value of the inner criterion on the same rows, then ∇wJ(w*) = 0. The fit path is then exactly zero, and autograd through the solve equals the stopped-fit gradient.

**(b) FB_PFD / FB_PFO: yes, their gradient is a stopped-fit gradient.**
- FB_PFD's statistic is D² = Δᵀ(S+εI)⁻¹Δ = max_w [2wᵀΔ − wᵀ(S+εI)w].
- The code detaches ε and the poly2 moments (`sar/fb/pfd.py:174, 189`).
- So FB's gradient is exactly the gradient of an in-sample best response with the fit stopped.
- Check E1: full autograd and the frozen-w* gradient differ by a relative 2.0e-15. The investigator's selftest `fb_pfd_gradient_is_stopped_fit` agrees.
- FB_PFO uses the same functions.

**(a) NEURAL_RAW.**
- Persistent, lagging critics (80 warm rounds, then 5 Adam rounds per update), evaluated on a fresh batch.
- Critic weights are frozen (`requires_grad_(False)`), so the gradient is direct path only.
- The risk is one-sided cross-entropy with a min(CE, H) clamp. This is the alternating-critic class.

**(c) Plain continuation.** FULL at λ = 0 is TASK.

**Is the held-out evaluation plus the fit path the only genuine difference?** Yes, with one qualification. A fit path exists whenever the outer objective differs from the inner one: held-out rows, AUC versus BCE, or a ridge used only inside the fit. RFA has all three. Even an in-sample AUC of a BCE-fitted reader has a fit path (E1: per-step share 0.26).

**What the expected fit path contains:**
1. **Population steering:** ∇wJ_pop(w̄)ᵀ dw̄/dθ. It is non-zero whenever the BCE + ridge population reader is not AUC-optimal, through misspecified features or ridge. It makes the fitting procedure land on a worse direction; it does not remove information.
2. **Learnability:** the derivative of the inner reader's estimation loss at n_A, of order p/n_A.

**How the three objectives compare.** All three converge to ∇ max_w AUC_pop. Beyond that:

| Objective | Extra term |
|---|---|
| In-sample best response (FB) | +∇optimism, which pushes toward coarser outputs (EDG1's bias in gradient form) |
| STOPFIT | Roughly neutral |
| FULL | −∇(estimation loss) plus steering |

**When does the fit path vanish?**
- **Exactly:** in the envelope case (FB, OptNet-ARL; PCRL's in-batch ridge R² up to its 1e-4 ridge).
- **In expectation:** only when n_A is large relative to p and the inner reader is AUC-consistent. Measured at p = 5: 0.092 at n_A = 2048, falling to 0.052 at 8192. With random features (misspecified) a floor of about 0.2 remains at 8192.
- **Strong ridge does not remove it.** AUC is scale-invariant, so ∇wAUC ⊥ w and scales as 1/|w|, while dw*/dθ scales as 1/κ. The product tends to the mean-difference reader's steering term. In the well-specified case it was 0.28 at κ = 1 versus 0.09 at κ = 1e-4.
- **Same law for E and A** is needed for the "learnability only" reading, because no A–E shift can then be created. It does not make the term vanish.
- **Per step it never vanishes** at fixed n_E. The E noise in ∇wJ is about √(p/n_E), whatever n_A is.

### Numbers from the synthetic check

**Setup:**
- Data: 6 inputs, two ReLU networks 6-16-K (K = 2 and 5), 5 pair logit differences.
- Reader: ridge logistic regression on standardized u plus m random-ReLU features, giving p = 5, 21 or 69.
- Sizes: n_A = n_E = 2048.
- Surrogate: smooth AUC on std-normalized held-out scores.
- Repetition: 240 episodes per configuration, unbiased cross-group mean norms, jackknife SE over 8 groups.

**IFT check (E2).** The implicit-function term matches a central finite difference of the re-solved objective to 0.5% (κ = 1e-2) and 0.13% (κ = 1). In one episode the direct path alone gave 0.00050 against a total of 0.00192.

**Columns:**
- **per-step fit/dir:** per-step median of |g_fit| / |g_dir|.
- **per-step cos:** per-step cosine between g_fit and g_dir.
- **mean ratio:** |E g_fit| / |E g_dir|, SE in parentheses.
- **orthogonal:** the part of E g_fit orthogonal to E g_dir, as a share of |E g_dir|.
- **cos(E tot, E dir):** cosine between the expected total update and the expected direct update.
- **T2σ:** steps until the systematic fit-path drift exceeds twice the full-trace per-step noise.

| law | n_A | p | κ | AUC_E | per-step fit/dir | per-step cos | mean ratio | orthogonal | cos(E tot, E dir) | noise dir→tot | T2σ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nonlin | 2048 | 5 | 1e-4 | .550 | .56 | −.11 | .165 (.017) | .097 (.026) | .996 | .0266→.0286 | 76 |
| nonlin | 2048 | 5 | 1e-2 | .550 | .62 | −.05 | .225 (.016) | .113 (.026) | .996 | .0248→.0277 | 50 |
| nonlin | 2048 | 5 | 1 | .541 | .23 | −.19 | .173 (.012) | .162 (.008) | .985 | .0101→.0111 | 41 |
| nonlin | 2048 | 21 | 1e-4 | .563 | .76 | −.17 | .353 (.012) | .294 (.015) | .971 | .0438→.0466 | 43 |
| nonlin | 2048 | 21 | 1e-2 | .561 | .72 | −.14 | .287 (.012) | .212 (.017) | .985 | .0317→.0355 | 42 |
| nonlin | 2048 | 21 | 1 | .542 | .31 | −.17 | .202 (.013) | .198 (.012) | .979 | .0118→.0137 | 41 |
| nonlin | 2048 | 69 | 1e-4 | .575 | .84 | −.28 | .443 (.017) | .414 (.019) | .941 | .0664→.0704 | 57 |
| nonlin | 2048 | 69 | 1e-2 | .567 | .73 | −.20 | .268 (.016) | .182 (.019) | .989 | .0392→.0396 | 49 |
| nonlin | 2048 | 69 | 1 | .542 | .47 | −.17 | .225 (.009) | .224 (.011) | .977 | .0138→.0182 | 65 |
| nonlin | 512 | 69 | 1e-2 | .550 | .81 | −.24 | .573 (.046) | .316 (.024) | .978 | .0394→.0374 | 24 |
| nonlin | 8192 | 69 | 1e-2 | .578 | .69 | −.17 | .228 (.015) | .221 (.014) | .979 | .0336→.0430 | 52 |
| well-specified (linear) | 2048 | 5 | 1e-4 | .618 | .54 | −.44 | .092 (.010) | .088 (.013) | .996 | .0219→.0104 | 55 |
| well-specified (linear) | 2048 | 5 | 1 | .591 | .29 | −.25 | .282 (.006) | .273 (.005) | .959 | .0075→.0076 | 6 |
| well-specified (linear) | 8192 | 5 | 1e-4 | .620 | .47 | −.38 | .052 (.008) | .051 (.009) | .999 | .0186→.0110 | 167 |

**What the table shows:**
- **Systematic share.** The systematic fit path is 5–57% of the mean direct gradient. Its orthogonal part is 5–41%.
- **Direction.** The expected total update stays within 3–20° of the stopped-fit direction.
- **At the investigator's settings** (κ = 1e-2; p ≈ 6 and 70) the measured values are:
  - systematic share 0.23–0.27;
  - orthogonal share 0.11–0.18;
  - a 5–9° turn.
- **Not swamped by noise.** The systematic drift clears the noise after about 6–170 steps (typically about 50), well within 400. It is near the noise only at large n_A in the well-specified case.
- **Per-step behaviour.** Per step, the fit path is large (0.2–0.8 of the direct norm) but mostly noise. It is anti-correlated with the direct path: it cancels "chase the current reader" moves that a refit would undo. In the well-specified case this halves the per-step noise.
- **Consequence for logging.** A per-step `fit_share` therefore overstates how much the fit path matters systematically.

**Closest closed relative.** STOPFIT is the held-out analogue of FB_PFO_R1. It shares FB_PFO's:
- START, occupation-only loop and frozen income;
- optimizer and 10-step path;
- budgets and bracket rule.

They differ only in the reader family, the surrogate, and held-out versus in-sample evaluation. SAR1 found FB not separated from NEURAL_RAW's own recipe continued.

## 2. Repository search: nothing differentiates through a reader fit

- **No higher-order differentiation anywhere outside `results/`.** There is no `create_graph`, `torch.func` or functorch, hvp, jvp, vjp, unrolling, or implicit/IFT code. The only Hessian is a numpy Fisher = Hessian check at `nrm/fb/channel.py:332`.
- **Differentiable solves inside training are all in-sample best responses**, so the envelope applies and they are stopped fits:
  - `sar/fb/pfd.py:180-195`, reused by `sar/fb/pfo.py`;
  - `pcrl/training/losses.py:476-560` (VerificationRegularizer, in-batch ridge R²);
  - `scripts/run_pcrl_bios_layer12.py:189`.
- **Critics with frozen weights during the release gradient** (direct path only):
  - `acs_replication/leafcorr.py` with `neural_raw.py`;
  - `rgj/train.py` (refreshed critics: bounded refit plus CRITIC_VAL selection);
  - `smf/train.py:175, 391`;
  - `osf/train.py:187, 362, 433`;
  - `jcv/train.py`.
- **Held-out or best-response quantities with no gradient path:**
  - `pcrl/language/dual_controllers.py` HeldoutR2Controller: a held-out R² used for the dual variable only.
  - `drf/solver.py` and `drf/finite_law.py`: AUC* is an in-sample best response on finite outputs, optimized by discrete search.
- **Numpy solves that never touch a release gradient:** in `nrm`, `ihs`, `ipr`, `jcv/project.py` and the guard modules.

`rfa/fam/rfit.py` is the first code in the repository with a fit path.

## 3. Prior work

- **OptNet-ARL** (arXiv:2109.05535). Method read via ar5iv.
  - Kernel-ridge adversary and target, both solved in closed form on one minibatch.
  - The encoder minimizes (1−λ)‖P_Ms[S̃ᵀ;0]‖² − λ‖P_My[Ỹᵀ;0]‖²: the adversary's minimum regularized MSE on the same rows.
  - The gradient passes through the projection (Golub–Pereyra).
  - Held-out adversaries are used only at final evaluation.
  - **Relation:** an in-sample best response, so the envelope applies and it is a stopped fit. It is the published analogue of FB, not of the fit path.
- **RecUP-FL** (arXiv:2304.05135). Method read via ar5iv.
  - FGSM-style sign perturbations of the shared FL gradients against a zoo of 20 fixed pre-trained attackers.
  - Q = 5 attackers are sampled per iteration: "meta-train" on Q−1, "meta-test" on the last.
  - Attacker weights are never updated, so only the direct path is used. "Meta-learning" is an analogy.
  - There is no held-out selection.
  - **Relation:** a heuristic over an ensemble of fixed readers.
- **Closest methods that do differentiate a held-out loss through a learner's fit:**
  - **NTGA** (Yuan and Wu, ICML 2021; method read from the PMLR PDF).
    - A closed-form NTK-GP mean is fitted on the perturbed training set; the loss is evaluated on a separate validation set.
    - Gradient ascent runs through (Kⁿⁿ)⁻¹ and e^(−ηKt).
    - **Key difference:** NTGA perturbs only the training data, so it can exploit a train/validation shift. In RFA the same network produces A and E, so no shift exists and only steering and learnability remain.
  - **R2-D2 / LR-D2** (Bertinetto et al., ICLR 2019) and **MetaOptNet** (Lee et al., CVPR 2019). From prior knowledge; I did not fetch these.
    - A ridge, logistic (Newton) or SVM head is fitted on the support set and scored on the query set.
    - Gradients pass through the solve.
    - **Relation:** RFA's FULL with the sign flipped.
  - **IFT hypergradients** (Bengio 2000; Pedregosa 2016; Lorraine et al. 2020; iMAML): the same as RFA's fit path, with the release playing the hyperparameter.
  - **Unrolled GANs** (Metz et al. 2017): the extra term vanishes as the discriminator converges on the same objective (the envelope).
  - **Learnability attacks** (Unlearnable Examples 2021, MetaPoison 2020, back-gradient poisoning 2017): their protection is known to be tied to the surrogate learner (Fu et al. 2022).
  - **Elazar and Goldberg (2018):** fresh probes recover attributes that adversarial training had supposedly removed.

None of this establishes novelty or suitability for this contract.

## 4. Falsifiers to register before the first fit

- **F0, mechanism gate at START** (MODEL_FIT only, before any arm is fitted).
  - Over at least 64 episodes, using unbiased cross-group estimates, measure:
    - the systematic share;
    - the orthogonal share;
    - cos(E g_tot, E g_dir).
  - If the orthogonal share is below 0.05 and that cosine is above 0.995, the mechanism is inert (MECHANISM_INERT). FULL then equals STOPFIT up to strength, and no fit-path claim is possible.
- **F1, logged mechanism.** Log the same statistics every 40-step epoch along FULL. If the orthogonal share falls below 0.05 before the selected checkpoint S_k*, the fit path is inert at the released point.
- **F2, primary test.** Compare FULL and STOPFIT at matched MODEL_FIT task spend (bracketed), never at matched step or matched λ.
  - Readers: fresh-bank pair AUC (maximum over the 16 recipes, oriented on VALIDATION, FIT fraction 1.0).
  - Credit the fit path only if FULL is lower than STOPFIT by the material margin and the difference is resolved.
- **F3, inner-family overfitting.** The fit path is fooling the inner family rather than removing signal if either holds:
  - At matched spend, FULL's gain over STOPFIT on the inner families (refitted on fresh held-out rows at n_A) is at least twice its gain on the fresh bank.
  - HGB15, ET25 or LOOKUP fall by less than half of what LOG_LR falls (SAR1's R1 trigger).
- **F4, learnability only.** The benefit shrinks monotonically toward zero as the FIT fraction rises from 0.25 through 0.5 to 1.0 (about 1,000 to 4,000 rows), and is about zero for readers trained on all 20,000 MODEL_FIT rows. Optionally, also refit the inner reader at 4×n_A.
- **F5, inversion.** Any of the following signals inversion rather than protection:
  - the held-out E orientation is negative on most steps;
  - the unoriented fresh AUC falls below 0.5;
  - the oriented AUC does not change.
- **F6, surrogate hacking.** Any of the following counts:
  - The binormal AUC falls but the exact Mann-Whitney AUC of the same reader does not.
  - A held-out degree-2 Fisher monitor or HGB rises while the binormal AUC falls, meaning the signal moved into higher moments.
  - Any coordinate's sd_A shrinks more than 10× from START.

## 5. Risks

- **The inner reader is weaker than the evaluation bank.** SIMPLE and RFF-64 at n_A = 2048 compare with 16 recipes fitted on about 4,000 rows with VALIDATION selection (HGB, ET, MLP, LOOKUP, decision and categorical-interaction readers).
- **The signal can move to where SIMPLE cannot see it.** SIMPLE plus a moment-based AUC is blind to variance or interaction signal, so a release can push SEX there. My "nonlin" law is exactly this case; RFF guards only partly.
- **Decision cells are never differentiated**, because the argmax is detached.
- **The arms run at different effective speeds at a common λ.**
  - The parallel part of the fit path rescales strength by about 0.9–1.2×.
  - Adam's noise normalization also differs between arms: FULL's per-step noise was halved (well-specified) or 0–30% higher (nonlin).

  Only matched-spend comparisons are valid.
- **Per-arm calibration would bias against FULL.** Calibrating λ separately per arm on the median per-step norm penalizes FULL, because its per-step norm includes the noisy fit path.
- **Alias.** With a converged inner solve on the same A, an alternating critic re-solved from any warm start is an exact alias of STOPFIT, because the objective is strictly convex. ALTCRIT with 'newton' is that alias, as `rfit.py` already notes. ALTCRIT with 10 warm-started full-batch Adam steps may be close to converged, i.e. a near-alias.
- **Inherited cap.** The occupation-only design with frozen income inherits FB_PFO's limit: the frozen income can carry the pair's signal.

## 6. Required changes

Checked against `rfa/fam/rfit.py` and `rfa/fam/fit.py`. Already in place:
- a common λ;
- the scale- and shift-invariant binormal AUC;
- exact AUC logged;
- Newton and condition-number guards;
- refusal never rewarded;
- the ALTCRIT alias note;
- finite-difference checks.

Still needed:
- **C1:** Add the F0 gate and register F1–F6 in the card.
- **C2:** Log the systematic shares per epoch from running sums of g_dir and g_fit in parameter space, over 8 groups. The per-step output-space `fit_share` alone is not enough.
- **C3:** Log ALTCRIT's lag per step: max|∇L_A(w_alt)| and ‖w_alt − w*_A‖/‖w*_A‖. If the median relative lag is below 1%, declare it a NEAR_ALIAS and do not count it as an independent control.
- **C4:** State in the card that the A-moment path is stopped. Because mu_A and sd_A are detached, FULL's fit path runs through w* only, and STOPFIT shares this convention.
- **C5:** Add guard readouts at matched spend:
  - per recipe (HGB15, ET25, LOOKUP, DEC_LR, CATI);
  - a held-out degree-2 Fisher monitor;
  - the per-coordinate sd_A ratio to START.
- **C6:** Make every FULL claim against STOPFIT, ALT or ALTCRIT at matched spend, with the bracket rule unchanged. Optionally add an in-sample STOPFIT (E = A) to separate the held-out effect from the family change; otherwise cite FB_PFO_R1 as the closed in-sample relative.
- **C7:** Choose κ and the reader families label-free, or on START MODEL_FIT only, never on outcomes. Note that κ = 1e-2 and the RFF width set the steering share.

## Challenger priors (not registered)

| Outcome | P |
|---|---|
| F0 passes (not inert) | 0.80 |
| FULL vs STOPFIT at matched spend: tie | 0.75 |
| FULL vs STOPFIT: FULL better, resolved | 0.10 |
| FULL vs STOPFIT: FULL worse | 0.15 |
| F3 or F4 triggers | 0.50 |
| FULL resolved better than ALT (RAW-J continued) | 0.10 |

**CPU used:** about 46 CPU-seconds in total.
