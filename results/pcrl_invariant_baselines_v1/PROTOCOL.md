# PROTOCOL — rotation-invariant objective and fair external baselines

Prospective. Written and hashed into `PROTOCOL_FREEZE.json` **before** any new
interface was fitted and before any new ACS performance outcome was opened.

**This is development, not final confirmation.** Directional forecasts are registered
below because registration is what makes an outcome interpretable, but *their*
success or failure is **not** the validity criterion for this study. A protocol
remains useful without a dramatic called shot.

---

## 1. The three questions

1. **Does an objective that cannot improve merely by rotating the same channel change
   actual disclosure recovery?** The predecessor established that a mean 63% (range
   37-91%) of its optimiser's training-objective progress was reachable by a rotation
   that provably changes nothing an attacker can recover. Removing that slack is a
   closed-form change. Whether it moves *measured* recovery is open.
2. **Does the corrected mechanism offer a useful tradeoff relative to `J`, the local
   controls, and fair published-method adaptations?** The predecessor had no external
   method baselines at all. Without them "our penalty helps" has no method-level
   control.
3. **Does the result persist descriptively across the already-used 2018 and 2017
   pools?**

---

## 2. Registered directional forecasts

Registered before fitting. Each is falsifiable and each is reported with its outcome
whether or not it was met.

| # | Prediction |
|---|---|
| **Q1** | The measured rotation-only share of the training gain is `< 0.01` in **every** seed, rank and policy — far below the predecessor's 0.10 gate — because the repaired objective is invariant by construction rather than by convergence. This is structural and is checkable without any outcome. |
| **Q2** | The repaired family attains a **higher** (worse) final training objective than the defective family at matched seed, rank and policy in at least 15 of 18 conditions, because roughly 63% of the predecessor's descent direction is no longer available. A *lower* training objective for the repaired arm would be surprising and would itself need explaining. |
| **Q3** | Despite Q2, the repaired family's **measured** additional `A/SEX` recovery is **not worse** than the defective family's at matched rank and policy in at least 2 of 3 seeds — i.e. the inert 63% really was inert, and removing it costs no measured protection. |
| **Q4** | The repaired family still does **not** satisfy the primary point-estimate coordination rule against both `L1` and `L2` in any cell. The predecessor failed all 8 registered coordination cells; the defect it repairs is not the reason it failed. |
| **Q5** | The repaired best candidate remains **significantly worse than `J` on `A/SEX`** under at least one weighting. `A/SEX` is the endpoint on which the predecessor's best candidate lost to `J`, and nothing in the repair targets it specifically. |
| **Q6** | The **LEACE** auxiliary-channel baseline achieves near-zero *linear* recovery of the erased attributes from the transformed channel alone, while its **selected** (nonlinear, full-slate) additional recovery from the augmented `A` wire remains **above** zero — because `H_A` is unchanged and still discloses, and the attacker slate is not linear. |
| **Q7** | At least one of the four external adaptations proves to be an **alias** of an arm already fitted, or **infeasible** under its own stated conditions, and is therefore credited-and-reused or reported as scoped-infeasible rather than refitted. |

Forecast **Q1** is the only one whose failure stops work: it is the mechanism gate
(§5). The others are scientific bets whose outcome is reported either way.

Honest disclosure required by standing practice: these are registered **now**, before
fitting. Anything discovered later that was not registered is disclosed as **post-hoc**
at the point it is reported.

---

## 3. Frozen constants and objects

Everything in `METHOD.md` §1-§5, plus:

| Constant | Value |
|---|---|
| Seeds | `0, 1, 2` |
| Ranks | `16, 8` (the positive-eigenvalue rule already returns `r_plus = 16`; **no alias is refitted** and compression is never presented as the rule's outcome) |
| Policies | `L1`, `L2`, `C1` |
| Kernel subset size | `min(512, n_j)` per role |
| Kernel subset seed | `20260930 + 100*seed + 10*rank + role_index` |
| Bandwidth subset | `min(1024, n)`, `default_rng(20260918 + 100*seed + r)` (predecessor recipe, reused) |
| Bandwidth multipliers | `(0.5, 1.0, 2.0)` |
| Kernel estimator | **V-statistic**, diagonal retained (declared; never swapped on outcome) |
| Block weights | quadratic `0.5`, kernel `0.5` |
| Linear/nonlinear weights | `0.5` / `0.5` |
| `FLOOR` | `1e-12`, with the **nondividing** degenerate conventions of `METHOD.md` §4 |
| Optimiser budget | 200 full-objective updates per start, 2 deterministic starts, 4 eligible checkpoints |
| Perturbation seed | `20260920 + 100*seed + r`, magnitude `0.05` |
| Selection | lowest **training** objective only |
| BLAS/OpenMP threads | 1 |
| Fitting workers | 1 |

Training subsets, representation-fitting rows, nuisances, service predictors, channel
inputs and household splits are **identical** to the prior experiment. No new seed
sweep, strength sweep, hidden-task tuning, bandwidth search on development scores, or
architecture expansion is authorised.

---

## 4. Mathematical validation, with tolerances declared in advance

These run **before** any large fitting and before any performance outcome. Tolerances
are scale-aware and are set from documented floating-point expectations, not from
observed results.

### 4.1 Declared tolerances

Let `eps = 2.22e-16`. The quadratic moment accumulates over `n ~ 1.05e4` rows in
float64, so a pairwise-summation relative error of order `sqrt(n) * eps ~ 2.3e-14` is
expected; QR retraction and the `r x r` conjugation add `O(r * eps) ~ 3.5e-15`. The
declared tolerances carry roughly 1-2 orders of headroom over that.

| Check | Declared tolerance |
|---|---|
| Quadratic full-symmetric vs packed `sqrt(2)` implementation | relative `1e-12` |
| Kernel block vs independent dense-Gram reference | relative `1e-12` |
| Rotation / sign / permutation invariance of each component and of `L` | `abs(delta) <= 1e-10 * max(1, abs(value))` |
| Analytic vs central finite-difference gradient (`h = 1e-6`) | relative `1e-6` |
| Derivative along pure rotation directions `W A`, `A` skew | `abs(<G, WA>) <= 1e-9 * ||G||_F * ||WA||_F` |
| Chunk-size and repeated-serial-evaluation invariance | exact bitwise, else relative `1e-12` |
| `H` parity on every released pool | **bitwise** (`np.array_equal`) |
| Feasibility `max abs(W'W - I)` | `<= 1e-13` |

A **rotation-share ratio is not the correctness gate.** It is unstable when the total
gain is tiny and it is not a causal decomposition. The gate is the direct
absolute/relative invariance identities above.

### 4.2 Required checks

1. Quadratic Frobenius identity vs packed `sqrt(2)` implementation.
2. Kernel objective vs an independently computed small dense Gram formula.
3. Objective, reconstruction and **each** penalty component under **multiple** random
   rotations, sign flips and permutations, across ranks and seeds. Not one chosen `Q`.
4. Analytic/autodiff gradients vs central finite differences.
5. Near-zero derivative along pure rotation directions `W A` for skew-symmetric `A`.
6. Exact `H` parity and legal input dependence (`A` inference reads `X` and `H_A` only).
7. The four falsification fixtures, carried over and re-pointed at the new objective:
   **magnitude leakage**, **XOR with side information**, **conditional-null with an
   oracle nuisance**, and **deliberately misspecified nuisance**. The last two exist
   to show the penalty can be **wrong**; their expected behaviour is declared in the
   fixture, not inferred afterwards.
8. Original fixed-matrix solver replay, and old rank/alias behaviour (the `original`
   family at rank 16 must still reproduce the historical arms bitwise).
9. Repeated serial CPU evaluations and alternate chunk sizes on the same saved objects.

**If an implementation fails, it is repaired and the relevant fixture re-run.** If an
intended mathematical property cannot be satisfied by this formulation at all, the
reason is recorded, that arm stops being fitted, and the independent baselines and
evidence work continue. A fixable bug does not stop the assignment.

---

## 5. Mechanism acceptance gate (outcome-free)

After fitting and **before any 2018 score is read**, `rotation_share` is computed on
the fitted maps. The repaired arm is eligible to be scored only if the measured
rotation-only share is below `0.10` in every seed, rank and policy — the
predecessor's own predeclared gate, reused. Forecast Q1 sharpens this to `< 0.01`.

If the gate fails, the construction is still misspecified, the repaired arm is **not**
scored, and that is itself the reportable result. The baselines and the evidence work
continue regardless.

---

## 6. The bounded matrix

### 6.1 Repaired arm — 18 new mapper fits

`rotation_invariant` family x `{L1, L2, C1}` x `{rank 16, rank 8}` x `{seed 0,1,2}`.

Reused, never refitted:

* matched-rank **original** (linear-moment) controls — at rank 16 these **are** the
  historical `spectral_L1`/`L2`/`C1`;
* matched-rank **defective nonlinear** controls from
  `results/pcrl_nonlinear_rank_v1/`, so the fix is attributable to the fix;
* frozen `H`, `E`, `A0`, `L025`, `L20`, `J`, and `spectral_S0`.

Both objectives are compared **within their own definitions**. Values from
differently normalised objectives are **not** directly comparable measures of privacy,
and are never presented as such. The comparison that carries weight is between
**independent attacks on the actual released wires**.

### 6.2 External adaptations — up to 15 further fits

Detailed in `BASELINE_ADAPTATIONS.md`, written after primary-source review and before
the corresponding fits. Fairness conditions binding on **all** of them, declared now:

* **Every compared release must preserve `H` exactly.** Baselines are applied to the
  *auxiliary channel*, and the transformed channel is appended to **unchanged** `H_A`.
  `H_B` is unchanged. The predecessor's specification said to apply erasure to the
  concatenated `[H_A, Z]` wire; that is **not done here**, because editing the
  combined wire can alter `H`.
* **No representation or baseline fitting may use residence or commute labels.** The
  predecessor's specification had SPLINCE preserving covariance with the residence
  label; that is **not done here**, because it would defeat the held-out-task test.
  Covariance preservation targets the authorised **training** tasks
  `income_binary` and `civilian_at_work`.
* A baseline's guarantee concerns only the channel it transformed. **No guarantee is
  claimed for the augmented release**, which still contains `H`.
* Every baseline is evaluated at a **declared width**, scored by the **same** attacker
  families on the **same** validation pool and selection rule, with absolute and
  incremental recovery reported separately.
* Shared policy loss coefficients do **not** imply identical effective regularisation
  across methods. This is disclosed as a limitation, not resolved.
* Primary papers and official code are read before implementation; existing correct
  repository implementations are used where possible. **A homegrown substitute is
  never labelled as the published algorithm.** If a method cannot be implemented
  faithfully with accessible materials, the other arms complete and the concrete
  limitation is reported.
* Any authorised-task variant is named an **adaptation**, not a verbatim reproduction.
* **No extra baseline tuning sweep is authorised.**

| Arm | Nominal new fits |
|---|---:|
| `rotation_invariant` x 3 policies x 2 ranks x 3 seeds | 18 |
| LEACE auxiliary-channel x 3 seeds | 3 |
| SPLINCE auxiliary-channel x 3 seeds | 3 |
| OptNet-ARL adaptation x 3 policies x rank 16 x 3 seeds | 9 |
| **Nominal total** | **33** |

Fewer where an alias is **proved** or a formulation is documented **infeasible**. The
spectral/SARL audit (§6a of the assignment) deliberately aims to *avoid* fits: if the
existing marginal spectral arms already are SARL-style residual-teacher adaptations
under their exact parametrisation, they are credited and reused, not duplicated to
attach a published name.

This bounded comparison is **not** a comprehensive benchmark of every cited method.

---

## 7. Evaluation and reporting rules

* **2018** is the development evaluation, completed first.
* **2017** is then evaluated under its already-established fitting/validation/
  evaluation partitions. For these new methods 2017 is **development** and is labelled
  so everywhere. It is **not** a new test merely because the new methods have not been
  scored there. The original locked 2017 study keeps its historical status and is not
  relabelled or replaced.
* **2016 is not scored, fitted, transformed through models, inspected for outcome
  distributions, or used for any performance diagnostic in this assignment.**
* Compatible historical outputs are reused by **exact hashes and recipes**. Evaluation
  models are refitted for every genuinely new wire.
* Common fresh exposure is kept separate from historical catch-up exposure.
* Preserved and reported: raw absolute recovery, increments over `H`, selected `H`
  attack scores, all protected classes and support flags, source-vector parity,
  source-probe losses, the `.01` residence reference, the half-headroom criterion, and
  the commute findings.
* The fixed withholding `p` grid and its actual branch-routed mechanism are kept. If
  expected utility is interpolated for a post-hoc matched point, feasibility
  `p in [0,1]`, the post-hoc status of the selection, and the uncertainty scope are all
  reported. Arbitrary averaged probability predictions are never substituted.
* The original **point-estimate** coordination rule is reported for continuity, and is
  described as a one-sided point-estimate rule — **not** statistical equivalence and
  **not** noninferiority within `.001`. Equivalence and noninferiority are reported
  **separately** with correct bounds. Absence of significance is neither.
* **No forbidden endpoint is ignored to nominate a winner.**
* Cached source-pass/fail wording and recovery-label errors are corrected in new
  reports **without** rewriting the historical evidence.
* All uncertainty intervals are **descriptive development uncertainty conditional on
  fitted systems**. They do not undo repeated use and do not cover all possible
  retraining runs. **Every seed and every weighting is shown**, not only pooled means.

---

## 8. Numerical integrity

One fitting worker until memory measurements support more. Exact kernels and
prediction batches chunked. Audit suites never concurrent with a model job. Isolated
write paths, atomic checkpoints, reproducible seeds, CPU reference calculations, and
independent replay for headline predictions and selected scores. Inconsistent outputs
are **preserved or quarantined** and affected dependencies rescored serially;
numerical disagreements are never silently averaged.

Memory pressure was observed on this machine. Its causal role in the predecessor's
rare faults is **suspected, not demonstrated**, and nothing here upgrades that.

---

## 9. Amendments

Any amendment is appended to `RUN_STATUS.md` with its **timing** relative to the fit
and score phases and the **units it affects**. An amendment made after an outcome is
read is labelled as such. The registered forecasts in §2 are never altered.

---

## 10. What would make a candidate worth confirming

A candidate merits a fresh confirmation proposal only if a complete development result
identifies a **specific better practical option** against `J` **and** both local
controls **and** the external adaptations, with a stated utility/disclosure tradeoff
and **no silently ignored harmful role**. A lower training objective, or one
favourable role, is insufficient. Mixed evidence is frozen and reported, not
re-metricised until it reads as a win. A repaired invariant alone is **not** a clean
scientific win.

**2016 is not authorised by this protocol.** Any confirmation proposal would need its
own prospective protocol, its own registered predictions, and an outcome-free
admission pass.
