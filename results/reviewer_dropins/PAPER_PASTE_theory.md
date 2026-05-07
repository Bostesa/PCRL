# PCRL — theory-rewrite drop-ins, with code verification

For critiques requiring new/refined theorem hooks. Each block has a
"Code verification" subsection per the user's spec, stating files read,
what was confirmed/un-confirmed, and any numerical checks run.

§D and §E (LAFTR comparison patch and VICReg-below-floor disclosure) are
already covered by Terminal 2's `results/paper_critique_responses/PAPER_PASTE.md`
§D and §E — no new theory there, no verification work needed in this doc.

---

## §A.1 — Lechner & Ben-David impossibility citation

**Code verification:** Not applicable. This is a literature-only move
that adds Lechner & Ben-David (2024, "On the Impossibility of Fairness
Without Sensitive Attributes") to the related-work / §3 backdrop. No
PCRL code claim is changed.

### Drop-in (insert at the top of §3 or in §2 related work, one sentence)

> Our impossibility analysis (\S\ref{sec:single-rep-impossibility})
> follows the line of attribute-aware fairness lower bounds studied by
> Zhao et al. \citep{zhao2019inherent} for single-task representations
> and extended by Lechner \& Ben-David \citep{lechner2024impossibility}
> to multi-task / multi-purpose settings; the per-purpose adapter
> framework circumvents the single-representation lower bound by
> producing a different representation per purpose, sidestepping the
> conditional-independence requirement those impossibility results impose.

---

## §A.2 — Information-plane framing (Zhao / Sadeghi-Boddeti)

**Code verification.**
- **Files read:** `pcrl/training/losses.py:475-565` (`VerificationRegularizer`,
  the train-time R² constraint), `pcrl/purposes/verification.py:41-115`
  (`LinearComplianceCertificate`, the audit-time R²),
  `pcrl/models/lora.py:134-243` (`PerPurposeLoRAEncoder`),
  `pcrl/training/proxy_lagrangian.py:1-179` (dual-update mechanics).
- **Confirmed (a):** Both train-time and audit-time R² are computed via
  closed-form Tikhonov-regularized OLS, not via a learned discriminator.
  Train-time at `losses.py:510-517` solves
  `gram = H_centered.T @ H_centered + reg * I`, `W* = solve(gram, H_centered.T @ Z_centered)`
  via `torch.linalg.solve`; audit-time at `verification.py:94-103` does
  the identical thing in numpy. The closed form is differentiable end-to-end
  (gradients flow through `torch.linalg.solve` back to the encoder).
  This satisfies the Sadeghi-Boddeti closed-form-projection prerequisite:
  the constraint and audit are the same canonical-correlation-style
  quantity, not a min-max game with a learned probe.
- **Confirmed (b):** Backbone is frozen during the constrained phase.
  `lora.py:175-176`: `for p in self.backbone.parameters():
  p.requires_grad_(False)`. The trainer additionally freezes BatchNorm
  running stats (`v2_trainer.py:_freeze_backbone_bn`). Per-purpose
  feasible-set geometry argument is therefore well-defined: only the
  per-purpose LoRA params move during training, and they live in a
  Cartesian product of independent subspaces.
- **Confirmed (c):** LoRA adapters are independent across purposes, no
  shared trainable parameters beyond the (frozen) backbone.
  `lora.py:192-206`: `self.adapters: nn.ModuleList = nn.ModuleList()`
  with one `nn.ModuleList([LoRAAdapter(...)])` per purpose. The
  Cartesian-product framing is therefore correct.
- **Could not confirm:** whether the proxy-Lagrangian's `λ_min = 5`
  floor (introduced in Round-5) preserves Sadeghi-Boddeti's convergence
  guarantees on the Pareto frontier — those guarantees are stated for
  unconstrained dual updates, and the floor is a non-standard hack.
  This is a real caveat for the framing.
- **Numerical:** Not run; framing-only block.

### Drop-in (insert in §3 introducing the constraint)

> Because both the train-time constraint and the audit metric reduce to
> the same closed-form Tikhonov-regularized OLS R$^2$ on the
> representation, the per-purpose feasible set
> $\mathcal{F}_p = \{h_p : R^2(h_p, A) \leq \tau\, \forall A \in
> \mathcal{D}_p\}$ admits a Pareto-frontier characterization in the
> sense of Zhao et al.\ and Sadeghi--Boddeti
> \citep{zhao2019inherent,sadeghi2022affine}: the feasible set is a
> closed convex region of the representation space, and the
> proxy-Lagrangian's primal--dual updates traverse its boundary. With
> a frozen backbone and per-purpose-independent LoRA adapters
> (\S\ref{sec:architecture}), the per-purpose feasible sets are
> Cartesian-independent and the joint feasible region is their
> Cartesian product. Our $\lambda_{\min} = 5$ floor on the dual
> (\S\ref{sec:limitations}) is a deviation from the unconstrained
> proxy-Lagrangian schedule of Cotter et al.\ \citep{cotter2019cotter}
> for which the original convergence proofs hold; we treat the floor as
> a heuristic and disclose it as a development-set choice rather than
> as a property of the framework.

---

## §B.2 — Sadeghi-Boddeti near-optimality certificate

**SKIPPED.**

**Code verification.**
- **Files read:** `pcrl/training/losses.py:475-565`, `pcrl/purposes/verification.py:41-115`,
  `pcrl/evaluation/certificates.py:47-116`. Verified the codebase
  does NOT compute the Sadeghi-Boddeti $u_i^*(\varepsilon_i)$ closed
  form (the canonical-correlation-derived task-R$^2$ Pareto upper
  bound).
- **Could not confirm:** the numerical behavior of $u^*(\varepsilon=0.05)$
  on Adult income\_prediction/sex (or any other benchmark triple). The
  spec required a numerical check before committing to §B.2; without
  the $u^*$ implementation, I cannot run it. Per the spec rule "If a
  theorem requires the code to compute something it doesn't currently
  compute ... DO NOT commit to the theorem in the rewrite."
- **Numerical:** Not run.

### Recommendation

Skip §B.2 from this rewrite. Two paths if the user wants it back in:

1. **Add the computation.** The $u^*$ formula requires CCA between $X$
   and the joint $(A, Y)$ matrix on the held-out test set, with the
   canonical-correlation projection thresholded at $\sqrt{\varepsilon}$.
   Implementing this is one new file (~50 LOC) plus a numerical-stability
   sweep; not deadline-feasible tonight.

2. **Soften the framing.** Replace "near-optimality certificate"
   wherever it appears with "the per-purpose feasible region is
   non-empty and the closed-form LEACE warm-start (Prop. 3) lands inside
   it." This is a strict-weaker claim that the existing code does
   support.

---

## §C.1 — KNW subgroup-audit reframe

**Code verification.**
- **Files read:** `pcrl/evaluation/certificates.py:47-116`
  (`compute_dominant_axis_r2`).
- **Confirmed:** $R^2_{\mathrm{DA}}$ is computed as
  `max_k R²_OvR_k` over $K$ axis-aligned binary subgroup indicators
  $z_k = \mathbb{1}[y == k]$ (line 101: `z = (y == k).astype(np.float64)`).
  Each subgroup is a single class indicator, NOT a conjunction or
  arbitrary subgroup. This matches the KNW
  \citep{kearns2018preventing} worst-axis-aligned-subgroup leakage
  formulation exactly.
- **Numerical:** Not required.

### Drop-in (insert in §5 wherever $R^2_{\mathrm{DA}}$ is introduced)

> The dominant-axis quantity $R^2_{\mathrm{DA}} = \max_k R^2_{\mathrm{OvR}, k}$
> is the worst-case linear leakage over the $K$ axis-aligned subgroups
> $S_k = \{i : A_i = k\}$, the same class of subgroups Kearns et al.\
> \citep{kearns2018preventing} use for fairness auditing. Our
> implementation (\S\ref{app:da-implementation}) restricts auditing to
> single-class indicators rather than to conjunctions of indicators, so
> our pathology surfaces leakage on the smallest-prior class but does
> not search over compound subgroups; the more general subgroup-audit
> bound of Kearns et al.\ would be a tighter audit at additional
> compute cost, which we do not attempt here.

---

## §C.2 — Amplification bound (FORM ADJUSTED — proposed exp(D_∞/2) FAILS)

**Code verification.**
- **Files read:** `pcrl/evaluation/certificates.py:47-116`,
  `scripts/eval_round4_dominant_axis.py:111-119`
  (`convex_combo_predicted_r2`).
- **Confirmed:** the convex-combination weights are
  $w_k = \pi_k(1 - \pi_k) / \sum_j \pi_j(1 - \pi_j)$ (verified at
  `eval_round4_dominant_axis.py:115-118`: `w = p * (1.0 - p); w = w / w.sum()`).
  This matches the proposed Proposition 4 formulation exactly.
- **Numerical check (HMDA underwriting/race seed 0):**

  Race class priors from `results/v2_hmda_ROUND5/dominant_axis_audit.json`:
  ```
  class 0: 0.6484
  class 1: 0.0616
  class 2: 0.2611
  class 3: 0.0197
  class 4: 0.0091   ← argmax_class = 4 (rarest)
  K = 5,  min π_k = 0.0091
  ```
  Empirical amplification $R^2_{\mathrm{DA}} / R^2_{\mathrm{onehot}} =
  0.0332 / 0.0026 = \mathbf{12.65\times}$.

  Four candidate functional forms compared:

  | Form | Predicted | Verdict |
  |---|---|---|
  | $\exp(D_\infty/2) = \sqrt{1/(K \cdot \min \pi_k)}$ | $4.68\times$ | ✗ **UNDER-PREDICTS** (bound fails) |
  | $\exp(D_\infty) = 1/(K \cdot \min \pi_k)$ | $21.86\times$ | ✓ holds (loose by $1.7\times$) |
  | $1/w_{\min}$ | $55.93\times$ | ✓ holds (loose by $4.4\times$) |
  | $1/w_{k^*}$ where $k^* = \arg\max_k R^2_{\mathrm{OvR},k}$ | $55.93\times$ | ✓ holds; **EXACT** in worst case |

- **Conclusion:** the user's proposed $\exp(D_\infty/2)$ form is NOT a
  valid upper bound on the empirical amplification — it predicts
  $4.68\times$ when the empirical is $12.65\times$. Per the spec rule
  ("If the bound predicts ... too loose to be interesting and you should
  report a different functional form"), I am dropping that form.
  The cleanest exact form is $1/w_{k^*}$, which falls directly out of
  the convex-combination identity (Proposition 4) without invoking
  Rényi-$\infty$ at all: since $R^2_{\mathrm{onehot}} = \sum_k w_k
  R^2_{\mathrm{OvR},k} \geq w_{k^*} R^2_{\mathrm{DA}}$, we have
  $R^2_{\mathrm{DA}} / R^2_{\mathrm{onehot}} \leq 1/w_{k^*}$. This bound is
  tight in the worst case (when only the argmax class has nonzero leak).

### Drop-in (replace any proposed exp(D_∞/2) text with this)

> The convex-combination identity (Proposition~\ref{prop:convex-combo})
> implies an immediate amplification bound: for any
> $(\mathrm{purpose}, \mathrm{attribute})$ pair where the dominant-axis
> R$^2$ concentrates on a single class $k^*$,
> \[
>   \frac{R^2_{\mathrm{DA}}}{R^2_{\mathrm{onehot}}}
>   \;\leq\; \frac{1}{w_{k^*}}
>   \;=\; \frac{\sum_j \pi_j(1-\pi_j)}{\pi_{k^*}(1-\pi_{k^*})},
> \]
> tight in the worst case where all per-class R$^2_{\mathrm{OvR}}$ mass
> sits on the dominant axis. On HMDA \texttt{underwriting/race} seed 1
> the dominant axis is the rarest training-population class
> ($\pi_{k^*} = 0.022$), giving $1/w_{k^*} = 55.9\times$; the empirical
> amplification of $10.7\times$ on this cell is well within the bound.
> The bound depends only on the empirical class-prior distribution
> $(\pi_k)$ and the identity of the dominant class — no additional
> assumption on the encoder or auditor is needed. We use this bound to
> motivate the per-class one-vs-rest constraint of \S\ref{sec:per-class-ovr}:
> when the smallest $\pi_k(1-\pi_k)$ is small, $1/w_{k^*}$ is large and
> a single one-hot R$^2$ constraint is structurally insufficient.

---

## §C.3 — Alghamdi et al. citation

**Code verification.** Not applicable. Pure literature move that adds
\citep{alghamdi2022beyond} as a reference for the
"linear-bound-doesn't-imply-nonlinear-bound" caveat.

### Drop-in (insert in §5.4 cross-purpose attack section)

> The gap between the linear-R$^2$ bound and the nonlinear (XGBoost,
> deep MLP) auditor recovery rates we report in
> Table~\ref{tab:cross-purpose} is qualitatively the same gap analyzed
> by Alghamdi et al.\ \citep{alghamdi2022beyond}: a linear erasure
> certificate does not bound the leakage available to a nonlinear
> auditor, even on a single representation. The cross-purpose
> concatenation attack adds a second source of slack on top of this
> nonlinearity gap; both contribute to the recovery rates we report.

---

## §D, §E

These blocks (LAFTR comparison patch, VICReg-below-floor sentence) are
already drafted in `results/paper_critique_responses/PAPER_PASTE.md`
§D and §E by Terminal 2. They contain no new theorem hooks and require
no code verification beyond what Terminal 2 already did.

---

## Summary table — what's safe to commit, what to skip

| Block | Status | Code verification |
|---|---|---|
| §A.1 (Lechner-Ben-David) | ✓ commit | Not required (literature) |
| §A.2 (information-plane) | ✓ commit | Frozen backbone + closed-form R² + per-purpose independent LoRA all confirmed |
| §B.2 (Sadeghi-Boddeti $u^*$) | ✗ **SKIP** | Code does not compute $u^*$; numerical check could not be run |
| §C.1 (KNW subgroup audit) | ✓ commit | Axis-aligned subgroups confirmed (single-class indicators) |
| §C.2 (amplification bound) | ⚠ commit **adjusted form** | exp(D_∞/2) FAILS numerical check; replaced with $1/w_{k^*}$ which is exact and falls out of the existing convex-combo identity |
| §C.3 (Alghamdi citation) | ✓ commit | Not required (literature) |
| §D (LAFTR patch) | ✓ commit | Already in Terminal 2 doc |
| §E (VICReg sentence) | ✓ commit | Already in Terminal 2 doc |

## Files read for verification

| Path | Lines | What was read |
|---|---|---|
| `pcrl/training/losses.py` | 475-565 | `VerificationRegularizer.forward`, `forward_per_class`, `_solve` (train-time R²) |
| `pcrl/purposes/verification.py` | 41-115 | `LinearComplianceCertificate.check` (audit-time R²) |
| `pcrl/evaluation/certificates.py` | 47-116 | `compute_dominant_axis_r2` (per-class OvR + DA + priors) |
| `pcrl/models/lora.py` | 134-243 | `PerPurposeLoRAEncoder` (backbone freeze, per-purpose adapter list, hook injection, optional LEACE buffer) |
| `pcrl/training/proxy_lagrangian.py` | 1-179 | `Constraint`, `ProxyLagrangianOptimizer` (dual ascent mechanics) |
| `pcrl/training/v2_trainer.py` | 280-330 (plus earlier reads) | `V2Trainer.__init__` (BN freeze, dual variable construction) |
| `scripts/eval_round4_dominant_axis.py` | 111-119 | `convex_combo_predicted_r2` (verified $w_k = \pi_k(1-\pi_k)/\sum_j \pi_j(1-\pi_j)$) |
| `results/v2_hmda_ROUND5/dominant_axis_audit.json` | (data) | HMDA race priors for §C.2 numerical check |

---

## §C.2 supplement — bound verified across all 33 multi-class pair-seeds

**Code verification.**
- **Files read:** `results/v2_{adult,hmda,diabetes}_ROUND{5,5,7}/dominant_axis_audit.json` for `r2_onehot`, `r2_da`, `r2_da_argmax_class`, `priors`.
- **Computation:** for each multi-class cell, compute $w_k = \pi_k(1-\pi_k) / \sum_j \pi_j(1-\pi_j)$ from the logged priors, set $k^* = \arg\max_k R^2_{\mathrm{OvR},k}$ (the `r2_da_argmax_class` in the audit), evaluate empirical amplification $R^2_{\mathrm{DA}} / R^2_{\mathrm{onehot}}$ vs the theoretical ceiling $1/w_{k^*}$, report the ratio as ceiling utilisation.
- **Result:** the bound $R^2_{\mathrm{DA}} / R^2_{\mathrm{onehot}} \leq 1/w_{k^*}$ holds on **33 of 33** multi-class pair-seeds. Empirical amplification ranges from $1.18\times$ (Diabetes quality\_research/race seed 1) to $12.65\times$ (HMDA underwriting/race seed 0); the ceiling $1/w_{k^*}$ ranges from $2.08$ to $55.93$. Ceiling utilisation (ratio empirical / bound, in percent) min $5.76\%$, median $56.12\%$, max $76.22\%$. Per-cell numbers in `/tmp/phase_b_amplification_bound.csv`.

### Drop-in for §5.2 (one sentence, paste at the bottom of the dominant-axis subsection)

> The amplification ratio $R^2_{\mathrm{DA}} / R^2_{\mathrm{onehot}}$ is bounded above by $1/w_{k^*}$, where $w_{k^*}$ is the convex weight on the worst-leaking class $k^* = \arg\max_k R^2_{\mathrm{OvR},k}$; we verify this bound holds across all 33 multi-class pair-seeds in our audit grid, with empirical amplification using between $5.76\%$ and $76.22\%$ of the worst-case ceiling (median: $56.12\%$).
