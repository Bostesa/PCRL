# Method delta: ordinary-penalty joint training without mandatory erasure (PN) and its local control (LN)

## Changes from the predecessor (`research/pcrl-joint-complete-view-method-v1` @ 568f970)

| Component | Predecessor candidate J | This study's candidate PN | Local control LN |
|---|---|---|---|
| Update rule | Privacy gradient projected off active task guards (GEM/A-GEM adaptation) + backtracking | **Ordinary penalty**: `u = −∇(L1+L2) − β∇P` (the predecessor's JP rule) | Ordinary penalty |
| Privacy objective P | R_v1 + R_v2 + R_pair | R_v1 + R_v2 + R_pair (unchanged) | R_v1 + R_v2 |
| Critic banks | v1 {A, B}, v2 {A, B}, pair {A, B} | Same | v1 {A, B, B2}, v2 {A, B, B2} (equal critic-update budget) |
| Linear erasure | Official LEACE in the loop and as the final transform | **None at any point.** Identity maps in training and finalisation; the finaliser asserts that no map is reinserted. | None |
| Released view | `[r_i, centred logits]`, r_i = LEACE(g_i(X)) | `[g_i(X), centred logits]` | Same as PN |

**Unchanged components.**
- Encoders: separate 64-64-16 MLPs.
- Training heads: affine.
- Warm starts: the predecessor's hashed units, shared bitwise.
- Seeds 0/1/2 and minibatch order.
- Loss: task cross-entropy.
- Optimisation: plain SGD at learning rate 0.05, global clip 5, 20 protection epochs × 76 steps.
- Critics: ZCA-whitened, refreshed before every step; 5 critic steps at learning rate 3e-3.
- Deployed heads: refitted as StandardScaler + LR with C selected on defense_val.
- Release contract: features plus centred logits computed only from those features.

## Paired controls (corrected after the mathematical review, before the lock)

| Contrast | What it measures |
|---|---|
| PN vs JP (erased, existing units) | **The whole LEACE package**: in-loop refits plus the final map. The arm specs differ only in erasure (the guard list is inert in penalty mode), and the warm start, data order, schedule, β, critic initialisation and head-refit rule are identical. **The gradients are not identical:** JP's in-loop maps feed the task gradient and head training, the critic whitening reference views, the critic inputs and the penalty gradient, so the trajectories diverge from the first step. In-loop and final-map effects are not separated. A PN→E arm (PN encoders + final LEACE) would separate them; it is not run here. |
| PN vs LN | **Not a clean isolation of coalition coupling.** At equal β the arms also differ in penalty mass (3 vs 2 surrogate terms), in local critic-bank size (2 vs 3 critics per local view), and in how often the global clip binds, which changes the task share of each step. PN's pair bank also has no ignore-other-view critics. Both β-matched and selected comparisons are reported, with the decomposition Δcoalition = Δ(max local) + Δ(synergy). |
| PN vs projected J | Changes two components (update rule and erasure). History only. |

## Guarantees lost

- PN and LN have **no exact linear guarantee**. LEACE's fitted-moment identity no longer holds by construction: zero cross-covariance with one-hot SEX on the fitting rows for each r_i, for the primary view by affine closure, and for the coalition [r1, r2] by stacking. That is exactly what is lost.
- Nothing held-out, nonlinear or decision-level was guaranteed before either.
- The fit-row cross-covariance for PN and LN is reported as a measurement.
- Any privacy advantage is evidence only against the declared fitted attackers on these reused rows.

## Novelty

There is none in the components: an adversarial critic surrogate (Song et al. 2019), a fixed-weight penalty (LAFTR-style Lagrangian), a coalition critic adapting PCRL's own cross-purpose constraint, and ZCA-whitened critics. An empirical win would be a property of this integration on this data, not algorithmic novelty.
