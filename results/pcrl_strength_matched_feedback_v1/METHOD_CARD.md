# Method card: J-F (norm-controlled joint protection with active per-recipient AUC feedback)

**In plain language.** Two recipients each receive their own 16-number feature vector plus their task outputs.
Training adds a protection push to each encoder whose size is a fixed fraction (rho) of that encoder's own task push,
so different attacker-training schedules can be compared at equal protection strength. A controller periodically fits
small readers on frozen features and, when one recipient's measured SEX recovery is above its target, shifts more of
the fixed total protection effort to that recipient. Only models keeping useful accuracy on both tasks and staying
locally close to the comparators can be nominated.

| Component | Specification |
|---|---|
| Encoders / heads | two MLPs 83-64-64-16 (ReLU), affine heads; fresh warm start on NEW_DEFENSE_FIT |
| Release | r_i = g_i(X); deployed head (StandardScaler + LR, C on HEAD_VALIDATION); view [r_i, centred logits] |
| Critic views | [r_i, centred(W_w r_i + b_w)] with the fixed warm-start head; pair = [v1, v2]; critics A (MLP 32), B (MLP 64-64) |
| Proxy | R = (CE_const − min(CE_const, CE_A, CE_B)) / H; P = normalised Σ c_v R_v (joint w1, w2, 1; local w1, w2) |
| Update | encoder i: −lr (t_i + q_i), q_i = a_i p_i, a_i = rho s_i ‖t_i‖/‖p_i‖ (float64, no gradient, cap 100); heads: task gradient; global clip 5 |
| Strength | rho ∈ {0.25, 0.75, 1.5} = privacy/task encoder-gradient norm ratio (Phase B: RMS over the two encoders) |
| Schedules | ONLINE, REFRESHED (bounded refits), ONLINE_MATCHED (online + refit-count-matched extra updates) |
| Controller | probes (whitened LR / MLP) fit on CRITIC_FIT, chosen on CRITIC_VAL, AUC on CONTROLLER_CALIB; b_i = max(0.5, ref AUC_i − 0.01); w_i ← clip(w_i + clip((AUC_i − b_i)/0.01, −1, 1), 0.25, 8) at 20 epoch starts; s_i = w_i / RMS(w) |
| Selection | inner AUC attackers; task gates vs U; J-F local AUC ≤ L-F + 0.005 and ≤ C* + 0.005; lowest coalition AUC |

**What it is not.** A heuristic. Norm matching equalises the size of the protection push, not its direction, the
optimisation trajectory or task interference. The controller uses a measured AUC to steer a CE-proxy gradient; it is
not Cotter et al.'s proxy-Lagrangian method and inherits none of its guarantees. No population, DP, MI, optimality or
linear-guardedness property follows. A training critic or probe at chance is not evidence of protection; only the
independent audits count.
