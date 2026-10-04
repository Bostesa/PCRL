# Method card: J-G (refreshed critics + per-recipient local feedback, joint coalition penalty)

**In plain language.** Two recipients each receive their own learned 16-number feature vector plus their task outputs.
Training tries to keep SEX hard to read from each recipient's release and from the two releases combined (the
coalition), while keeping both tasks useful. Every four epochs the features are frozen so that strong attackers can
relearn to read them; training then resumes against those refreshed attackers. If either recipient's own protection
slips past a budget set by the local-only reference, that recipient's protection weight is raised. Only models that keep
useful accuracy on both tasks and are locally no worse than the comparators can be nominated.

| Component | Specification |
|---|---|
| Encoders / heads | two separate MLPs 83-64-64-16 (ReLU), affine heads; warm start shared with every arm |
| Release | r_i = g_i(X) (no erasure); deployed head = StandardScaler + LR (C on HEAD_VALIDATION); view [r_i, centred logits_i] |
| Critic views | v_i = [r_i, centred(W_w r_i + b_w)] with the seed's fixed warm-start head (review R1-a); pair = [v_1, v_2] |
| Critics | per view (v1, v2, pair): A = MLP(32), B = MLP(64, 64); constant prior predictor |
| Surrogate | R_v = (CE_const − min(CE_const, CE_A, CE_B)) / H_fit(S) — an empirical recovery surrogate, not mutual information |
| Objective | L_task + β(R_1 + R_2 + R_pair)/3 + λ_1(R_1 − c_1) + λ_2(R_2 − c_2) |
| Refits | epochs 0, 4, 8, 12, 16, 20: frozen snapshot, new floored ZCA transform held for the block, continued (transported) vs fresh restart per kind, ≤ 15 epochs, patience 3, choose by CRITIC_VAL CE |
| Online steps | 5 critic Adam steps (3e-3) per encoder step between refits |
| Multipliers | λ_i ← clip(λ_i + β(R_i − c_i), 0, 3β) at refits, from CALIB rows; c_i = R_i(L-R) + 0.005 |
| Encoder step | SGD 0.05 on −∇(L_1 + L_2) − ∇P, global-norm clip 5, 20 epochs × 76 steps from the frozen L-R checkpoint |
| Selection | inner AUC attackers; gates vs U; local AUC ≤ min(L-R, L-G, C*) per recipient; lowest coalition AUC |

**What it is not.** A bounded nonconvex primal/dual heuristic. It does not enforce population constraints, prove
optimality, compute best responses, provide fitted-row linear guardedness (that was LEACE), or give DP/MI/universal
nonlinear guarantees. Refitting the critics changes optimisation; it does not by itself remove information. A training
critic at chance is not evidence of protection; only the independent audits count.

**Known proximity to prior work** (bounded check, see `MATH_REVIEW.md`): adversarial representation learning with
alternating adversary updates (e.g. LAFTR, Madras et al. 2018); constrained fairness via Lagrangian/reductions with
multiplier updates (Agarwal et al. 2018). No algorithmic novelty is claimed; novelty is assessed separately from whether
the recipe works.
