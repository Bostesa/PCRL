# Method delta versus the predecessor (PN / LN, research/pcrl-penalty-no-erasure-v1 @ b628ff5)

| Item | Predecessor PN / LN | This study |
|---|---|---|
| Data roles | defense_train / defense_val (heads) / attacker_fit / attacker_val / assessment (closed) | DEFENSE_FIT (= defense_train, split into CRITIC_FIT/VAL/CALIB) / HEAD_VALIDATION (30% of defense_val) / DEVELOPMENT_ASSESSMENT (70% of defense_val, sealed) / AUDIT_FIT / INNER_SELECTION; old assessment and cert dropped |
| Surrogate | R = 1 − min CE / H on the step minibatch | R = (CE_const − min(CE_const, CE_A, CE_B)) / H (same rows); additive offset on a batch plus clamping at 0; applied identically to every new arm |
| Base weights | PN: β on each of R_v1, R_v2, R_pair (sum 3β); LN: β on each of R_v1, R_v2 (sum 2β) | joint β/3 on each of three; local β/2 on each of two; sums equal β |
| Critic banks | PN: v1, v2, pair × {A, B}; LN: v1, v2 × {A, B, B2} | every arm: v1, v2, pair × {A, B}; local arms' pair bank is a shadow (never in the encoder gradient) |
| Critic views | [r_i, centred logits of the current (drifting) training head] | [r_i, centred logits of the FIXED warm-start head] (review R1-a; same information, constant null space) |
| Critic rows | all defense_train | CRITIC_FIT (training), CRITIC_VAL (refit choice), CALIB (budgets/multipliers) |
| Critic schedule | 5 online steps per encoder step; ZCA recomputed every step | online arms: unchanged (inherited); refreshed arms: frozen-snapshot bounded refits at epochs 0, 4, 8, 12, 16, 20 with continued (transported) vs restart choice and the transform held fixed within blocks |
| Local feedback | none | guarded arms: λ_i ← clip(λ_i + β(R_i − c_i), 0, 3β), c_i = R_i(L-R) + 0.005 |
| Grid | β ∈ {0.1, 1, 10}, final epoch only | β ∈ {0.03, 0.1, 0.3} × checkpoints {5, 10, 15, 20} |
| Ladder | PN and LN from the warm start | Stage B: L-O, L-R from the warm start; Stage C: J-G, L-G, J-R, J-O from the frozen L-R checkpoint |
| Task reference | U = warm + 20 task epochs | U-B = warm + 20 (stage B gates); U = warm + 20 + e_LR (stage C gates and final utility clauses) |
| Nomination buffer | candidate local ≤ LN + 0.01 | candidate local ≤ min(L-R, L-G, C*) (zero buffer); final +0.01 clause unchanged |
| Inner attackers | selected by validation log loss | selected by INNER_SELECTION AUC (fixed orientation); final slate dual (AUC primary, CE for log-loss reporting) |
| Head C selection | defense_val | HEAD_VALIDATION |
| Inner audit of finite (FARE) releases | jcv inner slate (LR, MLP, HGB; the cell-conditional attacker was not included at the inner stage) | inner slate + cell-conditional attackers for finite views (same selection rule; recovery values differ from the predecessor's) |
| Audit controls | null ≤ 0.55, planted leak detected | the same, plus a planted clue at amplitude 1e-6; the null bound applies to the selected (maximum) inner AUC, which is optimistic for large slates (FARE pair: 51 candidates) |

Unchanged: architecture, warm starts (reused, hash-verified; trained on DEFENSE_FIT only), SGD 0.05, clip 5, 20 epochs
× 76 steps, critic learning rate 3e-3, the floored ZCA for the main refreshed arms (saved exactly), the head rule
(except its validation rows), the 83-column input contract, tasks, and the 18-slot nine-clause primary structure.
