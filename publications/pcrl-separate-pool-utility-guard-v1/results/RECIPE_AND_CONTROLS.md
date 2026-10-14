# Recipe and matched controls (frozen in SCIENCE_LOCK)

**Start.** The admitted RAW_JOINT strength-0.3 epoch-5 packages:
- seed 0: `a1a0dfba…4c`
- seed 1: `b10a389a…bd`

Each seed keeps its inherited compressed frozen-income interface and admitted per-seed head-row-space basis byte for byte (geometry bindings `75ad1421…` and `2a3a667c…`).

**Release.**
- Income: frozen RAW encoder/head, with q = z B^T and released z = qB. Its own logits, probabilities and decision are computed from the released features. Effective feature rank is 1, which is known compression.
- Occupation: the actual learned z16 and its own centered logits, probabilities and decision.
- Nominal widths are 21 for income, 29 for occupation and 50 for the aligned pair.
- No raw income z, discarded residual, labels, ids or teacher outputs are co-released.

## Shared across all five arms

- **Task objective:** (N_UPDATE/N_TOTAL) × mean CE on the 256-row UPDATE minibatch + (N_GUARD/N_TOTAL) × mean CE on the whole GUARD pool. The weights are 8180/10223 and 2043/10223. Every control therefore learns from GUARD task labels.
- **Optimizer:** no-momentum SGD with learning rate 0.01. The actual occupation displacement norm is at most 0.01, with the inherited norm-only rounding repair in all arms.
- **Critics:** six actual-wire critics (linear, MLP32 and MLP64, on the occupation29 and pair50 views), trained with Adam 0.003. Each gets 80 warm updates, then five updates per critic before each proposal. They train on UPDATE rows and UPDATE SEX only, and they keep training in TASK_ONLY.
- **Critic normalization:** the reference is 256 fixed UPDATE rows, refreshed initially and each epoch, with the inherited positive scale floors.
- **Coefficient:** one per seed, 3 × median ||g_shared_task|| / median ||g_sensitive|| over 16 frozen warm draws, with the shared task objective inside. It is shared by all arms. The medians must be above 1e-12 and the coefficient at most 1e4.
- **Sensitive objective:** the sum over views of 1 − min(critic CE, SEX entropy)/entropy.
- **Proposal:** d = −0.01(g_task + λ g_sensitive). TASK_ONLY applies g_sensitive = 0.
- **Schedule:** 32 proposals per epoch (ceil(8180/256)), 5 epochs, checkpoints at epochs 0/1/3/5. All arms share starting weights and random streams.

## Arms

| Arm | Privacy term | Task acceptance |
| --- | --- | --- |
| SEPARATE_GUARD | applied | absolute anchored budgets on the whole fixed GUARD pool |
| GLOBAL_UPDATE_GUARD | applied | same rule on the whole fixed UPDATE pool |
| OLD_BATCH_GUARD | applied | predecessor strict NLL/Brier nonincrease on the current UPDATE minibatch |
| ORDINARY | applied | none (norm bound only) |
| TASK_ONLY | zero | none (norm bound only) |

## Absolute global safeguard

- **Anchor:** exact starting metrics on the arm's fixed pool. Budgets are NLL ≤ NLL_0 + 0.010, summed Brier ≤ Brier_0 + 0.005, and correct ≥ correct_0 − floor(0.010 N). The anchor is never reset.
- **Projection:** at each state, take whole-pool NLL/Brier gradients (fixed 1024-row chunks, float64 sums) and slacks b. Then solve min ||d − p||² subject to g·d ≤ b and ||d|| ≤ 0.01, by exact KKT active-set enumeration. The solver was tested against SLSQP.
- **Acceptance:** apply the real mixed-dtype displacement, read the actual movement, and verify the norm and the linearized constraints (tolerance 1e-10). Then recompute whole-pool NLL, Brier and accuracy and accept only inside the frozen budgets (NLL/Brier tolerance 1e-10). Backtrack at fractions 1/0.5/0.25/0.125. A rejection restores the exact weights.
- **Tolerances:** slack in [−1e-10, 0) is treated as 0 and logged; slack below −1e-10 restores the state and is disclosed.

These are internal fitting budgets and do not establish the original U0/Ucal criterion. A passed GUARD check guarantees only that empirical fitting-pool check.

## Privacy-bearing movement (registered before fits)

- **Probe:** the first 512 UPDATE rows in SHA256(`SPG1|UPDATE-wire-probe|20261013|` + row_id) order.
- **Wire tolerance:** max(1e-10, 10 × repeat-forward noise). The measured noise was 0 in both seeds, so the tolerance is 1e-10 for representation, logits and probabilities. Any decision change also counts.
- **Counterfactual scan:** SEPARATE_GUARD, the first eight accepted nonzero updates per seed. At the same state, remove the applied privacy gradient and keep the identical task gradient, pool, budgets, solver and ladder, then restore the recorded state. The contribution counts if the max abs difference of actual displacements exceeds max(1e-10, 10 × repeat-computation proposal noise), and the update must change the probe wire.

There is no accepted-step percentage requirement and no prior AUC requirement.

## Gate (VALIDATION opened once after TRAINING_COMPLETE_LOCK)

- **Incremental caps:** versus the unchanged start, accuracy loss ≤ 0.010, NLL increase ≤ 0.010 and summed Brier increase ≤ 0.005, for both tasks and both seeds.
- **Selection:** each arm takes the latest common saved epoch passing every cap. Epoch 0 is the fallback.
- **SEPARATE_GUARD passes only if** its chosen epoch is > 0, its occupation wire changed, privacy-bearing accepted wire-changing updates exist in both seeds, every proxy passes, and the independent feasibility replay is complete.
- **Fail:** COMPLETE_NEGATIVE_FEASIBILITY, privacy comparison NOT_TESTED.
- **Pass:** Stage B runs automatically, with no new fitting recipe.

No local-only twin is registered, so no coalition-term attribution and no joint-versus-sequential claim is made. The OLD_BATCH_GUARD comparison changes both pool scope and budget form.
