# Method card: joint complete-view release (JCV) — experimental prototype

**Status.**
- An experimental adaptation, tested on reused Adult development data.
- Its algorithmic novelty is **low**: every ingredient is established (see `PRIOR_WORK_DELTA.md`).
- Any empirical result is a separate judgment from novelty.
- The registered code is `jcv/` (`train.py`, `project.py`, `finalize.py`) at the lock commit.

## What runs

```
X (83 permitted raw-input columns) ──► g1 (MLP 64-64-16) ──► LEACE_1 (official, vs one-hot SEX) ──► r1 ──► h1 (affine) ──► centred logits c1
                                  └──► g2 (MLP 64-64-16) ──► LEACE_2 (official, vs one-hot SEX) ──► r2 ──► h2 (affine) ──► centred logits c2
recipient 1 (income) receives v1 = [r1, c1]        recipient 2 (occupation group) receives v2 = [r2, c2]
coalition = [v1, v2]; no recipient ever receives a pre-erasure feature or a clean (untreated) output
```

## Inputs, outputs, permissions

| Item | Value |
|---|---|
| Inputs | Adult permitted raw columns only: age, workclass, education, education-num, marital-status, relationship, capital-gain, capital-loss, hours-per-week, native-country. One-hot plus standardisation fitted on defense_train. Removed: sex, race, income, occupation, fnlwgt, ids (`DATA_ADMISSION.json`). |
| Recipient 1 | Purpose `income` (binary; >50K). |
| Recipient 2 | Purpose `occupation_group` (6 classes: the pinned loader's occupation grouping; not unemployment). |
| Primary protected attribute | SEX, forbidden to **both** recipients. Race is a secondary stress audit only. |
| Release (primary contract) | `v_i = [r_i, centred logits of h_i(r_i)]`. Centred logits are affine in `r_i`. |
| Secondary contract | Probabilities only, or the hard decision only, per recipient. |

## Model family and parameter counts

- **Encoders.** `g1`, `g2`: separate MLPs, `Linear(83,64)-ReLU-Linear(64,64)-ReLU-Linear(64,16)`, 10,576 parameters each.
  - Initialisation: PyTorch defaults under per-(seed, module) manual seeds.
  - They are optimised together; there is no shared backbone.
  - The backbone is trained, not "pretrained".
- **Heads.** Affine `Linear(16, K)` during training (34 and 102 parameters).
  - The deployed head is refitted at finalisation as StandardScaler + multinomial logistic regression (affine).
  - It is fitted on defense_train, with `C ∈ {0.01, 0.1, 1, 10, 100}` selected by defense_val log loss.
- **Critics** (training only, never released). Per view:
  - kind A, `MLP(dv,32,2)`;
  - kind B, `MLP(dv,64,64,2)`, interaction-capable;
  - the constant fitting-prior predictor.
- **Critic input.** Each critic's input is ZCA-whitened, with statistics from a fixed fitting-role reference subset of 4,096 rows. The statistics are refreshed before every encoder step and treated as constants within the step.
- **Critic optimisation.** Adam, learning rate 3e-3, 5 critic steps per encoder step.

## Training objective

**Surrogate.**
- For each view `v`: `R_v = 1 − min(CE_A, CE_B[, CE_B2], CE_const) / H_fit(S)`, computed on the step's fitting-role minibatch.
- `H_fit(S)` is the fitting-prior entropy. It is one common constant across views: it rescales β across attributes but does **not** reweight views.
- `R_v` is a training surrogate. It is not mutual information and not an AUC bound. It is nonnegative only on the rows the prior was fitted on.
- The privacy objectives are:
  - L: `P_L = R_v1 + R_v2`;
  - J and JP: `P_J = R_v1 + R_v2 + R_pair` (coalition/local ratio fixed at 1);
  - S12, S21: stage-wise subsets (see below).

**Encoder step** (plain SGD, learning rate 0.05, global-norm clip 5):
- Task direction `t = −∇(L_1 + L_2)`, applied to encoders and heads.
- Privacy direction `p = −β ∇_enc P`. Head weights are detached inside the views, so heads learn from the task only.
- **Proposed update (L, J, S12, S21).**
  - `d = Proj(p)` solves `min ½‖d − p‖²` s.t. `a_j·d ≤ 0` for every active guard.
  - `a_j = ∇_enc L_j` is computed on a 512-row minibatch of the fixed 4,096-row guard subset.
  - A guard is active when its guard-subset loss exceeds budget − 0.002, where budget = warm-start loss + 0.01 nats.
  - Because the encoders share no parameters, `a_1·a_2 = 0`, and the exact projection equals A-GEM applied per encoder.
  - Update: `u = (t_enc + d, t_head)`. A backtracking acceptance with step factors {1, ½, ¼, ⅛} accepts a step only if every guard loss ≤ max(budget, its previous value). If no factor is accepted, a task-only step is taken and recorded as a rejected protection step.
- **JP (matched penalty ablation).** `u = (t_enc + p, t_head)`, with no projection and no acceptance rule. This is the familiar weighted loss `L_1 + L_2 + βP_J`.

**LEACE.**
- The official concept-erasure LeaceFitter (defaults) is fitted on defense_train representations against one-hot SEX.
- It is refitted at the start of every protection epoch and frozen within the epoch; gradients pass through the fixed affine map.
- It is refitted once more after the last update, in float64, and that final map is the deployed transform.
- Heads and every attacker see only post-transform features.
- The final map is not optimised exactly by the training; differentiating through a frozen map does not do that.

## Arms (identical warm start, data order, critic initialisation, update counts)

**Shared schedule.**
- Warm start: 20 task-only epochs (Adam, learning rate 1e-3), shared bitwise by every neural arm of the same seed.
- Protection: 20 epochs, 76 steps each.

| Arm | Description |
|---|---|
| U | Task-only: 20 further task-only SGD epochs, no erasure, deployed head refit, centred outputs. |
| E | U's encoders + final official LEACE + refitted heads. |
| L | Local critics only: banks v1: {A, B, B2}, v2: {A, B, B2}. The coalition weight is 0, and an equal-budget extra local critic replaces the coalition critic's compute. |
| J | Local + coalition critics: banks v1: {A, B}, v2: {A, B}, pair: {A, B}. Projected update. |
| JP | Same critics as J, plain penalty update. |
| S12 / S21 | Neural sequential controls (not Taylor et al.'s algorithm). Stage 1: the first recipient's encoder with its local bank. Stage 2: the first encoder frozen, the second trained with its local bank + the pair bank. Each encoder gets 20 epochs, and critic steps match J. Both purposes are known in advance to every arm. |
| F | Official FARE (eth-sri pin), one tree per purpose on the permitted raw inputs, fitted on defense_train with the purpose label and SEX, using the admitted 6-configuration grid. The release is the one-hot cell plus centred logits of an affine head fitted on cells only. |
| F0 | Zero-fairness twin (γ = 0) at F's selected configuration per purpose. |

**β grid.** {0.1, 1, 10} for L, J, JP, S12 and S21. Seeds 0, 1, 2.

## Selection and comparisons

- **Rules.** See `PROTOCOL.md` §5. Inner roles only (attacker_fit → attacker_val); outer assessment is scored once, after `SELECTION_LOCK.json` is pushed.
- **Primary contrast: J vs L.** It holds information and compute fixed and isolates coalition coupling. Note that J also carries more penalty mass than L at the same β; the β grid lets L reach a higher strength.
- **Second contrast: J vs the validation-selected strongest feasible control.**

## Claims this prototype may and may not make

**May**, if the registered endpoints pass: on these reused development rows, against these fitted attackers, J lowered coalition SEX recovery relative to L or C* by ≥ 0.02 AUC while keeping both deployed tasks useful.

**May not:**
- population privacy, differential privacy or mutual-information bounds;
- protection of all attributes or of race;
- optimality or convergence;
- a new optimiser (the step is GEM/A-GEM);
- priority on collusion-aware release (Taylor et al. 2026; PCRL's own linear cross-purpose constraint);
- fresh confirmation.

**Deployment.** `python -m jcv.deploy --unit <selected unit> --X <permitted inputs>` (see `QUICKSTART.md`). The model is labelled EXPERIMENTAL unless the registered advantage is established.
