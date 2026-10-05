# Method card: online strength frontier (osf)

## Methods

| Method | Encoder update per step | Status in this study |
|---|---|---|
| U | Task gradient only | Admitted from smf (`tl__s{k}__e40`) |
| RAW-J β | Task gradient + β ∇[(R_income + R_occupation + R_pair)/3] | β 0.1 and 0.3 admitted (`rgj.train` J-O, unchanged); β 0.6 new |
| RAW-L β | Task gradient + β ∇[(R_income + R_occupation)/2]; the pair bank is a trained shadow | Same as RAW-J |
| NORM-J ρ, a | Task gradient + q_i per encoder, with q_i = ρ s_i(a) ‖t_i‖/‖p_i‖ · p_i and p_i = ∇_enc_i of the joint proxy | New |
| NORM-L ρ, a | The same, with the local proxy | New |

## Fixed components

| Component | Setting |
|---|---|
| Critics | ONLINE only: per-step floored ZCA on 4,096 CRITIC_FIT rows, 5 Adam steps per critic per encoder step, kinds A and B per view, fixed warm-start critic-view head |
| Surrogate | R_v = (CE_const − min(CE_const, CE_A, CE_B)) / H on the minibatch |
| Optimiser | SGD 0.05, global clip 5, batch 256, 40 epochs from the admitted warm start, salt 0 |
| Release | [features, centred logits of the deployed head]; deployed head = StandardScaler + LogisticRegression, fitted on OSF_DEFENSE_FIT, with C chosen on HEAD_VALIDATION |

The allocation is s_income = a/√((a²+1)/2) and s_occupation = 1/√((a²+1)/2), with a ∈ {0.5, 1, 2}. The RMS of the declared per-encoder ratios is ρ.

## What it is not

The method has none of the following:
- dynamic feedback or controller;
- critic refits or restarts;
- learned budget, Lagrange multiplier or gradient projection;
- erasure map in any trained arm.

The surrogate is not mutual information, a secrecy certificate or a privacy guarantee. Norm rescaling and a fixed coefficient bank are not claimed as algorithmic novelty. Adversarial representation learning follows Madras et al., ICML 2018.

## Deployment contract

**Input.** The input is exactly the 83 permitted columns in admitted order:
- the 5 numeric columns standardised with the OSF_DEFENSE_FIT statistics in `ROLE_MANIFEST.json`;
- the 78 one-hot columns over fixed category sets, where an unseen category is an all-zero block.

It needs no SEX, race, income, occupation, fnlwgt or identifiers, and no label.

**Output, per recipient:** features r_i (16 dimensions), centred logits, probabilities and hard decision.
