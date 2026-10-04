# Critic-gap diagnosis (registered; inner roles only; per-unit values in `CRITIC_GAP.csv`)

## Design (fixed before fits, after review R1/R2)

**Snapshot.** For every PN and LN unit (9 + 9) and every critic view (v1 and v2, plus the pair for PN), the diagnosis evaluates the state **θ_{T−1}**: the model at the last critic update.

**Inputs held fixed.**
- The views are the training-time views (encoder + training head, identity map).
- The input transform is the **saved** ZCA whitener.
- Every predictor is scored with SEX cross-entropy on the **same attacker_val rows**.

**Predictors compared.**

| Predictor | What it is |
|---|---|
| Online | The saved end-of-training critics |
| fresh_def | The same architectures, re-initialised and trained on the same defense_train rows of the frozen view (Adam 3e-3, batch 256, early stopping on a fixed 20% split) |
| fresh_att | The same protocol on attacker_fit rows |
| Slate | The sklearn inner slate on attacker_fit (secondary, optimistic) |

**Primary statistic.** The mean over the bank of paired CE(online critic j) − CE(fresh_def critic j). A positive value means a fresh critic of the same kind reads more SEX information from the identical frozen view than the online critic did.

## Results (seed means; the fitted-prior CE is 0.629 in every row)

| Arm, β | View | Online CE | fresh_def CE | Primary gap (range over seeds) | Sensitivity at θ_T, refitted whitener |
|---|---|---|---|---|---|
| PN 0.1 | v1 / v2 / pair | 0.551 / 0.457 / 0.451 | 0.518 / 0.444 / 0.422 | +0.027 / +0.014 / +0.022 | +0.028 / +0.017 / +0.027 |
| PN 1 | v1 / v2 / pair | 0.614 / 0.614 / 0.601 | 0.524 / 0.543 / 0.484 | **+0.067 / +0.048 / +0.091** | +0.069 / +0.041 / +0.094 |
| PN 10 | v1 / v2 / pair | 0.617 / 0.613 / 0.609 | 0.547 / 0.536 / 0.504 | **+0.048 / +0.051 / +0.081** | +0.037 / +0.052 / +0.088 |
| LN 0.1 | v1 / v2 | 0.538 / 0.436 | 0.515 / 0.422 | +0.018 / +0.010 | +0.025 / +0.012 |
| LN 1 | v1 / v2 | 0.606 / 0.609 | 0.506 / 0.546 | **+0.079 / +0.049** | +0.085 / +0.047 |
| LN 10 | v1 / v2 | 0.619 / 0.614 | 0.556 / 0.540 | **+0.046 / +0.055** | +0.048 / +0.058 |

- **The primary gap is positive for 45 of 45 view × unit cells**, mean **+0.047 nats**. This matches registered prediction 6.
- The sensitivity rows (θ_T with a refitted whitener) agree within about 0.01.

## Interpretation

**Yes: the online critics were genuinely weaker on the same metric.**

**At β ≥ 1 the training critics sat near the prior.** They reached CE 0.60–0.62 against the prior's 0.63, which corresponds to a training-surrogate R of about 0.01–0.05. Fresh critics of identical architecture, trained on the same rows of the same frozen views, reach 0.48–0.56. On those frozen views at least 0.05–0.09 nats of readable SEX information remained. The encoders mostly made SEX **hard for the current online critics to read**; they did not remove it.

**At β = 0.1 the gap is smaller** (+0.01 to +0.03), and the online critics read much of what was there.

**Consistent with the outer audit.** At β ≥ 1, fresh outer attackers still recover SEX at coalition AUC 0.72–0.79.

## What this supplies (prospective; no new grid was run)

The measured bottleneck is critic tracking under the penalty update at β ≥ 1: the encoder outpaces 5 Adam steps per encoder step, through a floored ZCA whitener.

Candidate next changes, each for a new registered protocol on new data, kept out of this erasure ablation:
- periodic critic re-initialisation or refitting to convergence on frozen snapshots ("fresh-critic restarts");
- more critic steps or a two-timescale schedule;
- removing the 10⁴× whitener floor amplification, for example by projecting out null directions instead of flooring them;
- auditing with refit critics during selection.
