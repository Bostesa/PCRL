# Team B numerical training design for the continuation

**Status: prefit proposal.** S1 must place exact values and source hashes in the effective campaign lock before any Adult fit. S3 must time the revised trainer first. This file does not itself admit a fit.

## Fixed proposal to smoke and lock

| Field | Proposed value | Reason |
|---|---:|---|
| Feature input | 83 ordered permitted columns | Same role-specific schema in every arm; S2 binds the manifest hash. |
| Encoders | two independent 83→64→64→(16 mean, 16 log variance) MLPs | Purpose-local acquisition; no cross-recipient runtime input. |
| Heads | affine 16→2 and 16→6 | Outputs depend only on each released feature draw. |
| Minibatch | 256 people | Covers the 15,434 defense-fit rows through a deterministic shuffled cycle. |
| Encoder/head optimizer | Adam, learning rate `0.001`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0` | Same optimizer and update count in matched arms. |
| Critic optimizer | Adam, learning rate `0.001`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0` | Separate optimizer state per local/pair critic. |
| Task CE coefficient | `1.0` | Mean across the two task losses. |
| Frozen teacher | `Ucal_1`, `Ucal_2` from S2-admitted role-only reference arrays | Uses the source's fixed global-temperature scores as soft targets on defense-fit people only. No old assessment values enter. |
| Frozen-teacher KL coefficient | `0.25` | Fixed confidence-retention pressure; every arm uses the identical Ucal teacher. |
| Gaussian KL coefficient | `0.001` | Fixed small rate penalty; do not interpret as a utility cap or SEX certificate. |
| Local SEX-critic coefficient | Joint stages: `0.05` times mean of two CEs; one-purpose LOCAL/SEQ stages: `0.025` times their one CE | Every active recipient gets the same effective local gradient coefficient `0.025`. |
| Pair SEX-critic coefficient | `0.05` | Multiplies pair CE with a negative sign in encoder loss. |
| Encoder steps per purpose | `T=600` provisional | Numeric value must survive S3 smoke and be fixed by S1 before fitting. No outcome-based extension. |
| Restarts/checkpoint | one start, exact terminal `T` | No best-epoch or best-seed selection. |
| Numerical threads | one per worker | Shared queue admits at most two heavy workers. |

The coefficients are a single scale-based design choice, not a tuned grid. A typical fitted KL of order 10 nats would contribute roughly 0.01 to the objective at `beta=0.001`; that is only a heuristic for the choice, not an observed Adult quantity or a guarantee. Actual KL values, task losses and critic losses must be reported. The common accuracy/NLL/Brier gates remain separate held-out development measurements against both U0 and Ucal.

S2's planned isolated exporter provides `U0_1`, `U0_2`, `Ucal_1`, `Ucal_2` for each allowed role after source hashes and exact row membership are verified. The trainer may read only the admitted defense-fit reference file. If Ucal fitting-row scores fail identity checks, the B fit is blocked until a prefit amendment selects a valid teacher; it must not silently substitute U0 or an all-role archive.

## Update schedule and matching

Each model seed fixes module initialization, minibatch permutations, Torch sampler streams and critic initialization by named deterministic sub-seeds; the same identifiers apply across arms. On each defense-fit minibatch:

1. Draw fresh code noise for stochastic arms. Update only the active SEX critics on detached codes. Local/pair critics receive the declared numeric training proxy: sampled `z`, raw head logits, `T=1` scores and decision one-hot. Protected labels are fitting-only.
2. Freeze critic parameters. Draw or reuse a predeclared independent code sample for the encoder step, compute mean task CE, mean teacher KL, mean analytic Gaussian KL, mean local critic CE and pair critic CE. Update the permitted encoder/head parameters by the signed objective in `METHOD_SPEC.md`. Zero gradients and restore critic trainability between phases.
3. Save a checkpoint/receipt at the fixed terminal step. Intermediate diagnostics may be logged but cannot become candidates.

`B-JVIB`, `B-DVIB`, and `B-RAWJ`: each of two local critics gets `T` updates and pair critic gets `2T`; each encoder/head gets `T`. Their local loss is the mean of two CE terms, weighted `0.05`, so each view contributes `0.025`. `B-LOCAL`: each single-purpose stage uses local coefficient `0.025`; each local critic gets `2T`, with no pair objective, and each encoder/head gets `T`. `B-SEQ12` and `B-SEQ21`: each single-purpose stage also uses local coefficient `0.025`; first purpose gets `T` encoder and `T` local-critic updates; freeze its encoder/head and its fixed first release; second purpose gets `T` encoder, `T` local-critic and `2T` pair-critic updates at pair coefficient `0.05`. Thus each joint/sequential arm receives `2T` encoder updates and `4T` total training-critic updates. LOCAL gives its local critics more refreshes than joint/SEQ because it has no pair critic; this is matched total critic compute, not an identical adversary trajectory. `B-TASK` has `T` encoder updates per purpose and no SEX critic, retaining stochastic KL. `B-RAW-TASK` is its deterministic ordinary task-only counterpart with `T` encoder updates per purpose, no KL and no SEX critic. Every required arm gets the same calibration and fresh attacker slate afterward.

The first-stage sequential release must be frozen as a one-draw code under the same release contract before training the second stage. Do not let its code resample to make the second stage easier. If exact first-stage identity permits a cache alias across sequential and local arms, record source/code/seed/optimizer/data-order/view hashes before sharing; otherwise charge a separate fit.

## Calibration, attacks and conclusions

Freeze each fitted model before drawing one code for each held-out calibration person and fitting one positive global temperature per task. Release and score the actual draw through `SavedRelease`. Fresh attack fit and model selection use a deterministic group-aware split **inside ATTACK_FIT_NEW**. Freeze every reader and binary orientation before INNER_SELECTION; primary AUC there is the maximum over the fixed slate, with attack-validation winner also reported. Readers of the actual complete view get `z`, raw logits, calibrated probabilities, decisions and all metadata. The training critics are only optimization surrogates.

The ideal stochastic Gaussian KL bound is about the model law with conditionally independent draws. A fixed HMAC-keyed saved batch is a computational implementation and carries no exact rate or SEX-privacy certificate. `B-DVIB`'s KL acts as mean-code regularization, not an information bound. A measured gain must survive matched deterministic, task-only, local, both sequential and ordinary raw-joint controls. All numerical endpoints remain exploratory on reused development roles.
