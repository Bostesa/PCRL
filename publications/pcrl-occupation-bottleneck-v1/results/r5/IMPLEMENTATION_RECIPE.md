# Conditional occupation repair implementation

Status: implemented for source review; no pilot fit or numerical result is
claimed. ROOT decides whether FIT/ATT diagnosis triggers this recipe before
INNER assessment. This is exploratory development on repeatedly used Adult
development people.

## Fixed mechanism and sign

Start from the admitted RAW_JOINT strength 0.3, epoch 5 checkpoint for each
source seed. Preserve the actual raw83 feature schema, float32 encoders,
float64 own heads, and class order. Freeze every income encoder/head tensor;
compare these tensors bitwise after every critic and model update.

On occupation encoder/head parameters only, let g_CE be the gradient of mean
occupation cross entropy. The full-wire sensitive penalty for a view is
`1 - min(CE_linear, CE_mlp32, CE_mlp64, H_SEX) / H_SEX`, minimized by the
occupation model. This increases the strongest critic's CE up to the entropy
cap. The full joint penalty sums occupation-local and aligned-pair penalties.
The frozen income wire is still present in the pair input, so the pair critic
can use or ignore it. Critic fitting minimizes its own CE.

The proposed actual SGD displacement is
`p = -0.01 * (g_CE + lambda * g_sensitive)`, without momentum, weight decay,
or preconditioning. In constrained arms, solve
`min_d 0.5 ||d-p||^2` subject to `g_NLL dot d <= 0`,
`g_Brier dot d <= 0`, and `||d|| <= 0.01`. Homogeneous halfspaces have zero
as a feasible witness. The implementation enumerates the active sets of up
to two normalized nonzero task gradients, solves their least-squares KKT
equations, and then clips radially to the norm ball. Tiny nonzero guards are
normalized after max-absolute scaling rather than discarded.

After assigning the displacement to float32 encoder and float64 head
parameters, read the actual rounded displacement back. Require its norm and
normalized guard products to pass tolerance `1e-10`. Recompute real NLL
(`1e-12` probability floor) and summed multiclass Brier on the same FIT-update
batch; each must be no greater than baseline plus `1e-10`. Try fixed fractions
`1, 0.5, 0.25, 0.125`, then restore the exact original parameters on rejection.
Record all attempts, active constraints, rounding stalls, zero fallbacks,
accepted fractions, and actual norms. This controls the measured update batch
only; it supplies no held-out utility or privacy guarantee.

## Matched controls and exposure

Run all four arms with identical initial model, critic initialization, random
draw stream, critic schedule, model proposal count, checkpoint budget and
training exposure. CONSTRAINED uses both privacy terms and the safeguard.
ORDINARY uses both terms with the same bounded SGD and no task safeguard.
LOCAL_TWIN uses the safeguard and removes only the pair term from the model
objective. TASK_ONLY disables the privacy model gradient. Both critic banks
still train in all four arms. The unchanged RAW checkpoint is a fifth control.

Six critics receive the actual occupation full29 wire or aligned full50 pair:
linear, one-hidden-layer MLP32, and two-hidden-layer MLP64 for each view.
Critic Adam uses learning rate `0.003`, 80 initial updates per critic and five
updates per critic before each model proposal. The model recipe is 20 epochs,
batch256 uniform draws without replacement within each draw from inherited
FIT folds 0/1, and `ceil(update_rows/256)` proposals per epoch. Checkpoints are
0/5/10/20. FIT fold2 is scored only for internal selection.

A deterministic fixed 256-row FIT-update reference is refreshed initially
and at each epoch, using each arm's current wires. Means and scales are
detached, finite, and all-coordinate scales strictly positive. Per-coordinate
scale is at least `max(initial_std*1e-8, 1e-12)`.

Derive one lambda per source seed from 16 deterministic initial draws after
critic warmup: `3 * median(||g_CE||) / median(||g_local+g_pair||)`, measured on
occupation encoder/head parameters only. The same lambda applies to joint,
ordinary and local twin. Both median norms must be finite and greater than
`1e-12`, and lambda must be at most `1e4`. An unavailable normalization aborts
the recipe; no substitute coefficient, strength grid or outcome-driven
threshold change is permitted. The capped penalty can produce zero gradients;
this is a real failure condition. Preserve normalization tensors privately.

`0.01` SGD and the `0.01` displacement bound are fixed proposed magnitudes.
They are not asserted to equal a historical `0.0001` Adam movement. Proposal
and accepted norms must support interpretation of ordinary continuation's
strength and the safeguard's stalls. Equal control access does not establish
that this fixed adversary approximates the strongest post-training reader.

## Code and invocation API

`occupation_bottleneck.pilot.RECIPE` defines the one recipe;
`validate_recipe` rejects modifications. `displacement_api` lazily exports
R2's `project_displacement` and `finite_step`. `pilot_worker.run(spec_path)`
accepts a JSON spec with:

- `operation: occupation_bounded_pilot`, `seed: 0` or `1`;
- exact `recipe` object; role-safe `manifest` and `manifest_sha256`;
- `checkpoint: {path, sha256}`; fresh `private_output`; aggregate `output`;
- `trigger_lock: {path, sha256}` binding ROOT's decision.

The trigger JSON requires owner ROOT, status PILOT_TRIGGERED,
INNER_used_for_trigger false, diagnosis_sha256, exact recipe_sha256, and
starting_checkpoints with both source-seed pins. The worker checks the active
protection-stage queue before importing numerical libraries, opening arrays,
loading checkpoints, fitting critics, and applying model updates. Invocation
is `python -m occupation_bottleneck.pilot_worker --spec <private-spec.json>`
inside the sole ROOT-owned queue. Direct invocation fails closed.

Checkpoints preserve `TCR1-trainable-own-head-checkpoint-v1`, model tensors,
feature_names, class_orders, and source_bindings. Pilot metadata and critic
states are additional fields. Existing load_checkpoint, actual_wires, reader
and deployment paths can consume them without predecessor source edits.
Runtime consumes raw83 and each actual own encoder/head only; no teacher,
labels or metadata enter either recipient wire.

The worker produces private gradient normalization tensors, exposure receipt,
epoch references, per-step logs, checkpoints, FIT-fold2 metric receipts and
an aggregate training summary. It does not fit ATT readers, nominate an epoch,
open INNER, or claim a win. ROOT freezes a common per-arm configuration using
FIT-fold2 and matched ATT evidence before reporting admission.

Verification to date: source AST syntax parsing only. All numerical solver,
checkpoint roundtrip, frozen-income, controls and training checks require
ROOT-authorized queue work and independent replay before a pilot result.
