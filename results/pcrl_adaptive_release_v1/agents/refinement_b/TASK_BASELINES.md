# Task-only and matched partition baselines

`experiments.pcrl_adaptive_release_v1.task_baselines` provides independent
2018-development controls for Branch B. This module has not fitted ACS data
from this agent. It reads no 2016 row and does not open the outer assessment.

The task-only model is a fixed, PWGTP-weighted logistic residence predictor
trained on the global `nuisance_train` households using exactly the allowed
`X_A32,H_A4` inputs. Its 16 weighted training-prediction quantile cuts map
the score deterministically to one of the existing 17 token labels. All
training labels, scaling constants and cuts are pinned in a private model
receipt. `TaskOnlyPredictor.release(RuntimeInputs)` emits only copied H_A and
one token. The continuous score is an allowed-input task-headroom diagnostic;
the 17-token release is the comparable wire mechanism. Neither uses a
person's task or protected label at deployment.
The private task-only unit has an immutable `COMPLETE.json` file inventory
for archive read-back and rejects altered artifacts on resume.

The task-only token can differ between people with the same historical T32
code. Audit it using `task_only_control_law(model, inputs, mode=...,
publish=...)`, which returns a **private** original-person × 17 exact token
law. It must not be represented as a T32×17 channel or handed to a recipient.
`mode` is `unmodified`, `constant_replacement`, or `randomized_response`.
Constant replacement emits existing token 0 by default when withheld;
randomized response uses a uniform existing-token draw. Rates `.50,.75,.90,1`
can be passed directly, and every distinct law needs the common independent
audit slate through a per-person-law audit adapter. A sampled implementation
must maintain one persistent token per record for its stated release; exact
law scoring is an evaluation method.

`TaskOnlyReleaseSession` implements the sampled wire for these modes. It
requires a private 32-byte-or-longer replay key, an immutable `release_id`,
and only `RuntimeInputs(X_A,H_A)` plus caller-held record IDs. It HMAC-samples
the registered private law and returns exactly `{h_a, token}`; H_A bytes are
preserved. Repeating the same ID and inputs reuses the token, including
across restarts with the same key. A private `cache_path` additionally binds
each ID to its input digest across restarts and rejects a changed X_A/H_A;
without that path, changed-input rejection applies within the current
session. The cache holds keyed ID digests, input hashes and tokens, with no
raw records or secret key. It is an implementation of one persistent release,
not a repeated-independent-query privacy guarantee.

```python
from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
from experiments.pcrl_adaptive_release_v1.task_baselines import (
    TaskOnlyReleaseSession, load_task_only)
model, receipt = load_task_only("/private/study/task_only_anchor0")
session = TaskOnlyReleaseSession(
    model, mode="randomized_response", publish=.75,
    replay_key=private_key_bytes, release_id="development-release-1",
    cache_path="/private/study/task_only_anchor0_replay.json")
wire = session.emit(RuntimeInputs(x_a, h_a), record_ids)
```

`select_partition_control` returns a nested T32 child partition for each
registered policy. All three consider the same allowed frozen H_A,
residual/posterior, SEX2 and full-RAC1P9 risk features, training-only
quantile thresholds, and fixed 100-unique/100-effective-household child
floor. `task_only` ranks by the fixed decoder's exactly weighted task-cost
freedom score with no privacy dual price. `joint_risk` ranks by the frozen
11-dimensional nuisance-score between-child variance reduction and never
uses a protected outcome on split rows; “nonadaptive” means independent of
the candidate's dual prices and task losses, although its nested partition
is built sequentially. `random_eligible` samples from the same eligible
feature/threshold/support set with a pinned seed. Each uses
`inner_selection` for a separately recorded stability check where the
policy has a score; none reads `inner_check`. Zero-gain supported splits may
fill a matched leaf budget. A support-limited policy keeps its smaller
realized leaf count and reports that mismatch rather than lowering the
floor.

`task_contributions(frozen_decoder, rows)` produces the original-person × 17
task loss contributions with half U and half PWGTP weighting, with joint
state mass included once. `fit_partition_controls_from_roles(...)` builds
all three policies, saves JSON routing rules and an immutable source/hash
inventory, and resumes only the same exact unit. The central queue gives
the resulting partition to the same finite stochastic, deterministic and
gradient controls before common independent audit. Copying a 32-row Q to
children with `refinement.copy_parent_kernel` is the exact parent witness.

Queue-friendly CLI on the pinned Linux x86 environment:

```sh
python -m experiments.pcrl_adaptive_release_v1.task_baselines fit-task \
  --anchor 0 --index-path results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json \
  --output-dir PRIVATE_TASK_MODEL --seed 20260924
python -m experiments.pcrl_adaptive_release_v1.task_baselines fit-partitions \
  --anchor 0 --delta 0.001 \
  --index-path results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json \
  --a-center-dir PRIVATE_A_CENTER --b-center-dir PRIVATE_B_CENTER \
  --output-dir PRIVATE_PARTITION_CONTROLS --seed 20260924
```

The second command verifies the completed A/B receipts, reuses B's frozen
nuisance model and A's selected decoder, and targets B's realized leaf count.
Its saved partitions are **inputs** to matched channel fits, not an audit or
evidence that the adaptive split policy improves the measured frontier.
