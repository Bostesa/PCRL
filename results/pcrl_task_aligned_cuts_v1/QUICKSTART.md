# Task-aligned cuts: executable quickstart

This is a **2018 development** experiment. The trained anchor-0 U1P1, delta-zero channel is an `EXPERIMENTAL_UNVALIDATED` checkpoint until the selected panel and independent control comparisons are complete. It is technically callable; its initial 312-cut numerical certificate is not a population privacy guarantee or evidence of an improved independently measured tradeoff. The closed 2016 prospective evaluation is not an input to these commands.

Run from the `research/pcrl-task-aligned-cuts-v1` worktree root. The Python path used by the completed local scientific queue is `/Users/nathansamson/PCRL/.venv/bin/python`. Frozen 2018 T0 codes were produced on Linux x86 and loaded from the pinned archive; re-encoding them on Mac is not an equivalent scientific replay. For deployment with the historical encoder, use a verified compatible Linux x86 environment.

## Public fit-to-release-to-score smoke

This entry point was executed and checked in a fresh private directory. It constructs a public toy 32×17 channel and encoder, emits one token per synthetic person, checks byte-identical `H_A`, replays the keyed draw across fresh sessions, and enumerates all 17 tokens for exact expected loss. Choose a new output directory each time:

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m experiments.pcrl_task_aligned_cuts_v1.release_cli synthetic --output-dir results/pcrl_task_aligned_cuts_v1/private/synthetic_demo_new
```

The checked synthetic output had three people, wire fields exactly `h_a,token`, and exact expected task losses `U=0.6974134620115667`, `PWGTP=0.8086663982261932` nats. [SYNTHETIC_DEMO.json](SYNTHETIC_DEMO.json) contains only aggregate toy results. The fixture encoder and its publicly known key are **not** an ACS model or a deployment secret. A separate tested synthetic LP-to-release-to-audit fixture is available with:

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m experiments.pcrl_task_aligned_cuts_v1.audit synthetic
```

## Load and emit from the trained checkpoint

The local checkpoint, exact file digests, original frozen encoder archive version, and archive status are in [TRAINED_MODEL_MANIFEST.json](TRAINED_MODEL_MANIFEST.json). The private input schema is [RELEASE_INPUT_SCHEMA.json](RELEASE_INPUT_SCHEMA.json). Do not put person-level inputs, wire archives, fitted predictors, replay keys, or IDs in Git.

Create a private `inputs.npz` with **only** `x_a` (`N×32` allowed PCA coordinates), `h_a` (`N×4` public A service coordinates), and unique `release_ids` (one-dimensional strings or integers). `emit` rejects other fields, including labels or `H_B`. It loads the 32×17 channel, verifies its bytes, checks the frozen encoder hash *before* deserialization, and emits only the original `h_a` bytes plus one integer token in `[0,16]`. Keep the replay key private, mode `0600`, and stable across restarts; keep IDs and input snapshot immutable. The mathematical channel samples once per person; the HMAC mode implements a stable computationally pseudorandom draw with negligible 64-bit discretization, not independent repeated releases or a composition guarantee.

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m experiments.pcrl_task_aligned_cuts_v1.release_cli key-init --output results/pcrl_task_aligned_cuts_v1/private/release_key.bin
/Users/nathansamson/PCRL/.venv/bin/python -m experiments.pcrl_task_aligned_cuts_v1.release_cli emit \
  --channel-dir results/pcrl_task_aligned_cuts_v1/private/run/a0_u1p1_z000/channel \
  --encoder /Users/nathansamson/PCRL/.worktrees/pcrl-final-prospective-v1/results/pcrl_final_prospective_v1/private/restore/results/pcrl_task_directed_release_v1/private/run/anchor_0/encoder/encoder.joblib \
  --input-npz results/pcrl_task_aligned_cuts_v1/private/inputs.npz \
  --key-file results/pcrl_task_aligned_cuts_v1/private/release_key.bin \
  --output-npz results/pcrl_task_aligned_cuts_v1/private/wire.npz
```

The CLI `key-init` and `emit` paths were exercised end to end on synthetic inputs in separate processes: repeated calls returned the same token and preserved `H_A` bytes. The command above for private ACS inputs is a documented callable path, **not** a claim that a new person-level release was run. The output path must be unused. Retain the key outside the public channel package. The recipient A uses `(H_A,Z)`; the coalition uses `(H_A,H_B,Z)`; B's historical service is unchanged. The hidden code, row of `Q`, seed, and unused draws never appear on the wire.

## Fit, audit, and reproduce the completed development unit

The registered [RUN_QUEUE.json](RUN_QUEUE.json) supplies the exact fit and audit argv and immutable unit IDs. Its scheduler verifies each existing `COMPLETE.json` and skips valid units, so use the queue to **resume/verify**, not to refit a completed result. The center chain is `a0_admit → a0_u1_decoder` and `a0_reference_bank → a0_u1p1_z000 → a0_u1p1_z000_audit`. The exact fit entry point is `pipeline fit-arm --anchor 0 --arm U1P1 --budget 0`; the audit entry point is `pipeline audit-inner` with the same anchor, arm, and budget. These completed units used `downstream_fit` for independent predictors, `inner_selection` for model selection, and `inner_pilot` for descriptive scoring; no outer 2018 assessment was opened by this checkpoint. The original 2018 pools were used upstream, so this is development evidence. The completed fit/audit commands can be verified through the scheduler by selecting their IDs:

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m experiments.pcrl_task_aligned_cuts_v1.scheduler \
  --root . --queue results/pcrl_task_aligned_cuts_v1/RUN_QUEUE.json \
  --state results/pcrl_task_aligned_cuts_v1/private/QUEUE_STATUS.json \
  --workers 1 --deadline-utc 2026-09-23T17:00:00Z \
  --only-id a0_u1p1_z000 --only-id a0_u1p1_z000_audit
```

The scheduler command is for this registered run and its fixed deadline; after that timestamp, verify the stored receipt directly rather than extending the scientific deadline. [VALIDATION.md](VALIDATION.md) explains the exact expected-token scorer, independent replay fixtures, role routing, and supported/unsupported classes. The private audit output is `private/run/a0_u1p1_z000_audit/INNER_PANEL.json`; the person and household contributions plus fitted audit models remain private. Public aggregate results and any final decision must be generated from verified machine-readable records, not from this checkpoint alone.

For a local numerical replay without refitting, the following command independently multiplies the stored fixed cost matrix by the stored channel and checks the registered center objective (approximately `0.467850596423` nats). It also loads the channel through its hash-verified API. This is the **fixed-bank training objective**, not an independent audit result:

```sh
/Users/nathansamson/PCRL/.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from experiments.pcrl_task_aligned_cuts_v1.release import ChannelArtifact
root = Path('results/pcrl_task_aligned_cuts_v1/private/run/a0_u1p1_z000')
q = ChannelArtifact.load(root / 'channel').Q
with np.load(root / 'cost.npz', allow_pickle=False) as archive:
    value = float(np.sum(archive['cost'] * q))
assert abs(value - 0.4678505964229197) < 1e-10
print(value)
PY
```

For a complete byte-and-receipt replay, compare `private/run/a0_u1p1_z000/COMPLETE.json` artifact hashes against every file it lists; check `FIT_P1_ARM.json` against the bank/cost/channel hashes in the manifest; and verify `INNER_PANEL.json`'s `channel_sha256` against the contiguous float64 array digest of the loaded `Q`. The private source receipts and a new-study verified archive are needed for a full restored replay.

## Restore inputs and archive status

The pinned [REUSABLE_INPUTS_PINNED.json](REUSABLE_INPUTS_PINNED.json) records the exact historical 2018 object member names, SHA-256 values, S3 bucket/key/version IDs, and restore instructions. Restore only required members with the authorized `AWS_PROFILE=vein`; verify the pinned private manifest and archive-part hash before extracting, then verify each member SHA before deserialization. No new-study private archive had been verified at the time this quickstart was written; [TRAINED_MODEL_MANIFEST.json](TRAINED_MODEL_MANIFEST.json) points to retained local fitted objects and must gain a verified archive pointer from the execution/data owner before local originals are removed. Do not treat a historical input archive as a backup of this new fitted channel or its audit weights.
