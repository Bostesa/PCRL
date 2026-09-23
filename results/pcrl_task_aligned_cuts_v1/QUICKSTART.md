# Task-aligned cuts: executable quickstart

This is a **2018 used-data development** experiment. The named trained release is the anchor-0 U1P1, delta-zero, one-round 324-cut `exchange_candidate`, marked `EXPERIMENTAL_NO_ADVANTAGE`. It is technically callable, but its independent inner-pilot task loss was worse than the matched same-bank deterministic MILP in both weightings, with no measured sensitive-recovery advantage. The round-zero 312-cut center is a separate checkpoint with the same experimental status. Three `BASELINE_CONTROL` modes are also packaged: unprotected D_U1, initial-bank MILP, and same-bank exchanged MILP. These are not relabeled PCRL improvements. The numerical fixed-bank integrality gap is not a population privacy guarantee. The closed 2016 prospective evaluation is not an input to these commands.

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

## Load and emit from the trained release

The five local mode directories, exact source/package digests, original frozen encoder archive version, and archive status are in [TRAINED_MODEL_MANIFEST.json](TRAINED_MODEL_MANIFEST.json). The private input schema is [RELEASE_INPUT_SCHEMA.json](RELEASE_INPUT_SCHEMA.json). Do not put person-level inputs, wire archives, fitted predictors, replay keys, or IDs in Git. This tested hash-verified load command identifies the named candidate without releasing a person:

```sh
/Users/nathansamson/PCRL/.venv/bin/python - <<'PY'
from experiments.pcrl_task_aligned_cuts_v1.release import ChannelArtifact
a = ChannelArtifact.load('results/pcrl_task_aligned_cuts_v1/private/packages/exchange_candidate/channel')
assert a.Q.shape == (32, 17) and a.model_status == 'EXPERIMENTAL_NO_ADVANTAGE'
print(a.model_status, a.Q.shape)
PY
```

Create a private `inputs.npz` with **only** `x_a` (`N×32` allowed PCA coordinates), `h_a` (`N×4` public A service coordinates), and unique `release_ids` (one-dimensional strings or integers). `emit` rejects other fields, including labels or `H_B`. It loads the 32×17 channel, verifies its bytes, checks the frozen encoder hash *before* deserialization, and emits only the original `h_a` bytes plus one integer token in `[0,16]`. Keep the replay key private, mode `0600`, and stable across restarts; keep IDs and input snapshot immutable. The mathematical channel samples once per person; the HMAC mode implements a stable computationally pseudorandom draw with negligible 64-bit discretization, not independent repeated releases or a composition guarantee.

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m experiments.pcrl_task_aligned_cuts_v1.release_cli key-init --output results/pcrl_task_aligned_cuts_v1/private/release_key.bin
/Users/nathansamson/PCRL/.venv/bin/python -m experiments.pcrl_task_aligned_cuts_v1.release_cli emit \
  --channel-dir results/pcrl_task_aligned_cuts_v1/private/packages/exchange_candidate/channel \
  --encoder /Users/nathansamson/PCRL/.worktrees/pcrl-final-prospective-v1/results/pcrl_final_prospective_v1/private/restore/results/pcrl_task_directed_release_v1/private/run/anchor_0/encoder/encoder.joblib \
  --input-npz results/pcrl_task_aligned_cuts_v1/private/inputs.npz \
  --key-file results/pcrl_task_aligned_cuts_v1/private/release_key.bin \
  --output-npz results/pcrl_task_aligned_cuts_v1/private/wire.npz
```

The CLI `key-init` and `emit` paths were exercised end to end on synthetic allowed inputs for **all five** packaged modes, in separate processes: repeated calls returned the same token and preserved `H_A` bytes. The private package receipt pinned by [TRAINED_MODEL_MANIFEST.json](TRAINED_MODEL_MANIFEST.json) records their hashes; `private/packages/SMOKE_RECEIPT.json` pins every synthetic smoke output. To select another mode, change only the last directory component of `--channel-dir` from `exchange_candidate` to `round_zero_center`, `baseline_D_U1`, `baseline_a0_MILP`, or `baseline_exchanged_MILP`. The command above for private ACS inputs is a documented callable path, **not** a claim that a new person-level release was run. The output path must be unused. Retain the key outside the public channel package. The recipient A uses `(H_A,Z)`; the coalition uses `(H_A,H_B,Z)`; B's historical service is unchanged. The hidden code, row of `Q`, seed, and unused draws never appear on the wire.

## Fit, audit, and reproduce the completed development unit

The registered [RUN_QUEUE.json](RUN_QUEUE.json) supplies the exact historical fit and audit argv and immutable unit IDs. The completed center chain is `a0_admit → a0_u1_decoder` and `a0_reference_bank → a0_u1p1_z000 → a0_u1p1_z000_audit`. Its fit entry point was `pipeline fit-arm --anchor 0 --arm U1P1 --budget 0`; its audit entry point was `pipeline audit-inner` with the same anchor, arm, and budget. The exchange and its matched deterministic re-solve have separate frozen sidecars and receipts. All independent inner audits fitted on `downstream_fit`, selected on `inner_selection`, and scored only `inner_pilot`; no outer 2018 assessment was opened. After the scientific cutoff, use this **receipt verification only** command with a deliberately expired deadline. It rehashes completed artifacts and cannot launch a missing fit:

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m experiments.pcrl_task_aligned_cuts_v1.scheduler \
  --root . --queue results/pcrl_task_aligned_cuts_v1/RUN_QUEUE.json \
  --state results/pcrl_task_aligned_cuts_v1/private/RECEIPT_REPLAY_STATUS.json \
  --workers 1 --deadline-utc 2026-09-23T00:00:00Z \
  --only-id a0_u1p1_z000 --only-id a0_u1p1_z000_audit
```

That exact expired-deadline command was tested after completion: it returned `status=selected_complete`, with 49 valid completed queue units and no new fit. [VALIDATION.md](VALIDATION.md) explains the exact expected-token scorer, independent replay fixtures, role routing, and supported/unsupported classes. The exchange, same-bank MILP, and their private inner audit receipts are pinned in [EXCHANGE_PILOT_SCREEN.json](EXCHANGE_PILOT_SCREEN.json), with aggregate results only. Fitted weights and person/household contributions remain private.

For a local numerical replay without refitting, the following command multiplies the stored fixed cost matrix by both frozen stochastic channels and checks their registered objectives. It also loads them through the hash-verified API. These are **fixed-bank training objectives**, not independent audit results:

```sh
/Users/nathansamson/PCRL/.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from experiments.pcrl_task_aligned_cuts_v1.release import ChannelArtifact
base = Path('results/pcrl_task_aligned_cuts_v1/private')
for relative, expected in (
    ('run/a0_u1p1_z000', 0.4678505964229197),
    ('exchange_r01', 0.4682873058695374),
):
    root = base / relative
    q = ChannelArtifact.load(root / 'channel').Q
    with np.load(root / ('cost.npz' if relative.startswith('run/') else 'coefficients.npz'), allow_pickle=False) as archive:
        value = float(np.sum(archive['cost'] * q))
    assert abs(value - expected) < 1e-10
    print(relative, value)
PY
```

For a complete byte-and-receipt replay, compare each private `COMPLETE.json` artifact hash against every file it lists; check the fitted bank/cost/channel hashes in the manifest; and verify `INNER_PANEL.json`'s `channel_sha256` against the contiguous float64 array digest of the loaded `Q`. `agents/release_math/package_channels.py` rebuilds or re-verifies all five packages only after checking both independent exchange audit receipts; its exact required receipt hashes are in `TRAINED_MODEL_MANIFEST.json`. The private source receipts and a verified new-study archive are needed for a full restored replay.

## Restore inputs and archive status

The pinned [REUSABLE_INPUTS_PINNED.json](REUSABLE_INPUTS_PINNED.json) records the exact historical 2018 object member names, SHA-256 values, S3 bucket/key/version IDs, and restore instructions. Restore only required members with the authorized `AWS_PROFILE=vein`; verify the pinned private manifest and archive-part hash before extracting, then verify each member SHA before deserialization. No new-study private archive had been verified at the time this quickstart was written; [TRAINED_MODEL_MANIFEST.json](TRAINED_MODEL_MANIFEST.json) points to retained local fitted objects and must gain a verified archive pointer from the execution/data owner before local originals are removed. Do not treat a historical input archive as a backup of this new fitted channel or its audit weights.
