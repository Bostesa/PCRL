# Adaptive PCRL quickstart

This is a **2018 used-data development** study. The trained T32 candidates are 17-token releases beside unchanged `H_A`; their hidden state, token law, and private replay key are not wire fields. The verdict is **`EXPERIMENTAL_NO_ADVANTAGE`**: neither registered route passed the locked independent assessment. The synthetic command below proves the fit/release/exact-score software loop without ACS data. It does not establish a population guarantee or untouched confirmation result. The private trained-model restore pointers are now hash-pinned in `MODEL_MANIFEST.json` and `ARCHIVE_INDEX.json`; D17 remains a named baseline.

Run from the `research/pcrl-adaptive-release-v1` worktree root with the pinned compatible Python environment. These CLI flags were checked against the executable modules.

## Restore and emit the locked trained T32 release

`MODEL_MANIFEST.json` gives each anchor's exact encrypted S3 key, version, archive SHA, selected Q file SHA, and frozen encoder SHA. Restore the A center plus the mode unit (`A_center_d001` for `U`, `privacy_first_d001` for `P`, or `A_controls_d001` for baseline `D17`) into an owner-only `private` directory on compatible Linux x86. Check the tarball SHA before extraction and the archived per-file manifest afterward. Restore the frozen encoder by the exact member/version pins in `results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json`, using the pinned Linux x86 environment in `ARCHIVE_INDEX.json`. Confirm the restored encoder reproduces stored T0 codes on an allowed-input fixture before claiming historical bitwise parity. The current model is already trained; these steps restore bytes, not refit or select it.

The actual emitter command is:

```sh
python -m experiments.pcrl_adaptive_release_v1.artifact_cli emit \
  --model-manifest results/pcrl_adaptive_release_v1/MODEL_MANIFEST.json \
  --manifest-sha256 54121dfddbff85787dc79e7aafcce12e65b8fdd1ce07fd2dd3eeb845c0370a91 \
  --selection-lock results/pcrl_adaptive_release_v1/SELECTION_LOCK.json \
  --lock-sha256 4f894f98af67fd2a79e35ad3697971041aa53451b292b8bdf9691c8dd761f484 \
  --anchor 0 --mode U \
  --a-center-dir "$PRIVATE/a0_A_center_d001" \
  --artifact-dir "$PRIVATE/a0_A_center_d001" \
  --encoder "$PRIVATE/encoder/encoder.joblib" \
  --input-npz "$PRIVATE/allowed_inputs.npz" \
  --replay-key "$PRIVATE/replay.key" \
  --wire-out "$PRIVATE/wire_U.npz"
```

Set `PRIVATE` to the restored owner-only directory. The input archive has exactly `x_a` (`N×32`), unchanged `h_a` (`N×4`), and unique stable `release_ids`; see `INPUT_SCHEMA.md`. Create a private random 32-byte-or-longer key with owner-only file permissions using `python -m experiments.pcrl_task_aligned_cuts_v1.release_cli key-init --output "$PRIVATE/replay.key"`. The wire archive has only `h_a` and integer `token`; the receipt labels `U/P` as `EXPERIMENTAL_NO_ADVANTAGE`. For the privacy-first diagnostic set `--mode P`, use its same-anchor `privacy_first_d001` as `--artifact-dir`, and choose a different wire path. For the baseline set `--mode D17`, use `A_controls_d001`; its receipt says `BASELINE_CONTROL`, never a new PCRL improvement. Repeating a release for the same immutable ID/input/channel and private key replays the same token. The production ACS emit was not rerun after cloud shutdown; the synthetic CLI path and archived channels were tested independently.

## Public synthetic fit → release → exact audit

Choose a new output directory. The study-owned `private` path component is required because fitted toy weights and original-person fixture values are saved for replay.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m experiments.pcrl_adaptive_release_v1.demo \
  --output-dir /tmp/pcrl-adaptive-demo/private/run1
```

This command fits a calibrated A center and B center on five disjoint synthetic household roles, loads the saved B channel, emits exactly `{h_a, token}`, verifies byte-identical `H_A`, replays the same token in a fresh keyed session, rejects changed inputs in a live session, and scores a fitted task decoder by enumerating all 17 possible tokens. `SYNTHETIC_DEMO.json` contains aggregate toy results only. The small fixture cannot meet the registered 100-household-per-child floor, so B normally records an explicit A alias. The fixture's public key is **never** a deployment key. Rerunning the same command verifies immutable fit receipts and the same summary rather than refitting a new scientific unit.

The executed CLI smoke used 20 synthetic check people and reported `SUPPORT_LIMITED_ALIAS_A`, exact task CE `U=0.5815649052732799` and `PWGTP=0.6083083556850262` nats, with a `0.0` maximum difference from direct 17-token enumeration. These toy values have no ACS or comparative interpretation.

`python -m pytest -q tests/pcrl_adaptive_release_v1/test_demo.py` checks both the callable loop and CLI. The exact expected-loss check compares `audit.expected_token_loss` with a separate person/token sum to machine precision. It is a fixed-decoder smoke, not the independent multi-role ACS attacker/probe audit.

## Fit and verify a private 2018 development unit

The coordinator's registered queue supplies the already pinned input index and output unit directories. Run these only for a declared missing unit on the verified Linux x86 host; resume complete units through their immutable receipts. `fit_a` uses `--index`, while `fit_b` uses `--index-path` and an explicit `fit` subcommand.

```sh
python -m experiments.pcrl_adaptive_release_v1.fit_a \
  --anchor 0 --delta 0.001 --index "$INDEX" \
  --output-dir "$A_CENTER" --max-rounds 6
python -m experiments.pcrl_adaptive_release_v1.fit_b fit \
  --anchor 0 --delta 0.001 --index-path "$INDEX" \
  --a-center-dir "$A_CENTER" --output-dir "$B_CENTER" \
  --max-states 64 --max-rounds 6
python -m experiments.pcrl_adaptive_release_v1.fit_b parity \
  --anchor 0 --index-path "$INDEX" --center-dir "$B_CENTER" \
  --receipt-path "$B_PARITY"
```

The registered queue may specify a different bounded round count or state limit; use its immutable unit definition. The B parity command separately compares archived Linux T0/child routing with the live frozen encoder and checks service bytes. A Mac float32 re-encoding cannot substitute for the archived Linux codes. A B fit without a verified Linux x86 parity receipt is not a deployable runtime.

## Emit one token from the verified B unit

This Python API requires a private, stable 32-byte-or-longer key and immutable caller-held person IDs. Keep the key, fitted nuisance, encoder, inputs and outputs private. `wire` contains only the original `H_A` and one integer token. The mathematical law describes one draw per person; keyed replay is a computationally pseudorandom implementation of one persistent release, not a guarantee for repeated independent releases. Bind person IDs to unchanged inputs across processes in the deployment storage layer; the runtime rejects changed inputs within a live session.

```python
import json
from pathlib import Path
import numpy as np
from experiments.pcrl_task_aligned_cuts_v1 import data, release
from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
from experiments.pcrl_adaptive_release_v1 import fit_b, nuisance, release_specs

spec = release_specs.load_refined_b_spec(
    B_CENTER, B_PARITY, anchor=0, delta=.001, a_center_dir=A_CENTER)
index = data.index(INDEX)
encoder_member = data.member_record(index, 0, "encoder")
encoder = release.load_encoder_verified(
    data.verified_member(encoder_member), encoder_member["sha256"])
frozen_nuisance, nuisance_receipt = nuisance.load_frozen_nuisance(
    Path(B_CENTER) / "nuisance")
parity_receipt = json.loads(Path(B_PARITY).read_text())
session = fit_b.RefinedReleaseSession(
    spec["Q"], spec["_partition"], encoder, frozen_nuisance,
    parity_receipt=parity_receipt,
    encoder_sha256=encoder_member["sha256"],
    nuisance_sha256=nuisance_receipt["model_sha256"],
    replay_key=Path(PRIVATE_KEY_FILE).read_bytes())
wire = session.release(RuntimeInputs(x_a, h_a), person_ids=private_ids)
assert set(wire) == {"h_a", "token"}
assert wire["h_a"].dtype == h_a.dtype and wire["h_a"].tobytes() == h_a.tobytes()
```

`x_a` must be `N×32`, `h_a` `N×4`; the local encoder sees only those fields. Neither labels nor `H_B` are accepted in `RuntimeInputs`. Recipient A uses `(H_A,Z)`, coalition AB uses `(H_A,H_B,Z)` from separately held B service, and B alone keeps unchanged `H_B`.

## Exact independent inner audit and replay

`evaluate` exposes a Python callable rather than an invented CLI. The central execution queue constructs hash-pinned specs from completed units, deduplicates exact aliases, then dispatches the common audit once per canonical release. `aliases` maps every declared name, including D17, back to its audited canonical ID for reporting.

```python
from experiments.pcrl_adaptive_release_v1 import evaluate, release_specs

bundle = release_specs.build_release_specs(
    0, .001, A_CENTER, b_center_dir=B_CENTER,
    b_parity_receipt=B_PARITY, a_controls_dir=A_CONTROLS,
    b_controls_dir=B_CONTROLS, task_only_dir=TASK_ONLY)
report = evaluate.audit_panel(
    0, bundle["releases"], INDEX, PRIVATE_INNER_AUDIT_DIR,
    slate="catchup")
```

The audit fits compatible predictors on `audit_fit`, selects routes on `inner_selection`, and scores `inner_check` using each original person's exact 17-token expectation and both U/PWGTP weightings. It is a **new scientific job**, so execute it only if the queue marks that unit missing; a completed output is immutable. Replay a finished unit by verifying its `COMPLETE.json` inventory, release and model SHA-256 pins, and stored per-person contributions through the independent scorer. The public synthetic command above is the safe, input-free fit/release/score replay.

The historical Branch B child-state deployment example above describes the implemented API, but its external Linux parity receipts were not included in the verified archive. All three Branch B fits were exact T32 aliases of Branch A under the registered support floor. The final package therefore exposes the actual locked T32 U/P channels and D17 control through `artifact_cli` without claiming a restored child-state release. The restored model is **`EXPERIMENTAL_NO_ADVANTAGE`**; no algorithmic advantage is claimed.

## Reproduce the locked public aggregate result

The following command was executed after closeout with the committed lock, inner selection, and inference files. It regenerates `FULL_RESULTS.csv` and `PLOT_DATA.json` byte-for-byte. The table is also byte-identical to the published `ENDPOINT_TABLE.json` when the optional three private aggregate outer reports and their SHA pins are supplied, enabling the independent crosscheck.

```sh
python -m experiments.pcrl_adaptive_release_v1.artifact_cli reproduce-aggregates \
  --selection-lock results/pcrl_adaptive_release_v1/SELECTION_LOCK.json \
  --lock-sha256 4f894f98af67fd2a79e35ad3697971041aa53451b292b8bdf9691c8dd761f484 \
  --inner-selection results/pcrl_adaptive_release_v1/INNER_SELECTION.json \
  --inner-selection-sha256 1a17ed90edf83fb1f875350ee3a7705f1001d340b26d87cfae76b8b1aecbea8d \
  --inference results/pcrl_adaptive_release_v1/INFERENCE.json \
  --inference-sha256 f5e8fc0704f1666dac636e9cd5b6111315b0db9b389a52ee7ebe7c487015f2b1 \
  --output-csv /tmp/pcrl-adaptive-replay/FULL_RESULTS.csv \
  --output-table /tmp/pcrl-adaptive-replay/ENDPOINT_TABLE.json \
  --output-plot /tmp/pcrl-adaptive-replay/PLOT_DATA.json
```

The aggregate renderer reads no person records or fitted weights. The independent private replay over archived original-person contributions, including 10,000 common-household bootstrap draws, is recorded in `agents/independent_review/LOCKED_BOOTSTRAP_REVIEW.md`. Full private reruns should use the pinned queue and a newly authorized development protocol, not overwrite the completed unit outputs.
