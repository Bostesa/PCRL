# Quickstart and commands used

Work in the isolated `research/pcrl-method-portfolio-v1` worktree. The existing virtual environment is two directory levels above that worktree. Every real fit, preprocessing job, attack, score and independent numerical replay entered the durable S3 queue; the one-slot admission accounts for the unrelated heavy process on the same machine. A direct worker invocation is not an admitted reproduction.

## Read the result

Start with `START_HERE.md`, `RESEARCH_DECISION.md`, `COMMON_PROTOCOL.md`, `INTERFACE_CONTRACTS.md`, `DATA_ACCESS_MANIFEST.json` and `CANDIDATE_REGISTRY.json`. The full B and C screen rows are in `SCREEN_RESULTS.csv` and `ALL_ARMS.csv`; additional B and C aggregate/per-class files are named there. S4 evidence is in `INDEPENDENT_VERIFICATION.json` and `s4/`. The effective C-R1 fit lock is `CAMPAIGN_LOCK.json` at SHA-256 `98365edba38b01e100f50e4ca3c3d21abea931fb0b8534a9b188d35f179beb54`. Its 135 exact R1 units are complete; the earlier failed C reader and 22 successful engineering units remain archived separately.

## Reproduce custody checks

These commands were run in the worktree, or with the same arguments by S3's queued runner:

```sh
git rev-parse HEAD
git cat-file -t 1baf5bbdaf59cfa6a664cae07d652a4712bdfdc5
git cat-file -t d85f0a6f44415f2f9e334870f4e68504cd6b6080
shasum -a 256 results/pcrl_method_portfolio_v1/CAMPAIGN_LOCK.json
git diff --check
```

The role-only data gate and queue synthetic regressions ran under S3 receipts. This read-only suite was run from the shared directory:

```sh
cd results/pcrl_method_portfolio_v1/shared
PYTHONPATH="$(cd ../../.. && pwd)" python3 -m unittest -v test_safe_data_guard
```

The durable queue was driven with the equivalent of:

```sh
../../.venv/bin/python results/pcrl_method_portfolio_v1/shared/compute_queue.py --root results/pcrl_method_portfolio_v1/shared/.private/compute run-once
```

Each unit has an immutable spec, admitted parent hashes, resource cap and S3 receipt under the private queue root. The C-R1 manifest, finalizer, and 135-unit registry are listed in `RESOURCE_RECEIPTS.json`. To replay historical C-R1 units, use the queue source byte identity `969a09daf86872bdfd0ec9499c3d5e2041e653aad4d56590ecffb89945818b09` pinned in Git; later S4 queue amendments do not alter that fit receipt.

S4's B replay checked 18 arm/anchor outputs and 822 aggregate values without importing the candidate scorer. C's independent R3 replay checked 16 arm-seed outputs and 264 simultaneous clauses, plus the paired aggregate, without importing the candidate scorer. `S4_C_R3_INDEPENDENT_REPLAY_INDEX.json` binds the replay receipts and the immutable aggregate wording erratum. Two dated amendments separately document the verifier corrections. Public aggregate extraction does not require reopening role arrays. Old assessment person arrays and HEAD_VALIDATION labels remain embargoed.

The local `origin` is a filesystem path whose research-branch pushes fail with a remote unpack error in the current sandbox. The `github` remote also failed DNS resolution. Local branch commits and same-device restore are the current custody; neither is a verified remote tip. Off-device backup is pending.
