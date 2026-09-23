# Reproduce this study

Run from the pinned branch worktree root with Python 3.14.3. Use an isolated environment; this study used a 19 MB temporary target under `/private/tmp/pcrl_fullview_py` and did not modify the repository's existing environment.

```sh
python3 -m pip install -r experiments/pcrl_full_view_protection_v1/requirements-finite.txt
python3 -m unittest discover -s tests/pcrl_full_view_protection_v1 -v
python3 -m experiments.pcrl_full_view_protection_v1.run_synthetic --output-dir results/pcrl_full_view_protection_v1/synthetic
python3 -m experiments.pcrl_full_view_protection_v1.summarize --input-dir results/pcrl_full_view_protection_v1/synthetic --results-dir results/pcrl_full_view_protection_v1
python3 -m experiments.pcrl_full_view_protection_v1.verify --input-dir results/pcrl_full_view_protection_v1/synthetic --output results/pcrl_full_view_protection_v1/VERIFICATION_REPLAY.json
```

`run_synthetic` skips existing per-fixture JSON checkpoints. To produce a fresh independent replay, pass a new empty output directory within this study's results area; do not overwrite the recorded results. The original eight fixture definitions are in `PROTOCOL.md`; the ninth is in `AMENDMENT_2_PRIOR_SENSITIVITY.md`. The finite suite needs no ACS records, network or private archive.

For the unchanged 2018 Q/D17 radius check, use an independently verified local copy of Terminal 3's `REUSABLE_INPUTS.json` and its referenced private `Q.npz` members. The script checks each file's SHA-256 before reading:

```sh
python3 -m experiments.pcrl_full_view_protection_v1.radius_existing --manifest /path/to/REUSABLE_INPUTS.json --output results/pcrl_full_view_protection_v1/EXISTING_RADIUS_REPLAY.json
```

The original run used the already restored members in the completed evaluation worktree and did not copy them here. If absent, use only the existing authorized archive profile and restore those six small kernel members through the owning archive procedure. No 2016 data or outputs are needed for any computation in this study.

`SYNTHETIC_CHECKS.json` records numerical residuals and wide brackets. CLARABEL numerical brackets are not interval-arithmetic proofs. New fits on ACS are not part of this replay because the protocol's pilot gate failed.
