# Portable verifier publication note

The published verification programs are packaging adaptations. The original executions and their source hashes remain preserved in [INDEPENDENT_VERIFICATION.json](INDEPENDENT_VERIFICATION.json), [VERIFICATION.md](VERIFICATION.md), and [SCORE_REPLAY.json](SCORE_REPLAY.json). Those historical records were not rewritten, and the adapted programs were not used to retrain or reevaluate the experiment during publication.

| Program | Original executed source SHA256 | Published adaptation SHA256 |
| --- | --- | --- |
| [Artifact verification](../../scripts/verify_acs_transfer_artifacts.py) | `739d1f2e9623990dfb12e9a4ef6d0e36ee506c56c555f68cdab39f15094936cf` | `c1d52bc8246cd637f080923408e95e3a6b6a417a60d650a6c3fcc0891fb5de94` |
| [Score replay](../../scripts/verify_acs_transfer_scores.py) | `4f8c4667f4753ff87a80943386fb774fe3213b2237a1b2d5cf5884995f2ca634` | `268823828325e2b5a09b59706e43c32d8cf9f0fe04395e083950db932bf1e2d3` |

Both programs derive the repository root from their published file location. `--out` accepts the existing results directory and defaults to `results/redesign_20260907_acs_transfer_v1` in that checkout. The artifact verifier also remaps result paths in original manifests when the result directory is relocated. The score verifier accepts an optional `--root` override.

By default, both commands print a new JSON report to stdout. `--report` writes only to a fresh file and refuses an existing path; neither program automatically writes into the original records. Run from the repository root, for example:

```sh
python scripts/verify_acs_transfer_artifacts.py --out results/redesign_20260907_acs_transfer_v1 --report /tmp/acs_artifact_replay_new.json
python scripts/verify_acs_transfer_scores.py --out results/redesign_20260907_acs_transfer_v1 --report /tmp/acs_score_replay_new.json
```

Choose unused report names. Exact replay requires the original local arrays/checkpoints and raw ACS data referenced by the manifests; compact GitHub evidence alone cannot supply those omitted artifacts. Fresh experiment regeneration is a different execution and does not establish byte-identical recovery of the original checkpoints.

The artifact adaptation retains the label reconstruction, split/mask, schedule, checkpoint, validation-selection, file-hash, and inference checks. Changes are portable paths, a callable verifier/CLI, a new report destination, explicit adaptation provenance, and removal of automatic historical report/Markdown writes. The new report records the current program hash separately from the original executed hash. The score adaptation retains its label reconstruction, sklearn metrics, tolerance, comparisons, and selection/schedule checks; it adds the portable CLI, fresh report output, and a nonzero exit on metric errors.

Packaging validation used syntax parsing, artifact-verifier `--help` from another working directory, and stubbed artifact-verifier CLI checks for stdout, fresh-file output, relocated manifest paths, and refusal to overwrite. No model fit or inference replay was executed for these adaptations.

Historical record SHA256 values checked unchanged after packaging validation:

- `INDEPENDENT_VERIFICATION.json`: `87c66514dfcf9b2447bf9c9fa0ed0d677768c3e2c74a9741b98ff42ae511850a`
- `VERIFICATION.md`: `36f307d47559c0e9c118bf2e390a8a1d8dc542874894bce4cb3d555e47a10c94`
- `SCORE_REPLAY.json`: `f041cd18329394df9e2063f8d756b926e62cdc78da77bb2ea6c2b16df2be8067`
