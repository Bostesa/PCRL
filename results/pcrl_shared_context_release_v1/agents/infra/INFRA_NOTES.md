# Infra notes: shared-context release v1

Written 2026-09-24, around 18:10Z, by the infrastructure agent. No instance was launched. No outer or 2018 assessment label was read.

**Git ignore.** The root `.gitignore` line `infra/` ignores this whole directory. Force-add only this file:

    git add -f results/pcrl_shared_context_release_v1/agents/infra/INFRA_NOTES.md

Never force-add `agents/infra/private/`.

Two private paths are involved:

- `agents/infra/private/` holds the private manifest, the rendered user data and the resource, SSM and cost records. It is ignored, and that is intended.
- `results/pcrl_shared_context_release_v1/private/` is **not** ignored. The predecessor used a nested `.gitignore` containing `/private/`. The coordinator should add the same file before anything writes there. Units are written there only on the host.

## Files (package `experiments/pcrl_shared_context_release_v1/`)

### `laws.py`

- `person_law(source, inputs)` returns an `(n, 17)` array. Only the six legal fields are allowed: `x`, `ha`, `token_codes`, `teacher_p`, `residual` and `risk`. The fields `hb`, `labels`, `ids`, `households`, `weights`, `role`, `aux` and `J` are refused before any artifact is loaded.
- The unit kinds come from `release.json`, schema `pcrl-sc-release-descriptor-v1`, whose pins map each file to its SHA and are re-verified on every load. The kinds dispatch as follows:
  - `nested` goes to `release.load_law`;
  - `deterministic_policy` goes to `rd.load_law`;
  - `adv_mlp` goes to `adv.load_law`.
- There are two in-memory kinds:
  - `historical_map` is D17 or Q, loaded through the pinned index or a pinned `.npz` and indexed by the stored T0 code.
  - `h_only` is the one-column H wire. It is the only law that does not have 17 columns.
- J is not a kind. It stays on the predecessor path, `AR/external_audit.py`.
- Every law is validated: shape, float64, finite, nonnegative, and row sums equal to 1 within 1e-9.
- If a descriptor is missing, the runner writes it after a successful unit. It sets `pins` to every output file.

### `audit_panel.py`

- The AR slate is unchanged: `ar_audit.fit_role_slate`, `select_frozen_routes`, `score_frozen_route`, `evaluate._registry_routes` and `_private_contributions`, with the same seven H ancestor roles.
- Seeds are `36000 + 1000*anchor + role_index`. This is a new registered base, and the seeds are common to every release.
- **De-duplication.** A release's identity is a SHA-256 over three items for each of the roles `audit_fit`, `inner_selection` and `inner_check`:
  - the role name;
  - the SHA of the ordered person ids;
  - the SHA of the float64 law bytes and shape.
- Identity must match exactly. `ALIAS_LEDGER.json` keeps every declared name, its canonical name, the law SHA for each role, the maximum absolute difference to the canonical law (for aliases), and the nearest distinct canonical law.
- The report adds `logical_to_canonical`, `aliases` and `law_identity`.
- **hist_gb minimum leaf.** The inherited rule is unchanged and runs through one code path for every candidate and for H: exact `hist_gb_{20,5}` use `leaf × max per-person support` on the fit rows, and the sampled variants use `leaf`.
  - For each release and role, `hist_gb_min_leaf` records the support histogram, the effective leaf and the leaf observed in `slate.json`. The observed value must equal the expected one or the run raises.
  - A stochastic NM law with full support gets leaf 340 and 85. Historical Q has full support on every anchor (minimum mass about 1e-12), so it gets the same treatment. No mass floor is applied.
- **Inner mode.** It loads only the sanitized prepared object and `roles.pooled_role` for the three inner roles, and it asserts that every household belongs to its role.
- **Outer mode.** `score_outer` runs only after `verify_outer_gate` succeeds. The gate requires:
  - `results/pcrl_shared_context_release_v1/SELECTION_LOCK.json` with schema `pcrl-sc-selection-lock-v1`, `status` LOCKED, `outer_assessment_authorized` true, and `anchors[a]` pinning the inner `COMPLETE.json` and `INNER_AUDIT.json` SHAs;
  - `private/OUTER_UNLOCK.json` with schema `pcrl-sc-outer-unlock-v1`, a matching lock SHA, `remote_verified` true, the study branch, and a 40-hex commit.

  After the gate, outer mode reuses the routes the inner panel selected (`AR/outer_audit.reconstruct_selected_routes`) and loads outer rows with `AR/outer_pool.load_verified_outer`. It never refits or reselects.

### `runner.py`

- Queue JSON has schema `pcrl-sc-queue-v1`. Each unit runs as a single-threaded subprocess with `OMP`, `MKL`, `OPENBLAS`, `NUMEXPR`, `VECLIB` and `BLIS` threads all set to 1, and writes into `<units>/<id>/`.
- `COMPLETE.json` is published atomically. It records:
  - `inputs_sha256`: the argv, declared input files and dependency receipts;
  - `code_commit` and `code_tree_sha256`, the hash of the package `*.py` files;
  - `data_role_hash`: the AR salt, cuts, `roles.py`, the declared roles and the index SHA;
  - `outputs_sha256`.
- Resume re-verifies all of these, blocks on any mismatch, and never overwrites.
- `--reuse-commit SHA` accepts receipts from an older commit, but only if the tree hash is identical.
- Retries are bounded. Two attempts in total are counted across invocations. Exit code 3 is a deliberate refusal and is not retried. An interrupted unit directory is kept as a failed attempt under `.attempts/<id>/attempt-k/` together with its logs.
- Outer roles are refused in queues. The unit root must be under `private`.
- `plan` builds the per-anchor queue:
  - `bank` feeds the six NM/T32 variants, RD_TASK, RD_PRIV and ADV_B1/ADV_B2;
  - those feed `inner_audit`, whose sources are D17 and Q_HIST plus every fitted release;
  - positive controls for AB/SEX and AB/RAC1P.
- CLI templates live in `DEFAULT_TEMPLATES`, which `--templates` can override.
  - RD and ADV match `rd.py` and `adv.py` as they exist now: `--anchor`, `--variant`/`--beta`, `--bank` and `--out`.
  - **`fit_nm` (`build-bank`, `fit --variant`) and `poscontrol` (`--role`) are guesses.** Align them once those modules exist.

### `stage.py`

- It reuses AR's 17 label-stripped objects under `pcrl_adaptive_release_v1/inputs/00..16`: the index, the sanitization receipt, three sanitized anchors, and the encoder plus the Q/D17/D33 maps for each anchor.
- Excluded by allowlist: `original_prepared_*`, which carry outer labels, and the external J index.
- **Verified 2026-09-24.** All 17 pinned versions are present with SSE-AES256 and matching sizes. Each object was also downloaded and SHA-256-verified against AR's manifest. Total size is 79,492,352 bytes.
- The study manifest (`agents/infra/private/INPUT_STAGE.json`, sha `15e1e655…4cfe`) was uploaded to `s3://pcrl-ux-archive-ed9d21fd/pcrl_shared_context_release_v1/control/INPUT_STAGE.json` as version `ZA_VMPGR5FHwB7RmCfkH0l5uvJt0oN3n`.
- The host `restore` command fetches each object by version and checks its SHA before moving it into place.

### `archive.py`

- `unit` and `sweep` produce, for each completed unit, a tar.gz under `pcrl_shared_context_release_v1/units/<id>.tar.gz`.
- Each tarball is put with SSE and a version id, re-downloaded by version, SHA-compared, extracted and re-inventoried. The manifest goes to `archive_manifests/`.
- Failed attempts are archived as well. `STATUS.json` is mirrored to `control/`.
- A host timer runs `sweep` every 20 minutes. Use `--local-only` for a dry run; it was smoke-tested locally.

### `cloud.py` and `cloud_user_data.sh`

These run one c7i.8xlarge on SSM only. Resources:

- AMI **`ami-0b2c9d1f3edcfd709`**: AL2023 2023.12.20260918.0, the AMI the predecessor actually ran (CloudTrail RunInstances at 02:54:24Z). `ami-025d99823a4caad37` is the Ubuntu AMI of the task-directed study and is not used here.
- Instance profile **`pcrl-ux-ec2`**: its role has SSM core access plus bucket-only S3 Get, Put and List, and no Delete.
- Subnet `subnet-092b1557c0ffb806f`, in the default VPC, us-east-1a.
- A **new** security group with zero ingress, created at launch. The predecessor's `sg-0b9c7f84f4290d9ab` was deleted at its closeout.
- 120 GiB encrypted gp3, IMDSv2 required, shutdown behavior `terminate`.
- Tags `Project=pcrl`, `Study=pcrl_shared_context_release_v1`, `Name` and `DeadlineUTC`.

The user data runs these steps in order:

1. Arm the watchdog timer. At the chosen time (default: launch plus 10 h) it stops the runner, runs `aws s3 sync` of `results/pcrl_shared_context_release_v1` to `…/emergency/results` with SSE into the versioned bucket, uploads the logs, and shuts down.
2. Arm a hard-stop timer 30 minutes after the watchdog.
3. Install the predecessor's exact stack: Python 3.11 venv, numpy 2.4.2, scipy 1.17.1, sklearn 1.8.0, pandas 3.0.1, joblib 1.5.3 and torch 2.10.0+cpu.
4. Make a sparse, blobless clone of the pushed branch and check out the pinned commit, which must be an ancestor of `origin/<branch>`. This flow was tested locally against GitHub.
5. Restore the inputs.
6. Start the archive timer.
7. Write `READY.json` to `control/`.

The user data is 7.3 KB.

## Verified 2026-09-24

- `aws sts get-caller-identity` with profile `vein` works: SSO AdministratorAccess.
- `cloud.py preflight` reports `ok: true`. The standard vCPU quota is 32; 0 are in use and 32 are needed. **The instance uses the entire quota.**
- `launch --dry-run` returned EC2 DryRun `authorized` for both `create-security-group` and `run-instances`, and the rendered user data passes `bash -n`.
- Tests: 34 pass in 13 s (`test_laws` 17, `test_runner` 12, `test_audit_panel` 5). One of them runs the **real** AR/TAC/TDR standard slate on a synthetic 48-person-per-role anchor.

## Follow-up (2026-09-24, about 18:20Z)

### Runner receipts moved out of unit directories

- `fit_nm` owns `COMPLETE.json` inside its unit directory, and DET_SEL re-checks that directory's inventory. The runner therefore adds nothing to a unit directory, except `release.json` when a release is declared and the module did not write one.
- The runner receipt is now `<units>/_receipts/<id>.json`, and it covers every file in the unit directory. Attempt logs stay in `.attempts/<id>/attempt-k/`.
- `archive.py` checks each unit directory against its receipt: every file must match and no extra file may be present. It uploads the receipt as `units/<id>.RECEIPT.json`, and it also archives successful attempt logs.

### Plan templates

- The `fit_nm` command lines were confirmed with the method owner.
- The `rd`, `adv` and `poscontrol` command lines were taken from their files at 18:17Z. `rd.py` was still being edited, but its flags were stable.
- Dependency order:
  1. `aA_bank`, one per anchor.
  2. `decide_k`, which waits for all three anchors' banks. It wraps `fit_nm decide-k` with `runner capture`, which writes `DECISION.json`.
  3. The six NM/T32 variants. NM4 variants take `--nm4-k {json:decide_k/DECISION.json:nm4_K}`, which is resolved at launch time and hash-checked against the `decide_k` receipt.
  4. DET_SEL1 depends on NM1_U, and DET_SEL4 depends on NM4_U.
  5. RD_TASK, RD_PRIV, ADV_B1 and ADV_B2 depend only on the bank.
  6. `aA_inner_audit` depends on every release above, plus D17 and Q_HIST.
- POS (AB/SEX and AB/RAC1P) and `aA_J_inner` depend only on the staged inputs.
- `plan --nm4-k 2|4` skips the barrier, which is needed for a plan that covers only some anchors.

### J continuity reference

- J16 is already part of the sanitized anchors as `pool["J"]`; the sanitized copies were checked locally. The staged prepared objects therefore already contain J.
- AR's `external_audit` also needs its external index, a 32 KB metadata JSON of paths, hashes and shapes with no person rows or labels.
  - It is now staged as a pinned 18th object: version `GELw62dAVhylHvcpI83mTbTeoX3.cU3j`, sha `461b06f0…ad8a`. This is exactly the SHA that AR passed as `--external-index-sha256`, and it was re-verified.
  - The manifest was regenerated: 18 files, sha `9ddb13a3…d3fa`, S3 version `frEADbCBRCcJowSwg_znRdN8ivLnAi2R`.
- The J unit runs AR `external_audit --methods J --slate standard` unchanged.
- To keep one H reference across panels, the main panel's `SEED_BASE` is now 26000. That is the value `external_audit` hard-codes, so the H-only slates of the main and J panels share rows, seeds and code.
- Outer J scoring is `audit_panel outer-j`. It opens nothing before this study's gate accepts the lock and unlock files. The lock must pin the J inner panel under `external_context.J[anchor]`.
- AR's own `score_locked_j_outer` uses AR's gate and is not used.
- Outer scoring of any kind still requires the coordinator to restore the original prepared objects, which carry outer labels, into `results/pcrl_shared_context_release_v1/private/original_2018_restore` after the lock. This happens only after the lock.

## Coordinator launch sequence

Run everything from the worktree with `PYTHONPATH=.` and `/Users/nathansamson/PCRL/.venv/bin/python -m experiments.pcrl_shared_context_release_v1.<module>`.

1. `cloud preflight`
2. `cloud launch --commit <PUSHED_SHA> --dry-run`
3. `cloud launch --commit <PUSHED_SHA> --watchdog-utc <T>`. The watchdog time must be no later than the 2026-09-25T13:26Z ceiling.
4. Wait for `cloud status` to show `READY.json`, or for `control/BOOTSTRAP_FAILED` to appear.
5. Build the queue on the host with `cloud send "plan queue" "cd /opt/pcrl/work && PYTHONPATH=. /opt/pcrl/venv/bin/python -m experiments.pcrl_shared_context_release_v1.runner plan --index /opt/pcrl/work/results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json --units-root /opt/pcrl/work/results/pcrl_shared_context_release_v1/private/units --plan-dir /opt/pcrl/work/results/pcrl_shared_context_release_v1/private/plan --queue-out /opt/pcrl/work/results/pcrl_shared_context_release_v1/private/QUEUE.json"`.
6. Start the runner with `cloud start-runner --queue /opt/pcrl/work/results/pcrl_shared_context_release_v1/private/QUEUE.json --workers 12`.
7. Record spend with `cloud cost --record`.
