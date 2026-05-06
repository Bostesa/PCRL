# BIOS_LAYER12 launch guide

Launches the layer-12 [CLS] PCRL extension on a single g4dn.xlarge.
**The launcher script (`launch_bios_layer12.sh`) is gitignored** — the user
must keep it local and never commit it.

## Pre-launch checklist

Run these locally (not on EC2) before invoking the launcher.

1. AWS credentials work and you're in the right account.
   ```
   aws sts get-caller-identity
   ```
2. No `BIOS_LAYER12`-tagged instance is already running.
   ```
   aws ec2 describe-instances \
     --filters Name=tag:Name,Values=BIOS_LAYER12 \
               Name=instance-state-name,Values=pending,running,stopping,stopped \
     --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output table
   ```
3. List currently running G-family instances. Note the
   `PCRL_VARCONSTRAINT` instance if present — **do not touch it**.
   ```
   aws ec2 describe-instances \
     --filters Name=instance-state-name,Values=running \
               'Name=instance-type,Values=g4dn.*,g5.*,g6.*' \
     --query 'Reservations[].Instances[].[InstanceId,InstanceType,Tags[?Key==`Name`].Value|[0]]' \
     --output table
   ```
4. Branch is pushed.
   ```
   git push -u origin bios-pcrl-layer12-2026-05-05
   ```
5. Dry-run the launcher to inspect the user-data without launching.
   ```
   DRY_RUN=1 ./launch_bios_layer12.sh
   ```

## Configuration

Override defaults by setting env vars before invoking the script. Defaults are
chosen to match the spec; deviations are flagged below.

| Var | Default | Notes |
| --- | --- | --- |
| `S3_BUCKET` | `pcrl-bios-layer12-20260505` | results + live logs prefix |
| `S3_PREFIX` | `bios_pcrl_layer12` | sub-prefix |
| `BRANCH` | `bios-pcrl-layer12-2026-05-05` | must be pushed |
| `INSTANCE_TYPE` | `g4dn.xlarge` | $0.526/hr on-demand |
| `HARD_CAP_HOURS` | `5` | watchdog + backup sleeper auto-terminate |
| `AMI_ID` | `ami-012ba162b9cd2729c` | DL Ubuntu 22.04, used by head-aware LEACE 2026-05-05 |
| `IAM_INSTANCE_PROFILE` | `pcrl-overnight` | needs `s3:PutObject` on the bucket |
| `INSTANCE_TAG` | `BIOS_LAYER12` | enforced unique by the launcher |
| `KEY_NAME`, `SUBNET_ID`, `SECURITY_GROUP_ID` | _(unset)_ | optional; AWS picks defaults |

## Stage 1 — Smoke test (~30 min, ~$0.30)

Runs immediately after the cache step. P1 seed=0, 5 epochs, batch=4096.

**Pass criteria** (the user-data writes a verdict to S3 and bails on FAIL):

| Criterion | Threshold | Where measured |
| --- | --- | --- |
| Compliance | `r2_gender_dev <= 0.05` | full BIOS dev (~39k) |
| Variance health | `per_dim_std_median > 0.5` | dev representations |
| Effective rank | `eff_rank_pr > 100` | participation ratio on dev |
| Linear utility | `task_dev_acc >= 0.55` (~vanilla 0.79 - 3pp = 0.76 ideal) | sklearn LR on patched dev |
| Linear gender probe | `<= base_rate + 10pp` | sklearn LR class_weight=balanced |

**MLP probe accuracy in [85%, 97%] is EXPECTED**, not a fail. RLACE / Kernelized
CE / TaCo / Obliviator empirics: a linear-R² constraint cannot bound non-linear
extractability. We report it but do not gate Stage 2 on it.

## Stage 2 — Full run (~3-4h, ~$2)

Runs autonomously **only on Stage 1 PASS**. 9 cells (3 purposes × 3 seeds),
3 concurrent on the GPU. Then closed-form LEACE, K-LEACE-union (= LEACE here,
documented), R-LACE rank-1 (n_iters=5), MLP probe across all conditions.

## Hyperparameter deviations from the deep-research spec

The launcher uses a few values that differ from the original spec — all
documented and justifiable:

| Field | Spec | Used | Reason |
| --- | --- | --- | --- |
| `batch_size` | 256 | **4096** | the differentiable in-batch R²(gender) requires `N >= ~2·d` to avoid the rank-deficient saturation at R²=1; with d=768 batch=256 is mathematically broken (verified empirically). Cached features make 4096 trivial on GPU. |
| `lambda_init` | 0.0 | **1.0** | the LoRA-realises-eraser warm-start is a saddle point that CE pulls out of in <2 steps; the dual ascent at 5e-3 takes ~2200 steps to reach λ=10, far longer than the 630-step training budget. Starting at 1.0 keeps the constraint binding from step 0 without forcing a hard project. |
| `dual_lr` | 5e-3 | **5e-2** | matches the spec's autonomous-debug directive #4 ("COMPLIANCE FAILURE: raise dual LR to 1e-2"); 5e-2 is empirically close enough to keep λ growing through the warmup window. |

If the smoke fails on compliance, the autonomous-debug directives in the spec
(λ_max → 20, dual_lr → 1e-1, λ_floor enabled) can be applied without
relaunching by re-running with overridden flags.

## Monitoring

The instance syncs `/var/log/v2_bios.log`, `train.log`, `watchdog.log`, and the
`results/bios_pcrl_layer12/` tree to S3 every 60s.

```
# tail the boot log
aws s3 cp s3://pcrl-bios-layer12-20260505/bios_pcrl_layer12/live_logs/<INSTANCE_ID>_v2_bios.log -

# tail the training log
aws s3 cp s3://pcrl-bios-layer12-20260505/bios_pcrl_layer12/live_logs/<INSTANCE_ID>_train.log -

# smoke verdict (after ~30 min)
aws s3 cp s3://pcrl-bios-layer12-20260505/bios_pcrl_layer12/SMOKE_VERDICT_<INSTANCE_ID>.json -

# stage markers
aws s3 ls s3://pcrl-bios-layer12-20260505/bios_pcrl_layer12/ | grep STAGE_

# full results sync
aws s3 ls s3://pcrl-bios-layer12-20260505/bios_pcrl_layer12/results/ --recursive
```

## Expected timeline

| Stage | Wall | Output |
| --- | --- | --- |
| Bootstrap (apt + pip) | ~3 min | stage markers 00..04 |
| BERT layer-12 cache | ~10 min on T4 | `cls_layer12_embeddings.npz` (~750MB) |
| Stage 1 smoke (P1, 5 ep) | ~5 min | `smoke/metrics_P1_0.json` + verdict |
| Stage 2 full (9 cells, 3-concurrent) | ~2-3h | 9× `metrics_*.json`, 9× `eval_reps_*.npz` |
| Baselines + MLP probe | ~10 min | `leace_results.json`, `rlace_results.json`, `mlp_probe_results.json` |
| Tar + final S3 sync | ~1 min | `full_logs/<INSTANCE_ID>.tar.gz` |
| **Total** | **~3-4h** | (5h hard cap) |

## Cost ceiling

`g4dn.xlarge` on-demand at $0.526/hr × 5h hard cap = **$2.63 max**. Typical
expected ~$2 if Stage 2 finishes in 3-4h.

## Cleanup

The instance auto-terminates via the watchdog on hard cap or 10 min after the
done-flag is touched. To force-terminate sooner:

```
aws ec2 terminate-instances --instance-ids <INSTANCE_ID>
```

**Never use `--filters` with `terminate-instances`.** Always pass the explicit
`<INSTANCE_ID>` from the launcher output, so you cannot accidentally terminate
the `PCRL_VARCONSTRAINT` instance.

## Troubleshooting

* **Smoke FAIL — compliance violated**: re-run with `--lambda-init 5.0
  --dual-lr 1e-1 --lambda-max 20`, or with `--use-post-projection` (registers
  a frozen LEACE post-projection on the head; LoRA learns inside the null
  space). Both knobs are exposed by `scripts/run_pcrl_bios_layer12.py`.
* **Smoke FAIL — collapse (`per_dim_std_median < 0.5`)**: raise `--lambda-var`
  to 10; reduce `--lambda-max` to 5 (autonomous-debug directive #2).
* **Stage 2 cell stuck at majority for 100+ steps**: re-run that cell only
  with `--use-post-projection`.
* **Instance never gets past stage 04 (HF prefetch)**: HF Hub rate limit (no
  HF_TOKEN). Same root cause as the head-aware LEACE attempt #1; the user-data
  bumps the prefetch timeout to 900s, which has been sufficient on retries.

## Deliverables (in `results/bios_pcrl_layer12/`)

* `cache/cls_layer12_embeddings.npz` + `cache/labels.npz`
* `encoder_{P1,P2,P3}_{0,1,2}.pt` (9 cells)
* `eval_reps_{P1,P2,P3}_{0,1,2}.npz` (9 cells)
* `metrics_{P1,P2,P3}_{0,1,2}.json` (9 cells)
* `leace_results.json`, `klu_results.json`, `rlace_results.json`,
  `vanilla_results.json`
* `eval_reps_LEACE.npz`, `eval_reps_RLACE.npz`
* `mlp_probe_results.json`
* `smoke/metrics_P1_0.json`, `smoke/SMOKE_VERDICT.json`

## Honest framing reminder

When all 3 purposes share gender as the disallowed attribute, K-LEACE-union
mathematically reduces to single LEACE. PCRL's value-add in this setup is
**purpose-specific task preservation** (P1 keeps 28-way variance, P2 keeps
5-way, P3 keeps binary), not "more directions erased." Frame §5.6 around
that — `klu_results.json` carries an explicit `reduction_note` field for this
reason. Do not dress K-LEACE-union up as a different operator from LEACE here.
