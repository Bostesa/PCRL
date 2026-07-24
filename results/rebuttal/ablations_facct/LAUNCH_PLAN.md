# AWS launch plan — FAccT resubmission ablations (NOT YET LAUNCHED)

Status 2026-07-24: code + smokes done on branch `ablations-facct-2026-07-24`
(needs push before launch — user_data clones from GitHub). Predictions
registered at ecaeba4 BEFORE smokes. Awaiting user approval for AWS spend.

## Run matrix

All runs: erase-layer-era protocol, 200 epochs × seeds {0,1,2}, batch 256,
`--lora-target repr_proj_only`, λ_vicreg=1.0, CPU (c5.4xlarge, Standard
quota — same route as the rank-8 ablation).

### Ablation 1 — linear adapter (60 cells)
Per dataset ∈ {adult, hmda, diabetes}:
```
run_v2_dataset.py --dataset <ds> --use-erase-layer --lora-target repr_proj_only \
  --adapter-type linear --out-tag _ABL1_LINEAR --device cpu
```
Comparator: published erase-layer pilot (LoRA arm) — no re-run needed.

### Ablation 2 — no frozen LEACE layer (60 cells)
```
run_v2_dataset.py --dataset <ds> --lora-target repr_proj_only \
  --no-leace-init --warmup-epochs 0 --out-tag _ABL2_NOLEACE --device cpu
```
Single-variable change vs pilot: erase layer removed, schedule identical
(no warmup), LoRA at published ranks (Adult/HMDA 8, Diabetes 24).

### Ablation 3 — held-out selection (16 candidate configs, registered grid)
Candidates (all with --use-erase-layer --lora-target repr_proj_only
--eval-split val, explicit flags so run_config is self-describing):
- adult / hmda: (--lambda-min {0,5}) × (--warmup-epochs {0,5}), --lora-rank 8 → 4 each
- diabetes:     (--lambda-min {0,5}) × (--warmup-epochs {0,5}) × (--lora-rank {8,24}) → 8
Tag: `_ABL3_lm{0|5}_wu{0|5}_r{8|24}`. Checkpoints MUST be archived
(eval-only needs them).
Afterwards (local, cheap): `select_hyperparams.py --dataset <ds>` on pulled
val results → selected config per dataset → local
`--eval-only --eval-split test --results-tag ..._TEST` on pulled checkpoints
→ delta table vs published test numbers.

## Instance layout (max 2 c5.4xlarge concurrent, 2 waves)

Timing basis: Diabetes ≈70 min/seed observed (rank-8 run); Adult ≈35,
HMDA ≈50 min/seed estimated from N. Caps at ~1.7× estimate.

Wave 1 (parallel):
- **Instance A** — Ablation 1, adult→hmda→diabetes sequential ≈ 8h. Cap 14h.
- **Instance B** — Ablation 2, same layout ≈ 8h. Cap 14h.

Wave 2 (parallel, after Wave 1 shuts down):
- **Instance C** — Ablation 3 adult (4 cfg ≈ 7h) + hmda (4 cfg ≈ 10h) ≈ 17h. Cap 26h.
- **Instance D** — Ablation 3 diabetes 8 cfg ≈ 28h. Cap 36h.

Cost @ $0.68/h on-demand: Wave 1 ≈ 28 inst-h ≈ $19; Wave 2 ≈ 45 inst-h ≈ $31.
**Total ≈ $50** (+ caps headroom worst case ≈ $61).

## Hardening (per instance, mirrors rank-8/vicreg pattern)

- user_data with NO exec-redirect-only logging: bootstrap log cp'd to S3.
- 60s watchdog: live logs + partial results sync.
- Per-stage `rc=` echo; STATUS.txt (strict/clean/meanR²/std/eff_rank per
  dataset) written to run prefix AND lifecycle-exempt ARCHIVE_DEST.
- Final dual-sync of results_final/ + **checkpoints/** (Abl 3 requires
  checkpoints for the local eval-only pass; 7-day bucket lifecycle —
  pull within 7 days).
- IAM profile pcrl-bios-s3-writer (required; silent failure without it).
- Backup sleeper hard-cap shutdown.

## Post-run analysis (local)

1. Pull all results + checkpoints from ARCHIVE_DEST.
2. Ablation 1: side-by-side per-cell table (strict pass, eff_rank,
   per_dim_std, task acc) linear vs published LoRA pilot.
3. Ablation 2: per-cell compliance survival table vs pilot; flag saturated λ.
4. Ablation 3: selection per dataset → eval-only test pass → delta table
   (held-out-selected vs eval-grid-selected). Compare against PREDICTIONS.md;
   disclose any deviation as such.
