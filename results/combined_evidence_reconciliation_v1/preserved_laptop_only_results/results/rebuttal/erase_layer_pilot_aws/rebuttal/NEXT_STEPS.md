# Erase-layer pilot — next steps after smoke

**Status as of 2026-05-18 00:21 local:** code surgery merged on branch
`erase-layer-pilot-2026-05-17` (commits `96c03d0`, `04cb7ef`).
5-epoch CPU smoke on Adult seed 0 passed:

- Construction-time joint LEACE fit drops linear R² to **exactly 0.0** on
  all 5 disallowed attributes (race / sex / age_group / marital_status /
  income).
- After 5 epochs of constrained training, R² stays in [0.0036, 0.0090]
  on all 8 (purpose, attr) pairs — well under the 0.05 paper threshold.
- `effective_rank = 18.0` (clean; 2.0 is the rank-collapse threshold).
- `per_dim_std_mean = 0.26` (below the 0.5 health threshold, but only
  5 epochs of warmup — full 200-epoch run needed to know if it lifts).

## What hasn't been done

The full 60-cell pilot (3 datasets × 3 seeds × pairs × 200 constrained
epochs). Expected wall-clocks:

- **Adult**: ~19 min/seed × 3 seeds = ~1 h on CPU
- **HMDA**: ~75 min/seed × 3 seeds = ~3.75 h on CPU
- **Diabetes**: ~75 min/seed × 3 seeds = ~3.75 h on CPU
- **Total**: ~8 h CPU, or ~7.5 GPU-h on `g4dn.xlarge` ($4).

## Two launch paths — pick one

### Path A: local CPU overnight (free, blocks the laptop)

```bash
# From repo root, on branch erase-layer-pilot-2026-05-17:
for d in adult hmda diabetes; do
  .venv/bin/python experiments/run_v2_dataset.py \
      --dataset $d --out-tag _ERASE_PILOT --seeds 0 1 2 \
      --use-erase-layer --lora-target repr_proj_only \
      --device cpu \
      > /tmp/erase_pilot_${d}.log 2>&1
done
```

Outputs land in `results/v2_{adult,hmda,diabetes}_ERASE_PILOT/`.

### Path B: AWS g4dn.xlarge overnight (~$4, doesn't block laptop)

Requires pushing the branch to GitHub first. The infra launchers were
untracked during the anonymization pass (`d09c5f7`), so the AWS path
needs either: (a) a fresh launcher checked in on a private branch,
or (b) manual `aws ec2 run-instances` with the params from
`CLEANUP_PLAN.md:G` (those AMI/IAM/bucket values are still in
`.git/logs/HEAD` reflog if needed). Adds ~30 min of setup overhead.

## Post-pilot: dominant-axis audit + aggregate comparison

After either launch path finishes, run:

```bash
.venv/bin/python scripts/eval_round4_dominant_axis.py \
    --datasets adult hmda --seeds 0 1 2 --tag ERASE_PILOT
.venv/bin/python scripts/eval_round4_dominant_axis.py \
    --datasets diabetes --seeds 0 1 2 --tag ERASE_PILOT

# Compare to Round 5/7 baseline
.venv/bin/python -c "
import json
for d in ['adult','hmda','diabetes']:
    base_tag = 'ROUND7' if d == 'diabetes' else 'ROUND5'
    old = json.load(open(f'results/v2_{d}_{base_tag}/summary.json'))
    new = json.load(open(f'results/v2_{d}_ERASE_PILOT/summary.json'))
    print(f'{d}: pass_count_mean Round{base_tag} -> ERASE = {old[\"pass_count_mean\"]} -> {new[\"pass_count_mean\"]}')
"
```

## What the rebuttal text needs (drafted regardless of outcome)

Three variants of the rebuttal paragraph for
`results/rebuttal/erase_layer_pilot/PAPER_PASTE.md`:

1. **Pilot succeeded** (cleanly-compliant count ≥ 30/60 with task acc
   within 1pp): "The framework's compliance can be strengthened from the
   submitted 7/60 to N/60 by porting the §5.5 erase-layer architecture."
2. **Pilot partial improvement** (compliance up but task acc drops 2–5pp):
   "The framework supports a tradeoff curve; the submitted version is the
   high-accuracy operating point, the erase-layer variant is the
   high-compliance one."
3. **Pilot showed no architectural improvement**: "We tested the obvious
   architectural fix; collapse-compliance turns out to be a metric-design
   issue rather than an architectural one. The dominant-axis audit
   already gives a stronger operational certificate." (Less likely given
   the smoke numbers above, but worth pre-drafting.)

## Recommendation

**Path A, fire-and-forget overnight on the laptop.** Cheap, no AWS setup,
and the smoke confirms the pipeline is healthy. ~8 hours from kickoff
to results. Wake up, run the dominant-axis audit + aggregator, draft
rebuttal text against the actual numbers.
