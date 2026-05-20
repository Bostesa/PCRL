# Cross-purpose constraint rebuttal pilot — plan

Branch: `cross-purpose-rebuttal-2026-05-18` (created today from `origin/main` @ 17ef7d4).
Goal: extend the §5.5 cross-purpose-constraint footnote (currently Adult-only,
trained without the erase-layer architecture) to all three tabular datasets
(Adult + HMDA + Diabetes) **under the current erase-layer pilot architecture**.
This addresses reviewer R3's strongest complaint (22–26 of 33 cross-purpose
attack cells flag in §5.4 of the submission).

---

## 1. Verified current state

### 1a. Cross-purpose constraint implementation — already on `main`

| Piece | File:lines |
|---|---|
| Config fields `cross_purpose_attrs`, `cross_purpose_threshold` | `pcrl/training/v2_trainer.py:139–152` |
| Constraint registration (low-K joint + high-K per-class OvR) | `pcrl/training/v2_trainer.py:419–476` |
| Training-step forward on `h_concat` | `pcrl/training/v2_trainer.py:772–810` |
| Validation logging on `h_concat` | `pcrl/training/v2_trainer.py:984–998` |
| CLI flags `--cross-purpose-attrs`, `--cross-purpose-threshold` | `experiments/run_v2_dataset.py:521–538` |

The constraint is **fully dataset-agnostic**: cardinalities resolve from
`purpose_configs[purpose]["disallowed_attr_dims"]`, and every constraint shares
the same dual-update hyperparameters as per-pair constraints. No Adult-specific
hardcoding in the trainer.

### 1b. Adult-only pilot — outcome on file

`results/v2_adult_CROSSPURP/HEADLINE.txt` (3 seeds, 75 epochs, **no erase-layer**):

| Metric | Baseline R5 | CROSSPURP | Δ |
|---|---|---|---|
| per-pair R² ≤ 0.05 pass | 23/24 | 19/24 | −4 |
| h_concat R² (race) pass | — | 3/3 (mean 0.053) | — |
| h_concat R² (sex) pass  | — | 1/3 (mean 0.147) | — |
| h_concat R² (age_group) pass | — | 1/3 (mean 0.087) | — |
| Adult-only attack flags >1pp (15 cells) | 14/15 | 12/15 | −2 |

Honest framing already lives in `results/v2_adult_CROSSPURP/PAPER_PASTE.md`:
linear-R² constraint on `h_concat` is too weak to defeat MLP/XGB attackers on
sex/age_group, but works on race; per-pair lost 4 cells.

### 1c. Erase-layer architecture — NOT on `main`

`grep -rn "use_erase_layer\|--use-erase"` against `pcrl/` and `experiments/`
on this branch returns **zero matches** for the tabular path (only vision /
language ProLoRA hits). The tabular erase-layer code lives on branch
`erase-layer-pilot-2026-05-17` as 5 commits ahead of `main`:

```
0ca91ee Add NeurIPS rebuttal pilot plan: LAFTR with hard linear-R² ...
65dd5c0 Add erase-pilot vs Round 5/7 aggregator
71b813f Document erase-layer pilot next steps + two launch paths
04cb7ef Add 5-epoch CPU smoke artifact for the erase-layer pilot
96c03d0 Add §5.5 erase-layer architecture for tabular (rebuttal pilot)
```

Diff vs `main`: `experiments/run_v2_dataset.py` (+43), `pcrl/models/encoder.py`
(+33), `pcrl/models/lora.py` (+22), `pcrl/training/v2_trainer.py` (+115).
Key commit is `96c03d0`; the other four are aggregator / docs / smoke /
LAFTR-baseline plan and are *not* required for our pilot.

### 1d. Erase-layer pilot baseline (the new headline)

From `results/rebuttal/erase_layer_pilot_aws/rebuttal/HEADLINE.txt`:

| Dataset | Strict R² pass (R7 → erase-layer) | Cleanly-compliant | Total wall |
|---|---|---|---|
| Adult | 21/24 → **24/24** | 1/24 → 0/24 | 8339 s (3 seeds) |
| HMDA  | 16/18 → **18/18** | 2/18 → 0/18 | 11041 s (3 seeds) |
| Diabetes | 17/18 → **18/18** | 2/18 → 0/18 | 9382 s (3 seeds) |
| **Total** | **54/60 → 60/60** | 5/60 → 0/60 | ~7.9 h |

The cross-purpose pilot starts from this strict 60/60 floor, not the Round-5/7
baseline.

---

## 2. Architectural decision: how to combine the two branches

**Recommended approach:** keep the new branch rooted at `main`, then **merge
the single erase-layer architecture commit (96c03d0) from the pilot branch**.
Skip the four trailing commits (aggregator, docs, smoke, LAFTR plan) — they
are pilot artifacts that don't belong on the rebuttal extension branch and
will conflict with our own aggregator.

```bash
git cherry-pick 96c03d0   # adds erase-layer code, leaves cross-purpose code intact
```

Rationale: cherry-picking gives us a clean composition `main + erase-layer +
new-cross-purpose-work` with one explicit provenance commit, instead of a
big merge that drags in aggregator/docs/smoke from the pilot. The cherry-pick
will need a conflict resolution in `experiments/run_v2_dataset.py` (cross-purpose
flags live there too) and `pcrl/training/v2_trainer.py` (cross-purpose code
lives there too) — both are additive and should resolve cleanly. I'll verify
on dry-run before committing.

**Alternative if cherry-pick conflicts are messy:** `git merge --no-ff
96c03d0` (single-commit merge). Both are reversible. I won't do either until
the plan is approved.

---

## 3. Implementation task breakdown

### Phase A — wire erase-layer onto cross-purpose branch *(local, ~1 h)*

| # | Task | Files | Verify |
|---|---|---|---|
| A.1 | Cherry-pick `96c03d0` from `erase-layer-pilot-2026-05-17` | `pcrl/models/encoder.py`, `pcrl/models/lora.py`, `pcrl/training/v2_trainer.py`, `experiments/run_v2_dataset.py` | `grep -n "use_erase_layer" experiments/run_v2_dataset.py` non-empty |
| A.2 | Smoke test: 5-epoch CPU run with `--use-erase-layer --cross-purpose-attrs race` on Adult, confirm no constructor errors and both code paths fire (look for both `[V2/INIT] cross-purpose attr=...` and erase-layer fit logging) | n/a | log lines present |
| A.3 | Commit (one commit, message: `Combine erase-layer architecture with cross-purpose constraint`) | n/a | `git log --oneline -1` |

### Phase B — generalise eval script to all three datasets *(local, ~2–3 h)*

The current `scripts/crosspurp/run_eval.py` is Adult-hardcoded
(`ATTRS = ["race","sex","age_group","marital_status","income"]`,
`CROSS_ATTRS = ["race","sex","age_group"]`, AdultDataset loader).
Approach: rewrite as a `--dataset` switched script that:

- imports the right `Dataset` class + `get_*_purposes()`,
- reads `ATTRS` and `CROSS_ATTRS` from purpose configs (auditor uses every
  declared `disallowed_attrs` for `ATTRS`; the cross-purpose subset is a CLI
  arg with a sensible default per dataset),
- loads checkpoints from `checkpoints/v2_<dataset>_CROSSPURP_ERASE_s{seed}/`,
- writes `results/v2_<dataset>_CROSSPURP_ERASE/results.json` + `HEADLINE.txt`.

| # | Task | Files |
|---|---|---|
| B.1 | New script `scripts/crosspurp/run_eval_multi.py` taking `--dataset` flag | new |
| B.2 | Per-dataset default `CROSS_ATTRS` choices: Adult `[race, sex, age_group]`, HMDA `[ethnicity, race, sex]`, Diabetes `[race, gender, age_bucket]` (note: `age_bucket` K=10 triggers per-class OvR — 10 duals; confirms feasibility before committing) | n/a |
| B.3 | Aggregator `scripts/crosspurp/build_report_multi.py` producing a 3-dataset comparison table + `PAPER_PASTE.md` | new |
| B.4 | Sanity test: run B.1 on existing `v2_adult_CROSSPURP` checkpoints to confirm I reproduce the 19/24 + 12/15 numbers before launching training | n/a |

### Phase C — defaults / hyperparameter sweep decisions *(0.5 h thinking, no compute)*

Decisions I'll lock in before launch, with rationale:

| Knob | Adult | HMDA | Diabetes | Why |
|---|---|---|---|---|
| `--epochs` | 200 | 200 | 200 | Match erase-layer pilot standard; previous Adult CROSSPURP used 75 ep, which the §5.5 honest framing already flagged as undersized |
| `--seeds` | 0,1,2 | 0,1,2 | 0,1,2 | Match erase-layer pilot |
| `--cross-purpose-threshold` | 0.10 | 0.10 | 0.10 | Same as prior Adult pilot (so Adult numbers are comparable up to the erase-layer architectural change); raise to 0.15 only if pilot shows all-fail |
| `--cross-purpose-attrs` | race sex age_group | ethnicity race sex | race gender age_bucket | Mirrors which attributes appear in the §5.4 cross-purpose attack table for each dataset |
| `--use-erase-layer` | True | True | True | This is the headline architecture |

No threshold sweep in the pilot — first establish whether the constraint binds
across datasets at τ=0.10. If it's universally too tight, sweep in a follow-up.

### Phase D — training runs *(AWS, see §4)*

| # | Task | Compute |
|---|---|---|
| D.1 | Launch Adult on AWS (3 seeds, 200 ep, erase + cross-purpose) | ~3 h |
| D.2 | Launch HMDA  (3 seeds, 200 ep, erase + cross-purpose) | ~4 h |
| D.3 | Launch Diabetes (3 seeds, 200 ep, erase + cross-purpose; **note: age_bucket K=10 adds 10 duals — slowest cell**) | ~4 h |

I'll use the same instance type / user-data scaffold as the erase-layer pilot
(memory `project_erase_pilot_run.md` shows the AWS launch pattern). I'll
batch Adult + HMDA on one instance and Diabetes on a second to parallelise,
following the proven pattern from the pilot.

### Phase E — eval + report *(local, ~2 h)*

| # | Task | Output |
|---|---|---|
| E.1 | Run `run_eval_multi.py` on each dataset's checkpoints | `results/v2_<d>_CROSSPURP_ERASE/results.json` |
| E.2 | Run `build_report_multi.py` | `results/rebuttal/cross_purpose/HEADLINE.txt`, `comparison_table.tex`, `PAPER_PASTE.md` |
| E.3 | Compute three deltas per dataset: per-pair pass count (vs erase-only baseline), h_concat R² pass count (vs τ=0.10), attack-flag count (vs erase-only baseline) | n/a |
| E.4 | Write honest-framing prose for §5.5 — three-dataset version of the existing Adult `PAPER_PASTE.md`, including any dataset where the constraint actively *hurts* per-pair compliance | `PAPER_PASTE.md` |

---

## 4. Compute cost estimate

Scaling from erase-layer pilot wall times (1 erase-layer-only seed per
dataset on the same AWS instance type):

| Dataset | erase-only / seed | est. erase + cross-purpose / seed | 3 seeds |
|---|---|---|---|
| Adult (3 cross-purpose duals) | 2780 s | ~3500 s (×1.25, extra duals + auditor forward) | ~3 h |
| HMDA (3 cross-purpose duals) | 3680 s | ~4600 s (×1.25) | ~4 h |
| Diabetes (3 cross-purpose; 1 high-K → +10 OvR duals) | 3127 s | ~5000 s (×1.6, big OvR fanout) | ~4.2 h |

Total: ~11 h of single-instance wall-time, parallelised across 2 instances
as ~5 h real-time. AWS cost on c5.4xlarge ≈ $0.68/h × 2 instances × 5 h ≈
**$7 + buffer ≈ $10–12 total**. (Pilot rebuttal-budget memory shows
$1.75–$1 for prior runs; this is the largest single rebuttal pilot but still
under $15.)

CPU smoke (Phase A.2 + B.4): ~30 min local, free.

---

## 5. Risks

1. **High-K Diabetes age_bucket may saturate.** 10 OvR duals on `h_concat` at
   τ=0.10 may all hit `lambda_max` and never bind (Round-7 saw this with
   per-pair age_bucket). Mitigation: log `lambda_saturated_cells` count;
   if all 10 saturate by epoch 50, kill the run early and either drop
   age_bucket from CROSS_ATTRS for Diabetes or escalate threshold to 0.15.
2. **Cross-purpose constraint may tank per-pair compliance below
   60/60 floor.** Prior Adult run lost 4 cells. If three-dataset extension
   loses >5 cells total per-pair, paper paragraph must lead with that
   tradeoff. Report it whichever way it lands — no cherry-picking.
3. **Cherry-pick conflicts.** Both branches modify
   `experiments/run_v2_dataset.py` and `pcrl/training/v2_trainer.py`. Conflicts
   are likely additive but I won't know until I try. Fallback: `git merge`
   approach in §2.
4. **Linear-R² constraint is too weak vs MLP/XGB attackers**, as already
   documented in the Adult-only outcome. The extension may show the same
   pattern on HMDA/Diabetes — i.e., the cross-purpose attack flag count
   barely moves while per-pair compliance pays a real price. That outcome
   would weaken the rebuttal response to R3; I'd flag it immediately and we
   can decide whether to fall back to "constraint shown infeasible at scale,
   report negative result" or escalate to kernel-HSIC on `h_concat`.
5. **AWS instance availability / interruption.** Pilot pattern uses
   on-demand c5.4xlarge; previous pilots launched cleanly. Standard
   mitigation: 6 h hard cap, periodic `aws ec2 describe-instances` poll,
   IAM profile `pcrl-bios-s3-writer` required per memory note.
6. **The §5.5 honest-framing pressure goes up with the architecture
   swap.** Previous Adult `PAPER_PASTE.md` says "outcome (b/c): partial
   reduction." Three-dataset version may need to be even more cautious
   if results diverge between datasets — e.g., race works on Adult but
   ethnicity fails on HMDA. I'll write the prose to surface every dataset
   where the constraint underperforms.

---

## 6. Out of scope (explicit)

- Kernel-HSIC or vCLUB MI on `h_concat` (the obvious next-step extension if
  linear-R² fails — but a different pilot).
- Threshold sweep (τ ∈ {0.05, 0.10, 0.15, 0.20}).
- Cross-purpose attacks beyond LR/MLP/XGB (e.g., neural-net auditor with
  bigger capacity).
- Rerunning baselines without erase-layer for sanity — we already have those
  from the prior Adult pilot and the erase-layer pilot's R5/R7 baseline files.

---

## 7. Branch / commit strategy

```
cross-purpose-rebuttal-2026-05-18  (now @ 17ef7d4, branched from origin/main)
  ├── A.1 cherry-pick 96c03d0       Combine erase-layer + cross-purpose
  ├── A.2 smoke artifact (optional, gitignored or under results/_smoke/)
  ├── B.1–B.3 multi-dataset eval scripts + aggregator
  ├── B.4 sanity reproduction of Adult CROSSPURP numbers
  ├── D launch artifacts (results/v2_*_CROSSPURP_ERASE/, gitignored ckpts)
  └── E rebuttal report (results/rebuttal/cross_purpose/{HEADLINE.txt, PAPER_PASTE.md})
```

Do NOT push to `origin` until E.4 is complete and reviewed.

---

## 8. Awaiting approval

Please confirm:
1. Cherry-pick approach (§2) vs merge approach.
2. Hyperparameter defaults table (§3, Phase C).
3. Per-dataset CROSS_ATTRS choices, especially Diabetes `age_bucket` (the
   high-K risk in §5.1).
4. Two-instance parallelisation for AWS launches (§4).
5. Anything else you want changed before I start Phase A.
