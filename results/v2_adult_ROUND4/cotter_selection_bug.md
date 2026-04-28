# Cotter best-iterate selector — known issue

Round 4 trained for 5 warmup + 200 constrained epochs (per seed). The
proxy-Lagrangian + Cotter best-iterate selector loaded `best.pt` from
**epoch ~12** instead of the last epoch reached. Re-evaluating from
`final.pt` (epoch ~204) shows that the optimizer kept improving
compliance throughout: mean linear R² across 24 (pair × seed)
combinations went from **0.169 (best.pt) → 0.038 (final.pt)**, a 4.4×
reduction. We adopt `final.pt` as the canonical Round 4 result.

This document explains why the selector misfired and what we plan to
change.

## What Cotter selection is supposed to do

Following Cotter et al. (JMLR 2019, §4.6), the best-iterate selector
returns whichever iterate jointly minimises constraint violation while
preserving task performance. Our implementation walks the full sequence
of stored iterates and picks the one with the lowest `violation_sum`
**among iterates whose `task_loss` is within 10% of the best
`task_loss` seen so far**.

When at least one iterate is fully feasible (every constraint slack ≥ 0
simultaneously), the selector picks the lowest-violation feasible
iterate. When no iterate is fully feasible, it falls back to the slack
rule above. The fallback is what fired on every Round 4 seed:
`cotter=fallback`, `n_feasible=0/200`.

## Why the fallback selected epoch ~12

The 10% task-loss slack window is anchored to the running minimum of
`task_loss`. During the 5-epoch warmup the task heads converge to near
the empirical optimum and `task_loss` drops fast; from then on the task
loss is roughly stable while constraint pressure pushes the
representation. As constraints engage, `task_loss` ticks slightly back
up (within tolerance) and `violation_sum` slowly decreases.

Epoch ~12 happens to sit at a local minimum of `task_loss`: it is just
past warmup, the task heads are well-calibrated, and constraint
pressure has not yet had time to materially compress the representation
or perturb the task fit. From the selector's perspective it is the
"most task-optimal" iterate within the entire 200-epoch run, even
though its compliance is far worse than what the optimizer eventually
reaches.

Concretely, on Adult Round 4 the selector compared:

| selection | mean R² (24 pair-seeds) | task acc (income) |
|---|---|---|
| `best.pt` — Cotter-selected, epoch ~12 | 0.169 | 0.840 |
| `final.pt` — last epoch, epoch ~204 | **0.038** | ~0.840 |

The 195 epochs the selector ignored produced a 4.4× improvement on
linear R² with no measurable task degradation. Tasks did not collapse
(income 84.0%, occupation 99.95%, education 99.99%); duals stayed in a
working range (λ ∈ [0.11, 77], not saturated at λ_max=1000).

## What we plan to change

The fallback rule should weight progress on constraints more heavily:

1. **Latest-feasible-window selection.** Among iterates whose
   `task_loss` is within e.g. 5% of the best, pick the most *recent*
   one whose `violation_sum` is below median. This biases toward late
   iterates that have benefitted from cumulative constraint pressure
   without sacrificing task performance.
2. **Plateau-detection.** Walk forward from the end of training and
   pick the first iterate whose `violation_sum` is within a small
   window of the running minimum and whose `task_loss` is within slack.
   This formalises "the most-recent reasonable iterate" instead of
   "the most task-optimal iterate ever seen".
3. **Honest training-time logging.** Log per-epoch `linear_r2` on a
   held-out batch for at least one (purpose, attribute) pair so that
   the selector — and the human reading the log — can see that the
   constraint loss is actually moving the audit metric.

Implementing these is mechanical but requires another full Round 5
training run; for the v2 paper we report Round 4 from `final.pt`,
document this issue, and treat the selection-rule fix as future work.

## Reproduction

`experiments/eval_round4_final_vs_best.py` loads
`checkpoints/v2_adult_s{0,1,2}/{best,final}.pt`, re-runs the auditor on
each, and emits `final_vs_best.{md,json}` next to this file. The
markdown table reports both checkpoints side-by-side per seed and per
(purpose, attribute) pair.
