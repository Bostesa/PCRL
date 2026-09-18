# PENDING_ADDENDUM — what to do when Terminal 1 publishes a completed result

**Status at the time this package was completed: no committed Terminal 1 result exists.** The
worktree `/Users/nathansamson/PCRL-terminal-a` is clean at
`c37807e4f568ef38e5528fc09c1506083278bf4d`, which is the *already integrated* nonlinear/rank study.
No new experiment directory, protocol, handoff file or running job belonging to that terminal was
found. **No Terminal 1 result is incorporated anywhere in this manuscript, and none is claimed.**
This paper stands on the two completed studies alone.

This supersedes the pending-addendum section of
`results/pcrl_evidence_review_v1/REVIEW_INDEX.md`, whose statement that the nonlinear/rank study
was "not available … fit phase still running" is stale: it is complete and integrated here.

## Where to look

* `<git common dir>/pcrl_parallel_handoff_v2/terminal_1/` — local handoff, if written.
* A published `HANDOFF.json` on Terminal 1's own branch — for coordination from a separate clone.
* Never a running log, a partially written CSV, or a status message. **Final committed evidence
  only.**

## Before adding a single claim, check all of these

1. **Protocol and registration.** Was a protocol frozen and hashed *before* fitting? Were
   directional predictions registered, and how did they turn out? Report the wrong ones first, as
   Study 2 did.
2. **Rotation invariance, per component.** The defect in Study 2 was a penalty that was not
   invariant under `W → WQ`. Verify invariance for **each** component and **each** normalisation
   separately — utility, every penalty term, and any whitening or trace normalisation — not only
   for the assembled objective. The outcome-free acceptance gate (measured rotation share < 0.10
   before any outcome is read) must have been evaluated *before* outcomes, or the claim that the
   defect is repaired is not supported.
3. **Baseline status, per baseline.** Classify each as: exact alias of an existing arm, a valid
   adaptation, infeasible under the constraints, or unimplemented. An unimplemented baseline is a
   limitation, never a benchmark. In particular, check the two constraints in
   [`RELATED_WORK_SCOPE.md`](RELATED_WORK_SCOPE.md) §3: a linear-erasure baseline may act only on
   the auxiliary channel `Z` if `H` is immutable, and its guarantee then has that scope; and no
   baseline in the main comparison may consume residence labels. A privileged-label variant is an
   oracle and belongs outside the main table.
4. **`H` unchanged.** Verify bitwise release identity on every view, as both studies did (210 of
   210). If any released number moved, the setting changed and the result is not comparable.
5. **Held-out labels.** Residence and commute labels must be absent from representation fitting.
   And residence remains a repeatedly used project endpoint, so it is still not a pristine unseen
   task, however the new arms were fitted.
6. **Attack exposure comparable.** Same families, same budgets, same selection pool, same scopes.
   Absolute and incremental recovery reported separately, negative increments unclipped, the
   routed-`H` control present.
7. **Negative results retained.** Check that no arm, seed or endpoint was dropped after the
   outcome was seen, and that the denominators are printed beside every count.
8. **Does it improve the intended practical comparison?** Not "did some endpoint move" — did the
   registered coordination rules pass, against both local controls *and* against `J`, with the
   residence criterion stated as the one-sided point rule it is?
9. **Evidence status.** Any 2017 numbers are **exploratory** — that seal is spent. **2016 must be
   unscored.** If a 2016 outcome, fit, model-scored value or outcome distribution appears anywhere
   in the new work, that is a protocol violation and must be reported as one, not integrated.

## How to integrate

```bash
cd /Users/nathansamson/PCRL-terminal-2-manuscript          # this worktree, not Terminal 1's
git merge --no-ff <terminal-1-branch>@<exact SHA>          # only the intended committed range
PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.make_assets_v2 --repo . --out papers/pcrl_manuscript_v2
PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.build_claim_ledger --repo . --out results/pcrl_manuscript_review_v2
PYTHONPATH=. python -m pytest tests/pcrl_evidence_review_v1 tests/pcrl_nonlinear_rank_v1 -q
(cd papers/pcrl_manuscript_v2 && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.build_integration_manifest --repo . --out results/pcrl_manuscript_review_v2
```

Then update, in this order: the claim ledger (new rows with their own data pool, attack scope and
evaluation status), the figures and tables, §8 or a new §9, the withdrawn-claims list, the
limitations, and **the title and abstract**. Run no scientific refit from this terminal; the build
and test steps above are integration checks only.

## Two outcomes, two ways to write them

* **A win.** State its actual scope — which pool, which attacks, which registered rule — and state
  the confirmation gap explicitly: a development win is not a confirmation, and 2016 is the only
  unspent year. Do not let a development result justify spending it.
* **Another negative.** It belongs in the paper. It does **not** license an impossibility
  conclusion: a finite collection of failed variants rules out no method family, and a maximally
  leaky channel is not an upper bound on attainable utility at a privacy constraint.

In either case, **do not keep a title or abstract that no longer fits the data.** The current title
and abstract are written for one confirmatory coalition result plus two negative method results; a
third result changes what they should say.
