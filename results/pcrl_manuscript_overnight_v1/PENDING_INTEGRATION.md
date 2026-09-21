# PENDING_INTEGRATION — stage R of the replacement study

**Status at closeout: NOT INTEGRATED, because the commit is not pushed.**

Terminal 1 committed `1c925fb3962e63a59a8901563c9ab82c4a74d046` ("Stage R closes the constrained ACS
fitting branch, with two corrected positives") on the local branch
`research/pcrl-stochastic-replacement-overnight-v1`. At closeout,
`git ls-remote origin refs/heads/research/pcrl-stochastic-replacement-overnight-v1` returns **nothing**:
the branch exists only locally. The standing rule is that a result is integrated only from a normal
pushed commit whose remote SHA and manifest verify, so nothing from it is in the paper.

Its status file is also stale relative to the commit: `STATUS.json` was written at
`2026-09-21T06:50:23Z` saying "registered; starting stage R", with all unit counts zero, while the commit
is dated `02:56:18 -0400`. Trust the commit, not the status file, and re-read both.

## Why this matters more than a normal pending item

The commit message reports that the **unquantized** standardised A-side view attains strictly lower
residence loss than the same-host `J` channel, with adjusted one-sided upper bounds below zero under the
frozen family-of-68 correction, aggregated over shared households. If that verifies, it **quantitatively
contradicts** the precursor's "already subsumed by `J`" reading — the reading this session had already
withdrawn as unsupported on precision grounds (`CORRECTION_LOG.md` N1).

Because that contradiction was visible before closeout, the paper was changed: §VII and the conclusion no
longer assert the "published output already carries the signal" explanation. They state the measured
result, name quantization loss and genuine redundancy as the two open explanations, and say a successor
experiment separates them. **No unpushed number appears anywhere in the paper.**

## Exact steps once it is pushed

```
cd /Users/nathansamson/PCRL && git fetch origin
git ls-remote origin refs/heads/research/pcrl-stochastic-replacement-overnight-v1   # must match locally
SHA=<full remote sha>
git show $SHA:results/pcrl_stochastic_replacement_overnight_v1/gates/G_R.json
git show $SHA:results/pcrl_stochastic_replacement_overnight_v1/stage_R/INTERVALS_ADJUSTED.json
git show $SHA:results/pcrl_stochastic_replacement_overnight_v1/stage_R/SUMMARY.json
```

Artifact keys to verify, not to trust:

1. **Recompute the two positives** from `stage_R/INTERVALS*.json`: the unquantized-view-versus-`J`
   residence difference and the 64-state-code-versus-`H` difference, both weightings, with adjusted
   one-sided upper bounds. Confirm the adjusted critical value (`z = 3.18` is claimed) against the frozen
   family size of 68 and Bonferroni — and see open item **M3**: that family size was not reconstructible
   from the documented factors at registration time.
2. **Confirm the aggregation unit**: 5,443 households, shared-household resampling across anchors, not
   independent per-anchor bootstraps (`STATISTICAL_PLAN.md` §2).
3. **Check the screen verdicts per family and resolution** (`pca32`/`zj` × `k=64`/`k=256`) and that the
   `k=256` fallback was the single registered one.
4. **Confirm no `Q` was fitted on ACS** and that stages A and Q are recorded as not triggered rather than
   failed.
5. **Check the identity control**: the `zj` unquantized view reproducing `J` with zero variance is an
   identity confirming wiring, and must not be reported as evidence (`STATISTICAL_PLAN.md` §9).
6. **Check the label-use record** for the unquantized-view probes, and that the residence result is
   labelled development, task-informed, on repeatedly used 2018 pools.
7. **Resolve the registered predictions** as the study states them (P1 refuted on both halves, P2
   confirmed), and record that the refutation was registered in advance.

## Then, in the paper

* §VII gains the separation of quantization loss from redundancy **as a measured result**, replacing the
  current "two open explanations" wording.
* The conclusion's open question narrows: if the A-side inputs carry residence signal beyond `J` but a
  64-state code destroys it, the obstacle is the code, not the contract.
* `CORRECTION_LOG.md` N1 gains a line: the subsumption claim moves from *unsupported* to *contradicted*.
* Claim ledger rows `S7-01`…`S7-03` are updated, and a new row records the stage-R outcome with its own
  commit.
* The abstract's "adds nothing measurable once the already-released service predictions are in the view"
  must be re-examined: it remains true of the **code**, but the unquantized result would make the
  sentence misleading without its qualifier. Note the venue rule — after abstract registration,
  substantial abstract changes are barred, so this is a reason to register the abstract only once this
  verifies, or to keep the current qualified wording, which survives either outcome.
