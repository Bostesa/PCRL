# PENDING_INTEGRATION — the fresh-year prospective comparison

**Status: nothing from it is in the paper, because it has published neither a protocol nor an outcome.**

Terminal 1's status file (`pcrl_finish_handoff_v1/terminal_1/STATUS.json`, written 2026-09-22T21:30Z)
records study `pcrl_final_prospective_v1` on branch `research/pcrl-final-prospective-v1`, phase
"Tier A started ... no 2016 outcome accessed", with **`protocol_commit: null`** and
**`evidence_commit: null`**. The branch exists locally and is not on `origin`.

Because `protocol_commit` is null there was nothing to review; the protocol review requested of this
terminal could not be performed and must not be reported as done.

## When a protocol is committed — review it for (before any outcome)

1. Frozen $Q$ and $D_{17}$ hypotheses, unchanged release objects, and the full control set.
2. One shared household fit/validation/final partition across all anchors.
3. Fresh-year fitting restricted to auditors and downstream probes; **no release reselection**.
4. The two operating-point conjunctions, each with two task and eight privacy clauses. Candidate-level
   $\alpha=.025+.025$ controls the declaration over the two candidates **provided component tests have
   their stated size**; the bootstrap gives nominal control under its assumptions, not an exact
   finite-sample guarantee. The intersection–union argument within a conjunction is valid; component
   bounds are pointwise, not simultaneous.
5. Secondary matched-control intervals in a separate, disclosed family with no global error-control claim.
6. Loss-difference signs, $H$-baseline cancellation, exact expected one-token loss, and **no
   loss-of-average-probability substitution**.
7. Continuous $H$ in attacks, appropriate ancestor predictors, the full race schema, and no token or
   anchor pseudoreplication.
8. A technical-validity process that does not halt full-panel evaluation merely because a result is
   unfavourable.

## When an evidence commit is published

```
cd /Users/nathansamson/PCRL && git fetch origin
git ls-remote origin refs/heads/research/pcrl-final-prospective-v1     # must match the local SHA
SHA=<full remote sha>
git show $SHA:results/pcrl_final_prospective_v1/HEADLINE.json          # or the published equivalent
```

Then: verify protocol-lock timing against the first outcome, data provenance, release hashes, chosen
auditors, partition rules, incidents and amendments; recompute both primary decisions and every cited
contrast from the per-person aggregates or bootstrap-ready household contributions; **do not fit another
attacker to verify a published number**.

## Outcome map, fixed in advance

| outcome | what the paper says |
|---|---|
| $Q$ passes its $J$ conjunction | a transported improvement for the selected stochastic release under the tested attacks; matched-control attribution stays separate |
| $D_{17}$ passes | a simpler task-directed release reaches the operating point — **not** disguised as a stochastic-optimisation contribution |
| both pass | describe both; use matched-control evidence to assess the added machinery |
| neither passes | report the task result and the exact unresolved or adverse privacy bounds; **unresolved does not become harm** |
| the year was not defensibly unused, or scoring was technically invalid | report that limitation with no fresh-year success language |

The fresh-year analysis would be prospective **conditional on** the 2018-fitted releases and freshly
selected audit models. It is not a guarantee about different people, future time, all attackers or
training-algorithm stability; a supervised task must not be described as unseen; and primary and secondary
intervals do not share one global 95% coverage property.

Any integration must stay inside the registered abstract and title scope, which were written to accommodate
every outcome above.
