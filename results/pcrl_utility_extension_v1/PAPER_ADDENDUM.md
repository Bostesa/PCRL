# PAPER_ADDENDUM — `pcrl_utility_extension_v1`

For Terminal 2. Everything here is 2018 development on repeatedly used pools. Cite
`RESEARCH_DECISION.md`, `PILOT_GATE.md` and `VALIDATION.md`; counts are in `TIER_STATUS.md`.

## What can be said

1. **A new, cleanly negative pilot.** Appending a small learned channel `R` to the frozen `J`
   release adds capability on the declared proxy (10-45% reduction in residual teacher-reconstruction
   MSE, every configuration, all three anchors) and passes the historical source allowances, but no
   configuration held the additional measured sensitive recovery over `J` within the declared .001-nat
   screen; the best was +0.0016 nats at the lowest capability gain. Utility and disclosure did not
   separate at any beta tried.
2. **The direction of the tradeoff is measured, not assumed.** Reconstruction falls monotonically
   from 0.451 (unprotected) to 0.111 as beta rises, while the worst sensitive increment falls from
   +0.036 to +0.0016. The unprotected control is the cleanest illustration: best capability, largest
   disclosure.
3. **No coalition-specific benefit on this mechanism either.** At matched beta, the coalition policy
   C1 is not better than the strength-matched local control L2 (at beta = 30: worst increment +0.0049
   vs +0.0016). This repeats the predecessor pattern in a new mechanism and is worth one sentence.
4. **The previous study contains no utility-first candidate.** Re-scored under a new utility-first
   criterion with a correctly enlarged family (m = 2480), 0 of 124 configurations pass against `J`,
   0 against `leace_A0`. The original negative verdict stands; this removes the "maybe a near miss was
   hiding in it" reading.
5. **A stronger attacker was sought and not found, again.** Eight prespecified recipes; none beat the
   standard slate on the frozen rule. State as "added stress strength was not established", never as
   "the release survived a stronger attack".
6. **An inclusion fact worth stating once.** Appending a deterministic function of already-permitted
   inputs cannot reduce optimal utility or optimal sensitive recovery; only finite-learner measurements
   can move either way. Any "appending R protected us" reading is wrong by construction.

## What must not be said

* Not "the extension is nearly compliant". It failed the declared screen at every configuration; the
  smallest gap is +0.0016 against .001, and buying it cost almost all of the capability gain.
* Not "protection improves with beta" as a general law — this is one finite family, one width (r = 2),
  three anchors, one budget.
* No claim about r = 4 or r = 8: **those were never run** (the pilot gate forbade the expansion).
* No confidence statement for the pilot: nothing was nominated, so no adjusted intervals were computed.
* Numbers here should not be differenced against the previous study's published audit values at a
  precision finer than ~0.003 nats (`VALIDATION.md`, cross-architecture refit noise).

## Blocking items from Terminal 2, as observed

* **One correction level** was declared before fitting for all primary comparisons (`PROTOCOL.md`
  section 7). No primary comparison was reached, so no level was applied selectively.
* **Measured rank and support**, never schema counting: LEACE-on-R was defined to record realised
  projection rank and per-class support; it belonged to Tier 3 and was not run.
* **Planned vs fitted vs distinct vs duplicate** are reported separately (`TIER_STATUS.md`).
* **Projection form**: not applicable here (this study appends a channel; it does not project one).
