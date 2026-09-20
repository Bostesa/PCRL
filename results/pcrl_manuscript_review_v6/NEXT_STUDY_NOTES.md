# NEXT_STUDY_NOTES — kept out of the PDF

Study 6's ladder is closed, not paused. Restarting it would re-audit the same releases. Anything below is a
*new* prospective study, because the outcome of this design is now known and changing a knob after seeing it
is outcome-driven.

## What the evidence actually constrains
* At r = 2 with these budgets and this adversarial term, capability and added disclosure fall together in all
  three policies; the closest point missed the .001 screen by +0.0006 (unweighted) while retaining 0.111 of
  0.451 available capability.
* The frontier is monotone in beta. A larger beta continues toward "release nearly-untouched J", which is the
  no-extension baseline, not a success.
* Nothing is known about r ∈ {4, 8}: Tier 3 never ran.

## If a follow-on is proposed, it needs (register before fitting)
1. A stated reason why a *different family* (not a larger beta on this one) should separate capability from
   disclosure — e.g. a narrower R, a different residual target, or a constraint applied at fit time rather
   than as a penalty.
2. The same release contract: H and Z_J byte-identical, R a function of permitted A-side inputs only, no
   out-of-fold lookup.
3. Both increments (over H and over J) on every endpoint, both weightings, with one declared correction level.
4. A utility-first decision rule with one-sided adjusted upper bounds, the .001-nat allowance named as an
   allowance, and a non-coalition extension included as the coalition-specificity counterexample.
5. Same-host references for every comparison (audit refits do not transport below ~0.003 nats).
6. 2016 stays sealed; any positive development result earns a prospective candidate, never a confirmation.

## If Terminal 1 reopens work
Read `pcrl_parallel_handoff_v5/terminal_1/STATUS.json` for the evidence commit, fetch only small reports via
`git show <sha>:<path>`, and model the verification on `checks/verify_study6.py` (re-derive decisions from raw
per-anchor numbers rather than reading recorded flags).
