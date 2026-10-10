# Pre-refit loss audit and code witness

R2's queued `R2_PRE_REFIT_LOSS` unit completed on the admitted 2,235
INNER_SELECTION people (2,234 groups) for both fixed source seeds. Its
508-row aggregate table is `LOSS_DECOMPOSITION.csv`; the public receipt is
`LOSS_AUDIT_RECEIPT.json`. The worker verified old F/T/D-K release bytes,
restored model packages, aligned rows/groups, and exact deployed output
reconstruction. It also decoded W_Q and W_TSTAR hard ten-bit releases from
their fixed public tables and verified their saved probabilities and
decisions exactly. It did not read an old assessment person array or fit a
new model/attacker.

The public CSV SHA-256 is
`2ffac156d033c4bc80c5724a68942b473235aee1af73ddbcea91cfb2d9888950`.
R2 independently recomputed every CSV difference row from its reported
absolute rows, and every aggregate metric in the receipt agreed with its
CSV row to `1e-12`. The receipt's private hash-only counterpart is retained
under owner-only custody. Source pins and the queue unit are in the receipt.

## Unchanged point utility caps

Each cell below refers to **both** U0 and Ucal references for the named
seed/task: accuracy loss at most 0.010, natural-log NLL excess at most
0.010, and summed multiclass Brier excess at most 0.005. Boundary equality
passes. These are development point checks, without intervals.

| Arm | Seed 0 income | Seed 0 occupation | Seed 1 income | Seed 1 occupation |
| --- | --- | --- | --- | --- |
| Continuous F | pass | pass | pass | pass |
| Old task-only ten-bit T | pass | **fail** | pass | **fail** |
| Old protected D-K | **fail** | **fail** | pass | **fail** |
| Historical W_Q bank, ten-bit identifier | pass | pass | pass | pass |
| Historical W_TSTAR bank, ten-bit identifier | pass | pass | pass | pass |

The strongest adverse old T result is seed-zero occupation: NLL excess is
0.025228 versus U0 and 0.026591 versus Ucal; Brier excess is 0.010329 and
0.009682. Seed-one occupation also loses 0.017002 accuracy and exceeds both
loss caps. D-K increases seed-one occupation NLL by 0.033061/0.034602 and
Brier by 0.016800/0.016334 against U0/Ucal. The continuous F head meets all
point caps, so the frozen 16-dimensional task latents preserve useful task
information under this comparator.

The W banks preserve the original class decision on every audited INNER
row. Their largest occupation NLL excess against Ucal is 0.004389 for W_Q
and 0.004176 for W_TSTAR; their largest occupation Brier excess against
Ucal is 0.001632 and 0.002201. Both lie inside the unchanged point caps.
Their global public banks contain 16 income and 321 occupation states, so
one global ten-bit identifier per recipient is sufficient for these fixed
tables (minimum 4 and 9 bits respectively). On INNER, occupation occupies
264–275 states, compared with old T's 168/176. No extra class/source token
is appended to the disclosed ten bits. This shows a ten-bit code can meet
these specific current-role **point** utility constraints under the bank's
score-routing contract. It does not show that the old or new latent mapper
can learn such a code, and it says nothing yet about privacy recovery.

## Loss decomposition

The old task-only hard T loses occupation NLL relative to F by 0.024361
(seed zero) and 0.018916 (seed one); Brier increases by 0.009005 and
0.009231. Its uncalibrated teacher KL on occupation is 0.034128/0.030161,
compared with F's 0.004516/0.004289. The same pattern appears in actual
true-label NLL, not only in teacher matching. Held-out global temperature
changes these values modestly and does not repair old T occupation utility.

The D-K privacy term adds occupation NLL of 0.006217 and 0.011853 relative
to old T for seeds zero and one. These are matched-row diagnostics, not
isolated causal effects: F uses a different continuous head/fitting schedule,
and D-K changes the training objective. The old T receipt does not retain
its per-epoch validation curves, so these numbers and its 15/20 selected
epochs do not establish convergence. See `OLD_TRAINING_CONVERGENCE_AUDIT.md`.

The locked next unit is the two-seed, ten-bit task-only T0 refit using a
longer predeclared hard-forward training allowance. If its 24 point clauses
fail, the protocol permits the predeclared 16-bit ladder step. W bank utility
supports the capacity premise for a score-routed public code; it cannot be
silently substituted for the latent learner or for a protected result.
