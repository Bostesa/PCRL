# Protocol amendments

Each entry gives the UTC time, the outcome-access status at the time, the trigger, and the change. Operational fixes never change margins, endpoints, nominees or comparison rules.

## M1 — 2026-09-24 ~17:46Z (pre-registration commit; recorded in DESIGN_SPEC.md)

- **Trigger:** the independent math review.
- **Changes:**
  - policies are D17-anchored and switched;
  - policy task costs are cross-fitted;
  - DET_SEL1 and DET_SEL4 are added;
  - interpretation rules are registered.
- **Outcome access:** none.

## M2 — 2026-09-24 ~18:10Z (after the engineering smoke, before any candidate fit)

- **Outcome access:** a method-implementer engineering smoke on a 30% household subsample of anchor 0. It covered only nuisance_train, audit_fit, coefficient_split and inner_selection rows. It did not touch inner_check or outer. It produced no scientific numbers, and none are recorded.
- **Trigger:** a structural diagnostic, not a candidate comparison. The switched policies disagreed with D17 for about 90% of coefficient people, while the task-only policy's task loss was about equal to D17's.
- **Cause:** the registered oracle fits 17 *independent* per-token regressors of the raw priced cost g_i(z). Each carries its own estimation noise, and their argmin selects that noise rather than signal. The 0.002-nat switching margin does not guard against it, because the between-token spread in raw per-person costs is much larger than 0.002.
- **Change:** each per-token regressor now targets the paired improvement over the D17 token, Δ_i(z) = g_i(z) − g_i(D17(x_i)); for the D17 token this is identically 0. The policy deviates to argmin_z Δ̂(z|x) only if min_z Δ̂(z|x) < −tau, with tau = .002 nats, and ties go to D17.
  - The pairing removes the label noise common to all tokens.
  - Boosting shrinkage then pulls predictions toward the mean paired difference rather than toward an arbitrary level. D17 becomes the default unless there is predicted evidence.
- **Unchanged:** features, learner hyperparameters, roles, cross-fitting and prices.
- **Scope:** the same oracle change applies to RD_TASK and RD_PRIV wherever they use the cost-regression oracle, so matching is preserved.
- **Not a response to any candidate or route outcome:** no candidate has been fitted on real data under the registered configuration.
- **Known limitation, recorded at M2 time (18:15Z; no new rule added):**
  - tau = .002 is a fixed margin. It does not scale with the oracle's estimation error. In a synthetic probe with token-specific noise of about 0.1–0.3 nats, a paired policy can still deviate broadly.
  - This is not guarded by a further rule, for two reasons. The fixed-bank LP assigns column weights, and it gives zero weight to a column that does not lower the priced objective on coefficient rows. And the refit attackers and the independent audit judge the result.
  - Post-M2 structural smoke diagnostic (anchor 0, 30% subsample, coefficient rows, no losses reported): the policies differed from D17 for about 10% (task_only), 66% (local_priced, which runs on the registered all-zero-dual fallback price), 13% (coalition_priced) and 13% (all_priced_x2) of people. There was within-T32 token variation in 28–32 of the 32 states.

## O1 — 2026-09-24 18:19Z (operational, pre-fit; outcome access: none)

- **Audit seed base.** Set to 26000, the AR value that AR `external_audit` hard-codes, replacing the 36000 the infra draft had used. This way the main and J panels on this host get identical H-only reference slates. It is a matching fix and no scientific rule changes.
- **J continuity reference.** J is run through AR `external_audit --methods J --slate standard`, unchanged. The J values come from the sanitized anchors, plus the 32 KB external index (SHA pinned to AR's). Outer J scoring goes through this study's lock/unlock gate.

## M3 — 2026-09-24 18:36Z (pre-fit matching repairs from the baseline owner's smoke; outcome access: engineering smoke on a 30% a0 subsample, non-inner_check and non-outer rows only; no candidate fitted under the registered configuration)

1. **Closing refit for NM and T32.** After the last alternation round, best-response attackers are fitted on that round's own law. They are added to the bank and every cut is rebased. The AR final-round rule (lowest inner_selection task loss among rounds feasible on the final bank) is then applied on this enlarged bank. Without this, NM's last round was never judged against attackers refit on itself, while every RD and ADV candidate was.
2. **ADV route-specific checkpoint rules.**
   - ADV_B1 and ADV_B2 keep the registered task-selection rule and supply the **U-route** ADV representative.
   - New units ADV_B1_P and ADV_B2_P use the baseline owner's privacy-selection rule (`--select privacy`) and supply the **P-route** representative, under the same eligibility and ranking rules as PROTOCOL section 6.
   - Reason: under the task rule the smoke ADV always picked its near-D17 warm start. That would make ADV a vacuous P-route comparator.
3. **Baseline-strengthening changes accepted as registered**, disclosed in BASELINE_MATCHING.md section 3:
   - RD candidates are scored with a common receiver seed per unit;
   - RD_TASK's switch threshold is chosen on inner_selection from tau in {0.002, 0.01, 0.03, 0.1, 0.3};
   - RD_PRIV bisects tau when no mu is feasible, with D17 as the feasible limit and an alias flag;
   - ADV uses lr 3e-4 and weight decay 1e-3, adds the legal D17-token one-hot input, and warm-starts its decoder and adversaries with early stopping on inner_selection.

   These give the controls more opportunity than NM. That is the intended direction.
4. **Recorded, not changed:**
   - RD gets 3 rounds against NM's 6, and RD_PRIV keeps the round-0 dual direction. These favor NM and are reported as limitations.
   - ADV trains on nuisance_train ∪ coefficient_split and checks feasibility on coefficient_split.
   - Stochastic laws weaken the two exact hist_gb audit attackers through the min-leaf rule. The rule is identical for all releases.
   - Receiver-fit noise was about .004 nats on the smoke.
