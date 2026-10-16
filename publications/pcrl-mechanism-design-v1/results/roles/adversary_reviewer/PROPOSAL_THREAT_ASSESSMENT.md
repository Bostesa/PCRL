# Threat-model assessment of the two final proposals

Adversary and methodology reviewer, 2026-10-16, after the reveal. I assessed:

- `roles/designer_1/PROPOSAL.md`, "MMP", sha256 `2b7273ae…`;
- `roles/designer_2/PROPOSAL.md`, "RCCS", sha256 `83653f2d…`.

Both hashes match `DECISION_LOG.md`. I also read designer 2's `proto_finite_conditional.py` to see what its arms
compute. I ran nothing and assessed the threat model only: the mathematics is the verifier's job, and priority is the
prior-work reviewer's.

**Yardstick.** Under the primary contract, any injective deterministic release has Bayes recovery AUC*(S;V) =
AUC*(S;X). A claimed benefit is therefore a fitted-reader (learnability) claim unless the release **merges** input
cells that differ in P(S|x).

## Summary

| Question | MMP (designer 1) | RCCS (designer 2) |
|---|---|---|
| 1. Contract | **Kept.** Each recipient gets a deterministic probability vector and argmax computed from the six fields. All comparators use the same contract. | **Kept.** Same contract. Two caveats: the occupation service must also evaluate the income teacher and five SEX-propensity models, and its correction targets only the exact paired income release (T_TREE). Changing recipient 1's output mis-targets it, as happened with H_LJ. |
| 2. SEX or prohibited input, state, randomness, interaction at deployment | None. SEX-aware trees are evaluated at the constants 0 and 1 and mixed with the fixed π̄. No person's SEX is read. | None read, but deployment **computes an inferred SEX score π̂(x) for every person and steers the occupation output with it**. The package therefore ships a ready-made SEX predictor. This is legal under the contract, but it is a person-specific use of inferred SEX (designer 2 disclaims individual fairness). |
| 3. Survives the defense-aware refit? Removal or obfuscation? | **Partly removal.** No critic is fooled; the term Δ is deleted by an identity. Where the fitted SEX-aware trees ignore SEX-only fields, cells merge and the Bayes reduction is real. Everywhere else, including within-SEX effects of task-relevant proxies and any fitted dependence on proxies, the effect is learnability. Stated honestly (§3.5, §9), with one overstatement in row 6. | **Obfuscation.** The release is injective, as the proposal says (§3.5). The mechanism adds a π̂-driven shift chosen to cancel the **linear and quadratic** partial covariance of occupation with S given income. Readers refit on the defended releases, nonlinear ones included, are exactly what it does not target. In designer 2's own exact law the unpooled shift raises Bayes recovery (row 6). |
| 4. Untested access most likely to break it | Model access or chosen-input probing, which invert the full-precision output to its input cell and reach Bayes recovery wherever cells did not truly merge. Better readers or larger auxiliary samples exploit the residual through task-relevant proxies (failure mode 1). | Better or larger readers, which can learn above-second-order structure. Recipients holding **partial inputs or the true income label**, which can explain away z2 and isolate the injected π̂-term. Model access, which inverts to x. Each could push recovery *above* T_TREE's, not just back to it. |
| 5. Fresh, role-separated readers and coalition bank | Yes. 84 fresh fits of the 7-reader slate on ATTACK_FIT/VALIDATION; the source report code; selection on MODEL_SELECT only. The 14 ignore-one aliases are implied by the code but not stated. | Yes. A frozen 7×3 slate on ATTACK_FIT/VALIDATION (42 fits per nominee); stopping on MODEL_SELECT only. Ignore-one aliases implied, not stated. The selection proxy (OUT_LR) is the reader family RCCS optimises against, so the OUT_HGB gate is essential. |
| 6. Overstatements | Minor (see below). | Several (see below). |

## MMP: notes

- **Where the protection is genuine.** m = (1−π̄)q⁰ + π̄q¹ has no π(x)-dependent mixing weight. Compared with the
  SEX-blind teacher, it should depend *less* on fields that matter only through SEX. That is the right direction:
  pooling, not injection. Exact pooling still needs the fitted CatBoost ensembles (800 depth-4 trees, with SEX
  appended) never to split on such fields. With full-precision outputs, even tiny noise splits keep cells distinct for
  a Bayes reader. "Those cells genuinely merge" (§3.4) is therefore an aspiration until measured.
- **Overstatement.** §3.6: "No π-term remains in either output for the other to decode." True of the additive term,
  but the linked pair can still triangulate x, and hence π(x), through within-SEX effects and any residual dependence
  on proxies. Failure mode 1 concedes the first route; the sentence should concede both. §3.3(3), "the signal lies in
  the *shape* of the vectors", is a reading of which selected reader won. It is a fitted-reader observation, not a
  measurement of Δ's share, and the proposal correctly lists that share as "not established".
- **Deployment.** The package embeds SEX-conditional task models but no SEX classifier; the α = 1 twin is used only in
  evaluation. That is lower insider exposure than RCCS.

## RCCS: notes

- **The mechanism adds SEX information and cancels it linearly.**
  - ψ is built from r = logit π̂(x) − ℓ̂(base wires): the SEX propensity that the base releases do *not* linearly
    carry.
  - A is chosen so that this injected term offsets the occupation output's S-covariance, measured by a closed-form ridge
    critic with quadratic features.
  - The population release then encodes more of x's SEX-only information than T_TREE does, hidden only from
    low-order readers.
- **Evidence and mechanism do not match.**
  - The Bayes gains in §11 come from **partition** arms. COND reaches 0.596 by *pooling* cells (`proto_finite_conditional.py`:
    "the occupation release is the calibrated posterior of a partition").
  - The deployed RCCS is a continuous, unpooled shift. Its exact analogue is the INJ arm, which in the feasible law
    attains AUC*X = 0.644 against the TASK pair's 0.615.
  - The proposal reports "INJ = AUC*X" but never says what that means: in its own law, the deployed analogue **raises**
    Bayes pair recovery above the unprotected task release.
  - The only support for a benefit to *fitted* readers is the linear-Gaussian model.
- **Overstatements.**
  1. §1: "the outputs are strongly correlated (R² ≈ 0.82)" and "the published reader AUCs show the effect". The R² is
     implied by Gaussian-equivalent d′² readings of *different selected readers' AUCs* (§3.4(i)); it is not measured.
     The same applies to §3.3(b), "90% of the pair's signal is the occupation output's information given income".
     Failure mode 4 and §9 concede this, but the summary states it as fact.
  2. §1 and §3.1 say the occupation output "loses the signal unique to occupation" and that the release-conditional
     SEX information is "removed". Only linear and quadratic partial covariance is targeted, and §3.5 itself says no
     Bayes information is removed.
  3. The §3.5 heading, "does not merely fool its own critic". The closed-form critic removes critic *lag*, but the
     defense is still tuned to a fixed reader family.
  4. §3.5: injectivity "is equally true of NEURAL_RAW and the trees". The proposal's own TASK arm pools cells, so tree
     teachers need not be injective. Whether T_TREE is injective on the populated support is UNVERIFIED. If it pools
     along SEX-only directions, RCCS un-pools exactly those.
- **Per-recipient threat.** By design the occupation-only AUC stays near 0.60, above MIN's predicted 0.58, and income
  stays at T_TREE's 0.569 against LOCAL_TREE's 0.558. RCCS buys coalition protection by giving up single-recipient
  protection relative to local defenses. That trade must be reported as a primary result, not a footnote.

## Conditions I would attach to either pilot

These are for a future registered pilot. None can run in this data-free campaign. Each uses MODEL_SELECT or ATTACK roles
only, and none needs DEV_REPORT labels or sealed data.
1. **Merge diagnostic (label-free), before any reader is fit.** On MODEL_SELECT, report how many distinct release
   pairs there are per distinct encoded input, for each arm and for T_TREE. Report each SEX-aware or propensity
   model's split usage on fields with no task value. This decides how much of a benefit can be Bayes-level, and
   whether RCCS refines T_TREE's partition.
2. **Learnability curve.** Refit the pair slate on 25, 50 and 100% of ATTACK_FIT, with ATTACK_VALIDATION selection
   unchanged. If recovery rises with auxiliary size markedly faster for an arm than for its comparators, the effect is
   obfuscation. This tests the "larger auxiliary samples" threat directly.
3. **A symmetric nonlinear gate.** Apply the OUT_HGB (and OUT_MLP) minus OUT_LR proxy gap on MODEL_SELECT to **every**
   arm, MMP included, not only RCCS.
4. **Report single-recipient recovery** for both views beside the coalition endpoint, and state the 21-candidate pair
   bank (7 pair readers plus 14 ignore-one aliases) explicitly in both plans.
5. **For RCCS only:** add a side-information diagnostic. On ATTACK roles, give a reader the occupation release plus
   one permitted field the recipient might hold (for example SCHL or WKHP). Report it descriptively, labelled as
   access outside the primary threat model.

**Bottom line (threat model only).** MMP's rationale is compatible with the defense-aware refit and partly removes
information. Its risk is that fitted trees do not truly pool, which would leave a learnability effect only. RCCS is
honest that its claims concern fitted readers, but its mechanism is linear cancellation of an injected SEX-propensity
term. Its own exact analogue raises Bayes recovery, and it is most exposed to the untested accesses: stronger readers,
side information and model access. If selected, it should be presented as an obfuscation result for a stated reader
family, never as removal.
