# MATHEMATICAL_REVIEW — overnight session

Three strands: (A) the elementary statements the paper relies on, (B) fixtures written to settle
specific inferential questions, (C) the review of the successor study's mathematics before outcomes.

## A. Statements the paper relies on

1. **Concatenation preserves the published output exactly.** Property of concatenation, checked bitwise.
2. **Inclusion fact.** For any target and any loss, Bayes risk under `[H, Z]` is at most that under `H`.
   Applied to a sensitive attribute, appending a deterministic channel **cannot reduce** optimal
   population recovery, and a constant appended channel has **exactly zero** population incremental
   disclosure. Consequences enforced in the text: no "appending protected us" reading; a strictly
   negative population increment is unavailable for an appended release, so a success condition
   demanding one is coherent only for a **replacement** release (`CORRECTION_LOG` N7).
3. **Risk difference equals `I(S;Z|H)` under Bayes prediction and log loss.** Finite validation-selected
   attackers do not attain those infima, so measured differences bound it in neither direction.
4. **Fitted-model conditional constraints are not population guarantees.** Binning a conditioning view
   can drop a full `log 2` of measured leakage to exactly zero (the precursor's own fixture). Hence the
   requirement that audits use unrounded continuous service predictions (sent as M2).

## B. Fixtures written this session

**B1 label leak** (`checks/b1_label_leak_fixture.py`). Conditioning on per-row log-loss deciles encodes
the label whenever the class is skewed. On a pure-noise label at the residence base rate (0.72) with
probe outputs carrying no information, a cross-fitted table conditioned on loss deciles reports 0.057
nats against a true conditional entropy of 0.590. The label-free control — deciles of the predicted
probability — reports 0.592. A balanced toy leaks nothing, which is why the defect survives casual
testing. This is the exact estimator shape used by the audited study.

**Replacement versus append** (`checks/replacement_vs_append_fixtures.py` §1). `H` constant, `J=(Y,S)`,
`T=Y`, independent fair bits. Appending `T` to `J` adds exactly zero task information; replacing `J`
with `T` preserves task utility in full and removes the sensitive bit entirely (0.693 nats). Therefore a
measurement of "does `T` add to `J`?" returning zero is consistent with `T` being a strictly better
release than `J`. An append test cannot reject a replacement channel.

**Additive-probe capacity** (§2). `J, T` independent signs, `Y = J·T`. The best score additive in `J`
and one-hot `T` gains `2.2e-16` over the prior, while the joint view determines `Y` exactly. A null from
one probe family is a statement about that family's capacity. This is a counterexample about inference,
not a claim that residence behaves like XOR.

## C. Review of the successor study's mathematics (before any outcome)

Full text sent as `REVIEW_REPLACEMENT_PROTOCOL.md`. Verified correct: row-stochasticity and
nonnegativity; conditional mutual information affine in the channel under the mechanism's Markov
property with a fixed fitted distribution, hence convex sublevel sets, with a linear objective for a
fixed cost table; the explicit exclusion of joint decoder/offset optimisation and task-MI maximisation as
convexity-destroying; absolute finite-model budgets never equated with empirical increments over `J`;
label-independent partitions with a fallback triggered by feature-cell counts alone; expected scoring
that averages **losses** rather than predictions; one sampled token as the released object, with
deployment randomness separated from published simulation seeds; household resampling with anchors
sharing people, so anchors are not independent replicates; and both success routes with margins fixed in
advance.

Three items raised: **(M1)** solver honesty — status, primal residuals, objective reconstruction,
optimality gap, and an independent recomputation of achieved CMI — is registered only for the `δ=0`
linear path and must cover `δ>0` before the constrained fits run; **(M2)** the registration should state
that audit attackers receive the unrounded continuous service predictions, never the partition cells;
**(M3)** the frozen Bonferroni family size of 68 is not reconstructible from the documented factors and
its arithmetic should be recorded before outcomes.
