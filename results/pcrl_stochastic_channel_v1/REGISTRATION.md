# REGISTRATION — `pcrl_stochastic_channel_v1`

**Role-constrained stochastic release (RCSR).** Locked before any new outcome is read. Written
after `VERIFICATION.md` (source check) and `STAGE_A.md` (repairs), before any machinery is built
and before any 2018 number for a new release exists.

Starting state: branch `research/pcrl-stochastic-channel-v1` from
`0517c06a79995dc574c857f890a2d6601e7d3c01`; pilot reviewed at `ba531ab42`; Stage A at `14b7528a2`.

This study does not revise any earlier verdict. `pcrl_utility_extension_v1` remains Tier 2 FAIL.

## 0. Boundaries

* **2016 is sealed**: no transform, label, fit, score, plot, selection, archive entry or cloud
  bundle file. 2018 is the development pool for everything here; 2017 is not opened.
* Residence and commute never enter a fit, decoder target, checkpoint selection, gate or nomination
  **except** in the explicitly supervised arm of §5, which is labelled a *development diagnostic*
  and can never support a reserved-task claim (§3).
* `H_A` and `H_B` are byte-identical to the stored release on every pool, asserted per unit.
* **`Z_J` stays in the release and in every conditioning view.** J was actually released to these
  recipients and remains accessible; the experiment may not pretend it was revoked. J is
  additionally audited as a same-host comparator, as before.
* **No rate constraint.** No `I(T;Z) ≤ R` term is ever added. The convexity of §2 relies on a fixed
  output alphabet only; the LP/convex characterization is known to fail when a rate constraint is
  active (`VERIFICATION.md` E7.3). If a rate constraint is ever wanted, this registration is void.
* The object is never called a conditional privacy funnel (`VERIFICATION.md` E5).

## 1. Question

Can a **randomized** finite release, optimized under explicit conditional-disclosure constraints,
reach an operating point that the pilot's deterministic appended channel could not — additional
useful capability beside the frozen service predictions, with additional measured sensitive
recovery over J within the declared budget?

Randomization is not assumed to help. The fixtures show it *can* strictly beat every deterministic
map in a finite model; they say nothing about ACS.

## 2. Mechanism and its assumptions

Release: `wire/A = [H_A, Z_J, onehot(Z)]`, `wire/B = H_B`, `wire/AB = [wire/A, H_B]`.

`T = g(x)` is a finite code over the permitted A-side inference inputs (PCA_32 through J's frozen
standardiser). No protected label, no residence label, no row identifier, no `H_B` at inference.
`Q_{tz} = Pr(Z = z | T = t)`, `Q ≥ 0`, rows sum to 1. For role `r` with protected attribute `S_r`
and existing view `C_r`:

```
minimise    sum_{t,z} p(t) Q_tz d(t,z)
subject to  I_Q(S_r; Z | C_r) <= delta_r   for every role r
```

`C_A = (H_A, Z_J)`, `C_AB = (H_A, Z_J, H_B)`.

**Assumptions, stated as assumptions.**

1. `Z — T — (S, C)` is Markov. This is the mechanism definition, not an empirical claim.
2. The program is solved on a **fitted finite model** `p̂_r(s, c, t)` in which `c` is a prespecified
   *partition* of the continuous `(H_A, Z_J[, H_B])`. Every guarantee is a statement about that
   fitted model. It is **not** a population statement and **not** a statement about continuous `H`
   (`VERIFICATION.md` §E8; the review's XOR fixture shows `I(S;Z|bin(H)) = 0` does not imply
   `I(S;Z|H) = 0`).
3. Convexity holds for fixed `d`, fixed alphabet, fixed `p̂`. Optimizing the codebook, optimizing
   decoders jointly, or maximizing a utility mutual information each destroy it and are out of
   scope.

**Released object.** The sampled token, as a fixed public one-hot. Never `Q[t,:]`, never its logits,
never an expected prototype. This is a gate, not a preference (§6 G3): the review's own fixture
records the probability-vector release leaking 0.4748 and 0.2499 nats in the two toys where the
sampled token leaks exactly 0.

**Repeated access.** One sampled release per individual, reused. Fresh independent draws are a
different mechanism and are not audited here. No composition claim is made: marginal
`I(S;Z|C) ≤ δ` does not imply `nδ` over a history.

## 3. Label use, and what each arm can support

| arm | labels used to build the release | what a success establishes | what it can never establish |
|---|---|---|---|
| **supervised diagnostic** (`SUP`) | residence labels, to build a fixed cross-fitted decoder/action library and expected log-loss costs `d` | that *some* channel in this fixed input code meets the operating point **when told what usefulness means** | anything about a task excluded from representation design. Not reserved-task transfer. Not a reusable-representation claim. |
| **label-free control** (`LF`) | none; `d` is a prespecified representation-distortion cost | that the task-agnostic objective obtains the benefit | that `A0` reconstruction is the right target |

At inference both arms use permitted A-side inputs only. Evaluation households are excluded from
fitting in both. **Residence-guided design makes this a development study**: the reserved-task label
is not restored by withholding it in one stage only, and no stage of this study may be described as
a blind or held-out task evaluation.

## 4. Registered directional predictions

Committed before any machinery exists. These are bets; they may be wrong, and a wrong bet is
reported as a wrong bet, not quietly dropped.

| # | prediction | confidence |
|---|---|---|
| P1 | The **modal outcome is "fitted constraints pass, full-continuous-H attacks fail"** — the convex program finds feasible `Q` meeting `δ` on the binned model, and the audited increment over J on full continuous `H` exceeds it. | 45% |
| P2 | Stage B (capacity, no privacy constraints) shows the quantizer retains a residence advantage over same-host J that is **positive but under .003 nats**. | 55% |
| P3 | Stage B shows **no** usable residence advantage at the declared resolution, stopping the study at the first gate. | 30% |
| P4 | The **supervised** arm reaches the operating point (utility gate and `.001` disclosure gate, both weightings, all four roles). | 20% |
| P5 | The **label-free** arm reaches the operating point. | 8% |
| P6 | A nonzero-`δ` feasible direction exists in the finite model for every role (the nullspace test at `δ = 0` is *infeasible* for at least one role, but small-`δ` feasibility holds). | 60% |
| P7 | Adding the Stage-A J anchors **raises** measured sensitive increments relative to a slate without them, on at least half the audited cells. | 75% |
| P8 | The paired-household precision check (§6 G0) shows `z × SE > .001` on at least one sensitive endpoint, i.e. the `.001` margin is **not** demonstrable at this sample size even if the point estimate is zero. | 50% |

Overall: **I expect this study to close negative.** It is run because the failure mode would be
informative and bounded, not because the candidate is likely to win. P1 and P8 are the predictions
I most want on the record, because both are reasons a *positive* fitted-model result would still
not be a result.

## 5. Stages

**Stage B — capacity, before any privacy optimization.** Fit one small prespecified A-side
quantizer (`|T| = 64`; exactly one predeclared higher-resolution fallback, `|T| = 256`) with no
sensitive and no residence labels in the quantizer itself. Evaluate whether the *unconstrained*
release retains a residence advantage over same-host J. Additionally solve the unconstrained
expected-task-loss problem within the same fixed input code and decoder/action library: its optimum
is a **ceiling on improvement for that fixed decoder family and fitted distribution**, not for
arbitrary predictors and not for the ACS population. Failure here is an operational warning about
this code, not a capacity theorem.

**Stage C — the two objectives through one solver.** `SUP` and `LF` as in §3, over one small
prospective budget ladder `δ ∈ {δ₁, δ₂, δ₃}` (calibrated in G0, not copied from `.001`), and
local-only versus local-plus-coalition constraint sets. References carried: `H`, `J`, an executed
strong erasure baseline, an unprotected code, and a simple randomization control (uniform `Q`
independent of `T`). Run matrix, distribution estimation, selection rule and stopping rule fixed
before any outcome is read.

**Stage D — audit the actual released token.** Full continuous `H`, both weightings, all forbidden
roles, source allowances, the historical attacker slate **plus the Stage-A J anchors**. Scoring
averaged exactly over the finite token distribution where possible; otherwise Monte Carlo
replication predeclared with its variance reported. Extra draws do not create additional households
or bootstrap units.

## 6. Gates

Machine-readable under `results/pcrl_stochastic_channel_v1/gates/`.

* **G0 — precision, before anything else.** From archived paired-household losses, compute the
  one-sided adjusted SE for each sensitive endpoint under the declared comparison family. **Report
  the margin that is actually demonstrable.** If `z × SE > δ` for the intended `δ`, that is recorded
  up front and the budget ladder is set accordingly; the margin is **not** relaxed later because an
  outcome was disappointing. This gate cannot fail the study; it can only constrain what may be
  claimed.
* **G1 — feasibility.** Per role and conditioning cell, the nullspace test
  `ker(A) ⊄ ker(B)` on the fitted model, with a declared tolerance and a sampling-sensitivity
  analysis of near-null directions. Reported for `δ = 0`; infeasibility at `δ = 0` does **not** stop
  the study, because the regime of interest is `δ > 0`.
* **G2 — capacity.** Stage B shows a residence advantage over same-host J at the declared
  resolution. FAIL → stop; report that this representation family is exhausted at the tested
  resolution. No grid expansion.
* **G3 — mechanism integrity.** `H_A`/`H_B` byte parity; inference-input restriction; probability
  integrity; and **the released array is a sampled one-hot token**, asserted per unit — not
  `Q[t,:]`, not logits, not an expected prototype.
* **G4 — solver honesty.** Feasibility residuals and a certified numerical optimality gap where the
  solver provides one. A solver success string alone is insufficient.
* **G5 — protection.** Audited increment over same-host J within the ladder budget on each of
  `A/SEX`, `AB/SEX`, `A/RAC1P`, `AB/RAC1P`, both weightings, on **full continuous `H`**. Passing the
  fitted-model constraints while failing this is a **failure**, and is reported as the modelled
  constraint not transferring — never as a certified mechanism.
* **G6 — source allowance.** Historical allowance on `income_binary`, `civilian_at_work`,
  `public_coverage`, unchanged from the predecessor.

**Stopping rules.** Finish the comparisons needed for each gate; do not expand after a failed gate;
do not change a threshold after seeing an outcome. A failed G2 stops at Stage B. A failed G5 stops
at Stage C/D with a negative report. Coalition benefit is claimed only against matched local
controls; a coalition win is not required to acknowledge a useful improvement.

## 7. Outcome table, fixed in advance

| outcome | interpretation | next action |
|---|---|---|
| Compression destroys task signal (G2 fail) | this representation family is exhausted at this resolution | stop; report |
| Supervised arm fails | no demonstrated reachable point in this finite family | stop; do **not** expand a grid, do **not** claim impossibility |
| Supervised passes, label-free fails | objective choice matters within this tested family | report; a reusable-method claim stays unsupported |
| Fitted constraints pass, full-`H` attacks fail (**P1**) | modelled constraints do not establish empirical protection | investigate partition/estimation, finite-sample generalization, baseline mismatch; **do not** call the mechanism certified |
| Label-free passes all audits and comparisons | a candidate exists | freeze it; only then plan independent confirmation |

## 8. Cost and compute

Incremental ceiling **$50 including storage**, against the previously established workflow
(`reference_pcrl_aws_launch`: AMI, SG, key, IAM profile `pcrl-bios-s3-writer` required). Heavy work
runs on AWS, not locally. The archive is reused and only necessary inputs restored. Task-created
compute is shut down after verified closeout, with the shutdown recorded. The S3 archive bucket's
7-day lifecycle applies to `pcrl-bios-overnight-*`; outputs are pulled locally or written to the
no-expiry UX archive before that window closes.

## 9. Reporting

A complete research-branch report is published **whether or not the result is positive**, stating
separately what the supervised diagnostic establishes, what the label-free candidate establishes,
and what remains unknown — specifically the continuous-side-information problem, which is a
substantive open research question and is not claimed to be solved here. Committed evidence is
handed to Terminal 2.
