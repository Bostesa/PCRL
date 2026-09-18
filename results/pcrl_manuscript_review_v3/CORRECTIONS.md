# CORRECTIONS — claims narrowed, withdrawn or repaired in v3

Branch `research/pcrl-manuscript-integrated-v3`, successor to manuscript commit
`3c6ada82e719489656da820b72ce8425d2079be0`.

Every correction below is applied **forward**, in this review directory and in
`papers/pcrl_manuscript_v3/main.tex`. **No historical protocol, result file or study
report is rewritten.** The completed studies' directories are preserved byte-for-byte:

| Study | Full SHA | Directory |
|---|---|---|
| 1 — locked 2017 transport | `349efa454afd907389760fd1f59fd8806a215efd` | `results/redesign_20260917_acs_spectral_transport_v1/` |
| 2 — nonlinear penalty / rank | `c37807e4f568ef38e5528fc09c1506083278bf4d` | `results/pcrl_nonlinear_rank_v1/` |
| 3 — invariant repair + externals | `73903b7f28df68284285f0610a4036beb32b208f` | `results/pcrl_invariant_baselines_v1/` |
| evidence-paper branch | `0d8f4b67b6d4961dfa133289d0167c874d2f4794` | `results/pcrl_evidence_review_v1/` |

Items **A1–A17** were already recorded in v2 and are carried unchanged (see
`papers/pcrl_manuscript_v2/main.tex` §"Claims this revision withdraws or narrows").
Items **B1–B14** are new in v3.

---

## The vocabulary audit

Every occurrence of *dominates, matches, equivalent, no utility cost, preserves
utility, causes, implicit regulariser, confirmation, optimal, certificate,* and *novel*
in the incoming study-3 documents and in the v2 manuscript was resolved against the
machine-readable comparison files. The outcome is tabulated in `CLAIM_LEDGER.csv`;
the substantive resolutions are B1–B14.

---

## B1. "LEACE dominates the repaired mechanism … with no statistically detectable residence cost"

**Where:** `results/pcrl_invariant_baselines_v1/RESEARCH_DECISION.md` §3 and
`HANDOFF.json:headline_verdict` / `changed_claims[2]`.

**Status: NARROWED, and the supporting comparator was wrong.**

The recovery half is correct and I reproduce it. `leace_A0 − spectral_riv16_C1` is
significantly lower on all four sensitive endpoints under **both** weightings
(`PAIRED_INTERVALS.csv`, family `external_vs_repaired`):

| weighting | A/SEX | AB/SEX | A/RAC1P | AB/RAC1P |
|---|---|---|---|---|
| unweighted | −0.00805 | −0.00856 | −0.01306 | −0.01584 |
| person-weighted | −0.00967 | −0.00928 | −0.01541 | −0.01659 |

The utility half does not support the word used. The residence-loss difference is

* unweighted **+0.00357**, adjusted \[−0.00284, **+0.00999**\]
* person-weighted **+0.00396**, adjusted \[−0.00312, **+0.01104**\]

Three separate reasons this is not "no utility cost", and not dominance:

1. **Non-significance is not absence.** The interval is compatible with LEACE costing
   up to ~0.010 nats of residence capability — *larger in magnitude than the recovery
   advantage it wins on three of the four sensitive endpoints.*
2. **SPLINCE's cost, which the same document calls significant, lies inside LEACE's
   interval.** SPLINCE's residence difference is +0.00888 \[+0.00317, +0.01460\]. A
   reader told that one arm has "no detectable cost" and the other a "significant cost"
   will infer the two differ. The evidence does not separate them.
3. **Against the correct reference the cost *is* significant.** `spectral_riv16_C1` is
   a weak utility comparator. LEACE's own source channel is `A0`, the untransformed
   16-coordinate auxiliary channel it was applied to, and against that reference
   (family `erasure_vs_its_own_source`):

   | contrast | weighting | residence loss difference | verdict |
   |---|---|---|---|
   | `leace_A0 − A0` | unweighted | **+0.00978** \[+0.00363, +0.01593\] | significantly worse |
   | `leace_A0 − A0` | person-weighted | **+0.00779** \[+0.00112, +0.01446\] | significantly worse |
   | `splince_A0 − A0` | unweighted | **+0.01509** \[+0.00888, +0.02131\] | significantly worse |

   Mean residence gain over `H` falls from 0.0318 (`A0`) to 0.0220 (`leace_A0`)
   (`CRITERIA_SUMMARY.csv`, unweighted). **Linear erasure of this channel costs
   measurable authorised capability**; it merely costs no *more* than an already
   utility-degraded spectral arm.

**v3 wording.** "Against the repaired spectral arm, LEACE recovers significantly less
on all four sensitive endpoints; the accompanying residence-loss difference is
unresolved, with an interval that admits a cost as large as 0.010 nats. Measured
against the channel LEACE actually transformed, it costs 0.0098 nats of residence
capability, and that difference is significant." The word *dominates* is not used.

---

## B2. "LEACE, SPLINCE and OptNet-C1 are statistically indistinguishable from `J`"

**Where:** `RESEARCH_DECISION.md` §3, `HANDOFF.json:changed_claims[2]`, and the
candidate-observation paragraph ("the only arm that **matches** `J` on every sensitive
endpoint").

**Status: NARROWED. These are unresolved differences, not demonstrated equivalence.**

No equivalence or non-inferiority test was run for any of these cells; the finding is
that adjusted simultaneous intervals contain zero. That is absence of evidence. It is
also not symmetric: the point estimates lean one way.

| contrast (unweighted) | A/SEX | adjusted interval |
|---|---|---|
| `leace_A0 − J` | **+0.00548** | \[−0.00063, +0.01158\] |
| `splince_A0 − J` | +0.00472 | \[−0.00151, +0.01095\] |
| `optnet16_C1 − J` | +0.00609 | \[−0.00150, +0.01367\] |

All three point estimates favour `J` on local sex, and two of the three intervals
exclude zero only by less than 0.0007. Person-weighted rows agree in direction. The
correct statement is "not separated at three seeds", and the paper says so once,
plainly, rather than distributing hedges.

`optnet16_L1 − J` on A/SEX is **significantly worse** (+0.01038 \[+0.00249, +0.01827\]
unweighted; +0.00935 \[+0.00076, +0.01795\] person-weighted), so the family of OptNet
adaptations is not uniformly indistinguishable from `J` either.

---

## B3. `J` is not Pareto-dominant

**Status: CORRECTED where it appeared; the v2 manuscript was already right and v3
keeps that reading.**

`J` recovers less, but it also supplies **less** authorised capability than several
spectral arms. On 2018 development data (`PAIRED_INTERVALS.csv`, unweighted):

* `spectral_riv16_C1 − J` residence loss = **−0.00406** — the spectral arm's residence
  loss is *lower*, i.e. it supplies more capability (unresolved: \[−0.01164, +0.00352\]).
* `spectral_riv16_L1 − J` residence loss = **−0.00797** \[−0.01569, −0.00025\] —
  **significantly** more capability than `J`, unweighted (person-weighted −0.00709,
  unresolved).
* Mean residence gain over `H`: `J` 0.0215 against 0.0245–0.0295 for the repaired arms
  (`CRITERIA_SUMMARY.csv`).

On the exploratory 2017 look the same shape holds: `J` gains 0.0189 residence against
0.0260–0.0289 for the repaired arms. So the comparison is a **trade**, not a
domination, and no full utility comparison in the evidence supports calling `J`
Pareto-dominant. v3 states the trade explicitly whenever `J` is invoked.

---

## B4. "Coordinate dependence was acting as an implicit regulariser"

**Where:** `RESEARCH_DECISION.md` §2, "The mechanism reading, stated carefully".

**Status: NARROWED to a hypothesis. The design does not isolate a cause.**

`METHOD.md` §0 is explicit that the repair changed **several** things at once:
Frobenius weighting of the off-diagonal monomials, a single scalar block scale in
place of per-feature standardisation, removal of centering, an exact three-bandwidth
radial kernel in place of a finite Fourier feature bank, and per-role frozen 512-row
subsets. Its own words: "Any measured ACS effect is attributable to the **package**,
never to `sqrt(2)` weighting alone."

The kernel substitution alone changes the estimand: the Fourier block averaged over
all ~10,500 valid rows, the exact kernel block averages over at most 512 frozen rows
and carries a measured `1/m` V-statistic floor (`METHOD.md` §3.7). No ablation
separates these ingredients. Therefore:

* **Supported:** the repaired package attains a lower value of the repaired objective
  in 18/18 matched cells, and its releases disclose significantly more on 4 of 24
  sensitive cells with 0 significantly less.
* **Not supported:** that *coordinate dependence* was the operative regulariser, or
  that rotation invariance is harmful in this or any objective.

v3 presents this as "surrogate–attacker mismatch remains the live explanation; the
study does not separate the five simultaneous changes", and drops the causal noun.

---

## B5. The 63 % rotation share

**Status: already corrected in v2 (item A13); restated and kept.**

0.628 is the **attained** mean share of training-objective gain that a *separately
optimised* information-preserving rotation also reaches, over 18 cells, range
0.367–0.907. Every rotation search stopped on budget exhaustion or a failed line
search (`MECHANISM_GATE.json` records `rotation_stop_reason` per cell). It is
therefore

* **not** a supremum over rotations,
* **not** a decomposition of the actual optimiser trajectory into effective and wasted
  parts,
* **not** a prediction about what a repaired objective would achieve — and study 3 is
  the direct experiment that refutes the prediction that was made from it.

---

## B6. "All four repaired coordination cells failed"

**Status: CONFIRMED and reported without inflation.**

Registered forecast Q4 is confirmed: 4 of 4 registered coordination cells NOT
SUPPORTED. v3 reports exactly that, and adds the sentence that bounds it: *four cells
of one construction on one interface is evidence about four cells. It is not evidence
that nonlinear conditional-dependence penalties cannot satisfy a coordination rule,
and the paper makes no such claim.* The v2 withdrawal A8 — that a nonlinear failure
generalises to "the closed-form spectral family" — stays withdrawn, and is not
reinstated in a new form for study 3.

Related, and not in the incoming reports: the repaired arm is **significantly worse
than its own linear-moment control** on A/race at rank 8 in three contrasts
(`spectral_riv8_L1 − spectral_lin8_L1` unweighted +0.01158 \[+0.00479, +0.01836\],
person-weighted +0.01053 \[+0.00237, +0.01870\]; `spectral_riv8_C1 −
spectral_lin8_C1` unweighted +0.00772 \[+0.00168, +0.01377\], person-weighted +0.00912
\[+0.00203, +0.01621\]). So "the repaired arms
beat their own linear controls" — true of the 2017 exploratory seed means — is **not**
true of the 2018 intervals, and v3 does not let the 2017 sentence stand unqualified.

---

## B7. Support counts refer to fitting pools

**Status: CORRECTED wording.**

`RESEARCH_DECISION.md` §4 says seed 2's `RAC1P` class 3 has "**zero population
support**". The recorded quantity is `coverage.RAC1P.support_complete_cases[3]` in
`ERASURE_BASELINES.json`, which is the count of complete cases in **that seed's own
representation-fitting pool**: 1, 1, 0 for seeds 0, 1, 2. Alaska Native alone is not
absent from California.

The substantive finding survives and is worth keeping: *protected-class support in the
fitting pool determines the realised width of a linearly erased channel* (6, 6, 7 of
16). v3 states it with the correct denominator, and repeats the Study-1 rule: **a class
without support is unmeasured, not protected.**

---

## B8. The historical 2017 counts

**Status: PRESERVED exactly, and kept separate from the new evidence.**

For the registered family F1 on the sealed 2017 partition: **14 of 16 sensitive cells
strictly better, 0 worse, 2 unresolved**, and the registered `advantage` rule fired in
4 of 4 contrast × weighting cells. Regenerated in this revision from the interval files
(`DERIVED_FACTS.json:family_counts.F1_primary`), unchanged.

The **point-estimate** residence rule (seed-mean difference ≤ 0.001) passed, with values
0.00029, 0.00010, 0.00059, 0.00017. **Interval-based non-inferiority at the same margin
holds in 0 of 4 comparisons** under the most permissive procedure computed. These are
different criteria and v3 never lets one stand in for the other.

Study 3's 2017 pass is **development on a spent partition**: no intervals, no
significance, no confirmatory status, and the erasure and OptNet arms are absent from
it entirely. It does not re-seal the year and does not restate the historical result.

---

## B9. The two carried numerical corrections

**Status: KEPT, verified independently at the pinned commit.**

* `spectral_nlr8_C1 − J` on `recovery/A/SEX`, person-weighted: **+0.01210**
  \[+0.00441, +0.01978\], significantly worse. The +0.0098 printed in study 2's prose
  is supported by no cell. Unweighted is +0.01065 \[+0.00331, +0.01799\].
* `spectral_nlr8_L1 − spectral_lin8_L1` on `recovery/A/RAC1P`, unweighted: **+0.00862**
  \[+0.00168, +0.01556\], significantly worse. The nonlinear penalty is **not**
  harmless, and "no endpoint significantly worse" stays withdrawn. (Person-weighted is
  +0.00681 \[−0.00141, +0.01503\], unresolved — the correction holds under one weighting
  and v3 says which.)

Both re-read from `c37807e4:results/pcrl_nonlinear_rank_v1/PAIRED_INTERVALS.csv`.

---

## B10. Numerical faults and the machine

**Status: REPORTED as association, not cause — and one incoming document is stale.**

Study 3's exploratory-2017 stage aborted four times with a probability row summing to
exactly zero inside an otherwise well-formed array
(`RUN_STATUS.md` "Failures and repairs" item 4). Every occurrence was detected by the
scorer's full-schema validation and aborted its unit **before** any metrics file was
written; four `QUARANTINE.json` records exist under
`results/pcrl_invariant_baselines_v1/exploratory_2017/`, and every quarantined unit was
recomputed by the unpatched pipeline. Memory pressure (swap at capacity throughout) is
the **observed condition**, not a demonstrated cause, and nothing here claims the
machine is now proved reliable.

**Defect found in the incoming package, reported to Terminal 1.**
`VALIDATION.md` §4 records "Inconsistent outputs: none observed; none quarantined" and
§5 records "**No such fault occurred in this run.**" Both are false as written.
`VALIDATION.md` was committed at `4e9dc127` (2026-09-17), before the 2017 stage ran;
`RUN_STATUS.md` was finalised at `34ff87d8` (2026-09-18). The 2017 stage's own
`EXPLORATORY_2017.md` and the handoff's
`known_limitations_terminal_2_must_carry[5]` both record the faults correctly, so the
package as a whole is internally recoverable — but a reader who stops at
`VALIDATION.md` is misinformed. **v3 carries the RUN_STATUS account, not the
VALIDATION account**, and this discrepancy is listed in `REVIEWER_RISKS`
(`REVIEW_INDEX.md`). No number changes.

---

## B11. The reporting scope, and what a paired contrast is invariant to

**New in v3; not stated in any incoming document.**

Study 3's published paired intervals are computed in scope `expanded_catchup`, budget
360, split `test`. I located this independently by reproducing a published estimate
from the per-seed file rather than taking it on trust (`VALIDATION.md`, check V3;
`leace_A0 − spectral_riv16_C1` on A/SEX unweighted reproduces as −0.00805 to five
decimals, and no other scope does).

That scope gives `H` and the frozen historical interfaces their saved-observer
catch-up attacks; the new arms have `own_catchup_trajectories: 0`. The consequences
are worth printing because two of them cut in opposite directions:

| quantity | fresh independent | reporting scope |
|---|---|---|
| `H` absolute A/sex recovery | 0.00142 | **0.00708** |
| `J` absolute | 0.00271 | 0.00236 |
| `leace_A0` absolute | 0.00531 | 0.00784 |
| `spectral_riv16_C1` absolute | 0.01589 | 0.01589 |

* **Paired contrasts between two arms are identical on both scales**, because the
  shared `H` baseline cancels. Every headline number in this paper is such a contrast,
  so none of them is scope-dependent. Verified: 0.00784 − 0.01589 = −0.00805.
* **`H`-relative levels and any ratio are scope-dependent**, by a factor of five on the
  subtracted baseline. Study 3's reports quote `H`-relative levels for the 2017 pass
  without flagging this. v3 prints absolute and additional recovery side by side
  (Table `scope_decomposition`), exactly as v2 did for study 1.
* **`J`'s absolute recovery *falls* when it is given more attack candidates.** A
  validation-selected attack maximum is a floor on leakage and never an upper bound —
  and here is a measured instance of the selected attack generalising worse.

---

## B12. "First method-level baselines" and what the adaptations are

**Status: NARROWED. These are adaptations, named as such.**

`BASELINE_ADAPTATIONS.md` is careful and v3 carries its distinctions into the paper
rather than compressing them into method names:

* **LEACE and SPLINCE transform the auxiliary channel alone** and are appended to
  unchanged `H_A`. They are *not* applied to the concatenated `[H_A, Z]` wire, because
  LEACE's `P*` is a single oblique projection over all coordinates and nothing
  constrains it to act as the identity on `H_A`. Their empirical linear guarantee
  therefore concerns **the fitted erasure of that channel**, on the moments it was
  fitted on. It is not a guarantee about arbitrary nonlinear prediction from the
  concatenated service-plus-channel view an attacker actually holds. Study 3's own
  fixture shows a quadratic probe recovering from an erased channel at which a linear
  probe reads `R² < 1e-8`.
* **SPLINCE's preservation target is the authorised training tasks**
  (`income_binary`, `civilian_at_work`), **not** the held-out residence task. That is
  an adaptation, and it is an asymmetry in SPLINCE's favour on utility: no other arm
  sees a task label during representation fitting.
* **SPLINCE shares LEACE's kernel.** Under SPLINCE Thm 2 the two give identical
  predictions after refitting an *unregularised* unique-minimiser model, so any
  difference measured here is attributable to attacker regularisation and
  nonlinearity, never to a stronger erasure guarantee.
* **The marginal spectral arms are SARL-style rebuilds** — verified bitwise — with one
  measured mismatch that is **not** absorbable into `λ̄`: per-attribute trace
  reweighting, moving the selected subspace by projector Frobenius distance 0.10–0.20.
  They are credited and reused, never refitted. "SARL adaptation" is therefore accurate
  and "exact replica of the published baseline" is not.
* **The OptNet multi-λ multi-attribute form is an extrapolation.** The paper asserts
  multi-attribute generalisation in a single sentence of its abstract with no equation,
  no normalisation rule and no code (verified against the arXiv abstract page); the
  specific per-role form used here is this project's.
* **Shared policy coefficients do not imply equal effective regularisation across
  methods.** Disclosed, not resolved.

The honest headline is: *a closed-form linear-erasure method from 2023, adapted to a
channel this project already had, recovers significantly less than the mechanism this
line of work developed across three studies — at a residence cost the design does not
bound, and under attackers its own authors disclaim.* That is narrower than "beats",
and it is what the evidence carries.

---

## B13. "Optimal", "certificate", "confirmation", "novel"

* **Optimal.** Ky Fan global optimality applies only to the *original* fixed-matrix
  family, whose released map is the top-`r` eigenvectors of a `W`-independent matrix.
  The repaired objective is nonconvex, has no fixed matrix, and is at best locally
  optimal (`METHOD.md` §5.2 says so). A suboptimal solution remains a live explanation
  for study 3's result, and v3 says so instead of asserting the optimisation was good.
* **Certificate.** Nothing in any study is one. Vanishing fitted finite moments imply
  neither `Z ⊥ S | H` nor any bound on `I(S;Z|H)`; no Type-I control is inherited from
  KCI or RCoT; and a misspecified nuisance manufactures a penalty more than ten times
  the oracle value under **exact** conditional independence, so a non-zero penalty is
  not evidence of disclosure.
* **Confirmation.** Exactly one result in this paper is confirmatory: the locked 2017
  transport. Study 3's 2017 pass is development on a spent partition. A positive
  development result may *motivate* a confirmation; it can never inherit the sealed
  year's status. **ACS 2016 remains unscored** and is not authorised by anything here.
* **Novel.** No priority claim is made. Known erasure, eigenvalue-sign rank selection,
  minimax adversarial training, conditional-moment tests and transferable adversarially
  protected representations are all prior art and are cited as such; Madras et al.
  (2018) is added explicitly so that "transferable useful representations with
  adversarial protection" is not presented as new. What the paper claims not to have
  found, after a bounded search, is a prior *evaluation* of recipient-specific and
  combined-access policies over an unchanged already-published output — phrased as
  absence of evidence after a bounded search, never as a first.

---

## B14. Stale claim in the predecessor's handoff

v2's `PENDING_ADDENDUM.md` recorded that Terminal 1's experiment "does not exist".
That was true when written and is **stale**. Study 3 is complete and committed at
`73903b7f28df68284285f0610a4036beb32b208f`; all 27 artifacts named in its
`HANDOFF.json` are present and their recorded sha256 values match. v3 integrates it in
full. The separate, genuinely pending item is Terminal 1's **fourth** study, which had
committed no protocol at the time of this revision; it is tracked in
`PENDING_EXPERIMENT_INTEGRATION.md` with empty, explicitly-marked cells.
