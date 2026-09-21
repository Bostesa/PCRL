# STATISTICAL_PLAN

Locked 2026-09-21 before any new validation outcome.

## 1. Development status

2018 is a repeatedly used development pool. This study is development evidence. **Neither a fresh
split of already-used people nor a new bootstrap restores independence from prior research**, and
internal selection being frozen does not change that. No confirmation is claimed anywhere.

## 2. Uncertainty

**Unit of resampling is the household**, always. Paired household bootstrap on the actual
candidate/comparator pair, 4000 resamples, computed for **every** nominated candidate against
**every** reported comparator. Intervals are never withheld because historical precision was poor
(`pcrl_stochastic_channel_v1/AMENDMENT_1.md` §4).

**More release draws, more optimizer seeds and more fitted channels are not more independent
households.** Monte Carlo replication counts are reported as such, never as sample size. `K` expanded
token rows for one person are one unit.

**Anchor aggregation.** The three anchors share people. Anchors are therefore **not** independent
pseudo-replicates: resampling is over the **shared household set**, and the registered aggregate is
the mean across anchors of the per-household paired difference, resampled jointly. Independent
per-anchor bootstrapping is not used for any aggregate claim.

The same selected predictions are used for both weightings. Family size and correction rules are
frozen across positive and negative decisions.

## 3. Correction family

Simultaneous one-sided bounds over the declared comparison family:
4 primary sensitive endpoints × 2 weightings × 2 routes × (number of nominees, at most 4) × external
baseline comparisons. The correction is applied over the **whole** declared family regardless of
outcome; choosing among baseline comparisons is itself corrected, and the same baseline identity is
retained across weightings. Bonferroni over the frozen family count, recorded in
`RUN_MATRIX.json` before outcomes.

Prespecified finalist comparisons are reported separately from exploratory whole-grid panels.

## 4. Screen rules vs confidence claims

The Section-6 representation screen and the Section-7 action-library check are **point-estimate
scheduling rules**. They are explicitly not confidence non-inferiority claims and are never reported
as such.

## 5. Precision condition

`estimate + critical_value * SE <= margin`, per actual pair. A negative estimate is **not** universally
required — a small positive estimate with a small enough `SE` also passes. Historical half-widths do
**not** constrain a new pair's `SE`.

## 6. Selection, frozen before outcomes

At most one SUP and one LF configuration per success route, hence **at most four nominees**. Same
configuration across all three anchors. Validation only.

* **Protection route:** among configurations meeting the validation utility and source screens,
  minimize the **worst** primary sensitive difference versus `J`.
* **Utility route:** among configurations meeting the validation disclosure and source screens,
  maximize the **smaller** of the two weighting-specific residence improvements.
* Ties: lower alphabet size, then configuration ID.

Both routes are corrected for, and all nominees are reported. No per-seed or per-weighting choice of a
different "best" configuration.

## 7. Success conjunctions, evaluated only after selection is frozen

**Protection route** — all of:
1. `H_A`/`H_B` parity exact; source-probe allowances pass as prespecified.
2. Residence non-inferior to `J` at the `.001`-nat margin by one-sided upper bound.
3. Sensitive recovery non-inferior to `J` within `.001` on **every** primary endpoint and **both**
   weightings, and **strictly lower** on at least one primary endpoint under the declared simultaneous
   correction.
4. The corresponding comparison against at least one **executed** external baseline also passes, with
   the same baseline identity across weightings, corrected for the choice among baselines, and results
   displayed against every reference.
5. A useful additional capability over `H` is actually present.
6. The `.01`-nat residence reference and the historical stricter half-headroom criterion are retained
   and reported **separately**; neither silently replaces the other.

**Utility route** — 1, 3 (non-inferiority only), 4, 5 hold, and instead of a sensitive improvement:
residence-loss reduction versus `J` of at least `.003` nats at the point estimate under **both**
weightings, with simultaneous upper bounds below zero. Against at least one fixed external comparator,
residence non-inferiority plus a strict improvement in residence or a primary sensitive endpoint,
under the same correction family.

Neither route relaxes the `.001` disclosure margin. Which route passed is always reported.

A candidate may improve usefulness at bounded disclosure without strictly reducing a sensitive
endpoint; that outcome is registered as legitimate and will not be discarded.

## 8. If intervals do not establish a conjunction

Report **point-estimate promise** separately from **competitiveness**. *Unresolved is not equivalent to
either.* An auxiliary outcome may be useful but cannot silently become the primary criterion.

Report all other forbidden roles and retain established policy allowances — passing four sensitive
endpoints does not establish compliance for an omitted role. Use a common class vocabulary, identify
unsupported categories explicitly, and do not claim complete race protection where support is absent.

SUP passing is **task-specific**. LF objective passing after residence-guided screening is still
development evidence, not untouched-task confirmation.

## 9. Things that are not evidence

* Selected-validation bootstrap intervals are not independent evidence for the selected maximum.
* Zero variance from selecting an identical ancestor is an **identity result**
  (`PRECURSOR_CORRECTIONS.md` C3), not proof that alternative predictors or information gains vanish.
* A cross-fitted smoothed-table loss difference is not a ceiling (C5) and a negative value is not
  evidence for a negative information quantity.
