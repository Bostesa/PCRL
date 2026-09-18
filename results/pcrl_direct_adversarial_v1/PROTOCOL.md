# PROTOCOL — direct adversarial refinement of the neural auxiliary channel

**Registered development, not a fresh confirmatory study.** Every 2018 and 2017
partition used here has been used repeatedly by this project. Registration makes the
analysis choices prospective; it does not restore independence.

This file, `METHOD.md` and `MATRIX.json` are hashed into `PROTOCOL_FREEZE.json` before
any new method outcome is opened.

---

## 1. Corrections carried into the starting interpretation

Recorded additively in `CORRECTIONS.md`; historical evidence is **not** edited.

1. **LEACE does not dominate on utility.** Its residence-loss difference against the
   repaired rank-16 `C1` arm is `+0.0036` with interval `[-0.0028, +0.0100]`
   unweighted. That interval contains zero **and** contains values well above the
   `.001` reference. It establishes **neither** dominance **nor** utility
   noninferiority.
2. **Nonsignificance versus `J` is not equivalence to `J`.** The erasure and OptNet
   arms being statistically indistinguishable from `J` on the five family endpoints is
   an absence of evidence, not an equivalence result.
3. **Coordinate dependence was not isolated as the cause.** The invariant repair
   changed quadratic weighting, feature scaling, kernel approximation and the kernel
   sample **together**. Implicit regularisation remains a *possible* explanation for
   the worse recovery, not an identified cause.
4. **The original coordination success used a one-sided point-estimate utility rule.**
   It was not an equivalence band and not an interval-based noninferiority result.
5. **The external baselines were never evaluated on 2017.** A pipeline boundary and a
   missing model serialisation are engineering tasks, not completed comparisons.
   §2 completes them.

## 2. Baseline transport completion (engineering, not a new method result)

Complete the 2017 evaluation of the existing frozen `leace_A0`, `splince_A0` and
`optnet16_{L1,L2,C1}` adaptations across all three fitted seeds: **15 interfaces**.

* **Reuse their 2018-fitted transformations on 2017.** No new 2017 eraser is fitted and
  then called transported.
* For `leace_A0` / `splince_A0`: recover the **frozen `A0` inference pipeline**, run it
  on the 2017 features, and apply the **saved affine map** (`projection`, `mean`) to
  that auxiliary channel alone. `H` is appended **unchanged**.
* For OptNet: recover saved encoders. If unavailable, **deterministic reconstruction is
  authorised**, and the reconstruction must first be verified against the recorded 2018
  artifacts — by weights, or by a comprehensive prediction reference. **A mismatch is a
  new fit and may not silently inherit the old identity.**
* Reuse the historical 2017 fitting/validation/evaluation partitions and the common
  attack recipe. Fit fresh 2017 attackers and utility probes, select **only** on the
  designated 2017 validation pool, then score the evaluation pool.
* Include `H`, `A0`, `J` and the relevant spectral references through compatible saved
  predictions or explicit re-evaluation.
* **Mark every new row exploratory.** Report utility costs as well as all
  forbidden-role recovery.

## 3. Prespecified research questions

**Q1 — refresh and ensemble.** Does refreshed multi-attacker training outperform the
same architecture trained against **one** attacker, at matched initialisation, utility
objective, role weights, refresh opportunities and alternation schedule?

**Q2 — coalition conditioning.** Does coalition conditioning (`C1`) help **beyond**
stronger local training (`L1`, `L2`) when architecture, initialisation, utility
objective, optimiser exposure and attacker slot count are controlled?

**Q3 — the frontier.** Does any candidate improve the **useful** frontier against `J`,
against erasure and against OptNet — **including local race** — rather than merely
moving to less utility and less recovery?

**Directional expectations, registered, with no invented confidence.** These are
weak priors, and being wrong about them is a reportable outcome, not a failure:

| # | Expectation | Basis |
|---|---|---|
| D1 | `C1` reduces `AB/*` recovery relative to `L1` at matched width and beta in at least 2 of 3 seeds | the penalty contains those roles and `L1`'s does not |
| D2 | At least one `C1` arm shows a **local race cost** relative to `L2` on `recovery/A/RAC1P` | coalition weight is bought from somewhere; this is the failure mode §8 forbids concealing |
| D3 | The ensemble arm's measured recovery is **not** lower than the single-attacker arm's at matched cells in the majority of cells | a training ensemble fools a wider family, which the fresh audit may or may not see |
| D4 | The declared nominee conjunction in §8 is **not** satisfied | four prior studies on this interface have not satisfied it |
| D5 | Width 8 shows lower recovery **and** lower residence gain than width 16 at matched policy/beta | compression moves both, which is why §9 checks whether compression accounts for any apparent improvement |

No prediction is registered about the *size* of any effect.

## 4. Fixed release contract

As `METHOD.md` §2. Restated here as the enforcement list: `H_A` 4 coordinates and
`H_B` 2 coordinates bitwise identical by construction in dtype, ordering and
probability coordinates; `Z` computable from permitted inference inputs; `H_B` may
condition coalition training attackers and is never added to the `A` release; no
sensitive label or row identifier at inference; residence and commute excluded from
every training, selection and calibration decision.

## 5. Multiplicity, before any result

**Inference.** Paired **household-cluster** bootstrap on stored per-person
predictions/losses, 2000 replicates, the transport study's own cohort-household
clusters. Loss arrays are streamed, never held whole.

**The candidate-wide family.** The simultaneous family is **every contrast searched
for a winning claim**, not the five endpoints of the eventual winner. Declared now:

```
family = { all prespecified contrasts of §6 }  x  { 5 family endpoints }  x  { 2 weightings }
```

Adjustment is a **conservative** simultaneous correction: per-comparison two-sided
bootstrap quantiles at level `alpha / m` with `alpha = 0.05` and `m` the realised
family size, reported alongside unadjusted per-seed signs and effect sizes.

**Decision rules, kept separate and never swapped:**

| rule | statement |
|---|---|
| historical `.001` coordination rule | a **point-estimate** rule, reused for continuity, labelled as such. A pass is not equivalence |
| new utility noninferiority | the relevant **one-sided adjusted interval bound** must be at most `.001` |
| equivalence | requires its own two-sided margin procedure; nonsignificance never suffices |
| residence reference gain | `0.01` over `H`, a **reference**, not a test |
| source requirement | the frozen source-probe allowance of the historical registry |
| old half-headroom reference | historical, reported only where the historical table used it |

Three anchor seeds do **not** establish broad training-population robustness.

## 6. Prespecified scientific contrasts

1. `C1` vs `L1` and `C1` vs `L2` at each matched width x beta.
2. Ensemble vs single-attacker at each ablation cell.
3. Each main `C1` vs frozen `J`, vs historical `leace_A0` / `splince_A0`, vs
   `optnet16_*`, and vs the **new** erasure controls.
4. Width 8 vs width 16 at each matched policy x beta.
5. Stability of the proposed mechanism across the extra optimiser seeds.

All configurations and all seeds are reported, never a chosen best row. The four
primary sensitive endpoints — `A/SEX`, `AB/SEX`, `A/RAC1P`, `AB/RAC1P` — are kept
**alongside** every other forbidden role. **A local race cost is never concealed behind
an average coalition SEX gain.**

## 7. Timing pilot and the frozen budget schedule

A representative **training-only** pilot was run before any outcome: seed 0, three
arms (`dax16_none`, `dax16_C1_b100`, `dax8_C1_b100`) at the **full nominal budget** of
600 mapper updates. It read only the internal `mapper_fit` and `monitor` folds. No
audit, no downstream pool, no test pool, no residence, no commute.

| measurement | value |
|---|---|
| seed context (recovery, folds, `p0_j`, warm start, both widths) | **5.2 s** |
| one fit at 600 mapper updates | **8.2 s** |
| projected 126 fits + 3 seed contexts | **~19 min** |
| 2018 audit per interface (historical measurement, `73903b7f`) | **~42 s** |
| 2017 fit + score per interface (historical measurement, `73903b7f`) | **~161 s** |

**The nominal 600-update budget is met in full. No budget reduction is applied and no
block is dropped.** The full **126-fit** matrix, all **126** 2018 audits and the fixed
**69-interface** 2017 panel are scheduled.

The reduction ladder is nevertheless frozen here, so that if wall-clock pressure
appears later it is resolved by a pre-registered rule rather than an improvised one:

1. reduce the mapper update budget **uniformly** (never per-arm), retaining all three
   seeds and all local/coalition controls;
2. then defer the **optimisation-repeat block** (24 fits);
3. then defer the **720-epoch catch-up extension**.

**A losing configuration is never selectively omitted, and a reduced matrix is never
called the full 126-fit study.** Any unfinished unit is listed explicitly in
`RUN_STATUS.md`.

**Resources.** One experimental worker per phase, one BLAS/OpenMP thread, `torch`
single-threaded. The machine is shared (14 CPUs, 24 GB, load average ~8 at start,
a Docker VM and other sessions present). Concurrency above one worker is taken only
after measuring free memory, is capped at two, and is recorded in `RUN_STATUS.md`.
Predictions are streamed and arrays released promptly. Earlier zero-sum probability
rows were observed under memory pressure and **the cause was never established**;
every probability matrix is validated for finiteness, bounds, nonzero row mass and row
sums before any atomic persist, and a suspicious unit is recomputed independently
rather than treated as data.

## 8. The nominee conjunction

A credible development nominee must satisfy **all** of:

1. preserve `H` exactly (bitwise, every pool);
2. meet the declared source-probe requirements;
3. add useful residence capability;
4. improve at least one declared sensitive endpoint against **both** local controls
   **and** a strong external/internal comparator, **without a declared sensitive-role
   regression**;
5. survive the stronger fresh attacks of the stress set.

The report states whether this conjunction is **supported**, **unsupported** or
**unassessable**. **It is not loosened after a near miss.** Frontier improvements that
fail the conjunction are reported accurately **without** a nominee. **No
confirmation-year execution follows automatically**, and 2016 is not opened by any
outcome here.

## 9. Audit program

After mapper and checkpoint choices are locked, run the established full 2018 audit on
every primary, ablation and erasure interface: full eleven-role forbidden registry,
identical splits, independent versus expanded pooled attack scopes, both unweighted and
PWGTP scoring **on identical predictions**. Unchanged `B` and `H`-only fits are reused
by validated identity. **Fresh audit models are entirely separate from the training
ensemble.**

Service-only candidates are included in every full-view audit. Selection on validation
can still yield a **negative** evaluation increment; those observed increments are
**never truncated to zero** in reports.

Default catch-up budget **360** epochs. **Stress set, prespecified**: `J`, historical
`leace_A0`, and the new width-16 `C1` and `L2` arms at `beta = 0.3` and `1.0` — two
fresh audit initialisations and a **720**-epoch continuation, selected on validation
only. The larger final attack suite is carried into **both sides** of every comparison
that uses it; competitors are never strengthened alone.

The 24 optimisation repeats receive the same primary independent attack families and
utility probes. Any shorter audit for the repeats is locked **before** any repeat
outcome and its comparisons use a matched common suite.

## 10. 2017 exploratory panel — membership fixed now

Fixed in advance, **not** selected by residence results:

* **36** main interfaces: `beta in {0.3, 1.0}` x `{width 8, 16}` x `{L1, L2, C1}` x 3 anchors;
* **6** no-protection interfaces;
* **12** new erasure interfaces;
* the **15** historical external-baseline completions of §2;
* compatible `H` / `A0` / `J` references.

Encoders are **not** retrained on 2017. Fresh probes train and select only on the
established 2017 fitting/validation partitions. Missing-category limitations are
preserved and reported.

The panel is **exploratory** even though its membership is fixed now. Any uncertainty
interval computed on it is labelled **conditional descriptive sampling uncertainty**;
intervals do not restore independence after repeated development, and they are not
mathematically forbidden on a used dataset either.
