# PENDING_EXPERIMENT_INTEGRATION — Terminal 1's fourth study

**Status at the time this revision was completed (2026-09-18): no protocol, matrix or
result for a fourth study is committed on any branch.** The manuscript is finished on the
three completed studies and does not wait for it.

The stale predecessor claim that Terminal 1 "has no experiment" referred to Study 3 and is
corrected: **Study 3 exists, is complete, and is fully integrated at
`73903b7f28df68284285f0610a4036beb32b208f`.** What follows concerns only the *next* study.

---

## 1. What is nominally assigned

A direct adversarial neural refinement with frozen `H`, multiple refreshed attackers, local
and coalition controls, two widths, four penalty strengths, single-attacker ablations,
erasure controls and optimiser-repeat diagnostics — nominally 126 new fits/transforms, plus
the baseline transport missing from Study 3's 2017 pass.

**That paragraph is a plan, not evidence.** Exact counts may be reduced prospectively by
measured resource limits, as Study 3's own OptNet budget was. **Read the committed matrix;
do not treat the assignment as a fit ledger.** Study 3 is the precedent: its nominal 33 and
realised 33 matched, but six SARL fits were deliberately avoided by a proved alias and that
reduction is recorded with its evidence rather than inferred.

## 2. Empty cells that exist in the manuscript today

`papers/pcrl_manuscript_v3/tables/pending_study4.tex`, rendered in Appendix~G, has six rows
with `pending` in every numeric cell and the precondition printed beside each. **No
forecast is entered anywhere.** Filling them requires all of §3 and §4 below.

| Row | Quantity |
|---|---|
| 1 | refined arm − `riv16-C1`, four sensitive endpoints |
| 2 | refined arm − `J`, four sensitive endpoints |
| 3 | coalition vs local controls, coordination cells |
| 4 | residence loss vs `J` and vs `riv16-C1` |
| 5 | penalty-strength and width ablations |
| 6 | erasure and single-attacker controls |

## 3. Protocol checks to run the moment a protocol is committed

Each is a **disqualifying** check: a failure means the corresponding cell is not filled,
and the reason is printed.

| # | Check | How to settle it |
|---|---|---|
| P1 | **No residence or commute label enters representation selection.** | Read the selection rule in the protocol and grep the fit module for the label symbols. Study 3's equivalent assertion is that selection "never consults attackers, residence, commute, development outcomes or the transport table"; the same standard applies. |
| P2 | **Baseline subtraction is not misrepresented as a new gradient or a certified CMI estimator.** | An `H`-relative difference is a reporting convention. If the objective subtracts a baseline, that is a variance-reduction term, not an estimator of `I(S;Z\|H)` and not a new optimisation principle. |
| P3 | **An attacker sees the exact recipient view, and `H_B` does not leak into A's release.** | Check released wire widths per arm (Study 3: A = 20 = `H_A`(4) + channel(16), AB = 22) and assert `H_B` is absent from every A-view release array. |
| P4 | **`L2`/`C1` nominal weight matching is not called exact gradient or capacity matching.** | Shared policy coefficients do not imply equal effective regularisation across constructions. Matching a reference penalty scale at one projection is not equal privacy strength away from it. |
| P5 | **Training attackers and fresh auditors are distinct.** | Assert disjoint pools and disjoint seeds; assert no auditor candidate was fitted on rows the training adversary also saw. |
| P6 | **Repeated optimiser seeds are nested inside the three fixed anchors.** | Optimiser repeats are within-anchor variability. They are **not** additional independent population seeds and must not enlarge a bootstrap denominator or a "seeds agreeing" count. |
| P7 | **Candidate-selection multiplicity covers the whole search used for any superiority claim.** | If the best of `k` configurations is reported, the adjustment family must include all `k`. Study 2's `nlr8-C1` was selected after inspecting 12 conditions and the manuscript says so; the same disclosure is required here. |
| P8 | **A hard maximum's tie behaviour, inner optimisation and attack refresh are described as approximations.** | A minimax formulation solved by alternating finite optimisation is not a solved minimax guarantee. Tie-breaking in a hard max, the inner solver's budget, and the refresh cadence all need stating. |
| P9 | **Runtime, excluded units and reused objects are accounted for honestly.** | A cache read is not a new fit; a resumed unit is counted once; an excluded unit is reported with its reason, not dropped. |
| P10 | **ACS 2016 is untouched.** | No label, output, transform or performance figure for 2016 may be read. A development result cannot authorise spending the one unused year. |

## 4. Artifact checks before any number is printed

1. **Resolve the full SHA** of the study's branch tip and record it.
2. **Recompute every artifact hash** in the study's `HANDOFF.json` from that commit
   (`experiments/pcrl_manuscript_v3/recheck_study3.py` generalises; point it at the new
   directory). Any mismatch blocks integration of the affected artifact.
3. **Count executed units from the committed matrix**, not from the plan, and print
   `realised / nominal` with the reason for any shortfall.
4. **Locate the reporting scope, budget and split by reproduction**, not by assumption —
   this was necessary for Study 3 and no incoming document stated it (`VALIDATION.md` V3).
5. **Check absolute against additional recovery** for the new arms. If they receive no
   catch-up trajectories while the historical comparators do, the asymmetry must be printed
   (`app:scope`).
6. **Read the run ledger and the validation record against each other.** Study 3's
   `VALIDATION.md` contradicted its own `RUN_STATUS.md` on whether numerical faults
   occurred (`CORRECTIONS.md` B10); do not assume internal consistency.

## 5. How the result will be classified, whichever way it comes out

* **A positive development result may motivate a confirmation. It cannot inherit the 2017
  seal.** The locked transport is the only confirmatory evidence in this paper and stays
  that way. A new 2017 pass is development on a spent partition.
* **A negative result is integrated in full**, at the same length as a positive one. Three
  negative method results are already reported; a fourth changes the count, not the
  standard.
* **An incomplete result is reported as incomplete**, with its executed and excluded unit
  counts, rather than as a smaller successful study.
* The conclusion will be compared with the earlier studies explicitly: Study 2 moved
  recovery and failed its coordination rule; Study 3 repaired Study 2's objective exactly
  and made disclosure worse. A fourth arm that beats `riv16-C1` has not thereby beaten `J`,
  and beating `J` on recovery is not beating it overall while `J` supplies less residence
  capability (`CORRECTIONS.md` B3).

## 6. Integration commands

```bash
# 1. resolve and pin
git rev-parse <terminal-1 branch>

# 2. hashes and counts
PYTHONPATH=. python -m experiments.pcrl_manuscript_v3.recheck_study3 \
    --repo . --out results/pcrl_manuscript_review_v3      # adapt STUDY3/D3 constants

# 3. regenerate assets with the new pinned commit
PYTHONPATH=. python -m experiments.pcrl_manuscript_v3.make_assets_v3 \
    --repo . --out papers/pcrl_manuscript_v3

# 4. ledger and manifest
PYTHONPATH=. python -m experiments.pcrl_manuscript_v3.build_claim_ledger_v3 \
    --repo . --out results/pcrl_manuscript_review_v3

# 5. rebuild and re-inspect every page
cd papers/pcrl_manuscript_v3 && latexmk -pdf main.tex
```

The clean follow-up integration is a **successor branch**
(`research/pcrl-manuscript-integrated-v4`), not an edit in place: v3 must remain the
truthful record of what was known on 2026-09-18.
