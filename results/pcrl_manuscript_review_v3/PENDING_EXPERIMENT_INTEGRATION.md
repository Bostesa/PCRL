# PENDING_EXPERIMENT_INTEGRATION — Terminal 1's fourth study

**Status at the time this revision was completed (2026-09-18): the protocol is REGISTERED
and the run is IN PROGRESS. No result has been read.**

| | |
|---|---|
| Branch | `research/pcrl-direct-adversarial-v1` |
| Worktree | `/Users/nathansamson/PCRL-terminal-1-adversarial` |
| Head at check time | `86f142c4a4324c4dc71c8d094eecd48b29af640d` |
| Registered at | `f5976c28` (protocol, method, 126-fit matrix), re-frozen at `b229a5c4` |
| Directory | `results/pcrl_direct_adversarial_v1/` |
| Phase log | phase 0 complete; phases 1–5 (transport completion, 126 fits, 126 audits, 69-unit 2017 panel, analysis) open |

The stale predecessor claim that Terminal 1 "has no experiment" referred to Study 3 and is
corrected: **Study 3 exists, is complete, and is fully integrated at
`73903b7f28df68284285f0610a4036beb32b208f`.**

---

## 1. What the committed matrix actually declares

Read from `MATRIX.json` at `86f142c4`, not from the assignment:

| block | per seed | fits | what it is |
|---|---:|---:|---|
| `main` | 24 | 72 | 2 widths × 3 policies × 4 `beta` × 3 anchor seeds |
| `no_protection` | 2 | 6 | `beta = 0`, same utility objective and budget |
| `single_attacker` | 4 | 12 | width 16 × {L2, C1} × `beta` {0.3, 1.0}, one MLP family |
| `new_erasure` | 4 | 12 | LEACE/SPLINCE refit on each width's no-protection channel |
| `optimizer_repeat` | 8 | 24 | width 16 × {L2, C1} × `beta` {0.3, 1.0} × 3 anchors × 2 extra optimiser seeds |
| **total** | | **126** | `declared_total: 126`; 2017 panel `panel_2017_count: 69` |

The nominal 126 and the declared 126 agree **as registered**. Counts may still be reduced
prospectively by measured resource limits, as Study 3's OptNet budget was; `MATRIX.json`
carries an empty `reduction_reason` field for exactly that. **Read the realised counts from
the run ledger at completion; do not treat this table as a fit ledger.**

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

## 3. Protocol checks — RUN against `86f142c4`, all ten satisfied

Each is a **disqualifying** check: a failure means the corresponding cell is not filled,
and the reason is printed. The outcome column records what was found in the registration.
These check the *protocol*; they do not and cannot pre-validate the results.

**Summary: 10 of 10 satisfied, 1 watch item (P6).** The protocol is unusually careful, and
its §1 already carries this revision's corrections B1–B4 into its own starting
interpretation, so the two records agree on what was and was not established by Study 3.

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

### Outcomes at `86f142c4`

| # | Found | Verdict |
|---|---|---|
| P1 | `METHOD.md` §133: "`same_residence` and `commute_over20` never enter encoder" training; §327–328: checkpoint selection uses "**No residence, no commute, no downstream pool, no test pool**", `argmin` with deterministic tie-breaking; `PROTOCOL.md` §4 repeats the exclusion for training, selection **and** calibration. | **satisfied** |
| P2 | `METHOD.md` §60: the incremental gain `g = CE(p0) − CE(q)` is "**not** a new conditional-information estimator", and subtracting the fixed `CE(p0_j)` "creates **no new encoder gradient at all**" because `p0_j` does not depend on `theta` — it changes only which constraints are active under the hinge and how the quantity reads. Explicitly not a bound on `I(S;Z\|H)`. | **satisfied**, and stated more precisely than the check required |
| P3 | `METHOD.md` §123–128: `A` receives `[H_A, Z]`, `B` receives `H_B`, `AB` sees the union; `H_B` **may** condition coalition training attackers "and is never added to the `A` release"; enforced by `wire/A[:, :4] == H_A` and `wire/AB[:, -2:] == H_B` on float64. | **satisfied**, with a structural assertion rather than a promise |
| P4 | `METHOD.md` §304: "Identical budgets across matched policies and strengths"; §218 locks a slot schedule giving matched attacker exposure. Described as matched **budget and slot count**, not as gradient or capacity equality. | **satisfied** |
| P5 | `METHOD.md` §316: each checkpoint is scored against "**its own freshly initialised attacker slate**"; §215: at least one independent audit architecture is out of training by construction; §323: training-attacker-versus-fresh-auditor analysis is a declared output. Four held-out audit architectures in `MATRIX.json`. | **satisfied** |
| P6 | `MATRIX.json` separates `seeds` (3 anchors) from `extra_optimizer_seed`, and the `optimizer_repeat` block is described as "3 anchor seeds × 2 additional optimiser seeds". `PROTOCOL.md` §5: "Three anchor seeds do **not** establish broad training-population robustness"; §129 lists repeat stability as its own question. | **satisfied — with a watch item.** The nesting is structurally correct and the repeats have their own question. What is not yet written anywhere is an explicit prohibition on counting the 2 extra optimiser seeds toward a bootstrap denominator or a "seeds agreeing" tally. Check this at analysis time; it is a reporting risk, not a design defect. |
| P7 | `PROTOCOL.md` §5: "The simultaneous family is **every contrast searched for a winning claim**, not the five endpoints of the eventual winner", declared as `{all prespecified contrasts} × {5 endpoints} × {2 weightings}`. | **satisfied**, and this is the strongest version of the check |
| P8 | `METHOD.md` §56: "Finite alternating optimisation is **not** a solved minimax problem and confers no protection certificate"; §193: the maximiser and `(slot, family)` order are fixed so "ties resolve deterministically"; §243–246 specify the refresh cadence (25/50/75% of budget, bounded 40-update catch-up) and the refreshed-family order. | **satisfied** |
| P9 | `RUN_STATUS.md` has a standing "Failures, quarantines and omitted units" section with "Corrupted scores are never treated as data"; `MATRIX.json` carries `reduction_reason` and a `reuse_policy`; resource decisions record load average, worker count and thread caps. | **satisfied as a structure.** Verify the realised entries at completion. |
| P10 | `METHOD.md` §379–380: "**2016 remains sealed and unused.** No transformation, fitting, scoring, label inspection or candidate selection involving 2016 occurs in this study"; `PROTOCOL.md` §192: no confirmation-year execution follows automatically and 2016 is not opened by any result here. | **satisfied** |

### One amendment already recorded there, and it is the right kind

`RUN_STATUS.md` declares, timestamped `2026-09-18T13:00Z` and **before any 2018 or 2017
result was read**, that the registered per-comparison bootstrap-quantile correction is
**not estimable** at the realised family size: `m = 86 × 5 × 2 = 860` gives
`alpha/m = 5.8e-5`, and with 2000 replicates the required `2.9e-5` tail quantile lies beyond
the most extreme replicate. The substitute is the conservative alternative the same section
authorises — a studentized Bonferroni bound, `z = 4.02` at `m = 860` — and **every row
carries a `percentile_estimable` flag so the substitution is visible rather than silent**,
with the predecessor's within-contrast max-`|t|` bounds retained under their own column
names for comparability.

That is an amendment declared before outcomes, with its reason, its arithmetic and a
per-row marker. It is recorded here so that it cannot later be mistaken for a post-hoc
loosening.

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

## 7. What was deliberately not done here

The protocol, method and matrix were **read**. No fitted object, intermediate checkpoint,
log or partial score from `results/pcrl_direct_adversarial_v1/seed_*` was opened, and no
number from that study appears anywhere in the manuscript or in this package. Reading a
registered protocol is not reading a result, and the distinction is the reason the ten
checks above could be run now rather than after the fact.
