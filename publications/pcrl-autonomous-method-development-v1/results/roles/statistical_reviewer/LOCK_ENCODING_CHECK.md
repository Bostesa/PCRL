# Analysis-lock encoding check (statistical reviewer)

The statistical reviewer's file write was blocked, so the coordinator saved this text verbatim from the reviewer's 17:42Z hand-back. It covers the uncommitted ANALYSIS_LOCK.json written by `amd/design.py` from the reviewer's draft.

**Verdict.** Not "ENCODING OK" yet. Four lock corrections and one correction in `amd/report.py` were required. The coordinator applied all five and regenerated the uncommitted lock before committing it (commit `345e0adb0`).

## Already correct

- **Counts.** 72 keys; K_P = 52, K_S = 20. Key names match the draft, with the placeholders resolved to TASKONLY and B2TWIN.
- **Signs.**
  - benefit: a = comparator, b = FINALIST;
  - worsening: a = FINALIST;
  - SEC AUC differences: a = PUB_IBSI or H_JJ;
  - loss differences: a = FINALIST.
- **Margins.** Utility caps 0.010 / 0.010 / 0.005; guard 0.005; benefit 0, with 0.020 as a reference only; worsening 0.010.
- **Flags.** IDENTICAL_RELEASE is on the occupation worsening keys against H_LJ and against B2TWIN, and on the SEC occupation NLL key against H_LJ.
- **Claims.**
  - E is a point gate over UG (24), SG (4) and RG (8).
  - B = CV ∧ E ∧ 4 BEN ∧ 8 NW, with the nested 0.020 test.
  - M is a fixed sequence, TO then SS. SS = H_LJ, read from B's keys. CV and E are inherited through step 1.
  - J is tested against B2TWIN.
  - N is not claimed; IBSI is descriptive.
- **Inference constants.**
  - B = 2048 and seed 2026101705, with a halt unless there are 3955 households and 5943 persons.
  - α = 0.025, n_claims = 3, σ = 0, ι = 1.
  - c1 = 2.3949991, c_P = 3.3040310, c_S = 3.0252810, marginal 1.9605641.
- **Control bands.** Planted ≥ 0.95 and |shuffled − 0.5| ≤ 0.05, for every view and seed, on the FINALIST wire.
- **TASKONLY eligibility.** It is point-eligible on MODEL_SELECT: ledger 43–44, worst normalized 0.397 / 0.315, both guards pass.
- **Refit ids.** These exist under the `amd/readers.py` naming.
- **No change needed.** The `refs [0, 0.020]` on the MECH_TO and JOINT keys are report-only. M_SS needs no separate CV/E condition, because the fixed sequence reaches it only after M_TO holds.

## Corrections (all applied before the commit)

1. **Design source.** `frozen_design_source` must point at `roles/statistical_reviewer/CLAIMS_AND_ENDPOINTS_DRAFT.md`, with `frozen_design_sha256` added. The draft is committed with the lock.
2. **REPLAY_OF_PUBLISHED flags.** Add the flag to `COMPETITIVE|occupation_worsening_vs_NEURAL_RAW|s0/s1` and to `SEC|occupation_nll_diff_vs_NEURAL_RAW|s0/s1`.
3. **Refit subset.** Add `PAIR__B2TWIN__OUT_HGB` for both seeds, so that J's comparator has an independently refit reader.
4. **`amd/report.py` endpoint().** Benefit failure categories must be tested in the order ESTABLISHED_SHORTFALL (family U < m) → POINT_SHORTFALL → UNRESOLVED. The previous order could never report ESTABLISHED_SHORTFALL.
5. **Two lock fields.**
   - (a) `inference.weights`: `task_headroom.inference.group_weights(DEV_REPORT unit, seed=bootstrap_seed, draws=bootstrap_B)`, shared across releases, endpoints and seeds.
   - (b) `in_force_by_reference`, which puts the draft's §2 E separate fields, §4, §5 and §6 in force, plus the marginal-descriptive set and its two mandatory qualifiers (FINALIST − TASKONLY occupation NLL for M; FINALIST − B2TWIN income Brier for J).
