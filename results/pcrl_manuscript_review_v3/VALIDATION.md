# VALIDATION — what this terminal checked itself, and what it did not

Terminal 2 owns the manuscript and the review. **Zero new scientific fits were run here**:
no encoder, eraser, attacker, utility probe or model-scoring job. Everything below is
arithmetic over committed evidence at one CPU worker and one BLAS/OpenMP thread.

Environment: Python 3.13.7, numpy 2.4.2, matplotlib 3.10.8
(`/Users/nathansamson/PCRL/.venv`), TeX Live with `latexmk`/`pdflatex`/`bibtex`.
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` set by the generators.

---

## 1. Provenance: every study-3 number is read at a pinned commit

Study-3 evidence is **never** read from Terminal 1's worktree. It is read out of the
shared object store with `git cat-file -p <commit>:<path>`, so the historical worktree is
untouched and each printed digit traces to a (commit, path, sha256) triple recorded in
`SOURCE_MANIFEST.json` and in `papers/pcrl_manuscript_v3/MANIFEST.json`.

| Study | Full SHA resolved |
|---|---|
| 1 — locked 2017 transport | `349efa454afd907389760fd1f59fd8806a215efd` |
| 2 — nonlinear penalty / rank | `c37807e4f568ef38e5528fc09c1506083278bf4d` |
| 3 — invariant repair + externals | `73903b7f28df68284285f0610a4036beb32b208f` |
| evidence-paper branch | `0d8f4b67b6d4961dfa133289d0167c874d2f4794` |
| manuscript base (v2) | `3c6ada82e719489656da820b72ce8425d2079be0` |

`73903b7f` is the branch tip of `research/pcrl-invariant-baselines-v1`; its own
`HANDOFF.json` (written at the parent, `34ff87d8`) lists 27 artifacts with sha256 values.

---

## 2. Checks this terminal performed

### V1 — Artifact integrity of the incoming study

Every one of the 27 artifacts listed in Study 3's `HANDOFF.json:artifact_sha256` was
recomputed from the pinned commit.

**Result: 27 of 27 match.** No artifact is missing and none has drifted.
(Script: `recheck_study3.py`; see §5.)

### V2 — Reproduction of the highest-impact paired estimates

The four contrasts that carry the paper's new claims were recomputed from
`PAIRED_INTERVALS.csv` and cross-read against `DEVELOPMENT_2018.md`'s rendered tables and
`RESEARCH_DECISION.md`'s prose, under **both** weightings and including direction and
denominator:

| contrast | checked | outcome |
|---|---|---|
| `leace_A0 − spectral_riv16_C1`, 5 endpoints × 2 weightings | 10 cells | agree; **the residence cell is unresolved, not costless** (→ `CORRECTIONS.md` B1) |
| `leace_A0 − A0` (its own source) | 10 cells | agree; residence cost **significant** both weightings |
| `spectral_riv16_C1 − J` | 10 cells | agree; residence favours the spectral arm (→ B3) |
| `repair_vs_defective`, 6 contrasts × 5 endpoints × 2 weightings | 60 cells | agree; **8 of 48 sensitive cells worse, 0 better** — the prose denominator of "4" counted contrast × endpoint rows, not weighted cells (fixed in v3) |

Two carried corrections were re-verified at `c37807e4`:
`spectral_nlr8_C1 − J` on `recovery/A/SEX` person-weighted `= +0.01210 [+0.00441,+0.01978]`,
and `spectral_nlr8_L1 − spectral_lin8_L1` on `recovery/A/RAC1P` unweighted
`= +0.00862 [+0.00168,+0.01556]`. Both stand.

### V3 — Independent location of the reporting scope

**No incoming document states which attack scope, budget and split the published paired
intervals were computed in.** Rather than assume, the value was recovered by brute force:
for each of the six scopes × two budgets × two splits, the seed-mean difference
`leace_A0 − spectral_riv16_C1` on A/sex unweighted was recomputed from `PER_SEED.csv` and
compared with the published `−0.00805`.

**Four of the 24 combinations reproduce it** (`−0.00805351`): the `expanded_catchup` and
`kernel_expanded_catchup` scopes, at budgets 120 and 360, both on `split=test`. For this
cell the budget and the kernel variant happen not to change the selected candidate, so the
scope is identified only up to that equivalence class — the load-bearing fact is that it is
a **catch-up** scope on the **test** split. No `*_independent` scope reproduces it; the
closest, `standard_independent/test`, gives `−0.01058`. The manuscript therefore names the
scope as `expanded_catchup`, budget 360, test, and the checks below are run in it, but the
conclusions drawn do not depend on resolving the remaining ambiguity (see V4).

Three consequences, all now in the manuscript and in `CORRECTIONS.md` B11:

1. The reporting scope gives `H` and the frozen historical interfaces their saved-observer
   catch-up attacks, raising the **subtracted** `H` baseline from `0.00142` to `0.00708`.
   The new arms have `own_catchup_trajectories: 0`, so their *absolute* recovery is
   identical across scopes.
2. **Every paired contrast between two arms is therefore scope-invariant**, because the
   shared baseline cancels. Verified directly:
   `0.00784 − 0.01589 = −0.00805` in absolute terms equals
   `0.00076 − 0.00881 = −0.00805` in additional terms. Every headline number in this paper
   is such a contrast.
3. `H`-relative **levels** and any **ratio** are scope-dependent by a factor of five on the
   baseline, so they are reported with both scales side by side rather than alone.

### V4 — A measured counterexample to "an attack maximum is a bound"

In the same recomputation, `J`'s absolute A/sex recovery **falls** from `0.00271` to
`0.00236` when it is given the larger candidate set. A minimum-validation-loss selection
over a larger set can pick a candidate that generalises worse on test. This is used in the
manuscript as the concrete instance behind the general warning, rather than the warning
alone.

### V5 — Internal consistency of the incoming package

Read for contradiction rather than for content. **One material discrepancy found.**

`VALIDATION.md` of Study 3 §4 records *"Inconsistent outputs: none observed; none
quarantined"* and §5 *"No such fault occurred in this run."* Both are false as written:
`RUN_STATUS.md` "Failures and repairs" item 4 documents four aborts during the
exploratory-2017 stage, and four `QUARANTINE.json` records exist under
`results/pcrl_invariant_baselines_v1/exploratory_2017/`. Commit dates explain it —
`VALIDATION.md` was committed at `4e9dc127` (2026-09-17), before that stage;
`RUN_STATUS.md` was finalised at `34ff87d8` (2026-09-18) — but a reader who stops at
`VALIDATION.md` is misinformed.

`EXPLORATORY_2017.md` and `HANDOFF.json:known_limitations_terminal_2_must_carry[5]` both
record the faults correctly, so the package is internally recoverable. **The manuscript
follows `RUN_STATUS.md`.** Reported to Terminal 1 through the handoff; **no number
changes**.

### V6 — Vocabulary audit

Every occurrence of *dominates, matches, equivalent, no utility cost, preserves utility,
causes, implicit regulariser, confirmation, optimal, certificate, novel* in the incoming
study-3 documents and in the v2 manuscript was resolved against the machine-readable
comparison files. Fourteen resolutions are recorded as `CORRECTIONS.md` B1–B14; the
per-claim outcome is `CLAIM_LEDGER.csv`.

### V7 — Claim ledger closure

28 rows, each with claim text, status, source commit, source file, source key, dataset and
year, statistical family, a **recomputed** value, a caveat, and a manuscript location.
Every number in the manuscript's prose that carries an argument appears in a ledger row or
in a generated table whose sources are hashed in `MANIFEST.json`.

### V8 — Build and page inspection

`latexmk -pdf` completes with exit status 0, with no undefined reference and no undefined
citation. **18 pages**: main text pages 1–11, appendices A–G on 12–17, references on 18.
Every page was rendered to an image and inspected.

Three material layout defects were found and fixed, not merely noted:

* Two tables ran `30`–`33`pt into the margin. Column widths narrowed; no overfull box now
  exceeds `15`pt and none is in body text.
* Long table notes inside the `tabular` stretched the table to the note's width and
  inflated whichever column absorbed the slack — visibly in the `external_vs_J` residence
  column. Notes are now emitted as LaTeX macros and set in the captions.
* Figure 3's per-point labels collided illegibly in the dense spectral cluster. Replaced
  with a shared legend and distinct marker shapes.

Cross-references, float placement, captions and the appendix/main split were checked once
each and are correct. **A clean build is not submission readiness** and no such claim is
made.

---

## 3. What this terminal did **not** check

* **Nothing was refitted, rescored or re-attacked.** Study 3's own independent score replay
  (3,264 checks, max abs difference `4.44e-16`) and its bitwise `H` parity checks are
  *reported*, not reproduced here; duplicating Terminal 1's verification pipeline was out
  of scope by assignment.
* The 40 validation fixtures and the invariance test suite were **read**, not executed.
* The bootstrap was not re-run. Intervals are taken as published.
* **ACS 2016 was not touched.** No label, output, transform or performance figure for that
  year was inspected, listed or computed by anything in this directory.
* Terminal 1's worktree was not switched, stashed, cleaned, reset or rebased, and no file
  under any historical study directory was modified.

---

## 4. Boundaries this validation cannot cross

* An internally consistent, fully hashed evidence package is not evidence that the
  underlying measurements are right. It is evidence that the reported numbers are the ones
  the pipeline produced.
* All intervals remain **development** uncertainty conditional on fitted systems. Hashing
  does not undo repeated use of the 2018 pools, and the 2017 seal is spent.
* Three seeds cannot resolve seed-to-seed variability on race recovery, which is large
  relative to the effects reported.
* The machine was at swap capacity throughout Study 3. Memory pressure is an **observed
  condition** associated with the captured faults, never a demonstrated cause, and nothing
  here establishes that the machine is now reliable.

---

## 5. Regeneration

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# shared Study 1 / Study 2 assets
PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.make_assets_v2 \
    --repo . --out papers/pcrl_manuscript_v3

# Study 3 assets, read at the pinned commit
PYTHONPATH=. python -m experiments.pcrl_manuscript_v3.make_assets_v3 \
    --repo . --out papers/pcrl_manuscript_v3

# claim ledger and source manifest
PYTHONPATH=. python -m experiments.pcrl_manuscript_v3.build_claim_ledger_v3 \
    --repo . --out results/pcrl_manuscript_review_v3

# artifact-hash and scope checks V1-V4
PYTHONPATH=. python -m experiments.pcrl_manuscript_v3.recheck_study3 \
    --repo . --out results/pcrl_manuscript_review_v3

cd papers/pcrl_manuscript_v3 && latexmk -pdf main.tex
```

Runtime: a few seconds for the assets, about a minute for the checks, well under a minute
for the PDF. No fitting, no scoring, no network access.
