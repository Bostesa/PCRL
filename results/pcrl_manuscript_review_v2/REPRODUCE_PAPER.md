# REPRODUCE_PAPER — regenerating every number, table, figure and page

Branch `research/pcrl-manuscript-integrated-v2`. Everything below is cheap: one CPU worker, one
BLAS thread, no model fitting, no scoring, no refits. The expensive evidence was produced by the
two completed studies and is consumed here as committed artifacts.

## 0. Environment

```bash
git clone <repo> && cd <repo>
git checkout research/pcrl-manuscript-integrated-v2
```

Python 3.13.7 with `numpy` 2.4.2 and `matplotlib` 3.10.8 (the project venv at
`/Users/nathansamson/PCRL/.venv` on the original machine). TeX Live 2026 with `latexmk`,
`pdflatex` and `bibtex` for the PDF. The generators set
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` themselves.

## 1. Tables and figures

```bash
PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.make_assets_v2 \
    --repo . --out papers/pcrl_manuscript_v2
```

Writes 13 `tables/*.tex` and 7 `figures/*.{pdf,png}`, plus:

* `papers/pcrl_manuscript_v2/MANIFEST.json` — for each asset, the generating function and a
  **sha256 of every evidence file it read**. This is the audit trail from a printed digit back to a
  committed artifact.
* `papers/pcrl_manuscript_v2/DERIVED_FACTS.json` — the recomputed headline counts (family
  denominators, equivalence outcomes, attribution counts, rotation shares, `p*` values), so a
  reviewer can diff them against the prose.

Runtime: a few seconds.

## 2. Claim ledger

```bash
PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.build_claim_ledger \
    --repo . --out results/pcrl_manuscript_review_v2
```

Writes `CLAIM_LEDGER.csv`: 25 claims, each with its type (structural / restricted-math / empirical
/ diagnostic / hypothesis / open), a **value recomputed from evidence rather than copied from
prose**, source commit, source artifact, endpoint or table, data pool, evaluation status, attack
scope, uncertainty procedure, limitation, and a regeneration command.

## 3. Corrected lock verification

```bash
PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.recheck_verification \
    --study-root /Users/nathansamson/.config/superpowers/worktrees/PCRL/residual-spectral-20260910 \
    --published results/redesign_20260917_acs_spectral_transport_v1 \
    --out results/pcrl_manuscript_review_v2
```

Read-only; hashes 17,639 inputs (1.80 GiB) in about 5 s on one worker and writes
`CORRECTED_INDEPENDENT_VERIFICATION.json`. `--study-root` must be a checkout where the study's
**local, untracked** fitted objects, releases and per-person arrays resolve — normally the worktree
the study ran in. Without it the check reports the 17,540 inputs it cannot see rather than passing
vacuously.

## 4. Tests

```bash
PYTHONPATH=. python -m pytest tests/pcrl_evidence_review_v1 tests/pcrl_nonlinear_rank_v1 -q
```

21 mathematical/sealed-loader tests from the evidence audit, plus the nonlinear-moment fixtures.
These test the *reasoning*, not the ACS results: exact enumeration of the XOR and magnitude
counterexamples, the conditional null under a correct versus a constant nuisance, the free-rank
optimum, the trace-form invariance argument, and thirteen sealed-year loader refusals.

## 5. The PDF

```bash
cd papers/pcrl_manuscript_v2
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: 22 pages, **0 overfull or underfull boxes, 0 undefined references or citations, 0 LaTeX
warnings**. Every `\input` table and every `\includegraphics` figure comes from step 1; no number
in the manuscript is typed by hand.

To inspect the rendered pages as they were checked here:

```bash
pdftoppm -png -r 100 main.pdf /tmp/page
```

## 6. Integration manifest

```bash
PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.build_integration_manifest \
    --repo . --out results/pcrl_manuscript_review_v2
```

Records the three source SHAs, the commit lists integrated from each, the merge method, and a
sha256 for every deliverable. Run it **last**, after the PDF exists.

## 7. What is deliberately not reproducible from this branch

* **The two studies themselves.** Their fitted objects, releases and per-person prediction arrays
  are large and stay local; only compact aggregates, code, tables and figures are published. The
  per-fit records of the nonlinear/rank study *are* committed under
  `results/pcrl_nonlinear_rank_v1/seed_*/` and `exploratory_2017/`, and their own `REPRODUCE.md`
  files describe rerunning them.
* **The 2017 evaluation.** Its seal is spent. The lock will refuse to reopen the final partition if
  any hashed input changed, and reopening it would not produce new confirmatory evidence in any
  case.
* **Anything on ACS 2016.** Admitted, split prepared, **unscored**, and nothing in this branch
  scores it.

## 8. File map

| Path | What it is |
|---|---|
| `papers/pcrl_manuscript_v2/` | the integrated manuscript: `main.tex`, `main.pdf`, `references.bib`, `tables/`, `figures/`, `MANIFEST.json`, `DERIVED_FACTS.json` |
| `results/pcrl_manuscript_review_v2/` | ledger, corrections, corrected verification, scope notes, risks, status, this file |
| `experiments/pcrl_manuscript_v2/` | the four generators above |
| `results/redesign_20260917_acs_spectral_transport_v1/` | Study 1, historical, unedited |
| `results/pcrl_evidence_review_v1/` | the independent evidence audit, historical, unedited |
| `results/pcrl_nonlinear_rank_v1/` | Study 2, historical, unedited |
