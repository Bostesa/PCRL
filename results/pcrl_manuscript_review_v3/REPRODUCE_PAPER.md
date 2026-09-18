# REPRODUCE_PAPER — regenerating every number, table, figure and page

Branch `research/pcrl-manuscript-integrated-v3`. Everything here is cheap: one CPU worker,
one BLAS/OpenMP thread, no model fitting, no scoring, no refits, no network. The expensive
evidence was produced by three completed studies and is consumed as committed artifacts.

## 0. Environment

Python 3.13.7 with `numpy` 2.4.2 and `matplotlib` 3.10.8 (the project venv at
`/Users/nathansamson/PCRL/.venv` on the original machine). TeX Live with `latexmk`,
`pdflatex` and `bibtex`. The generators set
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` themselves.

```bash
git checkout research/pcrl-manuscript-integrated-v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## 1. Where the evidence comes from

Nothing is read from another terminal's worktree. Study-3 artifacts are read out of the
shared object store at a pinned commit:

```bash
git cat-file -p 73903b7f28df68284285f0610a4036beb32b208f:results/pcrl_invariant_baselines_v1/PAIRED_INTERVALS.csv
```

| Study | Full SHA | Directory (preserved byte-for-byte) |
|---|---|---|
| 1 — locked 2017 transport | `349efa454afd907389760fd1f59fd8806a215efd` | `results/redesign_20260917_acs_spectral_transport_v1/` |
| 2 — nonlinear penalty / rank | `c37807e4f568ef38e5528fc09c1506083278bf4d` | `results/pcrl_nonlinear_rank_v1/` |
| 3 — invariant repair + externals | `73903b7f28df68284285f0610a4036beb32b208f` | `results/pcrl_invariant_baselines_v1/` |
| evidence-paper branch | `0d8f4b67b6d4961dfa133289d0167c874d2f4794` | `results/pcrl_evidence_review_v1/` |
| manuscript base (v2) | `3c6ada82e719489656da820b72ce8425d2079be0` | `papers/pcrl_manuscript_v2/` |

## 2. Tables and figures

Two generators, in this order. The first writes the Study 1 / Study 2 assets and
`MANIFEST.json`; the second adds the Study 3 assets and **merges** into that manifest, so
running them out of order loses the v3 entries.

```bash
# Study 1 and Study 2 assets (13 tables, 7 figures)
PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.make_assets_v2 \
    --repo . --out papers/pcrl_manuscript_v3

# Study 3 assets (9 tables incl. 8 caption-note macros, 2 figures)
PYTHONPATH=. python -m experiments.pcrl_manuscript_v3.make_assets_v3 \
    --repo . --out papers/pcrl_manuscript_v3
```

Writes into `papers/pcrl_manuscript_v3/`:

* `tables/*.tex` — each a bare `tabular`. Long notes are emitted separately as
  `tables/<name>_note.tex`, each defining a LaTeX macro, and are set in the **caption**.
  (A note inside the `tabular` stretches the table to the note's width and inflates
  whichever column absorbs the slack; this was a real defect in the first build.)
* `figures/*.{pdf,png}`
* `MANIFEST.json` — per asset, the generating function and a **sha256 of every evidence
  file it read**, plus `v3_pinned_commits` and `v3_pinned_sources` giving
  (commit, path, sha256, bytes) for each Study-3 blob. This is the audit trail from a
  printed digit back to a committed artifact.
* `DERIVED_FACTS.json`, `DERIVED_FACTS_V3.json` — recomputed headline counts, so a reader
  can diff them against the prose.

Runtime: a few seconds.

## 3. Claim ledger and source manifest

```bash
PYTHONPATH=. python -m experiments.pcrl_manuscript_v3.build_claim_ledger_v3 \
    --repo . --out results/pcrl_manuscript_review_v3
```

Writes `CLAIM_LEDGER.csv` (28 claims: claim text, status, source commit, source file,
source key, dataset/year, statistical family, a value **recomputed from evidence rather
than copied from prose**, caveat, manuscript location) and `SOURCE_MANIFEST.json`.

## 4. Independent re-checks

```bash
PYTHONPATH=. python -m experiments.pcrl_manuscript_v3.recheck_study3 \
    --repo . --out results/pcrl_manuscript_review_v3
```

Writes `VERIFICATION_V3.json` and exits non-zero on any failure. Checks V1–V5 of
`VALIDATION.md`: all 27 Study-3 artifact hashes; the highest-impact paired cells under both
weightings with their denominators; **brute-force location of the reporting scope** (24
combinations tried, 4 reproduce, all catch-up scopes on the test split); scope-invariance of
paired contrasts plus the measured instance of a selected attack generalising worse; and
the internal-consistency contradiction in the incoming package.

Expected output:

```
V1_artifact_hashes: PASS      V1 artifacts checked: 27 mismatches: []
V2_headline_cells: PASS
V3_reporting_scope: PASS
V4_scope_invariance: PASS
V5_internal_consistency: PASS  V5 quarantine records: 4 discrepancy: True
```

`V5 discrepancy: True` is the **expected** result: it records that Study 3's own
`VALIDATION.md` contradicts its `RUN_STATUS.md` about numerical faults. The check passes
because the discrepancy is detected and dispositioned, not because it is absent
(`CORRECTIONS.md` B10).

## 5. The PDF

```bash
cd papers/pcrl_manuscript_v3 && latexmk -pdf main.tex
```

18 pages: main text 1–11, appendices A–G on 12–17, references on 18. Exit status 0, no
undefined reference, no undefined citation, no overfull box in body text.

To re-inspect every page as an image, as this revision did:

```bash
pdftoppm -r 100 -png main.pdf /tmp/v3pages/p
```

## 6. Stale-asset detection

`MANIFEST.json` records a sha256 for every evidence file each asset read. To detect an
asset generated from evidence that has since changed:

```bash
PYTHONPATH=. python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
m = json.load(open('papers/pcrl_manuscript_v3/MANIFEST.json'))
bad = []
for key, rec in m.get('v3_pinned_sources', {}).items():
    raw = subprocess.run(['git','cat-file','-p',f"{rec['commit']}:{rec['path']}"],
                         capture_output=True, check=True).stdout
    if hashlib.sha256(raw).hexdigest() != rec['sha256']:
        bad.append(key)
for name, rec in m['assets'].items():
    for rel, want in rec.get('sources', {}).items():
        if isinstance(want, str) and Path(rel).exists():
            got = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
            if got != want:
                bad.append(f'{name} <- {rel}')
print('STALE:', bad or 'none')
PY
```

A pinned-commit source can only go stale if the commit is rewritten, which would itself be
a finding.

## 7. What is deliberately not reproducible here

No fitted object, release array or per-person prediction is regenerated: this terminal runs
no encoder, eraser, attacker, utility probe or model-scoring job, by assignment. Those come
from the three completed studies and each has its own `REPRODUCE.md`. Study 3's independent
score replay (3,264 checks, max abs difference `4.44e-16`) and its bitwise `H`-parity checks
are *reported* here, not re-executed.

**ACS 2016 is not reachable from anything in this directory.**
