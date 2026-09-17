# Review index — Terminal B independent evidence audit and manuscript

Branch `research/pcrl-evidence-paper-v1`. Baseline `349efa454afd907389760fd1f59fd8806a215efd`.
Nothing in `results/redesign_20260917_acs_spectral_transport_v1` was edited; this is a companion.

## Start here

| Question | File |
|---|---|
| What changed in the scientific claims, and what did not? | [ERRATA.md](ERRATA.md) |
| The corrected conclusion | [CORRECTED_RESEARCH_DECISION.md](CORRECTED_RESEARCH_DECISION.md) |
| The manuscript | [`papers/pcrl_evidence_v1/main.pdf`](../../papers/pcrl_evidence_v1/main.pdf) (source `main.tex`, bibliography `references.bib`) |
| Every claim, its verdict and a command that regenerates its number | [CLAIM_AUDIT.csv](CLAIM_AUDIT.csv), [EVIDENCE_LEDGER.csv](EVIDENCE_LEDGER.csv) |
| The statistics, reconstructed independently | [STATISTICAL_REANALYSIS.md](STATISTICAL_REANALYSIS.md) |
| Mathematical review of the proposed follow-up | [METHOD_REVIEW.md](METHOD_REVIEW.md) |
| Verified relation to prior work | [NOVELTY_MATRIX.md](NOVELTY_MATRIX.md), [CONTRIBUTION_ASSESSMENT.md](CONTRIBUTION_ASSESSMENT.md) |
| Is 2016 usable, and on what terms? | [DATA_2016_ADMISSION.md](DATA_2016_ADMISSION.md), [ACS_2016_ADMISSION_MANIFEST.json](ACS_2016_ADMISSION_MANIFEST.json) |
| What a reviewer will object to | [REVIEWER_RISKS.md](REVIEWER_RISKS.md) |
| How to rerun all of it | [REPRODUCE.md](REPRODUCE.md) |
| What was checked, and what the checks cannot show | [VALIDATION.md](VALIDATION.md) |
| Abstract claims linked to evidence | [`papers/pcrl_evidence_v1/CONTRIBUTION_LEDGER.md`](../../papers/pcrl_evidence_v1/CONTRIBUTION_LEDGER.md) |

## Generated evidence

| File | Contents |
|---|---|
| `INDEPENDENT_FAMILIES.csv` | all 90 F1–F4 endpoints, recomputed: estimate, per-seed, se, unadjusted and adjusted intervals, one-sided bounds |
| `FAMILY_AGREEMENT.json` | row-by-row agreement against the study's `FAMILIES.csv` |
| `EQUIVALENCE_F1.csv` | retrospective non-inferiority and equivalence at the 0.001 margin |
| `MODEB_TRANSPORT_TABLE.csv` | absolute, H-baseline and additional recovery per interface, both modes |
| `SELECTION_RECHECK.json` | selection re-derived from stored validation losses |
| `PER_SEED_DIRECTION.csv` | per-seed sign agreement with the seed mean and with development |
| `DEV_VS_TRANSPORT.csv` | development vs transport magnitudes and ratios |
| `SCOPE_COMPARISON.csv` | `common_fresh` vs `transport_all` additional recovery |
| `VECTOR_C1_J_CONTROLS.csv` | the full comparison vector, no exchange rate applied |
| `WITHHOLDING_MATCHING.csv` | post-hoc average-utility matching, both directions |
| `SERVICE_VS_PROBE.csv`, `SERVICE_QUALITY_SUMMARY.csv` | exact parity vs probe allowances; service accuracy after shift |
| `LOCK_RECHECK.json` | 17,639 locked inputs re-verified with all three amendments |
| `TRADEOFF_SUMMARY.json` | headline counts |

## Code and tests

`experiments/pcrl_evidence_review_v1/` — `independent_scorer.py` (own loss, selection, bootstrap and
max-|t| implementation), `reanalyse_transport.py`, `tradeoff_tables.py`, `recheck_lock.py`,
`build_claim_audit.py`, `admit_acs_2016.py`, `sealed_year_loader.py`, `make_paper_assets.py`.

`tests/pcrl_evidence_review_v1/` — 21 tests: 8 mathematical fixtures, 13 sealed-year loader tests.

## Pending addendum — Terminal A's new mechanism

**Status at the time this package was completed: not available.** Terminal A
(`research/pcrl-nonlinear-rank-v1`) had committed its prospective protocol and implementation at
`a6ce3a9e1afc55cec2775cea4d074d1f7d2c3852` and reported its fit phase still running, with 2018
development audits, replay and report pending. **No result of Terminal A is incorporated anywhere in
this package, and none is claimed.** This manuscript stands on the completed 2017 study alone.

When Terminal A publishes a committed, verified result:

1. Read its `PROTOCOL.md`, `METHOD.md`, per-seed table, validation record and research decision **at
   the stated SHA** — not from a status message, and not from a partially written CSV.
2. Add it as a clearly separated development section or appendix. **Its 2017 numbers are exploratory
   for the new method** and must not be placed under this study's independent-confirmation claim;
   Terminal A's own protocol already labels them that way.
3. Carry forward findings A1–A7 of the Terminal B handoff, in particular that eigenvalue-sign rank
   selection is prior work (SARL Thm 3, OptNet-ARL Thm 4.1, K-TOpt Cor 4.1), that a penalty nonlinear
   in `Z` admits no closed-form spectral optimiser, and that no family-level negative inference may be
   drawn. Terminal A has recorded its agreement with all seven.
4. Integration, if both sides are complete: create a **third** worktree from baseline `349efa4`,
   merge only the intended committed ranges of `research/pcrl-nonlinear-rank-v1` and
   `research/pcrl-evidence-paper-v1`, preserve both directory trees, run
   `pytest tests/pcrl_evidence_review_v1 tests/pcrl_nonlinear_rank_v1 -q` and rebuild the PDF as a
   lightweight check, rerun no scientific fit, push the integration branch, and record both source
   SHAs. Never merge to `main` and never reset either working terminal:

   ```bash
   cd /Users/nathansamson/PCRL
   git worktree add -b research/pcrl-integration-v1 ../PCRL-integration 349efa4
   cd ../PCRL-integration
   git merge --no-ff research/pcrl-evidence-paper-v1
   git merge --no-ff research/pcrl-nonlinear-rank-v1
   PYTHONPATH=. /Users/nathansamson/PCRL/.venv/bin/python -m pytest \
       tests/pcrl_evidence_review_v1 tests/pcrl_nonlinear_rank_v1 -q
   (cd papers/pcrl_evidence_v1 && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
   git push -u origin research/pcrl-integration-v1
   ```
