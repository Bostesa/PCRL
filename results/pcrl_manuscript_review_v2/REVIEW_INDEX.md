# REVIEW_INDEX — the integrated manuscript and everything behind it

Branch `research/pcrl-manuscript-integrated-v2`. Baseline
[`349efa45`](../../commit/349efa454afd907389760fd1f59fd8806a215efd). This directory and
`papers/pcrl_manuscript_v2/` are the only things it adds; all three historical trees are preserved
byte-for-byte.

## Start here

| Question | File |
|---|---|
| **The paper** | [`papers/pcrl_manuscript_v2/main.pdf`](../../papers/pcrl_manuscript_v2/main.pdf) (source `main.tex`, 22 pages) |
| What does this revision change, and on what evidence? | [CLAIM_CHANGES.md](CLAIM_CHANGES.md) |
| Every claim, its type, its scope and a command that regenerates its number | [CLAIM_LEDGER.csv](CLAIM_LEDGER.csv) |
| The conclusion both studies jointly support | [CORRECTED_RESEARCH_DECISION.md](CORRECTED_RESEARCH_DECISION.md) |
| What we claim about prior work, and what a fair baseline would need | [RELATED_WORK_SCOPE.md](RELATED_WORK_SCOPE.md) |
| What has been checked, and what each check cannot show | [VERIFICATION_STATUS.md](VERIFICATION_STATUS.md) |
| What a reviewer will object to — writing fixes vs scientific gaps | [REVIEWER_RISKS.md](REVIEWER_RISKS.md) |
| How to regenerate every number, table, figure and page | [REPRODUCE_PAPER.md](REPRODUCE_PAPER.md) |
| What was integrated, from which commits, with hashes | [INTEGRATION_MANIFEST.json](INTEGRATION_MANIFEST.json) |
| Restartable stage state and measured runtime | [RUN_STATUS.md](RUN_STATUS.md) |
| Terminal 1's experiment | [PENDING_ADDENDUM.md](PENDING_ADDENDUM.md) — **not started; nothing incorporated** |

## The two completed studies, at immutable SHAs

| Study | Branch | SHA | Evidence directory | Role |
|---|---|---|---|---|
| **Study 1** — locked temporal transport of a frozen 14-interface slate to ACS 2017 | `research/pcrl-evidence-paper-v1` | `0d8f4b67b6d4961dfa133289d0167c874d2f4794` | `results/redesign_20260917_acs_spectral_transport_v1/` | the only **confirmatory** evidence in this line of work |
| — its independent reanalysis, errata and novelty verification | same | same | `results/pcrl_evidence_review_v1/` | all 90 endpoints reproduced to 1.1e-15 |
| — its manuscript, superseded by this one | same | same | `papers/pcrl_evidence_v1/` | preserved; this revision extends it |
| **Study 2** — nonlinear-moment refinement and rank intervention on ACS 2018 | `research/pcrl-nonlinear-rank-v1` | `c37807e4f568ef38e5528fc09c1506083278bf4d` | `results/pcrl_nonlinear_rank_v1/` | **development**, prospectively registered, plus an **exploratory** 2017 reuse |

Both descend from `349efa45` and touch disjoint paths; the merge was clean. Nothing in any of the
three evidence directories was edited.

## Corrected companions to stale artifacts

Published beside the originals, never replacing them.

| Stale artifact | Why | Companion |
|---|---|---|
| `redesign_.../INDEPENDENT_VERIFICATION.json` | predates lock amendment 3 | [CORRECTED_INDEPENDENT_VERIFICATION.json](CORRECTED_INDEPENDENT_VERIFICATION.json) — 17,639 inputs, all three amendments, 0 changed |
| `pcrl_evidence_review_v1/LOCK_RECHECK.json` | its `amendments_covered_by_published_verification` field contradicts its own `note` | same file; see [VERIFICATION_STATUS.md](VERIFICATION_STATUS.md) §1 |
| `pcrl_evidence_review_v1/REVIEW_INDEX.md` "pending addendum" | says Study 2 is still running; it is complete and integrated | [PENDING_ADDENDUM.md](PENDING_ADDENDUM.md) |
| `pcrl_nonlinear_rank_v1/RESEARCH_DECISION.md` "2016 admission has never been run" | it has been run | `pcrl_evidence_review_v1/DATA_2016_ADMISSION.md`; status table below |
| Study 2's attribution and rotation readings | denominator and causal-claim errors | [CLAIM_CHANGES.md](CLAIM_CHANGES.md) A2, A3 |
| Study 1's "8 of 8 sensitive endpoints" | count overstated as written | [CLAIM_CHANGES.md](CLAIM_CHANGES.md) A1 |

## Data-pool status

| Pool | Status | Intervals? | Notes |
|---|---|---|---|
| ACS 2018 | **development**, used repeatedly | yes (development bootstrap) | cannot confirm anything |
| ACS 2017, original seal | **confirmatory**, spent | yes, simultaneous | historical status preserved; not restated or overwritten |
| ACS 2017, after the seal | **exploratory reuse** | **none, by design** | Study 2 only; never pooled with the frozen result |
| ACS 2016 | **admitted, UNSCORED** | n/a | provenance and schema checked, prospective label-blind split prepared; no fit, no outcome inspected, no diagnostic produced |

## Generators

`experiments/pcrl_manuscript_v2/` — `make_assets_v2.py` (all tables and figures, with per-asset
source hashes), `build_claim_ledger.py`, `recheck_verification.py` (read-only), and
`build_integration_manifest.py`. Tests: `tests/pcrl_evidence_review_v1/` (21) and
`tests/pcrl_nonlinear_rank_v1/`.

## Cross-terminal status

Terminal 2 (this branch) owns `papers/pcrl_manuscript_v2/`,
`results/pcrl_manuscript_review_v2/` and `experiments/pcrl_manuscript_v2/`. It ran no fit, no score
and no refit. Terminal 1's worktree is clean at `c37807e` — the study already integrated here — and
**no new Terminal 1 experiment exists**; no claim in this package depends on one. Stale assertions
in either direction have been removed from this index and are corrected in
[CLAIM_CHANGES.md](CLAIM_CHANGES.md) A7.
