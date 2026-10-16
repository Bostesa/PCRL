# Roles, file ownership and working rules

Campaign: PCRL mechanism design (data-free). Branch `research/pcrl-mechanism-design-v1`, based on the frozen-hybrids research tip `54f089d3e`. Results directory: `results/pcrl_mechanism_design_v1/` (written `R/` below).

## Roles

| Role | Who | Owns (writes only here) | Reads |
|---|---|---|---|
| Coordinator | main session | `R/DECISION_LOG.md`, `R/roles/coordinator/`, every top-level deliverable in `R/`, `mechanism_design/` package except the verifier's directory | everything permitted |
| Adversary and methodology reviewer | agent | `R/roles/adversary_reviewer/` | papers, code, published aggregates, private review sources (never copied into public files) |
| Mechanism designer 1 | agent | `R/roles/designer_1/` | evidence pack, problem specification, closed lanes, code, published aggregates. **Not** `R/roles/designer_2/` until the coordinator reveals it |
| Mechanism designer 2 | agent | `R/roles/designer_2/` | the same, and **not** `R/roles/designer_1/` until revealed |
| Closest-prior-work reviewer | agent | `R/roles/prior_work_reviewer/` | literature (primary sources, official code), repository's earlier literature assessments |
| Mathematical and implementation verifier | agent | `R/roles/math_verifier/`, `mechanism_design_independent/` | proposals, specifications, prototype code and outputs. Must recompute core quantities with separate code |

Agents write only inside the directories they own. The coordinator alone writes `R/DECISION_LOG.md` and the top-level deliverables, and records every hand-off there.

## Data rules (binding on every role)

- Use only published aggregates (JSON/CSV/MD summary tables), code and provenance metadata.
- **Never open** real-person feature, prediction, label or model arrays in any role: nothing under any `.private/` directory other than `.private/mechanism_design/`, nothing under `~/PCRL_eval_cache_private`, nothing on the external drive, no `.npz/.npy/.parquet/.pkl/.joblib/.pt/.cbm` file from earlier campaigns, no raw ACS file, no per-person CSV (predictions, scores, exports, labels). If unsure whether a file is per-person, do not open it; ask the coordinator.
- No real-person fits, no fresh attacks, no population admission, no SEALED_ASSESSMENT access, no DEV_REPORT label reads.
- Python that does numerical work runs through the campaign queue so that CPU is ledgered:
  `~/PCRL/.venv/bin/python -m mechanism_design.jobqueue run --unit <ROLE>_<NAME>_V<n> --stage <role> --rss-gib 1.0 -- ~/PCRL/.venv/bin/python <script>`
  from the worktree root. Unit ids are append-only; a failed unit stays in the ledger and a retry gets a new version. Scripts should call `mechanism_design.no_real_data.install()` first.
- Budget for the whole campaign: 8 CPU-hours, 6 elapsed hours from 2026-10-16T04:35:52Z, $0 cloud, at most two heavy processes, one numerical thread per job.

## Independence protocol for the designers

1. Both designers receive the same inputs: `R/EVIDENCE_PACK.md`, `R/PROBLEM_SPECIFICATION.md`, `R/CLOSED_LANES.md`.
2. Each writes `R/roles/designer_<k>/PROPOSAL.md` without reading the other designer's directory. Each proposal records its own write-completion time and the SHA-256 of the final file is logged by the coordinator in the decision log before any reveal.
3. Only after both hashes are logged does the coordinator reveal each proposal to the other designer and to the reviewers.

## Selection protocol

The coordinator selects at most one proposal, and only after the adversary reviewer, the prior-work reviewer and the mathematical verifier have each written an assessment of both proposals. A vote does not substitute for resolving a mathematical objection. If neither proposal supplies a credible distinct mechanism the decision is `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED`.
