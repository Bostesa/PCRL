# REVIEW_INDEX — final submission package

Branch `research/pcrl-submission-finish-v1`. Paper: `papers/pcrl_satml_final_v1/main.pdf`
(IEEEtran conference, 10pt, two columns; body ~5 pages of the 12 allowed; references and appendices
outside the limit). **Zero new ACS fits; 2016 not scored; no cloud compute; nothing submitted.**

| file | what it is |
|---|---|
| `REGISTRATION_READY.md` | **time-critical**: title, 228-word abstract, topics, deadlines in AoE and UTC, author checklist. Drafted only — not registered |
| `PROFESSOR_SUMMARY.md` | one paragraph plus an evidence table, no acceptance prediction |
| `CLAIM_LEDGER.csv` | 15 claims: source commit/file, estimand, split, weighting, uncertainty, correction family, verification, limitations, paper location |
| `TASK_DIRECTED_VERIFICATION.json` | 24 independent checks of the new study's headlines, all passing |
| `CORRECTIONS_FINAL.md` | C1–C9, including the refuted accuracy guarantee and the withdrawn subsumption reading |
| `REVIEW_TO_EVIDENCE.md` | review-concern → evidence map, and the status of the prior-review record |
| `NOVELTY_AND_SCOPE.md` | what is prior art (and whose), and what is actually ours |
| `ANONYMIZATION_CHECK.md` | scan results, allowlist rationale, residual risks |
| `SUBMISSION_READY_CHECK.md` | passed / missing, including the items only the author can do |
| `PENDING_INTEGRATION.md` | the fresh-year comparison: exact expected artifacts and commands |
| `checks/` | the generators and the verification script |
| `../../artifact/pcrl_satml_anon/` + `.tar.gz` | the anonymous artifact package |

## Verification actually performed

* Every new number recomputed from the smallest machine-readable source at pinned commit `f4bdf4cd5`;
  the "81 primary maps" figure traced to its protocol arithmetic (54+18+9) and its independent replay.
* Original contributions read from their pinned sources at `0176f149e`, not from an index.
* PDF: 0 errors, 0 undefined references, 0 undefined citations; every page rendered and inspected.
* Artifact scanned for author, institution, path, host and cloud identifiers: 0 hits.

## Not verified here

* The fresh-year prospective comparison (no protocol or outcome published at this writing).
* The Stage-B bootstrap of the predecessor study (per-row losses were never serialised).
* Terminal 1's cloud state — reported by its status file, not directly observed.
