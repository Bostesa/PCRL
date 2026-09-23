# Inner pilot implementation record

Recorded 2026-09-23 12:14 UTC before any new inner-pilot candidate scores were opened. This note does not alter the committed protocol, signed queue, endpoints, budgets, or final audit slate.

Each registered `audit-inner` unit fits the inherited `standard` slate uniformly for the candidate and its H-only and legal A/B ancestors on `downstream_fit`, selects predictor routes on the globally assigned `inner_selection` households, then computes exact expected-token losses once on `inner_pilot`. Both H and candidate use the same person and household masks for each role. It stores fitted weights, per-household contributions and original-person identifiers only in the ignored private unit directory. Final outer assessment, if reached after selection lock, uses the separate registered `catchup` slate. This inner pilot is descriptive 2018 development evidence, not a fresh confirmation.
