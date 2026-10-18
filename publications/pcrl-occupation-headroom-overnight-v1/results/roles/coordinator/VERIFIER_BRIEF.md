# Brief for the independent verifier (coordinator; verification starts as artifacts become available)

Read `AGENT_BRIEF_COMMON.md`. You verify with SEPARATE numerical code (your own implementation under
`results/pcrl_occupation_headroom_overnight_v1/roles/verifier/code/`, never importing `ohn.evaluate`, `ohn.devreaders`,
`ohn.rules` or `ihs.evaluate` for the quantity you check; you may use `ohn.common` loaders and hash helpers for data access,
which log every label read under your unit ids). Numerical PASS is separated from documentation review; every unresolved item
stays visible.

Checks, in the order the artifacts appear:
1. Source and household-role integrity: re-hash the admitted mirror against `SOURCE_ADMISSION.json` and the source's committed
   records; household disjointness of all six roles; the registered `DEV_GRID.json` split (FIT / VALIDATION / SCORE disjoint per
   repetition; counts) and the cross-fitting folds (households wholly within a fold).
2. Exposure: compare `EXPOSURE_TABLE.md` with the source modules and records; check every OCC.json's roles_read / exposure against
   the label-access logs (`.private/<campaign>/access/*.jsonl`): no ATTACK / DEV_REPORT label read before the gates, MODEL_SELECT
   read only by scoring / grid units, no SEX in task-only fits.
3. Utility and subgroup metrics: recompute the 12 clauses and 2 guards (values and household-clustered SEs) of every reference and
   of the selected witnesses / privacy-stage candidates on MODEL_SELECT from the stored arrays; cross-fitting diagnostics.
4. Reserve calculations and selection: recompute the whole-release reserve ratios (k = 2, H_LJ floor) and the witness labels with
   your own code from the registered RULES.json; the income-branch trigger.
5. Actual transmitted outputs: probability-vector contract, argmax decisions, wire hashes; `predict` from features reproduces the
   stored release for sampled configurations.
6. Development stability summaries: recompute paired benefits and the stability labels from the stored grid score vectors;
   independently refit a representative subset of readers (including the comparators and the strongest recipes) and compare.
7. If a final stage runs: the final readers, controls, household-aware intervals and the exact claim conjunctions, including a
   separately logged DEV_REPORT replay under the registered verifier event.
8. Raw-field deployment from restored artifacts (backup), refusals at both entry points.
Return the full report as the TEXT of your final response (the coordinator saves it); write your code and machine-readable check
records yourself.
