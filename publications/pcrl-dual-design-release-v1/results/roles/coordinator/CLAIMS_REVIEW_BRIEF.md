# Claims-review brief (coordinator, campaign `pcrl_dual_design_release_v1`)

You are the engineering and claims reviewer. Your job is to check that every sentence in the campaign's documents is supported by a
record. The campaign is a statistical machine-learning benchmark on admitted public-use ACS 2019 Florida records:
- two task recipients (income, occupation) receive probability vectors and argmax decisions;
- the privacy evaluation measures whether classifiers can infer the recorded SEX attribute from them;
- two release families were fitted (DA: SEX-standardized task posteriors; DB: propensity-tilt cancellation), each with matched
  controls and one revision, under the frozen DDR1 rules.

Review these documents against the records:
- `START_HERE.md`, `RESEARCH_DECISION.md`, `ADVISOR_BRIEF.md`, `HANDOFF.md`;
- the tables `TABLES.md` / `TABLES.json`;
- every number, label, time, commit and file path they state.

Records:
- `claims/*/DECISIONS.json`, `claims/*_D2R/D2R.json`, `claims/STRONGEST.json`, `claims/BESIDE_*.json`,
  `claims/CAMPAIGN_SUMMARY.json`, `claims/DB_R2_TRIGGER_*.json`;
- the fit records `da/`, `db/`, `rich/RICH_*.json`, `deploy/DEPLOYMENT_CHECK.json`, `dev/UTILITY_LEDGER.csv`;
- the cards, `PROTOCOL.md`, `PROTOCOL_AMENDMENT_1.md`, `PROTOCOL_DEVIATIONS.md`;
- `roles/verifier/` (both reports), `roles/statistical_reviewer/CLOSING_STATEMENT.md` (when present);
- `RESOURCE_ACCOUNTING.json`, `ACCESS_RECEIPT.json`, `EXPERIMENT_LEDGER.csv`, `CONFIG_LEDGER.csv`, `custody/`;
- `git log` of the branch `research/pcrl-dual-design-release-v1`.

Also regenerate the tables yourself and diff them:
`python3 -m ddr.report tables --extra-bank DDR1_R1 --extra-bank DDR1_DA_R1 --extra-bank DDR1_R1_BESIDE --extra-bank DDR1_DA_R1_BESIDE --extra-bank DDR1_R1_ZTILT --extra-bank DDR1_ROB_DB_R1`,
writing to a scratch copy, not over the campaign's files. This is label-free. If it would overwrite `TABLES.md`, copy the file first
and restore it.

Write `roles/engineering_review/REVIEW_CLAIMS.md`, or return it as text if the harness refuses report files. It should contain:
1. A numbered list of required corrections (R1, R2, ...). Each one quotes the exact sentence, names the record that contradicts or
   fails to support it, and gives the corrected wording.
2. Any claim stronger than the rules allow. Watch for:
   - a partial D2 outcome presented as a result;
   - an adaptive reading presented as registered;
   - point-only or unresolved differences called reductions;
   - "protection" beyond the declared readers;
   - novelty;
   - joint design;
   - the reserve.
3. Whether every user-required report item is present and kept separate:
   - task feasibility;
   - improvement beyond matched controls;
   - robustness to classifier refitting;
   - competitiveness against strong references;
   - the material criterion (0.020, kept apart from smaller gains);
   - joint-design evidence;
   - novelty;
   - favourable and adverse evidence;
   - actual resources;
   - untested questions;
   - the public commit.
4. Whether the documents say what was not run and why.

Read no label and run no real-data computation. Report CPU seconds.
