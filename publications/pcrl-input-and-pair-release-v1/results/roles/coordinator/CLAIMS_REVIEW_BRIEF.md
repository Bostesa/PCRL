# Claims-review brief (coordinator, campaign `pcrl_input_and_pair_release_v1`)

You are the engineering and claims reviewer. Your job is to check that every sentence in the campaign's documents is supported by a
record. The campaign is a statistical machine-learning benchmark on admitted public-use ACS 2019 Florida records:
- two task recipients (income, occupation) receive probability vectors and argmax decisions;
- the privacy evaluation measures whether classifiers can infer the recorded SEX attribute from them;
- two release families were fitted (IA: shared SEX-aware field generalization, SFG; IB: coordinated keyed selective flattening, CSF,
  SEX-blind), each with matched controls (IA: SBC, SPEND, RO; IB: SPEND), ablations and matched classifier refits, under the frozen
  IPR1 rules;
- IA's adaptive revision IA_OSG (card IA_OSG_R1: occupation-only generalization, the income output is the twin's; its SBC, SPEND
  and RO controls restricted to the occupation output) was read under the same rules (claims/IA_R1, tag ADAPTIVE_R1);
- a proposed replacement family (FS_SWAP) and IB's declared revisions were not run.

Review these documents against the records (all under `results/pcrl_input_and_pair_release_v1/`):
- `START_HERE.md`, `RESEARCH_DECISION.md`, `ADVISOR_BRIEF.md`, `HANDOFF.md`, `QUICKSTART.md`;
- the tables `TABLES.md` / `TABLES.json`;
- every number, label, time, commit and file path they state.

Records:
- `claims/*/DECISIONS.json`, `claims/*_D2R/D2R.json`, `claims/STRONGEST.json`, `claims/BESIDE_IPR1_MAIN.json`,
  `claims/CAMPAIGN_SUMMARY.json`, the plans `claims/PLAN_*.json`, `paths/SHARED_ROLES.json`;
- the fit records `ia/`, `ib/`, `rich/RICH_*.json`, `deploy/DEPLOYMENT_CHECK.json`, `dev/UTILITY_LEDGER.csv`;
- the cards, `PROTOCOL.md`, `PROTOCOL_AMENDMENT_1.md`, `PROTOCOL_DEVIATIONS.md`, `roles/coordinator/FS_SWAP_DECISION.md`;
- `roles/verifier/`, `roles/statistical_reviewer/CLOSING_STATEMENT.md` (when present), `roles/challenger/`;
- `RESOURCE_ACCOUNTING.json`, `ACCESS_RECEIPT.json`, `EXPERIMENT_LEDGER.csv`, `CONFIG_LEDGER.csv`, `custody/` (when present);
- `git log` of the branch `research/pcrl-input-and-pair-release-v1`.

Also regenerate the tables yourself and diff them: `python3 -m ipr.report tables --main IPR1_MAIN --extra-bank IPR1_ROB`, writing to a
scratch copy, not over the campaign's files (label-free). If it would overwrite `TABLES.md`, copy the file first and restore it.

Write `roles/engineering_review/REVIEW_CLAIMS.md`, or return it as text if the harness refuses report files. It should contain:
1. A numbered list of required corrections (R1, R2, ...). Each one quotes the exact sentence, names the record that contradicts or
   fails to support it, and gives the corrected wording.
2. Any claim stronger than the rules allow. Watch for:
   - a partial D2 outcome presented as a result;
   - point-only or unresolved differences called reductions;
   - "protection" beyond the declared readers;
   - a SEX-targeting or release-only claim for the SEX-blind family IB;
   - the adaptive revision IA_OSG presented as IA's result, or its refit or beside comparisons presented as D2;
   - IA's RO comparison read as an information bound (it is the declared generic-tilt fallback);
   - novelty; joint design; the reserve.
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
4. Whether the documents say what was not run and why (the replacement family, the revisions whose triggers did not hold).

Read no label and run no real-data computation. Report CPU seconds. Any time you write must come from `date -u` read first, never ahead
of the clock.
