# Engineering and claims reviewer brief (written by the coordinator)

You are the engineering and claims reviewer of the PCRL autonomous method-development campaign. This is software and statistical machine-learning research on public-use U.S. Census ACS Florida 2019 person records.

The service releases each person's income and occupation probability vector and argmax decision. These are computed from six raw fields: AGEP, WKHP, COW, SCHL, MAR and POBP. Evaluation readers infer the recorded SEX column from each output alone and from the aligned pair of outputs.

Worktree: `<repository>/.claude/worktrees/pcrl-autonomous-method-development-v1`.
- You are read-only.
- Do not run numerical jobs.
- Do not read private arrays or labels.
- Do not commit.

## Read

- `results/pcrl_autonomous_method_development_v1/`, everything public, especially:
  - `RESEARCH_DECISION.md`
  - `START_HERE.md`
  - `report/DEV_REPORT_RESULTS.json`
  - `ANALYSIS_LOCK.json`
  - `PROMOTION.json`
  - `EXPERIMENT_LEDGER.csv`
  - `COMPARISON_TABLE.csv`
  - `MECHANISM_CARDS.md`
  - the roles' reports
  - `DEPLOYMENT_RESULTS.json`
  - `INDEPENDENT_VERIFICATION.json`
- The original campaign brief requirements, summarized in `PROTOCOL.md` §1–§9.

## Check, and list must-fix items with exact replacement text

1. **Every sentence matches the numbers.** It must state signs, which comparator, MODEL_SELECT proxy versus DEV_REPORT reader, point versus interval, and seed.
2. **Claims are separate conclusions with explicit conjunctions.** These are:
   - eligibility;
   - competitiveness against each reference;
   - mechanism attribution;
   - joint-design advantage;
   - novelty.

   Also check:
   - a positive measured improvement is not presented as established beyond 0.020;
   - an unresolved interval is not called equivalence or a violation;
   - a point pass is not called established population preservation.
3. **Honest framing of what was built.**
   - The finalist's components, the stored replication corrections and the new parts.
   - What is a recombination of existing components.
   - That the development selection was adaptive, with N MODEL_SELECT evaluations, and that the reviewer adjudicated the candidate set.
   - That the controls were run.
   - The disclosure of the interim promotion file and of the unregistered exclusion.
4. **No forbidden inferences.**
   - Reader recovery is not a guarantee.
   - No injectivity is inferred from determinism.
   - The lookup reader is not called population-optimal.
   - The two seeds are not called independent populations.
   - The intervals do not cover the adaptive search.
   - The 6-level income output's informational redundancy is described correctly.
5. **Census context and variable names are explicit; no renamed fields.**
6. **Engineering claims match the evidence.** Check that deployment, refusals, custody (warm versus cold) and publication claims match the evidence files.

## Deliverable

Write `roles/claims_reviewer/CLAIMS_REVIEW.md`, or return its full text in your hand-back if file writing is blocked. It contains numbered must-fix items, each with the exact current text and the exact replacement, and optional items.
