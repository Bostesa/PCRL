# Quickstart (campaign `pcrl_input_and_pair_release_v1`)

How to rerun or use this campaign's code.
- **Where to run:** every command runs from the repository root of a checkout of the research branch, or of the publication's `code/`
  folder placed at the root as `ipr/`.
- **Data:** real-data steps need the admitted private copies, which are never published.
- **The queue:** every numerical unit goes through the campaign queue `ipr.jobqueue`: two slots, one thread each, CPU and time
  budgets, and an append-only ledger.
- **Lanes:** a lane file (`lanes/*.json`) is run with `python3 -m ipr.lane <lane>`, which submits its units to the queue in order.
- **Interpreters:** `TORCH` names the reader-grid interpreter and `PY` the default one.

## Label-free checks (no data needed beyond the code)

```
python3 -m ipr.test_rules              # the frozen IPR1 decision rules on invented data (28 tests)
python3 -m ipr.famkit                  # spends and the selection rule
python3 -m ipr.ctlkit                  # PSEUDO2 attribute, dithered quantization and the generic release-only tilt
python3 -m ipr.paths selftest          # bracket points, shared path points, occupation-restricted points
python3 -m ipr.ia.sfg selftest         # SFG ladder and composition (IA, registered)
python3 -m ipr.ia.sfg selftest_maps    # the generalization menu on admitted MODEL_FIT features (label-free)
python3 -m ipr.ia.osg selftest         # OSG (IA revision R1): occupation-priced ladder, income pass-through
python3 -m ipr.ib.csf selftest         # CSF keyed collapse, rates and the D5 collapse (IB)
python3 -m ipr.ctl.twin selftest       # the shared twin's payload algebra
python3 -m ipr.richreaders selftest    # data-rich readers
python3 -m ipr.release && python3 -m ipr.evaluate selftest
```

The end-to-end stage tests attach invented labels to admitted MODEL_FIT features (label-free) and write nothing to the campaign:
`python3 -m ipr.ia.sfg selftest_stage`, `python3 -m ipr.ia.osg selftest_stage` and `python3 -m ipr.ib.csf selftest_stage`.

## The campaign's run order (real data; after registration)

1. **The shared twin:** `lanes/CTL_TWIN.json` (`ipr.ctl.twin fit`; reads no label).
2. **Registered fits** (MODEL_FIT, under the committed cards): `lanes/FIT_V0.json`, `lanes/FIT_V1.json` (`ipr.ia.sfg fit`, then
   `ipr.ib.csf fit`, per version).
3. **Plans** (label-free, before any grid output): `python3 -m ipr.plan registered --bank IPR1_MAIN`. The robustness bank comes from
   `roles/coordinator/write_rob_bank.py`; the lanes from `roles/coordinator/write_eval_lanes.py`.
4. **MODEL_SELECT utility:** `lanes/EVAL_ALL.json` (`ipr.evaluate many` / `roles` / `references` / `source_ddr`). It includes the four
   units of the pre-written `lanes/REF_EVAL.json`, which was not run on its own.
5. **Reader grids:** `lanes/GRID_MAIN_V<v>.json` (`ipr.devreaders run`, locked code); data-rich readers `lanes/RICH_MAIN.json`.
6. **Registered decisions:** `lanes/CLAIMS_REG.json` (`ipr.claims run`); then `roles/coordinator/strongest.py`,
   `roles/coordinator/beside.py --bank IPR1_MAIN`, `roles/coordinator/write_d2r_plan.py --reading <reading>` and
   `python3 -m ipr.claims summary --decisions results/pcrl_input_and_pair_release_v1/claims/IA_REG results/pcrl_input_and_pair_release_v1/claims/IB_REG --out results/pcrl_input_and_pair_release_v1/claims/CAMPAIGN_SUMMARY.json`.
7. **Robustness of the registered releases:**
   - the refit grids: `lanes/GRID_ROB_V<v>.json` (units IPR_GRID_ROB2_V<v>; deviation 4);
   - the D2R readings: `lanes/D2R_REG2.json` (`ipr.claims d2r`). The first lane, `lanes/D2R_REG.json`, ran before the version-0 grid had
     finished; its NOT_RUN records are kept under `claims/superseded/` (deviation 9, `roles/coordinator/supersede_d2r.py`).
8. **IA revision R1** (card `cards/IA_OSG_R1.md`; adaptive):
   - fits: `lanes/FIT_R1_V<v>.json` (`ipr.ia.osg fit`);
   - plan: `python3 -m ipr.plan revision --bank IPR1_R1 --name IA_R1`;
   - lanes: `roles/coordinator/write_r1_lanes.py read`;
   - evaluation, grid, data-rich readers and claims: `lanes/EVAL_R1.json`, `lanes/GRID_R1_V<v>.json`, `lanes/RICH_R1.json`,
     `lanes/CLAIMS_R1.json`;
   - strongest release and beside records: `strongest.py '{"IA": [["IA_R1", "IPR1_R1"]]}'`, `beside.py --bank IPR1_R1`, and
     `python3 -m ipr.claims summary --decisions results/pcrl_input_and_pair_release_v1/claims/IA_REG results/pcrl_input_and_pair_release_v1/claims/IB_REG results/pcrl_input_and_pair_release_v1/claims/IA_R1 --out results/pcrl_input_and_pair_release_v1/claims/CAMPAIGN_SUMMARY.json`;
   - IA_OSG became IA's strongest feasible release, so its refits follow: `roles/coordinator/write_rob_r1.py` (bank IPR1_ROB_R1),
     `roles/coordinator/write_d2r_plan.py --reading IA_R1`, `lanes/GRID_ROB_R1_V<v>.json`, then `lanes/D2R_R1.json`; the descriptive
     SPEND-point refit record `roles/coordinator/osg_spend_refit.py` (writes `claims/IA_R1_SPEND_REFIT.json`).
9. **Tables:** `python3 -m ipr.report tables --main IPR1_MAIN --extra-bank IPR1_ROB --extra-bank IPR1_R1 --extra-bank IPR1_ROB_R1`.
10. **Deployment and custody:**
    - `lanes/DEPLOY_V1.json` (written by `roles/coordinator/write_deploy_lane.py`; 10 packages) and `lanes/DEPLOY_R1.json` (written by
      `roles/coordinator/write_deploy_r1_lane.py`; IA_OSG and IA_OSG_RO_HI);
    - `lanes/DEPLOY_V2.json` (written by `roles/coordinator/write_deploy_v2_lane.py`): the first ten rebuilt against the final code and
      all fourteen checked (deviation 12). A package pins the hashes of its code dependencies, so a code change after packaging needs a
      rebuild;
    - reader custody: `lanes/READER_CUSTODY2.json` (`ipr.readercustody run`). The first lane, `lanes/READER_CUSTODY.json`, failed on a
      composite winner recipe (deviation 10);
    - `python3 -m ipr.custody backup <dest>`, run as a queue unit (stage `custody`). Its record is written by
      `roles/coordinator/record_backup_attempt.py`. The private archive is compressed and its scope narrowed for disk space
      (deviation 11);
    - ledgers and receipts: `python3 -m ipr.ledger`, `python3 -m ipr.closeout`.

## Using a saved release (deployment)

`ipr.deploy` packages a configuration with its hash-pinned model files and code:
- `python3 -m ipr.deploy store --name IB/IB_CSF --version 0`
- `python3 -m ipr.deploy build --name IB/IB_CSF --version 0 --out DIR`
- `python3 -m ipr.deploy check --packages DIR ... --report OUT.json`: bitwise parity from the admitted DEV_REPORT raw fields, and
  refusal probes.

A package accepts exactly AGEP, WKHP, COW, SCHL, MAR and POBP per row. It returns, per recipient, the probability vector and the
argmax decision only. SEX, task targets and identifiers are refused. The 14 packages built here are under the private `deploy/` folder;
their final check is `deploy/DEPLOYMENT_CHECK_V2.json` (the earlier checks are `deploy/DEPLOYMENT_CHECK.json` and
`deploy/DEPLOYMENT_CHECK_R1.json`).
