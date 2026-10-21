# Quickstart (campaign `pcrl_dual_design_release_v1`)

How to rerun or use this campaign's code. Every command runs from the repository root of a checkout of the research branch, or of
the publication's `code/` folder placed at the root as `ddr/`. Real-data steps need the admitted private copies, which are never
published. Every numerical unit goes through the campaign queue `ddr.jobqueue`: two slots, one thread each, CPU and time budgets,
and an append-only ledger. A lane file (`lanes/*.json`) is run with `python3 -m ddr.lane <lane>`, which submits its units to the
queue in order. `TORCH` names the reader-grid interpreter and `PY` the default one.

## Label-free checks (no data needed beyond the code)

```
python3 -m ddr.test_rules              # the frozen DDR1 decision rules on invented data (22 tests)
python3 -m ddr.famkit                  # spends and the selection rule
python3 -m ddr.ctlkit                  # PSEUDO2 attribute and the dithered-quantization spend path
python3 -m ddr.da.std selftest         # STD algebra, weight rule, bracket points
python3 -m ddr.da.ipw selftest         # IPW weights (revision R1)
python3 -m ddr.db.ptc selftest         # PTC covariance cancellation, nested releases
python3 -m ddr.db.ptc_r1 selftest      # R1 pair selection
python3 -m ddr.richreaders selftest    # data-rich readers
python3 -m ddr.release && python3 -m ddr.evaluate selftest
```

The end-to-end stage tests attach invented labels to admitted MODEL_FIT features (label-free) and write nothing to the campaign:
`python3 -m ddr.da.std selftest_stage`, `python3 -m ddr.da.ipw selftest_stage`, and
`python3 -m ddr.db.ptc selftest_stage --n 4000 --max-iter 200`.

## The campaign's run order (real data; after registration)

1. **Registered fits** (MODEL_FIT, under the committed cards): `lanes/FIT_V0.json`, `lanes/FIT_V1.json`. These run DA
   (`ddr.da.std fit`) and DB (`ddr.db.ptc fit`) per version.
2. **Plans** (label-free, before any grid output): `python3 -m ddr.plan registered --bank DDR1_MAIN`. The robustness bank comes
   from `roles/coordinator/write_rob_bank.py`.
3. **MODEL_SELECT utility:** `lanes/EVAL_ALL.json` (`ddr.evaluate many` / `references`); `lanes/EVAL_DB_ZTILT.json` (deviation 5).
4. **Reader grids:** `lanes/GRID_MAIN_V<v>.json` (`ddr.devreaders run`, locked code). Robustness: `lanes/GRID_ROB2_V0.json` and
   `lanes/GRID_ROB3_V1.json`. Data-rich readers: `lanes/RICH_MAIN.json`. `ddr/richreaders.py` changed after this run (commit
   1c7064bd6, which added the revision reader files); reproducing the run with the current code was not checked.
5. **Registered decisions:** `lanes/CLAIMS_REG.json` (`ddr.claims run`).
6. **DB revision R1:**
   - fits: `lanes/DB_R1_FIT_V<v>.json`;
   - plan: `python3 -m ddr.plan revision --bank DDR1_R1 --family DB --rev R1 --card results/pcrl_dual_design_release_v1/cards/DB_PTC_R1.md --cand DB_PTC_R1 --sbc DB_SBC_R1_LO DB_SBC_R1_HI --spend DB_SPD_LO DB_SPD_HI`;
   - extras: `roles/coordinator/write_revision_extras.py --plan claims/PLAN_DB_R1.json --bank DDR1_R1`;
   - evaluation, grid and claims: `lanes/DB_R1_READ.json`, then `lanes/DB_R1_EXTRAS.json`.
7. **DA revision R1:**
   - trigger: `lanes/DA_R1_TRIGGER.json` (`ddr.da.std trigger_r1`);
   - fits: `lanes/DA_R1_FIT_V<v>.json` (`ddr.da.ipw fit`);
   - plan and extras: as in step 6, with `--family DA --cand DA_IPW`;
   - reading: `lanes/DA_R1_READ.json`.
8. **Strongest release and its additions:**
   - `roles/coordinator/strongest.py` writes `claims/STRONGEST.json`;
   - DB_PTC_R1's ablation: `lanes/DB_R1_ABL.json` (`ddr.db.ptc_r1 ablate`);
   - its refit bank and D2R plan: `roles/coordinator/write_rob_revision.py --reading DB_R1 --ablations DB/DB_PTC_R1_OCC` and
     `write_d2r_plan.py --reading DB_R1 --ablations DB_PTC_R1_OCC`;
   - its refit grid and D2R: `lanes/ROB_DB_R1.json`;
   - its release-only control DB_ZTILT_R1 (intended to match its spend; it overspent): `lanes/ZTILT_R1_FIT.json` (`ddr.db.ptc_r1 ztilt`), then `roles/coordinator/write_ztilt_r1_bank.py`
     and `lanes/ZTILT_R1_GRID.json`; its declared reading: `roles/coordinator/ztilt_r1_reading.py` (writes
     `claims/ZTILT_R1_READING.json`) and `roles/coordinator/beside.py --bank DDR1_R1_ZTILT --cand DB_PTC_R1`.
9. **Robustness readings of the registered releases:** `lanes/D2R_REG.json` (`ddr.claims d2r`; plans written by
   `roles/coordinator/write_d2r_plan.py`).
10. **Summary and tables:**
    - `lanes/SUMMARY.json` (`ddr.claims summary`);
    - `roles/coordinator/beside.py --bank <bank> --cand <release>`;
    - `roles/coordinator/db_r2_trigger.py --reading <reading> --bank <bank>`;
    - `python3 -m ddr.report tables --extra-bank DDR1_R1 --extra-bank DDR1_DA_R1 --extra-bank DDR1_R1_BESIDE --extra-bank DDR1_DA_R1_BESIDE --extra-bank DDR1_R1_ZTILT --extra-bank DDR1_ROB_DB_R1`.
11. **Deployment and custody:**
    - `lanes/DEPLOY_V1.json` (written by `roles/coordinator/write_deploy_lane.py`);
    - reader custody: `lanes/READER_CUSTODY.json`, `lanes/READER_CUSTODY_R1B.json`;
    - `python3 -m ddr.custody backup <dest>`;
    - ledgers and receipts: `python3 -m ddr.ledger`, `python3 -m ddr.closeout`.

## Using a saved release (deployment)

`ddr.deploy` packages a configuration with its hash-pinned model files and code:
- `python3 -m ddr.deploy store --name DB/DB_PTC_R1 --version 0`
- `python3 -m ddr.deploy build --name DB/DB_PTC_R1 --version 0 --out DIR`
- `python3 -m ddr.deploy check --packages DIR ... --report OUT.json`: bitwise parity from the admitted DEV_REPORT raw fields, and
  refusal probes.

A package accepts exactly AGEP, WKHP, COW, SCHL, MAR and POBP per row. It returns, per recipient, the probability vector and the
argmax decision only. SEX, task targets and identifiers are refused. The ten packages built here are under the private
`deploy/` folder; their check is `deploy/DEPLOYMENT_CHECK.json`. They cover DA_STD, DB_PTC_R1, the twin DA_TWIN, and the bracketing
control each release is not separated from (DA_SPD_HI and DB_SBC_R1_HI), in both versions.
