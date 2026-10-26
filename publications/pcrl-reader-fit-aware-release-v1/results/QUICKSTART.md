# Quickstart: reader-fit-aware release campaign (`pcrl_reader_fit_aware_release_v1`, RFA1)

Commands run from the repository root of the research worktree (branch `research/pcrl-reader-fit-aware-release-v1`). Every real-data
command goes through the shared queue (`python3 -m rfa.jobqueue run ...` or `python3 -m rfa.lane LANE.json`), which sets one numerical
thread per job, logs CPU and refuses work outside the time window. Private arrays and models live under
`.private/pcrl_reader_fit_aware_release_v1/` and are never published. Interpreters: `python3` (evaluation, claims, deployment) and the
torch venv `.venv/bin/python` at the root of the main checkout (fits and the reader grid).

## 1. Read the results (label-free)

```
python3 scripts/rfa1_tables.py RFA_v1:RFA_v1_final:RFA_v1_D2R RFA_R1:RFA_R1_final:RFA_R1_D2R   # ENDPOINTS.json, TABLES.md
python3 scripts/rfa1_summary.py                                     # claims/CAMPAIGN_SUMMARY.json (rules' campaign summary)
python3 scripts/rfa1_per_recipe.py                                  # claims/PER_RECIPE_PAIR_MEANS_corrected.json (R1 card prediction)
python3 -m rfa.rfclaims run --plan results/pcrl_reader_fit_aware_release_v1/claims/PLAN_RFA_v1.json \
    --bank RFA1_MAIN --lc-bank RFA1_LC --d2r-dir results/pcrl_reader_fit_aware_release_v1/claims/RFA_v1_D2R \
    --out /tmp/rfa1_recheck      # re-derive the governing v1 record (needs the private grid outputs)
```

## 2. Selftests (invented data only)

```
.venv/bin/python -m rfa.fam.rfit selftest       # v1 objective: complete-derivative finite differences, envelope identity, guards
.venv/bin/python -m rfa.fam.rfit_r1 selftest    # R1 objective (softbin family): the same checks; reproduces v1 with v1's families
.venv/bin/python -m rfa.fam.fit selftest_stage  # the fit stage end to end on invented labels
python3 -m rfa.test_rules                       # the frozen decision rules (statistical reviewer)
.venv/bin/python -m rfa.devreaders selftest     # the reader grid on invented data
```

## 3. Use a deployment package (six raw fields in, each recipient's probability vector and decision out)

```
python3 -c "
from rfa import deploy as D
rows = [{'AGEP': '41', 'WKHP': '40', 'COW': '1', 'SCHL': '21', 'MAR': '1', 'POBP': '012'}]   # a synthetic row in the admitted coding
print(D.release('.private/pcrl_reader_fit_aware_release_v1/deploy/RF_FULL_V0', rows))
"
python3 -m rfa.deploy check --packages .private/pcrl_reader_fit_aware_release_v1/deploy/RF_FULL_V0 --report /tmp/check.json
```

A package refuses SEX, task targets, identifiers, extra or missing fields and any output other than the probability vector and the
argmax decision. It binds every code module and model file its deployment function reads; a changed module makes it stale (rebuild
with `python3 -m rfa.deploy store --name FAM/<config> --version v`, then `build`). The release is deterministic: per-request keys are
refused.

## 4. Re-check custody

```
python3 -m rfa.custody backup <parent folder>                                  # same-device backup + restore check from the backup
python3 -m rfa.custody offdevice <backup folder> <existing folder on an attached drive>   # never mounts or forces anything
```

A check right after copying to a still-mounted drive is WARM. A COLD check needs an ordinary eject and reconnect first.

## 5. Re-run a stage (only inside the queue; the campaign's queue is closed at closeout)

Lanes are in `results/pcrl_reader_fit_aware_release_v1/lanes/` (FIT_*, FIT_R1_*, EVAL_*, GRID_*, LC_*, CLAIMS_*, D2R_*, DEPLOY_*,
READER_CUSTODY_*). A closed queue refuses every new unit; a re-run needs a new campaign queue.
