# Quickstart: autonomous method-development campaign (PCRL, Florida ACS 2019 development roles)

Everything here runs from the worktree root, with the project virtual environment:

```
PY=/Users/<you>/PCRL/.venv/bin/python
```

CatBoost is needed only to refit family C's teachers, and must be the pinned 1.2.8. Every numerical step runs through the shared queue. The queue allows at most 2 heavy jobs at once and forces one thread per job:

```
python3 -m amd.jobqueue run --unit <ID> --stage <stage> --rss-gib 2 -- $PY -m <module> ...
python3 -m amd.jobqueue summary
```

Person-level arrays, fitted models and reader artifacts live under `.private/pcrl_autonomous_method_development_v1/`. That directory is never published, and the commands below need it.

## 1. The finalist release (what it computes)

`B/B3CTL_UNCOUPLED_MIX50_C6A`. For one person, from the six permitted raw fields AGEP, WKHP, COW, SCHL, MAR and POBP:

1. Encode the fields with the replication's frozen 145-column encoder.
2. **Occupation output.** Route the person through the frozen CatBoost teacher trees (T_TREE). Add the stored JOINT_TREE E20 leaf-score corrections from the protected-tree replication. Take the softmax over the 5 occupation groups and release the vector with its argmax decision. This is bitwise the occupation output of the frozen reference H_LJ.
3. **Income output.**
   - The income statistic is d = 0.5 × (LOCAL_TREE income log-odds) + 0.5 × (JOINT_TREE income log-odds), using the stored E20 corrections on the income teacher.
   - c is one of 6 decision-aligned levels of d, with label-free MODEL_FIT quantile thresholds and 3 levels on each side of 0.
   - The income probability is a per-level value fitted on CALIBRATION income labels, using an L2 logistic regression on one-hot c. It is released with its argmax decision.
   - The income output therefore takes 6 distinct values. It does not depend on the occupation output.
4. SEX is never an input. Training-time SEX labels entered only through the stored replication corrections. The CALIBRATION fit uses income labels only.

To refit the release from the admitted pieces (deterministic) and score it on MODEL_SELECT (each scoring is logged):

```
$PY -m amd.fam_b fit --seed 0 --config <new id> --base JOINT_TREE --phi none --m 6 --coarse-kind aligned --stat bins --stat-source MIX --alpha 0.5
$PY -m amd.devrun score --family B --version B3 --config <new id> --seed 0 --counts 1 --hypothesis "..." --change "..." --training "..." --selection "..."
```

## 2. Deployment packages (raw fields to released vectors)

```
$PY -m amd.deploy --out-root .private/pcrl_autonomous_method_development_v1/deploy --report results/pcrl_autonomous_method_development_v1/DEPLOYMENT_RESULTS.json \
    --package FINALIST:B:B3CTL_UNCOUPLED_MIX50_C6A:family --package TASKONLY:B:B3CTL_TASKONLY_TR_C6A:family
```

The run also copies and reruns the frozen references NEURAL_RAW (seed 0) and H_LJ (both seeds). There is no NEURAL_RAW seed-1 package; the source never built one.

Entry points:
- API: `amd.deploy.release(pkg_dir, rows)`. Each row must have exactly AGEP, WKHP, COW, SCHL, MAR and POBP, in that order.
- CSV: `amd.deploy.release_csv(pkg_dir, path)`.

Both return only `probability` and `decision` per recipient. Each package is tested to refuse:
- extra columns;
- reordered columns;
- a missing column;
- a SEX column, alone or replacing a field;
- extra or reordered API keys;
- requests for any internal quantity: leaf ids, raw scores, teacher probabilities, propensities, SEX-specific probabilities or theta.

## 3. Development harness (MODEL_SELECT only)

- `amd.evaluate.measure` computes, per seed:
  - the 12 original clauses against U0 / Ucal;
  - both within-SEX occupation Brier guards;
  - household-split reader proxies on the pair, income and occupation views.
- `amd.baselines` measures the frozen references with the same code.
- `amd.promote --candidates all` applies PROTOCOL.md §6 to every fitted configuration and writes PROMOTION.json.
- The full record of trials is in `EXPERIMENT_LEDGER.csv` and `COMPARISON_TABLE.csv`.

## 4. Final stage (frozen; needs the committed locks)

```
$PY -m amd.final_bank                      # bank from FINALIST_LOCK (8 releases per seed)
$PY -m amd.readers --seed 0 --out .private/.../runs/READERS_0     # fresh readers: ATTACK_FIT fit, ATTACK_VALIDATION selection
$PY -m amd.dryrun                          # the report on INVENTED labels
python3 -m amd.locks reporting             # REPORTING_LOCK (then commit and push the receipt)
$PY -m amd.report --out results/.../report # the ONE logged coordinator DEV_REPORT evaluation (refuses a second run)
```

The DEV_REPORT truth is reachable only through `amd.common.dev_report_truth`. Each of the two registered events can be used once, after the committed reporting lock: the coordinator's evaluation and the independent verifier's replay.

## 5. Custody

```
AMD_PYTHON=$PY AMD_PYLIB=<private CatBoost lib> bash amd/custody.sh backup <dest parent>      # bundle + code + private tar + restore check
AMD_PYTHON=$PY AMD_PYLIB=<private CatBoost lib> bash amd/custody.sh offdevice <backup dir> <external drive dir>
```

A restore check redeploys every package from the backup alone and compares the result bitwise with the evaluated DEV_REPORT outputs, without reading labels.
