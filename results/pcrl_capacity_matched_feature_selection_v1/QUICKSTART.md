# Quickstart

Use the isolated branch `research/pcrl-capacity-matched-feature-selection-v1` and package `results/pcrl_capacity_matched_feature_selection_v1/`. Python commands below were run from the worktree with the pinned environment and role-only exports available locally. They are read-only or data-free unless a queue command is explicitly invoked.

```sh
git rev-parse HEAD
python3 -m json.tool results/pcrl_capacity_matched_feature_selection_v1/CAMPAIGN_LOCK.json >/dev/null
python3 -m json.tool results/pcrl_capacity_matched_feature_selection_v1/INDEPENDENT_VERIFICATION.json >/dev/null
../../.venv/bin/python -m pytest -q results/pcrl_capacity_matched_feature_selection_v1/m5/test_replay_task_no_go.py
```

The actual scientific work used the sealed `m0/dispatch.py` and `m0/queue.py` with private registered unit specifications `M2-FIT-S0` and `M2-FIT-S1`. After the separate verification lock, the same shared queue ran `M5-TASK-NOGO-REPLAY-S0` and `M5-TASK-NOGO-REPLAY-S1`. Reexecuting those units requires the private role-only export, original seal, unused output identity and available shared heavy slot; do not replace existing frozen outputs in place. The exact specs, receipts, stdout/stderr hashes and saved head packages are in the ignored `.private` study archive and its versioned backup. The public [unit registry](UNIT_REGISTRY.json) and [resource receipts](RESOURCE_RECEIPTS.json) identify all units and attempts. The short verifier test above passed 3/3 using the repository virtual environment; the system `python3` lacks pytest.

To inspect results without private rows, read [FIT_SCREEN_SUMMARY.md](m2/FIT_SCREEN_SUMMARY.md), [FIRST_REAL_COMPARISON.md](FIRST_REAL_COMPARISON.md), [ALL_ARMS.csv](ALL_ARMS.csv), and [INDEPENDENT_VERIFICATION.json](INDEPENDENT_VERIFICATION.json). The present `CAMPAIGN_LOCK.json` permits verification only. The frozen science lock is archived in `amendments/CAMPAIGN_LOCK_SCIENCE_NOGO.json`; neither lock authorizes a new design under an old identity.
