# Reproduce and inspect the local development package

Run from the isolated `research/pcrl-utility-reference-repair-v1` worktree with the repository's existing `.venv` Python. These commands were used in this study; the complete private numerical replay also needs the admitted role-only exports, saved model artifacts and the shared flock slot documented in the archived stage seals.

```sh
git status --short
/Users/nathansamson/PCRL/.venv/bin/python -m pytest -q results/pcrl_utility_reference_repair_v1/r3/test_privacy_replay_worker.py results/pcrl_utility_reference_repair_v1/r3/test_build_privacy_replay_retry_lock.py results/pcrl_utility_reference_repair_v1/r2/test_build_privacy_aggregate.py results/pcrl_utility_reference_repair_v1/r1/test_build_old_f_replay_lock.py
```

The actual sealed stage builders were run with `--repo` pointing to this worktree, `--python` pointing to the existing `.venv/bin/python`, and `--slot` pointing to the single shared portfolio `slot0.lock`. The run order was reference parity, task-only recipe choice, full-input refit/gate, two lead masks, remaining fixed mask bank, pair selection, release/raw union, local ATT readers, coalition ATT readers, INNER SEX scoring, independent privacy replay (failed q21 then corrected q22), optional old-F diagnostic and independent old-F replay. Each queue `QUEUE.json` and immutable unit spec is under `.private/`; `RESOURCE_RECEIPTS.json` gives its SHA-256 and unit totals. The final public `CAMPAIGN_LOCK.json`, `LOCAL_SEAL_RECEIPT.json` and `r3/PREFIT_REVIEW.json` preserve the terminal q24 seal; prior seals are archived by stage in `.private/stage_seals/` and their commits remain in Git history.

For a lightweight public audit, read `START_HERE.md`, `INDEPENDENT_VERIFICATION.json`, `FULL_INPUT_UTILITY_INTERVALS.csv`, `MATCHED_PRIVACY_RESULTS.csv`, `MATCHED_PRIVACY_CONTROL_SELECTION.json`, `RESEARCH_DECISION.md` and `HANDOFF.json`. Public artifacts contain aggregates/hashes only. Do not move private NPZ/checkpoint files into a public branch.
