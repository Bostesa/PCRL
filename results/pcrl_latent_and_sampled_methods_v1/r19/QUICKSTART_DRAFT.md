# R19 quickstart draft: commands verified in this role

These are R19 integration commands actually run in this worktree. They are **read-only or tiny local tests**; they do not fit a method, access protected labels or replace R00's campaign quickstart.

From `results/pcrl_latent_and_sampled_methods_v1/r19/`:

```sh
python3 -m unittest -v test_queue_unit_status.py
python3 queue_unit_status.py
python3 -m json.tool ROLE_AND_RESULT_TRACKER.json
```

The targeted queue-helper test passed 2/2. A read-only live export produced a CSV with the exact eight public columns, including finished receipt hashes; no `/Users/`, `/private/` or `.private/` path marker appeared in its CSV output. R03 queue state can change immediately, so rerun the helper when proposing current `UNIT_STATUS` rows. A finished unit's queue receipt and state must agree, or the helper exits with no CSV.

For scientific source evidence, inspect the public R04 JSON receipts named in `D_TWO_SEED_POINT_NOTE.md`. R19 has not run R00's D fit, R02 attack, R04 replay, R18 real deployment or A-channel commands and does not list them as verified invocations. R00 should replace this draft with only commands backed by final queue/deployment receipts.
