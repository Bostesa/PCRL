# R19 quickstart draft: commands verified in this role

These are R19 integration commands actually run in this worktree. They are **read-only or tiny local tests**; they do not fit a method, access protected labels or replace R00's campaign quickstart.

From `results/pcrl_latent_and_sampled_methods_v1/r19/`:

```sh
python3 -m unittest -v test_queue_unit_status.py
python3 queue_unit_status.py
python3 -m json.tool ROLE_AND_RESULT_TRACKER.json
python3 derive_d_k567_proposal.py
python3 build_primary_endpoint_proposal.py
python3 -m unittest discover -q -p 'test_*.py'
python3 a_220_status.py --attempt-output A_FIXED_ATTACK_PHYSICAL_ATTEMPTS_PROPOSAL.csv > A_220_UNIT_STATUS_PROPOSAL.csv
```

The targeted queue-helper tests passed 3/3. A read-only live export produced a CSV with the exact eleven public columns, distinguishing unit and separate charge receipts; no `/Users/`, `/private/` or `.private/` path marker appeared in its CSV output. R03 queue state can change immediately, so rerun the helper when proposing current `UNIT_STATUS` rows. A finished, failed or superseded unit's receipt and state must agree, or the helper exits with no CSV.

The interval derivation command reads R04's hash-pinned public `D_567_INDEPENDENT_INTERVALS.json`, checks its 567-clause scope and two-seed point parity, then updates only the three R19 D proposal CSVs. The unified endpoint builder reads those three hash-pinned CSVs and R02's sealed D clause order, producing 665 D-only candidate rows. Neither command recomputes the bootstrap or an attack. For scientific source evidence, inspect the public R04 JSON receipts named in `D_TWO_SEED_POINT_NOTE.md` and `D_ENDPOINTS_PROPOSAL.md`. R19 has not run R00's D fit, R02 attack, R04 replay, R18 real deployment or A-channel commands and does not list them as verified invocations. R00 should replace this draft with only commands backed by final queue/deployment receipts.

The current R19 test discovery passed 11 tests. The A helper checks the sealed 220 logical IDs, all 16 FIT outcomes, 18 fixed attack cells and 12 corrected physical aliases against public hashes and R03 queue metadata. It writes a 220-row logical proposal and a separate 20-row physical attack attempt ledger. These are accounting artifacts, not A endpoint or independent replay evidence. R00 should reconcile the two ledgers without adding the same completed attack charge twice.
