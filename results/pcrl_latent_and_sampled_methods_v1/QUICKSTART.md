# Quickstart and replay map

Run these from the campaign worktree. They are the public, data-free commands exercised during closeout; they do not refit models or open person-level labels:

```sh
cd results/pcrl_latent_and_sampled_methods_v1/r19
python3 -m unittest discover -q -p 'test_*.py'
python3 queue_unit_status.py > /tmp/pcrl_queue_unit_status.csv
python3 a_220_status.py --attempt-output /tmp/pcrl_a_physical_attempts.csv > /tmp/pcrl_a_220_status.csv
```

The first command passed 11 tests. The final two generate sanitized live receipts; their closeout outputs are copied into the shared unit ledger and its planned-A supplement after queue closure. They validate hashes and fail closed on missing or contradictory R03 receipts.

The real numerical jobs were launched through the single R03 queue with immutable owner-only specs. Their actual unit IDs, source hashes, exit codes, CPU charges and output hashes are in `UNIT_STATUS.csv` and the public R03 receipts. Representative entrypoints are `r18/run_real_d_acceptance.py` (one admitted D-K raw-schema input), `r18/run_restored_d_replay.py` (restored service), `r02/a_fixed_endpoint_retry_v3.py` (fixed A endpoint), and `r04/replay_a_fixed_endpoint.py` (independent endpoint). They require the role-only exports, private model package and registered R03 spec; an invocation without those pinned inputs is not a reproduction of this study.

`START_HERE.md` gives the result. `PROTOCOL.md`, `CAMPAIGN_LOCK.json`, the dated amendments and `SOURCE_AND_EXPOSURE.json` give the access and comparison definitions. `PRIMARY_ENDPOINTS.csv` carries public aggregate rows; private saved predictions, checkpoints, random coins and person labels are intentionally absent from Git. The D deploy adapter and its 83-column/schema checks are documented in `r18/ACCEPTANCE_PROTOCOL.md` and the hash-only `r18/D_DEPLOYMENT_CLOSEOUT_RECEIPT.json`. A protected sampled deployment is unavailable because no fitted policy was selected.
