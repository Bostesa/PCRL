# Run and inspect the bank-anchored campaign

Run from the PCRL research worktree with its project virtual environment. The public package is `results/pcrl_bank_anchored_release_v1/`; owner-only source inputs, models, labels, fixed draws and wire arrays are held under role `.private` directories and are not in Git.

The actual execution commands and exact recipient NPZ contract are in [R1 quickstart](r1/QUICKSTART.md). B0 accepts the pinned 83-column finite input matrix. J additionally requires a separately specified owner-only draw NPZ for a single fixed release per person. Both commands validate source/seed/schema and emit only global ten-bit code, fixed table probability and decision for the requested recipient. The public manifest is [R1_EXECUTABLE_RELEASE_MANIFEST.json](r1/R1_EXECUTABLE_RELEASE_MANIFEST.json). A different person batch needs its own admitted input and frozen draw contract; this campaign does not test repeated-query protection.

Commands executed for aggregate results:

```sh
../../.venv/bin/python -m results.pcrl_bank_anchored_release_v1.r0.build_final_tables
../../.venv/bin/python -m pytest -q results/pcrl_bank_anchored_release_v1/bank_anchored/test_r2_laws.py results/pcrl_bank_anchored_release_v1/bank_anchored/test_r2_fit_core.py results/pcrl_bank_anchored_release_v1/bank_anchored/test_r2_materialize.py results/pcrl_bank_anchored_release_v1/bank_anchored/test_r2_sequential.py
```

The fit, utility, attack and replay jobs were launched through the single [compute queue](r0/compute_queue.py) from source-bound private specs. The exact queued source hashes, job IDs, CPU charges and outputs are in the owner-only queue state and public role receipts. [R2 fit closeout](r2/R2_FIT_CLOSEOUT.json), [R3 utility gate](r3/R3_ORIGINAL_POINT_INTERVAL_ARM_TABLE.csv), [R4 attack aggregate](r4/ATTACK_AGGREGATE_ALL_H2.json), and [R5 numerical closeout](r5/NUMERICAL_CLOSEOUT_V2.json) show the executed stages. Run the table builder only after the public R5 V2 selection and recovery receipts exist; it creates [PRIMARY_ENDPOINTS.csv](PRIMARY_ENDPOINTS.csv) and [HEADLINE_COMPARISONS.csv](HEADLINE_COMPARISONS.csv).

[RESTORE_INDEX_V2.json](r1/RESTORE_INDEX_V2.json) records the last completed same-device byte restore. The fixed-source numerical successor and off-device custody statuses must be read from the latest R1 restore index and [HANDOFF.json](HANDOFF.json), once finalized. A local copy is not an off-device backup.
