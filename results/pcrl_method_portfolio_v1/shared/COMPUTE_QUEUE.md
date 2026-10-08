# Portfolio compute queue

S3 owns this single queue. Every fit, attacker, calibration, preprocessing, forward pass, synthetic numerical check and replay enters it. Other agents send S3 an exact command, worktree, dependencies, output list, maximum wall time and maximum RSS estimate. No job should start directly in a team terminal.

The private runtime is `<PORTFOLIO_WORKTREE>/results/pcrl_method_portfolio_v1/shared/.private/compute`, ignored by Git. All team worktrees must pass that same absolute directory to `--root`. The queue uses two `flock` slots, immutable unit specifications, append-only receipts, atomic per-unit status, one-thread numerical environment variables, a 20 CPU-hour ledger and a 10 elapsed-hour deadline. A conservative CPU charge uses the greater of child CPU and wall time. This avoids undercounting descendants after a signal but can overcharge idle time.

The queue begins closed, with two external slots reserved. `lsof` can inventory some processes, but it cannot measure their CPU use. S3 records the external reservation and refreshes the occupancy check within 120 seconds of each admission. Uncooperative external jobs cannot be controlled by this queue. It never signals their PIDs.

## Commands

Run from the campaign worktree, using the same `<QUEUE_ROOT>` in every team:

```sh
<python> results/pcrl_method_portfolio_v1/shared/compute_queue.py --root <QUEUE_ROOT> status
<python> results/pcrl_method_portfolio_v1/shared/compute_queue.py --root <QUEUE_ROOT> enqueue <UNIT_SPEC.json>
<python> results/pcrl_method_portfolio_v1/shared/compute_queue.py --root <QUEUE_ROOT> run-once
```

S3 opens **prelock synthetic admission** only after verifying the fixed Phase I law and team test hashes, plus a fresh occupancy receipt. The S1 cumulative Phase I queue index preserves the original lock and dated amendments. A prelock unit declares `mode: "prelock_synthetic"`, `data_access: "NONE"`, `team`, `test_file`, `test_sha256`, at most 120 seconds wall and 1.5 GiB RSS. It cannot read Adult data. The code checks file hashes and the registered path suffix across isolated team worktrees; S3 separately reviews imports and runtime inputs.

```sh
<python> results/pcrl_method_portfolio_v1/shared/compute_queue.py --root <QUEUE_ROOT> gate \
  --occupancy-confirmed --external-heavy <0|1|2> \
  --phase1-lock-path <PHASE1_QUEUE_INDEX.json> --phase1-lock-sha256 <SHA256> \
  --evidence <OCCUPANCY_RECEIPT.json>
```

S3 opens **real-data admission** only after S2's safe-input guard receipt and a remote-verified pushed CAMPAIGN_LOCK, with fresh machine occupancy. Those evidence identities belong in the gate receipt. The old assessment remains embargoed regardless of gate state.

```sh
<python> results/pcrl_method_portfolio_v1/shared/compute_queue.py --root <QUEUE_ROOT> gate \
  --safe-inputs --lock-pushed --occupancy-confirmed --external-heavy <0|1|2> \
  --evidence <ADMISSION_RECEIPT.json>
```

At a ceiling, S3 stops admitting new units and allows the wrapper to interrupt only its own campaign process group. Failed or interrupted units retain logs and receipts; retries need a new identity or a dated amendment. The public `RESOURCE_RECEIPTS.json` summarizes capacity and budget without publishing private command paths or per-person artifacts.
