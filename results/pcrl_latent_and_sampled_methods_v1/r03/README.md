# Shared compute queue

All campaign heavy numerical work runs through [compute_queue.py](compute_queue.py) in this worktree. The script fixes one shared private state root at `r03/.private/compute`; use this exact absolute script from every role or worktree. The queue starts closed. R00 alone opens it after the effective protocol seal and input admission are reviewed. Enqueueing records a plan and does not run it.

The budget epoch began **2026-10-10 00:42:38 UTC** and ends **2026-10-10 10:42:38 UTC**. The ledger ceiling is **72,000 CPU-seconds**. Admission protects 32,400 seconds for D and 10,800 for independent verification. A has an 18,000-second maximum. There is no automatic transfer of unused reservations. `d_first` jobs have first queue priority. A and optional jobs require both a D-first completion receipt and an independent verification completion receipt. R00 records those milestones only after checking them. D follow-up and verification retain their protected budget after the first package.

The process cap is two globally among participating roles. This sandbox denies `ps`, `pgrep`, `top`, and psutil's global PID inventory. Until R00 supplies a fresh, reliable external occupancy receipt, this queue allows **one** campaign worker. A 0/1/2 external-heavy receipt can permit 2/1/0 campaign workers for 120 seconds; then capacity returns to one. An unknown external count is not proof of a two-process machine-wide bound. Host observations are in the private `HOST_INVENTORY.json`; its snapshot at 00:48:48 UTC found 14 logical CPUs, 24 GiB physical RAM, about 4.24 GiB available RAM, and about 85.4 GiB free disk. The queue reserves at most 8 GiB aggregate RSS and keeps 1 GiB host available and 5 GiB free disk at admission.

## Commands

Set `Q` to the absolute path of `r03/compute_queue.py` and use the project's `.venv/bin/python`:

```sh
python "$Q" status
python "$Q" open <effective-seal-file> <sha256>       # R00 only
python "$Q" amend <dated-amendment-file> <sha256> <ISO-UTC-date> <stage[,stage...]> # R00 only
python "$Q" enqueue <exact-unit-spec.json>
python "$Q" run-once                                   # runs one eligible unit, then exits
python "$Q" mark d_first_complete <receipt.json>      # R00 only
python "$Q" mark verification_complete <receipt.json> # R00 only
python "$Q" occupancy <inventory-receipt.json> <0|1|2> # R00 only; external heavy count
python "$Q" cache-put <private-file>
python "$Q" charge <unique-id> <stage> <cpu-seconds> <receipt.json> # R00 ledger correction
python "$Q" close <reason>                             # R00; stops campaign workers at next poll
python "$Q" reconcile <unit-id>                        # only after wrapper, child, group and lock are gone
```

`open` stores the exact original seal path and SHA-256 and every admission rechecks its bytes. The original seal identity cannot be replaced in this budget epoch. `amend` appends a dated file path, hash and affected stages only between jobs. An affected queued unit must state the latest amendment hash in its spec as `amendment_sha256`; every admission rechecks all amendment bytes. Old seal and amendment entries remain in the private journal. `mark` stores immutable completion receipt hashes. `occupancy` stores an inventory receipt hash and time. `cache-put` copies an artifact under `cache/objects/<first-two-sha-characters>/<sha256>`, verifies it and makes it read only; it never mutates an existing object. Private unit logs and receipts live under `.private/compute/units/<unit-id>`.

## Unit contract

Example JSON, with paths filled from the effective R00 seal:

```json
{
  "unit_id": "D-FIRST-T-COMPRESS-S0",
  "stage": "d_first",
  "argv": ["/absolute/path/to/.venv/bin/python", "-m", "module.command", "--config", "/private/exact-config.json"],
  "cwd": "/absolute/path/to/campaign-worktree",
  "max_wall_s": 1800,
  "cpu_reserve_s": 1800,
  "max_rss_gib": 2.0,
  "projected_output_bytes": 100000000,
  "sources": [{"path": "/absolute/path/to/source.py", "sha256": "exact64characterhex"}],
  "deps": [],
  "outputs": ["/private/expected-output.json"],
  "env": {}
}
```

Stages are `admission`, `d_first`, `d_followup`, `verification`, `a`, and `optional`. `cpu_reserve_s` must cover `max_wall_s` under the one-thread contract. A unique unit ID fixes its full spec; a changed attempt needs a new ID and dated protocol amendment. Output hashes are written only after a successful exit. Source file hashes are checked at each admission. The command is an argv list, never a shell. All BLAS/OpenMP/Torch/NumExpr/Polars thread variables are forced to one, overriding inherited values. The wrapper holds a flock slot through its child process group, samples owned PID/PPID/start/CPU/RSS using `lsof` group discovery where global process enumeration is blocked, and records the limitations in its receipt. It stops its own group on wall, memory, CPU, signal or campaign closure; it never signals unrelated processes. A supervisor may have one active heavy child; concurrent heavy descendants need their own queue units and slots.

The final CPU charge is the greatest of child CPU, observed process-tree CPU and elapsed wall time. This is conservative for a one-thread job and covers short-lived descendants that sampling may miss. The ledger also accepts `charge` receipts for light or exceptional CPU use outside heavy units. Each job records PID, observed descendants, exit code, time, CPU, memory and output hashes. A crashed wrapper leaves its state and lock evidence; `reconcile` charges its full CPU reservation only after the slot, wrapper PID, child PID and process group are all gone.

## Current state and checks

At setup the seal is absent, the queue is empty, no science jobs have run, projected private output is zero, and campaign capacity is one. The private `SETUP_COST.json` records a conservative 60 CPU-second admission charge for scheduler setup and data-free checks. Seven isolated control tests pass for the seal gate, exclusive slot, milestone gate, child PID observation, resource receipt, immutable cache, amendment requirement, idempotent charge and orphan recovery. Run them with `python -m unittest -v test_queue.py` from `r03`.
