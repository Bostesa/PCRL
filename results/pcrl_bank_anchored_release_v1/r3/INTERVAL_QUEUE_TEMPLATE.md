# R3 fixed-family utility interval queue unit

Run one R3 interval unit after the chosen protected INNER utility units are
terminal and their private cost vectors have been retained. The worker is
`python3 -m bank_anchored.r3_interval_worker --config <private-config>`.
Register it in the shared queue under `verification` or `protection`, with a
dated R0 amendment naming that stage. The implementation computes exactly the
sealed 360 utility slots in the K=1,575 family. It does not read labels or
attack files, and missing arms remain null.

Use a fresh owner-only config file (`0600`) under `r3/.private`:

```json
{
  "schema": "pcrl-bank-r3-interval-job-v1",
  "unit_id": "R3_UTILITY_INTERVALS_R1",
  "stage": "verification",
  "seal": {"path": "<absolute CAMPAIGN_LOCK.json>", "sha256": "<locked SHA>"},
  "entries": [
    {
      "arm": "B0", "seed": 0,
      "vector": {"path": "<absolute private R3_B0_S0_INNER-R1_vectors.npz>", "sha256": "<hash>"},
      "receipt": {"path": "<absolute R3_B0_S0_INNER-R1_RECEIPT.json>", "sha256": "<hash>"}
    },
    {
      "arm": "B0", "seed": 1,
      "vector": {"path": "<absolute private R3_B0_S1_INNER-R1_vectors.npz>", "sha256": "<hash>"},
      "receipt": {"path": "<absolute R3_B0_S1_INNER-R1_RECEIPT.json>", "sha256": "<hash>"}
    }
  ],
  "outputs": {
    "results": "<absolute new public r3/R3_UTILITY_INTERVALS_R1.json>",
    "receipt": "<absolute new public r3/R3_UTILITY_INTERVALS_R1_RECEIPT.json>"
  }
}
```

`python3 -m bank_anchored.r3_make_interval_config` creates that config and
its queue spec from an owner-only source manifest with schema
`pcrl-bank-r3-interval-sources-v1`. Its ordered `entries` list contains
`arm`, `seed`, `receipt_path` and `vector_path` for each completed INNER unit.
Pass `--manifest`, `--seal`, `--amendment`, `--unit-id`, `--stage` and an
absolute `--interpreter` that imports both NumPy and SciPy. The builder checks
that import, resolves and hashes the interpreter, and puts that exact binary
path into the queue spec.
The builder hashes vectors as opaque private files and checks each public
INNER receipt before registration. It writes a public hash-only unit index.

Add each completed protected INNER arm/seed to `entries` in the frozen arm
order `B0,T,D,P,L,J,S12,S21,W_Q,P_STAR`, with seed 0 before seed 1. Do not
insert synthetic identity entries for unrun arms. Every entry needs the exact
private vector SHA and public receipt SHA. The receipt must say
`INNER_SELECTION` and bind the same arm, seed, campaign seal and private vector.
Both B0 seeds are mandatory.

The queue spec pins the private config, campaign seal, all vector and receipt
files, and `bank_anchored/r3_interval_worker.py`, `r3_intervals.py`,
`r3_family.py`. Include the dated amendment file as a source and set
`amendment_sha256` to its digest. Use unique output paths and unit ID; never
overwrite a prior result. Reserve 120 seconds CPU/wall, 1 GiB RSS and 1 MiB
output initially, then let R0 revise before registration if queued smoke timing
requires it. Numerical threads are one. R5 replays the intervals independently
from these same byte-pinned vectors without importing R3's interval code.
