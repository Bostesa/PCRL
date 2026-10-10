# R4 attack-stage unit builder

`spec_builder.py` is a receipt-only prefit step. It reads pinned public
tables/gates and private **receipt JSON and file hashes** for R2 policies,
cached releases and historical references. It never loads a person-level
array or a label, enqueues a job, or fits an attacker.

Prepare an owner-only `.private/ATTACK_BUILD_CONFIG.json` with schema
`pcrl-bank-r4-spec-build-config-v2` and these fields:

- `campaign_seal_path`, `campaign_seal_sha256`: locked campaign.
- `role_manifest_path`, `role_manifest_sha256`: role-specific admission map.
- `table_by_seed`: keys `"0"`, `"1"`, each `{path,sha256}` for its public
  W_TSTAR/W_Q table JSON.
- `gate_by_arm`: keys `B0,T,D,P,L,J,S12,S21,W_Q`, each `{path,sha256}`.
  B0 and the seven channel arms use R3's write-once
  `{ARM}_TWO_SEED_ATTACK_GATE_V2.json`, whose point admission is the 24
  original INNER clauses only. The separate FIT proxy result remains visible
  in that gate; a FIT failure does not erase an INNER pass. W_Q uses its
  historical-original-INNER admission gate. `prior_combined_gate_by_arm`
  supplies exact `{path,sha256}` pins for the eight original R3 combined
  gates. The builder verifies the V2 gate's `supersedes_combined_gate_sha256`,
  four source receipt identities, counts and unchanged FIT/INNER facts before
  admitting a release. Missing outcomes stay explicit. If R2 search
  did not produce a policy or an expected R3 utility receipt is absent, use an
  R4 operational marker with schema `pcrl-bank-r4-incomplete-arm-gate-v1`.
  It has `arm_id`, `source_seal_sha256`, `status` (`INCOMPLETE_SEARCH` or
  `MISSING_UTILITY`), a nonempty `reason`,
  `all_two_seed_task_reference_point_caps_pass:false`,
  `source_receipts_by_seed:{"0":...,"1":...}` with each available seed's
  exact `{unit:{path,sha256},fit:{path,sha256}}` R2 receipts (or null), and
  `missing_r3_unit_ids` (empty for incomplete R2 search; named absent R3 FIT
  or INNER units for missing utility). At least one R2 receipt pair is
  required. The marker is not a utility pass and must be listed in R0's dated
  attack-stage amendment. It skips only that arm.
- `pstar_historical_gate`: R3 separate P* reference gate `{path,sha256}`.
  `pstar_gate`: independent R5 P* INNER replay `{path,sha256}`.
- `selected_r2_root`: R2 public result root. The builder finds exactly one
  source package under this root.
- `selected_unit_by_arm_seed`: exact `{path,sha256}` for each of the fourteen
  channel arm/seed R2 public successor unit receipts, keyed `T_s0` through
  `S21_s1`. For a skipped incomplete arm, its unused entries may be null.
  The builder never infers a winner by globbing old and successor attempts.
  For each eligible arm/seed, it follows the selected receipt's
  `private_relative_dir` entries into the immutable nested
  `.private/units/<unit_id>/{policy,ROLE_Rdraw}` package. No alias or copied
  per-person array is created. Each attacked INNER law SHA must equal the
  R3 utility receipt's evaluated INNER law SHA; FIT law, CSV, source unit and
  policy receipt are independently bound as well. The R2 law archive excludes
  sampled codes and realization metadata; the data-free R2 materializer test
  verifies its SHA is identical across draws 0, 1 and 2. The worker checks
  that invariant for every attacked draw before it opens ATT SEX.
- `witness_source_root`, `pstar_source_root`: roots of the immutable inherited
  witness and P* source results, with their own private releases.
- `private_root`: owner-only R4 `.private` directory;
  `public_root`: R4 public result directory; `queue_root`: global R0 private
  queue root; `python_executable`: absolute admitted virtual-environment
  entrypoint, preserved without symlink dereference. Finalization runs that
  exact argv to verify `scipy` and `sklearn` imports before writing queue specs.
- `max_wall_s`, `cpu_reserve_s`, `max_rss_gib`,
  `projected_output_bytes`: fixed queue ceilings. `bounds_pin` is the exact
  `{path,sha256}` of `ATTACK_QUEUE_BOUNDS.json`; these four values must equal
  that prefit receipt. The public attack index pins its SHA, and each queue
  source list includes it. The queue requires CPU reserve at least wall
  limit; set these before attack-stage lock.

After all gates and R2 packages exist, run:

```sh
python -m results.pcrl_bank_anchored_release_v1.r4.spec_builder prepare \
  --config results/pcrl_bank_anchored_release_v1/r4/.private/ATTACK_BUILD_CONFIG.json
```

`ATTACK_UNIT_INDEX.json` is public and hash-only. Its `stage_lock_units`
mapping has exactly the worker identity fields R0 must place in a dated
`pcrl-bank-attack-stage-lock-v1` amendment. R0 must register that lock in
the single global queue before the second phase:

```sh
python -m results.pcrl_bank_anchored_release_v1.r4.spec_builder finalize \
  --config results/pcrl_bank_anchored_release_v1/r4/.private/ATTACK_BUILD_CONFIG.json \
  --stage-lock <R0-attack-stage-lock-path> \
  --stage-lock-sha256 <R0-attack-stage-lock-sha256>
```

`ATTACK_QUEUE_INDEX.json` then lists immutable private queue specs and their
hashes. R0 may enqueue and run them through the global semaphore. A queue
unit checks the stage lock, exact source/code pins, two admitted role-only
releases and point gate before opening ATT SEX. It opens INNER SEX only after
all readers and predictions are frozen. The builder allocates one cached
realization for deterministic B0/D/W_Q/P* and three for T/P/L/J/S12/S21;
the ceiling is 44 units and 1,320 reader/control fits. The actual count is
the point-eligible subset, never a best-seed or best-draw selection.
