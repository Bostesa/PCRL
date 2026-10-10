# Code utility deployment

Both commands accept a preprocessed NPZ with exactly `X` (83 permitted columns)
and `feature_names` in the pinned order. Each invocation serves one purpose:
`0` is income and `1` is occupation. The output NPZ has exactly `code`, `prob`,
and `decision`. Bank probabilities are indexed directly from the public code
table. Learned-service probabilities are computed from the code through its
fixed head and calibrator. Use the hashes in `DEPLOYMENT_MANIFEST.json` for the
chosen seed.

Frozen public bank (`W_Q` or `W_TSTAR`):

```sh
OMP_NUM_THREADS=1 python -m code_witness.bank_deploy \
  --bank W_Q --purpose 0 --seed 0 \
  --source-root <verified-hcal-root> \
  --source-sha256 <source-model-sha256> \
  --qpc-root <verified-qpc-root> \
  --public-table results/pcrl_code_utility_witness_v1/r1/W_BANK_PUBLIC_TABLES_S0.json \
  --public-table-sha256 <bank-public-table-sha256> \
  --schema <ordered-83-column-schema.json> \
  --schema-sha256 <feature-schema-sha256> \
  --input <permitted-preprocessed-input.npz> \
  --output <recipient-wire.npz>
```

Learned task-only service (`T0_B10` or `T0_B16`):

```sh
OMP_NUM_THREADS=1 python -m code_witness.deploy \
  --service-manifest <private-service-package>/manifest.json \
  --service-manifest-sha256 <service-manifest-sha256> \
  --source-model <verified-hcal-root>/admitted/rel__s0__U/model.pt \
  --source-model-sha256 <source-model-sha256> \
  --schema <ordered-83-column-schema.json> \
  --schema-sha256 <feature-schema-sha256> \
  --input <permitted-preprocessed-input.npz> \
  --output <recipient-wire.npz> \
  --bits 10 --purpose 0
```

The bank loader checks the historical source checkpoint, task heads, policy,
decoder and public table against their pinned identities. The learned loader
checks its private model package and source schema. Both reject extra or
reordered predictors and hidden intermediate-output flags. The full data-free
adapter test command run was:

```sh
../../.venv/bin/python -m pytest -q code_witness/test_model.py code_witness/test_bank.py code_witness/test_service.py code_witness/test_worker.py code_witness/test_deploy.py code_witness/test_bank_deploy.py
```

It passed 19 tests. The historical banks passed current-role point utility
caps. The learned ten-bit and sixteen-bit task-only arms failed the locked
two-seed utility gate, so their service packages are reproducible research
artifacts rather than eligible protected mechanisms. On an independent
T0_B16 seed-one replay, the original full INNER batch exactly reproduced the
saved wire. A 32-row subset reproduced code and decisions exactly; float32
head arithmetic changed probabilities by at most `2.3296e-08` (income) and
`9.3348e-08` (occupation). The learned service therefore has no bitwise
probability invariance claim across batch sizes. A dated verification
amendment requires exact hard bits and decisions, probability decoded from
the emitted bits, and a maximum absolute saved-wire probability gap at or
below `2e-7` for the fixed 32-row subset and singleton calls. This engineering
restore tolerance does not change any task utility or privacy threshold.
The independent T0_B16 seed-one replay passed: all 32 singleton calls had
exact bits and decisions and decoded probability from those bits; the largest
saved-wire gap was `9.9768e-08`. Wrong source, reordered columns and an extra
excluded column were refused. This covers the stated sample and seed; the
learned service remains utility-ineligible. The separate verification receipt
and privacy reporting live in the campaign closeout records.
