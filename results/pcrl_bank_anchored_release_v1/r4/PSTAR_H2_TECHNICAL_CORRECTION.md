# P* H2 defense-aware view correction

The 38 H1 units remain immutable. Thirty-six completed; the two P* H1
units failed with `required defense-aware reader has no view` after
`ATTACK_FIT_NEW` SEX was opened and before a complete reader freeze or
`INNER_SELECTION` SEX access. Their private partial output and queue failure
records remain in custody. They supply no P* AUC.

Root cause: the inherited finite-reference adapter encodes the complete
public P* wire but sets no `public_derived` columns. `prepare_views` then
sets `x_defense_aware=None`, while the frozen family-A slate requires
`DA_MLP_64x32`. The same failure was documented for an earlier continuous
reference; this H1 P* attempt had not used that corrected adapter.

The H2 adapter preserves every emitted P* code, probability, decision and
metadata field and all 30 planned reader/control fits. It adds only public
deterministic columns: each recipient's `public_log_probability` equals
`log(max(emitted_probability,1e-12))`. The pair also includes
`pair_code_bit_products`, the row-major outer product of the two public
code-bit vectors. The prepared pair derived-column order is
`r1.public_log_probability`, `r2.public_log_probability`,
`pair_code_bit_products`. Neither protected SEX nor a private fine cell
is needed to build these columns. The stronger P* reader is a technical
retry of this development comparison, not an H1 alias or a new population.

`PSTAR_H2_SOURCE_PLAN.json` pins the exact two failed H1 units and new H2
IDs, the unchanged source/seed/draw/attacker identity, the H1 indexes/lock,
and H2 source code hashes. `reference_worker_h2.py` is a new module; the
old `reference_worker.py` remains unchanged so the 36 other H1 source pins
remain valid. R0 must register a dated attacks-stage H2 correction amendment
binding the plan before H2 receipt-only `prepare`. R5 audits the two new
worker specs before R0 seals an exact H2 stage lock. H2 `finalize` then
sets each queue spec's `amendment_sha256` to that registered lock SHA;
R5 checks `_admit_reason=None` before R0 enqueues. The two units use the
existing 300s/2GiB/128MiB per-unit bounds.

H2 builder commands, after the corresponding registered amendments:

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m results.pcrl_bank_anchored_release_v1.r4.pstar_h2_builder prepare \
  --config results/pcrl_bank_anchored_release_v1/r4/.private/PSTAR_H2_BUILD_CONFIG.json

/Users/nathansamson/PCRL/.venv/bin/python -m results.pcrl_bank_anchored_release_v1.r4.pstar_h2_builder finalize \
  --config results/pcrl_bank_anchored_release_v1/r4/.private/PSTAR_H2_BUILD_CONFIG.json \
  --stage-lock <registered-H2-lock-path> \
  --stage-lock-sha256 <registered-H2-lock-sha256>
```

The owner-only config schema is `pcrl-bank-r4-pstar-h2-build-config-v1`;
its exact keys are `schema`, `h1_config_pin`, `h1_attack_index_pin`,
`h1_queue_index_pin`, `h1_stage_lock_pin`, `h2_source_plan_pin`, and
`h2_correction_amendment_pin`. Each pin has absolute `path` and `sha256`.
