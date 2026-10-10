# T0 point-gate queue inputs

Run one separate R2 gate unit after **both** T0 fit seeds finish at a given
capacity. The gate does no fitting or calibration. It restores each saved
hard-code service, checks its public output against the actual code bits,
then computes 24 point clauses on admitted INNER rows. Ten-bit failure is
the locked condition for trying sixteen bits; no T0 privacy fit starts until
a capacity passes all 24 point clauses.

CLI through R0 queue, stage `witness`:

```text
<repository-root-venv>/bin/python -m results.pcrl_code_utility_witness_v1.r2.t0_utility_gate_worker --config <owner-only-config>
```

Private config schema `pcrl-r2-t0-utility-gate-config-v1`:

- `unit_id`: same exact queue unit ID;
- `bits`: integer 10 or 16;
- `seal`, `role_manifest`: each `{path,sha256}`;
- `seeds`: keys `"0"` and `"1"`, each containing a public R1 fit `receipt`
  `{path,sha256}` and its matching `private_root` directory;
- `output_csv`: a **new** file in R2 namespace such as
  `TASK_ONLY_UTILITY_B10.csv`;
- `output_receipt`: a **new** file in R2 namespace such as
  `T0_B10_UTILITY_GATE_RECEIPT.json`.

Put absolute paths only in the owner-only config/spec, not in a public
result. The config file and parent directory must be owner-only. The queue
spec must register exact byte SHA sources for config, effective seal, role
manifest, this worker, `r2/loss_metrics.py`, four admitted INNER archives
(`income`, `occupation`, `references_s0`, `references_s1`), two R1 public fit
receipts and, for each seed, its private `service__T0_B<bits>__s<seed>`
manifest and weights plus `release__T0_B<bits>__s<seed>__INNER_SELECTION.npz`.
The receipt supplies exact package/release hashes and basename; the role
manifest supplies exact archive hashes and paths.

The worker checks the active queue seal, R1 fit receipt queue seal,
per-seed frozen source-model hashes, permitted predictor schema, role
manifest, class order, model/scaler package hashes, per-array release hashes,
ordered row/group identity, and exact code-to-output reconstruction. It
publishes the 24 accuracy-loss/NLL-excess/Brier-excess rows against both U0
and Ucal, with an inclusive `<=` cap convention. The public aggregate
receipt also reports candidate accuracy/NLL/Brier, constant accuracy/gain,
class counts/recalls and occupied code states. `POINT_ALL_CAPS_PASS` is a
development **point** gate, not a confidence-interval or confirmation pass.
