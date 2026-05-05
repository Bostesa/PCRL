# V2 Validation Summary

V2 = frozen StandardEncoder backbone + per-purpose LoRA adapters,
HSIC + vCLUB independence penalty, VICReg anti-collapse,
proxy-Lagrangian dual variables on HSIC ≤ 0.05.

Compared against three reference points:

- **v1 paper** — adversarial PCRL with FiLM (mean over 3 seeds).
- **LEACE-on-raw** — concept-erasure baseline (threat exp 1).
- **Separate encoders** — one frozen StandardEncoder per purpose, no conditioning (threat exp 2).

| dataset | v2 pass (3 seeds) | v2 status | v1 paper | LEACE | separate | v2 task | v1 task |
|---|---|---|---|---|---|---|---|
| adult | 0.00±0.00/8 | COLLAPSED | 6.0/8 | 0/8 | 0/8 (OK) | 84.6%±0.2% | 76.3% |
| diabetes | (not yet run) | — | 5.0/6 | 0/6 | 5/6 (COLLAPSED) | — | — |
| hmda | 0.00±0.00/6 | COLLAPSED | 5.3/6 | 0/6 | 0/6 (OK) | 90.8%±0.2% | — |

## Per-dataset verdict

- **adult** — **COLLAPSED** — representation health failed at least one threshold (per_dim_std < 0.5 or effective_rank < 2.0). Pass count is degenerate, same failure mode as separate encoders on diabetes.
- **diabetes** — not yet run.
- **hmda** — **COLLAPSED** — representation health failed at least one threshold (per_dim_std < 0.5 or effective_rank < 2.0). Pass count is degenerate, same failure mode as separate encoders on diabetes.
