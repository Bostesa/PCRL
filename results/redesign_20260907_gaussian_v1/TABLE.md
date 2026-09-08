# Gaussian conflict pilot

Final-test predictive R², mean ± sample SD across seeds. Negative scores are retained.

| Method | P1 task U | P2 task V | Worst individual prohibited | Combined S |
|---|---:|---:|---:|---:|
| no_erasure | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| shared_union | -0.000483 ± 0.000439 | -0.000361 ± 0.000537 | 0.000263 ± 0.002875 | -0.001933 ± 0.005278 |
| per_purpose_leace | 0.999634 ± 0.000158 | 0.999694 ± 0.000314 | 0.000038 ± 0.003287 | -0.002490 ± 0.005325 |
| pcrl_post_erase_lora | 0.999634 ± 0.000158 | 0.999694 ± 0.000314 | 0.000038 ± 0.003287 | -0.002490 ± 0.005325 |

Worst individual is the maximum of P1→V,S and P2→U,S within each seed. Combined U,V are authorized and recorded separately in metrics.json.
Empirical fitting/evaluation covariance matrices, ranks, and in-sample OLS geometry are recorded separately from these attacker-fit → test predictive scores.

PCRL here reuses the existing post-erasure LoRA and proxy-dual primitives with continuous squared-error tasks; it does not validate the full categorical V2 trainer. Linear post-erasure adaptation plus optimal affine heads is expected to tie ordinary per-purpose LEACE.
