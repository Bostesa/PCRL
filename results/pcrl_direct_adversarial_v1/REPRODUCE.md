# REPRODUCE

Branch `research/pcrl-direct-adversarial-v1`, based on
`73903b7f28df68284285f0610a4036beb32b208f`
(`research/pcrl-invariant-baselines-v1`).

All stages are resumable and idempotent. A completed unit is reused by hash; a failed
unit is quarantined rather than overwritten. Re-running a completed stage is a no-op
that re-verifies the recorded hashes.

## Environment

* Python 3.14 with the repository's `.venv` (`torch 2.10.0`, `numpy 2.4.2`,
  `scikit-learn 1.8.0`, `concept_erasure`, `threadpoolctl`).
* One BLAS/OpenMP thread, `torch.set_num_threads(1)`. Every stage calls `limit_threads()`.
* Run from the worktree root with that interpreter:
  `/path/to/PCRL/.venv/bin/python -m experiments.pcrl_direct_adversarial_v1.<stage>`

## Read-only historical inputs

Ignored local artifacts of the completed studies are resolved read-only across, in
order: this worktree, the residual-spectral worktree
(`PCRL_NLR_FALLBACK`), the original checkout `/Users/nathansamson/PCRL`, and — added by
this study — the invariant-baselines worktree (`PCRL_DAX_INVARIANT_ROOT`, default
`/Users/nathansamson/PCRL-terminal-1-invariant`). Nothing historical is written,
switched, stashed, reset or rebased. Every resolved file is hashed into the stage's
`inputs` registry.

## Stages, in order

```bash
# 1. Enumerate the matrix and hash-lock the protocol (must precede any outcome)
python -m experiments.pcrl_direct_adversarial_v1.freeze
python -m experiments.pcrl_direct_adversarial_v1.freeze --verify

# 2. Fit the 114 neural interfaces (2 widths x 3 policies x 4 betas x 3 seeds,
#    plus no-protection, single-attacker and optimiser-repeat blocks)
python -m experiments.pcrl_direct_adversarial_v1.run_fit \
    --seeds 0 1 2 --blocks main no_protection single_attacker optimizer_repeat \
    --mapper-updates 600

# 3. The 12 new erasure controls on the fitted no-protection channels
python -m experiments.pcrl_direct_adversarial_v1.erasure --seeds 0 1 2

# 4. 2018 development audits, utility probes and scores (126 interfaces)
python -m experiments.pcrl_direct_adversarial_v1.run_dev_2018 --seeds 0 1 2

# 5. Baseline transport completion + the fixed 2017 panel
python -m experiments.pcrl_direct_adversarial_v1.transport --seeds 0 1 2 --phase fit
python -m experiments.pcrl_direct_adversarial_v1.transport --seeds 0 1 2 --phase score

# 6. Endpoints, uncertainty and decisions
python -m experiments.pcrl_direct_adversarial_v1.run_report --seeds 0 1 2
python -m experiments.pcrl_direct_adversarial_v1.report_2017 --seeds 0 1 2

# 7. Mechanism checks and validation
python -m experiments.pcrl_direct_adversarial_v1.mechanism --seeds 0 1 2
python -m experiments.pcrl_direct_adversarial_v1.validate --seeds 0 1 2

# 8. Documents and the artifact manifest
python -m experiments.pcrl_direct_adversarial_v1.write_docs
```

## What is not reproduced by these commands

* **Nothing historical.** The frozen `A0`/`J` channels, the 2018 `H`/`E`/`L025`/`L20`
  and `spectral_*` units, and the invariant study's own 2018 units are read, hashed and
  reused; they are never refitted or rescored.
* **The OptNet encoders are reconstructed, not reused** — the invariant study's
  production run did not persist them. The reconstruction runs at that study's own
  **recorded** budget (`optnet_complete.json:budget_steps`) and is admitted only if it
  rebuilds the stored 2018 release **bitwise on all seven pools**. The proofs are in
  `OPTNET_RECONSTRUCTION.json` and `BASELINE_TRANSPORT_COMPLETION.md`.
* **The erasure affine maps are likewise reconstructed** by the same closed form and
  admitted under the same bitwise proof.
* **2016 is not touched by any command here** and must not be.

## Determinism notes

* Every RNG draw is seeded from a recorded formula; the seed constants are in
  `MATRIX.json:training_config` and in the module-level formulas.
* Minibatch orders are generated once per seed and shared by every policy, beta,
  ablation and optimiser repeat, so matched arms see paired minibatches. The order hash
  is in `seed_N/fit_context.json:order_hash`.
* `torch` on CPU with one thread is deterministic for these operations. The 2017
  scoring path additionally routes every prediction through the transport study's
  `stable_predict`, which computes each array twice and requires agreement.
