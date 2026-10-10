# D follow-up A2 and controls queue handoff

This is a public prefit specification. It does not register an amendment,
enqueue a unit, or read role files. The original campaign seal remains
immutable. R07 completed A1 and the separate seed-1 T/D-K worker; A2 binds
controls for both source seeds.

## Fixed identities already available

| Item | SHA-256 |
| --- | --- |
| Original `CAMPAIGN_LOCK.json` | `a75db45770d2ecb1385cf0bbbcbfec86a217b244ebf898ce61acd569609f5a48` |
| Role-only admission manifest bytes | `605bcdc9111f8d228a8526a649aa5cdec34c3f932d798d65e0cdf9672d86c45c` |
| Inherited `common_eval.py` | `fd5016ee0c041ffc18fc8d02fc31f57eb270bdc9977298efd9958f43cff5851e` |
| Source U seed 0 model | `5061a0074af74d21b53aca80b96d8b83b85a3e43cdb9e6758b2551be9045b9a7` |
| Source U seed 1 model | `f52bef54ef77faf51cb6f55273e54f75ea094c4379bbf73b57a3941ce2f7a74c` |
| Seed-0 first-pair receipt `D_FIRST_S0_R1_RECEIPT.json` | `c3abf8a3f3287b53a551e2e26c854cd22387da53fffd680cd1a4546db67879b5` |
| Seed-0 owner-only `warm_full_state.npz` | `8327c44125b5c72add62bc3c5b31e4b85e0cdfcf9669671649ce797440ed2339` |
| Seed-1 first-pair receipt `D_FIRST_S1_RECEIPT.json` | `8ae5e6ce1c57075702a9ec3933414fd1c9ee3264663ef71c0a7e003d80a0b007` |
| Seed-1 owner-only `warm_full_state.npz` | `6abf48ad77e3071d8ea7256873df3f8f4e7c71c2159a25cf6f0bd2e452cb258a` |
| Controls worker `run_followup.py` | `d6957aa155b53c14a55859c256916d9cf3f0b787d0f95aae45b5e751fea3d942` |

The role manifest path passed to both queue units must be the predecessor
repair-worktree copy with its sibling admitted archives:
`/Users/nathansamson/PCRL/.worktrees/pcrl-utility-reference-repair-v1/results/pcrl_utility_reference_repair_v1/m1/DATA_ACCESS_MANIFEST.json`.
The first private directories are campaign `.private/d_first_s0` and
`.private/d_first_s1`.

## A2 amendment

Use schema `pcrl-d-followup-amendment-v1`, status
`LOCALLY_SEALED_D_FOLLOWUP`, and these exact fields:

```json
{
  "effective_seal_sha256": "a75db45770d2ecb1385cf0bbbcbfec86a217b244ebf898ce61acd569609f5a48",
  "controls_worker_sha256": "d6957aa155b53c14a55859c256916d9cf3f0b787d0f95aae45b5e751fea3d942",
  "first_receipt_sha256_by_seed": {
    "0": "c3abf8a3f3287b53a551e2e26c854cd22387da53fffd680cd1a4546db67879b5",
    "1": "8ae5e6ce1c57075702a9ec3933414fd1c9ee3264663ef71c0a7e003d80a0b007"
  },
  "role_manifest_sha256": "605bcdc9111f8d228a8526a649aa5cdec34c3f932d798d65e0cdf9672d86c45c",
  "schema": "pcrl-d-followup-amendment-v1",
  "source_seed_sha256_by_seed": {
    "0": "5061a0074af74d21b53aca80b96d8b83b85a3e43cdb9e6758b2551be9045b9a7",
    "1": "f52bef54ef77faf51cb6f55273e54f75ea094c4379bbf73b57a3941ce2f7a74c"
  },
  "status": "LOCALLY_SEALED_D_FOLLOWUP"
}
```

R00 dates and byte-seals A2. R03 registers it as affecting `d_followup` only
between jobs. Every controls unit spec then carries A2's exact SHA-256 in
`amendment_sha256`, and the worker also verifies that A2 is the latest R03
amendment for `d_followup` and is listed in its active unit spec.

## Two controls queue units

Create one `d_followup` unit for each seed, with different unit IDs,
`--seed`, source model paths, first receipt paths/hashes, first private paths,
private output paths, and public receipt paths. Use
`results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.run_followup`
as the Python module. Pass these required arguments to its `argv`:

`--seal`, `--seal-sha256`, `--unit-id`, `--role-manifest`,
`--role-manifest-sha256`, `--shared-dir`, `--common-eval-sha256`, `--model-pt`,
`--source-sha256`, `--first-receipt`, `--first-receipt-sha256`,
`--first-seal-sha256`, `--first-private-output`, `--private-output`,
`--public-receipt`, `--amendment`, `--amendment-sha256`, and `--seed`.
`--first-seal-sha256` is the original seal SHA for both seeds.

Seed 0 uses `D_FIRST_S0_R1_RECEIPT.json`, `.private/d_first_s0`, source model
`/Users/nathansamson/PCRL_eval_cache_private/hcal_v1/admitted/rel__s0__U/model.pt`,
new `.private/d_followup_s0`, and a new public `D_CONTROLS_S0_RECEIPT.json`.
Its queue dependency is completed unit `D_FIRST_T_DK_S0_R1`. Seed 1 uses the
new A1 first-pair receipt, `.private/d_first_s1`, source model
`/Users/nathansamson/PCRL_eval_cache_private/hcal_v1/admitted/rel__s1__U/model.pt`,
new `.private/d_followup_s1`, and `D_CONTROLS_S1_RECEIPT.json`; its dependency
is completed unit `D_FIRST_T_DK_S1`. No controls unit is enqueued
before R00's A2 registration and R03's milestone/budget gate.

Each spec's `sources` pins the original seal, A2, the predecessor role
manifest, inherited CAL code, source model, matching first receipt and warm
state, and every imported training source: `impl/fit.py`, `impl/model.py`,
`impl/service.py`, `impl/run_first_seed.py`, `kernel/objective.py`,
`controls/adversary.py`, `controls/fit_c.py`, `controls/fit_f.py`,
`controls/f_service.py`, `controls/local_sequential.py`,
`controls/unchanged.py`, `controls/run_followup.py`, `r01/frozen_u.py`,
`r01/role_store.py`, `r01/reference_replay.py`, `r03/compute_queue.py`,
`r1/fit_splits.py`, and their imported `hcal/calib.py`, `jcv/train.py`, and
`safe_data_guard.py`. The `outputs` list contains the public receipt,
owner-only private index, five model manifests, and ten ATT/INNER release
NPZ files. Use one thread for each numerical library and `PYTHONHASHSEED=0`.

The first-pair queue unit used a 3 GiB RSS cap, 3,600-second wall/CPU reserve,
and 200 MB projected output. R03 must set the follow-up cap from observed
first-pair timing and its remaining D/A budget before queueing the five-arm
controls fit. The worker makes no runtime cap assumption.
