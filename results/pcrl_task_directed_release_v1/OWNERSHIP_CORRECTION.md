# Ownership correction: planned local coverage conditioning

Dated 2026-09-22, source review completed at 02:30 UTC. This addendum corrects a historical specification and the earlier search status. It changes no current fitted channel, audit, selection rule, or scientific source.

**Located: a planned-definition error in the replacement method. No offending executed ACS local-conditioner implementation was located.**

At full commit `e3415b94deb8d71c4870d4c392bb8ad4b464a847`, `results/pcrl_stochastic_replacement_overnight_v1/METHOD.md`, lines 56–62, the primary `C_A` definition clusters the nonredundant `H_A` probabilities into four cells and then says:

> crossed with public-coverage probability split at its `representation_fit` median → **2** bins, giving
> `4 × 2 = 8` cells. `C_AB`: that, crossed with `H_B` coarsened to **2** bins by its first coordinate's
> median → **16** cells.

The committed service ownership is explicit. `experiments/acs_coalition_training.py:26` assigns income/employment to A and public coverage to B. `experiments/run_acs_fixed_predictions.py:73–74` constructs and saves A from A's task predictions and B from `public_coverage`. Therefore public-coverage probability is not an A-local conditioning coordinate. The planned primary `C_A` includes a B-owned service quantity, contrary to the local release contract. This is an ownership error in the written partition definition, irrespective of its feature-derived or label-independent status.

The proper local partition is a function of `H_A` alone; coalition conditioning may additionally use `H_B`. A constraint conditioned on the unavailable B service is not the declared local incremental-information constraint. Conditional information is not monotone in added context, so this correction by itself does not establish the direction or size of any privacy effect. It also does not establish that a runtime encoder took B's input, that H was altered, or that any executed outcome changed.

## Planned versus executed

The prior run history supplied for this review records **zero ACS privacy-channel fits**. This bounded source-only check did not open execution artifacts or outcomes and does not independently re-adjudicate that count. The examined replacement source implements input construction, label-free codebooks, the representation screen, synthetic finite-channel solves, token utilities and bookkeeping. No implementation of the quoted ACS `C_A` coverage-crossing rule was located in that source subtree. The finding is therefore a correction to the **planned method**, not a claim that completed ACS Q maps used the wrong local conditioner.

The older stochastic extension registration at full commit `cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a`, `results/pcrl_stochastic_channel_v1/REGISTRATION.md:52`, correctly states `C_A = (H_A, Z_J)` and `C_AB = (H_A, Z_J, H_B)` for its different contract in which J remains released. That earlier definition is not the located error. At the replacement pin, `experiments/pcrl_direct_adversarial_v1/attackers.py:42–48` also correctly returns only `ha` for local A roles and concatenates `ha,hb` for coalition roles. Those executed-source functions must not be retrospectively labeled buggy on the strength of the planned-method error.

The current task-directed study already enforces the corrected ownership: `experiments/pcrl_task_directed_release_v1/encoding.py:150–161` fits and assigns local cells from `ha[:,[1,3]]` only, while only coalition cells use `hb[:,1]`; `tests/pcrl_task_directed_release_v1/test_encoding.py:36` checks local independence from B's service. No code change is needed for this finding.

## Correction to prior review wording

The earlier `ARTIFACT_DEFINITIONS.md` and `RELEASE_CONTRACT.md` statements that the offending operation had not been located should now be read as follows:

> The coverage-ownership error has been located in the replacement study's planned `C_A` specification at the pinned METHOD.md lines 56–62. An executed ACS implementation of that incorrect definition has not been located. The current study's local cells use only A-owned service predictions.

This addendum preserves the historical documents and records the more precise finding; it does not silently rewrite their chronology.

## Exact source references

All SHA-256 values below hash the complete committed file bytes, not an extracted excerpt.

| Full source commit | Path | Lines | SHA-256 | Relevance |
| --- | --- | --- | --- | --- |
| `e3415b94deb8d71c4870d4c392bb8ad4b464a847` | `results/pcrl_stochastic_replacement_overnight_v1/METHOD.md` | 56–62 | `a51265d1608ec89397d091d9fc4aa58cf42eaa30446ff9dca739908cb97f4ea0` | Offending planned local conditioning partition |
| `e3415b94deb8d71c4870d4c392bb8ad4b464a847` | `experiments/acs_coalition_training.py` | 26 | `c9e846cbed3c2f40c604f5dfb2630e35f3b827942bc3bd4d64bb6c42d15bc763` | A owns income/employment; B owns public coverage |
| `e3415b94deb8d71c4870d4c392bb8ad4b464a847` | `experiments/run_acs_fixed_predictions.py` | 73–74 | `1e33ba0cfdb97c8221b14634958a3a2d3d41512adb592300a74e38425c220bde` | Saved A/B service array construction |
| `e3415b94deb8d71c4870d4c392bb8ad4b464a847` | `experiments/pcrl_direct_adversarial_v1/attackers.py` | 42–48 | `0c9ca5055cb96568d10872c4028db44551764542ca516616bd34b87a918541d6` | Correct local/coalition service routing |
| `cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a` | `results/pcrl_stochastic_channel_v1/REGISTRATION.md` | 40–60 | `abc156a5db02cf4b4d04f6ab8974313f325e711da8bc8b4a4d9bd9d73d175bc2` | Earlier extension contract and correct C_A/C_AB ownership |
| `e3415b94deb8d71c4870d4c392bb8ad4b464a847` | `experiments/pcrl_stochastic_replacement_overnight_v1/replacement.py` | 1–23, 249–277 | `e509145ababcc24d8ee16fd917e276edfa6b2af79207cb804ef7ad9c77d2a98f` | Replacement code implements the representation screen, not the planned ACS privacy partition |

Search scope: the two named commits' stochastic-channel and stochastic-replacement experiment source subtrees, their registration/protocol/method/release-contract/correction documents, and the named service-ownership/routing source files. Search terms included `C_A`, `C_AB`, `partition`, `conditioning`, `coverage`, `H_B`, and `hb`; matched definitions and surrounding source were read. No dataset, model, per-person artifact, current-run outcome, or 2016/2017 data was opened. This is a bounded provenance finding, not a claim of exhaustive repository history coverage.
