# R5 worker formats

All numeric entry points require the new authenticated shared scheduler, stage `attacks`. R00 supplies current source hashes, immutable seals and queue specs. Config itself and every pin checked by `queue_config` must appear in the scheduler spec's `sources`. Private output directories must be fresh and contain `.private` in the resolved path. All JSON outputs are exclusive writes. Reader model/scaler and learned per-person archives remain private.

## New continuous reader fitting

Command: `python -m task_conditioned_representation.attack_worker --phase fit --spec PRIVATE_CONFIG`.

Config keys: `unit_id`, `arm_id`, `seed`, `manifest`, `manifest_sha256`, `checkpoint`, `checkpoint_sha256`, `science_lock`, `science_lock_sha256`, `attack_lock`, `attack_lock_sha256`, `private_output`, `public_output`.

Release lock: `status=RELEASES_AND_ENDPOINTS_FROZEN`, with `instances` containing `{arm_id,seed,checkpoint_sha256}` for every requested new arm/seed. Both source seeds are 0 or 1. The current source manifest excludes INNER SEX. The actual checkpoint determines both own features and predictions.

`build_attack_units --index PRIVATE_INDEX --output PRIVATE_PREPARED_INDEX` is a metadata-only builder. Its index has `manifest`, `science_lock`, `attack_lock` pins and `instances` records with `arm_id,seed,checkpoint,checkpoint_sha256`, optional `unit_id`, optional `deps` and optional current `amendment_sha256`. It writes configs/specs under the R5 private namespace, stages them as attacks, reserves 900 CPU seconds and caps wall time at 1200 seconds and RSS at 2 GiB per bank. R00 may prospectively lower these bounded reservations. It executes no queue unit.

## Original point utility

Command: `python -m task_conditioned_representation.utility_worker --spec PRIVATE_CONFIG`.

Config keys: `unit_id`, `manifest`, `manifest_sha256`, `nomination_lock`, `nomination_lock_sha256`, `instances`, `private_output`, `public_output`. Instance records have `arm_id,seed,checkpoint,checkpoint_sha256`. Global nomination lock must be `FROZEN`, contain nonempty `nominees`, and contain every requested checkpoint hash. Verify all hashes before any INNER task label is opened. Include both seeds for every arm ID.

Outputs: private `UTILITY_INPUTS.npz` with `row_id,unit,income,occupation`, reference probabilities `s0__U0__p1`, `s0__Ucal__p1` etc, and arm probabilities `s0__JOINT_CONDITIONAL__p1` etc; public original 24-clause utility eligibility for each family. To distinguish a preselected fallback from a nominee, use a distinct utility arm ID.

## Frozen true-label interpretation reference

Command: `python -m task_conditioned_representation.oracle_worker --spec PRIVATE_CONFIG`.

Config has unit/manifest/nomination lock/private/public keys as above. It trains Laplace1 SEX frequencies on all admitted FIT labels, then writes a private frozen `INNER_LABEL_ORACLE.npz` with `row_id,unit,income,occupation,r1,r2,pair`; these oracle probabilities require true labels and never enter primary readers. Add its `FROZEN_COMPLETE.json` pin to the global freeze and pass its score pin to reporting as `oracle_scores`.

## Protected-label reporting

Command: `python -m task_conditioned_representation.attack_worker --phase report --spec PRIVATE_CONFIG`.

Global lock keys: `status=ALL_READERS_FROZEN`, `K_primary=51`, `K_secondary=33`, `inner_sex_sha256`, and `reader_freezes` pins for every final bank. Every freeze must say `FROZEN_COMPLETE`. Config keys: `unit_id,arm_id,seed,point_eligible,reporting_lock,reporting_lock_sha256`, pins `inner_sex,freeze,scores,release`, `public_output`, optional `oracle_scores`, optional `FIT_surrogate_diagnostic`. The latter is the already frozen FIT-selected diagnostic, displayed beside fresh recovery without selecting by it. The reporting SEX archive must be owner-only and contain exactly `row_id,unit,sex`, 2235 rows / 2234 groups. `scores` pins `INNER_SCORES.npz`, `release` pins `INNER_RELEASE.npz`, and both hashes must match the corresponding freeze receipt.

## Historical separate score contract

Command: `python -m task_conditioned_representation.legacy_attack_worker --phase fit|report --spec PRIVATE_CONFIG`.

Fitting config has the continuous unit/manifest/science/attack/private/public fields, and pins `bank,att_law,inner_law,public_prior,law_origin_receipt`. Attack lock has `legacy_instances` records `{arm_id,seed}`. Laws contain exactly `row_id,unit,code_r1,code_r2,mass`. Public prior contains only `pair`. R5's private `LEGACY_ADMITTED_METADATA.json` supplies four prospective P0/S12 source pins from the closed previous campaign, without opening or hashing those NPZ files during preparation. S12 maps the old `S12_SELECTED` identity.

Reporting config has `unit_id,arm_id,seed,reporting_lock,reporting_lock_sha256,public_output`, current manifest path/hash and pins `bank,inner_law,freeze,score_tables,inner_sex`. These readers use the complete legacy wire: code IDs, probabilities, decisions and literal metadata. The contract has different capacity from new feature and new output-only services and stays separately marked in every result.

## Paired bootstrap inference

Command: `python -m task_conditioned_representation.inference_worker --spec PRIVATE_CONFIG`.

Config keys: unit/reporting lock fields as above, pins `endpoint_registry,utility_inputs,inner_sex`, `instances`, `private_output`, `public_output`. Registry structure: `K={primary:51,utility:240,secondary:33}` and `families={primary:[51 entries],utility:[240 entries],secondary:[33 entries]}`. Every slot remains present when dependencies are missing. An entry has unique `id`, `kind=utility|privacy`, `arm`, and either `cap` or `target`. Utility entries also have `seed,task=income|occupation,reference=U0|Ucal,metric=accuracy_loss|nll_excess|brier_excess`. Privacy entries also have `comparator,metric=pair_benefit|r1_worsening|r2_worsening`. Pair benefit is comparator minus JOINT; worsening is JOINT minus comparator. Primary family selects full wire, secondary selects new output-only or the separately identified complete legacy wire.

New inference instance keys: `arm_id,seed,point_eligible,is_nominee_or_exact_alias`, pins `release,scores`, optional `contract=new_continuous`. Eligibility and identity require both source seeds. A different fallback leaves primary privacy slots missing; an ineligible identity nominee stays diagnostic. Legacy instances have `contract=legacy_complete_score_wire`, `arm_id,seed,point_eligible`, pins `bank,inner_law,score_tables`.

The same `default_rng(2026101208).integers(0,2234,size=(2048,2234))` stream becomes group multiplicities in ascending unique-group order; duplicates share counts and all arms/seeds share the stream. Each seed's envelope is recomputed over fixed oriented readers inside each draw. Primary differences average paired seed quantities. Absolute AUC intervals are separately descriptive per seed and paired seed mean, with K=1; every primary conjunction endpoint uses K=51. Fixed 24 utility clauses per arm and all 240 utility slots use K=240. Secondary differences use K=33. Intervals use bootstrap SD with ddof1 times Bonferroni t, df2233, approximate conditional-development coverage. No kernel, head or reader is refitted here.

## Restore and independent replay

Model joblib files contain `{model,scaler}`. Registered scalar floor is float64 machine epsilon; every coordinate is retained. Replay matrix is `feature_tables(r1,r2)[view][reader_kind(name)]`; predict SEX class1 then apply `validation[view][name].orientation` from the freeze. A negative orientation maps p to 1-p. R6 uses separate scientific code and the same shared definitions, not imports from these scientific modules. R7 may use these deployment helpers for package restoration. Include their code and environment pins in private custody.
