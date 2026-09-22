# Historical published-score index

**Historical scores only. These are not the current-host matched audits and must not enter current claim selection or adjusted comparisons.**

All report content is pinned to full Git commit `e3415b94deb8d71c4870d4c392bb8ad4b464a847`. Exact report SHA-256, Git blob IDs, original source row numbers and published per-seed decimal values are retained in `HISTORICAL_SCORE_INDEX.json`. No model was loaded and no score was refitted.

## Original H/J table

Published fixed-prediction table, expanded catch-up360, historically reused2018 test-named pools. Sensitive columns below are **prior-relative recovery**, not H-relative increments. Lower CE and lower measured recovery are favorable.

| Historical system | Weighting | Residence CE | A/SEX prior gain | AB/SEX prior gain | A/RAC1P prior gain | AB/RAC1P prior gain |
| --- | --- | --- | --- | --- | --- | --- |
| H | unweighted | 0.529633030 | 0.007078766 | 0.011730995 | 0.018736132 | 0.035310257 |
| H | person_weighted | 0.505151206 | 0.007096184 | 0.011309031 | 0.018395203 | 0.034209756 |
| J | unweighted | 0.508104559 | 0.002359944 | 0.012840481 | 0.025879517 | 0.041004578 |
| J | person_weighted | 0.485634763 | 0.001851306 | 0.012049164 | 0.024575861 | 0.040158019 |

## Later historical baseline audit: expanded_catchup

Equal three-anchor means derived from the committed per-seed score report, test-named2018 pools, budget360. Sensitive columns are the source-reported **H-relative additional recovery**; they are not CMI. All five original task losses and all four primary attack losses, absolute recoveries and H-relative recoveries remain separately available in JSON.

| Historical system | Weighting | Residence CE | A/SEX H increment | AB/SEX H increment | A/RAC1P H increment | AB/RAC1P H increment |
| --- | --- | --- | --- | --- | --- | --- |
| H | unweighted | 0.529633030 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| H | person_weighted | 0.505151206 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| J | unweighted | 0.508104559 | -0.004718822 | 0.001109486 | 0.007143385 | 0.005694321 |
| J | person_weighted | 0.485634763 | -0.005244878 | 0.000740133 | 0.006180658 | 0.005948263 |
| leace_A0 | unweighted | 0.507614896 | 0.000757006 | -0.000894828 | 0.009011528 | 0.003489746 |
| leace_A0 | person_weighted | 0.484636520 | 0.000328741 | -0.000856044 | 0.009596501 | 0.003225559 |
| splince_A0 | unweighted | 0.512927872 | 0.000000000 | 0.000000000 | 0.003746302 | 0.002859525 |
| splince_A0 | person_weighted | 0.489755010 | 0.000000000 | 0.000000000 | 0.004864493 | 0.003496321 |
| optnet16_L1 | unweighted | 0.509775999 | 0.005658550 | 0.002858279 | 0.016489217 | 0.013376678 |
| optnet16_L1 | person_weighted | 0.485265369 | 0.004105606 | 0.001512182 | 0.013433231 | 0.012331786 |
| optnet16_L2 | unweighted | 0.513691549 | -0.001634328 | -0.001531927 | 0.003636734 | -0.000496842 |
| optnet16_L2 | person_weighted | 0.488850985 | -0.002486355 | -0.002732287 | 0.003971189 | -0.001374555 |
| optnet16_C1 | unweighted | 0.512614953 | 0.001369753 | -0.003309612 | 0.005970847 | 0.004792111 |
| optnet16_C1 | person_weighted | 0.488239142 | 0.001128926 | -0.003318481 | 0.006711236 | 0.004747890 |

## Later historical baseline audit: kernel_expanded_catchup

Equal three-anchor means derived from the committed per-seed score report, test-named2018 pools, budget360. Sensitive columns are the source-reported **H-relative additional recovery**; they are not CMI. All five original task losses and all four primary attack losses, absolute recoveries and H-relative recoveries remain separately available in JSON.

| Historical system | Weighting | Residence CE | A/SEX H increment | AB/SEX H increment | A/RAC1P H increment | AB/RAC1P H increment |
| --- | --- | --- | --- | --- | --- | --- |
| H | unweighted | 0.529633030 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| H | person_weighted | 0.505151206 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| J | unweighted | 0.508104559 | -0.004718822 | 0.001109486 | 0.007143385 | 0.005694321 |
| J | person_weighted | 0.485634763 | -0.005244878 | 0.000740133 | 0.006180658 | 0.005948263 |
| leace_A0 | unweighted | 0.507614896 | 0.000757006 | -0.000894828 | 0.009011528 | 0.003489746 |
| leace_A0 | person_weighted | 0.484636520 | 0.000328741 | -0.000856044 | 0.009596501 | 0.003225559 |
| splince_A0 | unweighted | 0.512927872 | 0.000000000 | 0.000000000 | 0.003746302 | 0.002859525 |
| splince_A0 | person_weighted | 0.489755010 | 0.000000000 | 0.000000000 | 0.004864493 | 0.003496321 |
| optnet16_L1 | unweighted | 0.509775999 | 0.005658550 | 0.002858279 | 0.016489217 | 0.013376678 |
| optnet16_L1 | person_weighted | 0.485265369 | 0.004105606 | 0.001512182 | 0.013433231 | 0.012331786 |
| optnet16_L2 | unweighted | 0.513691549 | -0.001634328 | -0.001531927 | 0.003636734 | -0.000496842 |
| optnet16_L2 | person_weighted | 0.488850985 | -0.002486355 | -0.002732287 | 0.003971189 | -0.001374555 |
| optnet16_C1 | unweighted | 0.512614953 | 0.001369753 | -0.003309612 | 0.005970847 | 0.004792111 |
| optnet16_C1 | person_weighted | 0.488239142 | 0.001128926 | -0.003318481 | 0.006711236 | 0.004747890 |

## Interpretation and reuse limits

- Historical committed scores only; these are not the current-host label-matched audits, are not selection inputs, and do not enter current adjusted comparisons.
- The original fixed-prediction table reports prior-relative recovery gains. The later baseline series retains attack loss, prior-relative absolute_recovery, and source-reported H-relative additional_recovery as distinct metrics.
- Scope names are retained literally. Both expanded_catchup and kernel_expanded_catchup at budget360 are included for every requested arm and weighting; no favorable scope, anchor, endpoint, or weighting was chosen.
- person_weighted is the historical name for PWGTP weighting. Historical predictions were selected by unweighted validation; current study uses balanced unweighted/PWGTP selection and expanded independent fitting pools.
- All three anchors partition the same repeatedly used 2018 cohort. The historical split label test denotes development evaluation, not fresh confirmation. Current matching is also development evidence, with different training/selection protocols.
- Historical J and A0 use income/employment source-task supervision. Historical SPLINCE-on-A0 preserves income/employment covariance; no residence or commute label entered those historical representation/baseline fits. The new programme explicitly supervises residence.
- Historical erasers transform only A0 auxiliary coordinates and append unchanged H. They are different objects from the new supervised PCA32-plus-residence-logit erasers, including mechanism40/union88 supplements.
- Executed OptNet here means the original stored2018 release arrays for L1/L2/C1. Original encoder weights were not saved; later reconstruction fits must not inherit these score identities.
- Same numeric width or policy label does not equalize method objectives or effective regularization. These adaptations were not an exhaustive method benchmark.
- Means and sample SDs below are descriptive summaries of published fitted-system aggregates. No historical confidence interval was recomputed, and historical multiplicity procedures differ from the current selected endpoint family.
- Historical RAC1P class-support limitations remain: the original report states code4 lacks independent fitting/validation support. Aggregate scores do not establish assessment of every race class.
- Candidate model names and fitting-seed IDs are model metadata, not person identifiers. Source paths and artifact hashes are provenance, not publication of those private artifact contents.

## Exact committed sources

| ID | Committed path | SHA-256 |
| --- | --- | --- |
| original_H_J_means | `results/redesign_20260909_acs_fixed_predictions_v1/MEANS.csv.gz` | `f9c8015275b8bc45c36d4f812d146b84a487b1d93593aa0185785ef3e551f529` |
| original_H_J_table | `results/redesign_20260909_acs_fixed_predictions_v1/TABLE.md` | `fa57abfa42d5d32238e78fa0ecfe54ac57ef17d13fd52b63528a690e715172bf` |
| historical_baseline_scores | `results/pcrl_invariant_baselines_v1/PER_SEED.csv` | `1aebf9d34977b2179b4011c9629bfb22507bf000d5260bf2c0e80ab80ea1ec48` |
| historical_baseline_report | `results/pcrl_invariant_baselines_v1/DEVELOPMENT_2018.md` | `d0f90292c210583ef07c832f2190c60299b667580baf61178a45b0d572cc9342` |
| historical_baseline_provenance | `results/pcrl_invariant_baselines_v1/DEVELOPMENT_2018.json` | `3714ec63735711641282cf31da310f65d608d0da6e9b51dc1b8b3822fe08cee7` |
| historical_baseline_audit_metadata | `results/pcrl_invariant_baselines_v1/DEV_2018_SUMMARY.json` | `5f25802764a7964654a3483252f8809f803f768f06921655ed83dc3609ce7ce0` |
| historical_baseline_adaptations | `results/pcrl_invariant_baselines_v1/BASELINE_ADAPTATIONS.md` | `4a322ff08bc43c40904ffaeb23e6d7cb5d5ca5f2769a522f3b9ff4e06d26853d` |
| historical_baseline_protocol | `results/pcrl_invariant_baselines_v1/PROTOCOL.md` | `121708d95ab535e2de4c7d0497a3d5b7a0c5cd71714033136f3f319fdabf37de` |

## Public-content inspection

No person IDs, individual labels, or person-level records found in the inspected report/index content. Per-seed rows and candidate IDs are aggregate/model metadata. Private array/model artifacts referenced by hashes were not opened.

Only the indexed committed aggregate report sources, this derived index, and owned ARTIFACT_DEFINITIONS.json/.md were inspected; this is not a repository-wide or live-output privacy audit.

No historical report was modified. The generated index stores aggregates and model provenance only.
