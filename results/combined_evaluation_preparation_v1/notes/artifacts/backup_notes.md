# Backup notes (artifacts role, 2026-10-02)

## EARLY FLAG: the cross-purpose constrained checkpoints exist on the laptop
The reconciliation (missing_files.csv, s3_references.csv) records the 9 cross-purpose constrained best.pt checkpoints as "referenced archive only (S3 lifecycle bucket, probably expired)". However, the main PCRL checkout holds all 18 of them under the git-ignored `checkpoints_archive/` (best.pt + final.pt × {adult, hmda} × s0–2 from CROSS_PURPOSE_AB, and diabetes × s0–2 from CROSS_PURPOSE_DIABETES; mtime 2026-05-31).
- They are not in any git ref.
- Their sha256 values appear in none of the 2026-09-30 drive inventories.

They were single laptop copies. After the 2026-10-02 drive remount they were copied by the private backup and are now PRESERVED on the drive (sha256 read back 2026-10-02T15:33:36Z). The hashes below were computed today. Location class: (b) present locally with a verified hash, plus a verified drive copy.

| file | size_bytes | sha256 | mtime (UTC) |
|---|---|---|---|
| checkpoints_archive/cross_purpose_ab/v2_adult_CROSS_PURPOSE_AB_s0/best.pt | 1256263 | 5aa994b146e72c5b5347eb96adc43e7a592c3369ac8ab8ea128737740d8904c4 | 2026-05-31T11:46:48Z |
| checkpoints_archive/cross_purpose_ab/v2_adult_CROSS_PURPOSE_AB_s0/final.pt | 1256356 | 7232549eea24be7a92d1576528f635e02a6e813a65e5944eff957b35f2dd5e39 | 2026-05-31T11:46:48Z |
| checkpoints_archive/cross_purpose_ab/v2_adult_CROSS_PURPOSE_AB_s1/best.pt | 1256263 | 5dc967e6d39a6fbbafcd39b492880672065fdf314a8ca066e76910a9a617d328 | 2026-05-31T11:46:49Z |
| checkpoints_archive/cross_purpose_ab/v2_adult_CROSS_PURPOSE_AB_s1/final.pt | 1256356 | e4a1b2f8a8ce01b265e5f8d6e18bdd72a9b9c2b349c6e7e4c1a0c3b374bbeeaf | 2026-05-31T11:46:49Z |
| checkpoints_archive/cross_purpose_ab/v2_adult_CROSS_PURPOSE_AB_s2/best.pt | 1256263 | 113a15ba941ef168e45a5a95b88caef67302af5edbcd92bb8638863b540f4c19 | 2026-05-31T11:46:49Z |
| checkpoints_archive/cross_purpose_ab/v2_adult_CROSS_PURPOSE_AB_s2/final.pt | 1256356 | 5ea0a250b3c566c61793b9b420fcd8ea35287a3598a50f7d829466feec9d2c9b | 2026-05-31T11:46:49Z |
| checkpoints_archive/cross_purpose_ab/v2_hmda_CROSS_PURPOSE_AB_s0/best.pt | 1011087 | a75b32fa5c30997ed9fd69d578e88e9508c7413b0dbf1a5dc0b5d6e9b3f10535 | 2026-05-31T11:46:50Z |
| checkpoints_archive/cross_purpose_ab/v2_hmda_CROSS_PURPOSE_AB_s0/final.pt | 1011168 | df351b6d066ca6ad5458970493c5e5be07675ff458289f774c006fc9bc5c7a46 | 2026-05-31T11:46:50Z |
| checkpoints_archive/cross_purpose_ab/v2_hmda_CROSS_PURPOSE_AB_s1/best.pt | 1011087 | 6d0bed7c21d3b69a78134205b7ec0d6f7589939f7b0d9f2688af83f0b920b373 | 2026-05-31T11:46:51Z |
| checkpoints_archive/cross_purpose_ab/v2_hmda_CROSS_PURPOSE_AB_s1/final.pt | 1011168 | 486313ddb655eefb142093ea67d1b60399442798d63b74a66547ad3d5e89eb77 | 2026-05-31T11:46:51Z |
| checkpoints_archive/cross_purpose_ab/v2_hmda_CROSS_PURPOSE_AB_s2/best.pt | 1011087 | 0dc1dd9118cf9d1f87f9c0ed9ea5bae715dee8e772e8131a37b0e17afdc81dee | 2026-05-31T11:46:52Z |
| checkpoints_archive/cross_purpose_ab/v2_hmda_CROSS_PURPOSE_AB_s2/final.pt | 1011168 | d63fce9cf5a7f40862f77485aa5d6efc6f26d80fd599646f21b06dae029cf375 | 2026-05-31T11:46:52Z |
| checkpoints_archive/cross_purpose_diabetes/v2_diabetes_CROSS_PURPOSE_DIABETES_s0/best.pt | 1106959 | 07aa03e249e11d91c515f5df37291d43071978c98e5487a608575a89e533960b | 2026-05-31T08:58:09Z |
| checkpoints_archive/cross_purpose_diabetes/v2_diabetes_CROSS_PURPOSE_DIABETES_s0/final.pt | 1107040 | 4003a28566c5e1141bf8a5d901924b8c24b2f2e502ec7499e36766b389a40ee9 | 2026-05-31T08:58:09Z |
| checkpoints_archive/cross_purpose_diabetes/v2_diabetes_CROSS_PURPOSE_DIABETES_s1/best.pt | 1106959 | 8670d7abc011d8a2e293e44b22245d75541fb7a875e66a14fcb64332de3761f3 | 2026-05-31T08:58:10Z |
| checkpoints_archive/cross_purpose_diabetes/v2_diabetes_CROSS_PURPOSE_DIABETES_s1/final.pt | 1107040 | c956ae3e7718f34ed0e787cc8dfb09e559f14274abfdd836190c54b052d35bd5 | 2026-05-31T08:58:10Z |
| checkpoints_archive/cross_purpose_diabetes/v2_diabetes_CROSS_PURPOSE_DIABETES_s2/best.pt | 1106959 | 509ad92f78858358a83ea8150e1bf82936d6cd273425edf5b8ff9e585af75f25 | 2026-05-31T08:58:11Z |
| checkpoints_archive/cross_purpose_diabetes/v2_diabetes_CROSS_PURPOSE_DIABETES_s2/final.pt | 1107040 | 0d7d53a28ef9953b9831b8f6d2364ee9ad6ea6e10c6227eb3df5ee12b303fc4d | 2026-05-31T08:58:11Z |

## Other notes
- The infra set is larger than recon record R-04 listed. Besides erase_pilot, erase_pilot_diabetes, erase_vicreg_sweep and erase_rank8_diabetes_cpu, `infra/` also holds inlp, splince, varconstraint and perdim_lagrangian. inlp and splince are in git history at d09c5f79f; varconstraint and perdim_lagrangian are not in any ref.
- The R-04 hash prefixes recorded in recon (erase_pilot/user_data 82e62d1f…, erase_pilot_diabetes/user_data 2b5f32a8…) match today's files, so they are unchanged since the reconciliation.
- `logs/heldout_s3/` (3 small logs from the held-out seed-3 run) is git-ignored and single-copy.
- The private Sept worktree launch state (pcrl-adaptive-release, privacy-first-selector, shared-context-release) is single-copy. It belongs to the ACS lineage and is included for completeness.
- The full per-file manifest (paths, modes, mtimes, sha256, git blob, secret-pattern categories) and the backup script are private: `~/Documents/PCRL_private_review_sources_20261001/backup_manifest_private/`.
