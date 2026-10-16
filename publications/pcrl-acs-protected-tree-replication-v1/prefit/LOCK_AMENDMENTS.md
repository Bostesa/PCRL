# Lock amendments (engineering only; no scientific definition changed)

Every amendment is dated (UTC), reviewed by the lead and rerun where it affects results. The SCIENCE_LOCK
(`4d50dea13`, prefit receipt `85b2a41` on GitHub) is never edited; code hashes at lock time are in
`SCIENCE_LOCK.code_sha256_at_lock`, and later engineering changes are listed here with their effect.

| id | utc | file | change | effect on results |
|---|---|---|---|---|
| A1 | 2026-10-16T00:11Z | `acs_replication/extract.py` | The locked-mapping check read `lock['schema']['occupation_mapping_sha256']`; the lock stores it under `lock['benchmark']`. Path corrected. | None. Unit `EXTRACT_FL2019_V1` failed at this check after verifying the lock and hashing the raw zip, before any record was read; rerun as `EXTRACT_FL2019_V2` with identical rules. |
