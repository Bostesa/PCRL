# Lock notes (engineering and process record)

Entries are added in time order. Nothing here changes a registered definition.

| step | evidence | note |
|---|---|---|
| Campaign start | first admission read 2026-10-16T02:10:34Z; queue opened 02:20:03Z | elapsed budget counts from the first action |
| Admission | unit `ADMISSION_V1` | 112 stored outputs recomputed from frozen models (96 bitwise, max abs diff 4.4e-16, all decisions equal); 17 pinned source documents byte-identical to publication f4ea199 |
| Bank | unit `BANK_V1` | four combinations plus controls; 56 bitwise source identities per seed; swaps change only the intended wire |
| Deployment before lock | unit `DEPLOY_V1` | label-free parity check of the H_LJ package (bitwise); no recovery result exists at this point |
| Statistics review | `STATISTICAL_REVIEW_PRELOCK.md` | ten required changes adopted in full: final K = 38 primary / 22 secondary, settled outcomes recorded, two-stage controls, separate status fields, distance rule, critical values, forecasts |
| Forecasts | `FORECASTS.json` | registered before the lock and before any reader fit |
| Forecast timestamp correction | `FORECASTS.json` (hash pinned in ANALYSIS_LOCK) | The file's `registered_utc` field reads 2026-10-16T02:50:00Z, a time typed as an estimate before checking the clock. The file was actually written at 2026-10-16T02:46:35Z (file modification time) and is bounded by the analysis-lock commit `88c86cf57` (2026-10-15T22:47:28-04:00), before any reader fit. The file is left unedited because its hash is pinned. |
