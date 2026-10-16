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
| Public prefit receipt | GitHub `publication/pcrl-acs-frozen-hybrids-v1` commit `46dc0b8` (pushed ~02:48Z) | Pushed after the analysis-lock commit and before any reader fit. Remote SHA equals local, and the served ANALYSIS_LOCK equals the local file byte for byte. The first link check returned HTTP 503 seconds after the push; three later checks returned 200. |
| Readers | units `READERS_S0_V1`, `READERS_S1_V1` | 125 effective fits per seed (70 marginal, 49 pair, 6 control); 28 aliased service-views not refit. Replay: 105/105 compared scores per seed bitwise equal to the source stored DEV_REPORT scores; winners agree on all 15 shared service-views. Stage-1 controls pass (planted ATTACK_VALIDATION AUC 1.0; nulls 0.499-0.512). |
| Engineering amendment E1 (before truth) | `acs_hybrids/report.py` sha256 `d415bb93eecd…` at lock -> `3dbd1fef9888…` | A synthetic-label rehearsal (`ENG_REPORT_SYNTHETIC_V1`) stopped at the source-reproduction halt, as designed, because random labels cannot reproduce published values; that hid the rest of the report path. Added a module constant `HALT_ON_SOURCE_REPRODUCTION = True` that only the synthetic rehearsal may switch off; a real run refuses to proceed if it is off. `ENG_REPORT_SYNTHETIC_V2` then completed the report but the rehearsal script's summary used two stale field names (script-only fix); `V3` ran end to end. No rule, roster, threshold or real-run behaviour changed; no DEV_REPORT truth was read. |
| Reporting lock | unit `REPORTING_LOCK_V1` | Written only after the label-free pre-truth checks passed (pair wires equal component concatenations, 125 fits per seed, all 128 score arrays finite and row-aligned) and Stage-1 controls passed. |
