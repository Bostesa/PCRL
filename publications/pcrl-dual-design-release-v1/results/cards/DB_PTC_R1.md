# Card DB_PTC_R1: recipient-split strengths of propensity-tilt cancellation (family DB, revision R1)

Committed 2026-10-21 before its fit and before any registered reader-grid output of this campaign was read. The MODEL_SELECT utility
records (18:09Z) had been read, so under RULES.md 7 this reading is tagged ADAPTIVE_R1 even though its trigger is a MODEL_FIT
condition declared in DB_PTC_v1 section 6. Code: `ddr/db/ptc_r1.py` (entry `python3 -m ddr.db.ptc_r1 fit --version V --card <this
file>`); deployment through `ddr.db.predict` (operations `ptc_r1`, `sbc_r1`).

## 1. Observed failure that triggers it (registered trigger; MODEL_FIT only)

`db/R1_TRIGGER.json` (from `db/FIT_V0.json`, `FIT_V1.json`): at alpha* = 0.35 the components failing at the next grid value (0.5) are
income only (V0: income NLL, accuracy and summed Brier; V1: income NLL and summed Brier), while every occupation component keeps at least
40% slack at alpha* (V0: 56-99%; V1: 60-95%). The single shared strength is held back by income while occupation has room.

## 2. Operation

The DB_PTC_v1 tilt law with separate strengths: log r_I = log p_I - alpha_I (q - qbar) w_I; log r_O = log p_O - alpha_O (q - qbar_k)
W_O[k]; same weights, strata, propensity and twin. (alpha_I, alpha_O) on STRENGTHS x STRENGTHS (36 pairs); nested MODEL_FIT spend as in
DB_PTC_v1; selection per version: the admissible pair maximizing alpha_I + alpha_O (ties: the smaller maximum, then the smaller
alpha_I), a pair being admissible iff it and every grid pair dominated by it meet the DB_PTC_v1 budget (S3 plus each task's summed
Brier 0.0045, estimate + 2 household-clustered SE). None admissible: NO_FEASIBLE_STRENGTH.

## 3. Controls (PROTOCOL 8 reuse rule)

- TWIN: `DA/DA_TWIN` (unchanged).
- SPEND path: reused, DB_SPD_LO / DB_SPD_HI (the twin is unchanged); its bracket is recomputed from DB_PTC_R1's MODEL_SELECT costs.
- SBC path: refitted, because the operation changed: the same target rule (fixed-seed permutation of SEX; PSEUDO2 with gamma = 1 if
  degenerate or short of spend), the split law along the candidate's ratio rho = alpha_O* / alpha_I* (points g x m: alpha_I = g m,
  alpha_O = g m rho, g in STRENGTHS; m by the DB_PTC_v1 scale ladder), two bracket points DB_SBC_R1_LO / DB_SBC_R1_HI.
- ZTILT and SHUF are not refitted (budget); DB_ZTILT / DB_SHUF are reported beside only for the registered reading.
- Descriptive: MODEL_FIT cross-fitted readers and data-rich readers on the nested releases.

## 4. Stop rule

This is DB's first of at most two revisions. It is read under the same rules (D1-D4), tagged ADAPTIVE_R1, and reported beside the
registered reading, never as the family's registered result. If DB_PTC_R1 is NO_FEASIBLE_STRENGTH or task-infeasible, the protection
comparison is reported as not run.

## 5. Predictions (coordinator, before the fit)

alpha_I* = 0.35 and alpha_O* in {0.65, 0.8} in both versions (p 0.7); MODEL_SELECT occupation view lower than DB_PTC's by 0.005-0.015;
pair benefit vs TWIN larger than DB_PTC's by 0.000-0.008; D2 met p 0.12. Configurations: 3 x 2 = 6 (DB_PTC_R1, DB_SBC_R1_LO,
DB_SBC_R1_HI). CPU about 0.1 h.
