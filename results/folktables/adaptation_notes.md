# Folktables adaptation notes

Source: `folktables.ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")`, California state, filtered with the ACSIncome standard mask (AGEP>16, PINCP>100, WKHP>0, PWGTP>=1). Expected ~195k rows after filtering; if CA proves too small, rerun with `--states NY,TX,CA` (or similar national subset).

## Feature mapping

Dropped from the ACSIncome 10-feature spec: **OCCP, POBP, RELP**. These three columns each have hundreds of raw codes and would dominate the one-hot feature vector while adding little to the privacy/utility story this run exists to demonstrate.

Kept features (post one-hot / normalization):

| ACS column | Role       | Encoding                                  |
|------------|------------|-------------------------------------------|
| AGEP       | feature    | z-scored on train stats                   |
| WKHP       | feature    | z-scored on train stats                   |
| COW        | feature    | one-hot over 1..9                         |
| SCHL_group | feature    | one-hot over 4 bucketed levels            |
| MAR        | feature    | one-hot over 1..5                         |
| SEX        | feature    | one-hot over {1,2}                        |
| RAC1P      | feature    | one-hot over 1..9                         |

## Label / sensitive-attribute derivations

- **income** (task for p1, sensitive for p3): binary, `PINCP > 50000`.
- **hours_band** (task for p2): 4-class, `pd.cut(WKHP, bins=[-1,20,35,45,200])` — (<=20, 21–35, 36–45, >45).
- **education_level** (task for p3): 4-class, SCHL bucketed as Adult's education (<HS / HS / some-college / bachelors+).
- **sex** (sensitive): binary, remapped 0=Female / 1=Male to match Adult convention.
- **race** (sensitive): 9-class, RAC1P remapped 1..9 → 0..8.
- **age_group** (sensitive): 4-class, `pd.cut(AGEP, bins=[0,25,45,65,150])`.
- **marital_status** (sensitive): binary, 1 if MAR==1 (married, spouse present) else 0.

## Purpose structure (parallels Adult exactly except "hours_band" replaces "occupation_group")

1. `income_prediction`: allowed=`income`, disallowed=`[race, sex]`.
2. `employment_analysis`: allowed=`hours_band`, disallowed=`[race, age_group, marital_status]`.
3. `education_assessment`: allowed=`education_level`, disallowed=`[sex, race, income]`.

## Hyperparameters

MLP [128,128], repr_dim=64, λ_adv=λ_verify=50, K=20, 200 epochs, patience=20. Generated via `experiments/run_folktables.py` with the post-a50813 `generate_report` fix for shuffle-misaligned repr/label extraction.
