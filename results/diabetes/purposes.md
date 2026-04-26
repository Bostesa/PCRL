# UCI Diabetes 130-US — PCRL setup

## Source

Strack et al. (2014). UCI ML Repository, Diabetes 130-US hospitals for
years 1999-2008 (101,766 encounters, 130 hospitals, deidentified, public).

https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

## Preprocessing (experiments/preprocess_diabetes.py)

1. Dropped high-missing columns: `weight`, `payer_code`, `medical_specialty`.
2. Dropped rows with missing `age` or `diag_1`.
3. Deduplicated by `patient_nbr` (first encounter only) to prevent patient-level
   leakage across splits.
4. Encoded sensitive attributes:
   - `race` (5 categories: Caucasian, AfricanAmerican, Hispanic, Asian, Other;
     "?" mapped to Other)
   - `gender` (binary: Female/Male)
   - `age_bucket` (10 decade buckets: [0-10), ..., [90-100))
5. Grouped primary diagnosis `diag_1` into 9 ICD-9 categories using the Strack
   protocol:
   - circulatory (390-459, 785)
   - diabetes (250.xx)
   - digestive (520-579, 787)
   - injury (800-999)
   - musculoskeletal (710-739)
   - respiratory (460-519, 786)
   - genitourinary (580-629, 788)
   - neoplasms (140-239)
   - other (V/E codes + everything else)
6. Targets:
   - `readmission_outcome`: 1 if `readmitted == "<30"` else 0.
   - `medication_change_outcome`: 1 if `change == "Ch"` else 0.
   - `primary_diagnosis_category`: 9-class (above grouping).
7. Feature matrix: z-normalized numericals + one-hot categoricals (admission/
   discharge/source IDs, 23 medication columns × 4 levels, A1Cresult,
   max_glu_serum, diabetesMed, age_bucket).
8. 70/15/15 stratified split on `readmission_outcome`, seed=42.

## Purposes (3 purposes, 6 disallowed-attr pairs)

| Purpose | Allowed task | Disallowed attributes |
|---|---|---|
| `billing_audit` | `primary_diagnosis_category` (9-class) | race, gender |
| `quality_research` | `readmission_outcome` (binary) | race, age_bucket |
| `clinical_decision_support` | `medication_change_outcome` (binary) | race, gender |

## Methods and hyperparameters

Shared: MLP [128, 128], repr_dim=64, batch=256, epochs=200, patience=20,
seed ∈ {0, 1, 2}, 3 disallowed-attr auditor hidden layers × 256 dim.

| Method | Notes |
|---|---|
| Standard | StandardEncoder, no adversarial loss. Reprs audited directly. |
| LAFTR | StandardEncoder, all purposes' auditors, λ_adv=λ_verify=50, K=20. |
| INLP | Applied post-hoc to Standard reprs, max_iters=50, min_acc=0.52. |
| LEACE | Applied post-hoc to Standard reprs, regularization=1e-4. |
| PCRL | PurposeConditionedEncoder (FiLM), λ_adv=λ_verify=50, K=20. |

Compliance gate: delta < 2% AND linear R² < 5%.
