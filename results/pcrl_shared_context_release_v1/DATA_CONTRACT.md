# Data contract

## Source and provenance
- **Data:** California ACS PUMS 2018. The three development anchors are the predecessor's sanitized `prepared.joblib` copies (`results/pcrl_task_aligned_cuts_v1/private/sanitized_2018_v2/anchor_{0,1,2}.joblib`, receipt `SANITIZATION.json`), in which outer-role labels are stripped.
- **Hash checks:** the census verified all 36 pinned local objects, all 606 `PREPARED.json` receipts, and 22 raw/preprocessing inputs against their pinned SHA-256 (`agents/support/SUPPORT_CENSUS.json`).
- **Per-person arrays used:**
  - `x`: X_A, 32 columns, float32, frozen preprocessing.
  - `ha`: H_A, 4 columns. Two probability pairs, so its effective rank is 2.
  - `hb`: H_B, 2 columns.
  - `weights`: PWGTP.
  - `households`, `ids`: never used as features.
  - Labels: `same_residence`, `SEX`, `RAC1P`, with a negative code for missing.
  - Stored, per pool: T0 code (32 states), teacher posterior `p`, residual `r = logit(p) - logit(b)`, and historical risk (11).

  The T0, p, r and risk arrays were produced on the historical Linux x86 host and are used as stored. There is no Mac re-encoding.
- **Legacy dependence:**
  - The historical teacher, risk models and codebook were fitted on earlier `teacher_fit`/`representation_fit` rows. 1,289 outer-role households overlap those historical teacher rows.
  - Every 2018 eligible person has been used somewhere before.
  - The outer role was opened once by the predecessor.

## Roles

Household roles are the predecessor's: SHA256(`pcrl_adaptive_release_v1|household_id`), first 64 bits divided by 2^64, cut at .20/.40/.65/.75/.85. Roles are global across anchors, with zero household overlap between roles (census). Global union, from the census:

| Role | People | Households | Household-weight ESS | Used for |
|---|---|---|---|---|
| nuisance_train | 5,758 | 3,884 | 1,961 | nuisances, decoder round 0, policy cost learners, contexts |
| audit_fit | 5,816 | 3,894 | 1,987 | optimization attack bank; independent audit attackers and probes (separate fits, shared households) |
| coefficient_split | 7,326 | 4,907 | 2,586 | LP coefficients, cut floors, DET_SEL choice, RD_PRIV feasibility |
| inner_selection | 3,021 | 2,003 | 1,042 | decoder/attacker/route validation; round choice |
| inner_check | 2,956 | 2,018 | 1,047 | candidate screening and nomination |
| outer | 4,403 | 2,968 | 1,524 | one locked development assessment |

- **Overlap:** 81% of coefficient people appear in at least two anchors. Anchors are not independent samples, and the inference resamples the union of households.
- **Exclusion:** attacker_validation rows (702–757 coefficient people per anchor) are excluded from non-outer roles, as in the predecessor.

## Normalization

- **Weights:** U uses w_i = 1/n_valid. W uses w_i = PWGTP_i / Σ PWGTP over the stated valid rows of that role, anchor and target.
- **Token-expanded weights:** these are w_i·q(z|x_i), with no renormalization by expanded row count.
- **Missing labels:** missing-label rows are excluded per target before any outcome is computed. Reference and candidate always use the same valid rows.
- **Class schemas:** protected classes keep the full schema (2 SEX, 9 RAC1P), with a 1e-9 floor for classes absent from a fitting role.

## Deployment contract

- **Encoder inputs:** the encoder may read only x, ha, the stored or recomputed T0, p, r and risk, and frozen models of these.
- **Forbidden inputs:** hb, labels, ids, households, roles and losses.
- **Wire:** exactly `h_a` (byte-preserved) and one integer token in 0..16.
- **Persistence:** one persistent token per record (keyed HMAC replay). Fresh redraws are a different, unanalyzed contract.
- **Recomputing inputs:** recomputing T0, p, r or risk on new data needs the Linux x86 historical environment (parity) and is outside this study's 2018 evaluation.
