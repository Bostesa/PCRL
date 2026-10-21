# Protocol amendment 1 (coordinator; adjudication of the challenger's design review)

Dated 2026-10-21, written 18:04-18:06:04Z after the registration receipt (verified 17:42:42Z) and BEFORE any label read, any fit and
any mechanism card commit of this campaign. It does not edit a registration file (PROTOCOL.md, RULES.md / RULES.json, the access
policy and the other registered files stay byte-identical to the receipt). It records how the challenger's review
(`roles/challenger/REVIEW_DESIGNS.md`, saved verbatim) was adopted. The decision rules (RULES.md) are unchanged: every change below
concerns how the matched controls and the descriptive checks are built, which PROTOCOL section 4 leaves to the cards.

## Adopted

| Finding | Status | What changes |
|---|---|---|
| 1 (both families test one idea family) | ADOPTED | Every campaign-level statement says that STD and PTC both subtract the propensity-proportional part of the posterior, so a joint negative closes one idea, not two. |
| 2 (ZTILT, PTC's analogue of the HQ control that closed PRC) | ADOPTED (REQUIRED) | DB saves DB_ZTILT: the same tilt law with the amplitude P(SEX \| the twin's released log-probabilities) (a cross-fitted HGB on the twin's log-probability features), spend-matched on MODEL_FIT. Declared reading: DB_PTC not resolved below DB_ZTILT repeats PRC's failure (the gain is reproduced by release-only information); resolved below means the gain needs propensity information beyond the release, which carries no removal guarantee (finding 3). Reported beside D2 (it is not one of D2's two control paths). |
| 3, 18 (information removal or obfuscation; data-rich readers) | ADOPTED (REQUIRED for DB; also run for DA) | `ddr.richreaders`: LOG_LR (three views), HGB15 and MLP64 (pair view) trained on the 20,000 MODEL_FIT out-of-fold (DA) or nested (DB) releases with MODEL_FIT SEX inside each fit unit, scored on the deployed MODEL_SELECT releases of the DDR1 SCORE parts of R1 and R2 (entry point added by `ACCESS_AMENDMENTS.json`, operation 'readers', before first use). Descriptive label: a D2 pass whose data-rich benefit over ZTILT (DB) or over a bracketing control (both) is <= 0 in either version mean is labelled READER_BUDGET_DEPENDENT and is never called information removal. |
| 4 (shared twin; disclosure of inherited SPEND rungs) | ADOPTED | One twin: `DA/DA_TWIN` (the cached task teachers) serves both families; DB saves no twin. The finite coarsening (CRS) SPEND path is dropped (see 5), so no inherited finite-routing wire is used as a control point; the twin's coarsenings at (24, 128) and (16, 64) would have reproduced the published TWB_C128 / TWB_C64 constructions, whose MODEL_SELECT outcomes are public (pair 0.6569 / 0.6555). |
| 5 (SPEND path against continuous candidates) | ADOPTED in a stronger form | The SPEND path of both families is DQ (`ddr.ctlkit`): subtractive-dithered lattice quantization of each recipient's twin log-ratio vector in its standardized principal-component coordinates (fitted on the twin's MODEL_FIT out-of-fold outputs; keyed six-field dither). Continuous outputs (no finite cells), information removing, SEX-blind. Grids: DA (0.35, 0.7, 1.4), DB (0.15, 0.25, 0.4, 0.6, 0.9, 1.4) (each the size of its candidate's grid). |
| 6, 8, 15 (SBC fallback attribute) | ADOPTED (REQUIRED) | Both families first try a fixed-seed permutation of the MODEL_FIT SEX column (protocol default; expected degenerate or short of the candidate's spend). The common fallback is PSEUDO2 (`ddr.ctlkit`): A = 1[u < sigmoid(gamma (beta z(x) + g(y)) + c)] with z a keyed random linear score of the standardized compact fields residualized on logit pi_S and standardized, beta = SD(logit pi_S), g the logit of the label-cell SEX shares permuted across the ten cells, c matching the SEX share, u a keyed uniform of the row. Rule (a): of 20 seeded candidates, the one minimizing max over gamma of \|corr(A, SEX)\| on MODEL_FIT; accepted if <= 0.03 (else flagged SBC_ATTRIBUTE_NOT_INDEPENDENT and reported). Profile (b) recorded beside SEX's: the cross-fitted propensity AUC of A and the R^2 of its logit on the twin's pair log-probabilities. DA: gamma = the smallest of (1, 2, 4) whose lam = 1 out-of-fold spend reaches CAND's + 2 SE. DB: gamma = 1; the tilt-scale ladder m0 x (1, 2, 4, 8, 16) (m0 equalizing the RMS log-shift with the candidate's) takes the smallest scale whose top grid point reaches CAND's nested spend + 2 SE. |
| 7 (the DA card did not describe the code) | ADOPTED | DA_STD_v1 rewritten from the code before its commit. |
| 10 (R1 trigger (a) income reference) | ADOPTED | DA R1 trigger (a) is restricted to occupation; R1 triggered by (b) is adaptive and labelled so. |
| 14 (one CAND for DB) | ADOPTED | DB's CAND is the S3 selection; the designer's added per-task summed-Brier components (0.0045) are a stricter, declared deviation from S3. |
| 16 (degeneracy threshold) | ADOPTED | Cov(q, s) <= 1e-4 is degenerate, globally and per stratum (a degenerate stratum uses the global vector). |
| 17 (SHUF) | ADOPTED | DB_SHUF kept (the same SEX direction with a SEX-blind keyed-donor amplitude, FB2's lesson). |

## Declined or deferred (with reasons)

| Finding | Decision | Reason |
|---|---|---|
| 9 (inner-CV T* for the SBC teachers) | DECLINED | The SBC teachers use the candidate's components' iteration counts (the same capacity); 10-30 extra fits per attribute per version would be spent on controls only. Recorded as a known difference in tuning effort that favours neither side systematically. |
| 11 (cross-fit w) | DEFERRED | The optimism of the men's guard at lam* is reported from the MODEL_SELECT guard values (which are out of sample); w is not cross-fitted. |
| 13 (UND on MODEL_SELECT) | DECLINED | Budget; UND stays a descriptive MODEL_FIT check in DA's fit record. |
| 19 (joint-decision strata as revision 0) | DECLINED for revision 0 | The registered operation stays the designer's; joint-decision strata remain DB's revision R2. |

## Configuration plan after the amendment

DA: DA_TWIN, DA_STD, DA_SBC_LO, DA_SBC_HI, DA_SPD_LO, DA_SPD_HI, DA_COMP, DA_STD_OCC: at most 16. DB: DB_PTC, DB_SBC_LO, DB_SBC_HI,
DB_SPD_LO, DB_SPD_HI, DB_ZTILT, DB_SHUF, DB_PTC_OCC: at most 16. At most 32 of 48; 16 left for revisions and a replacement family.
