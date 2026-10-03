# Registered directional predictions

**Registered 2026-10-03, before any new defense, head or attacker fit.**

These predictions do not select endpoints, configurations or execution order. They are scored after the run, and
they are reported whether right or wrong.

**Cells:**
- Adult income_prediction × sex;
- HMDA underwriting × race (supported classes {0, 1, 2}).

**Common definitions:**
- **AUC** means the benchmark's supported-class macro one-vs-rest AUC of the validation-selected nonlinear attacker,
  averaged over encoder, release and attacker seeds.
- **"Full output"** is the purpose head's historical logit vector.
- **"Hard prediction"** is that head's argmax.

| # | Prediction | Adult P(true) | HMDA P(true) |
|---|---|---|---|
| H1 | Outputs-only recipient: hard predictions lower recovery versus the full output by at least 0.02 AUC | 0.85 | 0.85 |
| H2 | FARE's representation (validation-selected operating point) lowers nonlinear recovery versus target-only LEACE by at least 0.02 AUC | 0.80 | 0.75 |
| H3 | FARE's common refitted task accuracy is within 0.01 of the untreated representation | 0.35 | 0.40 |
| H4 | FARE's common refitted task accuracy is within 0.01 of target-only LEACE | 0.35 | 0.40 |
| H5 | FARE plus a head fitted only on FARE features lowers recovery versus FARE plus the historical clean outputs by at least 0.02 AUC | 0.85 | 0.80 |
| H6 | Same output contract (protected features plus a head on those features): FARE lowers recovery versus LEACE by at least 0.02 AUC | 0.75 | 0.70 |
| H7 | Clean outputs bypass FARE: FARE plus clean outputs stays above 0.55 AUC | 0.90 | 0.90 |
| H8 | FARE is a competitive method: H2, H3 and H4 jointly | 0.25 | 0.25 |
| H9 | A defense-aware attacker (cell-conditional on FARE cells) is measurably stronger on validation-selected assessment AUC than the standard slate | 0.30 | 0.30 |
| H10 | Exposure sensitivity: removing training-overlapping groups changes no primary decision of the original benchmark | 0.95 | 0.95 |

**Reasoning, recorded before the run:**
- **H1:** the historical outputs-only AUC (0.78 / 0.76) is well above the label-only reference (0.60 / 0.52), and a
  hard prediction carries at most one bit.
- **H2:** FARE releases a finite cell index chosen with a fairness penalty, which coarsens the representation.
- **H3, H4:** a coarse tree usually costs task accuracy, and the one-point cap is tight.
- **H5, H7:** the clean outputs alone reach 0.76–0.78.
- **H9:** a cell-conditional predictor is close to what GBT already learns on one-hot cells.
