# VALIDATION_SUMMARY — v4

| check | result | evidence |
|---|---|---|
| Study 4 frontier table recomputed from PER_SEED.csv | PASS 50/50 cells ≤5e-5 | STUDY4_VERIFICATION.json |
| 2017 panel recomputed | PASS, cell for cell | STUDY4_VERIFICATION.json |
| Significance counts with denominators | 189 worse / 18 better of 400 rows, 850-row simultaneous family | same |
| Coalition family at both correction levels | positive narrower than reported (C1, C2) | same |
| Residence .001 claim | REFUTED (adj low 7.64e-5; cand-wide contains 0) | same |
| Channel accounting 126→123→66→60 | PASS both directions + per-seed sums | same |
| Erasure rank | measured; counting reason withdrawn | same |
| 39/72 unmoved | PASS; 3 at β=1 | same |
| Initialisation | 114/114 from A0, 0 from J | SELECTION_AUDIT.json |
| Published selection = fresh-probe argmin | 114/114 | SELECTION_AUDIT.json |
| T1 diagnosis: γ rescoring, identities, decomposition | reproduced exactly | SELECTION_RESCORING_CHECK.json |
| Step-0 trajectories moved (T1 claim) | 63/63, median rel. displacement 0.0991 [0.0753, 0.1419] — reproduced | DISPLACEMENT_CHECK.json |
| Transport identity proofs | 15/15 at 0.0 | STUDY4_VERIFICATION.json |
| Score replay (T1) | 4512 rows, max 4.44e-16 | same |
| Math fixtures F1–F7 | all run; verdicts in MATHEMATICAL_REVIEW.md | MATH_FIXTURES.json |
| Figure-caption claims | checked against data (both weightings) | build log / this file |
| Compile | 0 errors, 0 undefined refs, 0 overfull boxes | papers/pcrl_manuscript_v4/main.log |
| Page inspection | every page rendered and viewed; no blank pages, no clipping; one unverified caption claim found and corrected | — |

Not checked: person-weighted 2017 panel beyond the md table; external literature beyond v3's read set.
Zero ACS model fits. No 2016 access.
