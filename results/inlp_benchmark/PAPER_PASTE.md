# Section 5.3 INLP three-way comparison

## Coverage

- 27 INLP cells trained (3 datasets x 3 purposes x 3 seeds)
- 60 (cell, attribute) audit rows joined to LAFTR + PCRL benchmarks

## Headline numbers

- Mean linear R^2 across all audit rows: LAFTR 0.270, INLP 0.038, PCRL 0.014
- Strict-pass rate (R^2 <= 0.05): LAFTR 25%, INLP 72%, PCRL 93%
- Mean post-INLP task accuracy across cells: 0.815

## Paragraph for §5.3 (paste-ready)

We benchmarked INLP under the same protocol used for LAFTR and PCRL: three datasets (Adult, HMDA, Diabetes), three purposes per dataset, three seeds, and a held-out linear audit of the disallowed attributes from each purpose's specification. The shared backbone was the same MLP-128-128-64 encoder PCRL uses, so the comparison isolates erasure method, not capacity. Across the resulting twenty-seven cells, INLP's null-space projection drives the mean disallowed-attribute audit R^2 to 0.038, well below LAFTR's adversarial mean of 0.270 but distinctly above PCRL's 0.014. The strict pass rate at the R^2 <= 0.05 threshold tells the same story more sharply: LAFTR clears the bar on 25% of rows, INLP on 72%, and PCRL on 93%. Where INLP loses ground is exactly where its sequential per-attribute formulation expects to: in purposes whose disallowed set spans two or three correlated attributes (e.g. {race, sex} for income prediction or {race, ethnicity} for HMDA underwriting), iterating null-space projections one attribute at a time leaves residual leakage on the second and third attribute that PCRL's joint LEACE constraint suppresses in a single pass. Task accuracy after projection averages 0.815 across cells, so the residual leakage is not the price of preserved utility — it is a representational consequence of treating multi-attribute compliance as a sequence of single-attribute problems. The full per-cell numbers and the three-way table are in inlp_results.json and pcrl_vs_laftr_vs_inlp_table.tex.
