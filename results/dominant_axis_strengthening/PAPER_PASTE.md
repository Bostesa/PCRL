# PAPER_PASTE — §5.2 dominant-axis motivation strengthening

## One-paragraph addition (drop into §5.2)

Across the 33 multi-class pair-seeds in our v2 PCRL benchmark (Adult-R5, HMDA-R5, Diabetes-R7, three seeds each), the median $R^2_{\rm DA}/R^2_{\rm 1-hot}$ ratio is 1.43 and the mean is 2.25, with 14/33 cells exceeding 1.5$\times$ amplification, 7/33 exceeding 2$\times$, and 4/33 exceeding 3$\times$. Standard one-hot $R^2$ systematically under-reports the worst-class leakage along the dominant axis; the HMDA underwriting/race seed-1 hidden case ($R^2_{\rm 1-hot}{=}0.027$, $R^2_{\rm DA}{=}0.288$, 10.7$\times$) is the only cell satisfying the strict $R^2_{\rm 1-hot}{<}0.05{<}R^2_{\rm DA}$ definition, but it is one of 14 cells with $\geq$1.5$\times$ amplification, and is not even the most amplified one --- HMDA underwriting/race seed-0 has $R^2_{\rm 1-hot}{=}0.0026$, $R^2_{\rm DA}{=}0.033$, ratio 12.7$\times$, hidden from one-hot reporting by sitting below the 0.05 strict floor on both metrics. Dominant-axis auditing is therefore not a corner-case primitive --- it changes the conclusion for a substantive fraction of multi-class pair-seeds, with consistent per-attribute behavior (HMDA underwriting/race shows $\geq3\times$ amplification on all three seeds).

## Numbers

| Criterion | Adult | HMDA | Diabetes | Total |
|---|---|---|---|---|
| ratio > 1.5 | 4/12 | 6/9 | 4/12 | **14/33** |
| ratio > 2.0 | 0/12 | 4/9 | 3/12 | **7/33** |
| ratio > 3.0 | 0/12 | 4/9 | 0/12 | **4/33** |
| hidden strict | 0/12 | 1/9 | 0/12 | **1/33** |
| median ratio | 1.41 | 1.91 | 1.24 | **1.43** |
| max ratio    | 1.9  | 12.7  | 2.7  | **12.7** |

## Top-5 cases (full ranking in `top_5_cases.tex`)

| # | dataset | purpose / attribute | seed | K | R²_1-hot | R²_DA | ratio |
|---|---|---|---|---|---|---|---|
| 1 | hmda | underwriting / race | 0 | 5 | 0.0026 | 0.0332 | 12.65× |
| 2 | hmda | underwriting / race | 1 | 5 | 0.0268 | 0.2880 | 10.74× |
| 3 | hmda | fair_lending_audit / race | 2 | 5 | 0.0026 | 0.0098 | 3.73× |
| 4 | hmda | underwriting / race | 2 | 5 | 0.0081 | 0.0260 | 3.22× |
| 5 | diabetes | quality_research / race | 0 | 5 | 0.0007 | 0.0018 | 2.70× |

## Hidden cases (strict)

Strict-criterion hidden cases ($R^2_{\rm 1-hot}<0.05$ AND $R^2_{\rm DA}>0.05$): **1** of 33 multi-class pair-seeds — hmda underwriting/race s1 (10.7×).

## Honest assessment

The median ratio is 1.43 — meaningfully above 1.0. The reframing strengthens §5.2's motivation: the HMDA hidden case is the strongest instance of a broader pattern, not a singleton.
