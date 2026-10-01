# Literature B — summary (newer empirical analyses)

(Saved by the owner from the role's final report; the role's own write was refused by the harness.)

Files: `prior_matrix_rows.csv` (31 rows, LB01–LB32, no LB08; experiments section read for every row),
`SEARCH_LOG.md` (39 search strings; Semantic Scholar forward citations for 27 starting papers — 2,572
records, 1,953 unique titles screened; inclusion/exclusion rules; lead checks; saturation; gaps),
`source_index_rows.md` (33 statements).

## EARLY FLAGS
1. **Wang et al. (ICCV 2019, LB01) already makes the central output-side measurement**: an attacker on
   the model's task outputs vs one on true task labels matched to model accuracy; the difference is
   "bias amplification". Direct precedent for the AAAI claim that output leakage is predicted by
   AUC(s|y). Not cited in the AAAI-27 references or durable-guarantees@956f5c8 (string search).
2. **Ravfogel, Goldberg & Cotterell (ACL 2023, LB03)**: task predictions downstream of an
   R-LACE-protected representation can leak the erased attribute — small for honest binary task
   classifiers; an adversarially built multiclass classifier recovers it fully. Bears on the
   dominant-axis multiclass audit. Not in AAAI-27 references.
3. **Checking stated guarantees against held-out outcomes is already done for fair-representation
   certificates**: FRG (NeurIPS 2025, LB15) measures how often baselines' stated bounds fail on test
   data (incl. a task predicting the sensitive attribute); FARE (LB14) checks its certificate against 24
   downstream classifiers. Any novelty claim must be narrower than this.
4. **Johansson (arXiv 2403.16142, LB25)** recovers the erased grouping by anti-clustering when the
   projection is fitted on the same rows it releases. (Owner note: durable-guarantees fits LEACE /
   Obliviator / projection Q on all rows including attacker-test rows — DG-F10 — so this attack applies.)
5. **Ferry et al. (SaTML 2023, LB05)** compare a labels-only adversary with labels-plus-predictions, and
   use the published fairness constraint as attacker side information — precedent for defense-aware tiers.
6. **Closest combination precedent: Tian et al. (arXiv 2503.06150; arXiv page says accepted to IEEE
   TDSC, LB06)** — combining a biased and a fair version of the same model raises attribute inference.
7. **Lead corrections:** Obliviator is by Akbari, Afshari & Boddeti (not Kim et al.); LinEAS is
   Rodriguez et al., a steering method (out of scope); "Holstege optimal/reliable erasure" not found
   (only SPLINCE); "Gupta adaptive attacks" is FCRL (AAAI 2021: standard-scaling the representation
   defeats adversarial methods); the Aalmoes title in the brief is the v1 title of arXiv 2211.10209.

## Already established
- Stronger attackers break adversarial/deterministic fair representations: FNF (LB12), FCRL (LB13),
  Cerrato et al. AutoML (LB09), Reddy et al. (LB10), INLP/SAL/Lipstick in NLP (LB21, LB24, LB20).
- Task outputs leak: LB01–LB04, LB29, with the imputation-baseline caution of Jayaraman & Evans (LB07).
- Defense-exploiting attacks: LB25, LB05, LB30, LB03.
- Common utility protocols: fixed downstream model (LB11, LB13–LB15), selection rules (LB16, LB18),
  budget-matched comparison (LB27).
- Protection tests and other measures disagree: LB09, LB11, LB16, LB17, LB28–LB30.

## Appears open (hedged)
1. No single study found combining: LEACE-type certified erasure + defense-aware recipients + output
   recovery against a labels-only reference + one common utility operating point. Each piece has precedent.
2. Coalition recovery of several purpose-specific releases, each passing its own test, is least covered:
   only analogues (LB06, LB29, LB30) and Taylor 2026 (theory).
3. Label predictability as a cross-method predictor of output leakage not found; Wang 2019 is close.
4. Few papers test agreement between a stated test and recovery; none with a pre-registered agreement test.

## Coverage gaps
Semantic Scholar under-counts some records (amnesic 25, least-privilege 5, Taylor 0); Google Scholar
cited-by unavailable; OpenReview full text not searched; speech/recommender/graph/federated
attribute-inference literatures not searched systematically; some venues from Semantic Scholar or PDF
headers only (Aalmoes WISE, Johansson LREC-COLING, colluding-adversaries SoK USENIX 2026, Tian TDSC);
figures not checked (pdftotext extraction). Scratch downloads in scratchpad/litB/, not in the repo.
