# Literature B — search log (newer related empirical analyses)

Execution date: 2026-10-01. Role: LITERATURE part B. No compute beyond web reads; PDFs and API dumps
kept in the session scratchpad (`.../scratchpad/litB/`), not in the repo.

## Question decomposition used for screening

The organizing question (AGENT_CONTEXT.md) was split into components:

| Code | Component |
|---|---|
| (a) | protection of a single released representation re-evaluated with stronger / nonlinear / differently-specified attackers |
| (b) | leakage through task outputs, predictions, generated text or other side information (incl. a labels-only reference) |
| (c) | defense-aware (adaptive) attackers against fair / invariant / obfuscated / erased representations |
| (d) | utility–protection benchmarks with a common protocol (same downstream model, selection rule, operating point) |
| (e) | multiple releases: composition, collusion, combining views or artefacts about the same individual |
| (f) | critiques of linear-probe certification / concept-removal evaluation |

## Inclusion / exclusion criteria

Include (matrix row) when ALL hold:
1. Empirical analysis (experiments with measured recovery, leakage or fairness), dated 2019 to 2026-10-01;
   pre-2019 anchors (Elazar & Goldberg 2018, Moyer et al. 2018, Xie et al. 2017) are discussed only in SUMMARY.
2. Studies at least one component (a)–(f) for sensitive/protected attributes (not membership, not model
   extraction, not diffusion-model concept erasure, not LLM capability unlearning).
3. Primary text was opened and at least the experiments section was read (reading depth recorded per row).

Exclude: papers deep-read by Literature A (Ravfogel 2022 R-LACE and kernel; Elazar et al. 2021 amnesic
probing; Song & Shmatikov 2020; Stadler et al. 2024; Taylor/Vippathalla/Coon 2601.21859; Gitiaux & Rangwala
2021; FairNVT) — their forward citations were still harvested. Excluded also: the "What Can AI Hide?" LLM
ledger project (per AGENT_CONTEXT), theory-only papers, surveys, and anything whose experiments I did not read.

## Indexes and endpoints used

- Semantic Scholar Graph API, forward citations (`/paper/{id}/citations?fields=title,year,venue,externalIds`),
  paginated to exhaustion or 1,000:
  - Round 1 (seeds given in the brief): R-LACE arXiv:2201.12091 (111), kernel arXiv:2201.12191 (58),
    amnesic arXiv:2006.00995 (25 — S2 appears to under-count this record), overlearning arXiv:1905.11742 (182),
    least-privilege arXiv:2402.12235 (5 — also under-counted), LEACE arXiv:2306.03819 (295).
  - Round 2 (branches): Elazar & Goldberg ACL:D18-1002 (356), Kumar arXiv:2207.04153 (46),
    Jayaraman & Evans arXiv:2209.01292 (96), FARE arXiv:2210.07213 (23), FNF arXiv:2106.05937 (47),
    Chowdhury PEF arXiv:2503.20098 (6), SPLINCE arXiv:2506.10703 (3), INLP arXiv:2004.07667 (617).
  - Round 3: Wang 2019 arXiv:1811.08489 (18), Back-to-Drawing-Board arXiv:2405.18161 (0), 10 Years
    arXiv:2407.03834 (8), Ferry arXiv:2209.01215 (16), Aalmoes arXiv:2211.10209 (9), log-linear guardedness
    arXiv:2210.10012 (2), Taylor arXiv:2601.21859 (0), Gitiaux arXiv:2006.08788 (17), Obliviator
    arXiv:2603.07529 (1), Johansson arXiv:2403.16142 (0).
  - Round 4 (output-leakage branch): Wang 2019 ICCV S2 record 4ffa7587… (478), Hirota LIC arXiv:2203.15395 (66),
    Directional bias amplification arXiv:2102.12594 (87).
  - Totals: 27 seed records, 2,572 citation records, 1,953 unique titles screened by title (keyword filter
    `remov|eras|guard|leak|attribute inference|inference attack|fair represent|invarian|debias|protected|sensitive|demographic|privacy|obfusc|adversar|probe|nullspace|concept|bias|imputation|fairness|censor|collu|release|amplif`
    applied to the large sets; all ≤120-record sets read in full).
- Web search (US web index via the WebSearch tool; acts as a Google-Scholar-like general index), arXiv abs/PDF
  pages, ACL Anthology, NeurIPS proceedings / D&B proceedings, OpenReview (search-result level only; login
  pages not opened), PMLR, CVF open access, PoPETs site, ACM DL landing pages.
- arXiv export API (rate-limited, failed twice — venue checks fell back to PDF headers, arXiv abs pages and S2).

## Search strings (web)

1. benchmarking bias mitigation algorithms in representation learning through fairness metrics Reddy 2021
2. fair representation learning evaluation stronger adversary recovers sensitive attribute after training benchmark
3. attribute inference from multiple released representations collusion fair representations composition
4. "multiple representations" OR "multiple embeddings" combined attribute inference attack privacy recipients collude representation learning
5. concept erasure evaluation nonlinear probes recover protected attribute after LEACE INLP 2024 2025 empirical study
6. sensitive attribute leakage from model predictions fair classifier outputs attribute inference fairness constraint increases leakage
7. adaptive attack defense-aware adversary privacy-preserving representation obfuscation broken evaluation attacker knows defense
8. "linear guardedness" implications multiclass leakage downstream classifier recover
9. Wang 2019 "balanced datasets are not enough" leakage model predictions protected attribute
10. "leakage amplification" OR "model leakage" "dataset leakage" protected attribute inferred from predictions bias amplification metric
11. privacy funnel multiple users collusion representation learning empirical attribute leakage combined releases
12. "Are attribute inference attacks just imputation" Jayaraman Evans CCS 2022
13. "Probing Classifiers are Unreliable for Concept Removal and Detection" Kumar NeurIPS 2022
14. Obliviator nonlinear guardedness concept erasure arXiv NeurIPS 2025
15. Mehnaz 2022 "Are your sensitive attributes private" model inversion attribute inference USENIX
16. benchmark fair representation learning methods common protocol 2024 2025 … LAFTR FNF FARE comparison
17. "Adversarial Removal of Demographic Attributes Revisited" Barrett 2019 leakage
18. Gupta "controllable guarantees for fair outcomes via contrastive information estimation" adversary evaluation
19. MiMiC minimally modified counterfactuals representation space Singh Ravfogel arXiv
20. Holstege concept erasure reliable OR optimal arXiv 2025 2026
21. LinEAS Guerreiro activation steering end-to-end arXiv
22. GAN-Invert unveiling vulnerabilities privacy-preserving facial transformations PETS 2026
23. Hamman "Can querying for bias leak protected attributes" fairness auditing privacy
24. attacker aware of fairness defense adaptive attribute inference "fair representation" adversary knows encoder white-box …
25. combining predictions from multiple models to infer sensitive attribute "multiple models" attribute inference …
26. concatenating two debiased representations recovers protected attribute erasure composition multiple erasures leakage
27. "attribute inference" "multiple releases" OR "multiple data releases" machine learning representations …
28. Decouple-and-Sample protecting sensitive information task agnostic data release ECCV 2022 attacker
29. "Does Representational Fairness Imply Empirical Fairness" Shen Han Frermann Baldwin Cohn AACL 2022
30. "Learning to Attack" uncovering privacy risks sequential data releases arXiv 2510.24807
31. "Rethinking Fair Representation Learning for Performance-Sensitive Tasks" ICLR 2025
32. openreview 2025 2026 "concept erasure" evaluation adaptive attacker … retrained probe recover gender embeddings audit
33. "fair representations" leak sensitive attribute through downstream predictions "outputs" demographic parity … 2025
34. aclanthology 2024 2025 evaluating robustness of debiasing removal of protected attributes probing stronger attacker "leakage" benchmark
35. purpose limitation representation release multiple recipients coalition sensitive attribute inference machine learning empirical
36. "attribute inference" baseline from labels alone "task label" leaks sensitive attribute … "label leakage" chance level adjusted
37. "Adaptive Privacy of Sequential Data Releases Under Collusion"
38. auditing attribute removal certificates stronger attackers LEACE certified linear guardedness nonlinear recovery tabular Adult ACS 2026
39. "sensitive attribute" recovery from "model predictions" after "concept erasure" OR "nullspace projection" downstream classifier outputs …

## Counts

- Titles screened (citation graph): 1,953 unique; plus ~300 web-search result entries.
- Abstracts / landing pages read: ~75.
- Full texts downloaded and converted (pdftotext): 63 PDFs (one, Dikaios v-latest, initially returned HTML; v1 fetched).
- Included in matrix: 31 rows (LB01–LB32; LB08 retired — see below).
- Read but not included (experiments not read, or out of scope after reading): see list below.

## Lead verification (from the brief)

| Lead | Status | Primary source |
|---|---|---|
| Reddy et al. 2021 benchmarking bias mitigation in representation learning | VERIFIED, included LB10 | NeurIPS 2021 D&B proceedings |
| "FairBench-type" / "fair representation learning benchmark" | No paper named FairBench for FRL found; nearest: EvalFRL (Cerrato 2024, LB09), TransFair (Pouget 2024, LB11), ABCFair (NeurIPS 2024 D&B, predictions only), FFB (ICLR 2024, in-processing) | arXiv / NeurIPS / ICLR |
| "Cruz/Hardt?" | Not resolved to an attribute-recovery benchmark; dropped | — |
| LEACE evaluation sections | VERIFIED, LB23 | arXiv 2306.03819 |
| "concept erasure is not robust" type papers | Johansson 2024 (LB25), Kumar 2022 (LB22), Gonen & Goldberg 2019 (LB20), Hidden-not-Deleted 2609.27593 (screened, synthetic/unlearning, excluded) | arXiv / ACL |
| Kumar et al. 2022 | VERIFIED, NeurIPS 2022, LB22 | arXiv 2207.04153 |
| Gonen & Goldberg 2019 | VERIFIED, NAACL 2019, LB20 | ACL Anthology N19-1061 |
| Xie et al. 2017 controllable invariance | VERIFIED (NIPS 2017, arXiv 1705.11122); pre-2019 anchor, SUMMARY only | arXiv |
| Elazar & Goldberg 2018 | VERIFIED (EMNLP 2018); anchor; its branch harvested (356 citers) | arXiv 1808.06640 |
| Moyer et al. 2018 | VERIFIED (NeurIPS 2018, arXiv 1805.09458); anchor | arXiv |
| Balunović et al. FNF | VERIFIED, ICLR 2022, LB12 | arXiv 2106.05937 |
| Jovanović et al. FARE | VERIFIED, ICML 2023, LB14 | arXiv 2210.07213 |
| "Kim et al. Obliviator (NeurIPS 2025)" | CORRECTED: authors are Akbari, Afshari, Boddeti; NeurIPS 2025; LB26 | arXiv 2603.07529 v2 |
| SPLINCE | VERIFIED (Holstege, Ravfogel, Wouters, NeurIPS 2025); method paper, screened (already in PCRL prior-work matrix) | arXiv 2506.10703 |
| MiMiC | VERIFIED as Singh et al., arXiv 2402.09631 (affine steering/counterfactuals); not a protection evaluation; excluded | web search result |
| Shao et al. spectral removal | VERIFIED, EACL 2023, LB24 | arXiv 2203.07893 |
| Iskander et al. | VERIFIED (Shielded Representations, ACL 2023, arXiv 2305.10204); method paper with nonlinear probes; screened, not a row | arXiv |
| Chowdhury | VERIFIED two papers: KRaM (NeurIPS 2023, 2312.00194) and PEF (AISTATS 2025, 2503.20098); method papers, screened | arXiv |
| "Guerreiro LinEAS?" | CORRECTED: LinEAS is Rodriguez et al., NeurIPS 2025 (arXiv 2503.10679), activation steering; out of scope; dropped | arXiv |
| "Holstege et al. Optimal/Reliable concept erasure" | NOT FOUND under that name; only SPLINCE located; dropped | — |
| "Gupta et al. adaptive attacks on fair representations" | RESOLVED to Gupta, Ferber, Dilkina, Ver Steeg, AAAI 2021 (FCRL): standard-scaling preprocessing breaks adversarial FRL; LB13. No separate "adaptive attacks" paper found | arXiv 2101.04108 |
| Aalmoes et al. 2022 "Leveraging algorithmic fairness to mitigate blackbox attribute inference" | VERIFIED as v1 title of arXiv 2211.10209; v3 retitled "On the Alignment of Group Fairness with Attribute Privacy"; S2 venue WISE (unverified); LB04 | arXiv abs v1 |
| Ferry et al. 2023 | VERIFIED, SaTML 2023 per S2, LB05 | arXiv 2209.01215 |
| Mehnaz et al. 2022 | VERIFIED (USENIX Security 2022, arXiv 2201.09370); training-data model-inversion AIA; screened (full text downloaded, not read beyond abstract) — not included | USENIX PDF |
| Jayaraman & Evans 2022 | VERIFIED, CCS 2022, LB07 | ACM DL |

## Read but NOT included (reason)

- Hamman, Chen, Dutta, FAccT 2023 (2211.02139): fairness-audit queries leak attributes; experiments not read → excluded.
- Singh et al., Decouple-and-Sample, ECCV 2022: experiments not read → excluded.
- Kashyap & Ali, GAN-Invert, PoPETs 2026: identity reconstruction, experiments skimmed only → excluded (noted as a defense-aware precedent outside attribute removal).
- Cui, Zhang, Pei, Learning to Attack (2510.24807): sequential trajectory releases; experiments not read → excluded, cited in SUMMARY as context for (e).
- Malekzadeh et al., Honest-but-Curious Nets, CCS 2021: outputs secretly encode attributes (malicious model); abstract only → excluded.
- Duddu & Boutet / Aalmoes Dikaios (2202.02242): fairness auditing via AIA; partial grep only → excluded.
- Abascal et al., Black-Box Privacy Attacks on Shared Representations in MTL (2506.16460): task inference, not attributes.
- Vanagas et al., Capability-Gated LMs (2609.00445) and Rauba et al., Least-Privilege LMs (2601.23157): capability access in LLMs; coalition ("joins pool a coalition's reach") concept is analogous but not attribute recovery.
- Samanta et al., Hidden not Deleted (2609.27593): synthetic superposition/unlearning.
- Chen et al., mPL joint-consumption leakage (2605.01137): metric DP / LDP aggregation, not representations.
- Jiang et al., PriMask (2211.06716): collusion-resilient masking for cloud inference (collusion = cloud + other mobiles).
- ABCFair (NeurIPS 2024 D&B), FFB (ICLR 2024), U-FaTE (CVPR 2024): prediction-fairness benchmarks/trade-off estimation without attribute recovery.
- Jones et al., Rethinking FRL for Performance-Sensitive Tasks (ICLR 2025): distribution-shift critique; not read beyond abstract.
- KRaM, IGBP/Shielded, FaRM, PEF, SPLINCE, LEOPARD/density matching (ECAI 2025), Debiasing-without-protected-attributes (2606.12088): erasure method papers whose evaluations use standard post-hoc (MLP/kernel) probes; covered collectively by LB21–LB27.
- AccretionLink (2608.14735): exposure control on post ranking, single corpus.
- Struppek et al. class attribute inference, Directional Bias Amplification (Wang & Russakovsky ICML 2021), DPA (2412.11060): bias-amplification metrics; dirBA text did not mention leakage in extracted text — not assessed.

## Saturation evidence

- Web queries 32–39 (later, differently phrased queries for (c), (e), and output leakage) returned only
  already-seen items (LEACE, INLP, SAL, Obliviator, density matching, Dikaios, Elazar & Goldberg, Taylor 2026,
  Learning to Attack) or out-of-scope domains (diffusion concept erasure, membership inference).
- Round-3/4 citation branches (10 Years, Ferry, Aalmoes, log-linear guardedness, Wang ICCV, LIC, dirBA)
  yielded no new in-scope empirical analyses beyond those already found, except Tian et al. 2025 (found via
  FNF branch) and the bias-amplification metric line (DPA, captioning follow-ups), which repeats Wang 2019's design.
- For component (e) (multiple permitted releases / coalitions of recipients), every query and branch returned
  the same small set: Taylor et al. 2026 (theory+numerics, Literature A), Tian et al. 2025 (two model versions),
  Vectors-Are-Not-Neutral (two artefacts audited separately), EEG cross-encoder transfer, sequential-trajectory
  attacks, and LLM capability-gating. No empirical study of concatenating purpose-specific certified
  representations or task outputs across recipients was found.

## Coverage gaps (do not read as completeness)

1. Semantic Scholar under-counts several records (amnesic probing 25, least-privilege 5, Taylor 0,
   Back-to-Drawing-Board 0, Wang 2019 arXiv record 18 vs ICCV record 478). Google Scholar "cited by" was not
   directly accessible; forward citations of Stadler 2024 and Taylor 2026 are therefore thin.
2. OpenReview full-text search and submitted-but-unpublished ICLR 2026/2027 papers not searched beyond web hits.
3. Venues for several preprints/papers taken from S2 or PDF headers, not publisher pages (Aalmoes/WISE,
   Johansson/LREC-COLING, SoK/USENIX 2026, Tian/TDSC).
4. Not searched systematically: speech/biometrics (beyond EEG), recommender-system attribute unlearning,
   graph-embedding attribute leakage, federated settings — these have large attribute-inference literatures
   that may contain multi-release analyses.
5. Pre-2019 work (e.g., Edwards & Storkey 2016, Coavoux et al. 2018) only referenced, not reviewed.
6. Reading was via pdftotext extraction; tables and figures were read from extracted text, so some numbers
   (e.g., figure-only results) were not checked.
