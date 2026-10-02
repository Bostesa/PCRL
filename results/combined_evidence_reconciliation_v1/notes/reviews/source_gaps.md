# Source gaps in the review exports: dropped formulas, dropped links, unidentified references

Role B (reviewer mapping), 2026-10-01. The private review exports lost inline math and hyperlinks when
pasted: they show blank lines or blank positions where formulas were. This note records each gap,
whether a manuscript recovers it, and which source did.

**Rule.** Nothing is filled by guessing.
- A value counts as recovered only when the surrounding review wording matches a passage in the
  submitted manuscript that prints that value.
- Where the reviewer's exact symbol or formatting cannot be known, that is said explicitly.

## Sources and abbreviations (also used in `review_response_matrix.csv`)

| Abbrev. | Source |
|---|---|
| NP p.N | NeurIPS supplied PDF (sha256 397d85d5…894f). Its text is identical to the local "(23)" build, extracted to scratch `recon/b.txt`. Page = form-feed count. |
| AP p.N | AAAI supplied main PDF (sha256 2d86afad…b077, 9 pp). Extracted with `pdftotext`; page = form-feed count. |
| AS p.N | AAAI technical-supplement PDF (sha256 9921d6cc…, 5 pp). Its source is `appendix.tex` in the Aug-30 source zip. |
| PREV | `PCRL@f381bc266:results/combined_empirical_preparation_v1/`, the prior assessment. |
| DG | `durable-guarantees@956f5c88`. Last commit 2026-08-19; every experiment commit is ≤ 2026-08-02. |
| DRIVE | The 2026-09-30 relocation `archives/` directory on the external drive (full path in `../RECON_CONTEXT.md`). Members located via `inventories/*.json.gz`. |

## A. NeurIPS export (Appendix B of the private file)

### Abstract block in the submission metadata

Nine blank positions (eight rows below; the last row holds two). The words around them match the PDF abstract exactly, so the values are
recovered with high confidence.

| Gap (paraphrased context) | Recovered value | Source |
|---|---|---|
| the per-purpose linear leakage bound the dual enforces | R²(h_p, A) ≤ τ | NP p.1 abstract |
| the trained encoder achieves … | R² ≤ 0.05 | NP p.1 |
| … on … configurations | 56/60 | NP p.1 |
| of which … also clear a non-collapse check | 7 | NP p.1 |
| HMDA underwriting/race seed 1: standard score … | R²_onehot = 0.027 | NP p.1 (also NP p.6 Table 1 caption, p.7 §5.3) |
| … but dominant-axis score … on the rarest class | R²_DA = 0.288 | NP p.1 |
| concatenation auditor recovers above majority …pp | +1 (i.e., "+1pp") | NP p.1 |
| on … of … triples | 26 of 33 | NP p.1 |

### Third human review (N-R3), summary paragraph

- **Gap.** The paper motivates dominant-axis auditing by noting that "standard [blank] metrics"
  underestimate leakage on imbalanced multi-class attributes.
- **Probable value: one-hot R².** The manuscript's own phrase is "the standard one-hot R² metric
  underestimates leakage on imbalanced multi-class attributes" (NP p.3, Related Work; also p.1, p.7
  §5.3).
- **Confidence: medium.** The reviewer may have written plain R² or R²_onehot. The token is not
  recoverable verbatim.

### Second human review (N-R2): dropped links or citations

| Gap | Status |
|---|---|
| A link appears dropped after the "Zhang et al. (2024)" phrase (stray space before the full stop). | Not recoverable from the manuscript; the reference is external. See §C. |
| "Recent conditional LoRA generation methods [blank] operate on billion-parameter models" (double space where a citation or link was). | Not recoverable from the manuscript. The examples the reviewer names (a 7B-parameter LLM, Stable Diffusion) do not identify a paper. |

The first (N-R1) review uses plain-text "R^2" and lost nothing that affects meaning.

### Other export notes

The export header's PDF, BibTeX and supplementary-zip links are not preserved. The NeurIPS supplementary
zip was not available to this role.

## B. AAAI export (Appendix C)

### Human reviews (A-R1, A-R2)

- No math was dropped.
- A-R1 cites "[1,2]", and the export contains **no bibliography** for them. See §C.
- A-R2 gives its alternative thresholds in full (0.52, 0.55, 0.60, or CI-derived), so this item is no
  longer missing.

### AI review (A-AI)

Every blank and its recovery:

| Where in the AI review | Recovered value | Source | Confidence |
|---|---|---|---|
| Weakness 1: the standard check approves removal using linear [blank] in Eq. (1) | R²_lin (written R²_lin(h, s)) | AP p.2, "The standard check" + Eq. (1) | high for the quantity; the reviewer's exact typography is unknown |
| Weakness 1: the audit declares failure using AUC [blank] | the 0.55 bar. A failure is AUC above 0.55; a pass is "at or below the bar". | AP p.3 Attackers; AP p.3 audit | value high; the relation symbol the reviewer wrote (>, ≥) is not recoverable |
| Weakness 2: FARE leaks race pairs at AUC [blank]–[blank] | 0.603–0.610 | AP p.7 FARE paragraph | high |
| Weakness 2: pairwise excess [blank]–[blank] | 0.01 to 0.11 | AP p.7 Limitations | high |
| Weakness 4: cost ramp reports [blank] | r = 0.799 | AP p.5 cost ramp | high for the value; formatting unknown |
| … with a 95% interval of [blank]–[blank] | 0.60–0.90 | AP p.5 | high |
| … cost ranges from [blank] to [blank] (15 cells above the bar) | 0.018 to 1.000 | AP p.5 | high |
| … external Pearson interval [blank]–[blank] (seven cells) | 0.35–0.98 (r = 0.872, n = 7) | AP p.5 | high |
| Weakness 6: Fig. 4 middle-cell Tier-1 point retains [blank] percent | 56.6% | AP p.5 Fig. 4 (in-sample, middle cell, Tier 1) | high |
| … of a [blank] clean lift | 0.409 | AP p.2 Setup | high |
| … approximately [blank] accuracy | **not recovered.** This is a derived quantity that the manuscript does not print. Left blank. | — | — |
| … versus [blank] clean accuracy | 0.619 | AP p.2 Setup | high |
| Suggestion 2: the basis [blank] | Q (the learned subspace basis) | AP p.3 Methods, p.4 Eq. (2), p.6 | high |
| Minor 1: the Introduction attributes [blank] to output leakage | r = 0.799 | AP p.1 (attached to output leakage) vs AP p.5 (attached to removal cost) | high |
| Minor 3: verdicts near AUC [blank] | 0.55 | AP p.3 | high |
| … including VFAE at [blank]–[blank] | 0.551–0.556 (VFAE Tier-2 reading, "at the bar") | AP p.6 VFAE paragraph | high (the only VFAE range near 0.55 in the PDF; the Tier-1 point 0.536 is a single value) |

**Dropped links in the AI review's reference list.** The parenthetical hosts after the two citations
(an MLR proceedings site and an OpenReview site) had their URLs stripped. The full bibliographic text
survives.

## C. Unidentified or ambiguous references

| Reference | What the export gives | Status | Candidate (UNCONFIRMED unless stated) |
|---|---|---|---|
| "Zhang et al. (2024)", second NeurIPS review | Author and year, plus a topic description: an information-theory-based treatment of representations that are both robust and private, aimed at attribute inference, with theoretical privacy guarantees (the review does not name a venue or title). A link seems dropped. | **Unidentified.** Not cited in the PCRL paper, and not on any branch or in the prior literature notes. | B. Zhang, S. L. Noorbakhsh, Y. Dong, Y. Hong, B. Wang, "Learning Robust and Privacy-Preserving Representations via Information Theory", arXiv:2412.11066 (Dec 2024); AAAI 2025, 39(21):22363–22371. It matches the topic words and the arXiv year. **Unconfirmed** as the reviewer's intended paper. |
| "conditional LoRA generation methods", second NeurIPS review | No citation; link dropped. | Unidentified. | None asserted. |
| "NCSGD" and "NDR", second NeurIPS review | Acronyms only. | **Unmappable.** No hits in the submitted PDF text or in any searched PCRL branch (md/tex/py). | None asserted. Possibly carried over from another paper. |
| "[1,2]", first AAAI human review | Bracket numbers only. The context is prior work showing that linear probes are insufficient and that attributes stay recoverable after adversarial removal. **No bibliography in the export.** | **Unidentified.** | None asserted. Candidates must come from the original review page. |
| "Ravfogel et al. (2022)", second AAAI review | Author and year only. | **Ambiguous: two papers.** | (a) Ravfogel, Twiton, Goldberg, Cotterell, "Linear Adversarial Concept Erasure" (R-LACE), ICML 2022. (b) Ravfogel, Vargas, Goldberg, Cotterell, "Adversarial Concept Erasure in Kernel Space", EMNLP 2022. The AAAI paper cites only (b) (AP p.8); the PCRL paper only (a), with a malformed bib entry (PREV `notes/literature_A/SUMMARY.md`). Both read in full by PREV. |
| "Elazar et al. (2021)" | Author and year. | Identified. | Elazar, Ravfogel, Jacovi, Goldberg, "Amnesic Probing", TACL 9:160–175 (cited AP p.8). |
| "Song and Shmatikov (2020)" | Author and year. | Identified. | "Overlearning Reveals Sensitive Attributes", ICLR 2020 (AP p.8). |
| "Stadler et al. (2024)" | Author and year. | Identified. | Stadler, Kulynych, Gastpar, Papernot, Troncoso, "The Fundamental Limits of Least-Privilege Learning", ICML 2024, PMLR 235 (AP p.8). |
| Gitiaux & Rangwala (2021), AI review | Full citation given: AISTATS 2021, PMLR 130:253–261. | Identified. | Read in full by PREV (`notes/literature_A/gitiaux2021_agwn.md`). Not cited by the AAAI paper. |
| Tang et al. (2026), FairNVT, AI review | The AI review cites it as an **ICLR** paper. | **Venue discrepancy.** | PREV's primary-source check found (1) v2 "FairNVT: Fair Classification via Noise Injection in Vision Transformers", arXiv:2604.16780v2, with a TMLR (08/2026) header; and (2) v1, with a different title, presented at the **ICLR 2026 AFAA workshop**. No evidence of an ICLR main-conference paper. The OpenReview/TMLR record itself returned 403 and was not opened. |
| Zemel et al. 2013; Edwards & Storkey 2016, second NeurIPS review | Author and year. | Identified. | Both already cited in the NeurIPS paper (NP p.2), as the reviewer notes (cursorily). |

## D. Other provenance notes relevant to the reviews

- **Rebuttal record.**
  - The NeurIPS export contains no substantive author response, only a withdrawal (2026-08-21).
  - No AAAI rebuttal appears in the export (the decision was Phase-1 reject).
  - Branches named "rebuttal" are therefore evidence of work, not of a submitted response.
- **NeurIPS: work predating the reviews.** The reviews were written 2026-06-22..26 and released
  2026-07-24. These runs all predate them and targeted simulated or anticipated reviewers:
  - erase-layer pilot, VICReg sweep, rank-8 ablation, LAFTR-hard-R²;
  - cross-purpose rebuttal;
  - BIOS.
- **NeurIPS: the one response to the real reviews.** The FAccT ablation registration (2026-07-24,
  `ablations-facct@ecaeba46f`) names the real reviewer identifiers, so it responds to the real reviews.
  It was **registered and smoke-tested only, never run**.
- **AAAI: all experiments predate the human reviews.** Every DG experiment commit is ≤ 2026-08-02,
  before the human reviews (2026-08-23..30).
  - Several, such as the worst-pair rescoring and the max-over-seeds diagnostics (2026-07-28..29),
    post-date the AI review's timestamp (2026-07-27). That review was visible only at the decision
    (2026-09-24).
  - A local file of simulated AAAI reviews dated 2026-07-28 exists, so those commits answered simulated
    reviews.
- **AAAI supplement access.** One human reviewer says the supplementary material was not provided. The
  export metadata lists a technical-supplement PDF and a code/data zip. A 5-page supplement exists
  locally (AS). Whether the reviewer could open it is not verifiable.
- **LAFTR-hard-R² full outputs.**
  - The full outputs survive **on the external drive only**, in archive
    `wt-PCRL-superpowers-laftr-hard-r2-2026-05-17.tar`. Members:
    - `results/laftr_hard_r2/{HEADLINE.txt,comparison.json,PAPER_PASTE.md}`;
    - `results/laftr_hard_r2_{adult,hmda,diabetes}_LAFTR_HARD_R2/{summary,per_seed_results}.json`.
  - sha256 values are in the matrix row N-R2-07.
  - The GitHub branch holds only code, the aggregator and a 5-epoch smoke run.
  - This role recomputed the strict counts from the per-seed `attribute_results`: 0/24, 17/18, 18/18.
- **Missing source-control copy of the NeurIPS manuscript.** `paper-body/*.tex` for the submitted build
  is in no git ref (PREV encoder finding F0). Layout items such as Table 2's width cannot be checked from
  the repositories.
