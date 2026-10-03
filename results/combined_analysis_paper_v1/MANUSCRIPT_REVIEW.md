# Manuscript review: `papers/combined_empirical_v1/main.tex` (compiled `main.pdf`, 10 pages)

**Reviewer role:** read-only manuscript reviewer, 2026-10-03. Nothing else was edited or committed. Only aggregate values
were read from the private `cap_v1/run/inference.json` and `LEDGER.jsonl`; no per-person data was copied.

## Summary verdict

**Close to submittable. Revise before circulation: 21 required fixes, most of them one-line.**

- **Numbers.** Almost every number traces to a committed source and rounds correctly. Five are wrong or mislabelled:
  - abstract "0.81–0.87 under LEACE" should be 0.81–0.86;
  - abstract "0.77–0.79" should be 0.76–0.79;
  - Table 1's HMDA "+0.263" should be +0.262;
  - Table 1's HMDA frozen-head bound "(0.010)" hides that 0.0098 misses the 0.01 target;
  - the "top canonical correlation" values are squared correlations (ρ₁²).
- **Methods text.** The Roles and Inference paragraphs describe one uniform protocol: same roles, 5,243/4,764 assessment
  records, exposure rows removed, B = 1,999 normal intervals. That holds for the output diagnosis and the useful-head
  study. It does **not** hold for:
  - the matched benchmark: 5,250/4,778 records, exposure rows kept, B = 20,000 one-sided percentile bounds;
  - the output-aware study: B = 20,000 one-sided percentile bounds.

  Several Table 1 rows come from those two studies.
- **Disclosure.** The useful-head study reuses 48 aliased surfaces from the output-aware study, whose exploratory
  tables had already reported their levels:
  - full-logit output recovery 0.774 / 0.755 / 0.548 / 0.648;
  - features-plus-own-output 0.816 / 0.817 / 0.550 / 0.646;
  - the bypass values.

  The paper presents these as results of the new comparison and does not say they were already known when the
  predictions were registered.
- **Scope and wording.**
  - The "absent for multiclass heads" claim is contradicted by HMDA pricing/race, a 5-class head where the offset
    effect passed.
  - Two post-hoc comparisons in the Discussion are unlabelled.
  - One post-hoc diagnostic is written in causal language.
  - The "decisions say little" reading is stated more strongly than the useful-head data allow: useful decisions still
    recover only 0.536–0.546.
- **Citations.** All 25 cited keys resolve. `elazar2021amnesic` is in the bib but not cited, which is harmless. Two
  attributions overreach:
  - INLP and R-LACE described as *guaranteeing* linear erasure;
  - log-linear guardedness generalised beyond multiclass heads.
- **Anonymity.** There are no author names, emails, private paths, row IDs or review text in the .tex, .bib or PDF
  text. The method name "PCRL" appears in two figures but nowhere in the text (Fig. 1 node, Fig. 3 panel title).
- **Layout.** No "??", no overfull or underfull boxes (`main.log`: 0), all references resolve, and the section
  structure matches the brief. The only layout defect: in Fig. 3 the right-panel legend overlaps that panel's y-axis
  spine.

### Required-wording checklist

| Rule | Status |
|---|---|
| "pretrained" only negated | PASS (l.63, "the backbone is not pretrained"; PDF identical) |
| No R²-to-accuracy guarantee used | PASS (l.85–86 disclaims it) |
| No claim that the main-branch API is fixed | PASS (l.86: "a fix exists on an unmerged branch") |
| "not significant" / "no difference" never on an interval excluding zero | PASS literally (the phrase occurs only inside the wording rule, l.192). Borderline: "essentially unchanged" (l.228) covers HMDA B−A = −0.007 [−0.008, −0.006], whose interval excludes zero (fix 20). |
| FARE vs compression on the useful task: "smaller additional improvement, below the 0.02 target" | PASS (l.322–323). The abstract (l.44) does not use the phrase; optional O12. |
| fair_lending = constant decision, not constant scores | PASS in substance (l.237–239: "predicts the same class for everyone, although its scores still vary"); optional O5 |
| Offset effect: 5 of 6 seeds and 5 of 14 pairs | PASS (l.242–243) |
| Four bodies of work never one pipeline | PASS (l.58–62). Body (ii) is misdescribed (fix 10). |
| Post-hoc results labelled | PARTIAL: §5.1 is labelled; Discussion l.354–357 and l.360–361 are not (fix 13) |
| Reused rows not fresh confirmation | PASS (l.82–84, l.367–368). Aliased reuse inside the useful-head study is not disclosed (fix 9). |
| Absolute vs incremental coalition recovery separate | PASS (l.339–340) |
| Final-checkpoint vs best-validation counts not merged | PASS: neither count family appears |

## REQUIRED fixes

Each item gives the exact current text, then the suggested text.

1. **Abstract, LEACE range (l.36–37).** The LEACE levels are 0.817 / 0.863 (benchmark) and 0.812 / 0.864
   (output-aware), so 0.87 is the *untreated* HMDA value (0.870).
   - Current: "(AUC about 0.81--0.87 under official LEACE)"
   - Suggested: "(AUC about 0.81--0.86 under official LEACE, against 0.82--0.87 untreated)"

2. **Abstract, bypass range (l.39).** The values are noise plus clean outputs 0.778 / 0.758 and FARE plus clean outputs
   0.788 / 0.768. HMDA noise is 0.758, so the range starts at 0.76. The body (l.233) already says 0.76–0.79.
   - Current: "(recovery returns to 0.77--0.79)"
   - Suggested: "(recovery returns to 0.76--0.79)"

3. **Table 1, HMDA full vs hard (l.213).** `combined_output_aware_removal_v1/PRIMARY_ENDPOINTS.csv` P1-hmda gives point
   0.262472 and lower 0.253953. The 0.263 was inherited from a rounding slip in the source RESEARCH_DECISION.md.
   - Current: "0.768 vs 0.505: +0.263 (0.254)"
   - Suggested: "0.768 vs 0.505: +0.262 (0.254)"

4. **Table 1, frozen-head gain row (l.216), and the stated targets.** U-frozen-hmda has lower bound 0.009780 against a
   target of 0.01. Printing "(0.010)" next to "not established" looks self-contradictory. That endpoint's target is
   0.01, but §4 (l.190) gives only the useful-head gain target of 0.03.
   - Current: "Frozen head: accuracy gain over constant & 0.039 (0.031) & 0.016 (0.010), not established"
   - Suggested: "Frozen head: accuracy gain over constant (target 0.01) & 0.039 (0.031) & 0.016 (0.0098), not
     established"
   - Also add to l.190: "(the earlier frozen-head usefulness endpoints used a gain target of 0.01)".

5. **Roles / Inference paragraphs (l.165–169, l.187–190) and Contribution (1) (l.70).** These describe the diagnosis
   and useful-head protocols as if they held for every study. Table 1 also draws on the matched benchmark (12/42,
   0.815/0.870, 0.817/0.863, noise rows, 0.177/0.236) and on the output-aware study (0.249/0.254, FARE rows). Those
   studies differ:
   - **Matched benchmark:** assessment 5,250 / 4,778; the 17/42 exposure rows were *not* removed; cluster bootstrap with
     B = 20,000; one-sided Bonferroni percentile bounds at α = 0.05/24.
   - **Output-aware study:** B = 20,000; one-sided percentile at α = 0.05/12.

   Changes:
   - l.165, current: "Every analysis uses the same record-group roles:"
     → suggested: "The output-aware, output-diagnosis and useful-head analyses use the same record-group roles:"
   - After l.170, add: "The earlier matched benchmark used the uncleaned roles (assessment 5{,}250 Adult and 4{,}778
     HMDA records; the 17/42 overlapping records retained and disclosed)."
   - l.187–189, current: "Standard errors come from 1{,}999 paired multinomial bootstrap replicates ... intervals are
     two-sided Bonferroni normal intervals over the endpoint family"
     → suggested: "In the output diagnosis and useful-head studies, standard errors come from 1{,}999 paired
     multinomial bootstrap replicates ...; intervals are simultaneous two-sided 95\% Bonferroni normal intervals. The
     matched benchmark and the output-aware study used one-sided Bonferroni percentile bounds from 20{,}000 group
     bootstrap replicates (Table~\ref{tab:summary} bounds from those studies are of that kind)."
   - l.70, current: "A single evaluation protocol applied to the same stored models"
     → suggested: "A common evaluation protocol (one attacker slate and endpoint discipline; roles and bootstrap
     differ slightly across studies, Section~\ref{sec:protocol}) applied to the same stored models"

6. **Table 1 shows two different LEACE feature-recovery values (l.211 vs l.217) without saying why.** They are 0.817 /
   0.863 from the benchmark roles and 0.812 / 0.864 from the output-aware roles.
   - Suggested: add the source to each row, e.g. "Nonlinear recovery, LEACE features (benchmark roles; bound met on all
     60 maps)" and "FARE features vs LEACE features (output-aware roles)". Alternatively, add a caption sentence: "Rows
     1--5 are from the matched benchmark, rows 6 and 9--11 from the output-aware study, rows 7--8 from the output
     diagnosis."

7. **ρ₁² mislabelled (l.227–228).** The source (benchmark RESEARCH_DECISION §2; output-aware CORRECTION_ADDENDUM W5)
   reports the *squared* top canonical correlation ρ₁².
   - Current: "its held-out top canonical correlation is near zero (Adult 0.0006, HMDA 0.0001)"
   - Suggested: "its held-out squared top canonical correlation $\rho_1^2$ is near zero (Adult 0.0006, HMDA 0.0001, on
     one already-used held-out split)"

8. **Multiclass claim contradicted (l.243–244).** HMDA `pricing_analysis` is a 5-class head (FROZEN_HEAD_UTILITY K = 5)
   and S3-FC-hmda-pricing_analysis-race passes: 0.047 [0.041, 0.054]. The 9 non-passes include all 6 multiclass
   *Adult* pairs and HMDA pricing/sex.
   - Current: "it was absent for multiclass heads."
   - Suggested: "it was not established for any of the six multiclass Adult pairs, although it passed for the
     five-class HMDA pricing head with race."

9. **Undisclosed aliasing in the useful-head study (l.252–253, l.308–314).** `COVERAGE_AND_UNITS.csv` lists 48 alias
   units: output-only full logits, features only, features plus own full output, and features plus clean output, all
   reused from the output-aware study. That study's `EXPLORATORY.csv` had already reported their levels:
   - output-only R(O_head) A 0.774, B 0.755, F 0.548, F0 0.648;
   - rep+head A 0.816, B 0.817, F 0.550, F0 0.646.

   The "Complete release" paragraph's levels and the 0.788 / 0.789 bypass values are these reused units. The registered
   predictions were written while the full-logit levels were in committed files, although the protocol disclosed only
   the accuracies as "already known".
   - l.252–253, current: "The heads, LEACE maps and FARE trees are reused unchanged; 84 new attacker units (3{,}924
     model fits) were run after the protocol lock."
     → suggested: "The heads, LEACE maps and FARE trees are reused unchanged. Full-logit output-only, features-only,
     features-plus-full-output and bypass surfaces are reused hash-verified from the output-aware study, whose
     exploratory tables had already reported their levels. The centred, probability and hard surfaces and the banks
     are new: 84 attacker units (3{,}924 model fits), run after the protocol lock."
   - l.312, current: "Four of the ten primary recovery predictions were correct."
     → suggested: "Four of the ten primary recovery predictions were correct. The predictions were registered after
     full-logit recovery of these same heads had been reported as exploratory in the output-aware study."

10. **Body of work (ii) misdescribed (l.60–61).** Per CORRECTIONS_AND_SCOPE, scope statement 1:
    - body (ii) is the erase-layer and isolation/noise work, including the LEACE/INLP/LAFTR/SPLINCE comparisons;
    - the official LEACE + noise benchmark on stored representations belongs to body (iv).

    - Current: "attribute-removal and noise baselines applied to their stored representations; finite-release studies
      on later census data; and the stored-model output analyses that this paper centres on"
    - Suggested: "attribute-removal and isolation work on those encoders (an erase layer, noise, and comparisons with
      LEACE, INLP, LAFTR and SPLINCE); finite-release studies on later census data; and the stored-model analyses that
      this paper centres on (a matched removal benchmark, an output-aware study, an output diagnosis and the
      useful-head comparison)"

11. **Over-attribution: INLP and R-LACE described as guarantees (l.123–125).** INLP is iterative nullspace projection,
    and R-LACE solves a linear minimax game. Only LEACE proves linear guardedness. CLOSEST_PRIOR_ANALYSES records only
    their nonlinear-recovery findings.
    - Current: "Linear concept erasure guarantees that no linear classifier can recover a concept:
      INLP~\citep{ravfogel2020inlp}, R-LACE~\citep{ravfogel2022rlace} and LEACE~\citep{belrose2023leace}, which
      achieves this in closed form"
    - Suggested: "Linear concept erasure aims to stop linear classifiers from recovering a concept: INLP iteratively
      and R-LACE through a linear minimax game, while LEACE~\citep{belrose2023leace} guarantees it in closed form"

12. **Over-generalisation of log-linear guardedness (l.137).** CLOSEST_PRIOR_ANALYSES: binary log-linear heads cannot
    recover the concept; a constructed multiclass softmax head can.
    - Current: "A downstream classifier on a linearly guarded representation can still reveal the erased
      concept~\citep{ravfogel2023loglinear}."
    - Suggested: "A multiclass log-linear head on a linearly guarded representation can still reveal the erased
      concept, although a binary one cannot~\citep{ravfogel2023loglinear}."

13. **Unlabelled post-hoc comparisons in the Discussion.** The decision-vs-score comparison (0.544 vs 0.539 / 0.648) is
    not in any registered family. It is labelled post hoc in §5.1 but not here.
    - l.354, current: "\paragraph{A measured gap.} At matched accuracy"
      → suggested: "\paragraph{A measured gap (post hoc, descriptive).} At matched accuracy"
    - l.360–361, current: "and FARE's score leaks about as little as the untreated decision."
      → suggested: "and, post hoc, FARE's score leaks about as little as the untreated decision."

14. **Causal language from a post-hoc diagnostic (l.298–301).**
    - Current: "A post-hoc diagnostic suggests why LEACE barely changes the score: a linear attacker on the untreated
      head's margin recovers only 0.515, while nonlinear attackers on the same scalar recover 0.773; this is consistent
      with the encoders' linear constraint: the remaining signal in the score is nonlinear, outside the scope of a
      linear eraser."
    - Suggested: "Post hoc and descriptively, a linear attacker on the untreated head's margin recovers only 0.515,
      while nonlinear attackers on the same scalar recover 0.773. This is consistent with, but does not establish, the
      reading that the sex signal left in the score is not linearly decodable and so lies outside a linear eraser's
      scope."

15. **"Decisions say little" overstated (abstract l.40; l.239; l.352–353).** The useful heads' decisions (gain 0.073–0.082)
    still recover only 0.536–0.546. Low decision-level recovery therefore persisted when decisions were useful, so weak
    usefulness cannot be the explanation. The frozen-vs-useful comparison (0.513 vs 0.544) also crosses studies and
    heads.
    - Abstract, current: "(iii) Low recovery from hard decisions often reflects decisions that say little; when every
      arm releases"
      → suggested: "(iii) The frozen heads' hard decisions leak little but are also weakly useful; when every arm
      releases"
    - l.239, current: "A decision that says little also reveals little."
      → suggested: "Low decision-level recovery here coincides with decisions that barely beat a constant."
    - l.352–353, current: "Part of the earlier ``decisions leak little'' pattern came from decisions that barely beat a
      constant."
      → suggested: "The earlier ``decisions leak little'' pattern was observed on decisions that barely beat a
      constant; with useful heads, decisions still recovered only 0.536--0.546, so weak usefulness is not the whole
      story."

16. **Point presented as a bound (l.345).** Acc(B) − Acc(A) = −0.0006 with interval [−0.0016, 0.0003]. "Within 0.001"
    describes the point only; the interval extends to −0.0016.
    - Current: "LEACE preserved the deployed head's accuracy (within 0.001)"
    - Suggested: "LEACE preserved the deployed head's accuracy (difference $-0.0006$, lower bound $-0.0016$;
      non-inferior within one point)"

17. **FARE certificate overgeneralised (l.228–229).** NATIVE_TEST_VS_RECOVERY.md shows:
    - the Adult nominee bound is 0.497 / 0.504 (weak, not vacuous);
    - one informative bound, 0.175, on the degenerate single-cell HMDA s1 nominee.
    - Current: "FARE's native certificate was unavailable or vacuous at the available certification sizes."
    - Suggested: "FARE's native certificate was unavailable, vacuous or weak (Adult bounds about 0.50) at the available
      certification sizes; its only informative bound (0.175) belongs to a degenerate single-cell HMDA nominee."

18. **Seed-2 coalition sentence is wrong as worded (l.340–341).** Seed 2 contributes zero only to the pair-minus-income
    difference. For pair minus employment it contributes income minus employment.
    - Current: "validation selected the income-only attacker for the pair, contributing exactly zero to that endpoint."
    - Suggested: "validation selected the income-only attacker for the pair, so that seed's pair-minus-income
      difference is exactly zero."

19. **"PCRL" in figures only (fig_recipient.tex l.15; Fig. 3 left-panel title from `report/figures.py`).** The text
    never names or defines the method. Using the name of the authors' earlier system is an avoidable anonymity link and
    an undefined term.
    - Fig. 1 node, current: "historical clean\\PCRL task logits" → suggested: "historical clean\\task logits
      (untreated)"
    - Fig. 3 title, current: "Frozen PCRL heads: one recipient" → suggested: "Frozen task heads: one recipient"

20. **"Essentially unchanged" on an interval that excludes zero (l.228).** HMDA B − A nonlinear AUC is −0.007
    [−0.008, −0.006] (benchmark §3).
    - Current: "yet nonlinear recovery is essentially unchanged (Table~\ref{tab:summary})."
    - Suggested: "yet nonlinear recovery changes by less than 0.01 AUC (Adult $+0.002$, HMDA $-0.007$; the HMDA
      interval excludes zero; Table~\ref{tab:summary})."

21. **Fig. 3 layout (p.7).** The right panel's legend, placed `loc="upper center"` with `ncol=3`, extends left across that
    panel's y-axis spine: the "income alone" swatch sits on the axis line. Suggested fixes:
    - move the legend below the title or outside the axes, e.g. `bbox_to_anchor=(0.5, 1.02)` with the title raised;
    - or narrow it with `ncol=1`, `loc="upper right"`;
    - and add a y-axis label to the right panel ("Race recovery (macro AUC)").

## OPTIONAL suggestions

- **O1 (l.254).** "the 19 primary endpoints use the centred and hard formats": 9 of the 19 are utility endpoints. Write
  "the ten primary recovery endpoints use the centred and hard formats".
- **O2 (Fig. 2 caption, l.261–262).** The bars are ±1.645 SE (`figures.py`), i.e. *marginal* 90 % intervals. Table 2
  gives simultaneous 95 % intervals. Write "marginal 90\% intervals (Table~\ref{tab:useful} gives simultaneous ones)".
- **O3 (Table 1, l.215).** "−0.000" should be "$-0.0003$" (HMDA s1 FC = −0.00028), so it does not read as negative zero.
- **O4 (l.38, l.226).** "attribute--seed combinations" should be "pair--seed combinations": 14 purpose–attribute pairs
  × 3 seeds.
- **O5 (l.237–238).** Add the literal phrase: "the HMDA fair\_lending head's decision is constant (it predicts the same
  class for everyone), although its scores still vary ...".
- **O6 (generality).**
  - Abstract (iv): write "In one Adult recipient pair, two recipients pooling ...".
  - Abstract (ii) and l.361: "any feature defense" should be "every feature defense we evaluated". The logical point
    (features cannot remove what an unchanged output carries) holds, but the measured recovery is cell-specific.
- **O7 (l.145–146).** "LAFTR lack[s] guarantees against unseen classifiers" is not supported by CLOSEST_PRIOR_ANALYSES.
  LAFTR does bound downstream unfairness, but only for an optimal adversary. Suggest "LAFTR's bounds assume an optimal
  adversary~\citep{madras2018laftr}".
- **O8 (l.143–144).** Zhao et al. bound the accuracy cost of demographic parity, not of "protection". Suggest "fair
  representations that enforce demographic parity must trade accuracy when base rates differ".
- **O9 (Fig. 1).**
  - The dashed path (3) carries only the clean logits, but its label says "features + clean logits". Add a dashed
    branch from "features h"/"feature arm" or relabel it "(3) clean logits, added to the features (adverse control)".
  - The frozen-head analyses (Table 1, Fig. 3) are not represented. Optionally add to the caption: "Table 1 and Fig. 3
    release the stored frozen head's output through path (1)".
- **O10 (Fig. 2, p.5).** In the hard-decision panel the A and B large markers overlap (0.544 at gain 0.082 / 0.083), so
  B is hidden. Small seed markers at alpha 0.35 are faint in print. Jitter the markers or raise the alpha.
- **O11 (Fig. 3).** The bars start at 0.45, which exaggerates differences, and there are no intervals. Either start the
  axis at 0.5 (chance) with a note or add error bars.
- **O12 (abstract l.44).** Write "compression reproduced most of FARE's effect (FARE's additional improvement, 0.007
  AUC, was below the 0.02 target)".
- **O13 (l.292).** "seed~2 loses 2.0 points" should be "seed~2 loses 2.0 accuracy points".
- **O14 (Table 2 caption).** Write "simultaneous two-sided 95\% intervals (19 endpoints; Bonferroni $z=3.008$)".
- **O15 (l.375).** "about 0.2 CPU-hours": the only source is the private `LEDGER.jsonl` (0.169 CPU-h for 88 units,
  verification excluded). Put the number in a committed file (e.g. `USEFUL_HEAD_COMPARISON.md`) or write "about 0.2
  CPU-hours excluding verification".
- **O16 (references.bib).**
  - The header comment ("claims/literature reviewer, combined-analysis-paper-v1") is internal process text. It is
    harmless in the PDF but should be stripped from any source upload.
  - The comment says each entry has a `note` field, but no entry has one.
  - The Mehnaz title renders "private? novel model inversion" (lowercase n); brace it as `{N}ovel`.
  - The `gupta2021fcrl` volume+number BibTeX warning is harmless.
- **O17 (l.34, abstract).** "using one attacker slate, held-out roles, and paired group-bootstrap intervals" is
  acceptable, but it should match the qualified wording adopted in fix 5.
- **O18 (Setting l.90).** "Round~4 checkpoints" is internal versioning. Either explain it in one clause ("the fourth
  training round of the encoders") or drop it.

## Number audit

Sources are abbreviated as follows:

| Abbreviation | File |
|---|---|
| CAP | `combined_analysis_paper_v1/` |
| P | `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` |
| S | `SECONDARY_COMPLETE_RELEASE_ENDPOINTS.csv` |
| U | `ACTUAL_HEAD_UTILITY.csv` |
| INF | private `inference.json`, aggregate levels only |
| PH | `report/posthoc_diagnostics.json` |
| CR | `report/corrections_recomputed.json` |
| ODX | `combined_output_diagnosis_v1/` (PRIMARY_ENDPOINTS, S3, S4, S5, FROZEN_HEAD_UTILITY, OUTPUT_SURFACE_RESULTS) |
| OAR | `combined_output_aware_removal_v1/` |
| MRB | `combined_matched_removal_benchmark_v1/RESEARCH_DECISION.md` |

| # | Location | Manuscript value | Source and exact value | Verdict |
|---|---|---|---|---|
| 1 | Abstract (i) | AUC about 0.81–0.87 under official LEACE | MRB §3 B: 0.817 / 0.863; OAR 0.812 / 0.864. 0.87 is untreated HMDA (0.870). | **MISMATCH** → 0.81–0.86 (fix 1) |
| 2 | Abstract (i) | 12 of 42 | MRB §1 "12 of 42 fail" | OK |
| 3 | Abstract (ii) | 0.77–0.79 | MRB §4 noise + clean 0.778 / 0.758; OAR FARE + clean 0.788 / 0.768 | **MISMATCH** → 0.76–0.79 (fix 2) |
| 4 | Abstract (iii) | LEACE head score 0.759 | INF R\|B\|out\|centred 0.7587 | OK |
| 5 | Abstract (iii) | untreated 0.774 | INF R\|A\|out\|centred 0.7735 | OK |
| 6 | Abstract (iii) | FARE 0.539 | INF R\|F\|out\|centred 0.5390 | OK |
| 7 | Abstract (iii) | within one point (NI not established) | P U-acc-F-vs-A −0.009918 [−0.015201, −0.004635], NOT_ESTABLISHED | OK |
| 8 | Abstract (iii) | tree 0.648 | INF R\|F0\|out\|centred 0.6482 | OK |
| 9 | Abstract (iii) | 0.109 AUC beyond compression | P R-out-centred-F0-minus-F 0.109228 | OK (labelled AUC) |
| 10 | Intro / Setting | six encoders, seeds 0,1,2; 64-dim; R² ≤ 0.05; 14 pairs | MRB; OAR PROTOCOL; CORRECTIONS scope 1 | OK |
| 11 | Setting l.115 | concept-erasure 0.2.4 | MRB pins | OK |
| 12 | Setting l.116 | FARE nominee within one point of untreated validation accuracy | OAR PROTOCOL "U2 attacker_val accuracy ≥ untreated − 0.01" | OK |
| 13 | Protocol l.166 | 20 % head holdout | OAR RELEASE_CONTRACTS l.39 "minus a hash-held-out 20 %" | OK |
| 14 | Protocol l.167 | assessment 5,243 Adult / 4,764 HMDA | OAR PROTOCOL roles table; CAP PROTOCOL | OK for OAR/ODX/CAP; **MISMATCH for MRB rows** (5,250 / 4,778) (fix 5) |
| 15 | Protocol l.168 | 17 Adult, 42 HMDA excluded from every scored role | OAR PROTOCOL l.24; ODX PROTOCOL l.25 | OK for OAR/ODX/CAP; **MISMATCH for MRB** (not removed) (fix 5) |
| 16 | Protocol l.169 | further 20 % of attacker_fit for certification | OAR PROTOCOL l.25 | OK |
| 17 | Protocol l.172–174 | LR 5, GBT 20, MLP 18; three attacker seeds | CAP PROTOCOL §4 | OK |
| 18 | Protocol l.188 | 1,999 paired multinomial replicates, two-sided normal | CAP and ODX protocols | OK for CAP/ODX; **MISMATCH for MRB/OAR** (B = 20,000, one-sided percentile) (fix 5) |
| 19 | Protocol l.190–191 | targets −0.01, 0.03, 0.8/0.2 retention, 0.02 AUC | CAP PROTOCOL §5 | OK for CAP. ODX usefulness target was 0.01 (fix 4). |
| 20 | Table 1 | 12 of 42 | MRB §1 | OK |
| 21 | Table 1 | untreated NL 0.815 / 0.870 | MRB §3 A | OK |
| 22 | Table 1 | LEACE NL 0.817 / 0.863; 60 maps | MRB §3 B; "holds for all 60 maps" | OK (but see fix 6) |
| 23 | Table 1 | noise features 0.528 / 0.513 | MRB §3 D at σ\* | OK |
| 24 | Table 1 | noise + clean 0.778 / 0.758 | MRB §4 | OK |
| 25 | Table 1 | Adult 0.776 vs 0.513: +0.263 (0.249) | ODX OSR FH full 0.77571, FH hard 0.51293; OAR P1-adult 0.262778, lower 0.249244 | OK |
| 26 | Table 1 | HMDA 0.768 vs 0.505: +0.263 (0.254) | ODX OSR 0.76790 / 0.50542; OAR P1-hmda **0.262472**, lower 0.253953 | **MISMATCH** → +0.262 (fix 3) |
| 27 | Table 1 | offset +0.072 (0.062) / +0.067 (0.060) | ODX FC-adult 0.071852, lower 0.062270; FC-hmda 0.067018, lower 0.059537 | OK |
| 28 | Table 1 | per seed 0.079 / 0.064 / 0.072 | CR FC_per_seed adult 0.07900 / 0.06433 / 0.07222 | OK |
| 29 | Table 1 | per seed 0.058 / −0.000 / 0.143 | CR hmda 0.05818 / −0.00028 / 0.14315 | OK (O3: print −0.0003) |
| 30 | Table 1 | frozen gain 0.039 (0.031) | ODX U-frozen-adult 0.039290, lower 0.030582 | OK |
| 31 | Table 1 | frozen gain 0.016 (0.010), not established | ODX U-frozen-hmda 0.015603, lower **0.009780**, target 0.01 | **MISMATCH** (rounding hides the failure) → (0.0098) (fix 4) |
| 32 | Table 1 | FARE vs LEACE features 0.550 vs 0.812 / 0.501 vs 0.864 | OAR NATIVE_TEST_VS_RECOVERY l.15, l.18; P2 0.2618 / 0.3628 | OK |
| 33 | Table 1 | zero-fairness tree features 0.645 / 0.607 | OAR RESEARCH_DECISION §3; EXPLORATORY hmda R(FZ,rep) 0.60752 | OK |
| 34 | Table 1 | FARE + clean 0.788 / 0.768 | OAR RESEARCH_DECISION §3; NATIVE_TEST l.19 | OK |
| 35 | §5 l.226–227 | 12 of 42; HMDA underwriting/race on two of three seeds | MRB §1 (s1 0.090, s2 0.081 > 0.05) | OK |
| 36 | §5 l.227 | 60 maps | MRB | OK |
| 37 | §5 l.228 | "top canonical correlation" 0.0006 / 0.0001 | MRB §2: these are **ρ₁²** (squared) | **MISMATCH (label)** (fix 7) |
| 38 | §5 l.232 | output minus label-only: 0.177 / 0.236 | MRB §4 +0.177 [0.166, 0.188], +0.236 [0.227, 0.245] | OK |
| 39 | §5 l.233 | 0.76–0.79 (noise, FARE) | rows 24 and 34 | OK |
| 40 | §5 l.235–236 | hard 0.513 / 0.505 | ODX OSR FH hard 0.51293 / 0.50542 | OK |
| 41 | §5 l.236 | Adult frozen head gains 3–5 accuracy points | ODX FROZEN_HEAD_UTILITY gains 0.0366 / 0.0334 / 0.0479 | OK (labelled accuracy points) |
| 42 | §5 l.237 | recalls 14–22 % of high earners | FROZEN_HEAD_UTILITY per_class_recall[1] 0.2228 / 0.1399 / 0.2122 | OK |
| 43 | §5 l.239 | fair_lending centred recovery 0.646 / 0.655 | ODX OSR fair_lending race centred 0.64588; sex centred 0.65542 | OK |
| 44 | §5 l.242 | offset effect 0.072 / 0.067 AUC | row 27 | OK |
| 45 | §5 l.242–243 | 5 of 6 seeds; not HMDA s1 | CR FC per seed | OK |
| 46 | §5 l.243 | 5 of 14 pairs | CR offset_FC_pairs n 14, pass 5 | OK |
| 47 | §5 l.243–244 | "absent for multiclass heads" | ODX S3: HMDA pricing (K = 5)/race PASS 0.0473 [0.0411, 0.0536] | **MISMATCH (claim)** (fix 8) |
| 48 | §5 l.245–246 | refitted head gain 0.083; "about twice" | CR HEAD__A_gain_mean 0.082650; frozen 0.0393 (ratio 2.1) | OK |
| 49 | §5 l.246 | margin alone recovers 0.773 | CR RH centred 0.77349 | OK |
| 50 | §5.1 l.252–253 | 84 new units, 3,924 fits | USEFUL_HEAD_COMPARISON §1; LEDGER sum of fits 3,924 | OK (aliasing undisclosed, fix 9) |
| 51 | §5.1 l.254 | verified to 10⁻¹² | CAP PROTOCOL §3 | OK |
| 52 | §5.1 l.254 | 19 primary endpoints | P (19 rows) | OK (O1 wording) |
| 53 | Fig. 2 caption | three seeds; 90 % intervals | `figures.py` yerr = 1.645·se (marginal) | OK (O2: say "marginal") |
| 54 | Table 2 caption | 19; z = 3.008 | P z 3.007787 | OK |
| 55 | Table 2 | Acc(B)−Acc(A) −0.0006 [−0.0016, 0.0003] pass | P −0.000636 [−0.001563, 0.000291] PASS | OK |
| 56 | Table 2 | Acc(F)−Acc(A) −0.0099 [−0.0152, −0.0046] n.e. | P −0.009918 [−0.015201, −0.004635] | OK |
| 57 | Table 2 | Acc(F0)−Acc(A) −0.0040 [−0.0086, 0.0005] pass | P −0.004005 [−0.008551, 0.000541] | OK |
| 58 | Table 2 | gain 0.082 / 0.073 / 0.079; ≥ 0.067 / 0.057 / 0.063 | P 0.082014 / 0.072732 / 0.078645; lower 0.067193 / 0.057087 / 0.063386 | OK |
| 59 | Table 2 | retention 0.016 / 0.007 / 0.013; ≥ 0.013 / 0.001 / 0.007 | P 0.015894 / 0.006612 / 0.012525; lower 0.012895 / 0.000641 / 0.007283 | OK |
| 60 | Table 2 | R(A)−R(B) 0.015 [0.011, 0.019] n.e. (excl. 0) | P 0.014811 [0.010540, 0.019081] | OK |
| 61 | Table 2 | R(A)−R(F) 0.234 [0.217, 0.252] | P 0.234475 [0.216903, 0.252046] | OK |
| 62 | Table 2 | R(A)−R(F0) 0.125 [0.112, 0.139] | P 0.125247 [0.111535, 0.138958] | OK |
| 63 | Table 2 | R(B)−R(F) 0.220 [0.201, 0.238] | P 0.219664 [0.201121, 0.238207] | OK |
| 64 | Table 2 | R(F0)−R(F) 0.109 [0.092, 0.126] | P 0.109228 [0.092314, 0.126142] | OK |
| 65 | Table 2 | hard: −0.002 to 0.010; lower bounds < 0 | P hard points −0.000332 … 0.009670; lowers −0.001209 … −0.006317 | OK |
| 66 | §5.1 l.290 | accuracy 0.832 / 0.831 / 0.822 / 0.828; constant 0.749 | INF Acc A 0.8318, B 0.8312, F 0.8219, F0 0.8278, const 0.7492 | OK |
| 67 | §5.1 l.290 | balanced accuracy 0.72–0.73 | U means 0.723 / 0.722 / 0.721 / 0.728 | OK |
| 68 | §5.1 l.291 | about half of high earners | U recall means 0.506 / 0.503 / 0.519 / 0.527 | OK |
| 69 | §5.1 l.292 | lower bound −0.015 | P −0.015201 | OK |
| 70 | §5.1 l.292 | seed 2 loses 2.0 points | P seed2 −0.020408 | OK (O13: "accuracy points") |
| 71 | §5.1 l.293 | retention passes narrowly | P lower 0.000641 | OK |
| 72 | §5.1 l.295–296 | 0.774 / 0.759; difference 0.015; seed 0 alone | P seeds 0.0486 / −0.0002 / −0.0039 | OK |
| 73 | §5.1 l.297 | F 0.539, F0 0.648 | INF | OK |
| 74 | §5.1 l.297–298 | 0.234 = 0.125 + 0.109, on every seed | P; F0−F seeds 0.113 / 0.106 / 0.109 | OK |
| 75 | §5.1 l.299–300 | linear 0.515, nonlinear 0.773 | PH R_L\|A 0.51509; R_NL\|A 0.77349 | OK (labelled post hoc; causal wording, fix 14) |
| 76 | §5.1 l.303 | hard 0.536–0.546 | INF A 0.5438, B 0.5441, F 0.5361, F0 0.5458 | OK |
| 77 | §5.1 l.305 | +0.005, 90 % [−0.003, 0.013] | PH R(A,hard)−R(F,centred) 0.00474 [−0.00335, 0.01283] | OK (labelled post hoc) |
| 78 | §5.1 l.306 | about one point more accurate | PH Acc(A)−Acc(F) 0.00992 [0.00703, 0.01281] | OK (accuracy) |
| 79 | §5.1 l.308–309 | A 0.816, B 0.817, F 0.550, F0 0.646 | INF feat+out full 0.8164 / 0.8172 / 0.5501 / 0.6460 (aliased from OAR) | OK (fix 9) |
| 80 | §5.1 l.309 | offset changes nothing | S S-offset-out-* within ±0.003, intervals include 0; INF feat+out fullbank = iobank | OK |
| 81 | §5.1 l.310 | bypass 0.788 / 0.789 | INF feat+clean F 0.7878, F0 0.7891 | OK |
| 82 | §5.1 l.312 | 4 of 10 predictions correct | USEFUL_HEAD_COMPARISON §5 | OK |
| 83 | §5.2 l.318–319 | 0.645 vs 0.550 vs 0.812 | OAR RESEARCH_DECISION §3 | OK |
| 84 | §5.2 l.319 | 0.109 | row 64 | OK |
| 85 | §5.2 l.321 | accuracy 0.974 vs 0.975, constant 0.277 | ODX S5_SUMMARY Acc(F) 0.97431, Acc(A) 0.97476, const 0.27675 | OK |
| 86 | §5.2 l.321 | FARE 0.151 below LEACE | ODX S5-LEACE-minus-FARE 0.151224 | OK |
| 87 | §5.2 l.322–323 | 0.0074 [0.0019, 0.0129]; 0.02 target | ODX S5-FZ-minus-FARE 0.007372 [0.001888, 0.012856] | OK |
| 88 | §5.3 l.338 | full 0.917 vs 0.845 | ODX MULTI_RECIPIENT pair 0.917, income 0.845 (S4 +0.0727) | OK |
| 89 | §5.3 l.338 | centred 0.883 vs 0.792 | pair 0.883, employment 0.792 (S4 +0.0915) | OK |
| 90 | §5.3 l.338–339 | hard 0.611 vs 0.576; +0.034; lower 0.023; weakest of six | S4 hard pair−employment 0.034404, lower 0.023375; the minimum of 6 lowers | OK |
| 91 | §5.3 l.338 | > 0.02 AUC under every format | S4 all 6 lower bounds ≥ 0.0234 | OK |
| 92 | Fig. 3 bars | left FH full / prob / centred / hard; right S4 values | OSR FH rows; `figures.py` hard-codes 0.845 / 0.775 / 0.917, 0.731 / 0.792 / 0.883, 0.546 / 0.576 / 0.611, matching MULTI_RECIPIENT | OK |
| 93 | §6 l.345 | LEACE accuracy within 0.001 | P point −0.0006, lower −0.0016 | **MISMATCH (point presented as a bound)** (fix 16) |
| 94 | §6 l.346 | 0.759 vs 0.774 | INF | OK |
| 95 | §6 l.347 | 0.648; 0.539 | INF | OK |
| 96 | §6 l.355–357 | 0.648 vs 0.544; gap about 0.10 AUC | INF; 0.6482 − 0.5438 = 0.104 | OK (post-hoc label missing, fix 13) |
| 97 | §6 l.359–360 | 0.109 AUC on every seed | row 74 | OK |
| 98 | §7 l.366 | three encoder seeds | all sources | OK |
| 99 | §7 l.375 | about 0.2 CPU-hours | private LEDGER.jsonl: 608.9 CPU-s = 0.169 CPU-h over 88 units (verification excluded) | OK numerically; **UNSOURCED in the committed package** (O15) |
| 100 | §4 l.191 | recovery targets 0.02 AUC | CAP, ODX, OAR protocols | OK |

**Units check.** Every AUC difference is expressed in AUC; none is called "points". Accuracy differences are called
"points" at l.42, l.116, l.236, l.291–292, l.306 and l.346–361; all are accuracy. "2.0 points" (l.292) is accuracy by
context (O13).

## Citations

- **Key resolution.** All 25 distinct keys cited via `\citep`/`\citet` exist in `references.bib`. One entry,
  `elazar2021amnesic`, is uncited; CLOSEST_PRIOR_ANALYSES says it is kept only because a reviewer named it. BibTeX
  gives one harmless warning (`gupta2021fcrl` volume+number).
- **Claim-by-claim check against CLOSEST_PRIOR_ANALYSES.md:**

| Citation (line) | Claim in manuscript | CLOSEST_PRIOR says | Verdict |
|---|---|---|---|
| ravfogel2020inlp, ravfogel2022rlace (l.124) | "Linear concept erasure guarantees that no linear classifier can recover a concept" | INLP §6.1.1 and R-LACE §5.1 report nonlinear recovery; a guarantee is attributed to LEACE only | **Over-attribution** (fix 11) |
| belrose2023leace (l.124–127) | closed form, minimal change; restricts claim to linear adversaries | §7 limits scope to linear erasure | OK |
| ravfogel2020inlp / ravfogel2022rlace (l.126) | report nonlinear recovery after linear erasure | INLP §6.1.1 MLP 85 %; R-LACE §5.1 > 90 % | OK |
| ravfogel2022kernel (l.127) | does not transfer across kernels | §5.2, Table 2 | OK |
| elazar2018adversarial, song2020overlearning (l.129) | post-hoc attackers recover what in-training adversary could not | Tables 3–4; Tables 2–3 | OK |
| moyer2018invariant (l.130) | independent post-hoc adversaries a standard check | post-hoc adversaries, §5 Fig. 1 | OK ("standard" is mild) |
| fredrikson2014pharmacogenetics, fredrikson2015model (l.133) | attacks exploit confidence scores | D3 partly, confidence values | OK |
| mehnaz2022sensitive (l.134) | label-only can match confidence-based | §1 "on par" | OK |
| choquettechoo2021label (l.135) | masking confidences does not stop label-only membership inference | D4 partly | OK |
| wang2019balanced (l.135–136) | prediction leakage vs label leakage at matched performance | D2 "yes, in a specific sense" | OK |
| ravfogel2023loglinear (l.137) | downstream classifier on a linearly guarded representation can still reveal the concept | binary log-linear heads cannot; a constructed multiclass softmax head can | **Over-generalised** (fix 12) |
| jayaraman2022imputation (l.138) | many black-box AI results do not exceed imputation | Table 1 | OK |
| fredrikson2015model (l.139) | rounding confidences as countermeasure | §6 | OK |
| guo2017calibration (l.140) | calibration unchanged under prob release | textbook use | OK |
| goodfellow2016deep §4.1 (l.102) | softmax shift invariance | §4.1 | OK |
| zhao2019inherent, zhao2022inherent (l.144) | fair representations trade accuracy for protection | DP lower bound on group-wise error when base rates differ | Loose ("protection" should be DP) (O8) |
| stadler2024lpl (l.145) | a useful release reveals what the label reveals | Fig. 3 | OK |
| madras2018laftr (l.146) | LAFTR lacks guarantees against unseen classifiers | evaluated by downstream fairness gaps; no such statement | Not supported by the review file (O7) |
| gupta2021fcrl (l.146) | common downstream protocol | D2 partly | OK |
| jovanovic2023fare (l.147–149) | finite partition; certifies DP of cell-only classifiers; notes information loss; k-means comparison | §5, §6, App. C | OK |
| ganta2008composition (l.153) | combining independent releases breaches | D6 yes | OK |
| tian2025fairnessprivacy (l.154) | biased + debiased versions raise attribute inference | §6.3, Table 5 | OK |
| taylor2026collusion (l.155) | collusion-aware mechanisms for sequential finite-alphabet releases | D6 theory | OK |
| Position paragraph (l.157–161) | gap in a bounded search, not priority | §4 "gap in a bounded reading, not evidence that we are first" | OK, no priority claim |

## Anonymity and privacy

Searched `main.tex`, `fig_recipient.tex`, `references.bib` and `pdftotext main.pdf` for author or user names, emails,
`/Users`, `~/`, GitHub, venue or review labels (AAAI, NeurIPS, FAccT, "reviewer"), row or instance IDs (`canon_key`,
`i-0…`) and private paths.

| Item | Result |
|---|---|
| Author line | "Anonymous authors" |
| PDF metadata | Author, Title, Subject and Keywords empty |
| Names | Only third-party cited authors (bibliography) |
| Emails, private paths, row IDs, review text | None in .tex, .bib or PDF text |
| "anonymous repository" (l.374) | Fine |
| "PCRL" | Appears only in figures: Fig. 1 node "historical clean PCRL task logits"; Fig. 3 title "Frozen PCRL heads". Undefined in the text and names the authors' earlier system (fix 19). |
| `references.bib` header comment | Internal role text; not rendered (O16) |

## Structure

**Sections:**
1. Introduction
2. Setting
3. Related work
4. Evaluation protocol
5. Results (5.1 Matched useful-head comparison, 5.2 How much of FARE's effect is compression?, 5.3 Combining
   recipients)
6. Discussion
7. Limitations, reproducibility, ethics and LLM assistance

This matches the required order.

**Figures and tables:**
- **Recipient diagram.** Exactly one: Fig. 1, `fig_recipient.tex`. It shows the permitted paths (1) and (2) solid and
  the bypass (3) dashed red (O9 on the bypass label).
- **Other figures and tables: 4.** Table 1 (summary), Fig. 2 (useful head), Table 2 (primary endpoints), Fig. 3
  (formats and pairs). This is within 3–4.
- **Cross-references.** Sections 3, 4, 5.1 and 5.1–5.3, Figs. 1–3 and Tables 1–2 all resolve. No "??" in the PDF text
  or page images.

## Overclaiming (sentences claiming more than their evidence)

1. **l.298–301.** "A post-hoc diagnostic suggests why ... the remaining signal in the score is nonlinear, outside the
   scope of a linear eraser." This is causal and mechanistic language from a post-hoc scalar diagnostic (fix 14).
2. **l.40 and l.239.** "Low recovery from hard decisions often reflects decisions that say little" / "A decision that
   says little also reveals little." These are causal and general. They are contradicted in part by the useful heads,
   whose useful decisions still recover only 0.536–0.546 (fix 15).
3. **l.352–353.** "Part of the earlier 'decisions leak little' pattern came from decisions that barely beat a constant."
   This attributes a cause to a cross-study comparison (fix 15).
4. **l.70.** "A single evaluation protocol applied to the same stored models". Roles, exposure handling and bootstrap
   differ across the four stored-model studies (fix 5).
5. **l.243–244.** "it was absent for multiclass heads". This generalisation is falsified by HMDA pricing/race (fix 8).
6. **l.228–229.** "FARE's native certificate was unavailable or vacuous". This overgeneralises: the Adult bounds of
   about 0.50 are weak but not vacuous, and one bound is informative (fix 17).
7. **l.123–125.** INLP and R-LACE described as guarantees (fix 11). **l.137:** log-linear guardedness generalised
   beyond multiclass heads (fix 12).
8. **Abstract (ii), l.361.** "any defense is bypassed by an unchanged output". The logic is sound, but the measured
   recovery is shown only for noise and FARE on two cells (O6).
9. **Abstract (iv), §5.3.** "Two recipients pooling their outputs recover more than either alone". This is one Adult
   pair, one attribute and three seeds, and it holds on the mean (seed 2 full-logit difference is 0). It needs scope
   (O6).
10. **"Four findings recur" (l.35).** Finding (iii)'s useful-head result is a single cell. "recur" suggests replication.
    Consider "Four findings emerge".
11. **Priority.** None: the Position paragraph is correctly bounded (l.161).

## Page-by-page inspection log

PNG renders are at 60 dpi in `scratchpad/mr/p-01…p-10.png`, with 150/200 dpi crops of Figs. 1–3 and Table 1.

| Page | Content | Observations |
|---|---|---|
| 1 | Title, "Anonymous authors", abstract, §1 opening, Questions | Clean. The abstract is dense but has no layout issues. "pretrained" appears only negated (PDF line 41). |
| 2 | Fig. 1 (top), Contributions, Scope, §2 Stored models, Recipients, start of Feature arms | Fig. 1 is legible at 200 dpi; no overlaps. Node text "historical clean PCRL task logits" (fix 19). The dashed path (3) carries only logits while its label says "features + clean logits" (O9). The bypass label sits close to the recipient box but does not touch it. Section references resolve (Section 4, Section 5.1, Section 3). |
| 3 | Feature arms (end), §3 Related work (all paragraphs), §4 Roles | Clean. Citation numbers resolve [1]–[25]. |
| 4 | Table 1 (top), §4 Attackers, Utility, Native checks, Inference, §5 opening, Native checks paragraph | Table 1 fits the text width. The first column wraps onto two lines in two rows ("…all 14 / pairs)", "…bound met / on all 60 maps)"), which is acceptable. The "Full logits vs hard decision" cells are long but fit. No overfull box (`main.log`: 0 Overfull or Underfull). |
| 5 | Fig. 2 (top), end of Native checks, Outputs and the bypass, Usefulness, Offset paragraph, §5.1 Design, Utility start | Fig. 2 is readable: 8 pt font at text width, axes labelled, legend clear. In the hard panel the A and B large markers overlap (O10). Seed markers are faint (alpha 0.35). Caption "90% intervals" are marginal (O2). Fig. 2 is placed in §5 before §5.1, where it is first referenced, which is fine. |
| 6 | Table 2 (top), Utility (end), Recovery from useful scores, useful decisions, Complete release, Pre-registered predictions, §5.2 | Table 2 is aligned and the "n.e. (excl. 0)" note is clear. No issues. |
| 7 | Fig. 3 (top), §5.3, §6 Discussion | **Fig. 3 right panel: the legend ("income alone / employment alone / pair") overlaps that panel's y-axis spine** (fix 21). The right panel has no y-label. Bars start at 0.45 and have no intervals (O11). The left panel's legend sits clear of the bars. Title "Frozen PCRL heads" (fix 19). |
| 8 | §7 Limitations, Reproducibility, Ethics, LLM assistance; References [1]–[7] | Clean. |
| 9 | References [8]–[19] | Mehnaz title "private? novel" lowercase (O16). URLs wrap acceptably. |
| 10 | References [20]–[25]; rest of page blank | Clean. Tian 2025 is shown as an arXiv preprint, consistent with UNVERIFIED journal details. |
