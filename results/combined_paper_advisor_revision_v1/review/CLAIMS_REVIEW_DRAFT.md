# Claims review draft (phase 2): revised combined manuscript

**Role.** Claims and provenance reviewer, 2026-10-16. Suggestions only; no manuscript, table or brief was edited. No
priority or novelty claim is suggested.

## What was checked

| Object | Version | Notes |
|---|---|---|
| `papers/combined_empirical_v2/main.tex` | `a1c1540db` (SHA-256 prefix `2927435a`, 725 lines; equal to the working copy at review time) | Requested at `665ca13ed`, then at `0a4c36e02`. `a1c1540db` differs from `0a4c36e02` only in float settings (7 lines added at the top), Figure 1 labels and one clause at l.180. **All `l.` numbers below are lines of `a1c1540db`.** |
| `references.bib`, `fig_recipient.tex` | `a1c1540db` | — |
| `CLAIM_SOURCE_CHECK.md` | working copy, SHA-256 prefix `7b5e6f71`, rows 1–44 (uncommitted at review time) | The committed `665ca13ed` copy (prefix `c77f3706`) still had the pre-correction values in rows 2, 10, 14 and 15; those are now fixed. |
| `ADVISOR_BRIEF.tex`, `METHOD_SCOPE_PROPOSAL.md` | `a1c1540db` | — |
| `PROBLEM_SPECIFICATION.md`, `ADVERSARY_AND_MOTIVATION.md` | `c9709a29863a70ef2e88b5e53a8a6e20413e03a0` (`publications/pcrl-mechanism-design-v1/results/`) | — |

**Sources** are as in `CLAIM_SOURCE_INDEX.md` §1:
- MRB = `results/combined_matched_removal_benchmark_v1/` @`70f978ffc` (Experiment 1);
- OAR = `results/combined_output_aware_removal_v1/` @`f7425b15c` (Experiment 2);
- ODX = `results/combined_output_diagnosis_v1/` @`dc6a2e961` (Experiment 3);
- CAP = `results/combined_analysis_paper_v1/` (Experiment 4; results at `90253ed39`, final report at `5150f42e8`).

Code citations were read in this worktree. As in phase 1, only aggregate tables, reports, locks, protocols and code were
read.

**Severity:**
- **MUST-FIX**: false as written against the stored record, or contradicted elsewhere in the paper.
- **SHOULD-FIX**: misleading, imprecise or missing a qualifier that the record requires.
- **NOTE**: optional.

**Counts:** 4 MUST-FIX, 12 SHOULD-FIX, 13 NOTE. Of D1–D35, 32 are resolved (D7, D12 and D16 each with a residual item)
and 3 are partly resolved: D17, D21 and D29 (see §B).

**Confirmed without change** (spot-checked, every number against its source):
- the abstract's numbers;
- Tables 1–3, apart from the items below;
- the LRT description and values (item (f) of the request);
- the data, purpose, role and support statements (request item (a)), apart from item 3;
- the HMDA FARE configuration facts (request item (b)), apart from item 8;
- the family sizes and the definition of "registered" (request item (c));
- the prediction breakdown (request item (d));
- the S4 sentence (request item (e));
- all citation sentences re-scoped since phase 1, apart from item 12.

---

## A. Items

### MUST-FIX

**1. Post-run changes are miscounted and one is misattributed (l.454–459; also D17).**

> "Three changes were made after runs had begun, and each is recorded with its protocol: a guard in Experiment~2's FARE
> certification code …; a scoring repair to the easier-task comparison of Experiment~3, … from 0.0001 to 0.0019 without
> changing its verdict; and reporting repairs in Experiment~3 that changed no estimand, selection, role or verdict."

What the record shows:
- Experiment 2's lock was amended (L1, a map-identifier fix in `oar/study.py`) after its Adult run and one failed HMDA
  attempt, before any HMDA fit (OAR `ACTIVITY_LOG.md` l.25–30; OAR `RESEARCH_DECISION.md` l.12–13).
- Experiment 2's report script was fixed after an inference crash; the script is outside the lock (OAR `ACTIVITY_LOG.md`
  l.34).
- Experiment 3's lock was amended (L1, adding the easier-task code) after its mandatory stages, before any easier-task
  fit (ODX `VALIDATION.md` l.10; ODX `RUN_STATUS.json` l.13).
- The "reporting repairs" (amendment R) repaired **Experiment 2's** code and records: role-label truncation, the A1 count
  field, and two HMDA bounds in Experiment 2's decision file. They were made at the start of Experiment 3, before its fits
  (ODX `AMENDMENT_R_2026-10-03.md` l.1–13; commit `351116705`, "Stage 0").

Replacement:

> "Changes made after an experiment's runs had begun are recorded with its protocol. Two lock amendments preceded the
> fits they affected: a map-identifier fix in Experiment~2, made after its Adult run and before its HMDA fits, and the
> addition of the easier-task code in Experiment~3, made before those fits. Two changes followed results: a guard in
> Experiment~2's FARE certification code that identifies duplicate rows by record rather than by feature values, without
> which every certificate was unavailable; and a scoring repair to the easier-task comparison of Experiment~3, found by
> the independent replay, which applied the registered rule for candidates that ignore one part and moved that
> endpoint's lower bound from 0.0001 to 0.0019 without changing its verdict. Experiment~2's report script was also fixed
> after a crash, and its reporting records were repaired at the start of Experiment~3; neither changed an estimand,
> selection, role or verdict."

Mirror the change in `CLAIM_SOURCE_CHECK.md` row 26a.

**2. The measured-gap reasoning compares two different heads (l.682–686).**

> "…the lowest score-level recovery among the evaluated treatments is the zero-fairness tree's 0.648, while the
> untreated decision, which has the same accuracy and no scores, recovers 0.544. Because a decision is a function of its
> score, this gap of about 0.10 AUC measures how much more the scores reveal about the attribute than their thresholded
> decisions do, in this setting."

Why it is wrong:
- 0.648 is the tree's score and 0.544 is the untreated head's decision. The untreated decision is not a function of the
  tree's score, so the "function of its score" argument does not apply to this gap.
- "Reveal" is an information statement, which §3 ("What recovery measures", l.241–249) disclaims.
- The within-head comparison exists in committed data, and it also gives about 0.10. The tree's own decisions recover
  0.546 (CAP `USEFUL_HEAD_COMPARISON.md` l.80; also 0.5438 + 0.0020 from `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` l.18 and
  `report/posthoc_diagnostics.json` l.28, l.43), so its score adds 0.102.

Replacement:

> "…while the untreated decision, which has the same accuracy and no scores, recovers 0.544. The tree's own decisions
> recover 0.546, so for our attackers its scores add about 0.10 AUC over their thresholded decisions, in this setting."

Keep the next two sentences unchanged.

**3. The training split is mislabelled (l.131–132 and l.397–398; D21 residue).**

> "fits defenses and heads on the encoders' training split (24{,}127 Adult and 63{,}704 HMDA records)"; "the encoders'
> own training split of 24{,}127 Adult and 63{,}704 HMDA records"

- The training split has **24,145** Adult and **63,747** HMDA records.
- `defense_fit` excludes the 18 and 43 training records whose key also occurs in the test split, leaving 24,127 and
  63,704 (MRB `PROTOCOL.md` l.85, l.89; MRB `notes/admission/rows_provenance.json`: `n_train` 24145 / 63747,
  `train_rows_whose_key_occurs_in_test` 18 / 43).
- My phase-1 D21 wording ("the encoders' own training split (24,127 / 63,704)") was imprecise and is the likely origin.

Replacements:
- l.131–132: "…on the encoders' training split, without the 18 Adult and 43 HMDA training records that also occur in the
  test split (24{,}127 and 63{,}704 records)…"
- l.397–398: "…the encoders' own training split without its 18 and 43 records that also occur in the test split, 24{,}127
  Adult and 63{,}704 HMDA records…"

Mirror the change in `CLAIM_SOURCE_CHECK.md` row 38. The test-split sizes, 15,060 and 13,661, are correct (same JSON,
`n_test`).

**4. Not all frozen heads are weak (l.144–145).**

> "The frozen heads are weak predictors (Section~\ref{sec:frozen}), which is why Experiment~4 fits useful heads."

- True for the two primary settings and for HMDA fair-lending.
- The employment, education and pricing frozen heads gain 0.69, 0.63 and 0.37 over the constant (ODX `S3_ENDPOINTS.csv`
  l.19, l.20, l.34; ODX `RESEARCH_DECISION.md` l.29–31). Usefulness passed for 4 of 6 purposes (l.68).

Replacement: "The frozen heads of the two primary settings are weak predictors (Section~\ref{sec:frozen}), which is why
Experiment~4 fits useful heads."

### SHOULD-FIX

**5. "Exploratory" is used in two senses (l.52, l.119).** The abstract says "All results are exploratory evidence", and
l.119 says "exploratory development evidence".
- Elsewhere the paper uses "exploratory" in its narrow sense: values outside registered families, with unadjusted
  intervals (l.264, l.362, l.548–549). It also calls the experiments "registered" (l.451–453).
- `CLAIM_SOURCE_CHECK.md` uses the same vocabulary.
- Calling all results exploratory blurs registered vs exploratory with development vs confirmation.

Replacements:
- l.52: "All results are development evidence on data used by earlier analyses, registered endpoints included, under a
  declared threat model; …"
- l.119: "…so even the registered results are development evidence rather than fresh confirmation…"

**6. "For each" overstates coverage (l.87–90).**

> "For each, we first apply each method's own protection check and then measure recovery from … and the outputs of two
> recipients linked by record."

The pair and the useful heads are Adult only (Table 3). Replacement: "Across them, we first apply each method's own
protection check and then measure recovery from the surfaces a recipient receives under the evaluated contracts
(Table~\ref{tab:access}): …"

**7. The scope of the probe and the own head is inconsistent (l.149–152, l.155–156, l.91–93, l.426–429, Figure 1
caption l.172–173; D12/D29 residue).**

The current text:
- "(Experiments~2 and~4)" (own head);
- "it measures utility in Experiments~1 and~2" (probe);
- "(1) a task output only, from the frozen head in Experiments~1--3";
- "in Experiments~1--3 the released output is that of the frozen head";
- "In Experiments~3 and~4 we report … of the deployed head".

What the record shows:
- Experiment 3's easier task uses own heads: complete release = features plus own head, and the own-head output alone
  (ODX `PROTOCOL.md` l.210; ODX `FARE_USEFUL_TASK_RESULTS.md` Part B). Its utility is the probe's (l.629; ODX
  `PROTOCOL.md` l.210).
- Experiment 3 also has registered probe-usefulness endpoints (ODX `PRIMARY_ENDPOINTS.csv` l.3, l.9) and a refitted own
  head (Table 3, l.301).
- Experiment 2 released the own-head output alone (Table 3, l.293–294).

Replacements:
- "(Experiments~2--4)";
- "it measures utility in Experiments~1 and~2 and on Experiment~3's easier task";
- "(1) a task output only: from the frozen head in the frozen-head analyses of Experiments~1--3, and from an own head in
  Experiment~4, in Experiment~2's own-head surfaces and on Experiment~3's easier task";
- Figure 1 caption: "Path (1) is drawn for an own head; the frozen-head analyses of Experiments~1--3 release the frozen
  head's output, which reads the untreated features";
- Utility paragraph: "…of the deployed head (the frozen heads in Experiment~3, the own heads in Experiment~4). The
  common refitted probe measures utility in Experiments~1 and~2 and on Experiment~3's easier task…";
- l.91–93: "…in Experiments~3 and~4 mostly the utility of the head whose output is released, and otherwise that of a
  refitted probe".

**8. The HMDA gain is the probe's, from a replay (l.183; request item (b)).**

> "On HMDA the untreated task gain over the constant is only 0.6 to 2.3 points"

The cited table, ODX `FARE_USEFUL_TASK_RESULTS.md` l.16–23:
- is Experiment 3's Part A replay ("replay of the existing Adult and HMDA FARE results", l.9–11, from saved predictions)
  of **Experiment 2's primary-cell nominees**;
- defines "Task gain = U2 refitted-probe accuracy minus the attacker_fit-majority constant" (l.13);
- has HMDA gains 0.0202 / 0.0061 / 0.0231 (`FARE_REPLAY_EXISTING.csv` l.14, l.18, l.22).

The cell counts 4 / 1 / 4, the constant release on seed 1 and the 0.5 recovery are confirmed (`FARE_REPLAY_EXISTING.csv`
l.16, l.20, l.24; OAR `RESEARCH_DECISION.md` l.100–106). The frozen head's HMDA gains are 2.1 / 0.6 / 2.0 (ODX
`RESEARCH_DECISION.md` l.20).

Replacement: "On HMDA the untreated probe's accuracy gain over the constant is only 0.6 to 2.3 points". Cite OAR
`RESEARCH_DECISION.md` l.100–106 for the nominees.

**9. Noise did run on secondary pairs (Table 3, l.286 and l.290; l.390–391).**

> "noise: the two primary settings"; "Untested: … noise on the secondary pairs (stopped by a budget rule; not reported)"

Noise ran on three secondary Adult pairs in full and on a fourth in part: 201 of 648 units (MRB `RESEARCH_DECISION.md`
l.20, l.29–34). Eight pairs were not run.

Replacements:
- "noise: reported for the two primary settings";
- "Untested: …, noise on eight of the twelve secondary pairs (a budget rule stopped the run; the four Adult pairs that ran
  are not reported)".

**10. The cell-conditional attacker "matched" claim has no committed support for Experiments 3–4 (l.420).**

> "where both were fitted it matched the nonlinear attacker"

- Committed values exist only for Experiment 2. They differ from the nonlinear attacker by 0.000 to 0.0027 (OAR
  `EXPLORATORY.csv` l.4–5, l.26–27, l.35–36, l.78–79, l.87–88).
- Table 3 says the attacker was also fitted in Experiments 3 and 4 (l.305, l.311–312). Their values are not in committed
  tables (CAP `INDEPENDENT_VERIFICATION.json` l.295: "CC.joblib … not reloaded").

Replacement: "…we report it descriptively; in Experiment~2, where both were fitted, it was within 0.003 of the nonlinear
attacker."

**11. Two directional claims about the pair are unsupported (l.655–658).**

> "…which makes the pair estimate conservative for that seed"; "…so the comparison, if anything, understates the pair's
> increment"

- The bank chose the income-only attacker by validation loss. The unselected pair attacker's assessment AUC for seed 2
  is not in any committed table, so "conservative" is not established.
- The format asymmetry favours the single recipients in selection. Whether it lowers the measured increment on
  assessment is not measured.

Replacement: delete "which makes the pair estimate conservative for that seed"; write "Under the full and centred
contracts each single-recipient comparator chooses among formats on validation, while the pair attacker sees only the
contract's format; this asymmetry favours the single recipients in selection."

**12. LAFTR does have a guarantee, conditional on an optimal adversary (l.362; pre-existing, now checked against the
paper).**

> "Adversarial methods such as LAFTR~\citep{madras2018laftr} lack guarantees against unseen classifiers"

LAFTR (PMLR 80, PDF read):
- §1 contributions: objectives "that upper bounds the unfairness of arbitrary downstream classifiers in the limit of
  adversarial training".
- §5 "Theoretical Properties": "Given an optimal adversary trained to maximize (8), the adversary's loss will bound
  ∆DP(g) from above for any function g learnable from Z."

Replacement: "LAFTR~\citep{madras2018laftr} bounds the unfairness of any downstream classifier only for an optimal
adversary".

**13. The output-surface numbers in §6.2 come from two experiments, unlabelled (l.513–518).**
- 0.177 and 0.236 are Experiment 1 values: MRB `MATCHED_COMPARISONS.csv` l.468, l.660, exploratory, unadjusted 90 %
  intervals [0.166, 0.188] and [0.227, 0.245].
- 0.776 and 0.768 are Experiment 2 values (OAR `EXPLORATORY.csv` l.2, l.54).
- The Stadler sentence (l.361–362) cites Experiment 2 for the same label-only comparison: +0.174 / +0.246, OAR
  `EXPLORATORY.csv` l.41, l.93.

Replacement: "…by 0.177 on Adult and 0.236 on HMDA in Experiment~1 (exploratory; unadjusted 90\% intervals [0.166,
0.188] and [0.227, 0.245]) … because the clean output alone already recovers 0.776 on Adult and 0.768 on HMDA
(Experiment~2)."

**14. `CLAIM_SOURCE_CHECK.md` needs corrections.** Details are in §C. Rows 26a and 38 must change together with items 1
and 3.

**15. The advisor brief quotes only the second attacker (`ADVISOR_BRIEF.tex` l.41–43).**

> "two exploratory attackers that know the noise scale recovered more … (Adult 0.563, HMDA 0.575, against 0.528 and
> 0.513)"

These are the values of the second attacker only, the one fitted on clean auxiliary representations, which exceeds the
access. The first attacker recovered 0.546 / 0.526 (MRB `RECOVERY.csv` l.7101, l.7105, l.10199, l.10203).

Replacement: "…(the first, within the evaluated access, 0.546 on Adult and 0.526 on HMDA; the second, fitted on clean
auxiliary representations that no contract releases, 0.563 and 0.575; the slate 0.528 and 0.513; exploratory, no paired
test)."

**16. Three corrections to `METHOD_SCOPE_PROPOSAL.md` (l.45–63).**
- **(i) Sources of the access bullet.** It is attributed to `PROBLEM_SPECIFICATION.md`. But "random household splits"
  and "household and person identifiers are never reader features" come from `ADVERSARY_AND_MOTIVATION.md` l.89 and l.92
  @`c9709a29`; "one fixed … release per person" comes from the same file and from COR `CORRECTIONS.md` §6. Add that
  source.
- **(ii) The worsening condition.** "each recipient's worsening ≤ 0.010" should read "each recipient's worsening upper
  bound ≤ 0.010" (`PROBLEM_SPECIFICATION.md` l.103).
- **(iii) The stored-model bullet lists one set of targets.** The paper's targets are per experiment (main.tex
  l.444–450):
  - gain 0.03: Experiment 4 only;
  - retention: Experiment 4 and the easier task;
  - 0.02 AUC: the recovery contrasts of Experiments 2–4;
  - Experiment 1: level bars 0.55 / 0.05 / −1 point;
  - Experiment 3 frozen-head usefulness: 0.01.

  Experiment 3 also releases the offset and the (margin, offset) pair besides "the four output formats".

### NOTE

17. **Table 1 caption (l.472–473).** "Several other rows are registered levels or contrasts" omits the exploratory
    zero-fairness level (OAR `EXPLORATORY.csv` l.35, l.87), the descriptive per-seed row (Stage A) and the count row.
    Suggest: "Other rows are registered levels (Experiment 1), levels underlying registered contrasts (Experiment 2), an
    exploratory level (zero-fairness tree), descriptive per-seed points and a count."
18. **l.443–444, the pass rule.** "In every experiment an endpoint passes only if its lower bound exceeds its registered
    target." Experiment 1's level endpoints are three-way: established above, established below (using the upper bound)
    or unresolved (MRB `PRIMARY_ENDPOINTS.csv`, `decision`). A clause would cover it.
19. **l.530–531, generality of the offset effect.** "it depends on how the heads were trained" is untested. "It is
    relevant wherever a service returns raw logits" generalises an effect not established on 9 of 14 pairs. Suggest: "it
    is specific to these frozen heads and absent by construction for heads whose logits carry no free offset; services
    that return raw logits may want to check for it."
20. **l.627–628, interval type.** State that 0.096 [0.076, 0.116] is a simultaneous interval over Experiment 4's 33
    secondary endpoints (z = 3.172; `SECONDARY_COMPLETE_RELEASE_ENDPOINTS.csv` l.11).
21. **l.51, the abstract.** "two recipients granted perfect linkage recover more" would match §3 (l.241–242) better as
    "our attackers recover more from the pair".
22. **l.100–102.** "output-only recovery exceeds that of a label-only reference, so a defense applied to the features
    cannot be credited…" The bypass conclusion follows from the output alone recovering about 0.77 (§6.2), not from
    exceeding the label-only reference.
23. **l.310, Table 3 row 4.** "in four formats and two banks": Experiment 4 also released the (margin, offset) pair, so
    there are five formats (CAP `PROTOCOL_USEFUL_HEADS.md` l.56).
24. **l.116–118.** Label "the gap in recovery between useful scores and equally accurate decisions" as post hoc and
    descriptive, as §7 does (and see item 2).
25. **l.629, point vs bound.** Only point accuracies are given (0.974 / 0.975). The registered non-inferiority passed:
    −0.0004 [−0.0025, 0.0016] (ODX `S5_ENDPOINTS.csv` l.4). Adding the bound shows established, not only point,
    feasibility.
26. **l.63–64, the training constraint.** "were trained so that a linear probe's R² … stays below a threshold" would
    match l.141–142 ("enforced through a Lagrangian penalty") better as "were trained with a penalty that targets … at
    most a threshold".
27. **l.709–710, the public code statement.** "The encoders' public code still exposes …" holds on the local
    remote-tracking refs: `origin/main` `55e4cb1d1` defines `certified_accuracy_bound`, and the fix `5d4eda046` is not
    merged. The refs were not refetched; re-check before submission.
28. **`ADVISOR_BRIEF.tex` l.49.** "11 of 42 attribute--seed combinations": the paper now says "combinations of
    purpose–attribute pair and encoder seed".
29. **`CLAIM_SOURCE_CHECK.md` §D.** "no equivalence claim anywhere": the paper has descriptive "about as well / about as
    much" statements (l.45, l.97–98, l.506, l.608), each with its numbers. Suggest "no formal equivalence claim".

---

## B. D1–D35 (phase 1) at `a1c1540db`

| D | Phase-1 issue | Status | Evidence in `a1c1540db` |
|---|---|---|---|
| D1 | 12 of 42 → 11 | resolved | l.99, 478, 497, 666 |
| D2 | 0.774 → 0.773 | resolved | l.49, 529, 598, 602 (all four consistent) |
| D3 | 0.864 → 0.863 | resolved | l.483 |
| D4 | 0.607 → 0.608 | resolved | l.484 |
| D5 | "−0.000" | resolved | l.487 |
| D6 | 0.2 → 0.25 CPU-h | resolved | l.711 |
| D7 | Table 1 caption | resolved (NOTE 17) | l.469–473 |
| D8 | HMDA interval type | resolved | l.504–505 |
| D9 | ρ₁² points vs interval | resolved | l.501–504 |
| D10 | Experiment 1 bars | resolved | l.447–449 |
| D11 | Figure 2 caption | resolved | l.558–562; replay scope at l.718–719 |
| D12 | S5 probe accuracy | resolved at l.629 (item 7 for §2/§5) | — |
| D13 | like-for-like compression | resolved | l.626–631 |
| D14 | offset "changes nothing" | resolved | l.612–613 |
| D15 | 19 endpoints' formats | resolved | l.552–553 |
| D16 | generality | resolved (item 6 residue) | l.47–48, 51, 677 |
| D17 | post-run amendments | **partly** | disclosed at l.454–459, but miscounted and misattributed (item 1) |
| D18 | replay scope | resolved | l.717–720 |
| D19 | household identifiers | resolved | l.237–239 |
| D20 | attacker budgets | resolved | Table 3; l.229–231 |
| D21 | defense_fit = training rows | **partly** | disclosed; numbers mislabelled (item 3) |
| D22 | "usually knows" | resolved | l.72 |
| D23 | employment vs occupation | resolved | l.126, 302, 642, 648 |
| D24 | pair × seed | resolved | l.99, 497 |
| D25 | 60 maps | resolved | l.500 |
| D26 | 0.175 secondary certificate; A1 | resolved | l.507–510 |
| D27 | F0 description | resolved | l.186–187 |
| D28 | LRT attackers | resolved | l.235–237, 254–266, Table 3 caption |
| D29 | Figure 1 | **partly** | labels fixed; the caption's Experiments 1–3 statement is inaccurate (item 7) |
| D30 | typo | resolved | l.443 |
| D31 | Stadler | resolved | l.355–362 |
| D32 | AttriGuard | resolved | l.368–370 |
| D33 | Song / Elazar | resolved | l.336–338 |
| D34 | Li LLSE | resolved | l.372–373 |
| D35 | bibliography | resolved | `references.bib`: Tian (TDSC 22(5):4654–4670, DOI); Jia (pages 513–529, USENIX) |

---

## C. `CLAIM_SOURCE_CHECK.md`, row by row (working copy, prefix `7b5e6f71`)

**Confirmed as written:** rows 2, 4, 7, 8, 9, 11, 12, 14, 15, 19, 20, 21, 23, 24, 26b, 26–35, 37, 39, 42, 43 and 44. The
code and JSON line citations in rows 26–37 were opened and match:
- `EFFECTIVE_PROTOCOL.json` l.255–272, 489–496, 522–525, 645–652;
- `stored_model_eval/recipes.py` l.71–90 and `bench.py` l.1045–1051;
- OAR `EXECUTION_LOCK.json` l.291, `oar/fare_run.py` l.25–47 and `oar/study.py` l.325–344;
- `odx/run.py` l.199–214;
- `pcrl/training/losses.py` l.475–518 and `pcrl/purposes/verification.py` l.62–109.

**Corrections needed:**

| Row | Problem | Correction |
|---|---|---|
| 1 | The status "registered (24)" fits the levels. The ≈ comparison (B−A) is exploratory, with unadjusted 90 % intervals (MRB `MATCHED_COMPARISONS.csv` l.396, l.588). The location "§1 ¶6" is ¶5 (main.tex l.96–106). The 60-maps source is MRB `VALIDATION.md` l.22. | "levels registered; B−A exploratory (unadjusted 90 %)"; location "§1 ¶5"; cite `PRIMARY_ENDPOINTS.csv` l.3–4, 15–16 for the levels |
| 3 | Location "§1 ¶6" | "§1 ¶5" |
| 5 | The limitation omits that the certificates exist only after post-run amendment A1, and that 0.175 is the secondary race {0,1,2} certificate (OAR `NATIVE_TEST_VS_RECOVERY.md` l.26–39) | Add both |
| 6 | The noise-plus-clean levels are registered Experiment 1 endpoints (MRB `PRIMARY_ENDPOINTS.csv` l.10, l.22, established above 0.55). Only the FARE-plus-clean levels are exploratory (with contrast P5 registered). | "registered (Exp. 1 levels; Exp. 2 P5); FARE+clean levels exploratory" |
| 10 | The per-seed values are descriptive Stage-A points, not part of the registered family of 30 | Split the status |
| 13 | The hard-decision levels are exploratory levels (OAR `EXPLORATORY.csv` l.4, l.56; P1 is the contrast). The fair-lending recoveries come from ODX `OUTPUT_SURFACE_RESULTS.csv` l.170, l.181, not `FROZEN_HEAD_UTILITY.csv`. | Fix the sources and status |
| 16–18 | "CAP (C13–C15)": the ledger gives the lock commit `a0de44953` for these rows | Cite `PRIMARY_USEFUL_HEAD_ENDPOINTS.csv` @`90253ed39`, as row 15 does |
| 22 | "CAP inference levels" is the private inference file | Cite `USEFUL_HEAD_COMPARISON.md` l.77–80; `report/posthoc_diagnostics.json`; OAR `EXPLORATORY.csv` l.17, 19, 29–33 (aliased units) |
| 25 | C22 cites the private levels | Cite `posthoc_diagnostics.json` l.38 (0.6482) and ODX `OUTPUT_SURFACE_RESULTS.csv` l.21 (0.5438); see also item 2 |
| 26a | Same miscount and misattribution as item 1 | Mirror item 1 |
| 36 | "equals NL where both were fitted" | "within 0.003 in Experiment 2 (OAR `EXPLORATORY.csv` l.4–5, 26–27, 35–36, 78–79, 87–88); Experiment 3–4 values not committed" |
| 38 | "training split 24,127 / 63,704" | "training split 24,145 / 63,747; defense_fit 24,127 / 63,704 after removing 18 / 43 records that also occur in the test split (MRB `notes/admission/rows_provenance.json`; MRB `PROTOCOL.md` l.85, l.89)" |
| 40 | The cited table is Experiment 3's replay of Experiment 2's nominees, and its gain is the probe's | Say so; cite OAR `RESEARCH_DECISION.md` l.100–106 and ODX `FARE_REPLAY_EXISTING.csv` l.14–24 |
| 41 | Cites the reviewer's index | Cite MRB `PROTOCOL.md` l.137 and OAR `PROTOCOL.md` l.146–150 |
| format | A blank line before row 26a ends the table, and rows 38–44 follow a paragraph without a header, so neither renders as a table. IDs 26a and 26b collide with row 26 of §B. | Renumber, and join the rows to their tables |

**Suggested text for the PENDING "Verification" line** (for the owner to record):

> "Checked by the claims and provenance reviewer on 2026-10-16 against the committed sources (this file at SHA-256
> prefix `7b5e6f71`): rows 2, 4, 7–9, 11, 12, 14, 15, 19–21, 23, 24, 26b, 26–35, 37, 39 and 42–44 confirmed; rows 1, 3,
> 5, 6, 10, 13, 16–18, 22, 25, 26a, 36, 38, 40 and 41 need the corrections in `review/CLAIMS_REVIEW_DRAFT.md` §C."

---

## D. Citation sentences (request item 3)

Checked against `CITATION_PRECHECK.md`:
- **Stadler** (l.355–362): every qualifier is present.
  - finite discrete data with full support;
  - a strictly positive label posterior;
  - the γ-level trade-off, stated for every attribute, with the label granted to the adversary;
  - the constructed attribute;
  - the single-attribute remark;
  - "an analogue of their label-only adversary";
  - "descriptive comparison, not a floor".

  The added Experiment 2 sentence is supported in both directions in both cells: OAR `EXPLORATORY.csv` l.40–41 and
  l.92–93; Adult hard −0.0885 [−0.0975, −0.0798], output +0.174; HMDA hard −0.0162 [−0.0246, −0.0092], output +0.246;
  exploratory 90 %.
- **Song / Elazar split** (l.336–338), **AttriGuard** (l.368–370), **Li LLSE** (l.372–373) and **Tian** (l.373–375):
  correct.
- **LEACE** (l.59–61, l.333–334) and **FARE** (l.61–62, l.363–365): correct.
- **Jayaraman** (l.347–348): correct. Its new use at l.216–217 (inputs would allow imputation) matches the paper's
  imputation adversary.
- **LAFTR** (l.362): item 12.
- **Bibliography:** both fixes are correct (Crossref TDSC record; USENIX pp. 513–529). The Mehnaz title capitalisation
  fix is fine.

## E. Distinctions (request item 4)

- **Point vs confidence-bound feasibility.** Kept. Table 2 verdicts use lower bounds; "not shown to stay / be within one
  point" (l.51, l.675); "within one point, established" (l.682). NOTE 25 is optional.
- **Unresolved benefit vs established shortfall.** Kept. "Not established … is not evidence of no effect" (l.450–451);
  "excl. 0" is defined (l.570–571); "smaller difference … below the registered 0.02 target" (l.631–632). No "no
  difference" wording was found.
- **No demonstrated difference vs equivalence.** No formal equivalence claim. The "about as well / as much" phrases are
  descriptive and give their numbers; see NOTE 29 for the table's §D.
- **Scope-limited guarantees vs attack-based evidence.** Kept: l.46–47, 505–507, 517–519, 668–670; "What recovery
  measures", l.241–249. Item 2 is the one sentence that slips into information language.
- **Development evidence vs fresh confirmation.** Kept: l.118–120, 700. Item 5 is the terminology clash.

## F. Advisor brief and scope proposal (request item 5)

**`ADVISOR_BRIEF.tex` @`a1c1540db`.** Every number matches the manuscript and sources:
- labelled budgets 5,388–7,571 / 2,082–2,239;
- 0.817 / 0.863 vs 0.815 / 0.870;
- 11 of 42;
- 0.76–0.79;
- 0.539 / 0.773 / 0.648;
- 0.611 vs 0.576.

Exceptions are item 15 and NOTE 28. The `665ca13ed` copy's 0.774 and its unscoped coalition sentence are already fixed.

**`METHOD_SCOPE_PROPOSAL.md` @`a1c1540db`.** These definitions match `PROBLEM_SPECIFICATION.md` @`c9709a29`:
- release contract (§2);
- 8,001 / 3,000 roles (§5);
- the pair bank ignoring either recipient (§3);
- 24 inclusive clauses: 2 tasks × U0 / Ucal × 3 metrics × 2 seeds, with caps 0.010 / 0.010 nats / 0.005 (§4);
- the within-SEX Brier guard ≤ 0.005 (§4);
- the claims list, with the 0.020 recipe and nested competitive margins (§6).

Exceptions are item 16 (i)–(iii). The "post-hoc gap" sentence in its open choice 1 should follow item 2 once it is
fixed.
