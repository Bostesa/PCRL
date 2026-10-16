# Independent reader report

Source read: `papers/combined_empirical_v2/main.pdf` only (14 pages, compiled 2026-10-16). I read every page as images and as a text layer and looked at the figures and tables enlarged. I opened no other file and used no outside knowledge of the project. Page numbers below are the printed page numbers, which match the PDF pages.

## Summary of my understanding

**Research question.** If a model serves several recipients, each allowed one purpose (for example an income prediction), how much can one recipient learn about a protected attribute it may not use (sex, race)? The recipient may receive a representation, a task output (full logits, centred logits, probabilities or a hard decision), both together, or outputs pooled by record with a second recipient. The paper also asks how useful that release is for the recipient's task.

**Threat model.** Each record is released once. The recipient also holds a labelled auxiliary sample: several thousand other records' releases from the same population, each with the protected attribute. It trains supervised attackers (logistic regression, boosted trees, MLP), selects them on a validation split, and is scored by macro AUC on held-out records. In the two-recipient case the pair is given perfect linkage. Access to inputs, model parameters, repeated or adaptive queries, insiders and imperfect linkage is out of scope.

**Main findings.** The study uses six previously trained encoders (Adult and HMDA, three seeds each) and four treatments: LEACE, FARE, Gaussian noise, and a FARE tree with the fairness weight set to zero.
- LEACE meets its linear guarantee, but nonlinear attackers recover the attribute as well as from the untreated features.
- The encoders' own linear constraint fails its check in 11 of 42 cases.
- Releasing the untreated model's output beside defended features undoes the feature defense.
- The frozen heads' hard decisions leak little, but they are barely better than a constant.
- With useful logistic-regression heads, FARE's lower score-level recovery is about half explained by compression alone, and FARE's accuracy is not shown to stay within one point.
- In the one coalition tested, two linked recipients recover more than either alone.

The authors present this as an exploratory measurement study, not a new method or guarantee.

**Where I got lost** (on first reading; most of it resolved on a second):
- what the "stored" models are and why their architecture looks the way it does;
- which head produces which output: the "frozen", "untreated", "historical clean", "own", "deployed" and "useful" heads, and the "common probe";
- which tasks "fair_lending", "pricing" and "employment group" are, and what the 14 pairs are;
- where the owner fits its defenses;
- what "registered", "previously used", "development evidence" and "retired bound" refer to;
- whether the HMDA FARE release, which recovers 0.501, is a constant release.

### Answers to the review questions

- **Does it make sense on its own?** Mostly, after two readings. The main message is clear. Many details depend on project vocabulary the paper never defines (items 3, 4 and 9).
- **Problem and question.** The motivation comes before the findings, in both the abstract and Section 1. The research question (pp. 1-2) is precise. The scenario is clearly marked as an illustration, perhaps too defensively (item 12).
- **Threat model.** Section 3 and Table 1 are the strongest part of the paper. For each experiment I can tell what the recipient receives, the labelled-sample sizes, how attackers are fitted, selected and scored, and what is untested. The gaps are:
  - where the owner's fitting data come from (item 2);
  - which head an "output" comes from (item 3);
  - why each surface is relevant. The relevance argument is general, not per surface: for example, why a recipient would receive features plus its own output, which adds nothing beyond the features.
  - The paper does say clearly why "a stronger attacker" alone is not a justification (p. 4).
- **Logic.** Interpretations are usually careful, often over-hedged. A few claims go beyond the evidence (items 8, 14 and 28), and some headline results hold by construction (item 7).
- **Tables and figures.** Every number I checked in the text matches the tables and figures:
  - abstract against Table 2;
  - 0.234 = 0.125 + 0.109;
  - each gain equals its accuracy minus 0.749;
  - the retention values, recomputed from the accuracies;
  - the per-seed offset means (0.072 and 0.067);
  - Figures 2 and 3 against Section 6.5 and Section 6.7;
  - the gap of 0.648 - 0.544 in Section 7;
  - z = 3.008 for 19 two-sided endpoints.

  Two differences computed from rounded means are off by 0.001, which is ordinary rounding. There is one broken internal reference and the "excl. 0" label is wrong (items 6 and 34). The kinds of interval change from one experiment to the next, and no caption states what the intervals generalize over.
- **Writing.** No em dashes; en dashes are used correctly for ranges and compounds. The main problems are many "X, not Y" disclaimers, repeated summaries, a few slogans, and sentences that read like an audit log (items 9 and 42-47).
- **Layout.** No "??" and no "[?]". Nothing runs into the margins. There are two small collisions in figures (items 48-49).

---

## MAJOR

**1. The models and data are not described well enough to interpret the numbers (p. 2 Section 2; p. 8; p. 12).**
- Quotes:
  - "We use six stored purpose-specific encoders from earlier work" (p. 2)
  - "a frozen, seeded, randomly initialised backbone, which is not pretrained, and a trained adapter per purpose" (p. 2); Figure 1 calls it "backbone + LoRA"
  - "secondary analyses use all 14 stored purpose–attribute pairs" (p. 2)
  - "the collapsed HMDA seed 1" (p. 8) and "one HMDA encoder is heavily collapsed" (p. 12)
  - "The Adult frozen head gains 3–5 accuracy points over a constant and recalls 14–22% of high earners" (p. 8)
  - "the HMDA fair_lending head predicts the same class for everyone" (p. 8)
- Problems:
  - Adult and HMDA are never described: source, HMDA year or region, sizes, label definitions, number of race classes, preprocessing.
  - The "earlier work" is not cited, even in anonymized form.
  - The architecture (a random, untrained, frozen backbone with LoRA adapters) is unusual, and nothing motivates it.
  - The 14 purpose–attribute pairs are never listed, yet results are reported for "fair_lending", "pricing", "employment group", "age group" and "six multiclass Adult pairs".
  - "Collapsed" is not defined.
  - The frozen heads are weak. A refitted logistic-regression head on the same features is about twice as useful (gain 0.083, p. 9). That makes me ask whether these models resemble anything a provider would deploy.
- Suggestion: add a "Data and models" subsection or an appendix table with:
  - each dataset's source, size and splits;
  - all 14 purpose–attribute pairs, with class counts and base rates;
  - the encoder architecture and training objective, with an anonymized citation and one sentence on why this architecture;
  - frozen-head accuracy next to a standard baseline (for example logistic regression on the raw inputs);
  - a definition of "collapsed" and how it affects the results.

  Then say explicitly that the conclusions concern these models, and discuss whether they carry over to well-trained or pretrained encoders.

**2. The owner's fitting data have no visible source; the published role sizes leave no room for them (p. 3; p. 5 Table 1; p. 6 "Roles").**
- Quotes: "B: official target LEACE [1] ... fitted on the owner's defense_fit role" (p. 3). Table 1, stored-check row: "(15,060; 13,661)".
- Problems:
  - In Experiment 1, attacker_fit + attacker_val + assessment = 7,571 + 2,239 + 5,250 = 15,060 (Adult) and 6,794 + 2,089 + 4,778 = 13,661 (HMDA). These are exactly the test-split sizes in the stored-check row.
  - In Experiments 2-4: 6,065 + 2,235 + 5,243 + 1,500 (certification) + 17 (excluded) = 15,060, and 5,388 + 2,082 + 4,764 + 1,385 + 42 = 13,661.
  - So defense_fit, which holds the LEACE maps, the FARE trees, the heads and a 20% head holdout, is not part of the test split in any experiment. For Experiment 1 no defense_fit is mentioned at all.
  - The reader cannot tell whether defenses and heads were fitted on the encoders' training records (whose representations may be distributed differently from test records), on a validation split, or on attacker roles.
- Suggestion: add a role table per experiment giving each role's source split, its size, which party uses it (owner or attacker), and which labels that party sees. Say where LEACE was fitted in Experiment 1.

**3. There are too many kinds of heads and outputs, and Figure 1 does not show the release that most results use (p. 2-3 and Figure 1; p. 7; pp. 8-11).**
- One object (the untreated model's original output, as far as I can tell) has five names:
  - "the untreated head's full logits" (Table 1)
  - "historical clean task logits (untreated)" (Figure 1)
  - "the untreated model's task output" (abstract)
  - "historical clean outputs" (Table 2)
  - "the original output" (p. 8)
- There are also these heads and probes: "frozen task head" (p. 2), "common refitted probe" (p. 7), "the head fitted on each treatment's features" (p. 7), "deployed head" (p. 7 and Figure 2), "useful head" (p. 9), and "own task head LR on arm features only" (Figure 1).
- Problems:
  - Figure 1 draws path (1), output-only release, from the refitted "own task head". The untreated clean logits appear only on the dashed path, "evaluated only as an adverse control".
  - Yet Table 2 (frozen-head rows), Sections 6.2-6.4, Section 6.7 and Figure 3 are all about output-only release from the frozen heads on the untreated representation. At first I could not tell whether frozen-head output-only release is a permitted release or the adverse control.
  - "Besides features alone, which we report for interpretation" (p. 2) means the headline LEACE result is about a surface Figure 1 does not draw.
- Suggestion: add one table of every head or probe: who fits it, on which role and which features, in which experiments, and whether its output is released or only used to measure utility. Redraw Figure 1 with:
  - features alone;
  - frozen-head output only (Experiments 1-3 and Section 6.7);
  - the refitted-head paths (Experiment 4);
  - the bypass.

  Use one name for the untreated output throughout.

**4. Terms, code labels and defined-later labels are used without definition (throughout).**
- "Output Contracts" is in the title. "release contracts" (p. 2), "probability-output contract" (p. 4) and "an output contract for particular frozen heads" (p. 9) also appear. The term is never defined.
- "official target LEACE [1] (concept-erasure 0.2.4)" (p. 3): "official" and "target" are not explained, and concept-erasure is not identified as the authors' reference package. The same goes for "official FARE", which has no version.
- "feature arm" (Figure 1) is used for what the text calls a "treatment".
- "adverse control" (pp. 2-3) is not standard.
- "bank" is used in Table 1 (p. 5) but defined only on p. 7.
- "cell" has two meanings: a FARE partition cell, and an experimental condition ("The primary cells are ...", p. 2; "depends on the cell", p. 11; "on this cell", p. 11).
- Code identifiers in typewriter font:
  - attacker_fit, attacker_val, defense_fit and assessment are used on pp. 3-4 but defined only on p. 6;
  - income_prediction (p. 2) and fair_lending (p. 8) are task names;
  - "underwriting", "pricing" (p. 9) and "employment group" (pp. 1 and 10) are never described.
- "60 maps" (p. 7) matches no other count in the paper. "supported classes" (p. 7), "collapsed" (pp. 8 and 12) and "easier task chosen by a validation-only screen" (p. 10) are not defined.
- "decision" means both a hard-label output and an endpoint verdict: Table 3 has a "Decision" column, and p. 6 says "its decisions were stable when they were removed".
- Suggestion: define each term at first use, or add a short glossary. Replace code identifiers with plain names ("attacker training split", "owner fitting split"). Reserve "cell" for FARE and "verdict" for endpoints.

**5. Utility is missing for most rows of Table 2, although the paper says recovery must be read beside utility (p. 7; p. 8; p. 12).**
- Quotes:
  - "Experiments 1 and 2 report the accuracy of a common refitted probe, and Experiment 2 also the accuracy and log loss of the head fitted on each treatment's features" (p. 7). No such number appears anywhere in the paper.
  - The paper's own requirement: "Recovery should be reported beside the held-out utility of the output that is actually released" (p. 12).
- Problems:
  - Noise at σ* (0.528 / 0.513), FARE features (0.550 / 0.501) and the zero-fairness tree (0.645 / 0.607) are reported with no accuracy.
  - HMDA FARE features recover 0.501, which is chance. Section 6.1 says FARE's "only informative bound (0.175) belongs to a degenerate HMDA configuration with a single cell" (p. 8).
  - The HMDA underwriting frozen head is barely better than a constant (Table 2, gain 0.016). If the untreated probe's gain is similarly small, a one-cell, constant FARE release could pass the rule "within one point of the untreated probe's" accuracy (p. 3).
  - I cannot tell whether the selected HMDA FARE release is that constant release. If it is, 0.501 is expected and says nothing about protection.
- Suggestion:
  - add accuracy and gain over the constant to every row of Table 2;
  - give the number of cells in each selected FARE and zero-fairness configuration, and flag any degenerate ones;
  - report the noise scale relative to the representation's per-dimension spread, and the accuracy at σ*.

**6. The paper does not say what its intervals cover, and the seed-to-seed variation is large (p. 7; Table 2 p. 8; Figure 2 p. 9; Table 3 p. 10).**
- Quotes:
  - "standard errors come from 1,999 paired multinomial bootstrap replicates over assessment record groups" (p. 7)
  - "Intervals are conditional on the fitted heads and attackers" (p. 2)
  - "the encoder seeds differ strongly" (p. 12)
  - "a smaller difference of 0.015 ... carried by seed 0 alone" (p. 9)
  - "because seed 2 loses 2.0 points" (p. 9)
- Problems:
  - The intervals resample assessment records only, for three fixed encoders. In Figure 2 the individual-seed markers for the untreated and LEACE heads span roughly 0.63-0.83 in recovery and 0.05-0.10 in gain, while the plotted 90% intervals are about ±0.01. Readers will take "pass" and "established" to generalize beyond these three encoders, which this method does not support.
  - Four kinds of interval are used:
    - one-sided Bonferroni percentile bounds (Experiments 1-2), with the family size and α not given;
    - two-sided 95% simultaneous normal intervals (Experiments 3-4);
    - 90% intervals (Figure 2, and p. 10 "90% interval [−0.003, 0.013]");
    - unadjusted exploratory intervals (p. 4).
  - Table 2 shows lower bounds without the targets or verdicts. Only the last row says "not established".
  - Section 6.1 says "the HMDA interval excludes zero; Table 2" (pp. 7-8), but Table 2 has no interval for the LEACE rows.
- Suggestion:
  - in every caption, state what is being estimated: the mean over these three stored encoders, conditional on the fitted attackers and heads, without retraining variance;
  - print per-seed values, or the range, next to each primary endpoint;
  - add target and verdict columns to Table 2, and give the Bonferroni family size and α for Experiments 1-2;
  - add one sentence defining "not established": the lower bound did not exceed the registered target, which is not evidence of no effect;
  - fix the Section 6.1 reference.

**7. Several headline results hold by construction or are already known, and the new results are buried (p. 1; p. 2; p. 8; p. 10).**
- Results that hold by construction or are specific to these heads:
  - The abstract says: "Releasing the untreated model's task output beside noisy or FARE features restores recovery to 0.76–0.79". But Table 2 shows the full-logit output alone already recovers 0.776 / 0.768, so this is expected. The paper itself says so: "Changing the features cannot remove what an unchanged output carries" (p. 8).
  - "With features plus own output, the features dominate" (p. 10). This is also by construction: the own head reads only those features, so its output adds no information.
  - The offset effect (Section 6.4) "disappears for heads whose logits carry no free offset" (p. 9).
  - "The individual observations ... are known (Section 4)" (p. 2).
- Problem: the abstract and introduction give these the same weight as the results a reader could not have predicted:
  - the capacity-matched control showing that about half of FARE's reduction is compression on one cell and most of it on another (Sections 6.5-6.6);
  - the score-versus-decision gap (Section 7);
  - the linked pair (Section 6.7).

  "Our contribution is this evaluation and its measurements, with their qualifications" (p. 2) does not tell the reader what they learn.
- Suggestion:
  - lead the abstract and introduction with the non-obvious results and their sizes;
  - present the bypass as a control, and say it holds by construction;
  - move Section 6.4 to an appendix, or tie it to a concrete practice such as APIs that return raw logits;
  - state in two sentences what a practitioner should do differently.

**8. The introduction contradicts the results (p. 2).**
- "None of this shows that the evaluated methods are broken; where their promises could be checked, they held."
- But the same page says the encoders' "stored linear check fails on 11 of 42 attribute–seed combinations, which is a failure of the method's own test". Section 7 repeats this: "the encoders' own linear constraint failed its check on 11 of 42 combinations" (p. 11).
- Suggestion: "LEACE met its bound wherever checked, FARE's certificate was uninformative at our sample sizes, and the encoders' own check failed on 11 of 42 pair–seed combinations."

**9. Words describing the status of the evidence go unexplained, and several sentences read like internal project notes (pp. 1, 2, 4, 6, 7, 9, 10, 12).**
- Quotes:
  - "All results are development evidence on previously used data" (p. 1)
  - "Every assessment row had been used by earlier analyses of these models, and registering endpoints before fitting does not turn these rows into fresh confirmation" (p. 2)
  - "we make no use of an earlier, retired bound from linear R² to classifier accuracy" (p. 2) and "The encoders' public code still exposes the interface of the retired R²-to-accuracy bound" (p. 12)
  - "no experiment is credited with a capability that only a later evaluation added. One historical attacker exceeds ..." (p. 4)
  - "uncleaned roles ... the overlapping records were retained and disclosed" (p. 6)
  - "on one already-used held-out split" (p. 7)
  - "reused, hash-verified, from Experiment 2, whose exploratory tables had already reported their levels" (p. 9)
  - "the protocol did not disclose that; we disclose it here" (p. 10)
  - "admitted-input hashes" (p. 12)
- Problems:
  - The reader is not told what the earlier analyses were, what they looked at, or how they could bias these results.
  - "Registration" could mean a public registry or a time-stamped commit in the anonymous repository; the paper does not say which.
  - The "retired bound" is unexplained. It also hints at the authors' earlier work.
  - The "Registered predictions" paragraph cannot be checked. "Four of the ten primary recovery predictions were correct", but three wrong predictions are listed: a larger LEACE effect, a smaller fairness-term effect, and established hard-decision contrasts. There are five hard-decision contrasts, so that would leave three correct, not four, unless not all five were predicted to be established.
- Suggestion:
  - write one paragraph or appendix titled "Analysis history and registration". It should say what earlier analyses used these rows, where the protocols are (anonymized link with timestamps), and what registration does and does not guarantee;
  - add a table of the ten predictions with their outcomes;
  - delete the retired-bound remarks, or explain them in one sentence with an anonymized citation;
  - use "exploratory" consistently in place of "development evidence" and "fresh confirmation".

---

## MINOR

**10. p. 1, title.** "What a Recipient Can Recover from a Useful Prediction: An Evaluation of Feature Defenses and Output Contracts on Stored Representation Models". It runs to three lines, "Output Contracts" and "Stored" are internal terms, and most results are about features rather than "a useful prediction". Suggestion: shorten it and use only defined terms, for example "What Recipients Can Infer from Protected Representations and Their Task Outputs".

**11. p. 1, abstract wording.**
- "FARE lowers sex recovery from the head's score from 0.774 to 0.539" has a doubled "from".
- "When each treatment releases a useful logistic-regression head": the treatment releases the head's output, not the head.
- "(AUC 0.817 and 0.863, against 0.815 and 0.870)": say which value is Adult and which HMDA.
- "restores recovery to 0.76–0.79": "restores" suggests a return to the untreated level (0.815 / 0.870), which is not what happens.

**12. p. 1, the motivation hedges too much.** "We know of no source that documents a deployment with all of these properties, so we do not claim that one exists." This weakens the motivation more than necessary. Suggestion: keep the flag that the scenario is illustrative, but ground each property separately with a citation:
- scoring APIs that return logits or probabilities;
- the premise of fair representation learning that third parties receive the representation;
- data-protection purpose limitation (for example GDPR Art. 5(1)(b)) as the source of the idea of a "permitted purpose";
- record linkage across services.

**13. p. 1.** "Each of these promises is precise about two things, the object it covers and the attackers it covers, and neither has to coincide with what a recipient of the system holds." Attackers cannot "coincide with what a recipient holds". Suggestion: "...and neither need match what a recipient receives or how it can analyse it."

**14. pp. 2, 6 and 8, the label-only reference.**
- "output-only recovery exceeds what the true label alone allows" (p. 2) conflicts with "We report a label-only reference as a descriptive comparison rather than a floor" (p. 6). "Allows" implies a bound. Use "exceeds the label-only reference".
- For "exceeds the label-only reference by 0.177 on Adult and 0.236 on HMDA" (p. 8), give the label-only AUCs themselves, the output format being compared, and how the label-only attacker is fitted.
- Add a row for it to Table 2.

**15. pp. 2 and 7, "11 of 42 attribute–seed combinations" against Table 2 "(pair×seed, all 14 pairs)".** 42 = 14 pairs × 3 seeds, so the unit is a pair–seed, not an attribute–seed. List the 11 failures in an appendix.

**16. p. 2.** "a one-hot ridge R² computed on each training minibatch and held at or below 0.05": say how R² is aggregated across one-hot classes and what the ridge penalty is.

**17. p. 3, FARE selection and the noise-tuning sentence.**
- "whose probe accuracy on attacker_val is within one point of the untreated probe's": define the probe (model family, inputs, labels). Page 6 calls it the "common utility probe".
- "a selection that favours them when they are scored against the same slate": "them" could mean the defenses or the attackers. Say "favours the defenses".

**18. p. 3 against p. 5, output formats.** Section 2 defines four formats, but Table 1 lists six for Experiment 3: "full, centred, probabilities, offset, centred with offset, hard". Define "offset" and "centred with offset", and say why the latter differs from the full logits; by the definition of c on p. 3 it is an invertible re-parameterisation of l. Also define the margin for K > 2.

**19. p. 3.** "(σ* = 2 on Adult and 4 on HMDA)": an absolute σ means nothing without the representation's scale. Report σ relative to the per-dimension standard deviation, and give the task accuracy at σ*.

**20. p. 4, hard to parse.** "A deterministic or real-valued release need not be reversible: a function of the representation can merge records, and a deterministic map to probability vectors, which stays within the probability-output contract, can remove information about the attribute. Conversely, we do not establish that any evaluated release preserves the population distribution of the attribute given the record." Suggestion: say it in two plain sentences. Recovery is what this attacker slate achieves, so a better attacker might recover more. A low recovery does not show the information is gone, and a high one does not show all of it was kept.

**21. p. 4, the exploratory likelihood-ratio attackers.**
- "One historical attacker exceeds the access described above. On the noise treatment only, Experiment 1 also ran two exploratory Gaussian likelihood-ratio attackers": the text says "one" and then "two".
- It is not said which of the reported values (0.546 / 0.563 on Adult, 0.526 / 0.575 on HMDA) belongs to the attacker that is given clean, pre-noise data.
- Add a footnote to the noise row of Table 2 noting that these attackers beat the slate (HMDA 0.575 against 0.513).

**22. p. 4, the labelled-sample assumption.** "we do not show that a recipient could obtain such data in practice". Thousands of labelled records (7,571 / 2,239 in Experiment 1) is a strong assumption. Suggestion: on the primary cells, add a learning curve of recovery against labelled-sample size (for example 100 to 7,500). That would show how much the conclusions depend on the assumption, and it is the most direct evidence for "why this access".

**23. p. 5, Table 1 cells.**
- In the stored-check row, under "What the recipient receives", the entry is "a check, not an attack", which does not fit the column. Move it to a separate line or a footnote.
- "(after 1,500; 1,385 rows for FARE certification)" is ambiguous. Say "attacker_fit after removing 1,500 (Adult) / 1,385 (HMDA) certification rows".
- Experiment 2 lists "noise-aware attacks" as untested, but Experiment 2 has no noise treatment as described.
- "two format banks" and "easier task" appear before they are defined.

**24. p. 6, related work.**
- "We measure the purpose-permission version, as a property of our attackers." I could not tell what this means; please rephrase.
- "In a bounded reading of the closest work ... This is a gap in a bounded search, not evidence of priority." Either state the search scope (venues, years, keywords) or write "To our knowledge".
- Consider citing:
  - Kumar, Tan and Sharma (NeurIPS 2022), on probing classifiers being unreliable for concept removal;
  - Shao, Ziser and Cohen (EACL 2023), on spectral removal of nonlinearly guarded attribute information;
  - Edwards and Storkey (ICLR 2016) and Zemel et al. (ICML 2013), as classic censored and fair representation baselines;
  - Yeom et al. (CSF 2018), on attribute inference from model outputs.

**25. p. 7, how recovery is defined.**
- "Recovery is the macro one-vs-rest AUC on assessment over supported classes": define "supported" (a minimum count?), and say how many race classes are kept for Adult and for HMDA.
- Say whether the reported recovery always comes from the nonlinear attacker (Table 2 says "Nonlinear recovery") or from the best of the three families. Apart from the post hoc remark on p. 9, the role of the logistic-regression attacker is unclear.

**26. p. 7.** "For features plus output, candidates that ignore one part are also eligible on validation." Suggestion: "For features-plus-output releases, model selection also considers attackers that use only the features or only the output."

**27. p. 7, retention.** "Acc(treatment) − 0.8 Acc(A) − 0.2 Acc(const) > 0". Say in words that this means the treatment keeps at least 80% of the untreated head's gain over the constant. Repeat it in the caption of Table 3.

**28. p. 7, Section 6.1.**
- "It meets its bound on all 60 maps": explain where 60 comes from.
- "...on one already-used held-out split), so the linear guarantee also holds on held-out data". A near-zero ρ² on one split, for two cells, is weaker than "the guarantee holds". Suggestion: "the residual linear correlation on held-out data is near zero (one split, primary cells)".

**29. pp. 8 and 12, FARE's certificate.**
- "it was unavailable, vacuous or weak (Adult bounds about 0.50)": explain what "unavailable" means, what quantity the certificate bounds, and what sample size it needs.
- The limitation "FARE is applied to stored representations rather than raw inputs" (p. 12) deserves a sentence on what this changes for FARE's trees and guarantee.

**30. p. 8, Table 2 contents.**
- LEACE recovery appears twice with different values: 0.817 in Experiment 1 against 0.812 in Experiment 2 for Adult, and 0.863 against 0.864 for HMDA. Say in the caption that the role splits differ.
- "11 of 42" sits under both dataset columns although it counts all 14 pairs. Give it its own line.
- "−0.000" should be printed as 0.000 or with more digits.

**31. p. 8, Section 6.3.**
- "HMDA underwriting usefulness is not established": the lower bound is 0.0098 against a target of 0.01 (Table 2). Call it borderline rather than implying the head is useless.
- "the HMDA fair_lending head predicts the same class for everyone, although its scores still vary and are informative about race and sex" is interesting. Define the purpose and give its base rate.

**32. pp. 8-9, Section 6.4.** "on 5 of 14 stored pairs; it was not established for any of the six multiclass Adult pairs, although it passed for the five-class HMDA pricing head with race". List the pairs in an appendix. On moving the section, see item 7.

**33. p. 9, Section 6.5.**
- "Of FARE's 0.234 reduction, compression accounts for 0.125 and the fairness term for a further 0.109, on every seed." It is unclear what holds "on every seed". Say "the fairness-term contrast is positive on each seed (range ...)".
- "a linear attacker on the untreated head's margin recovers only 0.515, while nonlinear attackers on the same scalar recover 0.773" is striking. A small plot of P(sex | margin) would show the non-monotone relationship and make the reading "outside a linear eraser's scope" concrete.

**34. p. 10, Table 3 caption, and p. 7.**
- R(·) is not defined in the caption.
- The recovery target "(> 0.02)" appears only on the first recovery row; state it once for all rows.
- "F0" and "F₀" are mixed.
- "excl. 0: the interval excludes zero but not the target" is literally wrong for [0.011, 0.019], which also excludes 0.02 because it lies entirely below it. Say "lower bound above zero but not above the target". In fact the upper bound being below 0.02 is informative: LEACE's effect is shown to be smaller than 0.02, so say so.
- The same wording appears on p. 7.

**35. p. 10, Section 6.6.**
- "On an easier task chosen by a validation-only screen (Adult employment group with age group protected; ...)": define "employment group" and the screen (what was screened, and among how many candidates). Give the absolute recoveries; only "0.151 below LEACE's" is reported.
- "That task is easy, but no single input column determines it." This reads as a non sequitur unless the reader knows why it matters. Explain it, for example as ruling out a deterministic recoding of one input column.
- "on the head outputs above the fairness term removes 0.109": "above" should be "in Section 6.5".

**36. p. 10, Section 6.7.**
- "(pair minus best single +0.034, lower bound 0.023, the weakest of six passes)": name the six contrasts.
- "On seed 2 under full logits, validation selected the income-only attacker for the pair, so that seed's pair-minus-income difference is exactly zero": say what this implies, namely that the pair estimate is conservative for that seed.
- Give the race class counts for Adult, so that 0.917 can be interpreted.

**37. p. 11.** "AUC is not an information measure, the pair gain does not separate evidence ..., and we prove no composition or synergy result." Three independent clauses are joined by commas. Split them.

**38. pp. 11-12, Section 7.**
- "What lowered recovery at matched utility" restates Section 6.5 almost number for number.
- "What the findings ask of future methods" restates paragraph 6 of the introduction.
- Trim both to the new interpretation.
- "A measured gap (post hoc, descriptive)" is the most forward-looking paragraph. Because a decision is a function of its score, the gap measures how much attribute information the scores carry beyond their thresholded decision. Say so, and say what a method would have to do to close it.

**39. p. 12, Limitations.**
- "Rare HMDA race groups are not estimable": say which groups, and how dropping them affects the macro AUC.
- Add the limitation stated on p. 3: the defenses were tuned against the same attacker slate that scores them.

**40. p. 12, Reproducibility.**
- "per-record outputs, weights and checkpoints are kept privately with hash indexes": say how reviewers can get access.
- "The useful-head comparison used about 0.2 CPU-hours on a laptop": give compute for all experiments, or for none.

**41. References.**
- [15]: "Are your sensitive attributes private? novel model inversion ..." should capitalize "Novel".
- [26] and [27] are the conference and journal versions of the same paper, cited together. Cite one, or say the journal version extends the other.
- [11] gives USENIX Security 2018 as the venue but an arXiv URL. Use the USENIX link.

### Writing

**42. pp. 1-12, repeated disclaimers and the "X, not Y" pattern.**
- There are about a dozen ", not ..." constructions, including:
  - "a comparator, not a recipient", twice (pp. 3 and 5);
  - "not a refutation of LEACE's theorem" (p. 5);
  - "a gap in a bounded search, not evidence of priority" (p. 6);
  - "a property of the release, not of the defenses" (p. 8);
  - "not a general method" (p. 9);
  - "requirements for evaluating a method, not evidence ..." (p. 12);
  - "untested, not ruled out" (p. 12).
- The point that LEACE is not contradicted appears four times: p. 1, p. 5, p. 8 and p. 11.
- "No new method / guarantee / composition theorem" appears on p. 1, p. 2 and p. 11.
- Saying each of these once would be enough. Repeated, they read as defensive and formulaic.

**43. Slogans.**
- "Changing the features cannot remove what an unchanged output carries." (p. 8)
- "A claim about a fairness term therefore needs its capacity-matched twin." (p. 10)
- "the evaluation has to follow the release" (p. 12)
- "Utility changes how low recovery should be read." (p. 2)
- "which we leave untested rather than dismiss" (p. 2)

Replace each with a plain statement tied to the measurement.

**44. The findings are summarized five times:** in the abstract, in paragraph 5 of the introduction (p. 2), at the opening of Section 6.1 (p. 7), in paragraphs 1-2 of Section 7 (p. 11), and in the last paragraph of Section 7 (p. 12). Cut paragraph 5 of the introduction or paragraph 1 of Section 7 by about half.

**45. Abrupt transitions.**
- Section 2 opens with the models before the datasets are introduced.
- Section 6.4 (the offset) interrupts the line from Section 6.3 (weak decisions) to Section 6.5 (useful heads), which the last sentence of Section 6.3 sets up.
- "Registered predictions" (p. 10) drops methodological history into the middle of the results.

**46. Hard-to-parse sentences** (beyond those in items 13, 20, 26 and 37):
- "Its configuration is chosen from a frozen grid among those whose probe accuracy on attacker_val is within one point of the untreated probe's, as the one with the lowest validation recovery for our nonlinear attacker." (p. 3)
- "Full-logit output-only, features-only, features-plus-full-output and bypass surfaces are reused, hash-verified, from Experiment 2, whose exploratory tables had already reported their levels." (p. 9)
- "The effect appeared on 5 of 6 primary encoder seeds (not on the collapsed HMDA seed 1) and on 5 of 14 stored pairs; it was not established for any of the six multiclass Adult pairs, although it passed for the five-class HMDA pricing head with race." (pp. 8-9)

**47. Smaller points.**
- "the task output in four formats that carry different amounts of information" (p. 2). In Experiment 4, three of the four formats carry the same information (p. 9), so write "can carry".
- No em dashes were found. En dashes are used correctly.

---

## LAYOUT

**48. p. 3, Figure 1.** The label "(3) bypass: features + clean logits (adverse control)" runs into the left border of the "recipient (attacker)" box; the closing parenthesis touches the frame. Shorten the label or move it left.

**49. p. 11, Figure 3, right panel.** The legend's black "pair" swatch sits just above the "pair" bar for the centred format (0.883) and visually merges with it. Move the legend to the empty space above "hard decision", or outside the axes. Neither panel shows uncertainty, and the bars stand on a truncated axis that starts near 0.45. Add error bars, or say in the caption that none are shown.

**50. p. 9, Figure 2.** The 90% error bars are almost invisible at print size. The pale per-seed markers are hard to tell apart by colour, especially in greyscale, and a seed cannot be matched across treatments. Use a distinct marker shape per seed, or thin lines joining one seed's treatments. This would also make the seed effect in item 6 visible.

**51. p. 8, Table 2.** "11 of 42" is centred across both dataset columns. Long quantity labels wrap onto two lines, which makes the rows hard to scan. The missing target and verdict columns are covered in item 6. Consider shorter labels and a separate target column.

**52. p. 10, Table 3.** The row labels mix "F0" ("Gain over constant, B / F / F0") with "F₀" ("Acc(F₀) − Acc(A)").

**53. pp. 7-8.** Table 2 floats into the middle of a sentence ("the HMDA interval" | Table 2 | "excludes zero; Table 2)"). If possible, place the float after the paragraph.

**54. General.** There are no unresolved references ("??") or missing citations ("[?]"). No lines overrun, Table 1 fits within the margins, and the figure text is legible at 100%. The title wraps to three lines (see item 10).
