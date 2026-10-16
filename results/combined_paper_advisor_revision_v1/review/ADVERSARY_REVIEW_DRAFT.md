# Adversary review of the revised draft (Phase 2)

16 October 2026. Suggestions only: no manuscript, brief or proposal file was edited.

## How to read this

**Files reviewed**, all at commit `726ea2907`:
- `papers/combined_empirical_v2/main.tex`: SHA-256 `f05922d0…`, 619 lines;
- `results/combined_paper_advisor_revision_v1/ADVISOR_BRIEF.tex`;
- `results/combined_paper_advisor_revision_v1/METHOD_SCOPE_PROPOSAL.md`.

**Line numbers.** Bare numbers ("98") are lines of `main.tex` at that commit. "Brief n" is a line of
`ADVISOR_BRIEF.tex`. "Scope n" is a line of `METHOD_SCOPE_PROPOSAL.md`.

**Tables** are named by their LaTeX label rather than by number:
- `tab:access` (line 213; the request calls it Table 3);
- `tab:summary` (line 383);
- `tab:useful` (line 472).

**Evidence paths**:
- `MRB/` = `results/combined_matched_removal_benchmark_v1/`;
- `OAR/` = `results/combined_output_aware_removal_v1/`;
- `ODX/` = `results/combined_output_diagnosis_v1/`;
- `CAP/` = `results/combined_analysis_paper_v1/`;
- `ACCESS_FACTS.md` = my Phase 1 file in this directory;
- `ADV` and `CORR` = `ADVERSARY_AND_MOTIVATION.md` and `CORRECTIONS.md`, read with
  `git show c9709a2:publications/pcrl-mechanism-design-v1/results/<file>`.

**Labels.**

| Label | Meaning |
|---|---|
| MUST-FIX | The text is false as written, contradicts the paper's own evidence, or misdescribes an experiment's access or coverage |
| SHOULD-FIX | Over-stated, missing, or inconsistent with the 2026-10-16 correction |
| NOTE | Optional |

**Counts:** 7 MUST-FIX, 19 SHOULD-FIX, 8 NOTE.

**Data handling.** No per-record data was opened, nothing was fitted and nothing was run.

---

## A. MUST-FIX

### 1. MUST-FIX: the noise-aware attackers are misdescribed, and line 186–187 is now false (lines 185–189 and 204–211)

**Current text, lines 185–187:**
> "Knowing a defense's published algorithm and objective is different from holding its fitted parameters. A recipient
> may know the former, but the attackers whose recovery we report use neither beyond refitting to the defended
> releases."

**Current text, lines 204–211:**
> "One historical attacker exceeds the access described above. On the noise treatment only, Experiment~1 also ran two
> exploratory Gaussian likelihood-ratio attackers that know the noise scale; one of them is fitted on clean, pre-noise
> representations of the auxiliary records, which no contract releases. They are not insider, repeated-query or
> white-box attacks: each sees one noisy release per record and none uses model parameters or the target's clean
> representation. At $\sigma^*$ they recovered … The noise rows of Table~\ref{tab:summary} are therefore not the strongest
> attack that was run on that surface."

**Problems.**
- **(a) Line 186–187 is false.** Lines 208–209 now report the recovery of these attackers, and they use the noise law
  and σ\*.
- **(b) "One historical attacker" is wrong.** Two attackers are then described.
  - Only the second (LRT-A4) exceeds the access: it uses clean auxiliary representations.
  - The first (LRT-A2) stays within the access. It uses public knowledge of the algorithm and scale, which line 186 says
    a recipient may have.
- **(c) "Each sees one noisy release per record" is false for LRT-A4's fitting data.** LRT-A4 is fitted on clean
  representations of attacker_fit. Both attackers *score* one noisy release per assessment record.
- **(d) Coverage.** The attackers ran on the features-only surface only, and neither was selected on attacker_val. The
  "plus clean outputs" entry of the noise row in `tab:summary` had no likelihood-ratio attacker, so "the noise rows …
  are not the strongest attack" is wrong for that entry.
- **(e) The benchmark's code labels LRT-A4's access "white-box population".** A reader of the repository will see that
  label next to the draft's "not … white-box".

**Evidence:**
- `stored_model_eval/bench.py:1026-1043`;
- `stored_model_eval/recipes.py:133-170`;
- `MRB/RELEASE_ACCESS_TABLE.csv:8-9,16-17`;
- `MRB/EFFECTIVE_PROTOCOL.json:566-581`;
- `stored_model_eval/access.py:7,22`;
- `MRB/RECOVERY.csv:7057,7101,7105,10155,10199,10203`.

**Replacement for lines 185–187:**
> Knowing a defense's published algorithm and objective is different from holding its fitted parameters. A recipient
> may know the former. The attackers of the slate, whose recovery the tables report, use neither beyond refitting to the
> defended releases; the one exception, run only in Experiment~1, is described below.

**Replacement for lines 204–211:**
> Experiment~1 also ran two exploratory attackers with more knowledge of the defense than the slate. On the noise
> treatment and the features-only surface only, it fitted two closed-form Gaussian likelihood-ratio attackers that use
> the noise law: an isotropic Gaussian with the selected scale $\sigma^*$, which the benchmark treats as a published
> parameter of the mechanism. The first is fitted, like the slate, on the labelled noisy releases of
> \texttt{attacker\_fit}. It stays within the access above but uses knowledge of the defense's algorithm and scale. The
> second is fitted on clean, pre-noise representations of the \texttt{attacker\_fit} records, which no contract
> releases, and so exceeds the access above. [Optional: the benchmark's code labels this access ``white-box
> population'' for that reason.] Neither is an insider or repeated-query attack, and neither has white-box access to the
> model. Neither is selected on \texttt{attacker\_val}, both score one noisy release per assessment record, and neither
> uses the encoder's or heads' parameters or gradients, or any assessment record's clean representation. At $\sigma^*$
> they recovered 0.546 and 0.563 on Adult and 0.526 and 0.575 on HMDA, against 0.528 and 0.513 for the slate. These are
> exploratory values with unadjusted intervals and no paired test. The features-only noise entries of
> Table~\ref{tab:summary} are therefore not the strongest attack that was run on that surface. No noise-aware attacker
> was run on noisy features released with clean outputs.

### 2. MUST-FIX: the `tab:access` caption misdescribes the noise-aware attackers (lines 215–218)

**Current text:**
> "Common to every attack: one release per record (…); attackers fitted on labelled \texttt{attacker\_fit} releases,
> selected by \texttt{attacker\_val} log loss and scored on \texttt{assessment}; no attacker receives a record's inputs,
> model parameters, gradients or defense maps, and record keys are never attacker inputs."

**Problem.** The caption claims this holds for every attack. That is false for the two likelihood-ratio attackers:
- neither is selected on attacker_val;
- LRT-A4 is fitted on clean auxiliary representations, not on releases;
- both use σ\*.

Evidence: `bench.py:1027-1043`; `MRB/RELEASE_ACCESS_TABLE.csv:8-9`.

**Replacement:** replace "Common to every attack:" with "Common to every attack of the slate:". After "…never attacker
inputs." add:
> The exploratory likelihood-ratio attackers of Experiment~1 are the exception described in Section~\ref{sec:threat}:
> they use the noise scale, are not selected, and one is fitted on clean auxiliary representations.

Keep the label-only sentence as it is.

### 3. MUST-FIX: noise was not evaluated on all 14 pairs (`tab:access` row 1, lines 229–230, and lines 321–322)

**Current text, lines 229–230:**
> "1. Feature removal (all 14 pairs) & representation (untreated, LEACE, noise at $\sigma^*$); …"

**Current text, lines 321–322:**
> "Experiment~1 compares untreated, LEACE and noisy representations on all 14 pairs."

**Facts.**
- σ\* exists only for the two primary cells (`MRB/SIGMA_STAR_SUMMARY.json:7-42`).
- On the 12 secondary pairs, the noise runs used the full σ grid, not σ\*. They were cut by a budget rule (`MRB/RESEARCH_DECISION.md:20,29-34`):
  - complete for 3 Adult pairs;
  - partial for 1;
  - not run for 8.
- None of these secondary-pair noise runs is reported in the paper.

**Replacement, row 1, columns 1–2:**
> 1. Feature removal (untreated and LEACE: all 14 pairs; noise: the two primary cells) & representation (untreated,
> LEACE, or noise at $\sigma^*$ on the primary cells); representation with the untreated head's full logits; full logits
> only

**Replacement, row 1, column 4:** append "noise on the secondary pairs (stopped by a budget rule; not reported)" to the
untested list.

**Replacement, lines 321–322:**
> Experiment~1 compares untreated and LEACE representations on all 14 pairs, and noisy representations at $\sigma^*$ on
> the two primary cells.

### 4. MUST-FIX: the introduction credits Experiments 1–2 with utility measurement added later (lines 81–83)

**Current text:**
> "The same slate of linear, tree and neural attackers, fitted and selected on held-out roles, attacks every surface,
> and each recovery figure is read beside the held-out utility of the output that is actually released."

**Problem.**
- Utility of the output actually released exists only in Experiment 3 (frozen heads), Experiment 4 (useful heads) and
  Experiment 2's view-3 heads.
- Experiment 1 reported a common refitted probe and a frozen-head diagnostic (`MRB/EFFECTIVE_PROTOCOL.json:603-635`).
  The paper's own Utility paragraph (lines 354–357) says this correctly.
- Lines 203–204 promise that "no experiment is credited with a capability that only a later evaluation added". This
  sentence breaks that promise.

**Replacement:**
> The same slate of linear, tree and neural attackers, fitted and selected on held-out roles, attacks every surface.
> Recovery is read beside held-out utility. In Experiments~3 and~4 this is the utility of the deployed head whose output
> is released. In Experiments~1 and~2 it is the utility of a common refitted probe, and in Experiment~2 also of the head
> fitted on each treatment's features.

### 5. MUST-FIX: line 98 contradicts the encoders' failed check

**Current text:**
> "None of this shows that the evaluated methods are broken; where their promises could be checked, they held."

**Problem.** Lines 88–90, 409–412 and 559–561 report that the encoders' own stored check fails on 11 of 42
combinations. This sentence erases the separation between a method failing its own test and recovery outside a
guarantee. That separation is one of the corrected statements.

**Replacement:**
> None of this shows that LEACE or FARE fails its own promise: LEACE's held wherever we could check it, and FARE's
> certificate was too weak at our certification sizes to be checked. The encoders' stored check is the exception: it
> fails on 11 of 42 combinations.

### 6. MUST-FIX: lines 413–415 reintroduce an overclaim that was already corrected

**Current text:**
> "…its held-out squared top canonical correlation $\rho_1^2$ is near zero (Adult 0.0006, HMDA 0.0001, on one
> already-used held-out split), so the linear guarantee also holds on held-out data."

**Problem.**
- The 2026-10-03 correction replaced "the linear guarantee transfers to unseen rows" with measured values and caveats
  (`OAR/CORRECTION_ADDENDUM.md:89`). Its upper bounds are 0.013 and 0.033.
- On rank-deficient seeds, official LEACE is the identity outside its fitting support
  (`MRB/PROTOCOL.md:114-116`; `MRB/RESEARCH_DECISION.md:179-180`).
- The admitted v1 text (v1 lines 283–284) made no guarantee claim.

**Replacement:**
> …its held-out squared top canonical correlation $\rho_1^2$ is near zero (Adult 0.0006, HMDA 0.0001; 90\% upper bounds
> 0.013 and 0.033, on one already-used held-out split). This is a measurement on one split, not a guarantee about future
> records. On the seeds whose \texttt{defense\_fit} covariance is rank-deficient, LEACE leaves directions outside its
> fitting support unchanged.

### 7. MUST-FIX: the advisor brief keeps the old count (Brief 44–45)

**Current text:**
> "The encoders' stored linear check fails on 12 of 42 attribute--seed combinations."

**Problem.** The paper now says 11 of 42 (lines 89, 392, 411 and 561). The committed table and the source's own list
support 11 (`MRB/NATIVE_AND_HELDOUT_CHECKS.csv`; `MRB/RESEARCH_DECISION.md:56-61`; `ACCESS_FACTS.md` §10). This item is
outside the brief's Section 2, but the advisor will read it.

**Replacement:** "…fails on 11 of 42 attribute--seed combinations."

---

## B. SHOULD-FIX: the manuscript

### 8. SHOULD-FIX: the choice of access is presented as a consequence (lines 161–162)

**Current text:** "The claims we evaluate concern a release policy, so the adversary holds what the policy delivers."

**Problem.** The 2026-10-16 correction classed this inference as "conditional (a choice, not a consequence)"
(`CORR` line 185).

**Replacement:**
> The claims we evaluate concern a release policy, so we chose to evaluate an adversary that holds what the policy
> delivers.

### 9. SHOULD-FIX: stronger attackers within the same access, and published-objective attacks versus fitted parameters (lines 174–175 and 262–263)

**Current text, lines 174–175:**
> "A stronger attacker could recover more, but that alone does not justify granting more access. Each further
> capability defines a separate threat model that needs its own justification."

**Current text, lines 262–263:**
> "attacks that exploit a defense's published objective or its fitted parameters beyond refitting;"

**Problem.**
- The correction separates two cases (`ADV` lines 46 and 55; `CORR` lines 184 and 189):
  - a stronger attacker *within the same access* (a larger slate, or an attacker that uses a defense's published
    algorithm or objective while seeing only the releases) belongs to this threat model, so every figure is a lower
    estimate;
  - *more access*, such as fitted parameters, is a separate threat model.
- The draft never says the figures are lower estimates for this access.
- Its list at 262–263 classes attacks that use the published objective as a separate threat model. That blurs the
  distinction the draft itself draws at lines 185–186.

**Replacement for lines 174–175:**
> More access cannot lower what an optimal attacker recovers, so a stronger attacker could recover more. That alone does
> not justify granting more access: each further capability, such as a defense's fitted parameters, defines a separate
> threat model that needs its own justification. A stronger attacker within the same access is different. A larger
> slate, or an attacker that uses public knowledge of a defense's algorithm and objective while seeing only the
> releases, belongs to this threat model. Our slate is neither, so every recovery figure is a lower estimate for this
> access, not a guarantee.

**Replacement for lines 262–263:** "attacks that use a defense's fitted parameters, such as a LEACE map or a FARE tree;".
After the list, add:
> Within this access, attackers that exploit a defense's published algorithm or objective were not run, apart from the
> exploratory noise-aware attackers of Experiment~1.

### 10. SHOULD-FIX: "every surface" and "four formats that carry different amounts of information" (lines 78–81)

**Current text:**
> "…measure recovery from every surface a recipient could receive: the representation, the task output in four formats
> that carry different amounts of information, features and output together, and the outputs of two recipients linked
> by record."

**Problem.**
- Only the evaluated contracts were measured. For example, no pair included features or defended outputs, and only
  full logits were paired with features.
- The four formats form three levels of information, not four:
  - for the frozen heads, centred logits and probabilities determine each other (`odx/surfaces.py:1-7`;
    `oar/study.py:162-175`; `OAR/OUTPUT_SURFACES.md:29-32`);
  - for the useful heads, full logits, centred margin and probabilities are equivalent (lines 459–461).

**Replacement:**
> …measure recovery from the surfaces a recipient receives under the evaluated contracts: the representation; the task
> output in four formats (full logits, centred logits, probabilities and hard decisions; for the frozen heads, centred
> logits and probabilities carry the same information, full logits more and decisions less); features and output
> together; and the outputs of two recipients linked by record.

### 11. SHOULD-FIX: the label-only reference is cast as an allowance (lines 90–92)

**Current text:**
> "…releasing the untreated model's output beside defended features restores recovery, and output-only recovery exceeds
> what the true label alone allows."

**Problem.**
- "Allows" makes the label-only reference a permitted level or floor. That contradicts lines 293–295 (a descriptive
  comparison, not a floor; Stadler et al.'s scope).
- "Defended features" includes LEACE, but the bypass result is for noise and FARE (line 40).

**Replacement:**
> …releasing the untreated model's output beside noisy or FARE features restores recovery, and output-only recovery
> exceeds that of the label-only reference, which reads the true task label.

### 12. SHOULD-FIX: the training constraint is described as achieved (lines 116–117)

**Current text:** "a one-hot ridge $R^2$ computed on each training minibatch and held at or below 0.05;"

**Problem.** Training targets 0.05 through proxy-Lagrangian dual ascent and does not guarantee it is met
(`pcrl/training/v2_trainer.py:8-25,114`).

**Replacement:** "a one-hot ridge $R^2$ computed on each training minibatch, with the threshold 0.05 enforced through a
Lagrangian penalty;"

### 13. SHOULD-FIX: an untested claim that the selection favours the defenses (lines 147–149)

**Current text:**
> "Both defenses were therefore tuned against the evaluated attacker slate on the attackers' own selection rows, a
> selection that favours them when they are scored against the same slate."

**Problem.** That the selection favours them is plausible but not shown.

**Replacement:**
> Both defenses were therefore tuned against the evaluated attacker slate on the attackers' own selection rows, which
> can favour them when they are scored against the same slate. For noise, attackers outside the slate recovered more
> (Section~\ref{sec:threat}).

### 14. SHOULD-FIX: `tab:access` row 2 omits surfaces that Experiment 2 ran (lines 235–236)

**Current text:**
> "FARE or zero-fairness cell (one-hot) or representation; either with the untreated head's full logits; frozen-head
> output as full logits, probabilities or hard decision"

**Problem.** The row omits three surfaces Experiment 2 fitted and attacked (`oar/run.py:89-95,97-105`;
`oar/fare_run.py:93-97`):
- features with the output of a head fitted on them (A, B, F, F0);
- that output alone;
- noisy features with each output format.

The first two are the sources of Experiment 4's reused surfaces (lines 456–458) and of the head utility at lines
356–357.

**Replacement:** append
> ; features with the output of a head fitted on them, and that output alone (reused in Exp.~4); noisy features with
> each output format (not reported)

### 15. SHOULD-FIX: `tab:access` rows 3–4 omit the refitted head and the cell-conditional attacker (lines 241–251 and 346–348)

**Problem.**
- Row 3 omits the refitted logistic head's formats on the primary cells. They are used at lines 446–448
  (`odx/run.py:171-172`).
- Line 346 points to the table for where the cell-conditional attacker was fitted, but rows 3–4 never mention it. Where
  it was fitted:
  - Experiment 3: hard outputs, the pair's hard contract, and the easier task's FARE and zero-fairness cells
    (`odx/run.py:181,210`; `odx/fare5.py:123,143-147`);
  - Experiment 4: every new F / F0 output and every hard decision (`cap/run.py:118,125-132`).

**Replacement.**
- Row 3, column 2: add "; the same formats for a refitted logistic head on the primary cells".
- Row 3, column 4: add "a cell-conditional attacker on hard outputs and FARE cells (descriptive)".
- Row 4, column 4: add "a cell-conditional attacker on F and F0 outputs and hard decisions (descriptive)".

### 16. SHOULD-FIX: "differ slightly in roles" understates the budget difference (lines 320–321)

**Current text:** "…and they differ slightly in roles and in how intervals are computed."

**Problem.** attacker_fit falls from 7,571 to 6,065 rows (Adult) and from 6,794 to 5,388 (HMDA) between Experiment 1
and the later experiments (`tab:access`).

**Replacement:**
> …and they differ in roles (Table~\ref{tab:access} gives each experiment's labelled budget) and in how intervals are
> computed.

### 17. SHOULD-FIX: recovery is read as an amount of information (lines 416–418)

**Current text:**
> "The guarantee concerns linear classifiers, and this measurement shows how much of the attribute remains available to
> nonlinear ones; it is not evidence against the theorem."

**Problem.** Recovery is a property of the slate (lines 191–194). "How much … remains available" reads as an
information measurement.

**Replacement:**
> The guarantee concerns linear classifiers, and this measurement shows that our nonlinear attackers still recover the
> attribute about as well as from untreated features; it is not evidence against the theorem.

### 18. SHOULD-FIX: the pair's race scope is not stated (lines 545–547)

**Current text:** "…whose outputs are linked by record and pooled to recover race, which is disallowed for both"

**Problem.** Race recovery is the macro AUC over the three supported Adult race groups {1, 2, 4}. Groups 0 and 3 are not
estimable (`ODX/MULTI_RECIPIENT_RESULTS.md:20`; `ODX/COVERAGE_AND_SUPPORT.csv:6-10`).

**Replacement:**
> …pooled to recover race (macro AUC over the three Adult race groups with enough records; the other two are not
> estimable), which is disallowed for both

## C. SHOULD-FIX: the advisor brief (Section 2, plus two adversary-related bullets in Section 3)

### 19. SHOULD-FIX: choice versus consequence, and "needs no further justification" (Brief 28–29)

**Current text:**
> "The claims concern a release policy, so the adversary holds what the policy delivers. An authorized recipient
> necessarily possesses its own release, so that part needs no further justification."

**Problem.** The first sentence presents a choice as a consequence (`CORR` line 185). "Needs no further justification"
over-states.

**Replacement:**
> The claims concern a release policy, so we chose to evaluate an adversary that holds what the policy delivers. An
> authorized recipient necessarily possesses its own release, so that part describes the service rather than an
> assumption.

### 20. SHOULD-FIX: insiders are grouped with threat models that need not involve a breach (Brief 34–36)

**Current text:**
> "Inputs held by the recipient, published parameters, repeated queries and insiders are distinct threat models, not
> necessarily breaches; they are untested, not dismissed."

**Problem.** The correction says inputs, a published model and permitted queries need not involve a breach, while
others need security controls (`ADV` line 53; `CORR` line 180). Insiders belong to the second group.

**Replacement:**
> Inputs held by the recipient, published parameters and permitted repeated queries are distinct threat models that
> need not involve a breach; insiders are another distinct threat model. All are untested, not dismissed.

### 21. SHOULD-FIX: three access facts the advisor should see are missing (Brief 36)

**Problem.** Section 2 omits the labelled budgets, the tuning of the defenses against the slate, and the result from the
noise-aware attackers. The paper now reports all three.

**Replacement:** insert before "The paper now has a table":
> The labelled budgets are 7,571 / 2,239 rows (Adult) and 6,794 / 2,089 (HMDA) in Experiment~1, and 6,065 / 2,235 and
> 5,388 / 2,082 in the later experiments. The noise scale and FARE's configuration were chosen against the same attacker
> slate on its validation rows. In Experiment~1, exploratory noise-aware attackers recovered more from the noisy
> features (0.546--0.575) than the reported slate (0.513--0.528); the paper now reports this.

### 22. SHOULD-FIX: "defended features" is wider than the evidence (Brief 46–47)

**Current text:** "Releasing the untreated model's output beside defended features restores recovery to 0.76--0.79"

**Problem.** With LEACE features plus clean output the values are 0.814 and 0.864, not 0.76–0.79
(`OAR/OUTPUT_SURFACES.md:53-61`). The paper (line 40) says "noisy or FARE".

**Replacement:** "Releasing the untreated model's output beside noisy or FARE features restores recovery to 0.76--0.79"

### 23. SHOULD-FIX: the scope of the pair result is missing (Brief 51–52)

**Current text:**
> "\item \textbf{Linked recipients.} Two linked recipients recover race better than either alone in every output
> format, including hard decisions (0.611 against 0.576)."

**Replacement:**
> \item \textbf{Linked recipients.} In the one coalition evaluated (Adult income and employment recipients, race), two
> recipients granted perfect linkage recover race better than either alone in every output format, including hard
> decisions (0.611 against 0.576).

## D. SHOULD-FIX: the method-scope proposal (adversary-related statements)

### 24. SHOULD-FIX: learning curves cannot settle removal versus slower learning (Scope 22–24)

**Current text:**
> "Learning curves over the auxiliary budget help to tell the two apart in practice: a release that only slows learning
> tends to lose its advantage as the attacker receives more labelled data."

**Problem.** "Tends to" is an unsupported generalisation. The correction withdrew the claim that measured differences
are learnability differences (`CORR` line 19).

**Replacement:**
> Learning curves over the auxiliary budget can indicate, but not establish, which applies. An advantage that shrinks as
> the labelled budget grows is consistent with slower learning; one that persists is consistent with information
> removal or with a limit of the attacker families.

### 25. SHOULD-FIX: the ACS benchmark definition omits its access (Scope 44–48)

**Problem.** Commitment 1 (Scope 14–18) says the access model and the labelled budget are part of every claim. The ACS
definitions list neither (`ADV` lines 89–95).

**Replacement:** add a bullet after the release contract:
> access: readers are fitted on 8,001 labelled people (ATTACK_FIT) and selected and oriented on 3,000
> (ATTACK_VALIDATION), drawn from random household splits of the same cohort; one fixed deterministic release per
> person; the pair bank includes readers that ignore either recipient; household and person identifiers are never
> reader features.

### 26. SHOULD-FIX: the stored-model benchmark definition is incomplete (Scope 55–57)

**Current text:**
> "release contracts (output only; features with own head output; the clean-output bypass as an adverse control), the
> four output formats, …"

**Problem.** The definition omits features-only and each experiment's access.

**Replacement:**
> release contracts (features only, for interpretation; output only; features with own head output; the clean-output
> bypass as an adverse control), the four output formats, each experiment's access and labelled budget (the paper's
> access table), …

## E. NOTES

### 27. NOTE: an unsourced frequency claim (line 63)

"A recipient usually knows whom an output describes" was kept from patch v2, but it has no source. The correction
removed the similarly unsourced "often releases". Consider: "A recipient may know whom an output describes, for example
because it requested the output for that person."

### 28. NOTE: say why the scored record's inputs are withheld (Section 3, after line 175)

The motivation does not say why inputs are withheld. Consider adding:
> Withholding the scored record's inputs isolates what the release itself reveals. With the inputs, recovery could come
> from imputation rather than from the release~\citep{jayaraman2022imputation}. A recipient that holds inputs is a
> separate threat model, not an illegitimate one.

Source: `ADV` line 132.

### 29. NOTE: the information claim at lines 424–426 is defensible

"So the outputs carry information about the attribute beyond what the true label carries" holds as written. The
label-only table ranks records by the label posterior, which no function of the label can out-rank, and the interval
excludes zero. Saying this explicitly heads off the objection that "AUC is not an information measure".

### 30. NOTE: other owner-side uses of the attacker roles (lines 332–334)

Two points could be added:
- attacker_val's task labels also select the probe and decide which FARE configurations are admissible
  (`oar/study.py:325-344`; `oar/fare_run.py:25-47`);
- σ\* was chosen on Experiment 1's uncleaned attacker_val (`MRB/SIGMA_STAR_SUMMARY.json:44`), while this paragraph
  describes the roles of Experiments 2–4.

### 31. NOTE: repository terms that differ from the paper's

- Experiment 2's files call the cell-conditional attacker the "defense-aware attacker" (`OAR/PROTOCOL.md:109`;
  `OAR/RUN_STATUS.md:13`; `oar/study.py:180`), although it uses only released cells.
- The benchmark's code labels LRT-A4's access "white-box population" (`access.py:7,22`).

A footnote mapping these to the paper's terms would prevent an apparent clash with lines 185 and 207.

### 32. NOTE: the pair comparison is asymmetric, in the pair's disfavour (lines 548–552)

Under the full and centred contracts, each single-recipient comparator is a format bank, while the pair attacker sees
only the contract's format (`odx/run.py:205-214`). The asymmetry works against the pair, so the reported gains are
conservative in this respect. One sentence would record this.

### 33. NOTE: Experiment 3 used a stricter FARE rule (lines 143–145)

The rule at lines 143–145 is Experiment 2's. Experiment 3's easier-task nominee also had to keep at least 80 % of the
untreated probe's gain (`odx/fare5.py:126`; `ODX/PROTOCOL.md:205`). `tab:access` row 3 could say so.

### 34. NOTE: one more untested auxiliary variant (lines 263–264)

Add "or their pre-release representations (used only by the exploratory attacker of Experiment~1)" to the list of
untested auxiliary variants.

## F. Checked and found correct

These passages match the committed evidence and the corrections:

- **Abstract.** The LEACE scope (lines 38–40); the bypass limited to noise and FARE (line 40); the single coalition
  under granted perfect linkage (lines 43–44).
- **Introduction.**
  - The illustrative scenario is labelled as such and no documented deployment is claimed (lines 58–68).
  - Other access is named as a different threat model (lines 70–76).
- **Section 3, the recipient's release.** The noise draws and the label-only reference are described correctly (lines
  153–159).
- **Section 3, why this access.**
  - Possession of the delivered information is stated (line 162).
  - Linkage is granted, its direction is correct, and pooling is not claimed to be unavoidable (lines 164–168).
  - Auxiliary data is labelled as an assumption (lines 168–174).
- **Section 3, attackers.**
  - Fitting, selection and the internal 10 % early-stopping split are right (lines 177–180).
  - The sample is needed by this supervised bank, not by every attack (lines 182–184).
  - The defense-awareness sense, the algorithm-versus-parameters distinction and the identifiers are right (lines
    184–189, apart from item 1).
- **Section 3, what recovery measures.** Removal versus attacker limits, non-reversibility, removal within a
  probability contract, posterior preservation not established (lines 191–199).
- **`tab:access`.** The stored-check row and the labelled sizes in every row (lines 225–251).
- **Threat models outside the evaluation.** Not necessarily breaches; release-policy versus security controls (lines
  256–261).
- **Related work.** Stadler et al.'s scope and the label-only reference as a comparator, not a floor (lines 291–295).
- **Protocol.** Roles and owner-side use (lines 328–337); the attackers paragraph and what lies outside the slate (lines
  339–352); utility (lines 354–357); native checks with certification sizes (lines 359–362); the stored check reproduced
  (lines 372–375).
- **Results and discussion.** Separating the three outcomes (lines 409–412, 559–565); the pair's linkage granted by
  aligned rows and its no-synergy caveats (lines 545–555); limitations (lines 596–601).
