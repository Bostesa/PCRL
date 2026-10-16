# Adversary review: correction pass v1

Adversary reviewer, correction pass on the PCRL mechanism-design campaign, 2026-10-16. Paths are relative to `results/pcrl_mechanism_design_v1/` (R/) unless stated otherwise.

**Scope and conduct.**
- I read: `ADVERSARY_AND_MOTIVATION.md` (all of it), `MANUSCRIPT_ADVERSARY_PATCH.diff`, `ADVISOR_BRIEF.md` §3, `START_HERE.md`, `PROBLEM_SPECIFICATION.md` §2–§3, the `roles/adversary_reviewer/` files (`ACCESS_TABLE.md`, `ACS_SERVICE_ADVERSARY_SPEC.md`, `MOTIVATION_DRAFTS.md`, `MANUSCRIPT_PATCH.md`, `README.md`) and `CLOSEST_PRIOR_WORK.md` §1. I read the manuscript with `git show 3b5065805:papers/combined_empirical_v1/{main.tex,references.bib}`.
- I also read two committed metadata files at `3b5065805` to check attacker role sizes (see "Notes for the manuscript owner").
- I wrote only this file and `MANUSCRIPT_ADVERSARY_PATCH_v2.diff`, both in this directory. I made no commit, did not touch the manuscript branch, and did not use stash or switch branches.
- I opened no person-level array or model file, nothing under `.private/`, no private review source and no meeting transcript. Scratch work was in the session scratchpad only.

**Verdict.** The concrete access definition is correct and is kept unchanged. What needed correcting was the motivation around it:
1. The exclusion rationale ("a breach of the provider or information already present in the inputs ... a release policy controls neither") is false as stated.
2. "No reader can be calibrated" without the labelled sample is false as stated. Only our supervised reader bank needs it.
3. "Cannot take away" and "cannot forbid", applied to linkage and pooling, are unsupported.
4. "None knows a defense's ... training objective" is false as stated. Our readers simply do not use that knowledge.
5. No sentence says outright that this is the only realistic threat. The exclusion rationale implies it, though, and that implication is removed.
6. The scope is now explained positively: it isolates inference from the delivered outputs under a declared labelled-data budget, and other threat models are untested, not dismissed.

## (a) Classification of affected sentences

"AM" is `ADVERSARY_AND_MOTIVATION.md` and "AB" is `ADVISOR_BRIEF.md`. Line numbers for both refer to the committed versions at worktree HEAD `b2958c76d`. In the working tree the coordinator has since replaced AB §3 with `ADVERSARY_SECTION_PLACEHOLDER`, under the heading "## 3. The adversary, and why this access (corrected)", and the replacement body in (b) is meant for that slot. "P1" is the current `MANUSCRIPT_ADVERSARY_PATCH.diff` (line numbers inside the diff file; main.tex line numbers after applying it are in brackets), and "MT" is `main.tex` at `3b5065805`.

| # | File | Location | Short quote | Classification | Replacement (short form; full text in (b) or the v2 patch) | Reason |
|---|---|---|---|---|---|---|
| 1 | AM | §1 heading, l.14 | "Answer to [advisor's name]'s question: why this access, and not simply a stronger attacker" | not a claim (naming) | "Answer to the advisor's question: why this particular access is relevant, rather than simply why a stronger attacker can recover more" | This pass refers to the advisor as "the advisor". The new heading states the question as posed. The same applies to the §5 heading, l.86. |
| 2 | AM | §1, l.16 | "The adversary is therefore drawn at the release boundary." | conditional | "We chose to evaluate an adversary that holds what is released ..." | The release-policy framing motivates this scope. It does not force it, so it is a choice, not a consequence. |
| 3 | AM | §1, l.19 | "Two capabilities the policy cannot take away" | unsupported | "Two capabilities the evaluation grants rather than rules out" | Linkage can be hindered by policy and governance: separate identifiers per recipient, contractual bans and audits. We did not study how well such controls work. A release policy cannot erase knowledge of the attribute obtained elsewhere, but whether recipients hold such knowledge, how much and how well matched, is an assumption. |
| 4 | AM | §1, l.20 | "recipients who serve the same person can link their records" | conditional | "may link their records. We do not assume that contracts or governance prevent this, and we grant perfect linkage, the attacker-favourable limit" | Linking requires shared identifiers and cooperation. The benchmark grants it; it is not shown. |
| 5 | AM | §1, l.22 | "Nothing else." | correct (as a description of the evaluated access) | Add: "This is the scope of the test, not a claim that other access is unrealistic." | Without the addition, the list reads as exhaustive of realistic threats (item 5). |
| 6 | AM | §1, l.27 | "Perfect linkage is the attacker-favourable limit, so a failure to protect is not an artefact of linkage error." | unsupported (the reason points the other way) | "a low pair recovery cannot be blamed on poor linkage, while a high one may overstate what a coalition with imperfect linkage would recover" | The attacker-favourable limit protects a *protected* verdict from a linkage artefact. A failure measured under perfect linkage may not occur with error-prone linkage. |
| 7 | AM | §1, l.28 | "Without them no reader can be calibrated" | false as stated | "Our readers are supervised, so the sample is what lets this particular reader bank turn outputs into guesses. Attacks that use a known population law or analytic structure need no such sample, and none was run." | Only supervised readers need labelled releases. Analytic, known-law and other attacks need not. |
| 8 | AM | §1, l.28 | "and a 'protected' verdict could rest on the attacker simply lacking data" | conditional | "Granting a generous same-population sample keeps a 'protected' verdict from resting on readers that had too little data, for this slate and this budget." | It holds only for this slate at 8,001 + 3,000 labelled people. A larger sample or other readers could recover more, because fitted readers give lower estimates (`PROBLEM_SPECIFICATION.md` §3). |
| 9 | AM | §1, l.32 | "Giving the attacker more would raise recovery" | conditional | "could raise recovery (it need not ...)" | More information cannot lower the optimal attacker's recovery, but it need not raise it. For example, re-querying a deterministic release returns the same output (`ACS_SERVICE_ADVERSARY_SPEC.md` bullet 7). |
| 10 | AM | §1, l.32 | "The target's inputs or labels, model weights or gradients, repeated or chosen queries, and insiders would each measure either a breach of the provider or information already present in the inputs." | false as stated | "... are each distinct threat models. Several involve no breach: a recipient may legitimately hold a person's inputs, a model may be published, and a service may permit repeated queries." | Target-input side information, public parameters and permitted queries are not breaches. Target labels are not "information already present in the inputs". |
| 11 | AM | §1, l.32 | "A release policy controls neither." | false as stated | "Some are governed by release-policy choices, such as rate limits, one release per person, query auditing and whether to publish the model; others need security controls." | Query limits and model publication are policy choices. The sentence also rests on the false dichotomy of row 10. |
| 12 | AM | §1, l.32 | "Those accesses need other controls, so we list them as untested and claim no protection against them." | conditional (first clause); correct (second) | "They are outside this evaluation's scope. We list them as untested, claim no protection against them, and do not dismiss them." | Not all of them need other controls (row 11). Listing them as untested is right. |
| 13 | AM | §1, l.34 | "A stronger attacker within the same access, such as a larger reader slate, ... belongs to the threat model" | correct, incomplete | Add: readers that use public knowledge of the defense's algorithm and objective while seeing only the same outputs also belong here, and none of ours does. | Item 4: knowing a published method is compatible with output-only access. |
| 14 | AM | §4, l.80 (= P1 l.27–28 [main.tex l.60–61]) | "A recipient normally knows whom an output describes, so two recipients ... can link their outputs by record" | conditional | "usually knows ... could link their outputs by record" | Linking is possible, not established, and needs cooperation (item 3). |
| 15 | AM | §4, l.80 (= P1 l.31–33 [l.64–66]) | "we study the conditional threat model they define (Section 2) and report what a fixed set of attackers recovers under it." | correct, incomplete | Add the narrow-question sentence and the other-threat-models sentence (see (b) §4 and the v2 patch) | Item 6, plus items 1 and 5 for the Introduction. |
| 16 | AM | §4 census add-on, l.84 | "The benchmark grants a coalition the two outputs ... It also grants every recipient a labelled sample ..." | correct, incomplete | Add the budget (8,001 to fit, 3,000 to select, same cohort), the governance clause and the scope sentence | Item 6. "It also grants every recipient" is accurate. |
| 17 | AM | §5, l.88 | "Two recipients who serve the same person can link what they receive." | conditional | "may link what they receive; we grant perfect linkage ... and do not assume contracts would stop it" | As row 4. |
| 18 | AM | §5, l.88 | "That sample is what lets a reader turn a release into a guess, so without it a 'protected' result could mean only that the attacker lacked data." | false as stated (first clause); conditional (second) | "Our readers learn from a labelled sample ..." | As rows 7–8. |
| 19 | AM | §5, l.88 | "which a per-recipient policy cannot forbid" | unsupported | Delete. Use the perfect-linkage grant instead. | Item 3: pooling is not shown to be unavoidable. |
| 20 | AM | §5, l.88 | "Each of those would measure a breach of the provider or the information already in the inputs, not what the release adds." | false as stated | "Other access ... defines a different threat model and need not involve a breach. We leave those untested, not dismissed." | As rows 10–11. |
| 21 | AM | §7, l.111 | "Jayaraman and Evans ..., whose imputation critique is why target inputs are withheld" | conditional | "whose imputation critique is one reason this evaluation withholds target inputs: it isolates what the outputs reveal. A recipient that holds the inputs is a separate, untested threat model." | It is acceptable as a scope reason. It must not imply that input-holding recipients are an illegitimate threat. |
| 22 | AM | §8, l.137 | "Limitations: the full untested-access list." | conditional | "Limitations: the threat models outside scope, listed as untested rather than dismissed." | No such list is complete, and v2 rewords it. |
| 23 | AM | §8, l.141–143 | "SHA-256 values match ...: main.tex fedd71ed…, references.bib b7043344…"; "grows the paper from 10 to 11 pages" | unresolved until updated | Replace with the v2 values: main.tex `cf0b25f2…`, references.bib `b7043344…` (unchanged); 10 → 11 pages, re-checked | The v1 hashes become stale once v2 replaces v1 (see the verification record). |
| 24 | AM | §9, l.149–157 | "The following are untested: ..." | correct, incomplete | Full replacement in (b) §9 | Missing: objective-aware attacks, public (non-breach) parameters, larger auxiliary samples, auxiliary inputs and labels, and attacks that use no labelled sample. Not framed as distinct threat models. |
| 25 | AB | §3, l.43 | "so the adversary is drawn at the release boundary" | conditional | "so we chose to test an adversary that holds what is released" | As row 2. |
| 26 | AB | §3, l.45 | "something a per-recipient policy cannot forbid" | unsupported | "We do not assume governance prevents linkage; perfect linkage is the attacker-favourable limit." | Item 3. |
| 27 | AB | §3, l.46 | "the measuring instrument without which no attacker can be calibrated" | false as stated | "a same-population labelled sample ... for our supervised readers" | Item 2. |
| 28 | AB | §3, l.48 | "granted by the benchmark. It is not shown to be obtainable in practice." | correct | Keep (merged into the replacement) | — |
| 29 | AB | §3, l.50 | "because they would measure a breach of the provider rather than what the release adds" | false as stated | "... are distinct threat models, untested rather than dismissed" | Item 1. |
| 30 | AB | §3, l.50 | "The adversary is not new: ..." | correct | Keep | Matches `CLOSEST_PRIOR_WORK.md` §2. |
| 31 | P1 | l.45 [main.tex l.120–121] | "Each record is released once, and the recipient cannot submit inputs of its choosing." | conditional | "Each record is released once, and no attacker submits inputs of its choosing." | This is a restriction of the evaluation, not an incapacity of real recipients. |
| 32 | P1 | l.45–46 [l.121–122] | "To turn a release into a guess about the protected attribute, the recipient is granted labelled auxiliary data" | conditional | "Our attackers are supervised: each learns to turn a release into a guess ... from labelled auxiliary data that the benchmark grants" | Item 2: the original implies that labels are needed for any guess. The replacement also adds that attacks needing no sample are not evaluated, and names the roles as the data budget. |
| 33 | P1 | l.52–53 [l.128–129] | "perfectly linked; this is the attacker-favourable limit of record linkage, and we model no linkage error." | correct | Keep, and add "We do not assume that contracts or governance prevent such linkage" | Item 3. |
| 34 | P1 | l.55 [l.131] | "None knows a defense's parameters or training objective" | false as stated | "A recipient may know a defense's published algorithm and objective, but our attackers use no such knowledge beyond this refitting, and none uses fitted parameters or gradients." | Item 4: LEACE and FARE are published methods. Not running an objective-aware attack does not make recipients ignorant of the method. |
| 35 | P1 | l.55–56 [l.131–132] | "and no white-box, adaptive-query, repeated-release or insider attacker is evaluated." | correct | Folded into a sentence that names these as distinct threat models, untested and neither claimed nor dismissed | Items 1, 5 and 6. |
| 36 | P1 | l.57–58 [l.133–134] | "Recovery is therefore a measured property of a fixed attacker slate under this conditional threat model, not a privacy guarantee." | correct | Keep | — |
| 37 | P1 | l.132–136 [l.425–428] | "The threat model of Section~\ref{sec:threat} leaves untested repeated or adaptive queries, white-box access ..." | correct, incomplete | "... is one of several. We leave the following distinct threat models untested, not ruled out: ..." (adds public or leaked parameters, chosen-input queries, objective-aware attacks, larger auxiliary samples) | Items 1, 4, 5 and 6. |
| 38 | MT | abstract, l.29 (outside P1) | "A model that serves several purposes often releases two things to each recipient" | unsupported | "may release" | No source is given, and it conflicts with the patched Introduction's statement that no documented deployment with these properties is known. **Added as its own one-line hunk; drop it if the owner prefers.** |
| 39 | roles/adversary_reviewer/MOTIVATION_DRAFTS.md | (b), l.37–41 | "two capabilities that the policy cannot take away ... would measure a breach of the provider, or information already in the inputs. A release policy controls neither" | unsupported / false as stated | Superseded by (b) §1 below | A source draft. Not edited, because it is role output. |
| 40 | MOTIVATION_DRAFTS.md | (b) table, l.45 | "Weights, gradients and internal states: these belong to an insider or the provider and bypass the release." | conditional | Superseded | Published weights belong to anyone. |
| 41 | MOTIVATION_DRAFTS.md | (b) table, l.47 | "A recipient cannot calibrate a reader without labelled examples." | false as stated | Superseded | As row 7. |
| 42 | MOTIVATION_DRAFTS.md l.17; MANUSCRIPT_PATCH.md l.67, l.95 | (a); diff block | "normally knows ... can link"; "None knows a defense's parameters or training objective" | as rows 14 and 34 | Superseded by the v2 patch | These are copies of the P1 text. |

Checked and **not** affected:
- `ACCESS_TABLE.md` l.56, "No claim holds against attackers without such data or with different data", is a correct scope statement.
- `PROBLEM_SPECIFICATION.md` §2–§3 is correct. It already says that linkage is "an assumption of the threat model, not a measured capability" and that recovery is "a finite-sample lower estimate".
- `START_HERE.md` makes no motivation claim beyond "corrected the adversary definition".

## (b) Corrected text, ready to paste

### Replacement for `ADVERSARY_AND_MOTIVATION.md` §1 (keep the "Provenance of the question" paragraph after it unchanged)

> ## 1. Answer to the advisor's question: why this particular access is relevant, rather than simply why a stronger attacker can recover more
>
> The claims we test concern a **release policy**: what each recipient is sent. We therefore chose to evaluate an adversary that holds what is sent, plus two capabilities that the benchmark grants rather than rules out. The access has three parts:
>
> 1. **What the policy hands out:** each recipient's own outputs about a person, one fixed release per person.
> 2. **Two capabilities the evaluation grants:**
>    - recipients who serve the same person may link their records. We do not assume that contracts or governance prevent this, and we grant perfect linkage, which is the attacker-favourable limit;
>    - a recipient may already know the protected attribute for some people. The benchmark models this with a fixed labelled sample of other people from the same cohort: in ACS, 8,001 people to fit readers and 3,000 to select and orient them.
> 3. **Nothing else.** This is the scope of the test, not a claim that other access is unrealistic.
>
> Each element corresponds to one claim:
>
> - **A single recipient's own outputs** test the per-recipient claim. Every legitimate recipient holds them by design, and they are what the policy controls.
> - **Linked outputs of two recipients** test whether per-recipient protection carries over when recipients pool. Purpose limitation is written per recipient, but the person is shared. Perfect linkage is the attacker-favourable limit: a low pair recovery cannot be blamed on poor linkage, while a high one may overstate what a coalition with imperfect linkage would recover.
> - **The labelled sample** tests no claim directly. Our readers are supervised, so the sample is what lets this particular reader bank turn outputs into guesses, and its size is the attacker's data budget. Granting a generous same-population sample keeps a "protected" verdict from resting on readers that had too little data, for this slate and this budget. Other attacks, such as ones built from a known population law or from analytic structure, need no such sample; none was run.
>
> The labelled sample comes from the same population, and that is an assumption the benchmark grants. The experiment does not show that recipients can obtain such data in deployment.
>
> **Why this access is relevant.** Together these elements isolate one narrow, well-posed question: what can be inferred about the protected attribute from the delivered outputs alone, by one recipient or by a linked pair, under a declared labelled-data budget. The answer does not depend on assumptions about what else a recipient holds about the target.
>
> **Why not simply a stronger attacker.** More access cannot lower an optimal attacker's recovery, so it could raise it. It need not: re-querying a deterministic release returns the same output. Either way it would answer a different question. The following are each distinct threat models:
> - a recipient that also holds the target's inputs or later learns its label;
> - public model parameters;
> - permitted repeated or chosen queries;
> - attacks that use the published objective or the fitted parameters;
> - insiders.
>
> Several of these involve no breach of the provider: a recipient may legitimately hold a person's inputs, a model may be published, and a service may permit repeated queries. Some are governed by release-policy choices, such as rate limits, one release per person, query auditing and whether to publish the model. Others need security controls. All of them are outside this evaluation's scope. We list them as untested in §9, claim no protection against them, and do not dismiss them.
>
> A stronger attacker within the same access is a different matter. A larger reader slate belongs to this threat model, and so do readers that use public knowledge of the defense's algorithm and objective while seeing only the same outputs. None of our readers uses that knowledge beyond refitting to the defended releases. This is why every recovery figure is a lower estimate rather than a guarantee. A larger labelled sample changes the declared budget, and is listed with the untested variants in §9.

### Replacement for `ADVERSARY_AND_MOTIVATION.md` §4

> ## 4. Manuscript-ready motivation paragraph
>
> This paragraph replaces Introduction paragraph 1 of `papers/combined_empirical_v1/main.tex`. The full patch is `MANUSCRIPT_ADVERSARY_PATCH.diff` (§8).
>
> > Representation-level defenses are usually evaluated on the representation: a linear probe, an in-training adversary, or a certificate about classifiers that read the representation. A release policy, however, is defined by what each recipient receives. As an illustration, and not a description of any documented deployment, consider a service that holds records about people and serves two recipients with different permitted purposes: one receives a prediction of an income band, the other a prediction of an occupation group, and neither may use a declared attribute such as sex or race. For each person it serves, a recipient receives the output of its own task (a score, a probability vector or a decision) and possibly the features that produced it. The policy is written per recipient, but the person is shared. A recipient usually knows whom an output describes, so two recipients that serve the same person could link their outputs by record, and a recipient may know the protected attribute for some of the people it serves, which would let it learn how outputs relate to that attribute. Under these conditions a defense applied to the features addresses only part of what recipients see: the task output may carry what was removed from the features, a recipient may hold features and output together, and two recipients may pool linked outputs. We know of no source that documents a deployment with all of these properties, so we do not claim that one exists. We study one narrow question that these conditions raise: what a fixed set of attackers, each fitted on labelled releases of other records from the same population, recovers from the releases themselves, held by one recipient or by two whose releases are perfectly linked (Section 2). Other access, such as a recipient that also holds a person's inputs, public model parameters or permission to query repeatedly, defines a different threat model, which we leave untested rather than dismiss.
>
> If the ACS output service enters the manuscript, add:
>
> > In our census benchmark the two recipients receive, for each person, the probability vector and decision of an income task and of an occupation task respectively. Both outputs are computed from six permitted survey fields, and neither recipient may use sex. The benchmark grants a coalition the two outputs for the same person, perfectly linked; it does not assume that contracts or governance would prevent such linkage. It also grants the attacker a fixed labelled sample of other people from the same cohort, with their releases and their sex: 8,001 people to fit attackers and 3,000 to choose and orient them. This sample is the attacker's data budget. The benchmark withholds the target's inputs and labels and every model internal, and it releases each person once. These are assumptions of the evaluation, not properties established for any real service. The benchmark therefore answers one narrow question, namely what attackers with this budget recover from the delivered outputs alone. Other access is a distinct threat model that it leaves untested.

### Replacement for `ADVERSARY_AND_MOTIVATION.md` §5 (175 words)

> ## 5. Shorter explanation for the advisor
>
> > We draw the adversary at the release, because that is where our claims are made. A recipient gets exactly what the policy sends it for its purpose, once per person. Two recipients who serve the same person may link what they receive. We grant perfect linkage, the worst case for the defender, and do not assume contracts would stop it. Our readers learn from a labelled sample of other people from the same population, which the benchmark grants: 8,001 to fit and 3,000 to select. The single recipient tests the per-purpose promise; the linked pair tests whether it holds when recipients pool. This isolates one narrow question: what the delivered outputs alone reveal, given that much labelled data. Other access, such as a recipient that also holds the person's inputs, a public model, permitted repeated queries, or an insider, defines a different threat model and need not involve a breach. We leave those untested, not dismissed. Within our access a stronger reader could recover more, so every number is a lower estimate, not a guarantee.

### Replacement for `ADVERSARY_AND_MOTIVATION.md` §9

> ## 9. Threat models outside this evaluation's scope
>
> Every recovery figure is a measured property of a fixed reader slate under the access in §1. The following are distinct threat models that this evaluation does not test. They are not dismissed: several involve no breach of the provider, and some are governed by release-policy choices, such as rate limits, one release per person, query auditing and whether to publish the model. We claim no protection against any of them.
> - **Target-input side information:** recipients that also hold the target's raw inputs, or later learn its true label or outcome.
> - **Public or leaked parameters:** access to the model's parameters or gradients, whether published or obtained otherwise.
> - **Repeated or chosen queries:** permitted repeated, adaptive or chosen-input queries, and releases of the same person from several model versions. Repeating a query to the current deterministic service returns the same release.
> - **Objective-aware or parameter-aware attacks:** readers that exploit the published training algorithm or objective, or the fitted parameters, beyond refitting to the defended releases. Recipients may know a published method; our readers simply do not use it.
> - **Insiders** at the provider.
> - **Imperfect linkage:** coalitions that link records with error. Our pair is perfectly linked, which is the attacker-favourable limit.
> - **Larger coalitions:** more than two recipients.
> - **Different auxiliary data:**
>   - samples larger or smaller than ours (8,001 to fit, 3,000 to select);
>   - samples drawn from another population or period;
>   - samples that include the auxiliary people's raw inputs or task labels, as in de-censoring attacks;
>   - attacks that use no labelled sample at all, such as ones built from a known population law.

### Replacement for `ADVISOR_BRIEF.md` §3 (body 145 words, plus the pointer line; the heading may keep the coordinator's "(corrected)" suffix)

> ## 3. The adversary, and why this access
>
> Our claims concern a release policy, so we chose to test an adversary that holds what is released:
> - each recipient's own outputs, testing the per-purpose promise;
> - two recipients' outputs for the same person, perfectly linked, testing whether that promise holds when they pool. We do not assume governance prevents linkage; perfect linkage is the attacker-favourable limit;
> - a same-population labelled sample (8,001 people to fit, 3,000 to select) for our supervised readers. The benchmark grants it; it is not shown to be obtainable in practice.
>
> This isolates one narrow question: what the delivered outputs reveal under that data budget. Recipients holding a person's inputs, public parameters, permitted repeated or chosen queries, objective- or parameter-aware attacks and insiders are distinct threat models, untested rather than dismissed. The adversary is not new: it is the auxiliary-data attribute-inference attacker in the linked-recipient setting of prior work.
>
> Manuscript-ready prose and a clean patch are in `ADVERSARY_AND_MOTIVATION.md`.

## (c) Other overstatements in `ADVERSARY_AND_MOTIVATION.md`

- **§1, l.27, linkage reasoning.** This is row 6. The direction of the attacker-favourable argument was reversed, and the §1 replacement fixes it. The same sentence in `MOTIVATION_DRAFTS.md` l.46 ("a negative result is not an artefact of linkage error") is correct only if "negative result" means a protected verdict.
- **§6.** It does not overstate. It already disclaims synergy, and its numbers are those in `MOTIVATION_DRAFTS.md` (c). One precision is worth adding after "That justifies evaluating combined outputs": "These pair figures assume perfect linkage, the attacker-favourable limit; with error-prone linkage the pair would be expected to read lower."
- **§7.** This is row 21. The Jayaraman and Evans clause should read as a scope reason only. Nothing else in §7 overstates, and "What is specific to this work is the evaluation, not the adversary" is right.
- **§8.** These are rows 22–23. The hashes and the "full untested-access list" wording must be updated when v2 replaces v1. Suggested "What the diff changes" additions:
  - "Abstract opening: 'often releases' → 'may release' (unsourced frequency claim)";
  - "Introduction: the evaluated question stated positively, other access named as distinct threat models";
  - "Adversary and access: supervised readers and the data budget, the governance clause for linkage, public knowledge of the method versus unused, and the scope sentence".
- **§2 access table and §3 ACS specification.** No overstatement found. The "Why plausible" cells point to "As in §1", which stays valid after the rewrite. The §3 row 7 caveat already lists the untested access neutrally.
- **Naming.** The §1 and §5 headings name the advisor, and so do `roles/adversary_reviewer/README.md` l.13 and `MOTIVATION_DRAFTS.md` (b). The replacement headings above say "the advisor". The role files are historical outputs and were not edited.

## Manuscript patch v2: what changed relative to v1

`MANUSCRIPT_ADVERSARY_PATCH_v2.diff` is a full replacement for `R/MANUSCRIPT_ADVERSARY_PATCH.diff`. Against `3b5065805` it has nine hunks.

**Changed or added hunks.**
1. **New hunk, abstract l.29.** "often releases" becomes "may release" (row 38). It is self-contained and can be dropped without affecting the other hunks.
2. **Introduction paragraph 1.**
   - "normally knows ... can link" becomes "usually knows ... could link".
   - "which lets it learn" becomes "which would let it learn".
   - The closing clause is replaced by the narrow-question sentence and a sentence naming other access as a different threat model, left untested rather than dismissed.
   - The illustrative-scenario disclaimer and "we do not claim that one exists" are kept verbatim.
3. **"Adversary and access" paragraph** (rows 31–36).
   - "In our evaluation"; "no attacker submits inputs of its choosing".
   - Supervised attackers, with the two roles fixing the data budget (pointing to `sec:protocol`). Attacks that need no labelled sample are not evaluated.
   - Governance clause for linkage.
   - The published algorithm and objective may be known, but our attackers do not use it, and none uses fitted parameters or gradients.
   - A scope sentence, with the distinct threat models named (pointing to `sec:limits`), neither claimed nor dismissed.
   - Everything else in the access definition is unchanged: what is received and withheld, `attacker_fit` and `attacker_val`, same population, the grant as an assumption with its citations, perfect linkage, the defense-aware definition, identifiers not used as inputs, and recovery as a measured property.
4. **Limitations sentence** (row 37). Reworded as distinct threat models, untested and not ruled out. It adds public or leaked parameters, chosen-input queries, objective-aware attacks and larger auxiliary samples.

**Hunks kept textually identical to v1:**
- abstract finding (iv);
- Related work ("Adversary models and combined releases");
- Protocol ("Attackers and recovery");
- §5.3 pair paragraph and its caveat;
- the `references.bib` hunk, spliced byte-for-byte from v1 (`li2012multilevel`, `jia2018attriguard`).

Only their line offsets differ.

**Wording choices.**
- In the manuscript I wrote "releases", not "outputs only": the manuscript's contract (2) releases features together with an output, so "outputs only" would be false there.
- "Outputs only" is used in the campaign documents, which describe the ACS service.
- I omitted "auxiliary task labels" from the manuscript's untested list, because the manuscript reports a label-only reference and I did not verify how its labels are supplied.

## Notes for the manuscript owner

- **Data budget.** The manuscript does not state the sizes of `attacker_fit` and `attacker_val`, and the v2 text says only that these roles fix the budget. Committed metadata at `3b5065805` shows that the sizes differ by study:
  - Adult roles in `results/combined_analysis_paper_v1/INDEPENDENT_VERIFICATION.json` are `attacker_fit` 6,065, `attacker_val` 2,235 and cert 1,500;
  - `results/combined_matched_removal_benchmark_v1/EFFECTIVE_PROTOCOL.json` lists `attacker_fit` 7,571.

  I did not verify HMDA or the remaining studies. A per-study size statement, for example in the Roles paragraph, would make the budget explicit. I did not add numbers I had not checked.
- **Pointers.** `\label{sec:threat}` sits in a `\paragraph` of §2, so it prints "Section 2", as in v1. The new `Section~\ref{sec:limits}` prints "Section 6".
- **"Defense-aware".** Use the term only with the block C definition (unchanged from v1).

## Consistency with `CLOSEST_PRIOR_WORK.md` §1 (the Stadler sentence)

- The v2 patch does not touch the Stadler sentence (main.tex l.147–149, "a release useful for a task reveals what the task's label reveals").
- It is consistent with the reviewer's replacement, which treats the label-only adversary as a descriptive reference, not a floor. In v2, a recipient that also holds the target's true label is listed as an untested threat model, and the label-only reference remains a comparison, not part of the attacker's access.
- The two edits do not overlap: the v2 Related-work hunk's context begins at l.153. Either can be applied first; `git apply` handles the line offset.

## Verification record

All checks were run in `<session scratch>`.

1. **Inputs.**
   - `git show 3b5065805:papers/combined_empirical_v1/main.tex` has SHA-256 `76206d1b3b3b48dd87ab1663cf7694aa320bad9183101970f6be62417ee91d65`, matching v1's recorded original.
   - `references.bib` has `86944628be53c27c3c11001406e4b20a294baf3e33142c68e668b570fd818c02`.
2. **Baseline.** The current v1 patch still applies to a fresh extraction and reproduces its recorded outputs: `fedd71ed7f2fe69ada35c4ed5333e840fe0565f04a4425d61cb1b056a7dc8ad7` and `b7043344021b75388abf43e5292305eb6ae13c2c9b309ee793830c616881fea1`.
3. **v2 applies cleanly.**
   - Setup: a fresh directory laid out as `papers/combined_empirical_v1/{main.tex,references.bib}` from `3b5065805`, with a throwaway `git init` and a base commit.
   - `git apply --check` passed, then `git apply` succeeded with no offsets or fuzz.
   - `patch -p1 --dry-run` on a separate fresh copy also applies both files.
   - Patch file SHA-256: `3d296d2be026c381d8c5b3fa7bd6b3c6dc1d7ecd51d0f3efc32526f2b9d4e068` (188 lines).
4. **Resulting hashes.**
   - `main.tex`: `cf0b25f2323b8b78c23af915505d405473a0b3df330f443726e7d44e28918c86`.
   - `references.bib`: `b7043344021b75388abf43e5292305eb6ae13c2c9b309ee793830c616881fea1`. This is unchanged from v1, because the bib hunk is identical.
5. **Keys** (checked by script over the patched files).
   - Every `\ref` has a `\label`: 11 refs, none missing, no duplicate labels.
   - Every `\citep`/`\citet` key is in the patched `references.bib`: 27 keys, none missing, no duplicate entries.
   - Keys used by the new or changed text are all present: `sec:threat`, `sec:protocol`, `sec:limits`, `song2020overlearning`, `jia2018attriguard`.
6. **Scratch build.**
   - Setup: `git archive 3b5065805 papers/combined_empirical_v1` (including `fig_recipient.tex` and the two figure PDFs), with the patched `main.tex` and `references.bib` swapped in.
   - Sequence: `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` (TeX at `/opt/homebrew/bin`). Every step exited 0.
   - Result: **11 pages**, 0 LaTeX warnings (no undefined references or citations, no multiply defined labels), 0 overfull boxes.
   - The only BibTeX warning, "can't use both volume and number fields in gupta2021fcrl", already exists in the original.
   - The unpatched original builds to 10 pages, so the growth is 10 → 11, as with v1.
