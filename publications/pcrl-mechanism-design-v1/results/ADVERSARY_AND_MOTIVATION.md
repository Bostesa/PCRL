# Adversary definition and motivation (corrected)

> **Corrected again on 2026-10-16 in the correction pass** (`CORRECTIONS.md` §6; review in `roles/correction_v1/adversary_reviewer/ADVERSARY_REVIEW.md`).
> - **Unchanged:** the concrete access definition.
> - **Rewritten:** §1, §4, §5 and §9.
> - **What was wrong:**
>   - other access was dismissed as a provider breach;
>   - the labelled sample was called necessary for any attacker;
>   - linkage and pooling were called impossible to forbid;
>   - the linkage argument pointed the wrong way;
>   - the patch said no attacker knows the training objective.
> - **Smaller precisions:** §6, §7 and §8.
> - The manuscript patch is now v2. The original text is at public commit `e141330`.

**Who did what.**
- **The adversary and methodology reviewer** read the two submitted papers, the code behind each, the combined manuscript and the current ACS output service. Line-level evidence is in `roles/adversary_reviewer/`:
  - `ACCESS_TABLE.md`;
  - `ACS_SERVICE_ADVERSARY_SPEC.md`;
  - `MOTIVATION_DRAFTS.md`;
  - `MANUSCRIPT_PATCH.md`.
- **The coordinator** checked that the manuscript patch applies cleanly to `main.tex` at `3b5065805` and reproduced the reviewer's output hashes. The coordinator also re-read three of the reviewer's code claims at the cited lines:
  1. the NeurIPS concatenation attack keeps the best of three auditor seeds by test accuracy;
  2. the AAAI audit's stop rule reads only the representation;
  3. race is disallowed for all three HMDA purposes.

## 1. Answer to the advisor's question: why this particular access is relevant, rather than simply why a stronger attacker can recover more

The claims we test concern a **release policy**: what each recipient is sent. We therefore chose to evaluate an adversary that holds what is sent, plus two capabilities that the benchmark grants rather than rules out. The access has three parts:

1. **What the policy hands out:** each recipient's own outputs about a person, one fixed release per person.
2. **Two capabilities the evaluation grants:**
   - recipients who serve the same person may link their records. We do not assume that contracts or governance prevent this, and we grant perfect linkage, which is the attacker-favourable limit;
   - a recipient may already know the protected attribute for some people. The benchmark models this with a fixed labelled sample of other people from the same cohort: in ACS, 8,001 people to fit readers and 3,000 to select and orient them.
3. **Nothing else.** This is the scope of the test, not a claim that other access is unrealistic.

Each element corresponds to one claim:

- **A single recipient's own outputs** test the per-recipient claim. Every legitimate recipient holds them by design, and they are what the policy controls.
- **Linked outputs of two recipients** test whether per-recipient protection carries over when recipients pool. Purpose limitation is written per recipient, but the person is shared. Perfect linkage is the attacker-favourable limit: a low pair recovery cannot be blamed on poor linkage, while a high one may overstate what a coalition with imperfect linkage would recover.
- **The labelled sample** tests no claim directly. Our readers are supervised, so the sample is what lets this particular reader bank turn outputs into guesses, and its size is the attacker's data budget. Granting a generous same-population sample keeps a "protected" verdict from resting on readers that had too little data, for this slate and this budget. Other attacks, such as ones built from a known population law or from analytic structure, need no such sample; none was run.

The labelled sample comes from the same population, and that is an assumption the benchmark grants. The experiment does not show that recipients can obtain such data in deployment.

**Why this access is relevant.** Together these elements isolate one narrow, well-posed question: what can be inferred about the protected attribute from the delivered outputs alone, by one recipient or by a linked pair, under a declared labelled-data budget. The answer does not depend on assumptions about what else a recipient holds about the target.

**Why not simply a stronger attacker.** More access cannot lower an optimal attacker's recovery, so it could raise it. It need not: re-querying a deterministic release returns the same output. Either way it would answer a different question. The following are each distinct threat models:
- a recipient that also holds the target's inputs or later learns its label;
- public model parameters;
- permitted repeated or chosen queries;
- attacks that use the published objective or the fitted parameters;
- insiders.

Several of these involve no breach of the provider: a recipient may legitimately hold a person's inputs, a model may be published, and a service may permit repeated queries. Some are governed by release-policy choices, such as rate limits, one release per person, query auditing and whether to publish the model. Others need security controls. All of them are outside this evaluation's scope. We list them as untested in §9, claim no protection against them, and do not dismiss them.

A stronger attacker within the same access is a different matter. A larger reader slate belongs to this threat model, and so do readers that use public knowledge of the defense's algorithm and objective while seeing only the same outputs. None of our readers uses that knowledge beyond refitting to the defended releases. This is why every recovery figure is a lower estimate rather than a guarantee. A larger labelled sample changes the declared budget, and is listed with the untested variants in §9.

**Provenance of the question.** The recorded 2026-10-01 meeting transcript and the review exports do not contain this question in these words. The nearest recorded passages are the advisor asking for carefully specified adversaries, access and evaluation settings. Pointers to those passages are kept in a private, git-ignored note. The question is answered here as it was posed in this campaign's brief.

## 2. Access table: what each source says vs what its code gives the attacker

The full table, with `file:line` evidence, is `roles/adversary_reviewer/ACCESS_TABLE.md`.

| Source | What the paper says | What the code gives the attacker | Why plausible | Unverified | Claims that apply |
|---|---|---|---|---|---|
| **NeurIPS submission** ("One Encoder, Many Purposes"), linear certificate | §4.3: R²(h_p, A) ≤ ε bounds every linear classifier (Prop. 3) | A ridge fit scored **in-sample on the labelled test rows**, so the "linear attacker" uses the targets' own attribute labels | A consumer of h_p with labelled examples can fit a linear probe | That in-sample R² tracks held-out R² | Only the in-sample least-squares statement. **Prop. 3 was later shown false**, and its retirement exists only on an unmerged branch |
| NeurIPS, nonlinear auditors | "linear, MLP and gradient-boosted auditors" | LR, RF, RBF-SVM (+ XGBoost if importable), fitted on training-split representations. **There is no MLP**, so the paper and code differ | Off-the-shelf readers on labelled data | Which auditors actually ran is not stored | Empirical lower bounds only |
| NeurIPS, aggregating adversary | Holds every purpose's representation (§5.5, Prop. 6) | Row-aligned concatenation of all purposes, so linkage is perfect. **The best of 3 auditor seeds is chosen by test accuracy** | Several purposes' consumers pool | That any party receives every representation | Prop. 6 (linear) and empirical readings. The motivating "auditor must see race" conflict is not instantiated: race is disallowed for all three HMDA purposes |
| **AAAI submission** ("Outputs Leak What They Use"), Tier 1 | Attackers on representation h and outputs ŷ, fitted on 75% and scored on 25% | Rows come from the model's **training partition**. "ŷ" is the **full logits**, attacked separately from h | A vendor exposes embeddings or scores | Off-partition generalisation (checked later per the paper) | Empirical surviving points |
| AAAI, the 59/67 audit | Any attacker, on either surface | The stop rule is max(XGB, MLP) ≤ 0.55 on the **representation only**, so the paper and code differ | — | — | Only the representation-surface statement |
| AAAI, Tier 2 "defense-aware" | Knows how the defense works | A Gaussian LRT that knows the noise law. For deterministic defenses it reduces to a QDA probe, so the paper and code differ | Pipeline access | — | Noise channels only. There is no general adaptive or white-box attacker |
| **Combined manuscript** (`3b5065805`) | Three recipient contracts, four output formats, one slate, ignore-one candidates. **No explicit threat-model statement** | Fitted on `attacker_fit` releases with the attribute, selected on `attacker_val`. The pair is a row-asserted concatenation (perfect linkage) and the bank includes each single recipient's attacker | As in §1 | Auxiliary labelled data and perfect linkage are implicit, and the untested list is incomplete | Native checks cover only their own access. None covers outputs or coalitions |
| **Current ACS output service** | Release contract, threat and claims in the source SCIENCE_LOCK | See §3: all nine bullets hold in code | As in §1 | Availability and match of auxiliary data; linkage in deployment | Recipient-worsening clauses (single-recipient access) and coalition-benefit clauses (coalition access) |

**Terminology hazard.** "Defense-aware" means three different things across these sources:
- the AAAI Tier-2 Gaussian LRT;
- the combined study's cell-conditional attacker;
- a feature transform in the TCR1 service.

The ACS code never uses the term. Define it once, as "readers refitted to the actual defended releases", or avoid it.

## 3. The current ACS output service: verified specification

The release is computed from x = (AGEP, WKHP, COW, SCHL, MAR, POBP). SEX is never an input.

| # | Statement | Verified in code | Caveat |
|---|---|---|---|
| 1 | Each recipient receives its own task probabilities and argmax decision about a person | yes (reader views are built per task; the bank holds probability vectors only) | The base deployment call returns both tasks to one caller, so routing is an operator duty there. The H_LJ package releases one task per recipient |
| 2 | A coalition receives the perfectly linked pair | yes (same-row concatenation after row-order checks) | No linkage-error model; no coalitions larger than two |
| 3 | Auxiliary SEX-labelled examples are **granted** | yes: 8,001 people to fit and 3,000 to select, from random household splits of the same cohort | Availability, size, distributional match and access to the same service's releases are assumptions. Readers get no auxiliary task labels or raw inputs, so de-censoring attacks are untested |
| 4 | Readers fit on ATTACK_FIT, then are selected and oriented on ATTACK_VALIDATION | yes | A finite slate with fixed hyperparameters |
| 5 | Readers receive no target raw inputs, target SEX, true task labels, weights, gradients, teacher internals, leaf states or hidden representations | yes, as reader inputs | The reader process loads the target feature export only for row and household ids. X never reaches a reader |
| 6 | Household and person identifiers enforce separation and linkage and are not reader features | yes | — |
| 7 | One fixed, deterministic release per person | yes | Repeated or chosen queries, white-box access, insiders, and recipients that hold x or later learn the outcome are untested |
| 8 | "Defense-aware" means refitting readers to the defended auxiliary releases | correct in substance; the term is not in the ACS code | Not adaptive or white-box access |
| 9 | The coalition bank includes readers that ignore either recipient | yes: 7 pair readers plus 14 ignore-one aliases | Selection is on validation, so the reported pair AUC need not be at least the best single-recipient AUC |

## 4. Manuscript-ready motivation paragraph

This paragraph replaces Introduction paragraph 1 of `papers/combined_empirical_v1/main.tex`. The full patch is `MANUSCRIPT_ADVERSARY_PATCH.diff` (§8).

> Representation-level defenses are usually evaluated on the representation: a linear probe, an in-training adversary, or a certificate about classifiers that read the representation. A release policy, however, is defined by what each recipient receives. As an illustration, and not a description of any documented deployment, consider a service that holds records about people and serves two recipients with different permitted purposes: one receives a prediction of an income band, the other a prediction of an occupation group, and neither may use a declared attribute such as sex or race. For each person it serves, a recipient receives the output of its own task (a score, a probability vector or a decision) and possibly the features that produced it. The policy is written per recipient, but the person is shared. A recipient usually knows whom an output describes, so two recipients that serve the same person could link their outputs by record, and a recipient may know the protected attribute for some of the people it serves, which would let it learn how outputs relate to that attribute. Under these conditions a defense applied to the features addresses only part of what recipients see: the task output may carry what was removed from the features, a recipient may hold features and output together, and two recipients may pool linked outputs. We know of no source that documents a deployment with all of these properties, so we do not claim that one exists. We study one narrow question that these conditions raise: what a fixed set of attackers, each fitted on labelled releases of other records from the same population, recovers from the releases themselves, held by one recipient or by two whose releases are perfectly linked (Section 2). Other access, such as a recipient that also holds a person's inputs, public model parameters or permission to query repeatedly, defines a different threat model, which we leave untested rather than dismiss.

If the ACS output service enters the manuscript, add:

> In our census benchmark the two recipients receive, for each person, the probability vector and decision of an income task and of an occupation task respectively. Both outputs are computed from six permitted survey fields, and neither recipient may use sex. The benchmark grants a coalition the two outputs for the same person, perfectly linked; it does not assume that contracts or governance would prevent such linkage. It also grants the attacker a fixed labelled sample of other people from the same cohort, with their releases and their sex: 8,001 people to fit attackers and 3,000 to choose and orient them. This sample is the attacker's data budget. The benchmark withholds the target's inputs and labels and every model internal, and it releases each person once. These are assumptions of the evaluation, not properties established for any real service. The benchmark therefore answers one narrow question, namely what attackers with this budget recover from the delivered outputs alone. Other access is a distinct threat model that it leaves untested.

## 5. Shorter explanation for the advisor

> We draw the adversary at the release, because that is where our claims are made. A recipient gets exactly what the policy sends it for its purpose, once per person. Two recipients who serve the same person may link what they receive. We grant perfect linkage, the worst case for the defender, and do not assume contracts would stop it. Our readers learn from a labelled sample of other people from the same population, which the benchmark grants: 8,001 to fit and 3,000 to select. The single recipient tests the per-purpose promise; the linked pair tests whether it holds when recipients pool. This isolates one narrow question: what the delivered outputs alone reveal, given that much labelled data. Other access, such as a recipient that also holds the person's inputs, a public model, permitted repeated queries, or an insider, defines a different threat model and need not involve a breach. We leave those untested, not dismissed. Within our access a stronger reader could recover more, so every number is a lower estimate, not a guarantee.

## 6. What the pair-versus-single differences show, and what they do not

On ACS seed 0, the locally protected trees' recipients read 0.558 (income) and 0.582 (occupation), while their linked pair reads 0.657, against the teacher's 0.664. For services without any privacy term, the pair exceeds the best single recipient by only 0.013–0.022.

A defense judged one recipient at a time can therefore leave the linked pair close to the undefended level. That justifies evaluating combined outputs. These pair figures assume perfect linkage, the attacker-favourable limit; with error-prone linkage the pair would be expected to read lower.

It is not an information-theoretic synergy result:
- **AUC is not mutual information.** A pair AUC above both singles is expected whenever the outputs carry partly non-redundant evidence.
- **The readers are a finite slate selected on validation.**
- **Gaps of this size can arise from how the local defense was trained,** not only from the data.
- **The intervals are conditional.** The seeds share people, and DEV_REPORT was already opened.

A synergy statement would need an estimate of I(S; W_inc, W_occ) with a redundant, unique and synergistic decomposition, with uncertainty. None was attempted.

## 7. This adversary is not new

Primary sources were checked on 2026-10-16; `roles/adversary_reviewer/MOTIVATION_DRAFTS.md` §d has the links.

- **The recipient is the auxiliary-data attribute-inference adversary.**
  - Song and Shmatikov (ICLR 2020, overlearning): an adversary with labelled (x, s) pairs reads the attribute from a representation.
  - Jia and Gong (USENIX Security 2018, AttriGuard): attackers train on users who disclose the attribute.
  - Further antecedents: Fredrikson et al. (USENIX Security 2014), Yeom et al. (arXiv 1709.01604), and Jayaraman and Evans (CCS 2022). Their imputation critique is one reason this evaluation withholds target inputs, because it isolates what the outputs reveal; it is also why label-only references are reported. A recipient that holds the inputs is a separate, untested threat model, not an illegitimate one.
- **The coalition is the linked-copies and composition setting.**
  - Ganta, Kasiviswanathan and Smith (KDD 2008): composition attacks.
  - Li, Chen, Li and Zhang (IEEE TKDE 2012), multilevel trust: recipients pool differently perturbed copies of the same records, under a criterion that the pool should do no better than the best single copy.
  - Taylor, Vippathalla and Coon (arXiv 2601.21859, 2026): sequential releases under collusion. This is the work the repository already cites as `taylor2026collusion`.
  - Tian et al. (arXiv 2503.06150): combining fair and biased models.
- **What is specific to this work is the evaluation, not the adversary:** purpose-permission tables, with per-recipient and linked-pair recovery measured on the same models alongside held-out utility.
- **Unidentified citations.** Two references in the reviews are incomplete: a "Zhang et al. 2024" framework in one NeurIPS review, and "[1, 2]" in one AAAI review. They remain unidentified and were not guessed.

## 8. Manuscript patch (v2)

`MANUSCRIPT_ADVERSARY_PATCH.diff`, in this directory, is a unified diff against `papers/combined_empirical_v1/{main.tex, references.bib}` at `3b5065805`. It is version 2 (correction pass, 2026-10-16) and replaces v1, which remains at public commit `e141330`. The manuscript branch was not modified.

**What the diff changes:**
- **Abstract opening (new in v2, optional):** "often releases" becomes "may release". The original frequency claim had no source and conflicts with the Introduction's statement that no documented deployment is known. The hunk is self-contained and can be dropped.
- **Abstract finding (iv):** the pair is linked by record.
- **Introduction paragraph 1:** replaced by §4 above. The evaluated question is stated positively, and other access is named as a distinct threat model, left untested rather than dismissed.
- **New §2 paragraph "Adversary and access"** (`\label{sec:threat}`):
  - what a recipient receives and does not receive;
  - the attackers are supervised, and `attacker_fit` and `attacker_val` fix their data budget;
  - attacks that need no labelled sample are not evaluated;
  - the grant is stated as an assumption;
  - perfect linkage, without assuming that governance prevents it;
  - the narrow meaning of defense-aware;
  - a recipient may know a published algorithm and objective, but our attackers do not use it, and none uses fitted parameters or gradients;
  - one release per record;
  - identifiers are not attacker inputs;
  - a scope sentence naming the distinct threat models.
- **Related work:** the adversary is identified as existing; the multilevel-trust collusion model and AttriGuard are added.
- **Protocol:** attackers are fitted on labelled `attacker_fit` releases; the pair bank contains each single recipient's attacker; the cell-conditional attacker is defined.
- **§5.3:** "link their outputs by record", plus a no-synergy caveat.
- **Limitations:** the threat models outside scope, listed as untested rather than ruled out. Public or leaked parameters, chosen-input queries, objective-aware attacks and larger auxiliary samples are added.
- **Bibliography:** two entries added, both checked against primary sources. This hunk is byte-identical to v1.

**Checks (v2):**
- No number changes, and the paper does not become a chronology.
- The diff applies cleanly with `git apply` to a fresh extraction of `3b5065805`. The reviewer checked this, and so did the coordinator independently.
- The resulting SHA-256 values are `main.tex` `cf0b25f2323b8b78c23af915505d405473a0b3df330f443726e7d44e28918c86` and `references.bib` `b7043344021b75388abf43e5292305eb6ae13c2c9b309ee793830c616881fea1` (unchanged from v1).
- Every `\ref`, `\label` and `\citep` key resolves. The reviewer's scratch build compiles with 0 LaTeX warnings and grows the paper from 10 to 11 pages, as v1 did.
- **For the manuscript owner:** the paper does not state the `attacker_fit` and `attacker_val` sizes, which differ by study. A per-study size statement would make the budget explicit; the patch adds no unverified number.

**Not covered by the diff:** the Related-work sentence attributing a label-implied disclosure claim to Stadler et al. is reviewed separately in `CLOSEST_PRIOR_WORK.md`. It is consistent with v2, and the two edits do not overlap.

## 9. Threat models outside this evaluation's scope

Every recovery figure is a measured property of a fixed reader slate under the access in §1. The following are distinct threat models that this evaluation does not test. They are not dismissed: several involve no breach of the provider, and some are governed by release-policy choices, such as rate limits, one release per person, query auditing and whether to publish the model. We claim no protection against any of them.
- **Target-input side information:** recipients that also hold the target's raw inputs, or later learn its true label or outcome.
- **Public or leaked parameters:** access to the model's parameters or gradients, whether published or obtained otherwise.
- **Repeated or chosen queries:** permitted repeated, adaptive or chosen-input queries, and releases of the same person from several model versions. Repeating a query to the current deterministic service returns the same release.
- **Objective-aware or parameter-aware attacks:** readers that exploit the published training algorithm or objective, or the fitted parameters, beyond refitting to the defended releases. Recipients may know a published method; our readers simply do not use it.
- **Insiders** at the provider.
- **Imperfect linkage:** coalitions that link records with error. Our pair is perfectly linked, which is the attacker-favourable limit.
- **Larger coalitions:** more than two recipients.
- **Different auxiliary data:**
  - samples larger or smaller than ours (8,001 to fit, 3,000 to select);
  - samples drawn from another population or period;
  - samples that include the auxiliary people's raw inputs or task labels, as in de-censoring attacks;
  - attacks that use no labelled sample at all, such as ones built from a known population law.
