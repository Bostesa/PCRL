# Adversary definition and motivation (corrected)

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

## 1. Answer to the advisor's question: why this access, and not simply a stronger attacker

The claims we test are about a **release policy**: what each recipient is sent. The adversary is therefore drawn at the release boundary. It has three parts:

1. **What the policy hands out:** each recipient's own outputs about a person.
2. **Two capabilities the policy cannot take away:**
   - recipients who serve the same person can link their records;
   - a recipient may already know the protected attribute for some of the people it serves, which lets it learn how outputs relate to the attribute.
3. **Nothing else.**

Each element of the access corresponds to one claim:

- **A single recipient's own outputs** test the per-recipient claim. Every legitimate recipient holds them by design, and they are the object the policy controls.
- **Linked outputs of two recipients** test whether per-recipient protection survives pooling. This matters because purpose limitation is written per recipient while the person is shared. Perfect linkage is the attacker-favourable limit, so a failure to protect is not an artefact of linkage error.
- **Auxiliary labelled examples** test no claim directly. They are the measuring instrument for both claims. Without them no reader can be calibrated, and a "protected" verdict could rest on the attacker simply lacking data.

The labelled sample comes from the same population, and that is an assumption the benchmark grants. The experiment does not show that recipients can obtain such data in deployment.

Giving the attacker more would raise recovery, but the increase would measure something else. The target's inputs or labels, model weights or gradients, repeated or chosen queries, and insiders would each measure either a breach of the provider or information already present in the inputs. A release policy controls neither. Those accesses need other controls, so we list them as untested and claim no protection against them.

A stronger attacker within the same access, such as a larger reader slate, is a different matter. It belongs to the threat model, and it is why every recovery figure is a lower estimate rather than a guarantee.

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

This paragraph replaces Introduction paragraph 1 of `papers/combined_empirical_v1/main.tex`. The full patch is in §8.

> Representation-level defenses are usually evaluated on the representation: a linear probe, an in-training adversary, or a certificate about classifiers that read the representation. A release policy, however, is defined by what each recipient receives. As an illustration, and not a description of any documented deployment, consider a service that holds records about people and serves two recipients with different permitted purposes: one receives a prediction of an income band, the other a prediction of an occupation group, and neither may use a declared attribute such as sex or race. For each person it serves, a recipient receives the output of its own task (a score, a probability vector or a decision) and possibly the features that produced it. The policy is written per recipient, but the person is shared. A recipient normally knows whom an output describes, so two recipients that serve the same person can link their outputs by record, and a recipient may know the protected attribute for some of the people it serves, which lets it learn how outputs relate to that attribute. Under these conditions a defense applied to the features addresses only part of what recipients see: the task output may carry what was removed from the features, a recipient may hold features and output together, and two recipients may pool linked outputs. We know of no source that documents a deployment with all of these properties, so we do not claim that one exists; we study the conditional threat model they define (Section 2) and report what a fixed set of attackers recovers under it.

If the ACS output service enters the manuscript, add:

> In our census benchmark the two recipients receive, for each person, the probability vector and decision of an income task and of an occupation task respectively. Both outputs are computed from six permitted survey fields, and neither recipient may use sex. The benchmark grants a coalition the two outputs for the same person, perfectly linked. It also grants every recipient a labelled sample of other people from the same population. It withholds the target's inputs and labels and every model internal, and it releases each person once. These are assumptions of the evaluation, not properties established for any real service.

## 5. Shorter explanation for the advisor

> We draw the adversary where our claims live: at the release. A recipient gets exactly what the policy sends it for its purpose. Two recipients who serve the same person can link what they receive. Any recipient may know the protected attribute for some people, from a same-population labelled sample that the benchmark grants. That sample is what lets a reader turn a release into a guess, so without it a "protected" result could mean only that the attacker lacked data. The single-recipient view tests the per-purpose promise. The linked pair tests whether that promise survives pooling, which a per-recipient policy cannot forbid. We withhold the target's inputs and labels, model internals, repeated queries and insiders. Each of those would measure a breach of the provider or the information already in the inputs, not what the release adds. So we list them as untested instead of claiming protection against them. Within this access, a stronger reader could still recover more, which is why every number is a lower estimate and not a guarantee.

## 6. What the pair-versus-single differences show, and what they do not

On ACS seed 0, the locally protected trees' recipients read 0.558 (income) and 0.582 (occupation), while their linked pair reads 0.657, against the teacher's 0.664. For services without any privacy term, the pair exceeds the best single recipient by only 0.013–0.022.

A defense judged one recipient at a time can therefore leave the linked pair close to the undefended level. That justifies evaluating combined outputs.

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
  - Further antecedents: Fredrikson et al. (USENIX Security 2014), Yeom et al. (arXiv 1709.01604), and Jayaraman and Evans (CCS 2022), whose imputation critique is why target inputs are withheld and label-only references are reported.
- **The coalition is the linked-copies and composition setting.**
  - Ganta, Kasiviswanathan and Smith (KDD 2008): composition attacks.
  - Li, Chen, Li and Zhang (IEEE TKDE 2012), multilevel trust: recipients pool differently perturbed copies of the same records, under a criterion that the pool should do no better than the best single copy.
  - Taylor, Vippathalla and Coon (arXiv 2601.21859, 2026): sequential releases under collusion. This is the work the repository already cites as `taylor2026collusion`.
  - Tian et al. (arXiv 2503.06150): combining fair and biased models.
- **What is specific to this work is the evaluation, not the adversary:** purpose-permission tables, with per-recipient and linked-pair recovery measured on the same models alongside held-out utility.
- **Unidentified citations.** Two references in the reviews are incomplete: a "Zhang et al. 2024" framework in one NeurIPS review, and "[1, 2]" in one AAAI review. They remain unidentified and were not guessed.

## 8. Manuscript patch

`MANUSCRIPT_ADVERSARY_PATCH.diff`, in this directory, is a unified diff against `papers/combined_empirical_v1/{main.tex, references.bib}` at `3b5065805`. The manuscript branch was not modified.

**What the diff changes:**
- **Abstract finding (iv):** the pair is linked by record.
- **Introduction paragraph 1:** replaced by §4 above.
- **New §2 paragraph "Adversary and access"** (`\label{sec:threat}`):
  - what a recipient receives and does not receive;
  - the granted auxiliary data, stated as an assumption;
  - perfect linkage;
  - the narrow meaning of defense-aware;
  - one release per record;
  - identifiers are not attacker inputs.
- **Related work:** the adversary is identified as existing; the multilevel-trust collusion model and AttriGuard are added.
- **Protocol:** attackers are fitted on labelled `attacker_fit` releases; the pair bank contains each single recipient's attacker; the cell-conditional attacker is defined.
- **§5.3:** "link their outputs by record", plus a no-synergy caveat.
- **Limitations:** the full untested-access list.
- **Bibliography:** two entries added, both checked against primary sources.

**Checks:**
- No number changes, and the paper does not become a chronology.
- The diff applies cleanly with `git apply`. The coordinator re-checked this, and the resulting files' SHA-256 values match the reviewer's: `main.tex` `fedd71ed…`, `references.bib` `b7043344…`.
- The reviewer's scratch build compiled with no undefined references. It grows the paper from 10 to 11 pages.

**Not covered by the diff:** the Related-work sentence attributing a label-implied disclosure claim to Stadler et al. is reviewed separately in `CLOSEST_PRIOR_WORK.md`.

## 9. Untested access

Every recovery figure is a measured property of a fixed reader slate under the access above. The following are untested:
- repeated or adaptive queries;
- chosen-input probing;
- white-box access to parameters or gradients;
- insiders;
- recipients that also hold the target's inputs or later learn its true label;
- imperfect linkage;
- coalitions of more than two recipients;
- auxiliary data that are scarcer than ours or drawn from another population.
