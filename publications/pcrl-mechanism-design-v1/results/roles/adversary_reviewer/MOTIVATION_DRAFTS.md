# Motivation drafts: adversary and access

Adversary and methodology reviewer, 2026-10-16. The drafts use reviewer pseudonyms only. Every number below is checked
against the file named next to it.

## (a) Motivation paragraph for the manuscript

This paragraph replaces Introduction paragraph 1 of `papers/combined_empirical_v1/main.tex`, as in `MANUSCRIPT_PATCH.md`.

> Representation-level defenses are usually evaluated on the representation: a linear probe, an in-training adversary,
> or a certificate about classifiers that read the representation. A release policy, however, is defined by what each
> recipient receives. As an illustration, and not a description of any documented deployment, consider a service that
> holds records about people and serves two recipients with different permitted purposes: one receives a prediction of
> an income band, the other a prediction of an occupation group, and neither may use a declared attribute such as sex
> or race. For each person it serves, a recipient receives the output of its own task (a score, a probability vector or
> a decision) and possibly the features that produced it. The policy is written per recipient, but the person is
> shared. A recipient normally knows whom an output describes, so two recipients that serve the same person can link
> their outputs by record, and a recipient may know the protected attribute for some of the people it serves, which
> lets it learn how outputs relate to that attribute. Under these conditions a defense applied to the features
> addresses only part of what recipients see: the task output may carry what was removed from the features, a
> recipient may hold features and output together, and two recipients may pool linked outputs. We know of no source
> that documents a deployment with all of these properties, so we do not claim that one exists; we study the
> conditional threat model they define (Section 2) and report what a fixed set of attackers recovers under it.

The following variant would serve if the ACS output service enters the manuscript:

> In our census benchmark the two recipients receive, for each person, the probability vector and decision of an
> income task and of an occupation task respectively. Both outputs are computed from six permitted survey fields, and
> neither recipient may use sex. The benchmark grants a coalition the two outputs for the same person, perfectly
> linked. It also grants every recipient a labelled sample of other people from the same population. It withholds the
> target's inputs and labels and every model internal, and it releases each person once. These are assumptions of the
> evaluation, not properties established for any real service.

## (b) For the advisor: why this access, and not simply a stronger attacker

The claims we test are about a **release policy**: what each recipient is sent, and what recipients can do with what
they are sent. The threat model is therefore drawn at the release boundary. It contains exactly what the policy hands
out, plus two capabilities that the policy cannot take away: recipients can link records about the same person, and
recipients may already know the attribute for some people. Giving an attacker more would raise recovery, but the
increase would measure a breach of the provider, or information already in the inputs. A release policy controls
neither of those. They need other controls, and we list them as untested instead of claiming protection against them.

| Access element | Protection claim it tests | Why it is the right boundary for that claim | What it deliberately excludes, and why | What remains an assumption |
|---|---|---|---|---|
| Each recipient's own outputs about a person (in ACS: its probability vector and decision) | **Per-recipient:** a recipient permitted purpose p cannot infer the declared attribute from what it is sent for p. In ACS these are the recipient-worsening clauses (`SCIENCE_LOCK.json:286`); in the manuscript, the output-only and features-plus-output surfaces. | Every legitimate recipient holds this by design, and it is the object the policy controls. | The target's inputs and true labels: these would measure what the data reveal, not what the release adds, and a recipient that also holds the inputs is a separate, untested case. Weights, gradients and internal states: these belong to an insider or the provider and bypass the release. | That recipients have no other channel about the person. |
| Linked outputs of two recipients | **Coalition:** per-recipient protection also limits what recipients learn when they pool. In ACS these are the coalition-benefit clauses (`SCIENCE_LOCK.json:286`); in the manuscript, §5.3. | Purpose limitation is enforced per recipient, but the person is shared, and recipients serving the same person normally know whom an output describes. A multi-purpose design is only meaningful if composition is tested. Perfect linkage is the attacker-favourable limit, so a negative result is not an artefact of linkage error. | Coalitions of more than two recipients, partial linkage, and collusion with insiders. | That linkage occurs in a given deployment, and how often. |
| Auxiliary labelled examples from the same population (in ACS: 8,001 people to fit plus 3,000 to select) | No claim directly. They are the **measuring instrument** for both claims, because they turn information present in a release into an inference. | A recipient cannot calibrate a reader without labelled examples. A same-population sample approximates an adversary with distributional knowledge, so a "protected" verdict cannot rest on the attacker lacking data. Prior attack models assume exactly this: people who disclose the attribute, or labelled auxiliary data (see (d)). | The auxiliary people's raw inputs and task labels, which leaves de-censoring attacks untested. Repeated or chosen-input queries. | Availability, size, distributional match, and access to the **same service's** releases for those people. All are granted by the benchmark; the experiment does not show that recipients can obtain them. |

How to read results under this access:
- A low recovery figure says that, under the policy and these two capabilities, a fixed slate of readers recovers
  little. It is not a guarantee.
- A high figure says that the policy does not stop recovery even without any breach.
- The coalition comparison asks whether a per-recipient defense carries over to linked releases. On ACS seed 0 it does
  not: LOCAL_TREE's single recipients read 0.558 and 0.582, while the linked pair reads 0.657.

## (c) Pair-versus-single differences motivate studying combined outputs; they are not a synergy theorem

The table shows SEX recovery (AUC of the reader chosen on ATTACK_VALIDATION, with marginal 95% intervals) from
`results/pcrl_acs_frozen_hybrids_v1/SERVICE_TABLE.csv`.

| Seed | Service (income source / occupation source) | Income recipient | Occupation recipient | Linked pair | Pair minus best single |
|---|---|---|---|---|---|
| 0 | H_LL (LOCAL_TREE / LOCAL_TREE) | 0.558 [0.545, 0.571] | 0.582 [0.568, 0.597] | 0.657 [0.644, 0.670] | +0.074 |
| 0 | H_JJ (JOINT_TREE / JOINT_TREE) | 0.535 | 0.560 | 0.603 | +0.043 |
| 0 | TASK_REFIT (no privacy term) | 0.572 | 0.650 | 0.664 | +0.014 |
| 0 | T_TREE (teacher) | 0.569 | 0.650 | 0.664 | +0.013 |
| 1 | H_LL | 0.549 | 0.584 | 0.649 | +0.065 |
| 1 | T_TREE | 0.568 | 0.649 | 0.664 | +0.016 |

What the numbers motivate: a defense judged one recipient at a time can lower each recipient's recovery while the
linked pair stays close to the undefended level (0.657 against the teacher's 0.664 on seed 0). Combined outputs
therefore need their own evaluation. The manuscript's stored-model pair (main.tex §5.3) points the same way.

Why this is not an information-theoretic synergy result:
1. AUC measures how well a fitted reader ranks people; it is not mutual information. A pair AUC above both single AUCs
   is expected whenever the two outputs carry partly non-redundant evidence, for example two conditionally independent
   noisy signals of SEX. It does not show information that is available only jointly, which is synergy in the
   partial-information sense.
2. The readers form a finite slate chosen on validation, and the pair bank includes the single-recipient readers.
   Different readers could change every number.
3. The gap is consistent with being produced by the defense rather than by the data. LOCAL's training critics
   penalised single-view readability only (`acs_replication/leafcorr.py:16-19`, arm `L_TREE`, `:49`). Single-view AUCs
   fell more than the pair's, and for services without a privacy term the gap is 0.013–0.022 (TASK_REFIT and T_TREE,
   both seeds). This is not decomposed further.
4. The intervals are marginal and conditional on the fitted models and readers. The two seeds share people.
   DEV_REPORT had already been opened, so the hybrid figures are exploratory (`ANALYSIS_LOCK.json:1133, 1147`).

A synergy statement would need an estimate of `I(S; W_inc, W_occ)` together with a decomposition into redundant,
unique and synergistic parts, with uncertainty. That has not been attempted.

## (d) This adversary is not new: prior models, with sources checked on 2026-10-16

**Attribute inference with auxiliary knowledge**
- Fredrikson, Lantz, Jha, Lin, Page, Ristenpart, *Privacy in Pharmacogenetics: An End-to-End Case Study of Personalized
  Warfarin Dosing*, USENIX Security 2014 —
  [PDF](https://www.usenix.org/system/files/conference/usenixsecurity14/sec14-paper-fredrikson-privacy.pdf). The
  attacker holds the model plus "some demographic information about a patient". This is stronger target knowledge than
  ours, and there is no coalition.
- Song and Shmatikov, *Overlearning Reveals Sensitive Attributes*, ICLR 2020 —
  [arXiv:1905.11742](https://arxiv.org/abs/1905.11742) (the v3 PDF is headed "Published as a conference paper at ICLR
  2020"). In §3.1 the adversary observes the target's representation and holds "an auxiliary set D_aux of labeled
  (x, s) pairs" plus a black-box oracle; D_aux "need not be drawn from the same dataset". **Our recipient is this
  adversary applied to released outputs**, given auxiliary releases instead of an oracle, and without de-censoring.
- Gong and Liu, *You are Who You Know and How You Behave: Attribute Inference Attacks via Users' Social Friends and
  Behaviors*, USENIX Security 2016 — [arXiv:1606.05893](https://arxiv.org/abs/1606.05893).
- Jia and Gong, *AttriGuard: A Practical Defense Against Attribute Inference Attacks via Adversarial Machine Learning*,
  USENIX Security 2018 — [arXiv:1805.04810](https://arxiv.org/abs/1805.04810). The attacker "collects a dataset from
  users who disclose their attributes" and trains a classifier on it. This is the standard plausibility argument for
  auxiliary labels.
- Yeom, Giacomelli, Fredrikson, Jha, *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting* —
  [arXiv:1709.01604](https://arxiv.org/abs/1709.01604). A formal attribute-inference adversary. The arXiv page lists
  no venue, so the venue is UNVERIFIED.
- Jayaraman and Evans, *Are Attribute Inference Attacks Just Imputation?*, CCS 2022 —
  [arXiv:2209.01292](https://arxiv.org/abs/2209.01292). An adversary's distributional knowledge matters, and black-box
  attacks rarely beat imputation. This is why we withhold target inputs and report label-only and constant references.

**Collusion, linkage and composition**
- Ganta, Kasiviswanathan, Smith, *Composition Attacks and Auxiliary Information in Data Privacy*, KDD 2008, pp. 265–273,
  [doi:10.1145/1401890.1401926](https://doi.org/10.1145/1401890.1401926) (metadata confirmed through Crossref);
  [arXiv:0803.0032](https://arxiv.org/abs/0803.0032). The attacker combines independent releases of overlapping
  populations with auxiliary information.
- Li, Chen, Li, Zhang, *Enabling Multilevel Trust in Privacy Preserving Data Mining*, IEEE TKDE 24(9):1598–1612, 2012,
  [doi:10.1109/TKDE.2011.124](https://doi.org/10.1109/TKDE.2011.124) (Crossref);
  [arXiv:1104.0459](https://arxiv.org/abs/1104.0459). Recipients at different trust levels get differently perturbed
  copies of the same data, and a malicious recipient may combine copies. Their mechanism prevents joint reconstruction
  more accurate than the best single copy. **This is the closest prior coalition model for linked releases of the same
  records.** Our "pair minus best single" increment mirrors their criterion, although they release perturbed data and
  measure reconstruction error.
- Taylor, Vippathalla, Coon, *Adaptive Privacy of Sequential Data Releases Under Collusion* —
  [arXiv:2601.21859](https://arxiv.org/abs/2601.21859) (v1 29 Jan 2026, v2 10 Jul 2026). Sequential releases to
  parties that may collude, with mutual-information privacy and Blahut–Arimoto-style adaptive releases. **This is the
  sequential-collusion work already discussed in the repository**: see
  `results/pcrl_learned_decoder_constrained_release_v1/PRIOR_ART_AND_CLAIM_SCOPE.md:27, 89, 172` (no code release
  found), and it is cited at `main.tex:158-159` as `taylor2026collusion`.
- Tian, Zhang, Liu, Zhu, Ding, Zhou, *Do Fairness Interventions Come at the Cost of Privacy: Evaluations for Binary
  Classifiers* — [arXiv:2503.06150](https://arxiv.org/abs/2503.06150), accepted to IEEE TDSC per the arXiv page. The
  attack exploits prediction discrepancies between fair and biased models. Its access details beyond the abstract are
  UNVERIFIED.
- Narayanan and Shmatikov, *How To Break Anonymity of the Netflix Prize Dataset* —
  [arXiv:cs/0610105](https://arxiv.org/abs/cs/0610105). Record linkage needs little background knowledge.
- Internal predecessor: the NeurIPS submission's aggregating adversary (§5.5 and Prop. 6), which holds every purpose's
  representation.

**Context: downstream consumers and post-hoc readers**
- Jovanović, Balunović, Dimitrov, Vechev, *FARE: Provably Fair Representation Learning with Practical Certificates*,
  ICML 2023 — [arXiv:2210.07213](https://arxiv.org/abs/2210.07213).
- Elazar and Goldberg, *Adversarial Removal of Demographic Attributes from Text Data*, EMNLP 2018, pp. 11–21 —
  [ACL Anthology D18-1002](https://aclanthology.org/D18-1002/).

**Not identified, and not guessed**
- A 2024 information-theoretic framework attributed to "Zhang et al." in one NeurIPS review (N-R2). The review gives
  no full citation.
- References "[1,2]" in one AAAI review (A-R1). The export has no bibliography.

**Where the novelty lies.** The recipient is the auxiliary-data attribute-inference adversary. The coalition is the
linked-copies and composition setting. What is specific to this work is the evaluation, not the adversary: purpose
permission tables, with per-recipient and linked-pair recovery measured on the same models alongside held-out utility.
