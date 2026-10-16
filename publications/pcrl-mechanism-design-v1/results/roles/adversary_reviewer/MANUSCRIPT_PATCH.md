# Manuscript patch: adversary definition and motivation

Target: `papers/combined_empirical_v1/main.tex` and `references.bib` on `origin/research/combined-analysis-paper-v1` at
`3b5065805`. That branch belongs to the manuscript owner, so I have not modified it. This file is a proposal for the
coordinator.

- Original `main.tex` SHA-256 `76206d1b3b3b48dd87ab1663cf7694aa320bad9183101970f6be62417ee91d65`.
- Patched `main.tex` `fedd71ed7f2fe69ada35c4ed5333e840fe0565f04a4425d61cb1b056a7dc8ad7`; patched `references.bib`
  `b7043344021b75388abf43e5292305eb6ae13c2c9b309ee793830c616881fea1`.

**Checked on a scratch copy.** The diff below applies cleanly with `git apply` to a fresh `git archive` of
`3b5065805`. The patched paper builds with `pdflatex`, then `bibtex`, then `pdflatex` twice. It produces no undefined
references or citations; the only BibTeX warning, on `gupta2021fcrl`, already exists in the original. Length grows from
10 to 11 pages. No reported number changes.

## Locations, with line numbers in `main.tex` at `3b5065805`

| Block | Section / label | Lines | Change and reason |
|---|---|---|---|
| A | Abstract | 45 | Finding (iv) now states that the pair is linked by record and that the result is a property of our attackers. |
| B | §1 Introduction | 52–56 (paragraph 1) | Replaces unsourced generalisations about deployed systems ("typically", "often", "sees more than any one channel") with an illustrative scenario, explicitly labelled as such, and a statement that we study a conditional threat model. |
| C | §2 Setting (`sec:setting`) | insert after 105 | New `\paragraph{Adversary and access.}\label{sec:threat}`. It states what a recipient receives and what it does not, the granted labelled auxiliary data (an assumption, not a demonstration), perfect linkage for the coalition, the narrow sense in which attackers are "defense-aware", one release per record, and that identifiers are not attacker inputs. |
| D | §3 Related work (`sec:related`) | 156–159 | Retitled. States that the recipient is the existing auxiliary-data attribute-inference adversary (Song and Shmatikov; Jia and Gong). Adds the multilevel-trust collusion model (Li et al. 2012). Keeps Ganta, Tian and Taylor. |
| E | §4 Protocol (`sec:protocol`) | 178–183 | Attackers are stated to be fitted on `attacker_fit` releases labelled with the attribute. The pair bank is stated to contain each single recipient's attacker. The cell-conditional attacker is defined. |
| F | §5.3 Combining recipients (`sec:pairs`) | 349–354 | Adds "link their outputs by record". Adds a closing caveat: these are measurements with a fixed attacker slate, AUC is not an information measure, and there is no synergy or composition result. |
| H | §6 Limitations (`sec:limits`) | 384–385 | Replaces "We test no repeated-query or adaptive-query attacker" with the full list of untested access. |
| bib | `references.bib` | append | `li2012multilevel` (Crossref metadata: TKDE 24(9):1598–1612) and `jia2018attriguard` (arXiv:1805.04810, USENIX Security 2018). Both sources were checked; see `MOTIVATION_DRAFTS.md` §d. |

Each statement in block C traces to code at `3b5065805`:
- Fit, select and score roles: `oar/study.py:277-322`.
- Linkage asserted by identical rows, plus single-recipient candidates in the pair bank: `odx/run.py:193-216`.
- Formats: `odx/surfaces.py:37-45`.
- Cell-conditional attacker: `oar/study.py:179-219`.
- Slate: `stored_model_eval/bench.py:648-662`.

The same text also matches the ACS service, with the line evidence given in `ACS_SERVICE_ADVERSARY_SPEC.md`.

## Unified diff, applicable with `git apply` from the repository root

```diff
--- a/papers/combined_empirical_v1/main.tex
+++ b/papers/combined_empirical_v1/main.tex
@@ -42,7 +42,8 @@
 (untreated 0.774), while FARE's recovers 0.539 at an accuracy cost whose non-inferiority within one point is not established.
 A same-size tree without the fairness term recovers 0.648, so on this cell the fairness term removes 0.109 AUC beyond
 compression, whereas on an easier task compression reproduced most of FARE's effect.
-(iv) Two recipients pooling their outputs recover more than either alone, including from decisions only.
+(iv) When two recipients link their outputs by record, our attackers recover more from the pair than from either
+output alone, including from decisions only.
 All results are development evidence on repeatedly used data. We propose no new algorithm, no population privacy
 certificate and no composition theorem.
 \end{abstract}
@@ -50,10 +51,19 @@
 \section{Introduction}
 
 Representation-level defenses are usually evaluated on the representation: a linear probe, an in-training adversary,
-or a certificate about classifiers that read the representation. Deployed systems release more. A recipient permitted
-a purpose typically receives a task output for that purpose---a score, a probability or a decision---and often the
-features that produced it. Whatever is withheld from the features may be carried by the output, and a recipient who
-holds both, or who pools releases with another recipient, sees more than any one channel.
+or a certificate about classifiers that read the representation. A release policy, however, is defined by what each
+recipient receives. As an illustration, and not a description of any documented deployment, consider a service that
+holds records about people and serves two recipients with different permitted purposes: one receives a prediction of
+an income band, the other a prediction of an occupation group, and neither may use a declared attribute such as sex
+or race. For each person it serves, a recipient receives the output of its own task (a score, a probability vector or
+a decision) and possibly the features that produced it. The policy is written per recipient, but the person is shared.
+A recipient normally knows whom an output describes, so two recipients that serve the same person can link their
+outputs by record, and a recipient may know the protected attribute for some of the people it serves, which lets it
+learn how outputs relate to that attribute. Under these conditions a defense applied to the features addresses only
+part of what recipients see: the task output may carry what was removed from the features, a recipient may hold
+features and output together, and two recipients may pool linked outputs. We know of no source that documents a
+deployment with all of these properties, so we do not claim that one exists; we study the conditional threat model
+they define (Section~\ref{sec:threat}) and report what a fixed set of attackers recovers under it.
 
 This paper reports a controlled empirical analysis of that complete view on stored models. We evaluate four bodies of
 work in our repositories that were developed separately and are \emph{not} connected by any pipeline: purpose-specific
@@ -104,6 +114,25 @@
 invariance~\citep[\S4.1]{goodfellow2016deep}). The formats are \emph{full} logits, \emph{centred} logits,
 \emph{probabilities}, and the \emph{hard} decision.
 
+\paragraph{Adversary and access.}\label{sec:threat} A \emph{recipient} is a party permitted one purpose. For each
+scored record it receives that purpose's release under the contract in force, and nothing else about the record: not
+its inputs, its true task label or its protected attribute, and not the model's parameters, gradients or internal
+states (defense maps, heads, or a representation that the contract does not release). Each record is released once,
+and the recipient cannot submit inputs of its choosing. To turn a release into a guess about the protected attribute,
+the recipient is granted labelled auxiliary data: the releases of the records in \texttt{attacker\_fit} together with
+their protected attribute, and a second labelled set, \texttt{attacker\_val}, on which it chooses and orients its
+attacker. Both come from the same population as the scored records. This grant is an assumption of the benchmark. It
+models a recipient that knows the attribute for some of the people it serves, as in attribute-inference work in which
+attackers train on labelled examples or on people who disclosed the attribute
+\citep{song2020overlearning,jia2018attriguard}; we do not show that a recipient can obtain such data, in this amount
+or from the same population, in practice. A \emph{coalition} of two recipients holds both recipients' releases for the
+same record, perfectly linked; this is the attacker-favourable limit of record linkage, and we model no linkage error.
+Attackers are refitted to the releases of the defended system and so adapt to the distribution that the defense emits;
+only in that limited sense are they defense-aware. None knows a defense's parameters or training objective, and no
+white-box, adaptive-query, repeated-release or insider attacker is evaluated. Record and household identifiers assign
+roles and align releases; they are never attacker inputs. Recovery is therefore a measured property of a fixed
+attacker slate under this conditional threat model, not a privacy guarantee.
+
 \begin{figure}[t]
 \centering
 \resizebox{\linewidth}{!}{\input{fig_recipient.tex}}
@@ -153,9 +182,13 @@
 information and compare fairness-unaware $k$-means encoders. We add a same-budget zero-fairness tree and measure
 attribute recovery from the complete release.
 
-\paragraph{Combining releases.} Combining independent releases is a classical route to
-breaches~\citep{ganta2008composition}; combining a biased and a debiased version of one model raises attribute
-inference~\citep{tian2025fairnessprivacy}; collusion-aware mechanisms exist for sequential finite-alphabet
+\paragraph{Adversary models and combined releases.} Our recipient is not a new adversary. It is the auxiliary-data
+attacker of attribute inference, which learns from labelled examples to read a sensitive attribute from what a model
+exposes~\citep{song2020overlearning,jia2018attriguard}, applied here to released task outputs. Combining independent
+releases is a classical route to breaches~\citep{ganta2008composition}; recipients of differently perturbed copies of
+the same data can pool them, and mechanisms exist under which the pool reconstructs the data no more accurately than
+the best single copy~\citep{li2012multilevel}; combining a biased and a debiased version of one model raises attribute
+inference~\citep{tian2025fairnessprivacy}; and collusion-aware mechanisms exist for sequential finite-alphabet
 releases~\citep{taylor2026collusion}. We measure the purpose-permission version, as a property of our attackers.
 
 \paragraph{Position.} In a bounded reading of the closest work, we did not find one protocol that applies, on the same
@@ -175,12 +208,16 @@
 the uncleaned roles (assessment 5{,}250 Adult and 4{,}778 HMDA records; the overlapping records retained and
 disclosed; its decisions were stable when they were removed).
 
-\paragraph{Attackers and recovery.} Every surface is attacked by the same slate: logistic regression (5 regularisation
-values), gradient-boosted trees (20 configurations) and an MLP (18). The nonlinear attacker is the better of the tree
-and MLP families on \texttt{attacker\_val} log loss, refitted at three attacker seeds. Recovery is the macro one-vs-rest
-AUC on assessment over supported classes, averaged over attacker seeds and then encoder seeds. For features plus output,
-candidates that ignore one part are also eligible on validation. For finite releases we additionally fit a
-cell-conditional attacker and report it descriptively. Banks over output formats select on validation only.
+\paragraph{Attackers and recovery.} Every surface is attacked by the same slate, with the access of
+Section~\ref{sec:threat}: logistic regression (5 regularisation values), gradient-boosted trees (20 configurations) and
+an MLP (18), each fitted on \texttt{attacker\_fit} releases labelled with the protected attribute. The nonlinear
+attacker is the better of the tree and MLP families on \texttt{attacker\_val} log loss, refitted at three attacker
+seeds. Recovery is the macro one-vs-rest AUC on assessment over supported classes, averaged over attacker seeds and then
+encoder seeds. For features plus output, candidates that ignore one part are also eligible on validation; for a
+recipient pair, the bank also contains each single recipient's attacker, chosen on validation like any other
+candidate. For finite releases we additionally fit a cell-conditional attacker, which estimates the attribute's
+distribution within each released cell, and report it descriptively. Banks over output formats select on validation
+only.
 
 \paragraph{Utility.} We report the accuracy, balanced accuracy, minority recall, log loss and gain over the constant
 (majority class on \texttt{attacker\_fit}) of the \emph{deployed} head whose output is released, and, separately, the
@@ -346,12 +383,16 @@
 \label{fig:pairs}
 \end{figure}
 
-Two Adult recipients, each permitted only its own task output (income; employment group), pool their outputs to
-recover race, which is disallowed for both (Figure~\ref{fig:pairs}). The pair beats both single recipients by more
-than 0.02 AUC under every format: full logits 0.917 vs 0.845, centred 0.883 vs 0.792, and hard decisions 0.611 vs
-0.576 (pair minus best single $+0.034$, lower bound 0.023, the weakest of six passes). We report absolute pair recovery
-and its increment over the strongest single release separately. On seed~2 under full logits, validation selected the
-income-only attacker for the pair, so that seed's pair-minus-income difference is exactly zero.
+Two Adult recipients, each permitted only its own task output (income; employment group), link their outputs by
+record and pool them to recover race, which is disallowed for both (Figure~\ref{fig:pairs}). The pair beats both
+single recipients by more than 0.02 AUC under every format: full logits 0.917 vs 0.845, centred 0.883 vs 0.792, and
+hard decisions 0.611 vs 0.576 (pair minus best single $+0.034$, lower bound 0.023, the weakest of six passes). We report
+absolute pair recovery and its increment over the strongest single release separately. On seed~2 under full logits,
+validation selected the income-only attacker for the pair, so that seed's pair-minus-income difference is exactly
+zero. These are measurements with a fixed attacker slate. They show that linked outputs need an evaluation of their
+own; they do not show that race is carried only jointly by the two outputs. AUC is not an information measure, the
+pair gain does not separate evidence that each output carries on its own from evidence present only in their
+combination, and we prove no composition or synergy result.
 
 \section{Discussion}\label{sec:discussion}
 
@@ -381,8 +422,11 @@
 \paragraph{Limitations.} Two datasets, a small number of primary cells and three encoder seeds; the encoder seeds differ
 strongly (one HMDA encoder is heavily collapsed). All assessment rows were used before; nothing here is fresh
 confirmation. Intervals are conditional on fitted heads and attackers and omit retraining variance. FARE is applied to
-stored representations rather than raw inputs. Rare HMDA race groups are not estimable. We test no repeated-query or
-adaptive-query attacker. Several observations in Section~\ref{sec:useful} are post hoc and labelled as such.
+stored representations rather than raw inputs. Rare HMDA race groups are not estimable. The threat model of
+Section~\ref{sec:threat} leaves untested repeated or adaptive queries, white-box access to parameters or gradients,
+insiders, recipients that also hold a record's inputs or later learn its true label, imperfect linkage, coalitions of
+more than two recipients, and auxiliary data that are scarcer than ours or drawn from another population. Several
+observations in Section~\ref{sec:useful} are post hoc and labelled as such.
 
 \paragraph{Reproducibility.} For every study the protocol, code pin, admitted-input hashes, unit list, endpoint
 families and budget were committed before fitting; each study's result tables, inference code, verification replay and
--- a/papers/combined_empirical_v1/references.bib
+++ b/papers/combined_empirical_v1/references.bib
@@ -273,4 +273,24 @@
   pages     = {1321--1330},
   year      = {2017},
   url       = {https://proceedings.mlr.press/v70/guo17a.html}
+}
+
+@article{li2012multilevel,
+  title   = {Enabling Multilevel Trust in Privacy Preserving Data Mining},
+  author  = {Li, Yaping and Chen, Minghua and Li, Qiwei and Zhang, Wei},
+  journal = {IEEE Transactions on Knowledge and Data Engineering},
+  volume  = {24},
+  number  = {9},
+  pages   = {1598--1612},
+  year    = {2012},
+  doi     = {10.1109/TKDE.2011.124},
+  url     = {https://arxiv.org/abs/1104.0459}
 }
+
+@inproceedings{jia2018attriguard,
+  title     = {{AttriGuard}: A Practical Defense Against Attribute Inference Attacks via Adversarial Machine Learning},
+  author    = {Jia, Jinyuan and Gong, Neil Zhenqiang},
+  booktitle = {Proceedings of the 27th USENIX Security Symposium},
+  year      = {2018},
+  url       = {https://arxiv.org/abs/1805.04810}
+}
```

## Notes for the manuscript owner

- `\label{sec:threat}` sits inside a `\paragraph` of §2, so `Section~\ref{sec:threat}` prints "Section 2". If a
  numbered target is preferred, promote the paragraph to `\subsection{Adversary and access}`.
- If the ACS service results enter the manuscript, add one sentence giving the ACS auxiliary sample (8,001 people to
  fit and 3,000 to select, from the same Florida 2019 cohort) and the ACS release (probability vector and argmax
  decision). The variant in `MOTIVATION_DRAFTS.md` §a can be used.
- Do not use "defense-aware" for any study without the definition given in block C. The AAAI submission uses the same
  term for a Gaussian likelihood-ratio test, which collapses to a QDA probe for deterministic defenses (see
  `ACCESS_TABLE.md`).
