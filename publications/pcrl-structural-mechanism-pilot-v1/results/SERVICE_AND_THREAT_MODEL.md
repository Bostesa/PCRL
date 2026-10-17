# Service and threat model: the ACS two-recipient benchmark

**Status.** Version 1. The coordinator wrote draft v0 before reading any mechanism proposal. This version applies the recipient-use and adversary reviewer's suggestions (`roles/adversary_reviewer/SERVICE_REVIEW.md`: 6 must-fix, 5 should-fix, 4 notes, all applied). Nothing here is a finding.

**What this is.** A controlled multi-recipient benchmark built on Florida ACS 2019 PUMS person records. About ten minutes of searching found no documented deployment of an income and occupation prediction service computed from these six fields for two recipients. The recipient actions below are therefore **illustrative**. The benchmark definitions (cohort, fields, tasks, roles, references, caps, readers, endpoints) are unchanged from the source study (public `f4ea199`) and its frozen-hybrid follow-up (public `3a1f509`).

## 1. The service

| | Income recipient (R1) | Occupation recipient (R2) |
|---|---|---|
| Task | Y1 = 1[PINCP x ADJINC > $50,000] | Y2 = one of five broad civilian occupation groups |
| Computed from | the person's own six fields: AGEP, WKHP, COW, SCHL, MAR, POBP | the same six fields |
| Transmitted to this recipient | its own probability vector over 2 classes, f1(x) (float64, fixed class order), and its decision d1 = argmax f1(x) | its own probability vector over 5 classes, f2(x), and its decision d2 = argmax f2(x) |
| Not transmitted | the other recipient's output, SEX, labels, raw fields, teacher outputs, representations, hidden features, leaf or cell identifiers as separate fields, scores before the softmax, model parameters, seeds, flags, weights or any state | same |
| Permitted action (illustrative) | decide whom to contact with an offer whose payoff depends on the recipient's own costs, and rank people when outreach capacity is fixed | assign one of five occupation-specific programmes or materials when they have different costs or capacities, and route low-confidence cases to manual review |
| Not permitted to infer | the person's SEX | the person's SEX |

- **Deterministic, one release per person.** The same six field values always give the same transmitted pair (f_i(x), d_i(x)). No randomness is transmitted or shared between recipients.
- **Identifiers and cells.** No identifier field is sent. A recipient that serves a person knows who that person is through its own relationship with them; that is how linkage can happen (section 3). For a release that takes finitely many values, the vector itself identifies the cell whenever cells have distinct vectors, so every reader has cell membership. Only cells with bit-identical vectors are merged as far as a recipient can tell.
- **Routing.** Delivering each task's output only to its own recipient is a contract duty. A deployment package that computes both outputs must route them separately.
- **Deployment.** The release takes only the six permitted raw fields. It must refuse SEX or any other prohibited input.

## 2. Why the actions need probabilities rather than only the argmax

- **Unequal action costs.** With costs C(a, y), the expected-cost-minimising action is argmin_a sum_y C(a, y) P(y | x) (Elkan, "The Foundations of Cost-Sensitive Learning", IJCAI 2001, pp. 973-978, Eq. 1). For two classes it predicts 1 iff P(1 | x) >= p\* = (c10 - c00)/(c10 - c00 + c01 - c11) (Eq. 2). That threshold "in general is different from 0.5", the threshold at which accuracy-maximising classifiers "implicitly make decisions". p\* depends on the recipient's own costs, which the service does not know. A probability lets each recipient apply its own threshold, which a decision taken at 0.5 does not. If the service knew p\*, it could release a decision taken at p\* instead; we assume it does not. For binary tasks, Gao, Parameswaran and Peng (AISTATS 2017, PMLR 54:1367-1374) show that the calibration property of estimates that need not be the true conditional probabilities "provides adequate support for us to use Bayes Decision Theory" in cost-sensitive problems.
- **Ranking under capacity.** Ling and Li ("Data Mining for Direct Marketing: Problems and Solutions", KDD 1998, section 5) require classifiers that output "a probability estimation or certainty factor", which lets customers be ranked and "makes it possible to choose any number of likely buyers"; they evaluate with lift rather than accuracy. Ranking needs only the order of the scores.
- **Deferral.** Guo, Pleiss, Sun and Weinberger ("On Calibration of Modern Neural Networks", ICML 2017, PMLR 70:1321-1330, section 1) write that "in automated health care, control should be passed on to human doctors when the confidence of a disease diagnosis network is low", and that a network "should provide a calibrated confidence measure in addition to its prediction". Their confidence is the probability of the predicted class, the quantity the CONF_HGB reader reads.
- **What these sources do not show.** None concerns releases to third parties, privacy or this benchmark's caps; Gao et al. cover binary calibration only. They justify investigating probability releases. They do not show that the benchmark's numerical caps (accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier excess <= 0.005, against U0 and Ucal, both tasks and both seeds) are what any particular downstream action needs. No cap is a ranking metric, so a release can meet every cap and still reorder people at a recipient's capacity. With the threshold set to the cost proportion, the Brier score equals the expected loss under a uniform distribution of cost proportions (Hernández-Orallo, Flach and Ferri, JMLR 13:2813-2869, 2012, Theorem 14), so a summed-Brier cap bounds an average over cost ratios, not the loss at any one recipient's threshold. The caps are **benchmark requirements pending advisor feedback**; a recipient use does not retrospectively validate them.

## 3. The declared adversary (in scope)

| Element | Specification |
|---|---|
| Individual view | one recipient's (f_i(x), d_i(x)) |
| Coalition view | the perfectly aligned pair (f1, d1, f2, d2) for the same person. Linkage is an **explicit grant**, the attacker-favourable limit, not a measured capability |
| Ignore-one | the coalition may ignore either recipient: its reader bank contains each single-recipient reader as an alias |
| Auxiliary data | SEX labels and releases, from the same fitted service, for people in the same cohort. They come from household-level random splits, so no auxiliary household is a target household: ATTACK_FIT has 8,001 people in 5,319 households and fits the readers; ATTACK_VALIDATION has 3,000 people in 1,974 households and orients and selects them. For the coalition, each auxiliary person contributes both releases, linked. Readers receive no auxiliary raw inputs or task labels (fits use release tables and SEX only), so de-censoring attacks are untested. Availability, size, the linkage of auxiliary pairs and representativeness are all **explicit grants** |
| How such labels could arise (illustrative) | people the recipient serves who have told it their sex, and whose releases it receives in normal service; for the coalition, people both recipients serve. Such a self-selected subset need not be representative; representativeness is granted. A recipient that submits a labelled survey sample to the service would be making chosen-input queries, which are outside this scope |
| Releases | one fixed release per person; no repeated or chosen queries |
| Not available | the target's raw inputs or labels, fitted model parameters, gradients, seeds, hidden features, and cell or leaf identifiers as separate fields |
| Algorithm knowledge | in scope: a recipient may know the published algorithm, objective and training recipe, but not the fitted parameters, seeds or training data. Readers that use this knowledge while seeing only the declared releases and auxiliary sample belong to this threat model. Our slate uses none beyond refitting to the defended releases, so the recovery figures are lower estimates against such readers too. Attacks that use fitted parameters, gradients or seeds need access the contract withholds; they are a separate, untested threat model |
| Readers | the source slate per view: OUT_LR, OUT_MLP, OUT_HGB (probabilities plus the decision as an index column), LOG_LR, LOG_MLP (log-probabilities plus one-hot decision), DEC_LR (one-hot decision), CONF_HGB (maximum probability); for the coalition, 7 pair readers plus 14 ignore-one aliases. Added in this campaign for every arm, because the HGB readers bin each input into at most 255 bins and can merge vectors that a finite-valued release keeps distinct: one **lookup reader** per view, the smoothed ATTACK_FIT SEX rate of each distinct released vector (or linked pair). Readers are fitted on ATTACK_FIT, oriented and selected on ATTACK_VALIDATION (ties to the earlier candidate), and scored on DEV_REPORT. Shuffled-SEX and planted-bit controls check the pipeline. **Lookup reader, frozen before any fit:** key = the exact float64 bytes of the released probability vector (marginal view) or of the two vectors concatenated (pair view); decisions are functions of the vectors and add nothing to the key. Score of a key seen in ATTACK_FIT = (n_female + 2 pi) / (n + 2), where pi is the ATTACK_FIT female share (2 pseudo-counts centred on the base rate); an unseen key scores pi. Orientation on ATTACK_VALIDATION like every other reader. Candidate order (ties go to the earlier candidate): the 7 inherited readers, then LOOKUP; in the pair bank, the 8 pair readers, then the 8 income-only aliases, then the 8 occupation-only aliases. Applied identically to the candidate and every comparator. The winner over the original 7-reader slate is also reported, as a secondary descriptive figure comparable with the source study |
| What a recovery number means | the DEV_REPORT AUC of the selected reader: an estimate, with sampling error, of that reader's AUC. It is not a lower confidence bound, a population floor, a certificate or a formal privacy guarantee, and AUC is not mutual information. DEV_REPORT has been opened before, so a new mechanism's DEV_REPORT figures are exploratory, not confirmatory |

**Outside this scope (distinct threat models, not breaches):** target-input side information; attacks that use fitted parameters, gradients or seeds; permitted repeated or chosen queries; insiders; imperfect linkage; coalitions of more than two; auxiliary data from another population or period; auxiliary samples that include raw inputs or task labels; attacks that use no labelled sample (for example ones built from a known population law); releases of the same person from several model versions (the two seeds are two fitted services); and attackers with more or less labelled data than declared. They are untested here, not dismissed.

## 4. Motivation

Both tasks are associated with SEX in this population. Releases are computed from six fields that carry information about SEX, although SEX is never an input, so probabilities that are useful for the two tasks can carry SEX information. Fitted readers confirm this. For the teacher tree release T_TREE, a recipient reads SEX at AUC 0.569 from the income vector and 0.650 from the occupation vector, and a perfectly linked pair reads 0.664, 0.013 above the better single recipient. Protecting each recipient separately did not protect the pair here: the local leaf-score correction (LOCAL_TREE) lowered the two recipients to 0.558 and 0.582 but left the linked pair at 0.657. This can happen in general, because protection of one declared attribute in each release does not in general compose: two releases can reveal together what neither reveals alone. Existing methods leave the problem open: the eligible neural adversarial control lowers the pair only to 0.640, and the only release near 0.60 (JOINT_TREE, 0.603) fails income utility. The problem worth addressing is purpose limitation in a release that must stay useful: each recipient needs probabilities for its permitted action, and the same probabilities support an inference that neither may make. The case for evaluating this access is that it is what the recipients legitimately hold, plus linkage and a labelled sample of the declared size that we grant explicitly; it is not that it yields a stronger attack. SEX is the declared attribute because ACS records it. A recipient that knows whom it serves may learn sex by other means, and the benchmark measures what the release reveals on its own, not what it adds to such side information.

(DEV_REPORT, seed 0, from `results/pcrl_mechanism_design_v1/EVIDENCE_PACK.md` table A.)

## 5. Documented versus assumed

| Item | Status |
|---|---|
| Fields, tasks, cohort, household roles, references, caps, reader slate, endpoint rules | documented in the source study's code and locks |
| That income and occupation probabilities computed from these fields are delivered to two recipients | **assumption** (benchmark construction); no deployment found |
| The recipients' actions and cost structures | **illustrative** |
| Linkage of the two releases for the same person, and of the auxiliary pairs | **explicit grant** |
| A representative SEX-labelled auxiliary sample of the declared size | **explicit grant** |
| One release per person, no queries | contract choice |
| That SEX is the protected attribute and no other attribute is claimed | declared scope |

## 6. Closest methods versus this access

The adversary is the standard auxiliary-data attribute-inference adversary, and the coalition is the standard linked-release setting; neither is new. Any novelty must come from the mechanism and its evidence.

**Work that already studies this access.**
- **Single recipient with auxiliary labelled data:** AttriGuard's attacker trains on users who disclose the attribute (Jia and Gong, USENIX Security 2018); the overlearning adversary does the same (Song and Shmatikov, ICLR 2020). The released object differs from ours: user data and representations, not task probabilities.
- **Model-inversion attackers** (Fredrikson et al. 2014; Mehnaz et al., USENIX Security 2022) hold the target's non-sensitive attributes and query the model. That is the target-input-plus-query threat model, outside this scope.
- **Linked releases:** Ganta, Kasiviswanathan and Smith (KDD 2008); Wang and Fung (KDD 2006).

| Method | What its guarantee or objective covers | Readers covered | Linked pair of task outputs? |
|---|---|---|---|
| FARE (ICML 2023) | cells of a restricted encoder. With probability >= 1 - epsilon over a finite sample, every downstream classifier's demographic-parity distance is at most T\*; the bound comes from bounding the balanced accuracy of the optimal SEX predictor from the cells | every reader of the cells ("model-free") | yes, if both outputs are functions of the same cells |
| FAGTB (ICDM 2019) | one boosted output; an adversarial objective with no certificate | its adversary network | no |
| Edwards and Storkey; LAFTR | a representation; an adversarial objective | the critic class | only if trained on the pair |
| LEACE; INLP | features; LEACE: no linear classifier beats a constant on the fitting distribution | readers linear in the guarded coordinates | for those readers only: LEACE's exact zero covariance in each release carries over to the concatenation (INLP gives no exact guarantee); nothing for nonlinear readers |
| Privacy funnel (2014) | one release Y under a known P(S, X): min I(S; Y) s.t. I(X; Y) >= R | any reader (an information measure) | only if the pair is designed as one release |
| Li et al. (TKDE 2012) | Gaussian-perturbed copies of the data; joint linear least-squares reconstruction no better than with the best single copy | linear estimators given first and second moments | yes, for randomised copies of the data itself |
| Taylor et al. (arXiv 2026) | sequential randomised channels under a known p(x); I(R_k; X) <= epsilon_k and, under collusion, I(R_k, R_k-1; X) <= delta_k | any reader (mutual information about the whole record) | yes, for randomised releases, at the level of the whole record |
| AttriGuard (2018) | a user's released data vector, perturbed against the defender's classifiers | the defender's surrogate classifiers | no |

**Composition, stated precisely.** A per-recipient guarantee for one declared attribute does not in general certify the pair: two releases can be individually uninformative about SEX and jointly informative (secret sharing is the canonical case). Some guarantees do compose. If each release has exactly zero covariance with SEX in every coordinate (the linear guardedness that LEACE enforces on its fitting distribution), then their concatenation also has exactly zero covariance with SEX, so the linear guarantee holds for the pair without separately erasing the concatenation. Four limits apply (from the independent verifier's exact examples):
- **Readers covered.** The guarantee covers only readers that are linear in the guarded coordinates. Readers of log-probabilities, of the one-hot decision, or of probabilities plus an argmax column are not covered.
- **Which methods give it.** Exact zero covariance on the fitting distribution is LEACE's property. INLP does not guarantee it.
- **What the guarantee is.** It is the convex-loss notion: no linear predictor improves on a constant under such losses. Zero covariance still allows a linear score with AUC above 0.5 (0.75 in an exact example), and this benchmark reports AUC.
- **Approximate constraints.** They can fail badly rather than merely inexactly. In an exact example each single release allows a best linear AUC of 0.505 while a linear reader of the pair reaches 1.0.

Nothing here establishes protection against nonlinear readers. The FARE, Li et al., Taylor et al. and Mehnaz et al. rows were re-opened by the reviewer; the others rely on `results/pcrl_mechanism_design_v1/CLOSEST_PRIOR_WORK.md`, which an earlier reviewer checked against primary sources.
