# Review of SERVICE_AND_THREAT_MODEL.md (draft v0)

Recipient-use and adversary reviewer, 2026-10-16. I did not edit the draft. Line numbers refer to the draft. I opened no `.private/` or `.npz` file.

**What I checked.**
- **Code and records:** `acs_replication/readers.py`, `acs_hybrids/readers.py` and `task_conditioned_representation/attacks.py:117-120`. Also, from the source study's results, `SCIENCE_LOCK.json` (`threat` :9, `caps` :17-18, `release_contract` :896) and `POPULATION_AND_SCHEMA.md:96-97`.
- **Campaign records:** `results/pcrl_mechanism_design_v1/{ADVERSARY_AND_MOTIVATION, CORRECTIONS §6, EVIDENCE_PACK, CLOSEST_PRIOR_WORK}.md`.
- **Primary sources:** I opened every one listed in the last section.

**Overall.** The draft is a sound start. It already calls the benchmark a controlled multi-recipient benchmark (line 5), and its access core matches the code:
- individual views and a perfectly aligned pair;
- 7 pair readers plus 14 ignore-one aliases (`acs_hybrids/readers.py:109-115`);
- ATTACK_FIT and ATTACK_VALIDATION roles;
- no target inputs and one release per person.

Six items must change before designers rely on it.

## MUST-FIX

**M1. The cost argument misstates its rule, and its sources are unchecked (§2, lines 24-26).** Line 24 says the Bayes action "equals the argmax only for equal (0-1) costs". That is false. Elkan shows that a 2x2 cost matrix has "essentially only one degree of freedom from a decision-making perspective": the threshold p\*. The argmax is the case p\* = 0.5, and any matrix with c10 − c00 = c01 − c11 gives it. Optimal decisions are also unchanged when every entry is multiplied by a positive constant or has the same constant added. Replace lines 24-26 with:

> - **Unequal action costs.** With costs C(a, y), the expected-cost-minimising action is argmin_a Σ_y C(a, y) P(y | x) (Elkan, "The Foundations of Cost-Sensitive Learning", IJCAI 2001, pp. 973–978, Eq. 1). For two classes, it predicts 1 iff P(1 | x) ≥ p\* = (c10 − c00)/(c10 − c00 + c01 − c11) (Eq. 2). That threshold is one "that in general is different from 0.5", the threshold at which accuracy-maximising classifiers "implicitly make decisions". p\* depends on the recipient's own costs, which the service does not know. A probability therefore lets each recipient apply its own threshold, which a decision taken at 0.5 does not. If the service knew p\*, it could release a decision taken at p\* instead; we assume it does not. For binary tasks, Gao, Parameswaran and Peng (AISTATS 2017, PMLR 54:1367–1374) show that for cost-sensitive problems, the calibration property of estimates that need not be the true conditional probabilities "provides adequate support for us to use Bayes Decision Theory".
> - **Ranking under capacity.** Ling and Li ("Data Mining for Direct Marketing: Problems and Solutions", KDD 1998, §5) require classifiers that output "a probability estimation or certainty factor". This lets customers be ranked, which "makes it possible to choose any number of likely buyers"; they evaluate with lift rather than accuracy. Ranking needs only the order of the scores.
> - **Deferral.** Guo, Pleiss, Sun and Weinberger ("On Calibration of Modern Neural Networks", ICML 2017, PMLR 70:1321–1330, §1) write that "in automated health care, control should be passed on to human doctors when the confidence of a disease diagnosis network is low". A network "should provide a calibrated confidence measure in addition to its prediction". Their confidence is the probability of the predicted class, the same quantity the CONF_HGB reader reads.

What these sources do not support:
- None of them concerns releases to third parties, privacy, or the benchmark's caps.
- Gao et al. covers binary calibration only. It is not a source for ranking or deferral.

**M2. Knowledge of the algorithm is put out of scope against the source (lines 40 and 44).** Lines 40 and 44 call "objective-aware" attacks a separate threat model. `CORRECTIONS.md:189` (the scope, stated positively) and `ADVERSARY_AND_MOTIVATION.md:55` say the opposite: a reader that uses public knowledge of the algorithm and objective, while seeing only the outputs, "belongs to this threat model". The source contradicts itself here, because `CORRECTIONS.md:180` and `ADVERSARY_AND_MOTIVATION.md:181` list objective-aware attacks as distinct. The draft should follow :189 and say so. Replace the line 40 cell with:

> In scope: a recipient may know the published algorithm, objective and training recipe, but not the fitted parameters, seeds or training data. Readers that use this knowledge while seeing only the declared releases and auxiliary sample belong to this threat model. Our slate has none beyond refitting to the defended releases, so the recovery figures are lower estimates against such readers too. Attacks that use the fitted parameters, gradients or seeds need access that the contract withholds. They form a separate threat model and are untested.

In line 44, replace "objective- or parameter-aware attacks" with "attacks that use the fitted parameters, gradients or seeds".

**M3. The example route to auxiliary labels contradicts the declared access (lines 36-38).** "A public survey sample scored through the service" needs the recipient to submit chosen inputs, and line 38 and line 44 exclude chosen queries. The draft also never says that the coalition's grant is labelled linked pairs (`acs_hybrids/readers.py:99-103`). Replace the rows at lines 36-37 with:

> | Auxiliary data | SEX labels and releases, from the same fitted service, for people in the same cohort. They come from household-level random splits, so no auxiliary household is a target household: ATTACK_FIT has 8,001 people in 5,319 households and fits the readers; ATTACK_VALIDATION has 3,000 people in 1,974 households and orients and selects them. For the coalition, each auxiliary person contributes both releases, linked. Readers receive no auxiliary raw inputs or task labels (`ADVERSARY_AND_MOTIVATION.md:89`; fits use release tables and SEX only, `acs_replication/readers.py:79`), so de-censoring attacks are untested. Availability, size, the linkage of auxiliary pairs and representativeness are all **explicit grants** |
> | How such labels could arise (illustrative) | People the recipient serves who have told it their sex, and whose releases it receives in normal service. For the coalition, people both recipients serve. Such a self-selected subset need not be representative; representativeness is granted. A recipient that submits a labelled survey sample to the service would be making chosen-input queries, which are outside this scope |

**M4. Cell identity is effectively released (lines 14 and 39).** For any finite-valued release, the float64 vector itself identifies the cell whenever cells have distinct vectors. This covers the trees and the many-to-one partition with per-cell heads that `DESIGNER_BRIEF.md:22` invites. "No cell identifiers" therefore withholds no information. Append to line 14:

> No identifier field is sent. For a release that takes finitely many values, the vector identifies the cell whenever cells have distinct vectors, so every reader has cell membership. Only cells with bit-identical vectors are merged as far as a recipient can tell.

On line 39, change "cell or leaf identifiers" to "cell or leaf identifiers as separate fields".

**M5. The motivation has an unaddressed hole and one imprecise number (line 48).**
- **The hole.** Line 19 grants that a recipient knows who it serves. Such a recipient may know sex by other means, such as a name, so the motivation must say that it measures what the release reveals on its own.
- **The imprecise number.** "0.65-0.66" for LOCAL_TREE's pair is 0.657 on seed 0 (seed 1 is 0.649).
- **An ambiguous label.** "Task-only tree release" could mean T_TREE, the teacher, or TASK_REFIT, the task-only twin. Both give 0.650 and 0.664.
- **Unsupported claims.** "The income gap persists given the six fields" has no cited number. "Protecting each recipient separately does not protect the pair" is stated as a general law but rests on one instance.

The other numbers match `EVIDENCE_PACK.md` table A. Replacement paragraph (seed-0 numbers, all checked against table A rows 18-24):

> Both tasks are associated with SEX in this population. Releases are computed from six fields that carry information about SEX, although SEX is never an input. Probabilities that are useful for the two tasks can therefore carry SEX information. Fitted readers confirm this. For the teacher tree release T_TREE, a recipient reads SEX at AUC 0.569 from the income vector and 0.650 from the occupation vector. A perfectly linked pair reads 0.664, which is 0.013 above the better single recipient. Protecting each recipient separately did not protect the pair here. The local leaf-score correction (LOCAL_TREE) lowered the two recipients to 0.558 and 0.582, but left the linked pair at 0.657. This can happen in general: leakage of one declared attribute does not compose, and two releases can reveal together what neither reveals alone. Existing methods leave the problem open. The eligible neural adversarial control lowers the pair only to 0.640, and the only release near 0.60 (JOINT_TREE, 0.603) fails income utility. The problem worth addressing is purpose limitation in a release that must stay useful. Each recipient needs probabilities for its permitted action, and the same probabilities support an inference that neither may make. The case for evaluating this access is that it is what the recipients legitimately hold, plus linkage and labelled data that we grant explicitly; it is not that it yields a stronger attack. SEX is the declared attribute because ACS records it. A recipient that knows whom it serves may learn sex by other means, and the benchmark measures what the release reveals on its own, not what it adds to such side information.

**M6. The closest-methods table (§6) misdescribes two works and confuses studies of this access with guarantees.**
- **Li et al.** protect the original data values, not a separate attribute. Their copies are perturbed with additive Gaussian noise that is correlated across copies. Colluders must not be able to reconstruct the data "more accurately than the best effort using any individual copy". Their adversary uses only linear (LLSE) estimation, knowing the means and covariances. They note that this is optimal over all estimators only when the data are jointly Gaussian (arXiv 1104.0459 §3.2).
- **Taylor et al.** bound I(R̂k; X), and I(R̂k, R̂k−1; X) when party k colludes with the previous recipient. Here X is the whole database, the release channels are randomised, and the joint law is known (arXiv 2601.21859 v2, §I–II).
- **Neither** fits "Bayes-optimal under a known law". The privacy funnel's mutual-information objective needs no reader class.

Replace the table with:

> **Work that already studies this access.**
> - **Single recipient with auxiliary labelled data:** AttriGuard's attacker trains on users who disclose the attribute (Jia and Gong, USENIX Security 2018); the overlearning adversary does the same (Song and Shmatikov, ICLR 2020). The released object differs from ours: user data and representations, not task probabilities.
> - **Model-inversion attackers:** Fredrikson et al. (2014) and Mehnaz et al. (USENIX Security 2022) hold the target's non-sensitive attributes and query the model. That is the target-input-plus-query threat model, outside this scope. `CLOSEST_PRIOR_WORK.md:33-35` groups them with the auxiliary-data adversary; the prior-work reviewer should check this.
> - **Linked releases:** Ganta, Kasiviswanathan and Smith (KDD 2008); Wang and Fung (KDD 2006).
>
> | Method | What its guarantee or objective covers | Readers covered | Linked pair of task outputs? |
> |---|---|---|---|
> | FARE (ICML 2023) | Cells of a restricted encoder. With probability ≥ 1 − ε over a finite sample, every downstream classifier's demographic-parity distance is at most T\*. The bound comes from bounding the balanced accuracy of the optimal SEX predictor from the cells | every reader of the cells ("model-free") | yes if both outputs are functions of the same cells |
> | FAGTB (ICDM 2019) | one boosted output; an adversarial objective with no certificate | its adversary network | no |
> | Edwards–Storkey; LAFTR | a representation; an adversarial objective (LAFTR's downstream bound is stated for an optimal adversary; not re-checked by me) | the critic class | only if trained on the pair |
> | LEACE; INLP | features; LEACE: no linear classifier beats a constant on the fitting distribution | linear | only if applied to the concatenation, and then for linear readers only |
> | Privacy funnel (2014) | one release Y under a known P(S, X): min I(S; Y) s.t. I(X; Y) ≥ R | any reader (an information measure) | only if the pair is designed as one release |
> | Li et al. (TKDE 2012) | Gaussian-perturbed copies of the data; joint LLSE reconstruction no better than with the best single copy | linear estimators given first and second moments | yes, for randomised copies of the data itself |
> | Taylor et al. (arXiv 2026) | sequential randomised channels under a known p(x); I(R̂k; X) ≤ εk and, under collusion, I(R̂k, R̂k−1; X) ≤ δk | any reader (MI about the whole record) | yes, for randomised releases, at the level of the whole record |
> | AttriGuard (2018) | a user's released data vector, perturbed against the defender's classifiers | the defender's surrogate classifiers | no |
>
> A per-recipient guarantee for one declared attribute never certifies the pair (`CLOSEST_PRIOR_WORK.md:47`).

I re-opened FARE, Li et al., Taylor et al. and Mehnaz et al. The other rows and the antecedent list rely on `CLOSEST_PRIOR_WORK.md` and `ADVERSARY_AND_MOTIVATION.md` §7, which an earlier reviewer checked against primary sources.

## SHOULD-FIX

- **S1. The caps and the actions (line 27).** Add: "No cap is a ranking metric, so a release can meet every cap and still reorder people at a recipient's capacity. Also, with the threshold set to the cost proportion, the Brier score equals the expected loss under a uniform distribution of cost proportions (Hernández-Orallo, Flach and Ferri, JMLR 13:2813–2869, 2012, Theorem 14). A summed-Brier cap therefore bounds an average over cost ratios, not the loss at any one recipient's threshold."
- **S2. The out-of-scope list at line 44 is incomplete** against `ADVERSARY_AND_MOTIVATION.md:178-189`. Add:
  - coalitions of more than two;
  - auxiliary data from another population or period;
  - auxiliary samples that include raw inputs or task labels;
  - attacks that use no labelled sample (for example, ones built from a known population law);
  - releases of the same person from several model versions. Two seeds are two fitted services.
- **S3. Routing (lines 13 and 20).** Delivering each task only to its own recipient is a contract duty. The base deployment call returns both tasks to one caller, and only the H_LJ package releases one task per recipient (`ADVERSARY_AND_MOTIVATION.md:87`).
- **S4. Readers (line 41).** The decision in the OUT_\* readers is a numeric index column, not one-hot (`acs_replication/readers.py:32`). Name the tie rule (the earlier candidate wins; `acs_hybrids/readers.py:11-12`) and the shuffled-SEX and planted-bit controls. Because of M4, add an in-scope lookup reader for finite-valued releases: the smoothed ATTACK_FIT SEX rate per distinct released vector or pair. The HGB readers use sklearn's default of at most 255 bins per input (`task_conditioned_representation/attacks.py:120`), so they can merge vectors that the release keeps distinct.
- **S5. Line 48, "modest".** Replace "modest SEX-labelled sample" with "a labelled sample of the declared size", since 11,001 people is a generous budget by design.

## NOTE

- **No matching deployment.** About ten minutes of searching found none: no service that predicts income and occupation from these six ACS fields for two recipients.
  - Search snippets showed income estimators for lenders built on credit and payroll data, and occupation coders that work from job-title text. I did not open those pages, so they are not citable.
  - Lines 5 and 57 already state this correctly. Keep the scenario illustrative.
- **Line 13, the simplex.** A 2-class vector lies in the 1-simplex, and a 5-class vector in the 4-simplex. Write "probability vector over 2 (5) classes".
- **Line 42, what a recovery number means.** Add three points:
  - it estimates the selected reader's AUC, with sampling error, and is not a lower confidence bound;
  - DEV_REPORT has been opened before, so a new mechanism's DEV_REPORT figures are not confirmatory;
  - AUC is not mutual information.
- **Line 14, the not-transmitted list.** Add "teacher outputs" and "representations", as in `release_contract` (`SCIENCE_LOCK.json:896`).

## Primary sources opened (2026-10-16)

- **Guo et al.:** https://proceedings.mlr.press/v70/guo17a.html, and the PDF, §1.
- **Gao et al.:** https://proceedings.mlr.press/v54/gao17a.html (abstract).
- **Elkan 2001:** https://cseweb.ucsd.edu/~elkan/rescale.pdf (Eq. 1, Eq. 2, §2). The record is at https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations/.
- **Ling and Li 1998:** https://www.csd.uwo.ca/~xling/papers/kdd98.pdf (§5).
- **Hernández-Orallo et al. 2012:** https://jmlr.org/papers/v13/hernandez-orallo12a.html (Theorem 14).
- **FARE:** https://proceedings.mlr.press/v202/jovanovic23a.html (abstract; the DP definition and the certificate requirement, pp. 2–3).
- **Li et al.:** https://arxiv.org/abs/1104.0459 (§3.2).
- **Taylor et al.:** https://arxiv.org/abs/2601.21859 v2 (§I–II).
- **Mehnaz et al.:** https://arxiv.org/abs/2201.09370 (abstract).
