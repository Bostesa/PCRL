# Selection decision: no candidate survived

**Outcome: `NO_CANDIDATE_SURVIVED`** (selection rule 1 of `roles/coordinator/SELECTION_CRITERIA.md`, committed at `2d96c471e` before either proposal was readable).

**What failed: the candidate-selection prerequisite.** Both proposals fail hard criterion S3 (distinct ingredient) as committed, and that alone triggers rule 1. After reading the proposals and the closest-work review, the coordinator also rules that both fall outside the intent of S2 (substantive structure; section 3). The outcome does not depend on that ruling. This is not:
- an external service refusal;
- a missing artifact;
- an implementation failure;
- a failed utility prerequisite (no task-only twin was fitted);
- a negative scientific result. The family was never fitted, so nothing is known about its utility or recovery on these people.

**Consequences.** No pilot was locked or run. There were:
- no defense fits and no reader fits;
- no MODEL_SELECT label array loaded (the restore copied and byte-hashed the label and SEX export files without loading them);
- no DEV_REPORT truth read;
- no sealed-assessment access.

The campaign ends with the service and threat model, this decision record, the reviewed proposals and the implementation already produced.

## 1. The two proposals

Each designer received the same brief and an instruction not to open the other's folder, and each reported that it did not; those reports are the evidence of independence. Designer 2's proposal carries no non-access statement of its own, and proposal 1 was on disk 18 minutes before proposal 2. The service-model text designer 2 cites traces to the adversary review, written before proposal 1. Both proposals were hashed on delivery.

| | Designer 1: Shared-Cell Merging (SCM) | Designer 2: SCAR (shared-cell agglomerative release) |
|---|---|---|
| File and hash | `roles/designer_1/PROPOSAL.md`, `6bc0617b…` | `roles/designer_2/PROPOSAL.md`, `74b42067…` |
| Mapping | task-only CatBoost teachers (5-fold cross-fitted) on MODEL_FIT; KL k-means into 512 cells of the 7-dimensional teacher-posterior space; greedy all-pairs merges by SEX-variance gain per unit of exact plug-in cost, under per-metric budgets and a within-SEX guard | task-only CatBoost teachers on MODEL_FIT (in-sample posteriors); Euclidean k-means into 8,192 cells; Lagrangian greedy merges (task cost minus λ times the plug-in I(S; cell)) on a nearest-neighbour graph to a fixed K, under per-clause caps |
| Release | occupation recipient: the shared cell's pooled posterior; income recipient: one of 32 SEX-blind income groups of cells | the same, with 64 income bins |
| Novelty claimed by the designer | none | none |
| Designer's own forecast | most likely outcome: stops at a stage-0 gate (after the teacher and mechanism fits, before any reader fit) | the falsifier more likely than not triggers |

The two designers converged on the same family. The forecasts played no part in this decision.

## 2. The criteria applied

| Criterion | SCM | SCAR | Evidence |
|---|---|---|---|
| S1 contract | pass | pass | Adversary check (`roles/adversary_reviewer/PROPOSAL_THREAT_CHECK.md`): the deployed maps read only the six fields and refuse SEX, and the transmitted values are copies of stored vectors. There are routing and last-bit storage remarks, both fixable |
| S2 substantive structure | **outside S2's intent on a post-reading ruling** (section 3, ruling 1); read literally, arguably met | **outside S2's intent on a post-reading ruling** (section 3, ruling 1); read literally, arguably met | The release is h(q(x)), where q is a task-only teacher refit with the T_TREE registry and h is a SEX-aware many-to-one map of q's output: quantiser, merge table and pooled heads. No input distinction is merged unless the teacher has already mapped it to a nearby posterior. The closest-work reviewer places it inside the output-channel family (frozen-bank channels, measured gains 0.002–0.006; latent codes, log-loss cost 0.020–0.032). In the reviewer's words, refitting the teacher escapes the closed lanes' letter but not their substance (`roles/prior_work_reviewer/PROPOSAL_COMPARISON.md` §3) |
| S3 distinct ingredient | **fail** | **fail** | Verdict `ENGINEERING_COMBINATION_NO_NEW_INGREDIENT` for both. SCM is a privacy-funnel-type constrained greedy merge with a quadratic leakage functional and per-task budgets. SCAR is an IBSI-type Lagrangian merge whose leakage cost is the privacy funnel's. Per-task hard caps are an engineering variant. The shared two-task partition is a combination: the pair reveals exactly what the occupation recipient sees whenever occupation vectors are distinct across cells, a data-processing consequence that matches Li et al.'s best-single-copy criterion within FARE's scope. Output-space ties are not realised, and generalisation through teacher posteriors is an engineering variant. Against the privacy funnel or IBSI the released object does not change; only the partition selected does (`PROPOSAL_COMPARISON.md` §1.2, §2.2, §6) |
| S4 rationale | conditional: SCM §5 says NEURAL_RAW "merges nothing beyond the encoder's OTHER pooling", an unqualified injectivity-type statement that `CORRECTIONS.md` §1(e) and §2 call unknown and S4 excludes; also a scope correction (the dominance argument holds for the pair view, not for each recipient) | pass: SCAR §5 states the same point with the qualification that injectivity is unknown; the reviewer's caveats on in-sample posteriors apply | Both proposals; both reviews |
| S5 feasibility | plausible, untested (gate G0) | plausible, untested (gate G0); in-sample posteriors with about 2.4 people per fine cell | Both proposals |

**Rule 1:** a candidate must pass S1 to S5, and neither does. Rule 2 (choosing between two eligible proposals) is not reached. For the record, if the family were piloted, the closest-work reviewer judged SCM "narrowly" the more defensible representative and the adversary reviewer judged SCM "safer to pilot".

## 3. Rulings (1 and 2 requested by the reviewers; 3 added by the coordinator)

1. **Is a SEX-aware hard quantiser of a refit task-only teacher's posteriors a structural change under the brief? Ruled no, after reading the proposals.**
   - Read literally, S2 is arguably met: both proposals merge input distinctions, and both refit their teachers.
   - The brief's hypothesis reads: "Learn which input distinctions to retain or merge while fitting useful task probabilities, instead of only modifying continuous scores on frozen representations, frozen tree partitions or historical code banks." Neither proposal modifies continuous scores. The ruling rests on the first half: the merges are to be learned *while fitting* the task probabilities.
   - Both proposals fit a task-only teacher first and merge its outputs afterwards. Which input distinctions are merged is then decided only through the teacher's level sets.
   - That is the output-channel family the closed lanes already measured, whatever the teacher's seed.
2. **Is a deterministic release that takes few distinct probability values a contract change? No.** Deterministic many-to-one probability releases with the argmax are inside the primary contract (`results/pcrl_mechanism_design_v1/CORRECTIONS.md` row 6). Randomised outputs, shared randomness or non-simplex symbols would be declared contract changes. So the comparators would not have needed re-evaluation under a changed contract.
3. **The designers' forecasts.** The decision does not rest on either designer's pessimistic forecast. It rests on S3 as committed (the closest-work reviewer's finding), which alone triggers rule 1. Ruling 1 on S2 is an additional interpretation made after reading the proposals, and it is labelled as such.

## 4. What this decision does and does not establish

**Established:** neither submitted proposal meets this campaign's committed eligibility criteria.

**Not established:**
- that shared-cell merging fails on ACS;
- that structural mechanisms cannot be competitive;
- that the neural adversarial comparator is unbeatable;
- anything about utility or recovery of any new release.

Nothing was fitted. The corrected status from earlier campaigns stands: no competitive new mechanism has been established.

## 5. If this family is ever piloted as an engineering application (not authorised here)

The reviews leave a ready specification. Recorded for a future owner; nothing here authorises it.

- **Representative:** SCM, with out-of-fold posteriors, the within-SEX guard budgeted and all-pairs exact search.
- **Additions from SCAR:**
  - the NR_PROJECTED diagnostic: NEURAL_RAW's pair averaged within the cells, which names the information the neural control retains;
  - the lookup-reader gate.
- **Attribution arm (required):** PUB_IBSI, the published IBSI objective, merged greedily on the same fine cells with no per-clause caps and stopped at the same cell count.
- **Fixes for both designs:**
  - pre-fit proxy readers fitted on one household half of MODEL_SELECT and scored on the other, both ways, because MODEL_FIT SEX drove the merges;
  - at least five fixed-seed SEX permutations, with the candidate required to beat the best of them;
  - enough ATTACK_FIT people per cell (the adversary reviewer suggests a median of at least about 15), or a statement registered in advance that recoveries are reader-limited.
- **Framing:** the coalition result is presented as a consequence of sharing one partition, citing Li et al. and FARE, not as a collusion mechanism. Pair and occupation-recipient endpoints coincide for Bayes readers and are not independent evidence.
- **Expectation:** the closed frozen-bank-channel lane's measured gains (0.002–0.006) are the direct prior for the recipe effect. The designers' priors agree.

## 6. What a future candidate would need

**The core requirement.** A mechanism in which the many-to-one structure is learned on the inputs jointly with the task probabilities, not applied afterwards to a task-only model's outputs. It must also have an ingredient that changes the released object relative to the privacy funnel, IBSI, FARE and the closed lanes.

**What it would still have to face:**
- the 24 caps;
- the strongest eligible neural comparator (NEURAL_RAW ρ1 E2, pair 0.640 / 0.638);
- the frozen lookup reader;
- the matched task-only and local twins.

No such proposal was produced in this campaign. None is invented here.
