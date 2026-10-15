# Adversary and motivation: task headroom and protected prediction outputs

Role: adversary/methodology. Adopted into SCIENCE_LOCK on 2026-10-15, before any fit. Existing manuscripts are unchanged; section 8 is a standalone replacement passage.

Status tags: **[SI]** source inspected by this role; **[SR-2]** second-hand, from a stored
source-inspection audit committed in this repository (the original source is not mounted locally,
see section 7); **[LIT]** the public paper's abstract page was checked; **[DESIGN]** the campaign
design as briefed, not yet a frozen protocol.

---

## 1. What is released (the object under attack) [DESIGN]

Two prediction services score the same person from permitted raw83 Adult inputs. SEX is excluded
from the inputs.

- **Income service.** A CatBoost binary (Logloss) predictor.
- **Occupation service.** A CatBoost 6-class (MultiClass) predictor.

Privacy training never changes tree structure or routing. It adds a learned per-leaf logit
correction:

`f_r(x) = f_teacher,r(x) + Σ_t δ[r, t, leaf_t(x)]`, with δ = 0 at initialization.

At δ = 0 the released service is exactly the teacher. That gives an exact witness that the family
contains the useful predictor.

**Release contract.** Each recipient r receives, for each person, only:

- its own full-precision probability vector (2 values for income, 6 for occupation); and
- its argmax decision.

No recipient receives the other service's output, leaf indices, raw inputs or metadata beyond the
published format. The **pair** view is the aligned concatenation of both recipients' releases for
the same person.

## 2. Threat model, by access type

Each recovery number in this campaign must carry one access tag. We name attackers by what they
receive, not by tier labels.

### 2.1 Target access (what the attacker sees about the person being attacked)

Exactly one of the following three views, from one fixed release:

| View | Target-time input |
|---|---|
| R1 income recipient | the income probability vector (2) plus the income decision |
| R2 occupation recipient | the occupation probability vector (6) plus the occupation decision |
| PAIR coalition | the aligned concatenation (R1, R2) for the same person |

Nothing else is available about the target. In particular, the attacker does not see the target's
raw inputs, SEX label, leaves, teacher scores or the other recipient's view, unless the PAIR view is
being scored.

### 2.2 Auxiliary access (what the attacker learns from)

- **Records.** Disjoint person groups (attacker-fit and attacker-validation), separate from the
  INNER scoring people. Duplicate/person groups are respected so that no person appears on both
  sides.
- **Per auxiliary record.** The same released view the attacker will see at target time, produced
  by the same deployed service, plus the record's SEX label.
- **Use.** Recovery models are fitted on attacker-fit groups. Orientation and model choice
  (including max(AUC, 1−AUC) orientation and family selection) use attacker-validation groups only.
  Scores are computed once on INNER.

### 2.3 Knowledge

- **Known:** the release format (vector dimensions, decision rule) and the defense family. The
  attacker knows that a fixed-routing leaf-correction service is used.
- **Not supplied:** model weights. Readers receive no tree structure, leaf values, δ tables, teacher
  model or training hyperparameters. "Defense-aware" here means only that the attacker trains on
  outputs of the actual deployed mechanism (a Kerckhoffs-style A2 in the project's access
  vocabulary). It does not mean white-box access.

### 2.4 Absent access (explicitly out of scope)

- target raw inputs;
- target sensitive labels;
- hidden tree leaves or routing paths;
- encoder activations (there is no encoder; the neural reference arm's internals are never
  released);
- gradients;
- clean, unreleased predictions. These are teacher scores before δ for any arm other than T_TREE,
  where the release *is* the teacher.

### 2.5 Interaction

- One fixed release per target. The services are deterministic, so a repeated query returns the
  same vector and adds no information. We still make no repeated-query claim.
- No adaptive or interactive querying with chosen inputs.
- No insider claim (an attacker holding the target's pre-protection outputs).
- No claim about composition with other datasets or releases beyond the two named services.

### 2.6 Linkage

The PAIR view assumes perfect person-level alignment between the two services' releases. This is
an **assumption** that makes the coalition as strong as possible. We do not establish that real
recipients can link records. Imperfect linkage would generally weaken the PAIR attacker. We report
no number for partial linkage.

### 2.7 Training distribution

Auxiliary records come from the same population and split procedure as the targets (Adult
development people, disjoint groups). Transfer to an auxiliary population that differs from the
target population is **not established**. A distribution-shifted attacker could be weaker, or
stronger if the shift happened to align with SEX. We report neither case.

### 2.8 Attackers (the reader slate)

Fresh readers are fitted per release and view:

- linear (logistic);
- MLP;
- boosted-tree readers.

Each is trained on attacker-fit groups, oriented and selected on attacker-validation groups, and
scored on INNER. The **primary** view AUC is the INNER AUC of the single reader that won on
attacker-validation groups (pair candidates include readers that ignore either recipient); it is
never chosen by reporting AUC. The maximum over the oriented slate on INNER (the "envelope") is
reported descriptively only. This slate is finite. Its readings are lower bounds on recovery by the best possible
attacker with this access. They are not certificates.

## 3. Why this access is relevant, without overclaiming

- **Historical labelled records.** A recipient of a prediction service may already hold its own
  records of some people with the sensitive attribute recorded, for example from historical
  customer, applicant or survey data. Together with the service outputs for those people, those
  records are exactly the auxiliary access in section 2.2.
- **Linked recipients.** Services may share predictions about the same linked people, for example
  when two downstream parties combine scores. The PAIR view models that combination.
- **Limits of the claim.** We do **not** claim that every recipient holds such records, that
  linkage is easy, or that the same-population assumption holds in deployment. The model describes
  a plausible and commonly studied recipient. It is not a worst case.
- **Stronger adversaries are out of scope.** Stronger access (white-box weights, leaves, repeated
  noisy queries, insiders) is excluded. Results under this model say nothing about those
  adversaries.

## 4. We did not invent this threat model [LIT]

- **Song & Shmatikov.** "Overlearning Reveals Sensitive Attributes" (arXiv:1905.11742; published at
  ICLR 2020 per the project's method catalog, `METHOD_CATALOG.csv` row EVL-01 on
  `research/combined-evidence-reconciliation-v1`). The adversary observes a model's inference-time
  representations or outputs for a target. Using auxiliary data labelled with the sensitive
  attribute, it trains a classifier that infers that attribute (and can de-censor or repurpose
  models). Our attacker is the same pattern applied to released prediction vectors instead of
  intermediate representations: auxiliary labelled records plus one observed output per target.
- **FARE.** Jovanović, Balunović, Dimitrov and Vechev, "FARE: Provably Fair Representation Learning
  with Practical Certificates", ICML 2023, PMLR 202:15401–15420.
  - FARE considers a **downstream consumer** who trains any classifier on the released
    (restricted-encoder) representation.
  - It provides a finite-sample, high-confidence upper bound on the unfairness of any such
    downstream classifier. The metric is demographic-parity distance, as used in the project's
    FARE reproduction (`dp_ub`).
  - This consumer is the same kind of party as our recipients: someone who fits models on released
    data with labelled examples.

So we adopt an auxiliary-data recipient that is standard in this literature. Our additions are
narrower:

- per-recipient prediction-vector releases;
- an aligned two-recipient coalition;
- a reader slate that is fresh, disjoint-group and validation-oriented.

## 5. Recovery outside a guarantee versus failure of a guarantee

These two things must not be conflated.

1. **Recovery outside a method's stated guarantee.** The measured attack uses access, a metric or
   an object that the method's guarantee does not cover. This is evidence about the method's
   *scope*. It is not evidence that the method is wrong. Examples:
   - a nonlinear reader against a linear-erasure guarantee (LEACE);
   - an attacker on separately released task predictions against a representation certificate;
   - a coalition of two releases against a single-release certificate;
   - AUC of a continuous score against a bound stated for a different quantity.
2. **Failure to satisfy the guarantee.** The measured quantity is exactly the certified quantity,
   on the certified object, under the certificate's assumptions (distribution, confidence level,
   attribute arity), and it exceeds the bound beyond the stated slack. Only this case says the
   guarantee does not hold.

**A demographic-parity certificate is not an AUC-recovery certificate.**

- FARE bounds the demographic-parity distance of downstream classifiers on its representation.
- In the special case of a binary attribute, where the certified variable is exactly what the
  attacker sees and the bound is taken over *all* classifiers, a DP bound does constrain each
  threshold's TPR − FPR for predicting SEX. It therefore limits AUC indirectly.
- None of those conditions holds for our releases. They are task-probability vectors from separate
  CatBoost services, not FARE's certified cell representation. The PAIR view combines two releases.
  The project's own FARE reproduction found the multiclass certificate reading 0.000 while
  attackers read 0.61 (project notes, AAAI rebuttal evidence). That points to a scope or arity
  mismatch, not necessarily to an invalid certificate.
- The rule for this campaign: report FARE or any certified baseline's AUC recovery as **outside its
  stated guarantee** unless all certificate conditions are verified to match.

## 6. Defense-specific caveats the adversary role insists on [DESIGN]

- **Exact teacher witness.** δ = 0 reproduces T_TREE. The family therefore contains the useful
  predictor. That is a property of the *family*. It is not evidence that a protected member exists
  inside the utility caps.
- **Fixed routing is not hiding.** Leaves are unreleased, but recipients see a deterministic
  function of x. Readers that learn the teacher's partition from (probabilities, decision) on
  auxiliary data are in scope. Boosted-tree readers in particular can approximate tree-structured
  releases well.
- **Online critics are not the evaluation.** The training critics (linear and MLP against SEX on
  each recipient's vector and on the aligned pair) are optimization devices. Every reported number
  comes from fresh readers that are never used in training, fitted on attacker-fit groups only.
  Readers whose family matches a critic must still be refit.
- **No KL trust region.** The correction can move confidence freely within the utility caps. NLL
  and summed Brier against U0 and Ucal must be checked per task and per seed. All 24 clauses use
  the original caps.
- **Equal-exposure neural reference.** This arm is used for internal selection only. It must see
  the same people, labels and selection roles. Its numbers are diagnostic, never the incumbent;
  RAW_PROJ remains the incumbent.

## 7. The ORIGINAL AAAI "informed" attacker: actual access

**Where it lives.** The AAAI-27 submission "Outputs Leak What They Use" is in the separate
`Bostesa/durable-guarantees` repository, not in PCRL.

- `git -C ~/PCRL grep -i informed main` returns **zero** hits, as does
  `rebuttal-evidence` [SI]. PCRL's `main` contains no informed-attacker code or text.
- The local clone `~/durable-guarantees` was relocated on 2026-09-30 to an
  external volume, which is not mounted
  (`~/storage-relocation-20260930/RESTORE.md`:
  `/Volumes/YOTUO/.../archives/tree-durable-guarantees.tar`). This role therefore could **not**
  re-open the original source.
- The evidence below is a committed source-inspection audit pinned to
  `durable-guarantees@956f5c883f515646aa457db55ecbd74b913768b2`, stored in PCRL on branch
  `research/combined-evidence-reconciliation-v1` @ `07a9ca3ff` [SR-2]. That audit also
  reimplemented the LRT independently on a Gaussian fixture and confirmed the classification.

**What the "Tier 2 / informed" attacker actually had** (`dg:` = durable-guarantees paths, as cited
by the audit):

| Item | Access | Evidence |
|---|---|---|
| Fitting data | **pre-noise (clean) representations** h of the 75% attacker-train rows of the PCRL train partition, with their protected labels | `dg:utils/battery.py:60-78` (fits N(μ_k, Σ_k + Σ_noise) on clean train rows), cited in `results/combined_empirical_preparation_v1/notes/methodology_durable/ACCESS_CONTRACTS.md:26` and `INDEPENDENT_CHECKS.json:745` |
| Mechanism knowledge | exact noise covariance σ²I (or σ²QQᵀ for subspace channels) | `dg:experiments/two_tier_certification.py:147`, cited in `INDEPENDENT_CHECKS.json:745` |
| Target-time input | **one** noised release per held-out target (a single draw) | `dg:utils/battery.py:88-103` (`gaussian_lrt`); `ACCESS_CONTRACTS.md:26` |
| Target's clean representation | **never** given | `ACCESS_CONTRACTS.md:26` ("it never sees the target's clean h"); `INDEPENDENT_CHECKS.json:746` ("not_implemented_by_tier2: (b) clean test-time h; (c) multi-query averaging") |
| Repeated queries / averaging | **not** part of Tier 2. Averaging was a separate experiment (N ≤ 16, subspace channel only) | `dg:experiments/averaging_attack.py:110-129`; `ACCESS_CONTRACTS.md:36-44` |
| Against deterministic baselines | collapses to QDA on the release (Tier-1 access) | `dg:experiments/baseline_gauntlet.py:52-56,329-333`; `ACCESS_CONTRACTS.md:27` |

**Verdict.** The owner's understanding is confirmed:

- Tier 2 had clean auxiliary training representations plus one noisy target release.
- It is **not** access to the target's clean representation, so it is not an insider attack.
- It is **not** a repeated-query attack.
- The audit's best name for it is *"mechanism-aware with white-box encoder access on a labelled
  population sample, single release"* (A4 in the project's A1–A5 vocabulary,
  `METHODOLOGY_IMPROVEMENTS.md:33-38,57`).
- The README and docstring describe it otherwise ("has seen the representation before noise was
  added … or anyone who can query the same row repeatedly and average"; `dg:README.md:62-66`,
  `dg:utils/battery.py:15-21`). The paper's wording is ambiguous ("knows the clean representation
  and the noise we added", `paper.tex:251-254`). All three citations are from `ACCESS_CONTRACTS.md:45-60`.
- **Substantive point.** Because the release distribution given s is the clean distribution given s
  convolved with known noise, the clean population vectors add mainly a correctly specified noise
  model, not information about the target.
- The audit's independent Gaussian fixture supports this (`INDEPENDENT_CHECKS.json:753`):
  - single-release defense-aware attackers sit at ≈0.620–0.624: clean-fit LRT 0.620, and a
    deconvolution LRT from released data only also 0.620;
  - an insider with the clean target h reads 0.756;
  - fresh averaging at N=16 reads 0.731.
- For subspace channels the LRT is *not* an upper bound. A knows-Q attacker reads 0.982 vs the LRT's
  0.856 on the easy cell (`INDEPENDENT_CHECKS.json:748`).

**Relation to this campaign.** The current campaign's attacker is weaker than the AAAI Tier 2 on
knowledge:

- no clean (pre-correction) outputs for auxiliary people;
- no model parameters;
- the services are deterministic, so there is no noise model to exploit.

It is A2-style: it trains on the actual released outputs with SEX labels. We must not describe it
as "informed", nor import AAAI Tier-2 readings as comparable numbers.

**Before any manuscript reuses this section**, the cited `dg:` lines should be re-verified against
the restored durable-guarantees tree. This role relied on the committed audit.

## 8. Replacement methodology passage (standalone, ≤350 words)

> **Threat model.** We evaluate recipients of released predictions, not holders of model internals.
> Two services score each person from the same permitted inputs, excluding the sensitive attribute.
> Each recipient receives only its own service's full-precision probability vector and decision for
> each person, from one fixed release. A coalition attacker receives both recipients' releases,
> aligned by person; we assume perfect linkage rather than establish it. Attackers learn from
> auxiliary records: groups of people disjoint from the targets, for whom they hold the same
> released outputs and the sensitive label. They know the release format and the defense family
> but receive no model weights, leaves, intermediate activations, gradients, unreleased clean
> predictions, target inputs or target labels. Auxiliary and target people come from the same
> population; we do not test transfer across populations, repeated or adaptive queries, or insider
> access. This is the auxiliary-data adversary of Song and Shmatikov (2020) applied to released
> outputs. It is also the downstream-consumer setting of FARE (Jovanović et al., 2023), whose
> certificate bounds demographic-parity distance rather than attribute-recovery AUC.
>
> **Measurement.** For each release and view (each recipient alone, and the aligned pair), we fit
> fresh linear, MLP and boosted-tree readers on attacker-fit groups. We orient and select them on
> attacker-validation groups, and score them once on held-out evaluation people. A view's recovery
> is the held-out AUC of the reader selected on attacker-validation data. Readers never participate in training. These
> finite-reader readings bound recovery from below; they are not certificates. Uncertainty comes
> from a paired duplicate-group bootstrap with fixed multiplicity, conditional on the fitted models
> and readers.
>
> **Interpretation.** When an attack uses access, an object or a metric that a baseline's guarantee
> does not cover, we report recovery *outside* that guarantee, not a failure *of* it. We call a
> guarantee violated only when the certified quantity, measured under its stated assumptions,
> exceeds its bound. In earlier work our "informed" attacker combined clean representations of a
> labelled population with one noisy release per target. It had neither the target's clean
> representation nor repeated queries, and we label it accordingly.

(≈330 words.)
