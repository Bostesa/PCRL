# PCRL application selection: reusable release must earn its place

Date: 2026-09-07. Starting commit: `15dbcc3c3338f6707e7a0b3901d738d99651a77d`.
This document is a development screen, not an evaluation of a privacy method.

**Decision: select no real-data pilot in this stage.** Three applications were
screened: wearable activity analysis, diabetes encounter research, and ACS
microdata analysis. Available data and genuine labels make all three technically
accessible. None currently supplies evidence that an untrusted recipient needs
a reusable representation beyond a prediction release, adequate prediction bank,
or trusted computation service. None supplies both a defensible purpose policy
and an independently motivated held-out task family. This is a missing application
specification, not evidence that reusable representations are never useful.

The preceding synthetic experiments established conflicting purposes and exposed
nonlinear leakage and adversary optimization failures. Prediction-only releases
performed well on their fixed tasks. Those results do not establish a real need
for representation access. No new representation method was fitted here, and no
real-data final-test outcomes were inspected or used for selection. One fixed
learned-output reference was subsequently run on disjoint HAR development
participants; it is described separately from the original oracle checks below.

## Admission criteria

An application must pass all five gates before a method comparison:

1. **Release necessity.** Specify the data-holder/recipient roles and the work
   the recipient must perform, and evidence that per-task predictions, a suitable
   bank of predictions, and a trusted fitting/query API do not meet that work.
   Offline access, arbitrary adaptation, or latency cannot simply be assumed.
   For a scientific benchmark, a precise, consequential held-out task interface
   can justify this question without a deployment partner or signed contract.
2. **Purpose and threat.** Specify allowed tasks, forbidden variables, side
   information, release granularity, repeated access, and recipient collusion.
   State which information authorized outputs necessarily reveal. A public
   benchmark with protected columns is not itself evidence of such a policy.
   A research policy may be hypothetical if explicitly labeled and coherent.
3. **Transfer task family.** Name genuine labels and a meaningful family of tasks
   withheld from encoder fitting. Reweightings, thresholds, and coarsenings of
   an already released label are insufficient by themselves: a suitable
   conditional distribution or prediction bank can support them. Joint queries
   require an appropriate joint bank, not an unfairly weak bank of marginal
   predictions.
4. **Development feasibility.** Check label coverage and authorized-output
   leakage before choosing privacy thresholds. Distinguish absolute attribute
   privacy from limiting additional leakage beyond authorized outputs. The
   latter is a different objective and must be stated explicitly.
5. **Clean evaluation.** Reserve task labels as well as observations; use
   appropriate person/session/time groups, train-only preprocessing, separate
   calibration and attacker fitting, validation-only selection, and sealed final
   evaluation. Attribute protection must include nonlinear attacks and collusion.

The application gate separates **scientific usefulness** from **operational
necessity**. A public dataset may support a legitimate controlled research
question without an actual partner. Conversely, results on an invented task
interface do not prove a deployment needs embeddings. The rejection below rests
on inadequate independent task families and unexcluded output/API alternatives,
not merely on the absence of stakeholder contracts.

Prior work already studies fair representations transferred to unseen tasks:
[LAFTR](https://proceedings.mlr.press/v80/madras18a.html) motivates third-party
downstream objectives, and
[FFVAE](https://proceedings.mlr.press/v97/creager19a.html) adapts a representation
to new labels and subgroup definitions. These are relevant research precedents.
Their existence does not document an operational need in these local datasets.

## Local access inventory

Inventory used file names, sizes, documentation, and existing preprocessing
source. Sizes are actual aggregate on-disk file sizes in MiB; archives and
expanded versions are both counted. Inventorying a file is not evaluating it.

| Local directory | Available material | MiB | Screen disposition |
| --- | --- | ---: | --- |
| `data/UCI HAR Dataset/` | 28 text files; features, inertial windows, activity and subject labels | 269.47 | Candidate 1 |
| `data/diabetes/` | Raw encounter CSV, archive, ID mapping | 21.47 | Candidate 2 |
| `data/diabetes_processed/` | Three split NPZs, feature names, metadata | 1.93 | Candidate 2; train NPZ only used |
| `data/folktables/` | CA 2018 person CSV and cached processed parquet | 257.86 | Candidate 3; contents not analyzed |
| `data/adult/` | Official data/test files | 5.70 | Inventory only; no fourth application |
| `data/hmda_raw/`, `data/hmda_processed/` | CA 2023 CSV, three NPZs, metadata | 103.52 + 1.72 | Inventory only; no fourth application |
| `data/celeba/` | 202,599 images, attribute and partition CSVs | 1,371.01 | Inventory only; no fourth application |
| `data/bios/` | Absent | — | No local BIOS access; no reconstruction/download attempted |

No package installs, model downloads, data downloads, paid resources, or changes
to historical datasets/results were made. Existing local numerical dependencies
were sufficient for the small label checks below.

## Candidate 1: wearable activity analysis

**Data evidence.** UCI HAR supplies manually annotated activity and participant
IDs from 30 volunteers performing six activities. The official split separates
participants, and sensor windows overlap by 50%. The local official training set
contains 7,352 windows from 21 people. These are activity and identity labels;
fall risk, gait pathology, energy expenditure, and future health outcomes are
not supplied labels. The source establishes activity-recognition research, not
a contract requiring feature transfer to independent recipients.
[Official UCI dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
[original dataset paper](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf).

**Policy under review.** The existing `pcrl/data/har.py` gives an activity recipient
six-way activity classification while forbidding subject identity. A second
recipient gets active/sedentary classification while subject identity and the
full activity label are forbidden. The second task is deterministically computed
from the first. These purpose names are repository choices, not verified
stakeholder requirements.

For a hypothesized third-party recipient, the attack would be inference of a
known participant or fine activity from window releases, with labeled auxiliary
windows and repeated releases. A combined recipient authorized for six-way
activity necessarily gains that label; only identity can remain forbidden under
the proposed union of authorized tasks. A new-person deployment would require
a separately defined identity/linkage threat: closed-set identity classification
cannot evaluate unseen subject IDs as though they were supported known classes.

**Output compatibility.** Releasing the exact active/sedentary bit already reveals
which three activities are possible. On the development check below, this raises
six-way activity accuracy from 0.194016 to 0.357208 and gives 0.687253 nats of
held-development log-loss improvement. Absolute fine-activity independence is
therefore incompatible with preserving this exact coarse label. A defensible
replacement might permit the coarse label and prohibit *additional* fine-activity
information, conditional on that label; that would be a revised policy, not a
successful result under the existing absolute policy.

**Reuse and held-out tasks.** A six-way probability vector supports all deterministic
coarsenings and cost-sensitive decisions based on those six labels. The active
task therefore provides no convincing transfer test. Holding out one of the six
classifications from a bank selected to contain only five would also be an
artificially weak comparison. No local independent outcomes support the more
compelling proposed tasks above. A feature release might help a recipient learn
new genuine sensor tasks, but neither the labels nor that recipient's need is
established here. A trusted on-device classifier remains an unexcluded option.

**Evaluation blockers.** The current HAR loader combines official train/test,
normalizes on the combined data, then randomly splits overlapping windows. It
must not be reused for a new sealed evaluation. An eventual identity study needs
session-separated auxiliary/evaluation windows for known people, or an explicit
linkage evaluation on unseen people; adjacent overlapping windows must not cross
splits. The present archive does not supply an obvious session-ID manifest.

**Decision: reject for the current pilot.** Reconsider only with independently
annotated permitted tasks, a justified offline/third-party adaptation requirement,
and a coherent output-aware policy. The local README contains an older
noncommercial-use restriction, whereas the current UCI page labels the data
CC BY 4.0; retain both provenance records rather than silently resolving the
difference. No redistribution is proposed here.

## Candidate 2: diabetes encounter research

**Data evidence.** The public dataset records diabetes encounters from 130 US
hospitals/networks during 1999–2008. It includes diagnoses, medications,
readmission, and demographic variables. Its published motivating endpoint is
30-day readmission; this does not establish an external recipient's need for an
embedding or validate the repository's named organizational purposes.
[Official UCI dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008),
[original study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/).

**Policy under review.** `pcrl/data/diabetes.py` specifies diagnosis grouping for
`billing_audit` with race/gender forbidden; readmission for `quality_research`
with race/age forbidden; and medication change for `clinical_decision_support`
with race/gender forbidden. These are hypothesis labels for purposes, not
documented operational agreements. A single forbidden demographic attribute
common to all purposes is not a genuine task-versus-task conflict by itself.
The dataset does not establish which recipient is entitled to infer age, sex,
or diagnosis and which is not.

A hypothetical recipient would receive encounter representations and labeled
auxiliary patients; attacks infer the stated demographic labels, including from
task outputs. Combining the three releases authorizes their three outcomes.
Maintaining all three original privacy restrictions would then forbid race,
gender, and age. That combined policy requires an explicit definition, because
authorized outcomes can themselves be associated with those attributes.

**Output compatibility.** The train-only checks show demographic association
even in true task labels: diagnosis grouping raises held-development gender
accuracy from 0.522842 to 0.543620. A combined exact-label bank improves age
log loss by 0.050159 nats. Several accuracy figures stay at the majority rate
despite improved log loss, so majority accuracy is not a satisfactory absence-of-
information test. These are limited finite-sample associations, not a universal
privacy impossibility bound for noisy predictors or a medical conclusion.

**Reuse and held-out tasks.** The three existing outputs can be supplied by a
small prediction bank. Changing diagnosis groupings or the readmission horizon
category alone does not establish representation necessity; a richer output
distribution should be considered first. A meaningful new family might concern
independently measured prospective clinical endpoints unavailable during encoder
fitting, but these local encounter files do not provide that family or a partner
who must fit it outside a trusted service. Treating existing medication fields
as new labels while retaining their post-outcome proxies in features would be
misleading. No admission-time clinical-decision claim is justified from the
current feature set, which includes discharge information and encounter-level
medication measurements.

**Evaluation feasibility.** The current preprocessing already retains one
encounter per patient; a repeated-patient leakage defect must not be invented.
It does, however, compute imputation values and discovered categorical domains
before splitting, although numeric normalization uses training statistics. A
new pilot would need fresh split manifests, fitting-only preprocessing, outcome
timing review, and documented category coverage. The small processed data are
locally cheap to load, but CPU feasibility does not repair these specification
gaps. UCI currently lists CC BY 4.0; no new access request is needed for a local
research screen.

**Decision: reject for the current pilot.** The necessary missing evidence is an
independently motivated reserved endpoint family with adequate timing/provenance,
an external-fitting interface that provides measurable utility beyond a suitable
bank, and a coherent purpose-grounded privacy policy. An operational deployment
claim would additionally need evidence about an actual recipient. Do not relabel a
finite multitask benchmark as evidence of reusable clinical utility.

## Candidate 3: ACS microdata analysis

**Data evidence.** The local material is California 2018 ACS person microdata
plus a repository cache. Census explicitly supports custom estimates unavailable
in pretabulated products and already offers public microdata access and query
tools. This is the strongest evidence of a real reuse task among the three
candidates, but it does not establish that a purpose-specific learned embedding
is the required interface.
[Official Census PUMS description](https://www.census.gov/programs-surveys/acs/microdata.html).

Folktables provides multiple predictive tasks and deliberately distinguishes
algorithm benchmarking from substantive domain investigations. Its income,
employment, public coverage, mobility, and travel tasks are relevant labels,
not evidence of a deployment requiring transfer learning.
[Official Folktables scope and task definitions](https://github.com/socialfoundations/folktables),
[original paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/32e54441e6382a7fbacbbbaf3c450059-Paper.pdf).

**Policy under review.** The repository permits income prediction while forbidding
sex/race; employment while forbidding sex/race/disability; and public coverage
while forbidding sex/race/age group. None specifies a real data-holder/recipient
agreement, and all three outcomes are associated by plausible mechanisms with
some forbidden attributes. A hypothetical colluding recipient with all three
outputs would still need an explicit rule on sex, race, disability, and age.
Public source access also matters: if the adversary can link released rows to
the public microdata and labels, an erasure-only attack model is incomplete.
The existence or accuracy of such linkage has not been measured here.

**Reuse and held-out tasks.** The three existing binary tasks fit in a tiny
prediction bank. Mobility/travel labels could make a broader benchmark only if
their allowed purpose is justified and their targets/proxies are excluded from
encoder inputs; merely withholding a known column is not proof that a bank or
trusted query interface is inadequate. Custom subgroup estimates are more
credible reuse, but suppressing subgroup variables can directly frustrate that
authorized analysis. A survey-estimation study would also need sampling weights,
uncertainty, and a clear permitted query family, not just individual accuracy.

**Development status.** No ACS outcomes or correlations were analyzed in this
screen. The reusable-need and policy gates fail before statistical tuning, and
the cached parquet mixes the existing deterministic train/validation/test rows.
Its source was inspected without opening its outcome contents. Before any future
pilot, export and seal explicit row/group manifests, inspect missing-label
semantics, and compute task-output leakage within development only. Existing
repository preprocessing fills missing targets with zero and differs from some
official Folktables population filters; those choices need justification rather
than attribution to the canonical task definitions.

**Decision: reject for the current pilot.** This is the first candidate to revisit
if a bounded permitted query/adaptation family supplies independent task holdouts
and explains why released predictions, a sufficiently rich bank, existing
microdata tools, and a trusted fitting service cannot satisfy it. No operational
need of that kind is established by the available sources. A documented research
interface could justify a scientific pilot even before there is a deployment.

## Explicit candidate purpose mappings

These are the existing repository policies being screened, not approved
operational policies. `Required` names the utility target; `permitted` names
additional output uses consistent with that target. No unspecified attribute
inference is treated as approved merely because it is absent from the table.

| Candidate / recipient | Required prediction | Additionally permitted | Forbidden inference in the current policy |
| --- | --- | --- | --- |
| HAR activity | Six-way activity | Coarsenings/cost decisions from activity | Subject identity |
| HAR health monitoring | Active/sedentary | Decisions using that bit | Subject identity; six-way activity (incompatible in absolute form) |
| HAR combined | Both preceding tasks | Their joint decisions; six-way activity | Subject identity |
| Diabetes billing | Nine-way primary diagnosis group | Decisions/coarsenings from that group | Race; gender |
| Diabetes quality | 30-day readmission binary | Decisions using readmission output | Race; age bucket |
| Diabetes clinical support | Medication change binary | Decisions using change output | Race; gender |
| Diabetes combined | All three targets | Joint decisions from their outputs | Race; gender; age bucket, if all individual restrictions persist |
| ACS income | Income over $50,000 | Decisions using that binary output | Sex; race |
| ACS employment | Employment binary | Decisions using that output | Sex; race; disability |
| ACS public coverage | Public coverage binary | Decisions using that output | Sex; race; age group |
| ACS combined | All three targets | Joint decisions from their outputs | Sex; race; disability; age group, if all restrictions persist |

For all three, a representation recipient's hypothetical attack budget includes
independent labeled auxiliary fitting, nonlinear probes, repeated records where
available, and collusion. Public-data linkage and available side information
require separate specification. The mapping deliberately exposes unresolved
conflicts; it is not a claim that these policies are simultaneously achievable.

## Development-only output/label checks actually executed

For the original oracle checks, only `train/y_train.txt`, `train/subject_train.txt` from official HAR and
`data/diabetes_processed/train.npz` were analyzed. No features were fitted, no
privacy methods were compared, and no validation/test NPZs or official HAR test
files were opened. These are **oracle-label-release diagnostics**, not attacks
on learned prediction heads. Actual prediction-only and probability-bank leakage
must be measured if an application is later admitted; true-label results cannot
be silently substituted for those measurements.

Protocol: one `numpy.random.default_rng(9137)` generator, first permuting 7,352
HAR training rows, then 50,053 diabetes training rows. Each is split 70/30 into
conditional-frequency fitting and development checking: HAR 5,146/2,206;
diabetes 35,037/15,016. Fit add-one-smoothed `P(attribute | exact task label)`
and an add-one-smoothed attribute prior. Score their predictions on development
rows. `log-loss gain` is prior cross-entropy minus conditional cross-entropy in
nats; positive means improved attribute prediction. No parameters were tuned.
HAR's random window partition is adequate only for this elementary label-policy
check, not for claiming independent-session generalization. Small gains are
descriptive: no confidence intervals or significance claims were computed.

| Dataset, exact output → attribute | Prior accuracy | Conditional accuracy | Conditional balanced accuracy | Log-loss gain (nats) |
| --- | ---: | ---: | ---: | ---: |
| HAR active/sedentary → activity | 0.194016 | 0.357208 | 0.333333 | 0.687253 |
| HAR activity → subject | 0.058477 | 0.062557 | 0.056139 | -0.008732 |
| HAR active/sedentary → subject | 0.058477 | 0.061197 | 0.058312 | 0.001536 |
| Diabetes diagnosis → race | 0.749068 | 0.749068 | 0.200000 | 0.005105 |
| Diabetes diagnosis → gender | 0.522842 | 0.543620 | 0.538663 | 0.003356 |
| Diabetes readmission → race | 0.749068 | 0.749068 | 0.200000 | 0.000270 |
| Diabetes readmission → age | 0.250333 | 0.250333 | 0.100000 | 0.001255 |
| Diabetes medication change → race | 0.749068 | 0.749068 | 0.200000 | 0.000442 |
| Diabetes medication change → gender | 0.522842 | 0.522842 | 0.500000 | 0.000216 |
| Diabetes joint three-label bank → race | 0.749068 | 0.749068 | 0.200000 | 0.004475 |
| Diabetes joint three-label bank → gender | 0.522842 | 0.538159 | 0.531068 | 0.003147 |
| Diabetes joint three-label bank → age | 0.250333 | 0.250067 | 0.103196 | 0.050159 |

All schema classes above have support in the fitting and development subsets.
Full inspected training-pool counts (these are coverage, not final-test counts):

- HAR activity, source class order: `[1226,1073,986,1286,1374,1407]`;
  active/sedentary `[4067,3285]`; 21 subject counts
  `[347,341,302,325,308,281,316,323,328,366,368,360,408,321,372,409,392,376,382,344,383]`.
- Diabetes diagnosis, repository `DIAG_ORDER`:
  `[15257,4072,4611,3369,2859,6811,2426,1917,8731]`;
  readmission `[45650,4403]`; medication change `[27641,22412]`.
- Diabetes race, repository `RACE_ORDER`: `[37436,9041,1052,342,2182]`;
  gender `[26553,23500]`; age decade bins
  `[107,379,805,1917,4814,8733,11202,12730,8054,1312]`.

The empirical mutual information between HAR activity and its deterministic
active/sedentary coarsening on this training pool is 0.687480 nats. This statistic
is separate from the held-development predictive log-loss gain. Neither statistic
is a certificate against general classifiers on representations.

### Actual learned-output reference on new participant groups

The fixed reference protocol was written to the new results directory before
this model was fitted. It is distinct from the retrospective description of the
earlier label checks. Seed 9138 assigns the 21 official training participants to
13 task-fitting, four attacker-fitting, and four development participants. These
are 4,697, 1,361, and 1,294 windows respectively. Entire participants are separated,
so overlapping windows cannot cross these roles. The official test remains sealed.

One `StandardScaler` followed by binary logistic regression was fitted on the
task-fitting participants: `C=1`, `lbfgs`, `tol=1e-4`, `max_iter=2000`. It converged
in nine iterations. No validation selection, head search, or representation
learning was performed. Its hard active/sedentary output and its scalar
probability were released to a fixed add-one categorical attacker fitted on the
four attacker participants. The probability attack uses 20 fixed equal-width
bins, specified before fitting. All six activity classes have support in both
attacker and development groups. Unknown-person identity classification is not
measured under this participant-disjoint protocol.

The input consists of UCI's supplied 561 engineered, normalized features. The
new scaler was fitted only on task-fitting participants; this screen does not
reconstruct or audit the original provider's normalization procedure. The
development reference is therefore evidence about these supplied features, not
a fully reconstructed raw-sensor pipeline. The deterministic oracle-label policy
conflict is independent of that preprocessing caveat.

The task accuracy on the four development participants is **0.999227** (balanced
accuracy 0.999293; log loss 0.002496 nats). Fine-activity leakage is:

| Released output | Activity accuracy | Activity balanced accuracy | Log-loss improvement over prior (nats) |
| --- | ---: | ---: | ---: |
| No output: attacker-fitting prior | 0.198609 | 0.166667 | 0 |
| Oracle active/sedentary bit | 0.363215 | 0.333333 | 0.685420 |
| Actual learned hard output | 0.363215 | 0.333333 | 0.681384 |
| Actual learned probability, fixed 20-bin attacker | 0.363215 | 0.333333 | 0.681398 |

The actual task output therefore reproduces the expected policy conflict on
disjoint participants. The probability attack's tied accuracy does not establish
that full-precision scores add no information: this is one bounded diagnostic,
with no universal guarantee and no privacy pass declared. The learned reference
is useful for checking the policy, but creates no new held-out task family and
does not rescue the application-selection gate.

Total script time was **5.072659 seconds**, including inventory, hashing, loading,
oracle reproduction, and the learned reference; the tool command elapsed
**6.073571 seconds**. The learned task plus attackers took **0.067769 seconds**.
Execution used one numerical thread on the existing Apple M4 Pro host
(`macOS-15.6-arm64`, 14 logical CPUs). No methodological sweep was run.

Fresh artifacts, all relative to the repository root:

- `experiments/screen_pcrl_applications.py`: full reproducible screening program.
- `results/redesign_20260907_application_screen_v1/PROTOCOL.md`: explicitly
  retrospective oracle chronology and the fixed learned-reference protocol.
- `results/redesign_20260907_application_screen_v1/metrics.json`: raw scores,
  full class/support coverage, subject role assignments, configuration, exact
  source/data hashes, starting commit, and timings.
- `results/redesign_20260907_application_screen_v1/local_data_inventory.json`:
  local file counts/sizes, dataset versions/sources, and hashes of the training
  files whose contents were accessed.
- `results/redesign_20260907_application_screen_v1/learned_task_reference.npz`:
  task coefficients/intercept, fitted scaler, role indices and actual outputs.
- `results/redesign_20260907_application_screen_v1/reproduction.log`: captured
  command output. Earlier oracle observations were checked against their already
  viewed exact values and reproduced without corrections or superseded runs.

### Reproduction

The complete executed command was:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u experiments/screen_pcrl_applications.py --out-dir results/redesign_20260907_application_screen_v1
```

Use a new output directory to rerun; the script refuses to overwrite an existing
one. The command output was captured through `tee` and retained as
`reproduction.log`. No external test suite was run for this documentation/data
screen; the script asserted disjoint participant groups, class coverage, task
convergence, and exact reproduction of earlier oracle values. Its saved source
hash was independently checked against the executed file after completion.

Run from the repository root with the existing environment. The following
reproduces the predictive table without touching final-test files:

```python
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

rng = np.random.default_rng(9137)

def inspect(name, arrays, tasks, attrs, joint=False):
    n = len(arrays[tasks[0]])
    order = rng.permutation(n)
    fit, dev = order[:int(.7*n)], order[int(.7*n):]
    outputs = {k: arrays[k] for k in tasks}
    if joint:
        outputs['joint'] = np.ravel_multi_index(
            np.stack([arrays[k] for k in tasks]), (9, 2, 2))
    for task, y in outputs.items():
        for attr in attrs:
            z = arrays[attr]
            T, K = int(y.max()) + 1, int(z.max()) + 1
            assert len(np.unique(z[fit])) == len(np.unique(z[dev])) == K
            counts = np.ones((T, K))
            np.add.at(counts, (y[fit], z[fit]), 1)
            p = (counts / counts.sum(1, keepdims=True))[y[dev]]
            prior = (np.bincount(z[fit], minlength=K) + 1)/(len(fit) + K)
            base = np.tile(prior, (len(dev), 1))
            print(name, task, attr,
                  accuracy_score(z[dev], base.argmax(1)),
                  accuracy_score(z[dev], p.argmax(1)),
                  balanced_accuracy_score(z[dev], p.argmax(1)),
                  log_loss(z[dev], base, labels=np.arange(K))
                  - log_loss(z[dev], p, labels=np.arange(K)))

a = np.loadtxt('data/UCI HAR Dataset/train/y_train.txt', dtype=int) - 1
s = np.loadtxt('data/UCI HAR Dataset/train/subject_train.txt', dtype=int)
_, s = np.unique(s, return_inverse=True)
inspect('HAR', dict(activity=a, is_active=(a < 3).astype(int), subject=s),
        ['activity', 'is_active'], ['subject', 'activity'])
with np.load('data/diabetes_processed/train.npz') as source:
    keys = ['primary_diagnosis_category', 'readmission_outcome',
            'medication_change_outcome', 'race', 'gender', 'age_bucket']
    d = {k: source[k] for k in keys}
inspect('Diabetes', d, keys[:3], keys[3:], joint=True)
```

This intentionally prints some additional task/attribute pairs beyond the
existing purpose policy; the table highlights the policy pairs and combined
recipient. The oracle HAR activity→activity row is trivially perfect.

## Evidence needed before reopening the pilot gate

The most informative next application step is a **concrete reuse specification**,
not another method run: specify one recipient role and one independent, genuine
reserved task family, then write the allowed/forbidden and combined-access policy
alongside a documented reason that an adequate prediction bank or trusted fitting
service fails the specified work. ACS custom research is the best
starting point for this investigation, but is not a selected pilot.

Once that evidence exists, freeze the task family and compare prediction-only,
an adequate prediction bank, and trusted computation against reusable releases
on development data before selecting a representation method. Audit actual
outputs, task-conditional leakage if authorized, covariance, nonlinear attacks,
and combined releases separately. Keep all final-test tasks and records sealed
until that specification and method selection are complete.

## Provenance and scope

Actual starting repository commit:
`15dbcc3c3338f6707e7a0b3901d738d99651a77d`.
Relevant reads: `PCRL_REDESIGN_STATUS.md`, `docs/DATA.md`, `data/README.md`,
`pcrl/data/{har,diabetes,folktables}.py`,
`experiments/preprocess_diabetes.py`, and existing dataset metadata/README.
No `AGENTS.md` was found in the repository instruction search or direct ancestor
checks. Primary web sources above were checked on 2026-09-07. This screen used
the existing `.venv/bin/python`; original exploratory values were preserved in
the reproducible run described above. This screen wrote this document, the new
screening script, and the fresh application-screen results directory. Historical
source, data, and results were not changed.

## Amendment — 2026-09-07: admit the defined ACS transfer benchmark

The original screen above remains intact as a historical decision. A defined
one-time release setting is sufficient to investigate cross-task transfer.
We do not require proof that every possible trusted service is inadequate before
running a scientific benchmark. A trusted service remains a deployment
alternative, not an empirical admission gate.

The now-authorized setting is a hypothetical data owner releasing a fixed
per-record interface to a researcher who may fit new supervised heads but cannot
retrain or query the owner's encoder. Prediction banks and representations have
identical downstream fitting privileges. California 2018 ACS public records
simulate this interface; this experiment cannot claim to protect those public
records against linkage. Source families are personal income, labor-force status
and public coverage. Residential mobility and commute duration are fixed withheld
task identities in a common19–34 positive-person-weight cohort. They support
a controlled study of contemporaneous young-adult socioeconomic, residential
and commuting information, not a Census deployment or longitudinal prediction.

This specification admits an **unprotected transfer comparison**. It does not
require attribute privacy, prove representation necessity, establish PCRL novelty,
or justify a protection mechanism. The competitive bank is part of the test.
SEX/RAC1P recoverability is measured without collapsing the recorded schema.
No synthetic audit is repeated. See the [frozen ACS protocol](../results/redesign_20260907_acs_transfer_v1/PROTOCOL.md).
