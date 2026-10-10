# R01 source and permitted-input map

Status: read-only admission audit, 2026-10-09. `SOURCE_AUDIT.json` and
`DUPLICATE_UNIT_AUDIT.json` bind the byte and membership checks. No new fit,
task/SEX label decompression, raw feature matrix decompression, or mixed-role
source deserialization occurred in this audit. The old assessment remains
embargoed. A local Git identity is not a fresh remote publication check.

## Local source identities

| Role | Verified local branch tip | Use |
|---|---|---|
| Repaired utility/reference source | `research/pcrl-utility-reference-repair-v1` `ce1308a52fb97b4673d26f44c1466ce3b0cea379` | Common INNER evaluation, U0/Ucal parity, admitted role-only exports |
| Focused feature predecessor | `research/pcrl-capacity-matched-feature-selection-v1` `9e6bfea62e82297d024fd9588a5f4bfb30e0a9de` | Prior split/negative mask evidence; no new writes |
| Portfolio predecessor | `research/pcrl-method-portfolio-v1` `38d641189d3e253ddff0ca372d295a034752017f` | Safe export admission and T* preprocessing receipts |
| A prototype worktree | `research/pcrl-method-portfolio-a` `3a3e56f4811be94e520783723ce42573b8a577f1` | Source review only; protected Adult fit not run |
| D prototype worktree | `research/pcrl-method-portfolio-d` `094ac4fca52c7912b2d58c2a73309fb5381babe4` | Source review only; synthetic prototype, no Adult fit |
| Hcal source | `research/pcrl-heldout-calibration-v1` `1baf5bbdaf59cfa6a664cae07d652a4712bdfdc5` as pinned in admitted manifests | Frozen source/model/calibration history |

The A/D names come from the portfolio `AGENT_ROSTER.json` worktree registry and
local branch tips, not guessed labels. The D prototype implements a shared
VQ codebook on U encoder latents. The new campaign proposal uses a different
10-bit sigmoid mapper, so its outcomes cannot inherit the prototype's tests.

## Actual role-only source interface

Use the predecessor repair worktree's
`results/pcrl_utility_reference_repair_v1/.private/role_export_v1/{ROLE}/{FIELD}.npz`
until a byte-verified, campaign-owned private copy is staged. Its public
allowlist and hashes are in
`results/pcrl_utility_reference_repair_v1/m1/DATA_ACCESS_MANIFEST.json`.
R01 verified all 25 files against both that manifest and the portfolio's
`DATA_ACCESS_MANIFEST.json`. The mixed source file was byte-hashed only; it is
not a scientific input. The inherited all-role loader chain is prohibited.

| Role | Rows / exact-record groups | Permitted new use and fields |
|---|---:|---|
| `OSF_DEFENSE_FIT` | 15,434 / 15,428 | New task/defense fits: `features`, `income`, `occupation`, `sex`, `references_s0/s1`; use existing FIT-internal group folds for checkpoint/recipe selection. SEX is fitting supervision only. |
| `CALIBRATION_HELDOUT` | 2,000 / 2,000 | Calibration: `features`, `income`, `occupation`, `references_s0/s1`. No SEX export in this role. |
| `ATTACK_FIT_NEW` | 4,064 / 4,061 | Attack fit/tuning: `sex`, candidate public release built from `features` or fixed bank, `references_s0/s1`; inherited permitted audit-fitting rows require separate explicit role admission. No task-label export here. |
| `INNER_SELECTION` | 2,235 / 2,234 | Reused development screen: `features`, `income`, `occupation`, `sex`, `references_s0/s1`. No fitting or checkpoint choice from INNER outcomes. |

Every archive's ordered `row_id,unit` pair is hash-bound. R01 read only those
members from the four feature archives and found **zero row and group overlap
for all six role pairs**, including FIT/CAL/INNER. `unit` is the inherited
exact-record duplicate group and bootstrap unit, not a new individual split.
FIT has six doubleton groups, ATT three, INNER one; all other rows are unique
within those roles. For seeds 0 and 1, every member of each duplicate group
has bitwise identical U0 income/occupation vectors and T* fine cells. A
unit-keyed offline draw is therefore input-unambiguous for the audited source;
model seed and recipient must also be part of the key.

The ordered predictor schema has exactly 83 float32 columns and newline-join
SHA-256 `7101f2fb37b623a08221c32b33eedf132cf0c4c376bb4042ab9cf16da08ad975`.
Five numeric predictors are `age`, `education-num`, `capital-gain`,
`capital-loss`, and `hours-per-week`; 78 fixed one-hot columns come from
`workclass`, `education`, `marital-status`, `relationship`, and
`native-country`. The source preprocessing fits numeric moments on FIT.
Excluded source columns are SEX, race, income, occupation, and fnlwgt. The
source occupation target is `occupation_group`, a six-class map of occupation,
**not employment**. Income has classes `0: <=50K`, `1: >50K`; occupation
classes are ordered integers `0..5` per `pcrl/data/adult.py`'s
`OCCUPATION_GROUPS`. Source SEX coding is `0: Female`, `1: Male`.

## Frozen reference and D latent inputs

The admitted U0 source is hcal `rel__s{k}__U`, `smf tl__s{k}__e40`: a 20-epoch
task warm start and 40-epoch task-only continuation of separate
`83 -> 64 -> 64 -> 16` encoders, followed by fitted logistic deployed heads.
The encoder/head fitting used all FIT rows and labels. Historical deployed-head
C selection used HEAD_VALIDATION labels, which remain barred now. Verified
`model.pt` hashes are s0
`5061a0074af74d21b53aca80b96d8b83b85a3e43cdb9e6758b2551be9045b9a7`
and s1
`f52bef54ef77faf51cb6f55273e54f75ea094c4379bbf73b57a3941ce2f7a74c`.
`SOURCE_AUDIT.json` gives the exact head and record hashes. The frozen D
input is `Model.encode(0,X)` for income and `Model.encode(1,X)` for
occupation, each 16-dimensional **before** any erasure map or source head.
R01 verified the checkpoint and interface identity statically; a new
role-only latent forward parity receipt still belongs after the R00 seal.

Ucal applies one global inverse temperature per task/seed to U0. Its parameter
fit used CAL labels. The *calibration family* was historically selected using
INNER outcomes. The repaired package restored U0/Ucal on FIT/CAL/INNER with
48/48 bitwise comparisons and no difference. R01 inherits that result, rather
than claiming a fresh forward replay. U0 parameters, D frozen encoder and
Ucal temperature fit have zero direct row/group overlap with INNER, but the
Ucal family has previous INNER selection exposure. For all new candidate and
control fits, the intended membership is FIT for fitting and CAL for
calibration; actual overlap must be rechecked against each write-once fit
receipt before reporting a comparison. A role held out from a new mapper does
not erase exposure of its frozen source model.

## A source bank and channel inputs

Existing predecessor portfolio private exports
`shared/.private/tstar_score_bank_s{k}_v1/` contain `bank_r1.npz`,
`bank_r2.npz`, and `tokens_{OSF_DEFENSE_FIT,ATTACK_FIT_NEW,INNER_SELECTION}.npz`
for seeds 0 and 1. R01 verified their byte hashes, receipt source-binding
SHA, teacher checkpoint, 83-column schema and row/reference hashes. Each
fixed T* bank has 16 income token rows with 2-class vectors and 321
occupation token rows with 6-class vectors. Direct exact-vector-plus-decision
deduplication preserves all 16/321 output identities in both seeds; there are
no duplicate bank outputs. The fine input has 64 income cells (32 per class)
and 641 occupation cells (128 per class 0–4, one in class 5). Every output is
used by the inherited deterministic map; 48 income and 320 occupation fine
cells map to an output already used by another fine cell. Every class has a
strict-argmax bank entry. These fixed-table results and their bound array
hashes are in `BANK_IDENTITY_AUDIT.json`. The predecessor's independent S4
numerical bank parity remains **pending**; hash/role admission alone does not
establish a freshly replayed decoder.

The A channel input is the private qpc **fine cell** for each recipient,
computed from its U0 score. The historical coarse T* code is the deterministic
`fine_to_hcal_token` mapping and output-bank index, not a substitute input.
The fine-cell totals and class mapping were directly checked from the bound
fixed tables; their role-only per-person assignment parity still requires
independent replay. The bank's
probabilities are already H-GLOBAL-TEMP calibrated; another transform would
change the fixed bank and must be declared before fitting. The A sampler may
receive only its recipient's fine cell and fixed bank/channel parameters;
its public release contains only the sampled code, code-determined probability
and decision, and fixed/code-determined metadata. It must not transmit U0,
Ucal, fine cell, or per-person random coin.

The inherited qpc `FINE-TASK` fit uses a task-only objective but its historical
runner read FIT SEX and passed it through a sex-stratified table before the
task-only weights were applied. Describe it as a fixed inherited
task-objective bank. A newly fitted task-only control should enforce a
sex-free input contract. A's predecessor protected channel was never fitted
or freshly attacked on Adult.

## Exclusions and next admission checks

No scientific worker may open `jcv_v1/inputs/adult_jcv.npz`, mixed-role
`release.npz` or teacher/assignment archives, HEAD_VALIDATION labels, or old
assessment person-level arrays. The only prior mixed-array deserialization
was the disclosed isolated mechanical export, already independently admitted.
If any required permitted field is absent, use that same isolated frozen-ID
custodian route and disclose transient mixed-member reads; no all-role loader
followed by masking is an acceptable substitute.

Before actual D/A fits: R00 must seal source and role hashes; a worker must
verify the exact file, member and ordered-person contracts; D must replay its
latent interface on permitted roles; A must close independent bank parity and
fine-cell capacity checks; and the shared queue must admit numerical work.
No HEAD_VALIDATION or assessment labels become available through these steps.
