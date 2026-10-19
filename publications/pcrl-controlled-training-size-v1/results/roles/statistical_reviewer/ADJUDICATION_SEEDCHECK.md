# Adjudication A-SC1: scope of the effective-reader-seed check (RULES.md section 9, item 2)

Statistical reviewer. All times are `date -u` readings:
- the request arrived at about 01:33Z (the coordinator read 01:33:28Z right after sending it);
- I started at 01:34:35Z;
- this text was begun at 01:35:21Z;
- the completion time is at the end of this file.

## Disclosure (read first)

**When this ruling was made.** After the outcome of the seed check was known, and before any reader-stability label was computed
by the coordinator.
- The seed-check record is `results/pcrl_controlled_training_size_v1/readers/READER_SEED_CHECK.json` (utc 2026-10-19T01:31:29Z,
  unit TSZ_SEEDCHECK_COND_V1).

**I read no reader outcome.** I did not open any of these:
- `roles/verifier/CHECKS.json`;
- `roles/verifier/records/`;
- the reader grids under `.private/pcrl_controlled_training_size_v1/runs/GRID/`;
- any AUC, pair benefit, recipient difference or label.

My only inputs were READER_SEED_CHECK.json (per-recipe equality counts and the 16 equal checks), the coordinator's two messages
and my registered RULES.md / RULES.json / tsz/rules.py. This ruling changes none of the three.
- tsz/rules.py at HEAD (937257ecc) differs from my completion record by one line. That line is the coordinator's adjudicated
  change A2 / ENG1-05, which aligns IR4 with RULES.md 3.6. `reader_seed_check` is untouched by it.

**What the coordinator reports and I have not seen:**
- The independent verifier computed and stored its own reader-stability labels (01:32:47-01:33:09Z, per the coordinator). Nobody
  has read them.
- The verifier independently found the same 16 equalities and no others.

**A timing point to reconcile (not a finding).** The coordinator says the verifier's labels were computed "before the seed-check
record existed". The seed-check record carries utc 01:31:29Z, which is earlier than that window. The coordinator should reconcile
the two times in the log.

## Facts used

**From READER_SEED_CHECK.json:**
- 2,688 checks, 168 per recipe: 14 bank members x 3 views x 2 versions x 2 repetitions.
- Reader seeds: 2026101705, 2026101706, 2026101715, 2026101716. They are distinct and none is an OHN1 / CTS1 grid seed
  (MODEL_SEED_BASE + 300 + 10 v + k = 2026101505-2026101524).
- MLP64, MLP6432, OUT_MLP and LOG_MLP: equal under k0 / k1 in 0 of 168 checks each.
- The 10 seed-invariant recipes: equal in 168 of 168 checks each. No NONDETERMINISTIC_RECIPE flag.
- ET5 and ET25: equal in exactly 8 checks each, 16 in total. Every one is the INCOME view of B2TWIN or RETIRED_B3CTL (both
  versions, both repetitions).
- The registered function `tsz.rules.reader_seed_check` returned READER_SEEDS_NOT_EFFECTIVE.

**From the coordinator, not re-derived by me:**
- These two reference income wires take only 6 distinct probability vectors on MODEL_SELECT (a 6-level income code).
- Every other wire has 7,335-7,500 distinct income vectors and 7,411-7,500 distinct occupation vectors.

**Implied by the record itself:**
- In each of the 8 cells (wire, view, version, repetition) where ET5 / ET25 are equal, all four MLP recipes differ under k0 / k1.
  The MLPs are fitted in the same cell from the same seeds, so the reader seed demonstrably reaches those cells.
- ET5 / ET25 differ under k0 / k1 on the pair and occupation views of the same two references, and on all 14 x 3 views of every
  other member. Their equal count is exactly those 8 income-view cells.

## Classification: an AMENDMENT, not a clarification

The registered sentence covers "every release wire checked (at least W100_v and every selected release)", and continues: "Any
equality -> READER_SEEDS_NOT_EFFECTIVE".
- Read literally over the wires that were actually checked, the grid is NOT_EFFECTIVE, and the registered code says so.
- Limiting invalidation to the registered minimum set, or to unexplained equalities, changes that registered outcome. This is
  therefore an amendment of the check's scope, made after the check's outcome.
- It is recorded as amendment A-SC1.

## Ruling

### 1. Reading (a) is rejected

Reading (a) applies the literal text to every checked wire, which leaves the grid invalid until "an engineering repair".

**The check's purpose.** It was registered as a validity check of the grid machinery: does each reader seed reach the
seed-dependent fits? Its remedy was "an engineering repair", which presupposes a code defect.

**There is no code defect.**
- The seed reaches every one of the 8 affected cells: the MLPs differ there.
- ET5 / ET25 respond to the seed on every other view of the same two wires.
- ExtraTrees grown on an input with 6 distinct points, each with enough support, partition those points completely. Their
  predictions then equal the per-point frequencies on FIT, whatever random thresholds were drawn. They are seed-invariant by
  construction, as a property of the input wire.

**No engineering repair exists.** Every conceivable "repair" changes a registered, frozen element:
- dropping ET5 / ET25 for every release changes the locked 16-recipe bank;
- dropping or replacing the income views of B2TWIN / RETIRED_B3CTL changes the required comparator set (frozen since OHN1);
- perturbing the released wires changes the releases;
- changing the ExtraTrees settings changes the bank.

Each of these departs further from the registration than a scope amendment does. The literal reading would also leave no
stability label at all, which is a non-result, not a conservative result.

### 2. Amended validity criterion

This replaces the scope of item 2 only. Items 1, 3 and 4 are unchanged. The grid is READER_SEEDS_EFFECTIVE iff all of:
- **(i)** Item 1 holds: 4 distinct, recorded seeds, none reused from OHN1 / CTS1.
- **(ii)** On the registered minimum set, every seed-dependent recipe (ET5, ET25, MLP64, MLP6432, OUT_MLP, LOG_MLP) differs
  under k0 / k1 in every view, version and repetition. The minimum set is W100_v (bank member W) and every selected release
  (LOC_SEL, PAIR_SEL).
- **(iii)** Every equality of a seed-dependent recipe on any other checked wire is EXPLAINED. Explained means all three of:
  - (iii-a) in the same (wire, view, version, repetition), at least one other seed-dependent recipe differs under k0 / k1;
  - (iii-b) the same recipe differs under k0 / k1 on at least one other view of the same wire, in the same version and
    repetition;
  - (iii-c) the number of distinct input vectors of that view on MODEL_SELECT is recorded label-free beside the check.

  One unexplained equality keeps the grid NOT_EFFECTIVE, and the original remedy (an engineering repair) applies.
- **(iv)** Seed-invariant recipes follow item 3 unchanged: an inequality is flagged NONDETERMINISTIC_RECIPE and is not fatal.

### 3. Application to the recorded check

| Condition | Result |
|---|---|
| (i) | Holds. |
| (ii) | Holds: none of the 16 equalities involves W, LOC_SEL or PAIR_SEL. |
| (iii-a) | Holds: the MLPs show 0 equalities in 168 checks. |
| (iii-b) | Holds: the ET equalities are confined to the income view. |
| (iii-c) | Holds once the coordinator's counts are recorded: 6 distinct income vectors for B2TWIN and for RETIRED_B3CTL, against 7,335 or more elsewhere. The coordinator must add these counts, label-free, to the seed-check record or to a companion record before any label is computed. |
| (iv) | Holds: no flag. |

**Decision: READER_SEEDS_EFFECTIVE under A-SC1.**
- The 16 equal checks are recorded as the property `SEED_INVARIANT_BY_INPUT`: ET5 and ET25 x income view x {B2TWIN,
  RETIRED_B3CTL} x {v0, v1} x {R1, R2}.
- Both outcomes stay on record side by side: `literal_registered: READER_SEEDS_NOT_EFFECTIVE` and
  `amended_A-SC1: READER_SEEDS_EFFECTIVE`.
- Downstream code takes grid validity from this adjudication, not from the label of the unchanged function
  `tsz.rules.reader_seed_check`.

### 4. What changes for labels: no formula changes

- Every stability, target and recipient computation runs exactly as registered in section 9. That includes the per-pair
  `k_differs`, n_eff, the distinct-cell spread and the always-attached reader-seed qualifier.
- "Genuinely effective reader seeds" for a single comparison still requires that comparison's own qualifier to be
  `READER_SEEDS_EFFECTIVE_ALL_PAIRS`.
- New recorded flag `COARSE_REFERENCE_INCOME_WIRE`. It never gates and never changes a label. Attach it to:
  - every income-view recipient-gate result against B2TWIN or RETIRED_B3CTL;
  - every pair-view comparison against B2TWIN or RETIRED_B3CTL in which the comparator's pair winner in some cell is an
    income-view (IGNORE_occupation) reader.
- Correction to the request's wording: pair-view labels against these two references are not unaffected by construction. A pair
  winner can be an income-view reader, and its score can then be seed-invariant in that cell. Any duplicated cells this produces
  are already handled by the registered n_eff rule.
- The verifier should state whether its stored labels assumed the literal validity status or the amended one. This ruling changes
  none of its computed numbers.

### 5. Disclosure sentence (required wherever reader labels are reported)

> "The registered effective-reader-seed check, read literally over all 14 checked bank members, returned
> READER_SEEDS_NOT_EFFECTIVE: ExtraTrees readers (ET5, ET25) are seed-invariant on the income views of B2TWIN and RETIRED_B3CTL,
> which take 6 distinct values. On the registered minimum set (W100_v and every selected release) there was no equality.
> Amendment A-SC1, made after this check's outcome and before any reader-stability label was computed or read, limited grid
> invalidation to unexplained equalities. The grid passed the amended check. No stability formula changed."

### 6. What this ruling does not do

- It changes no threshold, z, cell set, recipe bank, comparator set, trigger, selection rule or claim rule. RULES.md, RULES.json
  and tsz/rules.py are unchanged.
- It does not depend on any reader outcome: none was read.
- If any reader-stability label had been computed and read before this ruling, the ruling would count as post-outcome for that
  label, and that would have to be disclosed. The coordinator states that none has been.

Completed 2026-10-19T01:36:00Z (`date -u`). Correction disclosed: the first save of this file (written at about 01:35-01:36Z)
carried a completion time of "01:37:35Z" that I had not read from the clock. That was an estimated time, which is a recorded error
class. I replaced it with the `date -u` reading above, taken after that first save.
