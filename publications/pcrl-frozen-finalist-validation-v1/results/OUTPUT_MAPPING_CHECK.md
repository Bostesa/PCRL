# Output-mapping check: frozen finalist `B3CTL_UNCOUPLED_MIX50_C6A`

**Data.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, in the development roles. The service predicts income class (2) and occupation group (5) from AGEP, WKHP, COW, SCHL, MAR and POBP. Each recipient receives its own float64 probability vector and argmax decision. SEX is the recorded column that evaluation readers try to infer.

**How this was done.** Label-free: no SEX or task label was read. All numbers use the exact transmitted float64 bytes from the evaluated batch outputs, with no rounding. They come from:
- `OUTPUT_MAPPING_RESULTS.json` (unit FFV_MAPPING_V2);
- `MAPPING_CONFLICTS.json` (unit FFV_MAPPING_CONFLICTS_V2, `ffv/mapping_conflicts.py`) for the exact union count of shared vectors and the identifier-free conflict details. That unit replaces the session-scratchpad script of unit FFV_MAPPING_CONFLICTS_V1, which printed the same details. Domain-wide statements (Part 2) are kept separate from empirical counts (Part 1).

## What the source said, and what is checked

The source QUICKSTART said the income output "is not computed from the occupation output. On the observed people it is nevertheless a function of the occupation vector, because that vector is essentially one-to-one in the inputs."

The source RESEARCH_DECISION said the 6-level income output adds no distinct split beyond the occupation output on any observed role.

This check audits both statements without changing the candidate.

## Part 1: empirical facts about the observed records

### Per role (both seeds)

Columns: records; distinct encoded permitted-input vectors; distinct occupation vectors; distinct linked pairs; occupation vectors carrying more than one income code; occupation vectors shared by distinct inputs; share of occupation vectors seen once.

| Role | Records | Inputs | Occ. vectors s0 / s1 | Pairs s0 / s1 | >1 income code | Shared by distinct inputs s0 / s1 | Singleton share |
|---|---|---|---|---|---|---|---|
| MODEL_FIT | 20,000 | 17,744 (raw 17,753) | 17,743 / 17,737 | 17,743 / 17,737 | 0 / 0 | 1 / 7 | 0.928 |
| CALIBRATION | 2,982 | 2,869 | 2,869 / 2,869 | 2,869 / 2,869 | 0 / 0 | 0 / 0 | 0.968 |
| MODEL_SELECT | 8,001 | 7,500 | 7,500 / 7,498 | 7,500 / 7,498 | 0 / 0 | 0 / 2 | 0.956 |
| ATTACK_FIT | 8,001 | 7,531 | 7,530 / 7,526 | 7,530 / 7,526 | 0 / 0 | 1 / 4 | 0.954 |
| ATTACK_VALIDATION | 3,000 | 2,902 | 2,902 / 2,902 | 2,902 / 2,902 | 0 / 0 | 0 / 0 | 0.971 |
| DEV_REPORT | 5,943 | 5,619 (raw 5,620) | 5,619 / 5,619 | 5,619 / 5,619 | 0 / 0 | 0 / 0 | 0.960 |

- Raw six-field vectors are admitted only for MODEL_FIT and DEV_REPORT. A few distinct raw vectors share an encoding through the OTHER category columns.
- Every role has exactly 6 distinct income vectors on both seeds.

### Union of the six roles (deduplicated)

| | Seed 0 | Seed 1 |
|---|---|---|
| Records | 47,927 | 47,927 |
| Distinct encoded inputs | 39,585 | 39,585 |
| Distinct occupation vectors | 39,579 | 39,536 |
| Distinct linked pairs | 39,579 | **39,538** |
| Occupation vectors with more than one income code | 0 | **2** |
| Occupation vectors seen in two or more roles | 3,186 | 3,211 |
| Encoded inputs seen in two or more roles | 3,182 | 3,182 |
| Encoded inputs with more than one output pair across roles | 0 | 0 |

**Within each role.** On both seeds and all six roles, the observed income code is a function of the observed occupation vector. This reproduces the source observation on DEV_REPORT, ATTACK_VALIDATION and ATTACK_FIT, and extends it to MODEL_FIT, CALIBRATION and MODEL_SELECT.

**Across roles, seed 1.** The relation fails. Two occupation vectors each accompany two different income codes:
- one is shared by a DEV_REPORT person and a MODEL_FIT person;
- the other by an ATTACK_FIT person and a MODEL_SELECT person.

In both pairs the encoded inputs differ only in POBP. The occupation vectors are bitwise identical, but the income levels are 2 and 3, which lie on opposite sides of the income decision threshold. These are real observed conflicts, not a last-bit artifact: identical encoded inputs always received identical outputs across roles.

**The mapping is deterministic across roles.** The same encoded input never received two different output pairs in different roles.

### Why the observed functional relation is weak evidence

**Most vectors occur once.** Within a role, 92.8–97.1% of distinct occupation vectors occur exactly once. A relation "income code = f(occupation vector)" that holds on a set where nearly every vector occurs once is close to vacuous: any assignment of codes to once-seen vectors is a function.

**Only repeated vectors can test it.** Most repeated vectors are repeated identical inputs, which receive identical outputs trivially. The informative cases are distinct inputs that share an occupation vector:
- seed 0: 5 such vectors in the union (distinct inputs exceed distinct vectors by 6; at most 3 inputs share one vector), with no conflict;
- seed 1: 46 such vectors in the union (excess 49; at most 4 inputs share one vector), of which 2 carry different income codes (`MAPPING_CONFLICTS.json`).

**Future records are not covered.** Whether the relation holds for future records depends on how often distinct inputs fall in the same occupation routing cell. That is a property of the frozen trees over the input domain (Part 2), not of these 47,927 people.

## Part 2: domain-wide structure of the frozen model

This part is separate from the empirical counts, and it is not a population theorem.

**The valid input domain.**
- AGEP 25–64; WKHP 1–99; COW 1–8; MAR 1–5. Every valid COW and MAR value is in the encoder vocabulary.
- SCHL 01–24. The codes 02, 03 and 05 are outside the vocabulary and encode as `SCHL=__OTHER__`.
- POBP: the 105 vocabulary codes, plus valid codes outside the vocabulary (for example U.S. state FIPS 030) that encode as `POBP=__OTHER__`.

Counting distinct encodings, the domain has 369,388,800 encoded inputs.

**Routing cells.** The frozen oblivious trees split on exact borders: depth-4 trees, 587 income and 800 occupation trees on seed 0, and 529 and 799 on seed 1. A routing cell is a set of inputs that every tree of a teacher sends to the same leaves. For each raw variable, the equivalence classes of its valid values are exact; so are the cell counts.

Classes per variable (seed 0 / seed 1):

| Variable | Distinct encodings | Occupation classes | Income classes | Joint classes |
|---|---|---|---|---|
| AGEP | 40 | 40 / 40 | 39 / 36 | 40 / 40 |
| WKHP | 99 | 73 / 72 | 64 / 60 | 81 / 80 |
| COW | 8 | 8 / 8 | 8 / 8 | 8 / 8 |
| SCHL | 22 | 22 / 22 | 22 / 22 | 22 / 22 |
| MAR | 5 | 5 / 5 | 5 / 5 | 5 / 5 |
| POBP | 106 | 103 / 101 | 83 / 75 | 105 / 103 |

Domain totals:

| | Seed 0 | Seed 1 |
|---|---|---|
| Encoded inputs | 369,388,800 | 369,388,800 |
| Occupation routing cells | 264,668,800 | 255,974,400 |
| Income routing cells | 182,307,840 | 142,560,000 |
| Joint (occupation × income) cells | 299,376,000 | 290,048,000 |
| Share of the domain in singleton occupation cells | 0.573 | 0.524 |

1. **The occupation output is not one-to-one in the inputs.** There are fewer occupation routing cells than encoded inputs. About 43% (seed 0) and 48% (seed 1) of valid encodings share their occupation cell with at least one other valid encoding, and therefore receive a bitwise-identical occupation vector.
   - Invented examples, confirmed through the frozen package: on seed 0, WKHP 3 and WKHP 4 give the same occupation vector with every other field fixed, and so do POBP 046 and POBP 217.
   - "Essentially one-to-one" describes the observed records, where few people share a cell. It does not describe the model.
2. **Over the valid domain, the income code is not a function of the occupation vector.** The joint cells outnumber the occupation cells, so income routing splits occupation cells. A bounded search found constructive counterexamples:
   - 9,000 (seed 0) and 8,400 (seed 1) invented input pairs were tested, each pair inside one occupation cell but in different income cells, built on 600 random invented valid base rows;
   - 689 (seed 0) and 1,610 (seed 1) pairs have bitwise-identical occupation vectors and different income codes;
   - they arise through WKHP and POBP;
   - some flip the income decision. On seed 0: AGEP 63, COW 2, MAR 1, POBP 109, SCHL 15 with WKHP 72 against WKHP 74 gives the same occupation vector, but income level 2 against level 3 (P(income = 1) 0.431 against 0.551).

   Every example is fully invented and was confirmed through the frozen FINALIST deployment package.

## Part 3: pair-trained reconstruction with occupation-only evaluation input (diagnostic; pair access)

**What it is.** An HGB classifier with the HGB15 reader settings, fitted on ATTACK_FIT linked pairs, predicts the finalist's income level from its occupation output. It uses no SEX labels. It is not an ordinary occupation-only adversary: it needs the pair-access auxiliary sample.

| Role | Level accuracy (seed 0 / seed 1) | Majority-level accuracy |
|---|---|---|
| ATTACK_FIT (in sample) | 0.753 / 0.769 | — |
| ATTACK_VALIDATION | 0.628 / 0.630 | 0.211 |
| DEV_REPORT | 0.625 / 0.640 | 0.214 |

Only about 10% of ATTACK_VALIDATION and DEV_REPORT occupation vectors occur exactly in ATTACK_FIT, so an exact lookup cannot reconstruct the code for most people. A finite-sample learner recovers the income level from the occupation output only partially. An information-level redundancy on the observed records does not translate into reader-extractable redundancy.

## Floating-point characteristics (characterized, not repaired)

**Batch-size last bit.** The package's per-level income probability depends on the request batch size at the last bit, for 2 of the 6 levels on each seed.
- On 200 DEV_REPORT rows requested one at a time or in 8-row batches, 60 (seed 0) and 47 (seed 1) rows differ bitwise from the evaluated batch outputs, by at most 1.1e-16.
- The affected levels are 2 and 4 (seed 0) and 3 and 4 (seed 1). Decisions are unchanged, and the occupation output is unaffected (`DEPLOYMENT_CHECKS.json`; confirmed by the claims reviewer's spot check).
- H_LJ was tested on seed 0 only and was unaffected. TASKONLY uses the same family-B arithmetic and was not tested; it probably shares the dependence.
- The counts above use the evaluated batch bytes. An exact-byte lookup keyed on batch outputs would see different keys for small-batch requests.

**Threshold ties** (source deviation D3). These are people exactly on a frozen label-free MODEL_FIT threshold:
- seed 0: 2 MODEL_FIT people; neither changes level under an exactly rounded or reversed summation;
- seed 1: 4 MODEL_FIT people; 2 move to the adjacent level under either alternative summation (the package value is 2.2e-16 and 4.4e-16 below the exact sum).

In every other role, the closest person is 4.5e-6 from a threshold (ATTACK_FIT, seed 0). Neither seed-1 move crosses the income decision: the levels change 0→1 and 1→2. Thresholds are unchanged: a midpoint-threshold variant would be a different release, needing its own evaluation.

## Consequences for the source wording

1. **Reproduced, within each role.** The observed income codes are a function of the observed occupation vectors within each evaluated role, so the source's statement about DEV_REPORT, ATTACK_VALIDATION and ATTACK_FIT holds as an empirical fact about those people.
2. **Not supported:**
   - "essentially one-to-one in the inputs": false for the model, which is many-to-one on the domain, and on observed records on seed 1;
   - any reading that the income output carries no information beyond the occupation output for future records. Over the valid domain the income code is not determined by the occupation vector, and on seed 1 even the observed union contains two counterexamples.
3. **The source's pair benefit over H_LJ.** H_LJ shares the occupation output. The benefit stays a reader-relative difference on these records. Part 2 removes the basis for describing it as "no added information" beyond the observed roles; Part 1 shows the observed relation is close to vacuous.
4. **No population-optimal privacy statement** is made from any empirical lookup table. Absence of observed conflicts within a role is a fact about those records only.
