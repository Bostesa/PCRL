# R2 method and implementation handoff

This is a development comparison on reused Adult roles. R2 fitted seven
frozen families on each of two source seeds. The exact P0 incumbent was an
identity control, not a new fit. All fourteen R2 queue units finished. The
scientific lock is `237feaf85dd19c497c5f377ab0f7fffab72bc41cb8d2c277b21976d6c0cd4ae6`;
the original queue admission seal remains separate. The final R2 code/source
index is `PROTECTION_SOURCE_INDEX_V2.json` (SHA-256
`967fd0d05b66c3b3db4f62d2e7f49a877ed232bab25a7076699a5ba80a4b6af9`).
The fourteen config/spec hashes are in `PROTECTION_SPEC_INDEX.json` (SHA-256
`d0c7699831d44d91f7020cc30de629ee94473d0718359c4baab51b7a2aa4df9c`).

## Methods and actual fitting

All families take only both private P0 original-ID codes and fixed decoded
probabilities as provider inputs. Each recipient's same-class support has at
most four original P0 codes, making at most sixteen released pairs. The
released probabilities, decisions, code IDs and fixed metadata are all
functions of the selected public codes. SEX and task labels are fit-only;
deployment needs neither. One fixed release per group is modeled, with three
separately cached development realizations for stochastic arms.

| Arm | Operation | Two-seed FIT outcome |
| --- | --- | --- |
| O_J | Joint 480-parameter gate, inherited exact finite-code Bayes objective | Selected nonidentity twice |
| V_J | Same joint gate, cross-fitted actual-wire reader CE objective | Selected nonidentity twice |
| V_P | Marginals of shared gate, independent recipient coins | Selected nonidentity twice |
| V_D | Truly hard deterministic gate with retained bias trials | Exact P0 identity selected twice |
| V_L | Recipient-local 360-parameter factors, independent coins | Selected nonidentity twice |
| V_S12 | Frozen recipient-1 first release, conditional recipient-2 second | Selected nonidentity twice |
| V_S21 | Frozen recipient-2 first release, conditional recipient-1 second | Selected nonidentity twice |

O_J's inherited finite-code objective already aggregates the actual global
public code IDs. V changes the loss and reader-generalization package; it is
not a repair of a mean-only old objective. All arms used two starts, three
retained-reader rounds, source-matched update allocation, FIT-only checkpoint
selection and a fixed geometric feasibility backoff including exact P0.
Central and sequential arms used 600 optimizer steps per completed unit;
V_L used 300 steps updating both factors, or 600 factor steps. All completed
units had 600 factor steps. Product/local samplers used independent
recipient-domain coins. Sequential second stages conditioned on the actual
first output while freezing its first-stage law.

All fourteen selected laws met the frozen FIT task-cost proxy and all retained
local-reader guards; for V_D this is the exact P0 identity. R3's separate
original INNER two-seed point gate reports 24/24 clauses for every arm. Those
point checks do not establish simultaneous interval passes or privacy gain.
R3's fixed FIT-reader nomination selected V_L before final attacks and states
that final attack results were unused. Its ranking is a development selection
on repeatedly exposed data.

## Exact R2 work count and custody

The fourteen R2 FIT units charged **855.589 CPU seconds** in the shared
queue. They performed **1,512 trained reader family-fold fits**, **756
constant-prior family-fold fits**, **7,800 optimizer steps**, and **8,400
factor steps**. They produced **126 role/realization law and release packages**
(three roles times three draws times fourteen units). Each unit receipt
reports policy forward calls, action positions, preprocessing calls, actual
reader fits, role reads, release rows, CPU, and every selected policy and
release hash. The public `FIT_UNIT_INDEX.json` (SHA-256
`f7618f1a8f9cd388594b28acd8279e0642a9019dd359b0b3399cc12d11bd0b64`)
binds all fourteen unit receipts and records the two D identity aliases.
Private policy, cached release, retained reader and checkpoint arrays remain
under `r2/.private/units/<UNIT_ID>/`; public files contain aggregate receipts
and hashes only.

R5's fourteen queued independent FIT audits report PASS with zero task and
reader receipt gap and no ATTACK_FIT_NEW or INNER_SELECTION labels read by R2.
R5's full campaign closeout, fresh R4 attacks and any deployment replay have
their own receipts. Do not treat this R2 handoff as those results.

## Runnable package and limits

`incumbent_view.r2_package.load_policy(directory,
expected_receipt_sha256=...)` checks the selected receipt, arrays, source
seed, bank and law family. Its `Policy.sample_codes` consumes original P0
codes with either no coin (identity/deterministic), one joint coin, or two
recipient-domain coins according to the package receipt. R1's raw83 adapter
routes through the frozen P0 provider and emits the source public wire; it
must be replayed for the nominated selected package before a deployment claim.
The role packages are finite-code score releases; they do not protect raw
feature sharing.

Adam, hard straight-through optimization, reader fitting and checkpoint
selection are approximate. The FIT utility cap is a mean empirical constraint
with a fixed numerical safety envelope, not a per-person or population
guarantee. The retained categorical/nonlinear readers and exact finite-code
Bayes objective cover their specified fitting-law views; fresh adaptive
recovery is measured separately by R4. The two seeds share people. No old
assessment or new confirmation population was opened in R2.
