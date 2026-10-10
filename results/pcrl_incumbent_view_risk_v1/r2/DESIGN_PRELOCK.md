# R2 policy and reader design for prefit seal

This document is a design proposal. No new FIT, ATT or INNER arrays have been
opened and no fixture algorithm has run. R0 owns its effective admission.

## Sources and scope

Immutable primary source: `dda8e450cdd398ec1153c69ae1a9dd732a35b955`.
P0 is `U|JOINT|i8o64|l0.1|H-CLASS-TEMP`, with original coarse IDs and
unchanged source wire. R1 reports table sizes 14/270 (seed 0), 13/299 (seed 1),
and source code lengths 4/9 bits. These are provisional until R1 admission.
Its public code, probability, decision, release/model/recipient identity,
source/bank hashes and code-length metadata are the complete reader view.
Constant metadata is enumerated and bound but cannot vary by person.

The old protected objective is precisely empirical finite-law
`G_pair + .5*(G_1+G_2)`, where each G sums the larger SEX-labelled probability
mass at every emitted global state. It is not AUC, CE, mutual information or
a population guarantee. O-J keeps this formula, on the new P0 support.

All policies consume only private original P0 IDs and their fixed decoded
vectors. Central features are the two vectors, their entropies and top-two
margins. Source teacher scores and fine-cell positions are unavailable to
the learned policy. Local factors use only their own vector/summaries/ID.
Support is R1's fixed identity plus at most three nearest same-class P0
states, original-ID tie breaking; sequential second support never changes.

## Readers and folds

Three group-disjoint folds are assigned by SHA-256 of the literal
`IVR1|FIT-crossfold|20261010|` plus decimal duplicate-group ID, modulo three.
Every group belongs to exactly one fold. A reader for fold k fits weighted
action counts on the other two folds and evaluates only k. All fitting and
validation are within OSF_DEFENSE_FIT. No ATT/INNER data participates in
learning, policy selection, guard fitting or reader selection.

For each round/start/view, fit (1) categorical likelihood with additive
one pseudo-observation of each SEX per state and unseen-state smoothed
training prior, and (2) a weighted nonlinear actual-wire reader with source
MLP 64/32 ReLU geometry. Nonlinear input is recipient one-hot global code(s),
fixed probability vectors, one-hot decisions, source wire code bits and all
nonconstant metadata. It never uses a code's arbitrary scalar numbering as
its only representation. Constant metadata has a byte-bound inventory.
Its float64 Adam has lr .001, weight penalty 1e-4, 50 fixed full-weighted
updates, no outcome-selected early stopping, one thread. Counts aggregate
all people/actions sharing the same actual emitted code tuple; the two SEX
counts are exact soft action weights. No sampled training view or averaged
task vector substitutes for actual actions. Retain all fitted weights,
count tables, fold assignments and out-of-fold action CE arrays privately.

The constant-prior reader is included in every view/round slate, fit on the
same fold-training people with one pseudo-observation per SEX. Clip reader
probabilities at 1e-12 only for scoring. Previous round reader checkpoints
remain in the slate. Views r1/r2/pair have their own readers; final attackers
are separate and do not feed back.

For each frozen reader j, `R_j(q)=mean_i sum_a q_ia CE_j(s_i,wire(a))` on
its held-out group-fold rows. Cross-fold losses are combined personwise.
V-J/V-P/V-D maximize `min_j R_pair,j(q)` over the retained pair slate,
including constants. This is an explicit nonsmooth max-min approximation,
not an exact global solve. For every retained local reader separately,
require `R_r,j(q)>=R_r,j(P0)-1e-10`. Never replace all guards by one best-reader
summary. Report strongest-reader CE gain over the constant-prior baseline.
O-J receives the same reader fitting schedule and all these local guards,
but minimizes its inherited G formula. Readers inform only O-J guards.
V-L maximizes each recipient's own min-reader CE independently and composes
independent factors; its input restriction is explicit.

## Utility, identity and numerical safety

Accuracy loss cap .002, NLL excess .001 and summed Brier excess .0005 are
P0-relative per recipient/seed on each group-held-out fold and all FIT rows.
Accuracy is checked from actual fixed action decisions, even though class
preservation makes it structural. Score expected per-action costs, never
the loss of the mean output. Calculate signed per-row excess against P0
before float64 `math.fsum` reduction; replay using `numpy.longdouble` where
available and report max disagreement. Bound engineering comparison error
at 1e-10; require nonidentity utility margins below cap minus 1e-10.
Identity has exact zero excess and remains admissible. No threshold is
relaxed by numerical safety, which is separate from statistical intervals.

Stochastic exploration starts at eta .05, with fixed bounded safety backoff
`.05 * .5**k` for k=0..40, then exact zero. Returned checkpoint feasibility
backoff starts at1, then `.5**k` for k=0..40, then exact zero. Initial .05 is
not a final mixture ceiling (R0 prefit ruling; no outcomes observed). Joint mixes identity with gate;
product/local mix each marginal independently. For a candidate checkpoint,
take the first guard-and-cost feasible value on this schedule; no eta grid
selected by privacy outcome. Each checkpoint is evaluated on all FIT folds
and all retained local readers. This conservative neighborhood is disclosed.
D never receives an identity random mixture: its forward release is hard
argmax of a genuinely trained straight-through deterministic gate, and it
has a separate stored P0 checkpoint. Deterministic checkpoints select only
when their exact one-hot laws meet the same guards/costs.

Sequential first stage has its own exact identity boundary and pre-frozen
local slate. Select first stage using its held-out first-view min CE and
first task costs, then freeze policy and its guards. Three stage-two rounds
refit actual pair and second local readers; backoff changes only conditional
second factor, never the first law or support. A compulsory new first guard
failure is a recorded dependency failure and may return separately archived
whole P0, not secretly change first release. Stage-two training conditions
on a sealed single cached first draw; exact law validation integrates all
possible first emitted codes, including unseen ones. Source public
probability embedding defines unseen-code behavior.

## Updates, starts and custody

Inherited policy optimizer: float64 Adam lr .01, rho100, gradient norm5,
dual/checkpoint cadence25, initial hidden scale .01, two starts
identity-biased(logit+4) and uniform-logit. Each central start has 300 updates
split three rounds of100; local and sequential factors have150 updates per
start split three rounds of50, total300 per arm/start. D temperature1→.5
and source accepted bias-pass candidates are retained. Readers refit at
round entry under the current feasible policy; exact P0 checkpoints are
archived before any fit. The retained reader slate used to select final
checkpoints is common within a start and evaluated on every saved candidate;
cross-start selection replays both starts against the combined retained
slate, feasible first, best objective, start index, update, identity on tie.
Preserve initial, every25-update, round-boundary, terminal/stopped and
accepted deterministic-polish weights, including infeasible candidates.

One queue unit per arm/seed: P0 identity x2 plus seven fit arms x2 =16.
Per arm resource ceiling proposed 1,800 CPU seconds including all policy and
reader fits, one numerical thread, 3GiB RSS. Total fitted reservation25,200
CPU seconds (7h); P0 materialization/admission charged separately. Forecast
must be measured by a counted queued first real arm before further dispatch;
no unqueued role-sized smoke. At ceiling record INCOMPLETE and resumable
weights, not a complete identity arm. No sweep or post-ATT reselection.

## Prospectively authorized architecture matching

Inherited J/O/D gate12→16→16 has480 parameters; old P had552 and sequential
had552+16*Z_first. R0 approved shared480 central O-J/V-J/V-P/V-D, with P
formed as the product of the two marginals of the shared gate's masked joint
logits. This changes source P parameterization prospectively; J/O/D retain
source geometry, with policy inputs changed to fixed P0 decoded values.
Sequential uses two12→12 tanh→4 factors,208 each,416 total including frozen
stage one. Stage two replaces the first recipient's source decoded block
with the actual first emitted decoded vector and recomputes its summaries;
the other source block remains. Source IDs still select the frozen support.
This removes source trainable first-code embeddings and their capacity
imbalance. V-L retains4→16→4(148) and8→16→4(212),360 total. All are below the
480 capacity ceiling; exact equal architecture holds only central arms.
Three-round reader schedule is identical for O-J and V-J. The old G already
aggregates actual global IDs and therefore covers this complete finite-law
view; new CE training changes loss/generalization, not a mean-only bug.

## Frozen correctness tests (before fixture algorithms)

1. Exact original P0 IDs/probabilities/decisions/metadata and identity eta0.
2. Actual expected NLL/Brier equals enumerated realized distribution loss.
3. Mean trap: SEX0 actions p=.55/.85, SEX1 p=.60/.80, means .70; categorical
   actual-state reader distinguishes supports, mean-only cannot.
4. Code categorical encoding/public-ID mapping stable under ID permutation.
5. D one-hot; P/L independent two coins; joint correlation not laundered.
6. Sequential factorization, fixed first marginal, fixed second support;
   eta_second0 cannot alter first output, unseen-first code defined.
7. Duplicate groups never cross folds; reader train/evaluation rows disjoint.
8. Exact identity utility zero; cap-near summation longdouble replay;
   geometric safety backoff and deterministic rejection remain distinct.
9. Constant/null reader and one-class-weight handling; all local-reader
   guards checked separately, constant prior limits claimed CE gains.
10. All initial/infeasible/stopped/checkpoint arrays survive package roundtrip;
    FIT-only selection independent of final role labels or release draws.
