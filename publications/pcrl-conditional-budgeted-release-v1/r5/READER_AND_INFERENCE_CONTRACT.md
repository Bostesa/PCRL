# Fresh complete-wire readers and exact conditional inference

This is prospective exploratory development on reused Adult people. Public
parameters do not supply an attacker with private per-person provider inputs.
Only ATTACK_FIT_NEW SEX trains/orients readers. INNER SEX remains embargoed
until every release, every reader and the endpoint family have been frozen.

## Readers and the actual wire

Per distinct arm/source seed, fit the source architecture slate in each of
recipient 1, recipient 2 and their aligned pair: LR_C1, MLP_64x32,
HGB_15x200, DA_MLP_64x32, CELL_A1, CODE_BIT_LR_C1, PROB_LR_C1 and
PROB_HGB_15x200. This is 24 reader fits plus six real LR probe fits. Local
readers transported to the pair are exact zero-fit aliases. Alias savings
require an independently verified exact output law/full wire identity.

The original global code ID determines its frozen probability vector and
first-index decision. Encode it with the full **public frozen bank** vocabulary
and its exact code bits. This prospective encoding differs from the
predecessor's ATT-only one-hot vocabulary. The public bank defines all possible
IDs before fitting; no reporting label supplies vocabulary. Both candidates
and controls use the same encoding rule. Probabilities, decisions and every
original metadata field are accounted for. Constant schema/recipient/model/
source-binding/bank-hash/bit-length fields are bound literally in receipts;
constant strings cannot create person-varying AUC within one arm/seed.
Neither private source codes, action positions, coins, provider cache keys nor
labels enter a candidate wire or reader. The planted probe is a separate
engineering copy that deliberately appends the true label.

DA features contain log(max(actual frozen probability, 1e-12)) and the
log output mass under a uniform prior over every possible original P0 input
code pair, using the publicly known kernel. This is a data-free computation
aid; it does not claim that the provider inputs have a uniform distribution.
The pair also receives the original code-bit products and joint output mass.

LR and categorical readers train with exact law weights aggregated by
emitted wire and fitting SEX. StandardScaler uses those same weights. HGB
trains on positive per-person/action rows with exact law weights. This avoids
collapsing a 14-code income bank to at most 28 wire-label cells under the
inherited min_samples_leaf=20. HGB's physical row count therefore means
positive action rows, not independent people; the rule is identical across
arms and is disclosed. HGB uses learning rate .05, 15 leaves, 200 iterations,
no early stopping. LR uses C=1 and max_iter=500.

The actual local sklearn 1.8.0 MLP accepts sample weights (added in 1.7,
verified by reading its installed fit signature before science). Both MLPs
therefore use exact law weights aggregated by emitted wire and fitting SEX,
with a weighted StandardScaler. Source MLP settings are 64x32 ReLU/Adam,
alpha .0001, 300 iterations, no early stopping, model seeds 310700 plus
source seed. Their stochastic minibatches contain aggregated wire/label
cells rather than independent people; actual law weights enter the fit,
and this identical rule applies to all arms. This is an empirical reader,
not an exact optimization oracle. No sampled or mean probability feature is
used. An earlier unsealed draft incorrectly assumed older sklearn lacked
weight support; source inspection corrected that before any reader fit.
The four-draw data-free sampler fixture remains an engineering check and is
not the registered scientific reader training method.

The source SHA-based ATT split retains 3046 fitting groups and 1015 validation
groups. Each reader orientation is chosen only on its exact-law validation
AUC; .5 ties choose the positive orientation. No orientation is chosen on
INNER. The reporting estimand is the maximum over this fixed, oriented slate
per seed/view; this is reader capability reporting, not release selection.
The pair includes every oriented local reader, so its envelope cannot be below
either local envelope. All model parameters, orientations, score tables and
role pairings are frozen before reporting labels open.

## Engineering probes

Per view, fit an LR against one group-wise shuffled fitting label vector and
score against a separately shuffled INNER vector, with the inherited fixed
salts and equal-size group buckets. Its oriented null diagnostic must be at
most .60. A separate true-label planted-bit LR must recover at least .99.
Constant-reader AUC must equal .5. Failed probes block dependent recovery
claims. The true-INNER AUC of the shuffled-fit model is not a valid null when
the original wire itself leaks SEX and is not the mandatory gate.

## Integration and inference

For each fixed reader, evaluate its public finite-bank score table on every
person's actual emitted action. Aggregate exactly equal scores, including
different codes that the reader maps to identical floating-point scores.
For ascending score s with positive mass P_s and negative mass N_s, AUC is
sum_s P_s (sum_{t<s} N_t + .5 N_s)/(sum_s P_s sum_s N_s).
This integrates a single independently cached query per distinct person.
Duplicates of one person share a coin; their group must have constant SEX
and the same actual law. Thus a positive and negative comparison always
uses distinct groups. This removes release-draw Monte Carlo error; it does
not remove model/reader fitting uncertainty, reuse bias or population error.

Use 2048 multinomial duplicate-group bootstrap samples, seed 2026101105,
shared across all arms, both model seeds and utility/recovery endpoints.
Duplicates retain all their rows, multiplied by their shared group count.
Recompute the fixed-oriented reader max inside each replicate, then take
the equal paired-seed average when the endpoint calls for it. A model seed is
not a separately sampled population. Utility uses the same group counts.

The coordinator freezes K=552 prospective primary slots, including NA
dependencies rather than dropping them after outcomes. For each slot use
point +/- bootstrap SD(ddof=1) times t.ppf(1-.05/(2K), df=2233).
This approximate conditional-development construction makes no exact
simultaneous coverage or post-selection confirmation claim. Reader fitting
is fixed during the primary intervals. Absolute AUCs and all reader rows are
descriptive; registered allocation, budget, support and competitive contrasts
retain their prospective primary family.

Upper-bound cap equality passes. A lower bound strictly above a cap is a
demonstrated violation; a crossing interval is unresolved. Benefit lower bound
at least .020 passes, and upper bound below .020 is a demonstrated shortfall
of that target. Utility point admission is distinct from interval passage.
Registered allocation, budget and support effect contrasts use a strict
positive-effect convention: lower bound >0 passes; equality at zero is
unresolved. Structural identity is reported separately when independently
proved, without relabeling that statistical boundary.

## Resource forecast and artifacts

Each distinct arm/seed is one compound queue unit, at most 300 reserved CPU
seconds, 600 wall seconds and 2 GiB RSS before smoke timing revisions made
prospectively. Up to 34 registered arm/seed units imply 1020 total fits;
untriggered/ineligible arms are recorded without claiming defeat. Scientific
model fits, feature preprocessing, full-bank public-prior enumeration,
predictions and probes all use the global one-thread numerical queue.
Bootstrap inference is a separate queued unit with a memory-limited batch.
Private fitted models, law arrays, labels, score tables and group counts stay
outside the public evidence tree. Public receipts contain settings, model/
input/table hashes, validation aggregates, fit counts and numerical outcomes.
