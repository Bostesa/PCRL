# Release and attacker interface contracts

## Score track

The provider privately computes each task's frozen source score from the 83 allowed predictors. Each recipient observes its actual released probability vector, decision, output code identity (when present), and every flag, fallback value, length or metadata transmitted by the release. The coalition observes both aligned recipient views. Public source models, channel matrices, codebooks and decoder tables are known. The source score, input bin and fine token are **not** transmitted unless a candidate explicitly discloses them, in which case attacks must receive them and the privacy benefit may disappear. If a sampled code is released, the receiver gets that code's probability, not the conditional mean over draws. Sampling is one fixed draw per person for the default contract, with seeds independent of SEX and task labels. A claimed shared-randomness scheme must specify all inputs each sampler sees and which random bits each recipient observes.

Team A belongs only to this track. A score-only adaptation of B would require a new registered identity and a separate matched comparison; it is not in the initial candidate registry.

## Feature track

Each recipient sees its actual purpose features (including exact raw selected values for C, stochastic latent draws for B, or repaired latent code/features for D), a head probability and decision computed solely from those purpose features, and every transmitted mask, code, flag, length and fallback. The coalition sees the aligned union of both views. Public encoders, transformations, heads and masks are known. The provider alone reads the 83 permitted input columns; no deployment path reads SEX, task labels, or unavailable other-recipient private inputs. If a candidate needs other-party inputs at inference, its locality premise fails unless the matched arms receive the same contract.

For C, a deterministic head of a public fixed mask's raw values carries no additional person-specific information beyond those values to an ideal reader. Privacy evaluation must therefore attack the raw union, not only head outputs. For D, an invertible transformation alone cannot erase information. For B, the released head must use the sampled latent actually transmitted.

## Attack views and scoring

Fit fresh SEX readers on ATTACK_FIT_NEW, evaluate on INNER_SELECTION only after access gates pass. Include linear, nonlinear, code/mask/flag/fallback readers, defense-aware readers and coalition readers allowed to ignore either recipient. Attack features are derived from the full public per-person view. The strongest eligible reader is selected by a frozen inner protocol; do not select a weak reader after outcomes. Constant, shuffled-SEX and planted-leak engineering controls must pass on a separate permitted copy. No old assessment arrays or HEAD_VALIDATION labels are touched. Finite attacker AUC is an empirical recovery result, not mutual information, demographic parity, differential privacy or a population guarantee.
