# Coordinate matching, affine recoverability and local gradients

**Teacher-only adaptation makes some removed structure easier to recover linearly, even with an independently verified closed raw-input route.** This distinguishes coordinate matching and decoder function class from information creation. qE=rawPCA16−E is not pure sensitive information, and these diagnostics cannot replace the actual source/transfer losses or SEX/RAC1P audits.

Every main I/W/final release gets separate affine decoders to original-standardized raw PCA16, retained E and qE; S conditions additionally report S/qS. Intercepts and coefficients use only representation-fitting rows, fixed float64 least squares with `rcond=1e-12`, and no outcome selection. Coefficients freeze before source-validation/development evaluation. Targets use original raw-PCA16 scales, never erased/residual variances. Ratios are undefined when fitting/evaluation target variance or evaluation fitting-prior MSE is at most `1e-12`.

[MECHANISM.csv](MECHANISM.csv) retains mean error, fitting-prior error, target variances, ratios, ranks and spectra for every split/seed; [MECHANISM_COORDINATES.csv](MECHANISM_COORDINATES.csv) retains every coordinate. [DIRECT_MATCHING.csv](DIRECT_MATCHING.csv) records direct coordinate error separately. Full spectra matter because float32 teacher/mapper output can leave tiny directions that count at the fixed numerical rank tolerance; rank alone is not an information or privacy measure.

## Stage affine recovery

Cells are mean per-seed **development MSE / fitting-prior MSE ratios**: both errors are evaluated on development rows, using a prior learned only on fitting rows. Smaller values indicate better affine prediction of that target. F/K share the exact initial parameters and teacher starting function; their rows differ after training. These are unweighted, label-free geometry diagnostics; PWGTP task/audit sensitivity is shown separately in [TABLE.md](TABLE.md).

| Release | Raw PCA16 | Retained E | Removed qE |
| --- | ---: | ---: | ---: |
| Direct E teacher, historical | .539961 | ~0 | .999196 |
| E I, shared F/K | .540001 | ~0 | .999288 |
| E F W | .431028 | .000854 | .795936 |
| E K W | .440352 | .000640 | .813107 |
| E F C | .437414 | .001091 | .807909 |
| E F D | .436092 | .001554 | .805525 |
| E K C | .440146 | .000620 | .812610 |
| E K D | .439088 | .000705 | .810720 |
| Direct S teacher, historical | .542809 | .612057 | .489204 |
| S I, shared F/K | .542213 | .611104 | .488888 |
| S F W | .435438 | .527844 | .362720 |
| S K W | .458851 | .540439 | .394916 |
| S F C | .438375 | .528790 | .367301 |
| S F D | .440754 | .527449 | .372952 |
| S K C | .453786 | .538655 | .387377 |
| S K D | .453524 | .538425 | .387106 |

The tiny initial-versus-direct differences follow the declared numerical teacher parity and separate fixed affine solves; they are not evidence of a new initial audit or access route. E-K final qE ratios near .81 are appreciably below direct E's .9992, while E structure remains accurately reconstructable. Because K is a frozen function of E alone, this can reflect nonlinear teacher structure becoming accessible to an affine decoder after source training. The [composed attack witnesses](COMPOSED_ATTACKS.md) make the same function-class distinction for actual protected-label prediction.

Direct normalized matching error to each condition's own teacher falls during continuation: E-K mean development error goes from **.025965 at W** to **.018027 C / .018062 D**; E-F goes from .034121 to .023596/.023912. S-K goes from .020195 to .015047/.015130; S-F from .029579 to .022298/.022688. Closer coordinates therefore coexist with recoverable qE; closeness is neither a privacy result nor a transfer guarantee.

## Geometry of the actual three-probability banks

The separately frozen [bank supplement](bank_geometry/BANK_GEOMETRY_SCOPE.md) fits exactly18 rectangular **3→16** affine decoders, using all six published banks and the same raw/E/qE targets, priors, original scales and numerical policy. It does not compute three-versus-16 direct coordinate error or read a hidden bank mapper release. All coefficients froze before held-out evaluation. The full process cost was **.915 seconds**, charged once to scientific time; independent replay passed all18 decoders in [SCORE_REPLAY.json](SCORE_REPLAY.json).

| Published bank | Raw ratio | Retained E ratio | Removed qE ratio |
| --- | ---: | ---: | ---: |
| E bank | .877803 | .887431 | .868592 |
| S bank | .882735 | .919011 | .851725 |

These banks preserve much less affine-recoverable teacher structure than the 16-coordinate releases, while still exposing some qE prediction from their three probabilities. Their source advantage and residential cost are measured directly in [SOURCE_BUDGET_CONTROL.md](SOURCE_BUDGET_CONTROL.md). High affine error does not rule out nonlinear recovery from a bank.

## Gradient scope

The fixed first fitting minibatch is reused at I, W, common fork, arm forks and finals. [GRADIENT_DIAGNOSTICS.csv](GRADIENT_DIAGNOSTICS.csv) separates source/teacher/protection raw and applied mapper norms, teacher/raw input-weight blocks, and dots/cosines. Raw protection means positive prior-entropy-normalized CE; D applies −.1 and C applies zero. Before observers are trained, explicitly hypothetical original initialized observers supply the diagnostic raw gradient. The common fork's D objective is prospective; no protection updates occur in common base/observer-only warmup. Banks have no protection component. Zero-vector cosines remain undefined.

All K raw blocks remain exactly zero; F raw blocks become active. At the final D diagnostic, mean applied source/protection cosines are **−.318563 E-F, −.044457 E-K, −.154497 S-F, −.030525 S-K**. Their small or opposing local gradients do not establish why a final audit improved or worsened. Source/teacher final cosines are positive on average in every main condition. All114 diagnostic points replay exactly without extra optimizer steps or RNG/state mutation. Controlled input/protection contrasts, task losses and the independently selected audit evidence carry the scientific comparison.
