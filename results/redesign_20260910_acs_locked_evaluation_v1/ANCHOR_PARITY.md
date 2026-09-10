# Anchor and release parity

Fresh-sample native anchor/readout scores are unavailable: the sample has0 rows. No vacuous empty-array equality is reported as scientific preservation success.

Historical plumbing replay independently reproduced **63 original full anchor arrays**, **630 release arrays**,21 teacher arrays and21 raw-preprocessor/PCA transformations exactly. All18 system release graphs preserve H's full anchor columns in float64. B selected outputs/scores pass960 historical parity checks across alternatives;1,050 raw H-projection witnesses and26 selected singleton witnesses match exactly.

[HISTORICAL_REPLAY.json](HISTORICAL_REPLAY.json) records exact output comparisons, selected saved states and hashes. The new graph itself was independently checked on all18 historical representation-fit fixtures (90 wire/derived arrays), with unchanged mapper/teacher state; see [CODE_REVIEW.md](CODE_REVIEW.md). Identical batching is required for bitwise float32 parity: a different GEMM batch shape produced rounding up to1.91e-6 in a preliminary small slice, within the original mapper tolerance1e-5. No tolerance was changed to affect a scientific comparison.

Fixed service preservation is a design property, distinct from accuracy on another population and from the selected source-readout PCA32+.01 allowance. Neither fresh quantity can be checked here.
