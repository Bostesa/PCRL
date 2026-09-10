# Final code, publication and report review

Scope: a bounded review by the transport-admission/publication implementation agent. This is not a substitute for the separate independent full model/prediction replay. No models were fitted or applied, no candidates reselected, and no 2017 model outcomes were inspected in this review.

## Findings resolved before packaging

1. **Compressed report inputs:** report outputs are `*.csv.gz`. The packager now explicitly recognizes each allowlisted compressed filename, preserves exact decompressed CSV bytes, and writes `PUBLICATION_EVIDENCE/NAME.csv.gz` with an empty gzip filename and mtime zero. There is no `.gz.gz` suffix. Original compressed-file hashes and decompressed payload hashes are distinct in the export manifest. A regression fixture and an actual PER_SEED archive check passed.
2. **Fitted preprocessing inside audit metadata:** audit selections include `feature_mean` and `feature_scale`. They are removed from public projections together with nuisance priors, fitted vectors/matrices and private row arrays, with hashes/shapes retained. Audit curves, original candidate IDs, supports, validation metrics and selections remain. These exports are explicitly marked as projections, not byte-identical copies of the full original JSON.
3. **Aggregate metadata retained:** the sanitizer now distinguishes an aggregate `raw_rows` count, exact dictionary-entry metadata and 64-character SHA256 values from actual row identifiers or fitted arrays. The admission JSON therefore remains publishable at its documented root path. Scalar-count, dictionary-hash and identifier-hash regression checks passed.
4. **Projection layout repair:** the receipt covers all 42 evaluation units, with zero fits and zero reselections. It records maximum probability change 2.220446049250313e-16 and maximum log-loss change 1.1102230246251565e-16. Current metrics/predictions have canonical contiguous projected input layout; historical originals and pre-repair generated files are preserved. Every `revisions/` subtree stays local; compact root/per-unit receipts are allowlisted.
5. **Reproduction boundary:** REPRODUCTION and TRANSPORT_ADMISSION distinguish public evidence checking from exact model/report reconstruction. Full report regeneration needs local predictions, labels/weights, matrix diagnostics, split metadata and aligned withholding loss archives. Public-only users can inspect aggregate exports and run the packager's `--verify`. Transport `--verify` requires the preserved `482f14d648ced588949ff23faeb8d0cb616a781f` historical checkout plus the originally inventoried ignored/local evidence. It has no frozen-inventory-only shortcut; a fresh clone or later research tree is insufficient.

## Concrete publication checks

An actual completed-tree classification scan found 84 per-unit metric/audit exports, 15 aggregate compressed CSV inputs, seven numerical diagnostic sources and 314 other allowlisted public inputs at the time of this review; 8,383 other files were local. These are classification snapshot counts, not final published byte totals, and will change when final replay/runtime/control files are added.

All `.npz`, `.npy`, `.pt`, `.joblib` and revision files in that scan classified local. Aggregate CSV headers had no SERIALNO/SPORDER/person/household/raw-row or per-person prediction/loss-vector columns. All seven numerical projections were idempotent and removed row/fold/bandwidth subsets and full moment matrices while retaining support, norms and eigenspectra. Seed `indices.json` files contain SHA256 strings only. The stable root admission, decision, withholding-validation, figure, report-count and report-input-hash JSON files need no private-field removal.

The explicit allowlist covers the 14 conditions in seeds 0–2, final report names and the 144 prescribed PNG/PDF filenames. Unknown files and partial execution logs remain local. Raw `data/` files are outside the packaging traversal. The final staging boundary is `PUBLICATION_FILES.txt`; adding the entire result directory would bypass this boundary and is not the publication procedure. Every public file is capped at 45,000,000 bytes. Self-tests cover deterministic repeated packaging, exact gzip payloads/header, safe output-root restriction, private-state exclusion, public-only verification and corruption rejection. Full package creation has deliberately not been run by this reviewer while final evidence is still being assembled.

## Decision-report factual checks

The completed report has 42/42 evaluation units, 24 globally frozen maps and 144 figures. The original protocol SHA256 still matches PREFIT_FREEZE. All three spectral fitted output ranks are 16; each seed records 15 nuisance fits and zero extra full-training nuisance fits.

Using the public aggregate CSV values with independent standard-library arithmetic, I verified that every spectral arm gains at least .01 residence log loss versus H in all three seeds under both weightings, yet every arm has at least one failing original source-allowance component. The decision records no qualifying arm, no nominee and no supported coordination claim. No numerical privacy guarantee or independent-population confirmation is claimed.

Representative report means match the saved per-seed evidence:

| Quantity | Unweighted | PWGTP |
|---|---:|---:|
| C1 residence gain versus H | 0.0264798689093 | 0.0241886280878 |
| L2 residence gain versus H | 0.0285367285448 | 0.0259983733449 |
| C1 minus L1 residence loss | 0.00230630560566 | 0.00184865519912 |
| C1 minus L2 residence loss | 0.00205685963556 | 0.00180974525705 |

The report correctly separates same-seed source allowance, H residence improvement, original PCA/rich-bank half-headroom, nonlinear recovery and local-race behavior. It retains both weightings, original versus kernel/catch-up scopes, signed recovery, and all-seed vector comparisons. The fixed stochastic withholding mechanism is described as routed per-person loss averaging with visible A/AB branch and unchanged B, not interpolated probabilities; this review did not replace the independent per-person routing replay. The proposed next study is explicitly a limitation study with frozen comparators, not a successful recipe nomination or a launch.

## Limits and final gates

This is reused California 2018 development evidence; three seed summaries are descriptive, and a partition reshuffle cannot create confirmation. Float64 spectral channels versus losslessly promoted historical float32 channels are disclosed. Full class-schema recall/AUROC remains unavailable where support is missing. Small training residual moments do not imply unrestricted conditional privacy, and the report treats held-out moment increases and empirical attacks as separate evidence.

California 2017 admission supports future file-level temporal transport only. Cross-year public keys cannot prove distinct humans, and GQ-person keys cannot establish facility independence. Income remains the original nominal unadjusted PINCP threshold. Alaska Native alone has zero attacker-validation support and one final-evaluation person. No 2017 final model outcome is available.

Exact fitted maps/attackers and person-level predictions, labels, row/fold assignments, branch uniforms and losses remain local. Their hash inventory does not make them downloadable or make public-only exact fitted-object replay possible. Public function knowledge in the threat model is distinct from distribution of fitted artifacts in this repository package. The final independent replay receipt and the final public manifest/archive verification must pass separately before publication; this review does not assert their completion.

The final full-authorized-panel arithmetic also checks: zero spectral-to-withholding and six withholding-to-spectral source-feasible directional passes among 960 fixed comparisons at delta .001. These are descriptive individual comparisons, not a common all-seed selected mechanism. The final documentation refresh adds 48 aggregate negative-increment count rows and explicit signed-negative recovery limitations. INDEPENDENT_SURROGATES.json reports 504 independently reconstructed surrogate rows and 1,728 class-by-arm checks, maximum error 1.1102230246251565e-16, zero fits. I inspected its exact published verifier source: it contains no embedded person records or fitted coefficients, performs no fitting, and requires the documented local raw/model artifacts to run. Only that exact verification_sources/verify_surrogates.py path is allowlisted; other source-directory contents are not implicitly admitted.

## Reviewed identities

- PROTOCOL.md SHA256: `f4cfbeefa2735056b7a4548aa535b88fe487a972f4d60dc7d2cad91e39460653`
- DECISION_REPORT.md SHA256: `de040056e25f01a11cf81313309b3fd7097d4021310718f7c51296e603f3339e`
- DECISION.json SHA256: `40642489b6c5f428356594548081b02ccc9630e4958316734a82b9dfe6c8ec51`
- REPORT_COUNTS.json SHA256: `6a764b280c70e3e6d2525fb86039c37a89f8bde71ba0d60e4d295ea880e41403`
- PROJECTION_LAYOUT_REPAIR.json SHA256: `bd9d79b152b26d273379c4939f4aa995d51f43d337375c7637e1c315e01357ef`
- scripts/package_acs_residual_spectral.py SHA256: `a2ec96792cbb2b861631faffb193093c0f1e88e1769fcea0fb86fead79386874`

No unresolved scientific-result discrepancy was found in this bounded review. Exact full replay and final byte-level publication verification remain the separate required checks above.

Final receipt-only additions: FINAL_INDEPENDENT_REVIEW.json and the exact negative-count/legacy-criteria verifier sources are explicitly allowlisted. Root inspected these sources: no embedded raw records or fitted coefficients; they read saved aggregate metrics and perform no fitting. The separately required full replay subsequently passed42/42 systems (INDEPENDENT_REPLAY.json); final archive verification is recorded by the packager.

Publication assembly correction: git correctly rejected ignored original selection_before_test.json files. The packager now always emits these aggregate selection records as compressed PUBLICATION_EVIDENCE exports, matching the existing handling rule. A classification regression covers the route; source selections and all scientific outputs are unchanged. No ignore override is used.
