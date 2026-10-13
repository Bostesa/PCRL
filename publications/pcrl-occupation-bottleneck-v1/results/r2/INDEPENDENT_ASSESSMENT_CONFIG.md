# Root-owned independent assessment configuration

This is a static interface review of `occupation_bottleneck_independent/replay.py`. It is not a numerical verification result and does not approve the R2 geometry implementation. Geometry approval must come from the separate R4 functional-forward/SVD replay and its actual numerical receipts.

Run two `verification` queue units with `python -m occupation_bottleneck_independent.replay --spec PRIVATE_CONFIG.json`. Each config binds `unit_id`, operation `assessment`, canonical committed `reporting_lock` pin, `endpoint_registry` pin, `unit_input` pin, `arrays` dictionary of named pins, a fresh `public_output` and optionally a fresh `private_output`. Register every code/config/registry/input pin as that queue unit's sources. The independent worker imports only the campaign queue authenticator, not runner scientific functions.

The canonical unit NPZ and every table in `arrays` must contain `row_id` and `unit` in exactly the same row order, as well as the relevant metric/score/truth vectors. The verifier checks unique row identity, alignment, and registered 2,235 rows / 2,234 groups. Do not assume that equal lengths prove person pairing. Newly opened INNER truth requires the globally frozen, committed reporting lock before hashing or reading it.

## Separate families with the same resampling

Privacy registry: `family:"privacy"`, `configuration_count:36`, `K:108`, `replicates:2048`, `bootstrap_seed:2026101301`, `alpha:0.05`, `expected_rows:2235`, `expected_groups:2234`, and exactly 108 unique endpoints. These are three endpoints per paired configuration: pair benefit against matched RAW ≥0.020 and income/occupation local worsening against matched RAW ≤0.010. Each endpoint has two `paired_seeds` parts; average the two seed-specific contrasts within each shared duplicate-group draw.

Utility registry: `family:"utility"`, `configuration_count:36`, `K:864`, and the same replicate count, seed, alpha and units. There are 24 original clauses per paired configuration: two tasks × two source seeds × two references × three metrics. Each endpoint therefore has exactly one `paired_seeds` part for its named source seed. Averaging utility across the two seeds would change the original requirement and is forbidden. Accuracy loss/NLL excess caps are 0.010; summed Brier excess cap is 0.005.

The two registry jobs independently regenerate identical group draws from the same canonical units and seed. Each endpoint's confidence interval uses paired-group bootstrap standard error, Student-t critical value with df=groups−1 and alpha/(2K), with its own actual family K. This matches the inherited registered arithmetic. The simultaneous scope is each frozen family; do not advertise a combined privacy-plus-utility coverage level that has not been declared.

## Endpoint records

Every endpoint contains `endpoint_id`, `kind`, `threshold`, `direction` (`at_most` or `at_least`) and nonempty `paired_seeds`.

For utility `kind:"mean_difference"`, each part contains `left_table`, `left_key`, `right_table`, `right_key`, optional `sign` (±1). Vector tables contain actual per-row accuracy/NLL/Brier outcomes. Encode accuracy loss as reference-minus-candidate; encode NLL/Brier excess as candidate-minus-reference. These are signed differences, not absolute distances.

For privacy `kind:"auc_envelope_difference"`, each part contains `truth_table`, `truth_key`, `left_table`, `left_keys`, `right_table`, `right_keys`, optional `sign` (±1). Key lists are the entire frozen eligible reader envelope for that contract, including pair local-ignore aliases. Encode pair benefit as RAW-minus-candidate; local worsening as candidate-minus-RAW. The threshold stays 0.010 for local worsening; subtract 0.010 only in the separately reported guard excess, not twice inside the interval endpoint.

ATT-frozen orientations are already reflected in score vectors. Do not reorient on INNER truth. Missing endpoints require explicit MISSING accounting and cannot silently reduce K or shrink envelopes after outcomes. An unavailable essential unit prevents a complete simultaneous claim.

## Static audit findings and remaining numerical work

Independent metric arithmetic uses probability floor 1e-12, summed multiclass Brier and first-class decisions. Its tie-block AUC computation differs from the runner implementation, and its bootstrap group-count construction is separate. Its paired averaging and Bonferroni-t construction match the frozen interpretation. Added guards reject table/person misalignment, invalid directions/signs, nonfinite metric vectors, empty/duplicate reader envelopes, wrong family sizes and incorrect seed averaging.

Remaining numerical checks must compare actual point/interval endpoints against the runner, account for all 36 configurations and missing/alias units, replay selection/gates, and independently refit frozen U0/RAW readers on both seeds. Actual source folding and geometry forward replay are R4 work. The R2 author cannot independently certify their own geometry merely by reading a second implementation.

Geometry audit config uses a public fixed-artifact JSON list `[income_geometry, occupation_geometry]`, with B/P serialized for each recipient. The reader worker currently constructs the list internally; the root must write and pin that fixed artifact if public basis/projector comparison is requested. No q or other row-dependent metadata belongs in that artifact.
