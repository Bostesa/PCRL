# Independent control-gate verification correction E

Source inspection before any new INNER truth access found incomplete verifier coverage: the prepared final worker replayed INNER real controls but did not independently replay ATT controls, check the combined ROOT gate vectors, or explicitly compare supplied truth and bank pins with the committed reporting lock. The runner already constructed these inputs under its locked reporting interface; no observed Adult outcome prompted this correction.

The original verifier source is preserved privately, with SHA256 `128ad3ecdd8aecd7c13ecf9524528707c96cded33e2fcac9ff7cd105cc51b687`. Corrected `final_worker.py` SHA256 is `423447b0f44bac26e6d67a3cf9b7330adc9140c3bfda57a4ad762ed8d6e0a0cb`.

The corrected worker independently recomputes all real-control gates for U0_FULL and RAW_FULL, both source seeds, using their ATT validation and INNER scores. It compares every aligned constant boolean ROOT gate vector with that conjunction. It writes independent gate vectors for K240 replay alongside independently computed row metric vectors. Supplied truth-label hashes and bank-freeze hashes must belong to the committed reporting lock before any truth arrays are materialized.

All original thresholds, endpoint membership, selection rules, model recipes, orientations and bootstrap definitions are preserved. This completes verification arithmetic and input binding; it does not alter a scientific criterion or train a model.

Finite queued tests are prepared. The RED test executes the preserved verifier against an engineered false-positive ATT gate and is expected to fail, demonstrating the omission. The GREEN test checks eight cases: ATT and INNER failures, nonconstant and nonboolean gates, uncommitted labels and banks before array access, valid true conjunctions, and valid false gates retained for MISSING inference. No real person arrays are opened by these fixtures. Only ROOT may enqueue and execute them; preparation alone does not establish that they pass.
