# Frozen release adapters for the common audit

`release_specs.build_release_specs(anchor, delta, a_center_dir, ...)` verifies the immutable A, optional B, fixed-bank control, and task-only fit receipts before returning an `evaluate.audit_panel`-compatible `releases` mapping. Its `aliases` mapping preserves every declared name while assigning identical laws to one canonical audit. A copied B child channel collapses to its exact T32 parent law for this purpose; task-only publish=1 modes collapse to the unmodified law. It does not fit or score anything.

`aliases` is a dictionary of every declared score name to the canonical name present in `releases`, for example `aliases["A_control_D17"]` and `aliases["TaskOnly_rr_075"]`. A score table should expand the canonical audited result back to every declared name through this map, marking those rows as exact aliases; it should not omit D17 merely because another name has the same law. Control receipts are checked against the selected center's *final* U/W task coefficients and frozen bank hash, in addition to the center receipt and saved Q hashes.

```python
from experiments.pcrl_adaptive_release_v1 import evaluate, release_specs

bundle = release_specs.build_release_specs(
    0, .001, "/private/study/A_anchor0_delta001",
    b_center_dir="/private/study/B_anchor0_delta001",
    b_parity_receipt="/private/study/B_anchor0_delta001_PARITY.json",
    a_controls_dir="/private/study/A_controls_anchor0_delta001",
    b_controls_dir="/private/study/B_controls_anchor0_delta001",
    task_only_dir="/private/study/task_only_anchor0")
# The execution owner alone dispatches this scientific audit:
# evaluate.audit_panel(0, bundle["releases"], index_path, output_dir)
```

The equivalent `python -m experiments.pcrl_adaptive_release_v1.release_specs --anchor 0 --delta .001 --a-center-dir ... --b-center-dir ... --b-parity-receipt ... --a-controls-dir ... --b-controls-dir ... --task-only-dir ... --output-index /private/.../RELEASE_INDEX.json` writes a private aggregate-only identity/receipt index. It does not serialize a live Python callable or issue an audit job. A coordinator must construct specs in the Python process that runs the common audit.

B routing uses the archived Linux T0 code, teacher probability and residual together with allowed local `X_A,H_A` and the frozen nuisance model. No task/protected label, `H_B`, private child ID or kernel row enters the audit predictor. A B spec requires a separately generated `fit_b parity` receipt on Linux x86. The parity receipt must bind the selected Q, immutable B COMPLETE, partition, frozen encoder and nuisance, code/child parity, and unchanged service bytes. Production spec construction fails on a Mac. The test-only `require_current_linux_for_b=False` flag does not create a production parity receipt or change the platform requirement.

All inputs and the optional index stay under a path component named `private`. `build_release_specs` verifies unit inventories; each audit invocation also rechecks the selected channel/model artifacts. B's callable router additionally rechecks its partition, nuisance and parity pins on every call. A control with a changed Q hash or a B parity mismatch is rejected. Exact aliases save repeated model fits while retaining the declared full comparison roster for reporting. Per-person laws, weights, inputs and predictions are never written by this adapter.
