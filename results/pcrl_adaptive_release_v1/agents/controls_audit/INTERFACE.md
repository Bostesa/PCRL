# Controls and inner audit interface

`controls.aggregate_coefficients(leaf_ids, per_person_token_loss, weights, n_states=...)` returns a state-by-17 matrix with normalized person mass already included. `controls.lift_parent_channel(D17_Q, partition.parent_of_leaf)` is the exact deterministic reference on every child partition. Pass the same calibrated `{id, role, weighting, coeff, rho, delta, floor}` bank and cost to `controls.solve_deterministic_p1` and `controls.optimize_gradient_bank`; preserve their incumbent, valid lower bound, feasibility replay and unresolved oracle error separately. `constant_replacement` and `randomized_response` retain 17 visible token values.

`evaluate.audit_panel(anchor, releases, index_path, output_dir, *, resume=False, slate='catchup')` accepts an in-process mapping from release ID to `{Q, channel_artifact_path, channel_artifact_sha256, router, ...}`. The channel artifact is an NPZ containing only `Q`. `router='T0'` uses stored Linux x86 T0 codes. A Branch B callable router requires `router_artifact_path`, its `router_sha256`, and `router_entrypoint`; it receives only `x`, `ha`, stored `token_codes`, `teacher_p`, and `residual`. The latter three are frozen functions of the allowed local inputs, retained from Linux x86 encodings for exact 2018 replay. Labels, H_B, household IDs and weights are withheld. The router's private leaf ID chooses a token law for exact audit and never reaches a predictor or the release wire. The receipt records relative artifact paths and SHA-256 digests. The coordinator must bind the callable entrypoint to the pinned frozen router object in its job manifest.

The panel reads only `audit_fit`, `inner_selection` and `inner_check` through `roles.pooled_role`. It fits seven shared H-only ancestor slates once per panel and five fresh slates per release. Each A route includes own and H-only predictors; each AB route includes own, H-only, A same-release and B H-only predictors. For the five scored roles it locks validation-selected routes on `inner_selection`, scores `inner_check` using expected token log loss, and writes aggregate `INNER_AUDIT.json`, private person and household `PANEL_CONTRIBUTIONS.npz`, and SHA-inventoried `COMPLETE.json`. Completed outputs are immutable. An explicit `resume=True` only re-enters incomplete per-slate fits under inherited hash verification.

From the study worktree, the tested public synthetic fit-to-release-to-exact-score command is:

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m experiments.pcrl_task_aligned_cuts_v1.audit synthetic
```

It produced a 32×17 fitted toy channel, a fixed-bank objective and dual lower bound of 0.5, byte-preserved H_A, one sampled token per each of four people, and independent exact task loss 0.7803238741323343. To exercise the new panel's full role, route, fit dispatch, score and receipt orchestration on synthetic rows with stubbed model fitting:

```sh
/Users/nathansamson/PCRL/.venv/bin/python -m pytest -q tests/pcrl_adaptive_release_v1/test_evaluate.py
```

No ACS audit has been dispatched by this worker. The `audit_panel` output is 2018 development evidence, not untouched confirmation.
