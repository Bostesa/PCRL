# Quickstart and executed checks

Work from the `research/pcrl-method-portfolio-v1` worktree. The campaign remains blocked for Adult fits. `CAMPAIGN_LOCK.json` has `adult_fit_permission: false`; this page does not grant permission to change that gate.

## Commands executed in this campaign

```sh
git rev-parse HEAD
git cat-file -t 1baf5bbdaf59cfa6a664cae07d652a4712bdfdc5
git cat-file -t 159537dcb61d030c870ae42e7addff4796651723
git cat-file -t d85f0a6f44415f2f9e334870f4e68504cd6b6080
git cat-file -t d5337e552e78b31830f94d85ef1cc4b335b1b8ec
git cat-file -t dc57711a1d80fe4868f93008339ff2d02d467f7b
```

The local object checks succeeded. `git ls-remote` and research-branch pushes to GitHub failed on DNS resolution, so current remote tips were not verified.

From `results/pcrl_method_portfolio_v1/shared`, the fail-closed guard suite was executed successfully with the repository source on `PYTHONPATH`:

```sh
PYTHONPATH="$(cd ../../.. && pwd)" python3 -m unittest -v test_safe_data_guard
```

All six synthetic tests passed. S3 also ran seven `test_compute_queue.py` engineering tests and verified the hashes in `PHASE1_QUEUE_INDEX.json`; see `RESOURCE_RECEIPTS.json`. Some teams ran small synthetic checks before the global queue rule was in place. Those deviations and their conservative CPU charge are recorded there. The final A/B/C/D numerical checks have **not** run and must enter the shared queue when a fresh occupancy receipt opens it.

Static checks used `python3 -m py_compile` or AST parsing for team code, `python3 -m json.tool` for structured files, and `git diff --check`. These do not establish numerical correctness.

## Safe continuation order

1. Find and independently verify a **pre-existing** private export containing only allowed role rows, permitted features and needed labels. Never open the all-role NPZ to construct one. Bind hashes and role fields through `shared/safe_data_guard.py`.
2. Refresh the machine-wide process inventory and reopen `shared/compute_queue.py` only if the two-heavy-process limit has capacity. Do not stop other sessions' jobs.
3. Run the queued synthetic law, adapter and trainer checks through that queue. Fill the D distortion-matched control and the common attack slate. Smoke-time every architecture and close the exact unit table within the 20 CPU-hour budget.
4. Replace the blocking draft with a complete `CAMPAIGN_LOCK.json`, commit and push it, and verify its remote tip before any Adult fit. If GitHub remains unavailable, keep fit permission false.
5. If all gates pass, run the two paired-seed development screen, apply the frozen within-track promotion rule, and create `FINALIST_LOCK.json` only for selected candidates. Report every attempted unit.

The old assessment person-level arrays and HEAD_VALIDATION labels remain embargoed. The source hcal/CCM and team artifacts are historical or prefit evidence, not new endpoints.
