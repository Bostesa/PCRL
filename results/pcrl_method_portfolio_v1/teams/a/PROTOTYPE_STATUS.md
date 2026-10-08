# Team A prototype execution status

The finite-channel implementation is finite_channel.py. It reads no files or Adult data. Its input is a supplied finite probability law, public same-class output bank, and local channel matrices. The solver is a fixed-other HiGHS LP or one-hot MILP with the registered two-stage tie rule; fit_alternating is a bounded coordinate heuristic.

## Pre-run custody

- S1 common synthetic law: shared/SYNTHETIC_LAWS.json, SHA-256 b2e68f5dec7dd690afa9389c44c0c2f98a78b30cde25c20a48751665f35fdc53.
- Original Team A tests: test_finite_channel.py, SHA-256 03181f42693d18eba06b432a8b9098f2c0dec962eb82a77061a65ed02236774c, recorded in S1's Phase I lock before the first red run.
- Additive algorithm tests: test_finite_channel_extensions.py, SHA-256 37794e3300eaa93f98a0af2c86479d372bff06cb4a27ff071f4194245b75aed4, recorded in Phase I amendment 001 before a run.
- Supplementary L2 witness test: test_l2_witness.py, SHA-256 8aca246e436f274fdb81af17499cf80fdf06f403a2dda92f9d4677f3d70e803e, recorded in Phase I amendment 002 before a run.

## Commands and outcomes

1. Red TDD check: /Users/nathansamson/PCRL/.venv/bin/python -m pytest -q results/pcrl_method_portfolio_v1/teams/a/test_finite_channel.py in the Team A worktree. Eight tests failed on deliberate NotImplementedError stubs; no candidate algorithm or numerical solver ran. This preceded the global queue-closure notice and is an unqueued engineering check.
2. Static syntax check: python3 with ast.parse on the implementation and all three test files. All four parsed.
3. Green and supplementary synthetic tests: **queued/pending** while S3's global numerical queue is closed. No green result is claimed.

No Adult input, role array, label or model checkpoint was opened. No real-data fit was run.
