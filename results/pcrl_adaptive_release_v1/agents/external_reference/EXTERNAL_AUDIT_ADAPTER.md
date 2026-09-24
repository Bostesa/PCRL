# Frozen continuous-release audit adapter

`experiments/pcrl_adaptive_release_v1/external_audit.py` audits J and the five
historical 16-coordinate continuous auxiliary releases on the current 2018
global household roles. They are contextual frontiers: they do not share the
adaptive candidate's 32-state/17-token release contract or supervision. Their
old 2018 aggregate tables are not paired results from this audit.

The private index is
`results/pcrl_task_aligned_cuts_v1/private/EXTERNAL_RELEASE_INPUTS.private.json`
in the completed cuts worktree, SHA-256
`461b06f0bba10416e4a6bec6bc0c69a3821ae2ef9733bc45030ffd0dfea4ad8a`.
It pins all 15 per-anchor NPZ files and their member paths. The files currently
exist in the older task-directed private input tree; the prior admission checked
15/15 source hashes and 225/225 service views with zero mismatches. The private
S3 manifest is
`s3://pcrl-ux-archive-ed9d21fd/pcrl_task_directed_release_v1/final-5a78a39412f7/MANIFEST.private.json`,
version `BbLrQtvGGEm.XHvCGuZpQyX6wZWHnYEi`, SHA-256
`504944807f9f9b9f087836d34768440647b6276db0fca2827be3084213b3e667`.
Restore only required members, checking each NPZ SHA before loading. J16 is
already in each pinned prepared pool as `ctx.pools[pool]['J']`; it is not a
separate NPZ. The J prepared source SHA for anchors 0, 1, 2 is respectively
`01d8bf65a6c2576f2e949264f1e09fbe604c8a6059251d1d54bb72e2edf75341`,
`cdede00b751f34dc4ed47e4dd6de9b85f3a31eeecf318085e4f464a28b11ad66`,
`12ffb406a447e1d7c9029fccd12d179a17240e08c291ae2c40d27c44ba9cf97c`.

The adapter verifies the index, each external NPZ, all four admitted historical
pool arrays, and exact H_A/H_B service bytes before assigning households.
The archive order is `A=(H_A4,aux16)`, `AB=(H_A4,aux16,H_B2)`, and `B=H_B2`.
The inherited same-host scorer concatenates AB features internally as
`(H_A4,H_B2,aux16)`: this is one fixed column permutation, used identically
for every fresh AB fit, validation selection, and score. No historical AB
predictor is reused across the order change. The same `audit_fit`,
`inner_selection`, `inner_check`, slate, seeds, role schema, weight estimands,
and H-only/A-only/B-only ancestor routes are used as in the finite panel.
The single-column all-ones law is bookkeeping for **no token**, not a token
release or a claim that the continuous output is private. Only inner roles
are exposed here; outer assessment remains sealed.

After placing the private index and selected member files on the verified host
under a private `EXTERNAL_ROOT` matching their index-relative member paths,
the coordinator can dispatch, for example:

```sh
/opt/pcrl/venv/bin/python -m experiments.pcrl_adaptive_release_v1.external_audit \
  --anchor 0 --methods J leace_A0 splince_A0 optnet16_C1 optnet16_L1 optnet16_L2 \
  --index INPUT_INDEX.json \
  --external-index EXTERNAL_RELEASE_INPUTS.private.json \
  --external-index-sha256 461b06f0bba10416e4a6bec6bc0c69a3821ae2ef9733bc45030ffd0dfea4ad8a \
  --external-root EXTERNAL_ROOT \
  --out results/pcrl_adaptive_release_v1/private/external_inner_a0 \
  --slate catchup
```

Use the actual private index paths in that command. The output contains fitted
weights, frozen route selections, aggregate inner scores, original-household
contributions, source hashes, and an immutable `COMPLETE.json`. The stage is
intentionally separate from `evaluate.py`, which rejects continuous auxiliary
data to protect the candidate's 17-token contract. Its results must be labeled
different-access operational references in the final tables.
