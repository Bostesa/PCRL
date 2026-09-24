# Quickstart

**Status: `EXPERIMENTAL_NO_ADVANTAGE`, development-only.** No route passed. Do not deploy any unit as a competitive or privacy-guaranteed release.

## 1. Tests (synthetic; no private data)

```bash
cd <worktree root>
PYTHONPATH=. python -m pytest -q tests/pcrl_shared_context_release_v1
```

On 2026-09-24 (about 20:10Z) this gave 126 passed on macOS, using `/Users/nathansamson/PCRL/.venv` (Python 3.13.7, torch 2.10, sklearn 1.8, scipy 1.17, numpy 2.4).

## 2. Restore a fitted release and replay its per-person law (tested on 2026-09-24)

Restoring needs the authorized `vein` AWS profile and the private sanitized 2018 inputs (`results/pcrl_task_aligned_cuts_v1/private/sanitized_2018_v2`, gitignored). Keys, versions and SHA-256 values are in `MODEL_MANIFEST.json`.

```bash
P=<owner-only private dir>/units; mkdir -p $P && cd $P/..
for u in a0_bank a0_NM4_P; do
  aws s3api get-object --bucket pcrl-ux-archive-ed9d21fd \
    --key pcrl_shared_context_release_v1/units/$u.tar.gz --version-id <version from MODEL_MANIFEST> $u.tar.gz
  shasum -a 256 $u.tar.gz   # must equal archive_sha256 in MODEL_MANIFEST.json
  tar -xzf $u.tar.gz -C units
done
cd <worktree root>
PYTHONPATH=. python - <<'EOF'
from experiments.pcrl_shared_context_release_v1 import audit_panel, laws, release
pools, _ = audit_panel.load_inner_pools('results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json', 0)
law = release.load_law('<P>/a0_NM4_P', bank_dir='<P>/a0_bank')
q = law(laws.legal_inputs(pools['inner_check']))   # (n, 17) per-person token law; legal inputs only
print(q.shape, laws.law_identity(q))
EOF
```

In the tested run, the law identities on audit_fit, inner_selection and inner_check matched the Linux host's alias ledger bit for bit (`MODEL_MANIFEST.json` → `representative_restore`).

## 3. Emitting one persistent token per record

`release.py` session objects emit one HMAC-keyed persistent token per record ID from a private key. The key must be owner-only and at least 32 bytes. The wire carries `h_a` plus `token`; changed inputs for a known ID are refused. The session is covered by `tests/pcrl_shared_context_release_v1/test_release.py`. No production ACS emission was run.

## 4. Re-generate the capacity and manifest reports (on a host holding the unit tree)

```bash
PYTHONPATH=. python -m experiments.pcrl_shared_context_release_v1.reports capacity \
  --units-root <units> --index results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json --out ENCODER_CAPACITY.json
PYTHONPATH=. python -m experiments.pcrl_shared_context_release_v1.reports manifest \
  --units-root <units> --queue <QUEUE.json> --out RUN_MANIFEST.json
```
