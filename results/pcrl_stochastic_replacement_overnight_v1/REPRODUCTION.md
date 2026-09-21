# REPRODUCTION

Commit: see `HANDOFF.json` `evidence_commit`. Branch
`research/pcrl-stochastic-replacement-overnight-v1`, based on `cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a`.

## Environment

Python 3.13, numpy 2.5.3, scikit-learn 1.9.1, torch 2.14.0, scipy 1.18.1, cvxpy 1.9.3
(CLARABEL + SCS), matplotlib. Isolated virtualenv; **no project dependency was upgraded**.

This differs from the historical study environment (`results/pcrl_utility_extension_v1/REPRODUCE.md`:
torch 2.10.0, numpy 2.4.2). Everything published here is either **data-free** (fixtures, the
synthetic comparison) or a **new measurement** rather than a reproduction of a historical number, so
the difference is immaterial. Reproducing a *historical* number requires the pinned environment.

## Inputs

The restored `exec_main__0000` chunk (2,545 files, archive
`pcrl-ux-archive-ed9d21fd`, object version `vqtdaPnnad06uT5j41O78Y8z35mI2XkN`) provides `anchors.npz`
and `training/J/releases.npz` for all three anchors plus the 2018 ACS extract. It was **verified in
place**, not re-downloaded: 9 key files plus a 40-file random sample all matched their manifest
SHA-256.

The resolver finds it through `FALLBACK_ROOTS`; point it there with:

```
export PCRL_NLR_FALLBACK=/Users/nathansamson/PCRL-terminal-1-stochastic
```

If that tree is gone, restore into any fallback root:

```
python infra/pcrl_utility_extension_v1/restore.py \
  --bucket pcrl-ux-archive-ed9d21fd --prefix pcrl_utility_extension_v1 \
  --chunks exec_main__0000 --roots roots.json --verify-dir /tmp/verify
```

## Commands

```
# fixtures (data-free, ~3 s)
python -m pytest tests/pcrl_stochastic_replacement_overnight_v1/ -q         # 45 tests

# the representation screen (~121 s, CPU, reads 2018 development pools)
python scripts/run_stage_r.py          # writes results/.../stage_R/

# the certified synthetic comparison (data-free, ~30 s)
python scripts/run_synthetic.py        # writes results/.../synthetic/

# intervals, panel, replay
python scripts/make_intervals.py
python scripts/make_panel.py
python scripts/replay.py               # writes results/.../REPLAY.json
```

Thread limits: `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`. The screen's peak
resident set is well under 4 GiB; no swap pressure was induced and no unrelated process was touched.

## Determinism

Codebook seeds are `CODE_SALT + 1000 * family_index + anchor` with `n_init` fixed; probe seeds are
`1250000 + 100 * anchor`; bootstrap seeds are fixed in `uncertainty.py`. The screen reproduces
bitwise on the same platform. k-means and the MLP carry the usual cross-architecture tolerance.

## Resumability

Every unit is ledgered under `results/.../ledger/` keyed by a **content hash of its configuration**,
so an edited configuration cannot silently reuse a stale result. A run lock (`run.lock`) refuses a
second live worker and breaks only a genuinely dead holder on the same host. Artifacts are promoted
atomically, so a partial unit can never be observed as complete.
