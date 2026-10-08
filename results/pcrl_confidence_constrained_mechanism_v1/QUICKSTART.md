# Quickstart: the commands actually run

## Setup

All commands run from the worktree root on branch `research/pcrl-confidence-constrained-mechanism-v1`:
- `<python>` is the repository venv (Python 3.13, numpy, scipy, scikit-learn, torch; versions pinned in
  FEASIBILITY_LOCK.json).
- `<PRIVATE_CACHE>` is the private evaluation cache (`PCRL_CCM_PRIVATE_CACHE` overrides `<PRIVATE_CACHE>/ccm_v1`).
- Heavy commands go through the semaphore `python -m ccm.sema`, which allows at most two heavy processes across all
  roles.

## 1. Tests (synthetic only; 168 tests, about 6 s)

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m pytest -q tests/pcrl_confidence_constrained_mechanism_v1

## 2. Access guard, manifest and toy laws (role F)

    PYTHONPATH=. <python> -m ccm.data manifest          # DATA_ACCESS_MANIFEST.json (label-free; U0 bitwise vs admitted teacher)
    PYTHONPATH=. <python> -m ccm.data toy-laws-check    # exact construction check of TOY_LAWS.json

## 3. Math review scripts (role B; synthetic; numpy, scipy and the standard library; 1–20 s each)

    for f in results/pcrl_confidence_constrained_mechanism_v1/math_review/r*.py; do PYTHONPATH=. <python> "$f"; done

## 4. Feasibility lock (written at commit 75cb8c5, committed and pushed as d5337e5)

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -c "from ccm import lock as L; L.write_lock('FEASIBILITY_LOCK')"
    git add results/pcrl_confidence_constrained_mechanism_v1/FEASIBILITY_LOCK.json && git commit && git push
    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -c "from ccm import lock as L; \
      print(L.verify_lock('results/pcrl_confidence_constrained_mechanism_v1/FEASIBILITY_LOCK.json', stage='geometry'))"

## 5. Locked Stage D runs

The runner refuses unless the lock verifies on origin and every loaded study module is locked.

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m ccm.sema --label A:geometry -- env OMP_NUM_THREADS=1 PYTHONPATH=. \
      /usr/bin/time -l <python> -m ccm.run --lock results/pcrl_confidence_constrained_mechanism_v1/FEASIBILITY_LOCK.json --stage geometry
    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m ccm.sema --label A:oracle -- env OMP_NUM_THREADS=1 PYTHONPATH=. \
      /usr/bin/time -l <python> -m ccm.run --lock results/pcrl_confidence_constrained_mechanism_v1/FEASIBILITY_LOCK.json --stage oracle

**Outputs.**
- Geometry: FEASIBILITY_GEOMETRY.json (public aggregates); 185 s wall.
- Private per-unit records: `<PRIVATE_CACHE>/ccm_v1/run/geometry/{G,G_exp}__s{k}__r{i}.json`.
- Oracle: ORACLE_RESULTS.json; 2.5 s.

## 6. Tables (post-run reporting; reads only the public aggregates)

    PYTHONPATH=. <python> -m ccm.report   # PRIMARY_ENDPOINTS.csv, GEOMETRY_METRICS.csv, ALL_ARMS.csv, PREDICTION_SCORES.json

## 7. Independent verification (role E; does not import ccm.guard, geometry, oracle or run)

See `verify/README.md` for the exact per-script commands. The output is INDEPENDENT_VERIFICATION.json.

## 8. Custody (same-device versioned copy, then restore from the copy alone)

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m ccm.custody copy
    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m ccm.sema --label A:restore -- env OMP_NUM_THREADS=1 PYTHONPATH=. \
      <python> -m ccm.custody restore <PRIVATE_CACHE>/ccm_v1_local_copy_<YYYYMMDD>
    PYTHONPATH=. <python> -m ccm.custody publish   # BACKUP_VERIFICATION.json, RESTORE_INDEX.json

## Not provided

**No deploy command.** No mechanism was fitted, because the go rule failed.
