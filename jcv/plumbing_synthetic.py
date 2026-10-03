"""End-to-end plumbing check on a fully synthetic dataset in a scratch units root (no real data, reduced epochs).
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m jcv.plumbing_synthetic <scratch_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

from jcv import data as DA, run as R, train as T


def synthetic_D(n=6000, d=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    sex = (X[:, 0] + rng.normal(size=n) > 0).astype(np.int64)
    race = rng.choice(5, n, p=[.8, .1, .05, .03, .02]).astype(np.int64)
    inc = (X[:, 1] + 0.7 * X[:, 0] + rng.normal(size=n) > 0.5).astype(np.int64)
    occ = np.digitize(X[:, 2] + 0.5 * X[:, 0] + 0.3 * rng.normal(size=n), [-1.2, -0.4, 0.3, 1.0, 2.2]).astype(np.int64)
    role = rng.choice(["defense_train", "defense_val", "attacker_fit", "attacker_val", "assessment"], n,
                      p=[.45, .1, .2, .1, .15])
    D = {"row_id": np.arange(n), "unit": np.arange(n), "role": role, "X": X, "sex": sex, "race": race,
         "y_income": inc, "y_occupation_group": occ}
    D["idx"] = {r: np.flatnonzero(role == r) for r in np.unique(role)}
    D["y"] = {"income": inc, "occupation_group": occ}
    return D


def main(root):
    root = Path(root)
    R.RUN, R.UNITS, R.PRIV = root / "run", root / "run" / "units", root
    T.HP.update(warm_epochs=2, prot_epochs=2, guard_size=512)
    R.FARE_GRID = R.FARE_GRID[2:4]
    D = synthetic_D()
    for k in (0, 1, 2):
        R.stage_warm(D, k)
        R.stage_train(D, k)
        R.stage_fare(D, k)
        R.stage_inner(D, k)
    from jcv.select import run_selection
    sel = run_selection(D, [0, 1, 2])
    print(json.dumps({k: {a: v["status"] for a, v in s["arms"].items()} | {"C*": s["comparator"]["arm"]}
                      for k, s in sel.items()}, indent=0))
    from jcv import outer as O
    SL = {"seeds": {str(k): sel[k] for k in sel}}
    for k in (0, 1, 2):
        for arm in O.ARMS:
            O.outer_unit(D, k, arm, SL["seeds"][str(k)]["arms"][arm])
    (root / "SL.json").write_text(json.dumps(SL, default=float))
    from jcv import infer as I
    DA.load = lambda: D
    R.PKG = root
    out = I.main(["--selection-lock", str(root / "SL.json")])
    print({k: out[k] for k in ("claim1", "claim2")})
    print([(e["id"], round(e["point"], 4) if e["point"] is not None else None, e["decision"]) for e in out["primary"]])


if __name__ == "__main__":
    main(sys.argv[1])
