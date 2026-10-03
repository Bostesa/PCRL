"""Real-data audit controls on attacker_fit / attacker_val only (no assessment rows).

For a release view X: (a) label permutation: SEX permuted within attacker_fit and within attacker_val (seed 20261014);
the secondary-slate attacker's validation AUC must be near 0.5 (flag > 0.55); (b) planted leak: append a one-hot copy of
SEX with 20% of rows replaced by a random class; validation AUC must exceed 0.75. Run on local and coalition views of
the selected J and L releases (seed 0) and of the selected FARE pair (finite).

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m jcv.controls --selection-lock <SELECTION_LOCK.json>
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jcv import audit as AU
from jcv import data as DA
from jcv import outer as O
from jcv import run as R


def control(X, S, D, finite):
    f, v = D["idx"]["attacker_fit"], D["idx"]["attacker_val"]
    rng = np.random.default_rng(20261014)
    Sp = S.copy()
    Sp[f] = rng.permutation(S[f])
    Sp[v] = rng.permutation(S[v])
    sel = O.fit_select(X[f], Sp[f], X[v], Sp[v], O.secondary_slate(finite), 2)
    null = AU.macro_auc(Sp[v], AU.proba(sel["model0"], X[v], 2), [0, 1])
    noisy = S.copy()
    flip = rng.random(len(S)) < 0.2
    noisy[flip] = rng.integers(0, 2, flip.sum())
    Xp = np.hstack([X, np.eye(2)[noisy]])
    sel2 = O.fit_select(Xp[f], S[f], Xp[v], S[v], O.secondary_slate(finite), 2)
    planted = AU.macro_auc(S[v], AU.proba(sel2["model0"], Xp[v], 2), [0, 1])
    return {"null_val_auc": null, "null_flag_above_0.55": null > 0.55, "planted_val_auc": planted,
            "planted_detected_above_0.75": planted > 0.75}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-lock", required=True)
    a = ap.parse_args(argv)
    SL = json.loads(Path(a.selection_lock).read_text())
    D = DA.load()
    out = {}
    arms = SL["seeds"]["0"]["arms"]
    for arm in ("J", "L", "F"):
        V, _, finite, _ = O.release_for(arms[arm])
        for w in ("v1", "v2", "pair"):
            out[f"{arm}/{w}"] = control(V[w], D["sex"], D, finite)
    (R.RUN / "controls.json").write_text(json.dumps(out, indent=1, default=bool))
    (R.PKG / "AUDIT_CONTROLS.json").write_text(json.dumps(out, indent=1, default=bool))
    return out


if __name__ == "__main__":
    main()
