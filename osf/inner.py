"""Inner audits (SELECTION_AND_AUDIT_LOCK): every release unit rel__s{k}__{config} (and lc__s{k}__E when present) is
audited with the study's inner slate (osf.audit.inner_audit: fitted on AUDIT_FIT, selected/scored on INNER_SELECTION;
sealed assessment labels refused) and its deployed heads' INNER_SELECTION utility. FARE units are audited by the
audit/baseline owner's selection path (osf.baselines). Units: inner__<unit>.
"""
from __future__ import annotations

import time

import numpy as np

from osf import run as R
from rgj import finalize as FN


def inner_utility(o, D):
    v, tr = D["idx"]["INNER_SELECTION"], D["idx"]["DEFENSE_FIT"]
    res = {}
    for i, t in enumerate(FN.TASKS):
        y = D["y"][t]
        assert (y[v] >= 0).all() and (y[tr] >= 0).all()
        const = int(np.argmax(np.bincount(y[tr])))
        res[i] = {"acc": float((o[f"hard{i + 1}"][v] == y[v]).mean()), "const_acc": float((y[v] == const).mean()),
                  "const_class": const}
    return res


def targets():
    return [d.name for d in sorted(R.UNITS.glob("*")) if d.name.startswith(("rel__", "lc__")) and R.done(d.name)]


def run_inner(D, shard_spec=None):
    from osf import audit as AU
    todo = [n for n in targets() if not R.done(f"inner__{n}")]
    for n in R.shard(todo, shard_spec):
        z = np.load(R.U(n) / "release.npz")
        assert np.array_equal(z["row_id"], D["row_id"]), f"{n}: release rows differ from the osf rows"
        V = FN.views_from_release(z)
        t0, c0 = time.time(), time.process_time()
        r = {"of": n, "recovery": AU.inner_audit({w: V[w] for w in ("v1", "v2", "pair")}, D),
             "utility": inner_utility(V["out"], D)}
        r["wall_s"], r["cpu_s"] = time.time() - t0, time.process_time() - c0
        FN.save_unit(R.U(f"inner__{n}"), {}, r)
        R.event("unit complete", unit=f"inner__{n}")
