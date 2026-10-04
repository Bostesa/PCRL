"""Runner for the focused no-erasure penalty study (resumable, atomic hash receipts).

    env OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m pnx.run --lock <LOCK.json> --stage parity|train|alias|inner|select [--seeds 0 1 2]

parity  real-data engineering checks BEFORE any nonzero fit: (a) this module's copy of the predecessor training loop
        reproduces the predecessor's JP (beta 1, seed 0) unit bitwise; (b) PN and LN at beta = 0 reproduce U bitwise
        (model, release) on every seed. Failing receipts are preserved; nonzero fits refuse to start until parity passes.
train   PN, LN x beta in {0.1, 1, 10} x seeds, from the predecessor's hashed warm starts; finalised with identity maps.
alias   hash-checked alias records for reused predecessor units (warm, U, E, JP x 3 beta, FARE F/F0 and their inner units).
inner   inner audit (attacker_fit -> attacker_val) of the new units with the predecessor's functions.
select  registered selection (pnx.select).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from jcv import data as DA
from jcv import finalize as FN
from jcv import run as JR
from pnx import train as PT

HOME = Path.home()
PRIV = HOME / "PCRL_eval_cache_private" / "pnx_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
PRED = HOME / "PCRL_eval_cache_private" / "jcv_v1" / "run" / "units"
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_penalty_no_erasure_v1"
KS = [2, 6]
BETAS = [0.1, 1.0, 10.0]
NEW_ARMS = ["PN", "LN"]
INPUT_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"


def event(msg, **kw):
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "ACTIVITY_LOG.jsonl", "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": msg, **kw}) + "\n")


def U(name):
    return UNITS / name


def udir(name):
    """Directory holding a unit's files: a new unit, or the predecessor directory named by an alias record."""
    d = U(name)
    a = d / "ALIAS.json"
    if a.exists():
        return HOME / json.loads(a.read_text())["source"].removeprefix("~/")
    return d


def done(name):
    d = U(name)
    if (d / "ALIAS.json").exists():
        rec = json.loads((d / "ALIAS.json").read_text())
        src = udir(name)
        return FN.unit_complete(d) and FN.unit_complete(src) and \
            hashlib.sha256((src / "COMPLETE.json").read_bytes()).hexdigest() == rec["source_COMPLETE_sha256"]
    return FN.unit_complete(d)


def unit_name(k, arm, beta):
    if beta == 0:
        return f"nn__s{k}__U"           # beta = 0 is an exact alias of U after the parity check
    return f"pn__s{k}__{arm}__b{beta:g}"


def load_D():
    if hashlib.sha256(DA.INPUTS.read_bytes()).hexdigest() != INPUT_SHA:
        raise SystemExit("REFUSED: admitted input hash mismatch")
    return DA.load()


def warm(k):
    src = PRED / f"warm__s{k}"
    if not FN.unit_complete(src):
        raise SystemExit(f"REFUSED: predecessor warm start {k} not hash-complete")
    st = torch.load(src / "warm.pt")
    rec = json.loads((src / "record.json").read_text())
    return st, {int(i): v for i, v in rec["budgets"].items()}


# ------------------------------------------------------------------ parity (engineering, before nonzero fits)
def _state_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a) and set(a) == set(b)


def stage_parity(D, seeds):
    data = JR.tensors(D)
    d_in = D["X"].shape[1]
    out = {"checks": []}
    ok = True
    for k in seeds:
        data.seed = k
        st, budgets = warm(k)
        refU = torch.load(PRED / f"nn__s{k}__U" / "model.pt")
        relU = np.load(PRED / f"nn__s{k}__U" / "release.npz")
        # environment check first (review A1): U re-trained here must equal the predecessor's U, so that a later failure
        # separates environment drift from critic interference
        mU, _, _ = PT.train_arm("U", 0.0, st, d_in, KS, data, k, budgets)
        envok = _state_equal(mU.state_dict(), refU)
        ok &= envok
        out["checks"].append({"check": "U re-trained here == predecessor U (environment)", "seed": k, "model_bitwise": envok,
                              "pass": envok})
        for arm in NEW_ARMS:
            model, diag, _ = PT.train_arm(arm, 0.0, st, d_in, KS, data, k, budgets)
            same_model = _state_equal(model.state_dict(), refU)
            rel, maps, heads, meta = FN.finalize_neural(model, PT.arm_spec(arm)["erasure"], D["X"], D, KS)
            diffs = {key: float(np.abs(rel[key] - relU[key]).max()) for key in ("r1", "r2", "c1", "c2", "p1", "p2")}
            hard = all(np.array_equal(rel[f"hard{i}"], relU[f"hard{i}"]) for i in (1, 2))
            passed = same_model and max(diffs.values()) <= 1e-12 and hard and not maps
            ok &= passed
            out["checks"].append({"check": f"{arm} beta=0 == U", "seed": k, "model_bitwise": same_model,
                                  "release_max_abs_diff": diffs, "release_bitwise": max(diffs.values()) == 0.0,
                                  "hard_equal": hard, "identity_maps": not maps,
                                  "protection_steps": diag["protection_steps_attempted"], "pass": passed})
            event("parity", seed=k, arm=arm, passed=passed)
    # fidelity of the copied loop: JP beta 1, seed 0 (with erasure) reproduces the predecessor unit
    k = 0
    data.seed = k
    st, budgets = warm(k)
    model, diag, _ = PT.train_arm("JP", 1.0, st, d_in, KS, data, k, budgets)
    same = _state_equal(model.state_dict(), torch.load(PRED / "nn__s0__JP__b1" / "model.pt"))
    ok &= same
    out["checks"].append({"check": "copied loop reproduces predecessor JP beta=1 seed 0", "model_bitwise": same, "pass": same})
    out["all_pass"] = bool(ok)
    out["tolerance"] = "model state bitwise equal; release max abs diff <= 1e-12; hard decisions identical; no LEACE map"
    out["on_failure"] = ("nonzero fits refuse to start; an environment failure (U != predecessor U) is repaired as an "
                         "environment issue; a beta=0 failure with environment parity is repaired in routing, "
                         "initialisation, randomness or finalisation; every failing receipt is kept")
    name = "parity__" + "_".join(map(str, seeds))
    tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / f"PARITY_{tag}.json").write_text(json.dumps(out, indent=1))      # every attempt kept
    if ok:
        FN.save_unit(U(name), {}, out)
    event("parity done", all_pass=bool(ok))
    return out


def parity_passed():
    return any(FN.unit_complete(d) and json.loads((d / "record.json").read_text())["all_pass"]
               for d in UNITS.glob("parity__*") if d.is_dir())


# ------------------------------------------------------------------ nonzero fits
def stage_train(D, k):
    if not parity_passed():
        raise SystemExit("REFUSED: beta=0 parity has not passed")
    data = JR.tensors(D)
    data.seed = k
    st, budgets = warm(k)
    d_in = D["X"].shape[1]
    for arm in NEW_ARMS:
        for beta in BETAS:
            name = unit_name(k, arm, beta)
            if done(name):
                continue
            t0, c0 = time.time(), time.process_time()
            model, diag, crit = PT.train_arm(arm, beta, st, d_in, KS, data, k, budgets)
            rescue = None
            if diag["nonfinite"] > 0:   # registered once-only engineering retry (nonfinite training only)
                rescue = {"reason": "nonfinite", "first_attempt": diag}
                model, diag, crit = PT.train_arm(arm, beta, st, d_in, KS, data, k, budgets, lr=PT.HP["sgd_lr"] / 2)
            diag["rescue"] = rescue
            out, maps, heads, meta = FN.finalize_neural(model, False, D["X"], D, KS)
            assert not maps, "a finaliser reinserted a map"
            files = {"model.pt": lambda p: torch.save(model.state_dict(), p),
                     "critics_final.pt": lambda p: torch.save(crit, p),
                     "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out)}
            for i, h in heads.items():
                files[f"head_{i}.joblib"] = (lambda p, h=h: joblib.dump(h, p))
            rec = {"unit": name, "arm": arm, "beta": beta, "seed": k, "erasure": False, "maps": "identity",
                   "diag": diag, "finalize": meta, "cpu_s": time.process_time() - c0, "wall_s": time.time() - t0,
                   "params": model.n_params(), "warm_start": f"predecessor warm__s{k} (hash-checked)"}
            FN.save_unit(U(name), files, rec)
            event("unit complete", unit=name)


# ------------------------------------------------------------------ reuse
def alias(name, src_name, why):
    d = U(name)
    src = PRED / src_name
    if done(name):
        return
    if not FN.unit_complete(src):
        raise SystemExit(f"REFUSED: predecessor unit {src_name} not hash-complete")
    d.mkdir(parents=True, exist_ok=True)
    (d / "ALIAS.json").write_text(json.dumps({"source": "~/" + str(src.relative_to(HOME)), "why": why,
                                              "source_COMPLETE_sha256": hashlib.sha256((src / "COMPLETE.json").read_bytes()).hexdigest()},
                                             indent=1))
    (d / "COMPLETE.json").write_text(json.dumps({"id": name, "files": {"ALIAS.json": hashlib.sha256((d / "ALIAS.json").read_bytes()).hexdigest()},
                                                 "alias": True}, indent=1))


def stage_alias(seeds):
    for k in seeds:
        for nm in [f"warm__s{k}", f"nn__s{k}__U", f"nn__s{k}__E"] + [f"nn__s{k}__JP__b{b:g}" for b in BETAS]:
            alias(nm, nm, "identical inputs, code and deployment (predecessor unit)")
            if not nm.startswith("warm"):
                alias(f"inner__{nm}", f"inner__{nm}", "identical inner audit (predecessor)")
        for i in (0, 1):
            for c in range(1, 7):
                nm = f"fare__s{k}__p{i}__c{c}"
                alias(nm, nm, "official FARE grid tree (predecessor)")
                alias(f"inner__{nm}", f"inner__{nm}", "identical inner audit (predecessor)")
            alias(f"fare__s{k}__p{i}__Z1", f"fare__s{k}__p{i}__Z1", "F0 frozen: zero-fairness twin of grid config 1 (predecessor)")
        for a in ("F", "F0"):
            alias(f"inner__pair__s{k}__{a}", f"inner__pair__s{k}__{a}", "identical inner coalition audit (predecessor)")
    event("aliases written", seeds=seeds)


# ------------------------------------------------------------------ views and inner audit
def release_views(name):
    z = np.load(udir(name) / "release.npz")
    v1, v2 = np.hstack([z["r1"], z["c1"]]), np.hstack([z["r2"], z["c2"]])
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2]),
            "out": {"p1": z["p1"], "p2": z["p2"], "hard1": z["hard1"], "hard2": z["hard2"]}}


def fare_pair_views(n1, n2):
    a, b = np.load(udir(n1) / "release.npz"), np.load(udir(n2) / "release.npz")
    v1, v2 = np.hstack([a["r"], a["c"]]), np.hstack([b["r"], b["c"]])
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2]),
            "out": {"p1": a["p"], "p2": b["p"], "hard1": a["hard"], "hard2": b["hard"]}}


def stage_inner(D, k):
    for arm in NEW_ARMS:
        for beta in BETAS:
            nm = unit_name(k, arm, beta)
            iname = f"inner__{nm}"
            if done(iname) or not done(nm):
                continue
            V = release_views(nm)
            rec = {"unit": iname, "of": nm, "recovery": JR.inner_recovery(V, D), "utility": JR.inner_utility(V["out"], D)}
            FN.save_unit(U(iname), {}, rec)
            event("unit complete", unit=iname)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--stage", required=True, choices=("parity", "train", "alias", "inner", "select"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    from pnx.lock import verify_lock
    v = verify_lock(Path(a.lock))
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    D = load_D()
    event(f"start {a.stage}", seeds=a.seeds, pid=os.getpid())
    if a.stage == "parity":
        stage_parity(D, a.seeds)
    elif a.stage == "alias":
        stage_alias(a.seeds)
    elif a.stage == "select":
        from pnx.select import run_selection
        run_selection(D, a.seeds)
    else:
        for k in a.seeds:
            (stage_train if a.stage == "train" else stage_inner)(D, k)
    event(f"end {a.stage}", seeds=a.seeds, pid=os.getpid())


if __name__ == "__main__":
    main()
