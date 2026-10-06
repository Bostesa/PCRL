"""Deployment of a fitted class-preserving code pair from the 83-column PERMITTED input.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.deploy \\
        --unit <release unit dir: model.pt, head_0.joblib, head_1.joblib, record.json, COMPLETE.json> \\
        --policy <policy_pair.json> --X <input.npz with arrays X and feature_names> \\
        --schema <pinned schema: source npz with feature_names, or JSON list> --out <release.npz> [--seed k]

The input npz must contain exactly the arrays X (n, 83) float and feature_names (83,) str, with feature_names equal
to the pinned schema in the same order: missing, extra or reordered columns and any extra array are refused.
The frozen teacher is the unit's jcv.train.Model(83, [2, 6], seed) encoder (float32 input, outputs cast to float64)
followed by the deployed StandardScaler+LogisticRegression head: p_i = head_i.predict_proba(encode(i, X)), d_i =
argmax p_i (first-index ties), exactly as rgj/finalize.py and osf/deploy.py. The policy pair encodes (p_i, d_i).

The output npz contains ONLY tokens_1, probs_1, decision_1, tokens_2, probs_2, decision_2. Fine-cell IDs, teacher
probabilities or scores, logits, latent features and assignment distances are never written; flags asking for them
are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from dpc.release import PolicyPair, encode, load_policy

N_PERMITTED = 83
KS = (2, 6)
ALLOWED_OUTPUT = ("tokens_1", "probs_1", "decision_1", "tokens_2", "probs_2", "decision_2")
FORBIDDEN_SUBSTRINGS = ("fine", "cell", "raw", "score", "logit", "latent", "feature", "distance", "debug", "export",
                        "teacher", "centr", "embed", "repr", "hidden", "intermediate", "dump", "include", "extra",
                        "sex", "label")
ALLOWED_FLAGS = ("--unit", "--policy", "--X", "--schema", "--schema-sha256", "--out", "--seed",
                 "--allow-unbound-policy", "--help", "-h")


class Refused(Exception):
    pass


def refuse_forbidden_flags(argv):
    """Refuse any flag that is not in the fixed allow-list, naming forbidden export attempts explicitly."""
    for tok in argv:
        if not tok.startswith("-"):
            continue
        name = tok.split("=", 1)[0]
        if name in ALLOWED_FLAGS:
            continue
        low = name.lower()
        hit = [s for s in FORBIDDEN_SUBSTRINGS if s in low]
        if hit:
            raise Refused(f"refused: {name} would export a non-released field ({', '.join(hit)}); the protected "
                          f"interface is token id, decoded probabilities and decision only")
        raise Refused(f"refused: unknown flag {name}")


def schema_names(path):
    p = Path(path)
    if p.suffix == ".json":
        z = json.loads(p.read_text())
        names = z["feature_names"] if isinstance(z, dict) else z
    else:
        with np.load(p, allow_pickle=False) as z:
            if "feature_names" not in z.files:
                raise Refused(f"schema {p.name} has no feature_names array")
            names = z["feature_names"].tolist()
    names = [str(x) for x in names]
    if len(names) != N_PERMITTED or len(set(names)) != N_PERMITTED:
        raise Refused(f"pinned schema must list {N_PERMITTED} distinct feature names; got {len(names)}")
    return names


def schema_sha256(names):
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def check_schema(names, pinned):
    names = [str(x) for x in names]
    if names == pinned:
        return
    missing = [x for x in pinned if x not in names]
    extra = [x for x in names if x not in pinned]
    if missing or extra:
        raise Refused(f"refused: input schema differs from the pinned schema (missing {missing[:5]}"
                      f"{'...' if len(missing) > 5 else ''}, extra {extra[:5]}{'...' if len(extra) > 5 else ''})")
    first = next(i for i, (a, b) in enumerate(zip(names, pinned)) if a != b)
    raise Refused(f"refused: input columns are reordered relative to the pinned schema (first difference at column "
                  f"{first}: {names[first]!r} vs pinned {pinned[first]!r})")


def load_permitted_input(path, pinned):
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        if files != {"X", "feature_names"}:
            raise Refused(f"refused: input npz must contain exactly X and feature_names; got {sorted(files)}")
        X = z["X"]
        names = z["feature_names"].tolist()
    if X.ndim != 2:
        raise Refused(f"refused: X must be 2-D; got shape {X.shape}")
    if X.shape[1] != len(names):
        raise Refused(f"refused: X has {X.shape[1]} columns but {len(names)} feature names")
    if X.shape[1] != N_PERMITTED:
        raise Refused(f"refused: deployment accepts exactly the {N_PERMITTED} permitted columns; got {X.shape[1]}")
    check_schema(names, pinned)
    if not np.issubdtype(X.dtype, np.floating) or not np.all(np.isfinite(X)):
        raise Refused("refused: X must be the finite preprocessed float matrix")
    return X.astype(np.float32)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_teacher(unit_dir, seed=None):
    import joblib
    import torch

    from jcv.finalize import unit_complete
    from jcv.train import Model
    unit_dir = Path(unit_dir)
    if not unit_complete(unit_dir):
        raise Refused(f"refused: {unit_dir.name} is not hash-complete (COMPLETE.json)")
    rec = json.loads((unit_dir / "record.json").read_text())
    if seed is None:
        if "seed" not in rec:
            raise Refused("unit record has no seed; pass --seed")
        seed = int(rec["seed"])
    model = Model(N_PERMITTED, list(KS), int(seed))
    model.load_state_dict(torch.load(unit_dir / "model.pt"))
    model.eval()
    heads = [joblib.load(unit_dir / f"head_{i}.joblib") for i in (0, 1)]
    return model, heads, sha_file(unit_dir / "model.pt")


def teacher_probs(model, heads, X):
    """p_i = head_i.predict_proba(encode(i, X) as float64) for i = 0, 1 (rgj/finalize.py convention)."""
    import torch
    with torch.no_grad():
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        R = [model.encode(i, Xt).double().numpy() for i in (0, 1)]
    return [np.asarray(heads[i].predict_proba(R[i]), dtype=np.float64) for i in (0, 1)]


def check_binding(pair: PolicyPair, model_sha, allow_unbound=False):
    bound = pair.config.get("teacher_model_sha256") or pair.p1.meta.get("teacher_model_sha256")
    if bound is None:
        if not allow_unbound:
            raise Refused("refused: policy is not bound to a teacher (config teacher_model_sha256); pass "
                          "--allow-unbound-policy only for a verified pairing")
        return "UNBOUND"
    if bound != model_sha:
        raise Refused("refused: policy was fitted for a different teacher model.pt")
    return "BOUND"


def release(unit_dir, pair: PolicyPair, X, seed=None, allow_unbound=False):
    """The protected release only: {tokens_i, probs_i, decision_i}. No teacher score leaves this function."""
    if pair.p1.K != KS[0] or pair.p2.K != KS[1] or pair.p1.recipient != 1 or pair.p2.recipient != 2:
        raise Refused("refused: policy pair must be (recipient 1 income K=2, recipient 2 occupation K=6)")
    model, heads, msha = load_teacher(unit_dir, seed)
    status = check_binding(pair, msha, allow_unbound)
    P = teacher_probs(model, heads, X)
    out = {}
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        tok, probs, dec = encode(pol, P[i - 1])
        out[f"tokens_{i}"], out[f"probs_{i}"], out[f"decision_{i}"] = tok, probs, dec
    del P
    return out, {"binding": status, "model_sha256": msha}


def write_release(path, out):
    keys = set(out)
    if keys != set(ALLOWED_OUTPUT):
        raise Refused(f"refused: release may contain only {list(ALLOWED_OUTPUT)}; got {sorted(keys)}")
    for i in (1, 2):
        t, p, d = out[f"tokens_{i}"], out[f"probs_{i}"], out[f"decision_{i}"]
        if t.ndim != 1 or not np.issubdtype(t.dtype, np.integer):
            raise Refused("tokens must be a 1-D integer array")
        if p.ndim != 2 or p.shape != (t.shape[0], KS[i - 1]) or p.dtype != np.float64:
            raise Refused("probs must be the (n, K) float64 decoded prototypes")
        if d.shape != t.shape or not np.issubdtype(d.dtype, np.integer):
            raise Refused("decisions must be a 1-D integer array")
    np.savez_compressed(path, **{k: out[k] for k in ALLOWED_OUTPUT})


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        refuse_forbidden_flags(argv)
    except Refused as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
    ap = argparse.ArgumentParser(prog="python -m dpc.deploy", allow_abbrev=False)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--X", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--schema-sha256", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--allow-unbound-policy", action="store_true")
    a = ap.parse_args(argv)
    try:
        pinned = schema_names(a.schema)
        if a.schema_sha256 and schema_sha256(pinned) != a.schema_sha256:
            raise Refused("refused: pinned schema hash mismatch")
        X = load_permitted_input(a.X, pinned)
        pair = load_policy(a.policy)
        if not isinstance(pair, PolicyPair):
            raise Refused("refused: --policy must be a dpc.PolicyPair")
        bound_schema = pair.config.get("feature_names_sha256")
        if bound_schema is not None and bound_schema != schema_sha256(pinned):
            raise Refused("refused: policy was fitted under a different pinned feature schema")
        out, info = release(a.unit, pair, X, a.seed, a.allow_unbound_policy)
        write_release(a.out, out)
    except Refused as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"rows": int(X.shape[0]), "policy_pair_fingerprint": pair.fingerprint(),
                      "alphabets": [pair.p1.T, pair.p2.T], "binding": info["binding"],
                      "written": list(ALLOWED_OUTPUT)}))


if __name__ == "__main__":
    main()
