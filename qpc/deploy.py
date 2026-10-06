"""Deployment of a fitted qpc class-preserving code pair from the 83-column PERMITTED input.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m qpc.deploy \\
        --unit <teacher unit dir: model.pt, head_0.joblib, head_1.joblib, record.json, COMPLETE.json> \\
        --policy <policy.json (qpc.PolicyPair)> --X <input.npz with exactly X and feature_names> \\
        --schema <pinned schema: npz with feature_names, or JSON list> --out <release.npz> \\
        [--schema-sha256 <hex>] [--seed k]

Provenance: adapted from dpc/deploy.py at source SHA 0a7b05a5; the teacher forward application
(dpc.deploy.load_teacher / teacher_probs: jcv.train.Model(83, [2, 6], seed), float32 input, float64 encoder output,
StandardScaler+LogisticRegression heads, d = argmax p with first-index ties) and the schema/input checks are imported
unchanged. qpc differences: only qpc.PolicyPair records are accepted (a dpc policy is refused), BOTH bindings are
mandatory (teacher model.pt sha256 and feature-schema sha256; there is no unbound override), and the flag allow-list
has no override flag.

Output: ONLY tokens_1, probs_1, decision_1, tokens_2, probs_2, decision_2. Fine-cell IDs, teacher probabilities or
scores, logits, latent features and distances are never written; flags asking for them, unknown flags, reordered,
renamed, missing or extra columns, extra input arrays and mismatched teachers or schemas are refused (exit code 2).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from dpc import deploy as DD
from qpc import release as RL

N_PERMITTED = DD.N_PERMITTED
KS = DD.KS
ALLOWED_OUTPUT = DD.ALLOWED_OUTPUT
FORBIDDEN_SUBSTRINGS = DD.FORBIDDEN_SUBSTRINGS + ("unbound", "allow", "prob-source", "continuous")
ALLOWED_FLAGS = ("--unit", "--policy", "--X", "--schema", "--schema-sha256", "--out", "--seed", "--help", "-h")
Refused = DD.Refused
schema_names = DD.schema_names
schema_sha256 = DD.schema_sha256
load_permitted_input = DD.load_permitted_input
load_teacher = DD.load_teacher
teacher_probs = DD.teacher_probs
write_release = DD.write_release


def refuse_forbidden_flags(argv):
    for tok in argv:
        if not tok.startswith("-"):
            continue
        name = tok.split("=", 1)[0]
        if name in ALLOWED_FLAGS:
            continue
        hit = [s for s in FORBIDDEN_SUBSTRINGS if s in name.lower()]
        if hit:
            raise Refused(f"refused: {name} would export a non-released field or bypass a binding "
                          f"({', '.join(hit)}); the protected interface is token, decoded probabilities and decision")
        raise Refused(f"refused: unknown flag {name}")


def load_pair(path):
    try:
        with open(path) as fh:
            z = json.load(fh)
    except (OSError, ValueError) as e:
        raise Refused(f"refused: cannot read policy ({e.__class__.__name__})")
    if z.get("kind") != "qpc.PolicyPair":
        raise Refused(f"refused: --policy must be a qpc.PolicyPair (got kind {z.get('kind')!r})")
    try:
        return RL.PolicyPair.from_dict(z)
    except ValueError as e:
        raise Refused(f"refused: policy failed its integrity check ({e})")


def check_bindings(pair, model_sha, schema_hash):
    try:
        b = RL.check_bound(pair)
    except ValueError as e:
        raise Refused(f"refused: policy is not bound ({e})")
    if b["teacher_model_sha256"] != model_sha:
        raise Refused("refused: policy was fitted for a different teacher model.pt")
    if b["feature_names_sha256"] != schema_hash:
        raise Refused("refused: policy was fitted under a different pinned feature schema")
    return b


def release(unit_dir, pair, X, schema_hash, seed=None):
    """The protected release only: {tokens_i, probs_i, decision_i}. No teacher score leaves this function."""
    if pair.p1.K != KS[0] or pair.p2.K != KS[1] or pair.p1.recipient != 1 or pair.p2.recipient != 2:
        raise Refused("refused: policy pair must be (recipient 1 income K=2, recipient 2 occupation K=6)")
    model, heads, msha = load_teacher(unit_dir, seed)
    check_bindings(pair, msha, schema_hash)
    P = teacher_probs(model, heads, X)
    out = {}
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        tok, probs, dec = RL.encode(pol, P[i - 1])
        out[f"tokens_{i}"], out[f"probs_{i}"], out[f"decision_{i}"] = tok, probs, dec
    del P
    return out, {"binding": "BOUND", "model_sha256": msha}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        refuse_forbidden_flags(argv)
    except Refused as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
    ap = argparse.ArgumentParser(prog="python -m qpc.deploy", allow_abbrev=False)
    for f in ("--unit", "--policy", "--X", "--schema", "--out"):
        ap.add_argument(f, required=True)
    ap.add_argument("--schema-sha256", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        pinned = schema_names(a.schema)
        sh = schema_sha256(pinned)
        if a.schema_sha256 and sh != a.schema_sha256:
            raise Refused("refused: pinned schema hash mismatch")
        X = load_permitted_input(a.X, pinned)
        pair = load_pair(a.policy)
        out, info = release(a.unit, pair, X, sh, a.seed)
        write_release(a.out, out)
    except Refused as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"rows": int(X.shape[0]), "policy_pair_fingerprint": pair.fingerprint(),
                      "config": pair.config.get("config"), "alphabets": [pair.p1.T, pair.p2.T],
                      "binding": info["binding"], "written": list(ALLOWED_OUTPUT)}))


if __name__ == "__main__":
    main()
