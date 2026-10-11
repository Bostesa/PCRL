"""Deployment of a cbp-registered class-preserving code pair (thin wrapper around qpc.deploy; role B).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m cbp.deploy \\
        --unit <teacher unit dir: model.pt, head_0.joblib, head_1.joblib, record.json, COMPLETE.json> \\
        --policy <policy.json (qpc.PolicyPair)> --X <input.npz with exactly X and feature_names> \\
        --schema <pinned schema: npz with feature_names, or JSON list> --out <release.npz> \\
        [--schema-sha256 <hex>] [--seed k]

Everything is qpc.deploy at d0c8a45 (imported unchanged; its teacher forward application, schema and input checks are
dpc.deploy's): the same flag allow-list and refusals (unknown flags; flags asking for fine/cell IDs, raw or teacher
scores, logits, features, latent, distances, debug, export, include/extra, SEX or labels, allow/unbound, continuous),
the same 83-column schema check (missing, extra, renamed or reordered columns and any extra input array are refused),
the same mandatory bindings (teacher model.pt sha256 and feature-schema sha256; no unbound override), the same
qpc.PolicyPair integrity checks and the same output writer.

The ONE cbp addition: the policy's configuration must be a registered cbp configuration (cbp.fit.registered_ids():
U|{LOCAL,SEQ-12,SEQ-21,JOINT}|i8o64|l{0.01,0.025,0.04,0.06,0.08,0.1}, U|DIRECT-TASK|i8o64, U|FINE-TASK|i8o64,
U|CLASS|i1o1), consistent with its family, caps and lambda; anything else is refused.

Output npz: ONLY tokens_1, probs_1, decision_1, tokens_2, probs_2, decision_2 (per recipient: categorical token,
decoded codebook probability vector, unchanged teacher decision). Every refusal exits with code 2.
"""
from __future__ import annotations

import argparse
import json
import sys

from cbp import fit as FT
from qpc import deploy as QD

ALLOWED_OUTPUT = QD.ALLOWED_OUTPUT
ALLOWED_FLAGS = QD.ALLOWED_FLAGS
Refused = QD.Refused
refuse_forbidden_flags = QD.refuse_forbidden_flags
schema_names = QD.schema_names
schema_sha256 = QD.schema_sha256
load_permitted_input = QD.load_permitted_input
load_pair = QD.load_pair
write_release = QD.write_release


def check_registered(pair):
    """Refuse any policy pair whose configuration is not in the registered cbp bank (or inconsistent with itself)."""
    cid = pair.config.get("config")
    try:
        own = FT.config_id(pair.family, pair.config.get("lam"))
    except ValueError as e:
        raise Refused(f"refused: policy family/lambda is not a cbp configuration ({e})")
    m = (1, 1) if pair.family == "CLASS" else (FT.M1, FT.M2)
    if cid != own or (pair.config.get("m1"), pair.config.get("m2")) != m or cid not in FT.registered_ids():
        raise Refused(f"refused: policy configuration {cid!r} is not a registered cbp configuration")
    if pair.config.get("teacher", FT.TEACHER) != FT.TEACHER:
        raise Refused("refused: cbp policies are bound to the frozen U teacher")
    return cid


def release(unit_dir, pair, X, schema_hash, seed=None):
    """qpc.deploy.release after the cbp registration check: {tokens_i, probs_i, decision_i} only."""
    check_registered(pair)
    return QD.release(unit_dir, pair, X, schema_hash, seed)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        refuse_forbidden_flags(argv)
    except Refused as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
    ap = argparse.ArgumentParser(prog="python -m cbp.deploy", allow_abbrev=False)
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
        cid = check_registered(pair)
        out, info = QD.release(a.unit, pair, X, sh, a.seed)
        write_release(a.out, out)
    except Refused as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"rows": int(X.shape[0]), "policy_pair_fingerprint": pair.fingerprint(), "config": cid,
                      "registered": True, "alphabets": [pair.p1.T, pair.p2.T], "binding": info["binding"],
                      "written": list(ALLOWED_OUTPUT)}))


if __name__ == "__main__":
    main()
