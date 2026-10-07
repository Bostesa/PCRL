"""Deployment of a registered lcr code release (D0 map, or map + D1 learned decoder) from the 83-column PERMITTED input
(role B; prompt sec. 15). Thin wrapper around qpc.deploy (whose teacher application, schema and input checks are
dpc.deploy's, imported unchanged).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lcr.deploy \\
        --unit <teacher unit dir: model.pt, head_0.joblib, head_1.joblib, record.json, COMPLETE.json> \\
        --policy <policy.json (qpc.PolicyPair: the token MAP)> [--decoder <decoder.json (lcr.DecoderPair)>] \\
        --X <input.npz with exactly X and feature_names> --schema <pinned schema: npz with feature_names, or JSON list> \\
        --out <release.npz> [--decoder-sha256 <hex>] [--schema-sha256 <hex>] [--seed k]

Bindings (all mandatory, no override flag):
  * teacher: the policy's teacher_model_sha256 must equal the unit's model.pt sha256 (qpc.deploy.check_bindings);
  * schema: the policy's feature_names_sha256 must equal the pinned schema's (missing / extra / renamed / reordered
    columns and any extra input array are refused by dpc.deploy.load_permitted_input);
  * policy: qpc.PolicyPair integrity (fingerprints, recomputed prototypes) and a REGISTERED lcr configuration;
  * decoder (D1 configurations): lcr.decoder.load_decoder_pair verifies its own sha256, re-solves every supervised
    token from its stored sufficient statistics (bitwise), and requires the SAME policy-pair fingerprint and the same
    teacher/schema binding as the policy; --decoder-sha256, when given, must match.
Registration (lcr.run ids): a policy whose config is a D0 id (lcr.run.d0_ids(), checked as cbp's registered bank) is
deployable WITHOUT a decoder (its released vectors are the pinned D0 means) or WITH a decoder whose config is that id
+ "|D1" (fixed-map calibration control). A policy whose config is a new-fit id (C-TASK, W-*, K-*) requires a decoder
with the same config. Anything else -- unregistered ids, a D1 id without a decoder, a decoder for a D0-only id, a
family / caps / lambda inconsistent with the id -- is refused.

Output npz: ONLY tokens_1, probs_1, decision_1, tokens_2, probs_2, decision_2 (per recipient: canonical token ID,
released decoded probability vector, unchanged teacher decision). Fine-cell IDs, teacher probabilities or scores,
logits, latent features, distances and debug fields are never written; flags asking for them and unknown flags are
refused. Every refusal exits with code 2.
"""
from __future__ import annotations

import argparse
import json
import sys

from qpc import deploy as QD
from qpc import release as RL

ALLOWED_OUTPUT = QD.ALLOWED_OUTPUT
ALLOWED_FLAGS = ("--unit", "--policy", "--decoder", "--decoder-sha256", "--X", "--schema", "--schema-sha256", "--out",
                 "--seed", "--help", "-h")
FORBIDDEN_SUBSTRINGS = QD.FORBIDDEN_SUBSTRINGS
Refused = QD.Refused
schema_names = QD.schema_names
schema_sha256 = QD.schema_sha256
load_permitted_input = QD.load_permitted_input
load_pair = QD.load_pair
write_release = QD.write_release
M1, M2 = 8, 64


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


def check_registered(pair, decoder_cid=None):
    """The deployable configuration (lcr id) of a policy (+ optional decoder), or Refused."""
    from cbp import deploy as CD
    from lcr import run as R
    pcid = pair.config.get("config")
    if pair.config.get("teacher", "U") != "U":
        raise Refused("refused: lcr releases are bound to the frozen U teacher")
    if pcid in R.d0_ids():
        try:
            CD.check_registered(pair)                       # family / caps / lambda consistent with the D0 id
        except Refused as e:
            raise Refused(str(e).replace("cbp", "lcr D0"))
        if decoder_cid is None:
            return pcid
        if decoder_cid != pcid + "|D1" or decoder_cid not in R.d1_fixed_ids():
            raise Refused(f"refused: decoder configuration {decoder_cid!r} is not the registered D1 control of "
                          f"{pcid!r}")
        return decoder_cid
    if pcid not in R.new_fit_ids():
        raise Refused(f"refused: policy configuration {pcid!r} is not a registered lcr configuration")
    p = R.parse_id(pcid)
    if pair.family != p["family"] or (pair.config.get("m1"), pair.config.get("m2")) != (M1, M2) or \
            pair.config.get("lam") != p["lam"]:
        raise Refused(f"refused: policy family/caps/lambda are inconsistent with {pcid!r}")
    if decoder_cid is None:
        raise Refused(f"refused: {pcid!r} is a D1 configuration; its decoder.json is required (--decoder)")
    if decoder_cid != pcid:
        raise Refused(f"refused: decoder configuration {decoder_cid!r} differs from the policy's {pcid!r}")
    return pcid


def load_decoder(path, pair, sha=None):
    from lcr import decoder as DC
    try:
        cid, d1, d2, h = DC.load_decoder_pair(path, pair, verify_solve=True)
    except (OSError, ValueError, KeyError) as e:
        raise Refused(f"refused: decoder failed its integrity/binding check ({e.__class__.__name__}: {e})")
    if sha is not None and h != sha:
        raise Refused("refused: decoder sha256 differs from --decoder-sha256")
    return cid, d1, d2, h


def release(unit_dir, pair, X, schema_hash, decoders=None, seed=None):
    """The protected release only: {tokens_i, probs_i, decision_i}; no teacher score leaves this function."""
    from lcr import decoder as DC
    if pair.p1.K != QD.KS[0] or pair.p2.K != QD.KS[1] or pair.p1.recipient != 1 or pair.p2.recipient != 2:
        raise Refused("refused: policy pair must be (recipient 1 income K=2, recipient 2 occupation K=6)")
    model, heads, msha = QD.load_teacher(unit_dir, seed)
    QD.check_bindings(pair, msha, schema_hash)
    P = QD.teacher_probs(model, heads, X)
    out = {}
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        if decoders is None:
            tok, probs, dec = RL.encode(pol, P[i - 1])
        else:
            tok, probs, dec = DC.encode_d1(pol, decoders[i - 1], P[i - 1])
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
    ap = argparse.ArgumentParser(prog="python -m lcr.deploy", allow_abbrev=False)
    for f in ("--unit", "--policy", "--X", "--schema", "--out"):
        ap.add_argument(f, required=True)
    ap.add_argument("--decoder", default=None)
    ap.add_argument("--decoder-sha256", default=None)
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
        if a.decoder_sha256 and not a.decoder:
            raise Refused("refused: --decoder-sha256 without --decoder")
        decs, dsha, dcid = None, None, None
        if a.decoder:
            dcid, d1, d2, dsha = load_decoder(a.decoder, pair, a.decoder_sha256)
            decs = (d1, d2)
        cid = check_registered(pair, dcid)
        out, info = release(a.unit, pair, X, sh, decs, a.seed)
        write_release(a.out, out)
    except Refused as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"rows": int(X.shape[0]), "policy_pair_fingerprint": pair.fingerprint(), "config": cid,
                      "decoder": "D1" if decs else "D0", "decoder_sha256": dsha, "registered": True,
                      "alphabets": [pair.p1.T, pair.p2.T], "binding": info["binding"],
                      "written": list(ALLOWED_OUTPUT)}))


if __name__ == "__main__":
    main()
