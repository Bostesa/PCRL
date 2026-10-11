"""Deployment of a registered hcal code release (frozen map + public decoder table) from the 83-column PERMITTED input
(role A; prompt section 12). Thin wrapper around qpc.deploy (teacher application, schema and input checks are
dpc.deploy's, imported unchanged) and lra.deploy (the admitted D1 decoder path).

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m hcal.deploy \\
        --unit <teacher unit dir> --policy <policy.json (qpc.PolicyPair)> \\
        [--decoder <decoder.json: hcal.CalibratedDecoderPair (H-TOKEN32 / H-GLOBAL-TEMP / H-CLASS-TEMP / T-TOKEN32) or
                    lra.DecoderPair (D1)>] [--decoder-sha256 <hex>] \\
        --X <input.npz with exactly X and feature_names> --schema <pinned schema> --out <release.npz> \\
        [--schema-sha256 <hex>] [--seed k]

Bindings (mandatory; no override flag): teacher model.pt sha256 and feature-schema sha256 (qpc.deploy.check_bindings);
the policy must be one of the 57 registered frozen partitions; the decoder must carry the SAME policy-pair fingerprint
and binding, its own sha256 (and --decoder-sha256 when given), and must REPRODUCE from public / registered inputs:
temperature tables are recomputed from the policy's frozen q0 (token_proto) and the stored alpha(s) and must match
bitwise; H-TOKEN32 tables are re-solved from the stored calibration counts and the policy's fixed teacher means
(token_S / token_n) and must match bitwise (fallback tokens must carry q0 exactly). Without --decoder the release uses
the frozen mean decoder (q0). Output npz: ONLY tokens_1, probs_1, decision_1, tokens_2, probs_2, decision_2; the
probability argmax must equal the preserved decision on every row (strict). Raw scores, fine IDs, teacher outputs and
debug fields are never written; flags asking for them, unknown flags, extra / reordered columns, mismatched teachers,
schemas or tables are refused (exit code 2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

from qpc import deploy as QD
from qpc import release as RL

from hcal import ids as I

ALLOWED_OUTPUT = QD.ALLOWED_OUTPUT
ALLOWED_FLAGS = ("--unit", "--policy", "--decoder", "--decoder-sha256", "--X", "--schema", "--schema-sha256", "--out",
                 "--seed", "--help", "-h")
FORBIDDEN_SUBSTRINGS = QD.FORBIDDEN_SUBSTRINGS
Refused = QD.Refused
KIND = "hcal.CalibratedDecoderPair"
SCHEMA = "hcal-decoder-v1"
FAMILIES = ("H-TOKEN32", "T-TOKEN32", "H-GLOBAL-TEMP", "H-CLASS-TEMP")


def refuse_forbidden_flags(argv):
    for tok in argv:
        if not tok.startswith("-"):
            continue
        name = tok.split("=", 1)[0]
        if name in ALLOWED_FLAGS:
            continue
        hit = [s for s in FORBIDDEN_SUBSTRINGS if s in name.lower()]
        if hit:
            raise Refused(f"refused: {name} would export a non-released field or bypass a binding ({', '.join(hit)})")
        raise Refused(f"refused: unknown flag {name}")


def partition_of(pair):
    cid = pair.config.get("config")
    p = cid[:-3] if isinstance(cid, str) and cid.endswith("|D1") else cid
    if p not in I.partitions():
        raise Refused(f"refused: policy configuration {cid!r} is not one of the 57 registered frozen partitions")
    if pair.config.get("teacher", "U") != "U":
        raise Refused("refused: hcal releases are bound to the frozen U teacher")
    return p


def decoder_hash(body):
    z = {k: v for k, v in body.items() if k != "decoder_sha256"}
    return hashlib.sha256(json.dumps(z, sort_keys=True, allow_nan=False).encode()).hexdigest()


def decoder_pair_dict(release_id, pair, tab1, tab2):
    """decoder.json body for a calibrated release: both recipients' hcal.calib.decoder_table records bound to the
    policy pair (and through it to the teacher model and the feature schema)."""
    p, dec = I.parse_release(release_id)
    if dec not in FAMILIES:
        raise ValueError(f"{release_id} is not a calibrated release")
    if partition_of(pair) != p:
        raise ValueError("decoder release id does not match the policy partition")
    body = {"kind": KIND, "schema": SCHEMA, "release_id": release_id, "family": dec,
            "policy_pair_fingerprint": pair.fingerprint(), "binding": RL.binding(pair), "r1": tab1, "r2": tab2}
    body["decoder_sha256"] = decoder_hash(body)
    return body


def _mu(pol):
    n = np.asarray(pol.token_n, dtype=np.int64)
    S = np.asarray(pol.token_S, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n[:, None] > 0, S / np.where(n > 0, n, 1)[:, None], np.nan), n


def reproduce_table(pol, rec):
    """The registered table of one recipient recomputed from public / registered inputs (module docstring)."""
    from hcal import calib as C
    q0 = np.asarray(pol.token_proto, dtype=np.float64)
    tc = np.asarray(pol.token_class, dtype=np.int64)
    K = int(pol.K)
    fam = rec["kind"]
    if rec.get("K") != K or rec.get("T") != int(pol.T):
        raise Refused("refused: decoder table shape differs from the policy")
    if list(rec["token_class"]) != tc.tolist():
        raise Refused("refused: decoder token classes differ from the policy")
    if fam == "H-GLOBAL-TEMP":
        return C.temp_apply_tokens(q0, rec["alpha"], tc)
    if fam == "H-CLASS-TEMP":
        return C.temp_apply_tokens_by_class(q0, rec["alphas"], tc)
    mu, n_fit = _mu(pol)
    y_cal = np.asarray(rec["y_cal"], dtype=np.int64).reshape(-1, K)
    n_cal = y_cal.sum(1)
    if not np.array_equal(n_cal, np.asarray(rec["n_cal"], dtype=np.int64)):
        raise Refused("refused: decoder calibration counts are inconsistent")
    q = q0.copy()
    fit = (n_fit > 0) & (n_cal > 0)
    if fit.any():
        _, Qs, _, _ = C.token32_solve(y_cal[fit].astype(np.float64), mu[fit], n_cal[fit], tc[fit])
        q[fit] = Qs
    return q


def load_decoder(path, pair, sha=None):
    try:
        with open(path) as fh:
            z = json.load(fh)
    except (OSError, ValueError) as e:
        raise Refused(f"refused: cannot read decoder ({e.__class__.__name__})")
    if z.get("kind") == "lra.DecoderPair":
        from lra import deploy as LD
        cid, d1, d2, h = LD.load_decoder(path, pair, sha)
        return "D1", (d1.q, d2.q), h, (d1, d2)
    if z.get("kind") != KIND or z.get("schema") != SCHEMA:
        raise Refused("refused: --decoder is neither an hcal.CalibratedDecoderPair nor an lra.DecoderPair")
    if z.get("decoder_sha256") != decoder_hash(z):
        raise Refused("refused: decoder sha256 mismatch")
    if sha is not None and z["decoder_sha256"] != sha:
        raise Refused("refused: decoder sha256 differs from --decoder-sha256")
    if z.get("policy_pair_fingerprint") != pair.fingerprint() or z.get("binding") != RL.binding(pair):
        raise Refused("refused: decoder is bound to a different policy pair / teacher / schema")
    if z.get("family") not in FAMILIES or I.parse_release(z["release_id"]) != (partition_of(pair), z["family"]):
        raise Refused("refused: decoder release id / family is not registered for this policy")
    tabs = []
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        rec = z[f"r{i}"]
        if rec.get("kind") != z["family"] or int(rec.get("recipient")) != i:
            raise Refused("refused: decoder recipient tables are inconsistent")
        q = np.asarray(rec["q"], dtype=np.float64).reshape(-1, pol.K)
        try:
            ref = reproduce_table(pol, rec)
        except (ValueError, ArithmeticError) as e:
            raise Refused(f"refused: decoder table does not reproduce ({e.__class__.__name__}: {e})")
        if not np.array_equal(ref, q):
            raise Refused("refused: decoder table does not reproduce bitwise from its registered inputs")
        tabs.append(q)
    return z["family"], tuple(tabs), z["decoder_sha256"], None


def release(unit_dir, pair, X, schema_hash, tables=None, d1=None, seed=None):
    """The protected release only: {tokens_i, probs_i, decision_i}; no teacher score leaves this function."""
    if pair.p1.K != QD.KS[0] or pair.p2.K != QD.KS[1] or pair.p1.recipient != 1 or pair.p2.recipient != 2:
        raise Refused("refused: policy pair must be (recipient 1 income K=2, recipient 2 occupation K=6)")
    model, heads, msha = QD.load_teacher(unit_dir, seed)
    QD.check_bindings(pair, msha, schema_hash)
    P = QD.teacher_probs(model, heads, X)
    out = {}
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        if d1 is not None:
            from lra import decoder as DC
            tok, probs, dec = DC.encode_d1(pol, d1[i - 1], P[i - 1])
        else:
            tok, q0rows, dec = RL.encode(pol, P[i - 1])
            probs = q0rows if tables is None else np.asarray(tables[i - 1], dtype=np.float64)[tok]
        rows = np.arange(len(dec))
        other = np.array(probs, copy=True)
        other[rows, dec] = -np.inf
        if len(dec) and not (np.array_equal(np.asarray(probs).argmax(1), dec) and np.all(probs[rows, dec] >
                                                                                          other.max(1))):
            raise Refused("refused: released probability argmax differs from the preserved decision")
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
    ap = argparse.ArgumentParser(prog="python -m hcal.deploy", allow_abbrev=False)
    for f in ("--unit", "--policy", "--X", "--schema", "--out"):
        ap.add_argument(f, required=True)
    ap.add_argument("--decoder", default=None)
    ap.add_argument("--decoder-sha256", default=None)
    ap.add_argument("--schema-sha256", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        pinned = QD.schema_names(a.schema)
        sh = QD.schema_sha256(pinned)
        if a.schema_sha256 and sh != a.schema_sha256:
            raise Refused("refused: pinned schema hash mismatch")
        X = QD.load_permitted_input(a.X, pinned)
        pair = QD.load_pair(a.policy)
        p = partition_of(pair)
        if a.decoder_sha256 and not a.decoder:
            raise Refused("refused: --decoder-sha256 without --decoder")
        fam, tables, dsha, d1 = "MEAN" if not I.is_legacy(p) else "D0", None, None, None
        if a.decoder:
            fam, tables, dsha, d1 = load_decoder(a.decoder, pair, a.decoder_sha256)
        elif not I.is_legacy(p) and pair.config.get("config", "").endswith("|D1"):
            fam = "MEAN"
        out, info = release(a.unit, pair, X, sh, tables if d1 is None else None, d1, a.seed)
        QD.write_release(a.out, out)
    except Refused as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"rows": int(X.shape[0]), "policy_pair_fingerprint": pair.fingerprint(), "partition": p,
                      "decoder": fam, "release_id": I.release_id(p, fam), "decoder_sha256": dsha, "registered": True,
                      "alphabets": [pair.p1.T, pair.p2.T], "binding": info["binding"], "written": list(ALLOWED_OUTPUT)}))


if __name__ == "__main__":
    main()
