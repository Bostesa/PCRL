"""Packaging of a registered hcal release for research deployment (role A; prompt section 12). Reads saved artifacts only
(no fit, no selection): the admitted policy pair of the partition and, for a calibrated decoder, the cal__ unit's
decoder_table records, and writes an hcal.CalibratedDecoderPair (hcal.deploy) into the PRIVATE store. The decoder file
holds aggregate per-token calibration label counts and therefore stays private; public files cite it by sha256.

    PYTHONPATH=. <python> -m hcal.package <seed> <release id>
"""
from __future__ import annotations

import hashlib
import json
import sys

from hcal import ids as I


def policy_path(k, p):
    """The admitted policy.json of partition p (legacy: the D0 pol__ unit; lra: the D1 new__ unit)."""
    rid = I.release_id(p, "D0") if I.is_legacy(p) else I.release_id(p, "D1")
    return I.ADM_UNITS / I.lra_unit(k, rid) / "policy.json"


def decoder_path(k, rid):
    p, dec = I.parse_release(rid)
    if dec == "D1":
        return I.ADM_UNITS / I.lra_unit(k, rid) / "decoder.json"
    if dec in ("D0", "MEAN"):
        return None
    return I.PRIV / "package" / f"decoder__s{k}__{I.safe(rid)}.json"


def build(k, rid):
    """(policy path, decoder path or None, decoder sha256 or None) of a registered code release at seed k."""
    from hcal import deploy as DP
    from hcal import run as R
    from hcal import stages as ST
    from qpc import release as RL
    p, dec = I.parse_release(rid)
    pol = policy_path(k, p)
    path = decoder_path(k, rid)
    if path is None:
        return pol, None, None
    if dec == "D1":
        return pol, path, json.loads(path.read_text())["decoder_sha256"]
    pair = RL.load_policy(pol)
    tabs = json.loads((R.U(ST.cal_name(k, p)) / "tables.json").read_text())
    body = DP.decoder_pair_dict(rid, pair, tabs[f"{dec}|r1"], tabs[f"{dec}|r2"])
    for i, polx in ((1, pair.p1), (2, pair.p2)):                 # the decoder must reproduce from registered inputs
        import numpy as np
        ref = DP.reproduce_table(polx, body[f"r{i}"])
        if not np.array_equal(ref, np.asarray(body[f"r{i}"]["q"], dtype=np.float64).reshape(-1, polx.K)):
            raise SystemExit(f"REFUSED: {rid} s{k} recipient {i} table does not reproduce")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, allow_nan=False))
    tmp.rename(path)
    return pol, path, body["decoder_sha256"]


def sha_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == "__main__":
    k, rid = int(sys.argv[1]), sys.argv[2]
    pol, path, sha = build(k, rid)
    print(json.dumps({"seed": k, "release": rid, "policy_sha256": sha_file(pol),
                      "decoder_sha256": sha, "decoder": None if path is None else "<PRIVATE_CACHE>/hcal_v1/" +
                      str(path.relative_to(I.PRIV)) if str(path).startswith(str(I.PRIV)) else "<admitted lra decoder>"}))
