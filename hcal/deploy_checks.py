"""Deployment checks of the held-out calibration study on the admitted 83-column input (role A; prompt section 12).
Runs the real `python -m hcal.deploy` CLI as subprocesses, compares outputs bitwise with the stored release arrays
(hcal.stages.release_arrays over all rows; label-free), and exercises the registered refusals. Writes outputs under
<PRIVATE_CACHE>/hcal_v1/run/deploy_test/ and a public receipt (no paths, no per-person values).

    PYTHONPATH=. <python> -m hcal.deploy_checks <seed> <release id> [<release id> ...] [--root <copy root>]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from hcal import ids as I


def _cli(args):
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=".")
    c = subprocess.run([sys.executable, "-m", "hcal.deploy"] + args, capture_output=True, text=True, env=env,
                       cwd=str(I.WT))
    out = (c.stdout.strip().splitlines() or [""])[-1][:500]
    err = (c.stderr.strip().splitlines() or [""])[-1][:300]
    return {"rc": c.returncode, "stdout": out, "stderr": err.replace(str(Path.home()), "<HOME>")}


def run(k, rids, root=None):
    from hcal import package as PK
    from hcal import stages as ST
    adm = Path(root) / "admitted" if root else I.ADM
    out = I.RUN / "deploy_test"
    out.mkdir(parents=True, exist_ok=True)
    X, SCH = adm / "inputs" / "deploy_input.npz", adm / "inputs" / "schema.json"
    unit = adm / f"rel__s{k}__U"
    rec = {"schema": "hcal-deployment-receipt-v1", "seed": k, "store": "copy" if root else "store",
           "deployments": {}, "refusals": {}}
    first = None
    for rid in rids:
        pol, dec, sha = PK.build(k, rid)
        if root:
            pol = Path(root) / pol.relative_to(I.PRIV)
            dec = None if dec is None else Path(root) / dec.relative_to(I.PRIV)
        args = ["--unit", str(unit), "--policy", str(pol), "--X", str(X), "--schema", str(SCH),
                "--out", str(out / f"{I.safe(rid)}.npz")]
        if dec is not None:
            args += ["--decoder", str(dec), "--decoder-sha256", sha]
        r = _cli(args)
        if r["rc"] == 0:
            z = np.load(out / f"{I.safe(rid)}.npz")
            tok, q, hard = ST.release_arrays(k, rid)
            r["written"] = sorted(z.files)
            r["bitwise_equal_to_registered_release"] = {
                "tokens": all(bool(np.array_equal(z[f"tokens_{i}"], tok[i])) for i in (1, 2)),
                "probs": all(bool(np.array_equal(z[f"probs_{i}"], q[i])) for i in (1, 2)),
                "decisions": all(bool(np.array_equal(z[f"decision_{i}"], hard[i])) for i in (1, 2))}
            r["argmax_equals_decision"] = all(bool(np.array_equal(z[f"probs_{i}"].argmax(1), z[f"decision_{i}"]))
                                              for i in (1, 2))
        r["decoder_sha256"] = sha
        rec["deployments"][rid] = r
        if first is None and dec is not None and rid.split("|")[-1] in ("H-TOKEN32", "H-GLOBAL-TEMP", "H-CLASS-TEMP"):
            first = (rid, pol, dec, sha)
    if first is not None:
        rid, pol, dec, sha = first
        base = ["--unit", str(unit), "--policy", str(pol), "--schema", str(SCH), "--out", str(out / "bad_out.npz")]
        inp = np.load(X)
        Xa, fn = inp["X"], inp["feature_names"]
        for nm, arrs in {"84 columns": {"X": np.c_[Xa, Xa[:, :1]], "feature_names": np.r_[fn, ["extra"]]},
                         "reordered columns": {"X": Xa[:, ::-1], "feature_names": fn[::-1]},
                         "extra sex array": {"X": Xa, "feature_names": fn, "sex": np.zeros(len(Xa))}}.items():
            pth = out / f"bad_{nm.replace(' ', '_')}.npz"
            np.savez(pth, **arrs)
            rec["refusals"][nm] = _cli(base + ["--decoder", str(dec), "--X", str(pth)])
            pth.unlink()
        dX = ["--X", str(X)]
        body = json.loads(Path(dec).read_text())
        body["r2"]["q"][0][0] += 1e-9
        from hcal import deploy as DP
        body["decoder_sha256"] = DP.decoder_hash(body)
        bad = out / "tampered_decoder.json"
        bad.write_text(json.dumps(body))
        rec["refusals"]["tampered decoder table (self-consistent hash)"] = _cli(base + dX + ["--decoder", str(bad)])
        bad.unlink()
        rec["refusals"]["decoder sha256 mismatch"] = _cli(base + dX + ["--decoder", str(dec), "--decoder-sha256",
                                                                        "0" * 64])
        other = "U|CLASS|i1o1" if not rid.startswith("U|CLASS") else "U|DIRECT-TASK|i8o64"
        rec["refusals"]["decoder bound to another map"] = _cli(
            ["--unit", str(unit), "--policy", str(PK.policy_path(k, other) if not root else
                                                   Path(root) / PK.policy_path(k, other).relative_to(I.PRIV)),
             "--schema", str(SCH), "--out", str(out / "bad_out.npz")] + dX + ["--decoder", str(dec)])
        rec["refusals"]["mismatched teacher (RAW-J unit)"] = _cli(
            ["--unit", str(adm / f"rel__s{k}__RAW-J_b0.3")] + base[2:] + dX + ["--decoder", str(dec)])
        rec["refusals"]["raw-score export flag"] = _cli(base + dX + ["--decoder", str(dec), "--export-scores", "1"])
        rec["refusals"]["fine-ID export flag"] = _cli(base + dX + ["--decoder", str(dec), "--fine-ids", "1"])
        rec["refusals"]["unknown flag"] = _cli(base + dX + ["--foo", "1"])
        rec["no_output_on_refusal"] = not (out / "bad_out.npz").exists()
        rec["all_refusals_exit_2"] = all(v["rc"] == 2 for v in rec["refusals"].values())
    rec["all_deployments_bitwise"] = all(v.get("rc") == 0 and all(v["bitwise_equal_to_registered_release"].values())
                                         and v["argmax_equals_decision"] for v in rec["deployments"].values())
    return rec


if __name__ == "__main__":
    args = sys.argv[1:]
    root = None
    if "--root" in args:
        root = args[args.index("--root") + 1]
        args = args[:args.index("--root")]
    print(json.dumps(run(int(args[0]), args[1:], root), indent=1))
