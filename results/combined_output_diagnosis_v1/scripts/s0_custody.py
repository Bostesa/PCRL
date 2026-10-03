"""Stage 0A: custody and admission (no new fits). Hash-checks admitted inputs and reused output-aware units, replays one
encoder forward pass (from the checkpoint), one release head and one attacker per dataset against saved outputs."""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
HOME = Path.home()
BENCH = HOME / "PCRL_eval_cache_private/bench_v1"
OAR = HOME / "PCRL_eval_cache_private/oar_v1/run"
PKG = WT / "results/combined_output_diagnosis_v1"
DRIVE = Path(os.environ.get("PCRL_DRIVE", "/Volumes/DRIVE/relocated"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    s = str(p); return "~" + s[len(str(HOME)):] if s.startswith(str(HOME)) else s


t0 = time.time()
out = {"schema": "odx_custody/v1", "source_pin": "f7425b15cb89bee841f5b1f9dd290d05a879ae1e", "fits_performed": 0}
idx = json.loads((BENCH / "inputs/INPUTS_INDEX.json").read_text())
mm, n = [], 0
for ds, d in idx["datasets"].items():
    refs = [(d["labels_npz"], d["labels_sha256"]), (d["roles_npz"], d["roles_sha256"]), (d["features_npz"], d["features_sha256"])]
    for s, info in d["encoders"].items():
        refs += [(info["forward_npz"], info["forward_sha256"]), (info["checkpoint"], info["checkpoint_sha256"])]
    for p, h in refs:
        n += 1
        if sha(p) != h:
            mm.append(rel(p))
out["admitted_inputs"] = {"checked": n, "mismatches": mm}
# reused output-aware units: every COMPLETE.json file hash, and the drive copy index
bad, nu, nf = [], 0, 0
for d in sorted((OAR / "units").iterdir()):
    c = d / "COMPLETE.json"
    if not c.exists():
        continue
    nu += 1
    for f, h in json.loads(c.read_text())["files"].items():
        nf += 1
        if sha(d / f) != h:
            bad.append(f"{d.name}/{f}")
out["reused_output_aware_units"] = {"units": nu, "files": nf, "mismatches": bad}
for name, sub in (("bench", "private_bench_v1_20261003/SHA256SUMS"), ("oar", "private_oar_v1_20261003/SHA256SUMS")):
    p = DRIVE / sub
    out[f"drive_index_{name}"] = {"present": p.exists(), "sha256": sha(p) if p.exists() else None, "path": f"<drive>/{sub}"}
# replays
from stored_model_eval.forward import FrozenPCRLv2, load_checkpoint, run_forward
import joblib
from oar import study as S
rep = {}
for ds in ("adult", "hmda"):
    e = idx["datasets"][ds]["encoders"]["0"]
    F = np.load(BENCH / f"inputs/{ds}_features.npz")
    C = np.load(e["forward_npz"])
    test = np.flatnonzero(F["split"] == "test")[:512]
    ck, _ = load_checkpoint(e["checkpoint"], e["checkpoint_sha256"])
    o = run_forward(FrozenPCRLv2(ck), F["features"][test], batch_size=512)
    c = S.CELLS[ds]
    enc_ok = bool(np.array_equal(o[c["rep"]].astype(np.float64), C[c["rep"]][test]) and
                  np.array_equal(o[c["logits"]].astype(np.float64), C[c["logits"]][test]))
    W = S.load_world(ds)
    H = C[c["rep"]].astype(np.float64)
    hd = joblib.load(OAR / f"units/{ds}__s0__HEAD__A/models/head.joblib")
    hs = np.load(OAR / f"units/{ds}__s0__HEAD__A/preds.npz")["head_outputs_all"]
    head_diff = float(np.max(np.abs(S.head_outputs(hd, H, 2) - hs)))
    at = joblib.load(OAR / f"units/{ds}__s0__O_full/models/NL_as1.joblib")
    a = W["idx"]["assessment"]
    P = S._proba(at, C[c["logits"]][a], c["K_s"])
    att_diff = float(np.max(np.abs(P - np.load(OAR / f"units/{ds}__s0__O_full/preds.npz")["P__NL__as1"])))
    rep[ds] = {"encoder_forward_from_checkpoint_bitwise_equal_cache (first 512 test rows, rep + frozen-head logits)": enc_ok,
               "release_head_HEAD__A_s0_max_abs_diff": head_diff, "attacker_O_full_s0_NL_as1_max_abs_diff": att_diff}
out["replays"] = rep
out["wall_s"] = time.time() - t0
(PKG / "CUSTODY.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
