"""Source admission for the learned-decoder constrained-release study (lcr). Lead-owned; locked in SOURCE_ADMISSION_LOCK.

Admits the hash-pinned cbp artifacts (cbp final tip 7f3ec67; which carry the qpc admissions) by VERIFIED COPY (no fit,
no refit, nothing written into the cbp store):
  per seed k in {0, 1, 2}:
    tea__s{k}__U, tea__s{k}__RAW-J_b0.3            teacher outputs (row_id, p1, p2, d1, d2, c1, c2, r1, r2)
    ref__s{k}__{E, F, F0}                           official LEACE / FARE / no-fairness FARE score-only releases
    fine__s{k}                                      fixed label-blind fine partitions (income 32, occupation 128 per class)
    pol__s{k}__U_DIRECT-TASK_i8o64, pol__s{k}__U_FINE-TASK_i8o64, pol__s{k}__U_CLASS_i1o1
    pol__s{k}__U_{LOCAL,SEQ-12,SEQ-21,JOINT}_i8o64_l{0.01,0.025,0.04,0.06,0.08,0.1}   the full cbp six-weight D0 bank
    inner__<each pol / ref unit above>              the cbp matched inner audits of those D0 releases (same pinned
                                                    slate and roles; reused only with hash parity; the composed SRC|U
                                                    sources are NOT admitted: they are rebuilt over the lcr bank)
  admitted/rel__s{k}__{U,RAW-J_b0.3}                deployed teacher model.pt + heads (deployment and parity)
  inputs/deploy_input.npz, inputs/schema.json       the authorised 83-column deployment input

Every copied file must hash-match BOTH (i) the unit's own COMPLETE.json in the live cbp store and (ii) the independent
cbp same-device copy SHA256SUMS (verified uncached at the cbp closeout). Units scored in the cbp assessment must also
match cbp's EVALUATION_LOCK unit_file_sha256 at the pinned source commit 7f3ec67.

Parity (bitwise): teachers by own forward application of the admitted model.pt + heads; every pol__ unit re-encoded
from its policy.json and the admitted teacher (qpc.release) equals its stored release.npz on all rows.

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m lcr.admit plan       hash-only precheck (no data load, no copy)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
SRC = HOME / "PCRL_eval_cache_private" / "cbp_v1"
SRC_COPY = HOME / "PCRL_eval_cache_private" / "cbp_v1_local_copy_20261006"
LCR = HOME / "PCRL_eval_cache_private" / "lcr_v1"
WT = Path(__file__).resolve().parents[1]
PKG = WT / "results" / "pcrl_learned_decoder_constrained_release_v1"
SRC_EVIDENCE = "7f3ec67b2ecd86d474e2ff27167091af9923f572"
SRC_EVAL_LOCK_REL = "results/pcrl_confidence_budgeted_privacy_v1/EVALUATION_LOCK.json"
SEEDS = (0, 1, 2)
TEACHERS = ("U", "RAW-J_b0.3")
REFS = ("E", "F", "F0")
LAMS = ("0.01", "0.025", "0.04", "0.06", "0.08", "0.1")
PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")


def code_units(k):
    u = [f"pol__s{k}__U_DIRECT-TASK_i8o64", f"pol__s{k}__U_FINE-TASK_i8o64", f"pol__s{k}__U_CLASS_i1o1"]
    u += [f"pol__s{k}__U_{f}_i8o64_l{lam}" for lam in LAMS for f in PRIVACY]
    return u


def units_for(k):
    base = [f"tea__s{k}__{t}" for t in TEACHERS] + [f"ref__s{k}__{r}" for r in REFS] + [f"fine__s{k}"]
    codes = code_units(k)
    inner = [f"inner__{u}" for u in codes] + [f"inner__ref__s{k}__{r}" for r in REFS]
    return base + codes + inner


def admitted_dirs():
    return [f"rel__s{k}__{t}" for k in SEEDS for t in TEACHERS]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def copy_sums():
    out = {}
    for line in (SRC_COPY / "SHA256SUMS").read_text().splitlines():
        h, rel = line.split("  ", 1)
        out[rel] = h
    return out


def src_eval_lock_files():
    r = subprocess.run(["git", "-C", str(WT), "show", f"{SRC_EVIDENCE}:{SRC_EVAL_LOCK_REL}"], capture_output=True)
    L = json.loads(r.stdout)
    files = {}
    for s in L["seeds"].values():
        files.update(s["unit_file_sha256"])
    return files


def check_dir(src: Path, rel_prefix: str, sums: dict, complete=True):
    """(file -> sha) after checking every file against COMPLETE.json (if present) and the copy SHA256SUMS."""
    got, bad = {}, []
    comp = json.loads((src / "COMPLETE.json").read_text())["files"] if complete and (src / "COMPLETE.json").exists() else None
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(src))
        h = sha_file(p)
        got[rel] = h
        if comp is not None and rel != "COMPLETE.json" and comp.get(rel) != h:
            bad.append(f"{src.name}/{rel}: COMPLETE.json mismatch")
        ref = sums.get(f"{rel_prefix}/{rel}")
        if ref is None:
            bad.append(f"{src.name}/{rel}: absent from the cbp copy SHA256SUMS")
        elif ref != h:
            bad.append(f"{src.name}/{rel}: differs from the cbp copy SHA256SUMS")
    return got, bad


def plan():
    sums, lockf = copy_sums(), src_eval_lock_files()
    out = {"units": {}, "admitted": {}, "inputs": {}, "bad": []}
    for k in SEEDS:
        for u in units_for(k):
            got, bad = check_dir(SRC / "run" / "units" / u, f"cbp_v1/run/units/{u}", sums)
            if u in lockf:
                bad += [f"{u}/{f}: differs from the cbp EVALUATION_LOCK" for f, h in lockf[u].items() if got.get(f) != h]
            out["units"][u] = {"files": got, "in_source_evaluation_lock": u in lockf}
            out["bad"] += bad
    for a in admitted_dirs():
        got, bad = check_dir(SRC / "admitted" / a, f"cbp_v1/admitted/{a}", sums)
        out["admitted"][a] = got
        out["bad"] += bad
    got, bad = check_dir(SRC / "inputs", "cbp_v1/inputs", sums, complete=False)
    out["inputs"] = got
    out["bad"] += bad
    return out


def _copy(src: Path, dst: Path):
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        raise SystemExit(f"REFUSED: {dst.name} already admitted (resume skips verified units)")
    tmp.rename(dst)


def teacher_parity(k, t, D):
    from dpc import deploy as DD
    model, heads, msha = DD.load_teacher(LCR / "admitted" / f"rel__s{k}__{t}", seed=k)
    p = DD.teacher_probs(model, heads, D["X"])
    z = np.load(LCR / "run" / "units" / f"tea__s{k}__{t}" / "teacher.npz")
    res = {"model_sha256": msha}
    for i in (1, 2):
        res[f"p{i}_bitwise"] = bool(np.array_equal(p[i - 1], z[f"p{i}"]))
        res[f"d{i}_bitwise"] = bool(np.array_equal(p[i - 1].argmax(1), z[f"d{i}"]))
    res["row_id_equal"] = bool(np.array_equal(z["row_id"], D["row_id"]))
    res["ok"] = all(v for kk, v in res.items() if kk.endswith(("_bitwise", "_equal")))
    return res


def reencode_parity(unit, k):
    from qpc import release as RL
    pair = RL.load_policy(LCR / "run" / "units" / unit / "policy.json")
    T = np.load(LCR / "run" / "units" / f"tea__s{k}__U" / "teacher.npz")
    out = RL.release_arrays(pair, T["row_id"], T["p1"], T["d1"], T["p2"], T["d2"])
    z = np.load(LCR / "run" / "units" / unit / "release.npz")
    res = {x: bool(np.array_equal(np.asarray(out[x]), z[x])) for x in z.files}
    res["keys_equal"] = sorted(out) == sorted(z.files)
    res["decisions_equal_teacher"] = bool(np.array_equal(z["hard1"], T["d1"]) and np.array_equal(z["hard2"], T["d2"]))
    res["ok"] = all(res.values())
    return res


def run(D):
    """Verified copies + parity. Refuses (SystemExit) on any mismatch. Writes the private receipt and returns it."""
    pl = plan()
    if pl["bad"]:
        raise SystemExit("ADMISSION REFUSED (hash mismatch): " + "; ".join(pl["bad"][:10]))
    (LCR / "run" / "units").mkdir(parents=True, exist_ok=True)
    (LCR / "admitted").mkdir(parents=True, exist_ok=True)
    copied = []
    for k in SEEDS:
        for u in units_for(k):
            dst = LCR / "run" / "units" / u
            if not dst.exists():
                _copy(SRC / "run" / "units" / u, dst)
                copied.append(u)
            if {str(p.relative_to(dst)): sha_file(p) for p in dst.rglob("*") if p.is_file()} != pl["units"][u]["files"]:
                raise SystemExit(f"ADMISSION REFUSED: copy of {u} differs from its source")
    for a in admitted_dirs():
        dst = LCR / "admitted" / a
        if not dst.exists():
            _copy(SRC / "admitted" / a, dst)
            copied.append(a)
    if not (LCR / "inputs" / "deploy_input.npz").exists():
        shutil.rmtree(LCR / "inputs", ignore_errors=True)
        _copy(SRC / "inputs", LCR / "inputs")
    rec = {"schema": "lcr-admission-v1", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source_evidence": SRC_EVIDENCE, "hash_sources": ["cbp unit COMPLETE.json",
                                                             "cbp same-device copy SHA256SUMS",
                                                             "cbp EVALUATION_LOCK unit_file_sha256 (scored units)"],
           "copied_now": copied, "teachers": {}, "codes": {}, "units": sorted(pl["units"])}
    for k in SEEDS:
        for t in TEACHERS:
            r = teacher_parity(k, t, D)
            rec["teachers"][f"{t}|{k}"] = r
            if not r["ok"]:
                raise SystemExit(f"ADMISSION REFUSED: teacher parity {t} seed {k}: {r}")
        for u in units_for(k):
            if u.startswith("pol__"):
                r = reencode_parity(u, k)
                rec["codes"][u] = r
                if not r["ok"]:
                    raise SystemExit(f"ADMISSION REFUSED: re-encode parity {u}: {r}")
    rec["verdict"] = "ADMITTED"
    (LCR / "admitted" / "ADMISSION_RECEIPT.json").write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n")
    return rec


def admission_record():
    p = LCR / "admitted" / "ADMISSION_RECEIPT.json"
    return json.loads(p.read_text()) if p.exists() else {}


if __name__ == "__main__":
    if sys.argv[1:] == ["plan"]:
        pl = plan()
        print(json.dumps({"units": len(pl["units"]), "admitted_dirs": len(pl["admitted"]), "inputs": sorted(pl["inputs"]),
                          "bad": pl["bad"][:20], "precheck_ok": not pl["bad"]}, indent=1))
