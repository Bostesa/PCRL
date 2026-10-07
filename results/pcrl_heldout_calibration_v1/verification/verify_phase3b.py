"""Role E phase 3b: deployment and custody of the hcal study, then the consolidated INDEPENDENT_VERIFICATION.json.

  1 COPY_SHA256SUMS     every file listed in the same-device copy's SHA256SUMS re-hashed here; no unlisted file; the
                        SHA256SUMS hash equals BACKUP_RECORD.json's; the copied pinned input equals the locked input hash.
  2 TEACHER_RESTORE     from the COPY ALONE: the copied seed-1 U teacher (model.pt + heads, loaded with the pinned
                        dpc.deploy.load_teacher / teacher_probs) applied to the copied 83-column deploy input reproduces
                        the copied teacher.npz p1 / p2 bitwise (d = argmax); the copied deploy input equals the sealed
                        D's X built by the pinned loader from the identical pinned input.
  3 DEPLOYMENT          the real CLI `python -m hcal.deploy` (subprocess; product CLI) on the copy's teacher, policy,
                        packaged decoder and input for P* and T* at seed 1, compared bitwise with an OWN rebuild:
                        tokens = copied frozen-bank tokens, table = softmax(alpha log token_proto) per recipient / class
                        from the copied policy.json and the decoder's stored alpha(s) (own decoder hash recomputed);
                        decisions = token classes = teacher argmax. >= 6 registered refusals exit 2 with no output.
  4 CUSTODY             off-device custody is PENDING (no content-matched drive) and is not claimed anywhere.
Then INDEPENDENT_VERIFICATION.json consolidates phase 0 (E_ENGINEERING_REVIEW.md + dispositions), phase 1/2
(E_INNER_REPLAY.json), phase 3 (E_PHASE3.json) and phase 3b (E_PHASE3B.json).

No hcal module is imported here (checked); the product CLI runs in a subprocess. Imports: numpy, json, hashlib,
subprocess; dpc.deploy (pinned; load_teacher / teacher_probs only); lra.data (pinned loader; load() only).

    ~/PCRL/.venv/bin/python -P <WT>/hcal/sema.py --label E:phase3b -- env OMP_NUM_THREADS=1 \\
        PCRL_HCAL_PRIVATE_CACHE=$HOME/PCRL_eval_cache_private/hcal_v1 ~/PCRL/.venv/bin/python \\
        <WT>/results/pcrl_heldout_calibration_v1/verification/verify_phase3b.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("verify_inner_e", HERE.parent / "verify_inner.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

COPY = Path.home() / "PCRL_eval_cache_private" / "hcal_v1_local_copy_20261007"
CS = COPY / "hcal_v1"
OUT = HERE.parent / "E_PHASE3B.json"
FINAL = HERE.parent / "INDEPENDENT_VERIFICATION.json"
SEED = 1
RELEASES = {"P*": "U|JOINT|i8o64|l0.1|H-CLASS-TEMP", "T*": "U|FINE-TASK|i8o64|H-GLOBAL-TEMP"}


def sha_file(p):
    return V.sha_file(p)


def pubc(p):
    return str(p).replace(str(COPY), "<PRIVATE_CACHE>/hcal_v1_local_copy_20261007").replace(str(Path.home()), "~")


# ------------------------------------------------------------------ 1
def check_copy(L):
    chk = V.Check("1_COPY_SHA256SUMS", "same-device copy: every listed file re-hashed; no unlisted file")
    sums = {}
    for line in (COPY / "SHA256SUMS").read_text().splitlines():
        h, rel = line.split("  ", 1)
        sums[rel] = h
    bad = [rel for rel, h in sums.items() if not (COPY / rel).is_file() or sha_file(COPY / rel) != h]
    chk.ok(not bad, f"{len(bad)} listed file(s) missing or differing: {bad[:5]}")
    on_disk = {str(p.relative_to(COPY)) for p in COPY.rglob("*") if p.is_file()} - {"SHA256SUMS", "BACKUP_RECORD.json"}
    chk.same("unlisted files", sorted(on_disk - set(sums)), [])
    rec = json.loads((COPY / "BACKUP_RECORD.json").read_text())
    chk.same("SHA256SUMS sha256 vs BACKUP_RECORD", sha_file(COPY / "SHA256SUMS"), rec.get("SHA256SUMS_sha256"))
    chk.same("file count vs BACKUP_RECORD", len(sums), rec.get("files"))
    dep = COPY / "dependencies" / "jcv_v1" / "inputs" / "adult_jcv.npz"
    want = (L.get("inputs") or {}).get("source_npz_sha256") or \
        json.loads((V.PKG / "SCIENCE_LOCK.json").read_text())["inputs"]["source_npz_sha256"]
    chk.same("copied pinned input sha256", sha_file(dep), want)
    chk.info.update({"files_listed": len(sums), "files_verified": len(sums) - len(bad), "label": rec.get("label"),
                     "copy_at": rec.get("at")})
    return chk.done()


# ------------------------------------------------------------------ 2
def check_teacher(D):
    chk = V.Check("2_TEACHER_RESTORE", "seed-1 U teacher forward from the copy alone, bitwise vs the copied outputs")
    from dpc import deploy as DD                     # pinned: load the serialized model only
    inp = np.load(CS / "admitted" / "inputs" / "deploy_input.npz", allow_pickle=False)
    X = inp["X"]
    chk.ok(np.array_equal(X, np.asarray(D["X"])), "copied deploy input != sealed D's X (pinned loader)")
    chk.ok(list(inp["feature_names"]) == list(D["feature_names"]), "copied deploy input feature names differ")
    model, heads, msha = DD.load_teacher(CS / "admitted" / f"rel__s{SEED}__U", seed=SEED)
    P = DD.teacher_probs(model, heads, X)
    tf = V.load_unit(f"tea__s{SEED}__U", CS / "admitted" / "units")
    T = V.npz(tf["teacher.npz"])
    for i in (1, 2):
        chk.ok(np.array_equal(P[i - 1], T[f"p{i}"]), f"p{i}: restored forward != copied teacher.npz (bitwise)")
        chk.ok(np.array_equal(P[i - 1].argmax(1), T[f"d{i}"]), f"d{i}: argmax != copied decisions")
    chk.ok(np.array_equal(T["row_id"], D["row_id"]), "copied teacher rows not in D order")
    chk.info["model_sha256"] = msha
    return chk.done(), T


# ------------------------------------------------------------------ 3
def decoder_hash(body):
    z = {k: v for k, v in body.items() if k != "decoder_sha256"}
    return hashlib.sha256(json.dumps(z, sort_keys=True, allow_nan=False).encode()).hexdigest()


def own_table(proto, cls, rec):
    q0 = np.asarray(proto, dtype=np.float64)
    cls = np.asarray(cls, dtype=np.int64)
    if rec["kind"] == "H-GLOBAL-TEMP":
        a = float(rec["alpha"])
        return q0.copy() if a == 1.0 else V.softmax_rows(a * np.log(q0))
    q = q0.copy()
    for c, a in enumerate(rec["alphas"]):
        m = cls == c
        if float(a) != 1.0 and m.any():
            q[m] = V.softmax_rows(float(a) * np.log(q0[m]))
    return q


def cli(args, wt=V.WT):
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=".")
    r = subprocess.run([sys.executable, "-m", "hcal.deploy"] + args, capture_output=True, text=True, env=env,
                       cwd=str(wt))
    return r.returncode, r.stdout.strip(), r.stderr.strip().splitlines()[-1:] if r.stderr.strip() else []


def check_deploy(T):
    chk = V.Check("3_DEPLOYMENT", "real hcal.deploy CLI on the copy vs an own rebuild (bitwise); refusals exit 2")
    tmp = Path(tempfile.mkdtemp(prefix="hcal_e_deploy_", dir=os.environ.get("E_SCRATCH") or None))
    X = CS / "admitted" / "inputs" / "deploy_input.npz"
    SCH = CS / "admitted" / "inputs" / "schema.json"
    unit = CS / "admitted" / f"rel__s{SEED}__U"
    receipt = json.loads((V.RUN / "deploy_receipt_s1.json").read_text())
    first = None
    try:
        for role, rid in RELEASES.items():
            p, dec = V.RID_INFO[rid]
            pol_path = CS / "admitted" / "units" / f"pol__s{SEED}__{V.safe(p)}" / "policy.json"
            dpath = CS / "package" / f"decoder__s{SEED}__{V.safe(rid)}.json"
            body = json.loads(dpath.read_text())
            sha = body["decoder_sha256"]
            chk.same("own decoder hash", decoder_hash(body), sha, rid)
            chk.same("decoder hash vs A's receipt", (receipt["deployments"].get(rid) or {}).get("decoder_sha256"), sha,
                     rid)
            chk.same("decoder release / family", (body["release_id"], body["family"]), (rid, dec), rid)
            out = tmp / f"{V.safe(rid)}.npz"
            rc, so, se = cli(["--unit", str(unit), "--policy", str(pol_path), "--decoder", str(dpath),
                              "--decoder-sha256", sha, "--X", str(X), "--schema", str(SCH), "--out", str(out)])
            chk.same("deploy exit code", rc, 0, f"{rid} {se}")
            if rc != 0:
                continue
            z = np.load(out, allow_pickle=False)
            chk.same("written arrays", sorted(z.files), sorted(["tokens_1", "probs_1", "decision_1", "tokens_2",
                                                                "probs_2", "decision_2"]), rid)
            pol = json.loads(pol_path.read_text())
            b = V.npz(CS / "admitted" / "bank" / f"bank__s{SEED}__{V.safe(p)}.npz")
            tz = V.npz(V.load_unit(f"cal__s{SEED}__{V.safe(p)}", CS / "run" / "units")["tables.npz"])
            for i in (1, 2):
                pi = pol[f"p{i}"]
                proto = np.asarray(pi["token_proto"], dtype=np.float64).reshape(len(pi["token_class"]), -1)
                chk.ok(np.array_equal(proto, b[f"q0{i}"]), f"{rid} r{i}: policy token_proto != copied bank q0")
                tab = own_table(proto, pi["token_class"], body[f"r{i}"])
                chk.ok(np.array_equal(tab, np.asarray(body[f"r{i}"]["q"], dtype=np.float64).reshape(tab.shape)),
                       f"{rid} r{i}: own table != packaged decoder table")
                chk.ok(np.array_equal(tab, tz[f"{dec}|q{i}"]), f"{rid} r{i}: own table != copied cal__ table")
                tok = np.asarray(b[f"tok{i}"], dtype=np.int64)
                chk.ok(np.array_equal(z[f"tokens_{i}"], tok), f"{rid} r{i}: deployed tokens != copied bank tokens")
                chk.ok(np.array_equal(z[f"probs_{i}"], tab[tok]), f"{rid} r{i}: deployed probs != own rebuild")
                chk.ok(np.array_equal(z[f"decision_{i}"], b[f"hard{i}"]), f"{rid} r{i}: decisions != bank decisions")
                chk.ok(np.array_equal(z[f"decision_{i}"], T[f"d{i}"]), f"{rid} r{i}: decisions != teacher argmax")
                chk.ok(np.array_equal(z[f"probs_{i}"].argmax(1), z[f"decision_{i}"]), f"{rid} r{i}: argmax != decision")
            info = json.loads(so.splitlines()[-1]) if so else {}
            chk.same("CLI release id", info.get("release_id"), rid)
            chk.info.setdefault("deployed", []).append(rid)
            if first is None:
                first = (rid, pol_path, dpath, sha, body)
        # refusals (the first calibrated release's inputs)
        rid, pol_path, dpath, sha, body = first
        inp = np.load(X, allow_pickle=False)
        Xa, fn = inp["X"], inp["feature_names"]
        bad_out = tmp / "bad_out.npz"
        base = ["--unit", str(unit), "--policy", str(pol_path), "--schema", str(SCH), "--out", str(bad_out)]
        cases = {}
        for nm, arrs in {"extra column (84)": {"X": np.c_[Xa, Xa[:, :1]], "feature_names": np.r_[fn, ["extra"]]},
                         "reordered columns": {"X": Xa[:, ::-1], "feature_names": fn[::-1]},
                         "extra array (sex)": {"X": Xa, "feature_names": fn, "sex": np.zeros(len(Xa))}}.items():
            pth = tmp / f"in_{len(cases)}.npz"
            np.savez(pth, **arrs)
            cases[nm] = base + ["--decoder", str(dpath), "--X", str(pth)]
        tb = json.loads(json.dumps(body))
        tb["r2"]["q"][0][0] += 1e-9
        tb["decoder_sha256"] = decoder_hash(tb)
        tpath = tmp / "tampered.json"
        tpath.write_text(json.dumps(tb))
        dX = ["--X", str(X)]
        cases["tampered table (self-consistent hash)"] = base + dX + ["--decoder", str(tpath)]
        cases["wrong teacher (RAW-J unit)"] = (["--unit", str(CS / "admitted" / f"rel__s{SEED}__RAW-J_b0.3")]
                                               + base[2:] + dX + ["--decoder", str(dpath)])
        cases["decoder sha256 mismatch"] = base + dX + ["--decoder", str(dpath), "--decoder-sha256", "0" * 64]
        cases["unknown flag"] = base + dX + ["--decoder", str(dpath), "--foo", "1"]
        cases["raw-score export flag"] = base + dX + ["--decoder", str(dpath), "--raw-scores", "1"]
        cases["fine-ID export flag"] = base + dX + ["--decoder", str(dpath), "--fine-ids", "1"]
        ref = {}
        for nm, args in cases.items():
            rc, so, se = cli(args)
            ref[nm] = {"rc": rc, "stdout_empty": so == "", "stderr": [s.replace(str(Path.home()), "~") for s in se],
                       "no_output_file": not bad_out.exists()}
            chk.ok(rc == 2 and so == "" and not bad_out.exists(), f"refusal '{nm}': rc {rc}, stdout {so[:80]!r}")
            if bad_out.exists():
                bad_out.unlink()
        chk.ok(len(ref) >= 6, "fewer than 6 refusals exercised")
        chk.info["refusals"] = ref
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return chk.done()


# ------------------------------------------------------------------ 4
def check_custody():
    chk = V.Check("4_CUSTODY", "off-device custody PENDING (drive absent) and not claimed")
    vols = [p.name for p in Path("/Volumes").iterdir()] if Path("/Volumes").exists() else []
    chk.info["volumes"] = vols
    sd = json.loads((V.RUN / "same_device_copy.json").read_text()) if (V.RUN / "same_device_copy.json").exists() \
        else {}
    chk.info["same_device_copy_off_device_field"] = sd.get("off_device_copy")
    chk.ok(str(sd.get("off_device_copy", "PENDING")).startswith("PENDING"), "same-device record claims off-device")
    bv = V.PKG / "BACKUP_VERIFICATION.json"
    if bv.exists():
        txt = bv.read_text()
        chk.ok("PENDING" in txt or '"off_device": null' in txt, "BACKUP_VERIFICATION.json claims off-device custody")
        chk.info["backup_verification"] = "present"
    else:
        chk.info["backup_verification"] = "absent (role F has not written it)"
    claims = []
    for p in list(V.PKG.glob("*.md")) + list(V.PKG.glob("*.json")):
        for ln in p.read_text().splitlines():
            low = ln.lower()
            if "off-device" in low and ("verified" in low or "complete" in low) and "pending" not in low \
                    and "when" not in low and "otherwise" not in low and "test_" not in low:
                claims.append(f"{p.name}: {ln.strip()[:160]}")
    chk.info["possible_claims"] = claims
    chk.ok(not claims, f"possible off-device custody claim(s): {claims[:3]}")
    chk.info["custody_status"] = "PENDING (drive absent; not claimed)"
    return chk.done()


# ------------------------------------------------------------------ consolidation
def consolidate(p3b):
    def load(name):
        p = HERE.parent / name
        return json.loads(p.read_text()) if p.exists() else None
    inner, p3 = load("E_INNER_REPLAY.json"), load("E_PHASE3.json")
    disp = json.loads((V.PKG / "REVIEW_FINDINGS_DISPOSITION.json").read_text())
    efind = {f["id"]: {"severity": f["severity"], "disposition": f["disposition"]} for f in disp["findings"]
             if f["id"].startswith("E-")}
    open_blocking = [i for i, f in efind.items() if f["severity"] == "BLOCKING" and
                     not f["disposition"].startswith(("FIXED", "AMENDED", "ADDED"))]
    phase0 = {"document": "verification/E_ENGINEERING_REVIEW.md",
              "document_sha256": sha_file(HERE.parent / "E_ENGINEERING_REVIEW.md"),
              "findings": {"BLOCKING": 1, "SHOULD-FIX": 7, "NOTE": 11}, "dispositions": efind,
              "open_blocking": open_blocking, "status": "PASS" if not open_blocking else "FAIL",
              "note": "E-B1 (controls verdict key) verified fixed in code: select.controls_all_ok reads verdict.all_ok "
                      "and eval_lock / stages use it; the inner replay and the assessment confirm controls_all_ok "
                      "true"}
    ph = {"phase_0_engineering_review": phase0,
          "phase_1_2_inner_replay": {"file": "verification/E_INNER_REPLAY.json",
                                     "sha256": sha_file(HERE.parent / "E_INNER_REPLAY.json"),
                                     "statuses": (inner or {}).get("statuses"), "verdict": (inner or {}).get("verdict")},
          "phase_3_assessment": {"file": "verification/E_PHASE3.json", "sha256": sha_file(HERE.parent / "E_PHASE3.json"),
                                 "statuses": (p3 or {}).get("statuses"), "verdict": (p3 or {}).get("verdict"),
                                 "label": ((p3 or {}).get("checks", {}).get("d_PRIMARY_FAMILY") or {}).get("label")},
          "phase_3b_deployment_custody": {"file": "verification/E_PHASE3B.json", "statuses": p3b["statuses"],
                                          "verdict": p3b["verdict"],
                                          "off_device_custody": "PENDING (drive absent; not claimed)"}}
    verdicts = [phase0["status"], ph["phase_1_2_inner_replay"]["verdict"], ph["phase_3_assessment"]["verdict"],
                p3b["verdict"]]
    final = "PASS" if all(v == "PASS" for v in verdicts) else ("FAIL" if "FAIL" in verdicts else "PENDING")
    scripts = {n: sha_file(HERE.parent / n) for n in ("verify_inner.py", "selftest_verify_inner.py", "verify_phase3.py",
                                                      "verify_phase3b.py")}
    rec = {"schema": "hcal-independent-verification-v1", "role": "E (independent verifier)",
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "study": "pcrl_heldout_calibration_v1",
           "head": V.git("rev-parse", "HEAD").stdout.strip(),
           "evaluation_lock_sha256": sha_file(V.PKG / "EVALUATION_LOCK.json"), "phases": ph, "scripts_sha256": scripts,
           "independence": {
               "statement": "Role E's verifier code imports no hcal module (checked at exit of every script: "
                            "hcal_modules_loaded = []); formulas, grouping, metrics, selection, union rules, bootstrap, "
                            "clause outcomes and the label are re-derived; only pinned loaders (lra.data.load, "
                            "dpc.deploy.load_teacher / teacher_probs) and sklearn / numpy are used; the product CLI "
                            "hcal.deploy is exercised only as a subprocess. Assessment labels were read only from the "
                            "hash-complete oprob__ units bound to the pushed EVALUATION_LOCK; no study gate was used or "
                            "bypassed.",
               "hcal_modules_loaded": {"verify_inner": (inner or {}).get("independence", {}).get("hcal_modules_loaded"),
                                       "verify_phase3": (p3 or {}).get("independence", {}).get("hcal_modules_loaded"),
                                       "verify_phase3b": p3b["independence"]["hcal_modules_loaded"]},
               "worktree_modules_loaded_inner": (inner or {}).get("independence", {}).get(
                   "worktree_top_level_modules_loaded")},
           "scope_limits": ["exploratory development evidence on repeatedly used Adult data (registered exposure "
                            "statement); verification establishes reproduction, not external validity",
                            "attack refits verified on the registered sample (20 winners), not every winner",
                            "off-device custody PENDING (drive absent)"],
           "PHASE_3_FINAL": final}
    txt = json.dumps(V.jsafe(rec), indent=1, allow_nan=False) + "\n"
    for bad in (str(Path.home()) + "/", "/Users/", "/private/"):
        if bad in txt:
            raise SystemExit("REFUSED: a private path would enter INDEPENDENT_VERIFICATION.json")
    FINAL.write_text(txt)
    return final


def main():
    t0 = time.time()
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    sys.path.insert(0, str(V.WT))
    from lra import data as LD
    D = LD.load(verify=True, unseal=False)
    L = json.loads((V.PKG / "EVALUATION_LOCK.json").read_text())
    check_copy(L)
    _, T = check_teacher(D)
    check_deploy(T)
    check_custody()
    loaded_hcal = sorted(m for m in sys.modules if m == "hcal" or m.startswith("hcal."))
    statuses = {k: v["status"] for k, v in V.CHECKS.items()}
    verdict = "FAIL" if loaded_hcal or "FAIL" in statuses.values() else (
        "PASS" if all(s == "PASS" for s in statuses.values()) else "PENDING")
    out = {"schema": "hcal-E-phase3b-v1", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "script_sha256": sha_file(HERE), "copy": "<PRIVATE_CACHE>/hcal_v1_local_copy_20261007", "seed": SEED,
           "releases": RELEASES, "independence": {"hcal_modules_loaded": loaded_hcal,
                                                  "imports": ["numpy", "dpc.deploy (load_teacher / teacher_probs)",
                                                              "lra.data (load)", "subprocess: python -m hcal.deploy"]},
           "statuses": statuses, "checks": V.CHECKS, "wall_s": round(time.time() - t0, 1), "verdict": verdict}
    txt = json.dumps(V.jsafe(out), indent=1, allow_nan=False) + "\n"
    txt = txt.replace(str(COPY), "<PRIVATE_CACHE>/hcal_v1_local_copy_20261007")
    for bad in (str(Path.home()) + "/", "/Users/", "/private/"):
        if bad in txt:
            raise SystemExit("REFUSED: a private path would enter E_PHASE3B.json")
    OUT.write_text(txt)
    final = consolidate(out)
    print(json.dumps({"verdict": verdict, "statuses": statuses, "PHASE_3_FINAL": final,
                      "wall_s": out["wall_s"]}, indent=1))


if __name__ == "__main__":
    main()
