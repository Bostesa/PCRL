"""Source admission of the held-out calibration study (hcal; role A; locked in SOURCE_ADMISSION_LOCK).

Admits the hash-pinned lra artifacts (lra final tip 9762025, evidence 1dee332) by VERIFIED COPY. Nothing is fitted or
scored, no label is read, and nothing is written into the lra store. Units per source seed k in {0, 1, 2}:
  tea__s{k}__U, tea__s{k}__RAW-J_b0.3          teacher outputs (row_id, p1, p2, d1, d2, c1, c2, r1, r2)
  ref__s{k}__{E, F, F0}                         LEACE / FARE / no-fairness FARE score-only references
  fine__s{k}                                    label-blind fine partitions
  pol__ (27 legacy D0), dec__ (27 legacy D1), new__ (30 lra fits, D1)   the 84 original releases
  d0s__s{k}__U_C-TASK_i8o64_D1_D0SAME           lra's mean-decoded C-TASK diagnostic (TEACHER-MEAN parity witness)
  aud__<every unit above except fine / d0s>     lra inner audits (complete-interface reader banks; admitted legacy
                                                readers, fitted on all historical AUDIT_FIT rows)
  admitted/rel__s{k}__{U, RAW-J_b0.3}           deployed teacher model.pt + heads;  inputs/ (83-column deploy input)
NOT admitted: outer__* (lra assessment predictions; they carry assessment labels and are never opened before this
study's pushed EVALUATION_LOCK), inner__* (cbp custody), cor__* (fixtures).

Every copied file must hash-match (i) its unit's COMPLETE.json in the live lra store, (ii) the lra same-device copy
SHA256SUMS, and (iii) for units scored in the lra assessment, lra's EVALUATION_LOCK unit_file_sha256 read at the pinned
evidence commit with git show.

PARITY (bitwise): teachers by own forward application of the admitted model.pt + heads (dpc.deploy); every legacy D0
release re-encoded from policy.json + the admitted teacher (qpc.release); every D1 release re-encoded from policy.json +
decoder.json (lra.decoder, certificate re-solve) with tokens equal to the partition's tokens.

FROZEN BANK (prompt section 6) per partition and seed, written once to <PRIVATE_CACHE>/hcal_v1/admitted/bank/ and hashed:
  tok_i, hard_i (all rows, from the admitted release), class_i, n_fit_i, S_i (canonical token sums of U probabilities on
  OSF_DEFENSE_FIT, policy.token_S), mu_i = S_i / n_fit_i (unsmoothed teacher mean; NaN for a reserved empty token),
  q0_i = policy.token_proto (the source-smoothed mean vector; smooth(uniform, class) for a reserved empty token), qD1_i
  (the admitted learned-decoder table). TEACHER-MEAN decoder = q0 (legacy: must equal the admitted D0 release on every
  row, bitwise; lra C-TASK: must equal lra's D0SAME release, bitwise). Row-level check: token counts and teacher sums on
  OSF_DEFENSE_FIT reproduce token_n exactly and token_S within 1e-9 n. No task or SEX label enters.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time

import numpy as np

from hcal import ids as I

TEACHERS = ("U", "RAW-J_b0.3")
REFS = ("E", "F", "F0")
STATS_ROW_TOL = 1e-9


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def d0same_unit(k):
    return f"d0s__s{k}__U_C-TASK_i8o64_D1_D0SAME"


def release_units(k):
    return [I.lra_unit(k, rid) for rid in I.original_ids()]


def units_for(k):
    base = [f"tea__s{k}__{t}" for t in TEACHERS] + [f"ref__s{k}__{r}" for r in REFS] + [f"fine__s{k}", d0same_unit(k)]
    rel = release_units(k)
    aud = ([f"aud__{u}" for u in rel] + [f"aud__tea__s{k}__{t}" for t in TEACHERS]
           + [f"aud__ref__s{k}__{r}" for r in REFS])
    return base + rel + aud


def admitted_dirs():
    return [f"rel__s{k}__{t}" for k in I.SEEDS for t in TEACHERS]


def copy_sums():
    out = {}
    for line in (I.LRA_COPY / "SHA256SUMS").read_text().splitlines():
        h, rel = line.split("  ", 1)
        out[rel] = h
    return out


def src_eval_lock_files():
    r = subprocess.run(["git", "-C", str(I.WT), "show", f"{I.SOURCE_EVIDENCE}:{I.SOURCE_REL}/EVALUATION_LOCK.json"],
                       capture_output=True, check=True)
    L = json.loads(r.stdout)
    files = {}
    for s in L["seeds"].values():
        files.update(s["unit_file_sha256"])
    return files, hashlib.sha256(r.stdout).hexdigest()


def check_dir(src, rel_prefix, sums, complete=True):
    got, bad = {}, []
    comp = None
    if complete and (src / "COMPLETE.json").exists():
        comp = json.loads((src / "COMPLETE.json").read_text())["files"]
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
            bad.append(f"{src.name}/{rel}: absent from the lra copy SHA256SUMS")
        elif ref != h:
            bad.append(f"{src.name}/{rel}: differs from the lra copy SHA256SUMS")
    if comp is not None and set(comp) - set(got):
        bad.append(f"{src.name}: files listed in COMPLETE.json are missing")
    return got, bad


def plan():
    sums = copy_sums()
    lockf, lock_sha = src_eval_lock_files()
    out = {"units": {}, "admitted": {}, "inputs": {}, "bad": [], "lra_evaluation_lock_sha256": lock_sha}
    for k in I.SEEDS:
        for u in units_for(k):
            src = I.LRA_PRIV / "run" / "units" / u
            if not src.is_dir():
                out["bad"].append(f"{u}: missing from the lra store")
                continue
            got, bad = check_dir(src, f"lra_v1/run/units/{u}", sums)
            if u in lockf:
                bad += [f"{u}/{f}: differs from the lra EVALUATION_LOCK" for f, h in lockf[u].items() if got.get(f) != h]
            out["units"][u] = {"files": got, "in_source_evaluation_lock": u in lockf}
            out["bad"] += bad
    for a in admitted_dirs():
        got, bad = check_dir(I.LRA_PRIV / "admitted" / a, f"lra_v1/admitted/{a}", sums)
        out["admitted"][a] = got
        out["bad"] += bad
    got, bad = check_dir(I.LRA_PRIV / "inputs", "lra_v1/inputs", sums, complete=False)
    out["inputs"] = got
    out["bad"] += bad
    return out


def _copy(src, dst):
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        raise SystemExit(f"REFUSED: {dst.name} already admitted")
    tmp.rename(dst)


def _files(d):
    return {str(p.relative_to(d)): sha_file(p) for p in sorted(d.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------ parity
def teacher_parity(k, t, D):
    from dpc import deploy as DD
    model, heads, msha = DD.load_teacher(I.ADM / f"rel__s{k}__{t}", seed=k)
    p = DD.teacher_probs(model, heads, D["X"])
    z = np.load(I.ADM_UNITS / f"tea__s{k}__{t}" / "teacher.npz")
    res = {"model_sha256": msha}
    for i in (1, 2):
        res[f"p{i}_bitwise"] = bool(np.array_equal(p[i - 1], z[f"p{i}"]))
        res[f"d{i}_bitwise"] = bool(np.array_equal(p[i - 1].argmax(1), z[f"d{i}"]))
    res["row_id_equal"] = bool(np.array_equal(z["row_id"], D["row_id"]))
    res["ok"] = all(v for kk, v in res.items() if kk.endswith(("_bitwise", "_equal")))
    return res


def _teacher(k):
    z = np.load(I.ADM_UNITS / f"tea__s{k}__U" / "teacher.npz")
    return {x: z[x] for x in z.files}


def d0_parity(unit, T):
    from qpc import release as RL
    pair = RL.load_policy(I.ADM_UNITS / unit / "policy.json")
    out = RL.release_arrays(pair, T["row_id"], T["p1"], T["d1"], T["p2"], T["d2"])
    z = np.load(I.ADM_UNITS / unit / "release.npz")
    res = {x: bool(np.array_equal(np.asarray(out[x]), z[x])) for x in z.files}
    res["keys_equal"] = sorted(out) == sorted(z.files)
    res["decisions_equal_teacher"] = bool(np.array_equal(z["hard1"], T["d1"]) and np.array_equal(z["hard2"], T["d2"]))
    res["ok"] = all(res.values())
    return res


def d1_parity(unit, T):
    from lra import decoder as DEC
    from qpc import release as RL
    pair = RL.load_policy(I.ADM_UNITS / unit / "policy.json")
    _, d1, d2, sha = DEC.load_decoder_pair(I.ADM_UNITS / unit / "decoder.json", pair=pair, verify_solve=True)
    out = DEC.release_arrays_d1(pair, d1, d2, T["row_id"], T["p1"], T["d1"], T["p2"], T["d2"])
    z = np.load(I.ADM_UNITS / unit / "release.npz")
    res = {x: bool(np.array_equal(np.asarray(out[x]), z[x])) for x in z.files}
    res["keys_equal"] = sorted(out) == sorted(z.files)
    res["decoder_sha256"] = sha
    res["decoder_resolved_bitwise"] = True          # load_decoder_pair(verify_solve=True) re-solves every token
    res["ok"] = all(v for x, v in res.items() if x != "decoder_sha256")
    return res, pair, (d1, d2)


# ------------------------------------------------------------------ frozen bank
def bank_path(k, p):
    return I.ADM / "bank" / f"bank__s{k}__{I.safe(p)}.npz"


def build_bank(k, p, D, T):
    """Frozen per-partition tables (module docstring). Returns (arrays, receipt)."""
    from qpc import release as RL
    units = I.partition_original_units(k, p)
    pol_unit = units[0]                                 # legacy: pol__ (D0); lra: new__ (D1)
    pair = RL.load_policy(I.ADM_UNITS / pol_unit / "policy.json")     # from_dict re-derives token tables (pinned)
    z0 = np.load(I.ADM_UNITS / pol_unit / "release.npz")
    fit = np.asarray(D["idx"]["OSF_DEFENSE_FIT"], dtype=np.int64)
    arr, rec = {"row_id": np.asarray(z0["row_id"])}, {"partition": p, "seed": k, "policy_unit": pol_unit,
                                                      "policy_fingerprint": pair.fingerprint()}
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        tok = np.asarray(z0[f"tok{i}"], dtype=np.int64)
        hard = np.asarray(z0[f"hard{i}"], dtype=np.int64)
        tc = np.asarray(pol.token_class, dtype=np.int64)
        n = np.asarray(pol.token_n, dtype=np.int64)
        S = np.asarray(pol.token_S, dtype=np.float64)
        q0 = np.asarray(pol.token_proto, dtype=np.float64)
        Tn, K = q0.shape
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(n[:, None] > 0, S / np.where(n > 0, n, 1)[:, None], np.nan)
        P = np.asarray(T[f"p{i}"], dtype=np.float64)
        n_rows = np.bincount(tok[fit], minlength=Tn).astype(np.int64)
        S_rows = np.stack([np.bincount(tok[fit], weights=P[fit, c], minlength=Tn) for c in range(K)], 1)
        chk = {"token_n_rows_equal": bool(np.array_equal(n_rows, n)),
               "token_S_rows_within_tol": bool(np.all(np.abs(S_rows - S) <= STATS_ROW_TOL * np.maximum(n, 1)[:, None])),
               "decisions_are_token_class": bool(np.array_equal(tc[tok], hard)),
               "decisions_equal_teacher": bool(np.array_equal(hard, np.asarray(T[f"d{i}"]))),
               "q0_rows_equal_d0_release": None, "reserved_empty_tokens": int((n == 0).sum()), "tokens": int(Tn)}
        if I.is_legacy(p):
            chk["q0_rows_equal_d0_release"] = bool(np.array_equal(q0[tok], np.asarray(z0[f"q{i}"])))
        # D1 table (admitted)
        d1u = I.lra_unit(k, I.release_id(p, "D1"))
        dj = json.loads((I.ADM_UNITS / d1u / "decoder.json").read_text())[f"r{i}"]
        qD1 = np.asarray(dj["q"], dtype=np.float64).reshape(-1, K)
        z1 = np.load(I.ADM_UNITS / d1u / "release.npz")
        chk["d1_tokens_equal_partition"] = bool(np.array_equal(np.asarray(z1[f"tok{i}"]), tok))
        chk["d1_rows_equal_table"] = bool(np.array_equal(qD1[tok], np.asarray(z1[f"q{i}"])))
        arr.update({f"tok{i}": tok, f"hard{i}": hard, f"class{i}": tc, f"n_fit{i}": n, f"S{i}": S, f"mu{i}": mu,
                    f"q0{i}": q0, f"qD1{i}": qD1, f"alpha{i}": np.int64(Tn)})
        rec[f"r{i}"] = chk
    if p == "U|C-TASK|i8o64":
        zs = np.load(I.ADM_UNITS / d0same_unit(k) / "release.npz")
        rec["mean_equals_lra_d0same"] = all(
            bool(np.array_equal(arr[f"q0{i}"][arr[f"tok{i}"]], np.asarray(zs[f"q{i}"]))
                 and np.array_equal(arr[f"tok{i}"], np.asarray(zs[f"tok{i}"]))) for i in (1, 2))
    oks = [v for i in (1, 2) for kk, v in rec[f"r{i}"].items() if isinstance(v, bool)]
    rec["ok"] = all(oks) and rec.get("mean_equals_lra_d0same", True)
    return arr, rec


def run(D):
    pl = plan()
    if pl["bad"]:
        raise SystemExit("ADMISSION REFUSED (hash mismatch): " + "; ".join(pl["bad"][:10]))
    I.ADM_UNITS.mkdir(parents=True, exist_ok=True)
    copied = []
    for k in I.SEEDS:
        for u in units_for(k):
            dst = I.ADM_UNITS / u
            if not dst.exists():
                _copy(I.LRA_PRIV / "run" / "units" / u, dst)
                copied.append(u)
            if _files(dst) != pl["units"][u]["files"]:
                raise SystemExit(f"ADMISSION REFUSED: copy of {u} differs from its source")
    for a in admitted_dirs():
        dst = I.ADM / a
        if not dst.exists():
            _copy(I.LRA_PRIV / "admitted" / a, dst)
            copied.append(a)
        if _files(dst) != pl["admitted"][a]:
            raise SystemExit(f"ADMISSION REFUSED: copy of {a} differs from its source")
    if not (I.ADM / "inputs").exists():
        _copy(I.LRA_PRIV / "inputs", I.ADM / "inputs")
    if _files(I.ADM / "inputs") != pl["inputs"]:
        raise SystemExit("ADMISSION REFUSED: inputs copy differs")
    rec = {"schema": "hcal-admission-v1", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source_tip": I.SOURCE_TIP, "source_evidence": I.SOURCE_EVIDENCE,
           "hash_sources": ["lra unit COMPLETE.json", "lra same-device copy SHA256SUMS",
                            "lra EVALUATION_LOCK unit_file_sha256 at the evidence commit (scored units)"],
           "lra_evaluation_lock_sha256": pl["lra_evaluation_lock_sha256"], "copied_now": copied,
           "units": {u: v["files"] for u, v in pl["units"].items()}, "admitted": pl["admitted"],
           "inputs": pl["inputs"], "teachers": {}, "d0": {}, "d1": {}, "bank": {}}
    for k in I.SEEDS:
        for t in TEACHERS:
            r = teacher_parity(k, t, D)
            rec["teachers"][f"{t}|{k}"] = r
            if not r["ok"]:
                raise SystemExit(f"ADMISSION REFUSED: teacher parity {t} seed {k}: {r}")
        T = _teacher(k)
        for rid in I.original_ids():
            u = I.lra_unit(k, rid)
            if u.startswith("pol__"):
                r = d0_parity(u, T)
                rec["d0"][u] = r
            else:
                r, _, _ = d1_parity(u, T)
                rec["d1"][u] = r
            if not r["ok"]:
                raise SystemExit(f"ADMISSION REFUSED: re-encode parity {u}: {r}")
        (I.ADM / "bank").mkdir(parents=True, exist_ok=True)
        for p in I.partitions():
            arr, br = build_bank(k, p, D, T)
            if not br["ok"]:
                raise SystemExit(f"ADMISSION REFUSED: frozen bank {p} seed {k}: {br}")
            path = bank_path(k, p)
            if path.exists():
                old = np.load(path)
                same = sorted(old.files) == sorted(arr) and all(np.array_equal(old[x], arr[x], equal_nan=True)
                                                                if np.asarray(arr[x]).dtype.kind == "f"
                                                                else np.array_equal(old[x], arr[x]) for x in arr)
                if not same:
                    raise SystemExit(f"REFUSED: frozen bank {path.name} exists with different content")
            else:
                tmp = path.with_name(path.name + ".tmp.npz")
                np.savez_compressed(tmp, **arr)
                tmp.rename(path)
            br["bank_sha256"] = sha_file(path)
            rec["bank"][f"{k}|{p}"] = br
    rec["verdict"] = "ADMITTED"
    (I.ADM / "ADMISSION_RECEIPT.json").write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n")
    return rec


def public_record(rec, D):
    """SOURCE_ADMISSION.json body: hashes, counts and parity verdicts only (no paths, no per-person data)."""
    return {"schema": "hcal-source-admission-v1", "verdict": rec["verdict"], "at": rec["at"],
            "source_tip": rec["source_tip"], "source_evidence": rec["source_evidence"],
            "hash_sources": rec["hash_sources"], "lra_evaluation_lock_sha256": rec["lra_evaluation_lock_sha256"],
            "store": "<PRIVATE_CACHE>/hcal_v1/admitted (verified copies; never written after admission)",
            "not_admitted": ["outer__* (lra assessment predictions; carry assessment labels)", "inner__* (cbp custody)",
                             "cor__* (fixtures)"],
            "units": {u: {"files_sha256": f} for u, f in rec["units"].items()},
            "admitted_teacher_dirs": rec["admitted"], "inputs": rec["inputs"],
            "teacher_parity": rec["teachers"], "d0_reencode_parity": {u: r["ok"] for u, r in rec["d0"].items()},
            "d1_reencode_parity": {u: {"ok": r["ok"], "decoder_sha256": r["decoder_sha256"]} for u, r in rec["d1"].items()},
            "frozen_bank": rec["bank"],
            "counts": {"units": len(rec["units"]), "original_releases_per_seed": len(I.original_ids()),
                       "partitions_per_seed": len(I.partitions()), "bank_tables": len(rec["bank"])}}


if __name__ == "__main__":
    if sys.argv[1:] == ["plan"]:
        pl = plan()
        print(json.dumps({"units": len(pl["units"]), "admitted_dirs": len(pl["admitted"]), "inputs": sorted(pl["inputs"]),
                          "bad": pl["bad"][:20], "precheck_ok": not pl["bad"]}, indent=1))
