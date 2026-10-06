"""Admission of the frozen teachers and official references for the confidence-capacity study (qpc).

Owner: data/custody role (E). Light compute only (hashing, copying, one forward pass per teacher). No task label and no
SEX value is read anywhere here (D's label arrays are never indexed).

Pins (all read with `git show <pin>:<path>`, never from the working tree):
  dpc ADMISSION.json        @ 0a7b05a  complete_files_sha256 + model_sha256 of every admitted teacher/reference artifact
  dpc EVALUATION_LOCK.json  @ 0a7b05a  unit_file_sha256 of the dpc teacher units tea__s{k}__{t} and reference units
                                       ref__s{k}__{E,F,F0} (the parity targets)
  osf MODEL_MANIFEST.json   @ 925e0fd  (via dpc.admit.resolve; the further source of the same artifacts)

Teachers t in {U, RAW-J_b0.3}, seeds k in {0, 1, 2}:
  1. source = <PRIVATE_CACHE>/dpc_v1/admitted/rel__s{k}__{t} (fallback: the osf store <PRIVATE_CACHE>/osf_v1/run/units/
     rel__s{k}__{t}); its COMPLETE.json file list must equal the pinned dpc ADMISSION entry and every file must re-hash
     to it. Sources are READ-ONLY (copied, never moved or modified).
  2. verified copy -> <PRIVATE_CACHE>/qpc_v1/admitted/rel__s{k}__{t}/ (re-read uncached; a differing existing copy is
     refused, nothing overwritten).
  3. own forward application from the qpc copy: state-dict keys/shapes, model_sha256 recomputed, encoder forward and
     deployed-head outputs with the pinned dpc.admit functions (forward, head_outputs, state_sha; dpc/admit.py blob at
     0a7b05a, itself the dpc re-implementation of the osf/rgj conventions) -> r_i (16-d features), c_i (centred logits),
     p_i (probabilities), d_i (first-index argmax).
  4. parity: BITWISE equality (values, dtype, shape) of all nine arrays with the dpc teacher unit tea__s{k}__{t}/
     teacher.npz (whose file hash must equal the dpc EVALUATION_LOCK pin) and of r/c/p/hard with the copied release.npz.
     Any mismatch refuses: no code may be fitted on a teacher that fails parity.
References E (official LEACE), F (official FARE), F0 (no-fairness FARE twin), seeds 0-2:
  verified copy of the dpc reference unit <PRIVATE_CACHE>/dpc_v1/run/units/ref__s{k}__{label} (hash = dpc
  EVALUATION_LOCK pin) -> <PRIVATE_CACHE>/qpc_v1/admitted/ref__s{k}__{label}/, plus array parity with the dpc admitted
  artifacts (dpc.admit.reference, which re-verifies those artifacts against the pinned osf manifest). Provenance validity
  = the pinned dpc admission passed for every underlying unit AND the current hashes and arrays match.
Inputs: the raw input <PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz (pinned sha256) is copied with verification to
  <PRIVATE_CACHE>/qpc_v1/admitted/inputs/; the only derived input (the 83-column deploy input + schema, removed as
  session scratch at the dpc closeout) is reconstructed from it through the pinned loader into
  <PRIVATE_CACHE>/qpc_v1/inputs/. Nothing is downloaded.

API (dpc.admit-compatible, as used by dpc/run.py stage_admit):
  teacher(t, k[, D]) -> dict(row_id, p1, p2, d1, d2, c1, c2, r1, r2, model_sha256, state_sha256, unit)
                       bitwise-parity-checked own forward application (model_sha256 = sha256 of the model.pt file)
  parity_with_source(t, k, T) -> {"ok": bool, ...}  bitwise vs the dpc teacher unit tea__s{k}__{t}/teacher.npz
  reference(label, k) -> dict(row_id, c1, c2, p1, p2, d1, d2, r1, r2[, cells1, cells2, n_cells1, n_cells2])
  admission_record() -> dict (private receipt; public copy provenance/ADMISSION_RESULT.json)
  run(D=None) -> admits everything and writes the receipts (call once at the top of the admit stage)

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m qpc.admit plan     -> SOURCE_ADMISSION.json (hash-only)
    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m qpc.admit run      -> admission (lead, via qpc.run)
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from dpc import admit as DA
from qpc import data as QD

HOME = Path.home()
CACHE = HOME / "PCRL_eval_cache_private"
DPC_PRIV = CACHE / "dpc_v1"
DPC_ADMITTED = DPC_PRIV / "admitted"
DPC_UNITS = DPC_PRIV / "run" / "units"
OSF_UNITS = CACHE / "osf_v1" / "run" / "units"
QPC_PRIV = CACHE / "qpc_v1"
ADMITTED = QPC_PRIV / "admitted"
DERIVED = QPC_PRIV / "inputs"
RECEIPT = ADMITTED / "ADMISSION_RECEIPT.json"
WT = QD.WT
PKG = QD.PKG
PLAN = PKG / "SOURCE_ADMISSION.json"
RESULT = PKG / "provenance" / "ADMISSION_RESULT.json"
SOURCE_SHA = QD.SOURCE_SHA
TEACHER_PIN = QD.TEACHER_PIN
DPC_REL = QD.DPC_REL
OSF_REL = QD.OSF_REL
SEEDS = (0, 1, 2)
TEACHERS = ("U", "RAW-J_b0.3")
REFERENCES = {"E": "official LEACE (dpc ref__s{k}__E <- lc__s{k}__E)",
              "F": "official FARE (dpc ref__s{k}__F <- fare__s{k}__p0__c1, fare__s{k}__p1__c1 + official trees)",
              "F0": "matched no-fairness FARE (dpc ref__s{k}__F0 <- fare__s{k}__p0__Z1, fare__s{k}__p1__Z1 + trees)"}
UNDERLYING = {"E": ("lc__s{k}__E",), "F": ("fare__s{k}__p0__c1", "fare__s{k}__p1__c1"),
              "F0": ("fare__s{k}__p0__Z1", "fare__s{k}__p1__Z1")}
TEACHER_KEYS = ("row_id", "p1", "p2", "d1", "d2", "c1", "c2", "r1", "r2")
DPC_ADMITTED_SUMS_SHA = "6b12328aa5d9702ec7988b9bff2ae13b98dbb6f8f91c0802bc66659c1d3e1963"
REUSED_CODE = ("dpc/admit.py", "dpc/data.py", "osf/data.py")


# ------------------------------------------------------------------ helpers
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(p, nocache=False):
    return DA.sha(p, nocache=nocache)


def jload(p):
    return json.loads(Path(p).read_text())


def scrub(text):
    if re.search(r"/Users/|/Volumes/|/private/|" + re.escape(HOME.name), text):
        raise SystemExit("REFUSED: identifying path or user name in a public file")
    return text


def write_json(path, obj, public=True):
    txt = json.dumps(obj, indent=1, default=str) + "\n"
    if public:
        txt = scrub(txt)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(txt)
    tmp.replace(path)


@functools.lru_cache(maxsize=None)
def pinned_bytes(pin, rel):
    return QD.pinned_bytes(pin, rel)


def pinned(pin, rel):
    return json.loads(pinned_bytes(pin, rel))


def dpc_admission():
    return pinned(SOURCE_SHA, f"{DPC_REL}/ADMISSION.json")


def dpc_eval_lock():
    return pinned(SOURCE_SHA, f"{DPC_REL}/EVALUATION_LOCK.json")


def tea_unit(t, k):
    return f"rel__s{k}__{t}"


def dpc_tea_unit(t, k):
    return f"tea__s{k}__{t}"


def ref_unit(lab, k):
    return f"ref__s{k}__{lab}"


def _check_tk(t, k):
    if t not in TEACHERS or k not in SEEDS:
        raise KeyError(f"teacher {t!r} seed {k!r}")


def _check_lk(lab, k):
    if lab not in REFERENCES or k not in SEEDS:
        raise KeyError(f"reference {lab!r} seed {k!r}")


def pinned_teacher(t, k):
    e = dpc_admission()["teachers"][f"{t}|{k}"]
    if not e["pass"] or e["unit"] != tea_unit(t, k):
        raise SystemExit(f"REFUSED: the pinned dpc admission of {t}|{k} did not pass")
    return e


def pinned_unit_files(unit, k):
    """dpc EVALUATION_LOCK pin of a dpc run unit's files (tea__/ref__)."""
    try:
        return dpc_eval_lock()["seeds"][str(k)]["unit_file_sha256"][unit]
    except KeyError as e:
        raise SystemExit(f"REFUSED: no pinned hash for dpc unit {unit}") from e


def files_match(d: Path, files: dict, nocache=False):
    """(ok, detail): COMPLETE.json lists exactly `files` and every file re-hashes to its pin."""
    if not (d / "COMPLETE.json").exists():
        return False, "COMPLETE.json absent"
    c = jload(d / "COMPLETE.json")
    if c.get("files") != files:
        return False, "COMPLETE.json file list differs from the pin"
    bad = [f for f, h in files.items() if not (d / f).exists() or sha(d / f, nocache=nocache) != h]
    return (not bad), ("all files match" if not bad else f"hash mismatch: {bad}")


def copy_verified(src: Path, dst: Path, files: dict):
    """Pinned dpc copy routine (copies, never moves, never overwrites a differing file; uncached re-read)."""
    return DA.copy_verified(src, dst, files)


# ------------------------------------------------------------------ data (sealed) cache
_D = {}


def _data(D=None):
    if D is not None:
        return D
    if "D" not in _D:
        _D["D"] = QD.load()
    return _D["D"]


def _fit_idx(D):
    return D["idx"][QD.FIT]


# ------------------------------------------------------------------ teachers
def teacher_source(t, k):
    """(dir, which, detail) of the first verified source: the dpc admitted copy, else the osf store."""
    files = pinned_teacher(t, k)["complete_files_sha256"]
    tried = {}
    for which, root in (("dpc_admitted", DPC_ADMITTED), ("osf_store", OSF_UNITS)):
        ok, why = files_match(root / tea_unit(t, k), files)
        tried[which] = why
        if ok:
            return root / tea_unit(t, k), which, tried
    raise SystemExit(f"REFUSED: no verified source for {tea_unit(t, k)}: {tried}")


def admit_teacher_copy(t, k):
    e = pinned_teacher(t, k)
    files = e["complete_files_sha256"]
    dst = ADMITTED / tea_unit(t, k)
    ok, why = files_match(dst, files)
    if ok:
        return dst, {"source": "already admitted (qpc copy verified)", "copy": {"files": len(files) + 1,
                                                                                "copied_now": 0}}
    src, which, tried = teacher_source(t, k)
    cp = copy_verified(src, dst, files)
    if not cp["uncached_reread_matches_pinned"]:
        raise SystemExit(f"REFUSED: the copy of {tea_unit(t, k)} does not re-read to the pinned hashes")
    return dst, {"source": which, "source_checks": tried, "copy": cp}


def forward_application(d: Path, D, model_sha256):
    """Own forward application from an admitted copy (pinned dpc.admit functions). Returns (arrays, checks)."""
    import joblib
    import torch
    torch.set_num_threads(1)
    st = DA.state_dict(d / "model.pt")
    keys = {kk: tuple(v.shape) for kk, v in st.items()}
    chk = {"state_dict_keys_and_shapes_as_expected": keys == DA.EXPECT_KEYS,
           "state_dict_float32": all(str(v.dtype) == "torch.float32" for v in st.values()),
           "model_sha256_recomputed_equals_pin": DA.state_sha(st) == model_sha256}
    H = DA.forward(st, D["X"])
    out = {"row_id": np.asarray(D["row_id"]).copy()}
    for i in (0, 1):
        head = joblib.load(d / f"head_{i}.joblib")
        c, P, hard = DA.head_outputs(head, H[i])
        out.update({f"r{i + 1}": H[i], f"c{i + 1}": c, f"p{i + 1}": P, f"d{i + 1}": hard})
    chk["decision_is_first_index_argmax"] = all(bool(np.array_equal(np.argmax(out[f"p{i}"], 1), out[f"d{i}"]))
                                                for i in (1, 2))
    chk["finite"] = all(bool(np.isfinite(out[x]).all()) for x in TEACHER_KEYS if x != "row_id")
    return out, chk


def bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b))


def parity(mine, dpc_unit_dir: Path, release: Path):
    """Bitwise parity with the dpc teacher unit (all nine arrays) and with the admitted release (r, c, p, hard)."""
    z = np.load(dpc_unit_dir / "teacher.npz", allow_pickle=False)
    vs_unit = {x: bitwise(mine[x], z[x]) for x in TEACHER_KEYS}
    extra = sorted(set(z.files) - set(TEACHER_KEYS))
    r = np.load(release, allow_pickle=False)
    vs_rel = {"row_id": bitwise(mine["row_id"], r["row_id"])}
    for i in (1, 2):
        vs_rel.update({f"r{i}": bitwise(mine[f"r{i}"], r[f"r{i}"]), f"c{i}": bitwise(mine[f"c{i}"], r[f"c{i}"]),
                       f"p{i}": bitwise(mine[f"p{i}"], r[f"p{i}"]), f"d{i}": bitwise(mine[f"d{i}"], r[f"hard{i}"])})
    diffs = {x: (float(np.abs(np.asarray(mine[x], np.float64) - np.asarray(z[x], np.float64)).max())
                 if np.asarray(mine[x]).shape == z[x].shape else float("inf")) for x in TEACHER_KEYS}
    return {"vs_dpc_teacher_unit": vs_unit, "dpc_unit_extra_arrays": extra, "vs_admitted_release": vs_rel,
            "max_abs_diff_vs_dpc_unit": diffs,
            "all_bitwise": all(vs_unit.values()) and all(vs_rel.values()) and not extra}


def p_sha(T):
    h = hashlib.sha256()
    for a in (T["p1"], T["p2"]):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def admit_teacher(t, k, D=None):
    """(arrays, receipt) for one teacher; raises unless every check passes."""
    _check_tk(t, k)
    D = _data(D)
    e = pinned_teacher(t, k)
    d, cp = admit_teacher_copy(t, k)
    dpc_dir = DPC_UNITS / dpc_tea_unit(t, k)
    upin = pinned_unit_files(dpc_tea_unit(t, k), k)
    uok, uwhy = files_match(dpc_dir, upin)
    if not uok:
        raise SystemExit(f"REFUSED: the dpc parity target {dpc_tea_unit(t, k)} fails its pin: {uwhy}")
    mine, fchk = forward_application(d, D, e["model_sha256"])
    par = parity(mine, dpc_dir, d / "release.npz")
    rec = {"unit": tea_unit(t, k), "teacher": t, "seed": k, "epoch": e["epoch"],
           "model_sha256": e["model_sha256"], "complete_files_sha256": e["complete_files_sha256"],
           "admitted_from_osf": e["admitted_from_osf"], "pinned_dpc_admission_pass": bool(e["pass"]),
           "admission_copy": cp, "dpc_parity_unit": dpc_tea_unit(t, k), "dpc_parity_unit_files_sha256": upin,
           "dpc_parity_unit_hash_check": uwhy, "forward": fchk, "parity": par, "p_sha256": p_sha(mine),
           "rows": int(len(mine["row_id"])),
           "release_row_order_equals_D": bitwise(mine["row_id"], D["row_id"])}
    rec["pass"] = bool(all(fchk.values()) and par["all_bitwise"] and rec["release_row_order_equals_D"]
                       and rec["pinned_dpc_admission_pass"])
    if not rec["pass"]:
        raise SystemExit(f"REFUSED: teacher {t}|{k} failed admission parity: "
                         f"{json.dumps({'forward': fchk, 'parity': par}, default=str)}")
    return mine, rec


def teacher(t, k, D=None):
    """Teacher outputs on all 39,170 osf rows (osf D order), rebuilt by the own forward application from the qpc
    admitted copy and required to be bitwise equal to the dpc teacher unit. p_i float64 probabilities, d_i first-index
    argmax decisions, c_i centred logits, r_i 16-d encoder features. Non-array keys: model_sha256 (sha256 of the
    admitted model.pt FILE, the binding dpc.deploy uses), state_sha256 (osf state-dict hash), unit."""
    out, rec = admit_teacher(t, k, D)
    out["model_sha256"] = rec["complete_files_sha256"]["model.pt"]
    out["state_sha256"] = rec["model_sha256"]
    out["unit"] = rec["unit"]
    return out


def parity_with_source(t, k, T):
    """{"ok": bool, ...}: bitwise parity (values, dtype, shape) of T's nine arrays with the dpc teacher unit
    <PRIVATE_CACHE>/dpc_v1/run/units/tea__s{k}__{t}/teacher.npz, whose files must first match the dpc EVALUATION_LOCK
    pin."""
    _check_tk(t, k)
    unit = dpc_tea_unit(t, k)
    upin = pinned_unit_files(unit, k)
    uok, uwhy = files_match(DPC_UNITS / unit, upin)
    if not uok:
        return {"ok": False, "unit": unit, "pin_check": uwhy}
    z = np.load(DPC_UNITS / unit / "teacher.npz", allow_pickle=False)
    eq = {x: (x in T and bitwise(T[x], z[x])) for x in TEACHER_KEYS}
    extra = sorted(set(z.files) - set(TEACHER_KEYS))
    return {"ok": bool(all(eq.values()) and not extra), "unit": unit, "pin_check": uwhy, "files_sha256": upin,
            "bitwise": eq, "dpc_unit_extra_arrays": extra}


# ------------------------------------------------------------------ references
def pinned_reference_entries(lab, k):
    refs = dpc_admission()["references"]
    return {u.format(k=k): refs[u.format(k=k)] for u in UNDERLYING[lab]}


def admit_reference(lab, k):
    _check_lk(lab, k)
    unit = ref_unit(lab, k)
    src = DPC_UNITS / unit
    files = pinned_unit_files(unit, k)
    ok, why = files_match(src, files)
    dst = ADMITTED / unit
    dok, _ = files_match(dst, files)
    if not dok:
        if not ok:
            raise SystemExit(f"REFUSED: dpc reference unit {unit} fails its pin: {why}")
        cp = copy_verified(src, dst, files)
    else:
        cp = {"files": len(files) + 1, "copied_now": 0, "already_identical": len(files) + 1,
              "uncached_reread_matches_pinned": files_match(dst, files, nocache=True)[0]}
    z = np.load(dst / "reference.npz", allow_pickle=False)
    rj = jload(dst / "record.json")
    under = pinned_reference_entries(lab, k)
    # parity with the dpc admitted artifacts (dpc.admit.reference re-verifies their hashes against the osf manifest)
    try:
        src_ref = DA.reference(lab, k)
        par = {x: bitwise(z[x], v) for x, v in src_ref.items() if isinstance(v, np.ndarray)}
        meta = {x: v for x, v in src_ref.items() if not isinstance(v, np.ndarray)}
        par_ok = bool(par) and all(par.values()) and set(par) == set(z.files) and meta == rj.get("admission", {})
        par_note = "bitwise vs dpc.admit.reference (dpc admitted artifacts, hash-verified against the osf manifest)"
    except (RuntimeError, KeyError, SystemExit, FileNotFoundError) as ex:
        par, par_ok, par_note = {}, False, f"dpc admitted artifacts unavailable: {type(ex).__name__}"
    prov = {u: {"pinned_dpc_admission_pass": bool(v["pass"]), "kind": v.get("kind"),
                "alias": v.get("checks", {}).get("alias"),
                "fit_role": v.get("checks", {}).get("fit_role")} for u, v in under.items()}
    rec = {"unit": unit, "label": lab, "seed": k, "description": REFERENCES[lab].format(k=k),
           "dpc_reference_unit_files_sha256": files, "dpc_unit_hash_check": why, "copy": cp,
           "arrays": sorted(z.files), "record_admission": rj.get("admission", {}),
           "parity_with_dpc_admitted_artifacts": {"arrays_bitwise": par, "pass": par_ok, "rule": par_note},
           "underlying_admitted_units": prov}
    rec["provenance_valid"] = bool(cp["uncached_reread_matches_pinned"] and par_ok
                                   and all(p["pinned_dpc_admission_pass"] for p in prov.values()))
    rec["pass"] = rec["provenance_valid"]
    out = {x: z[x].copy() for x in z.files}
    out.update(rj.get("admission", {}))
    return out, rec


def reference(label, k):
    """Admitted reference release (verified copy of the dpc reference unit, parity with the dpc admitted artifacts)."""
    out, rec = admit_reference(label, k)
    if not rec["pass"]:
        raise SystemExit(f"REFUSED: reference {label}|{k} provenance not valid: "
                         f"{json.dumps(rec['parity_with_dpc_admitted_artifacts'], default=str)}")
    return out


# ------------------------------------------------------------------ inputs
def admit_inputs(D=None):
    """Verified copy of the raw input + reconstruction of the authorised derived deploy input (no download)."""
    src = QD.SRC
    rec = {"raw": {"declared_source": QD.INPUT_PUBLIC, "sha256": QD.INPUT_SHA, "present": src.exists()}}
    if not src.exists():
        raise SystemExit("REFUSED: the declared private source input is absent; restore a verified copy (never "
                         "download a replacement)")
    rec["raw"]["source_matches"] = sha(src) == QD.INPUT_SHA
    dpc_copy = DPC_ADMITTED / "inputs" / "adult_jcv.npz"
    rec["raw"]["dpc_admitted_copy_matches"] = dpc_copy.exists() and sha(dpc_copy) == QD.INPUT_SHA
    q = ADMITTED / "inputs" / "adult_jcv.npz"
    if q.exists():
        rec["raw"]["copy"] = "already present" if sha(q) == QD.INPUT_SHA else "DIFFERS (refused)"
        if rec["raw"]["copy"] != "already present":
            raise SystemExit("REFUSED: the qpc input copy differs; nothing overwritten")
    else:
        if not rec["raw"]["source_matches"]:
            raise SystemExit("REFUSED: the declared source input fails its pinned hash")
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, q.with_name(q.name + ".tmp"))
        q.with_name(q.name + ".tmp").replace(q)
        rec["raw"]["copy"] = "copied"
    rec["raw"]["copy_uncached_reread_matches"] = sha(q, nocache=True) == QD.INPUT_SHA
    rec["raw"]["destination"] = "<PRIVATE_CACHE>/qpc_v1/admitted/inputs/adult_jcv.npz"
    rec["derived"] = reconstruct_deploy_input(_data(D))
    rec["pass"] = bool(rec["raw"]["source_matches"] and rec["raw"]["copy_uncached_reread_matches"]
                       and rec["derived"]["pass"])
    return rec


def reconstruct_deploy_input(D, out_dir=None):
    """The one authorised derived input: the 83-column deploy input (X, feature_names) and its schema, rebuilt from
    the pinned loader (the dpc QUICKSTART recipe). Deterministic; rewritten only if absent or different."""
    out_dir = Path(out_dir or DERIVED)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [str(x) for x in D["feature_names"]]
    X = np.ascontiguousarray(D["X"])
    xsha = hashlib.sha256(X.tobytes()).hexdigest()
    schema = json.dumps(names).encode()
    npz, sch = out_dir / "deploy_input.npz", out_dir / "schema.json"
    fresh = True
    if npz.exists() and sch.exists():
        z = np.load(npz, allow_pickle=False)
        fresh = not (np.array_equal(z["X"], X) and [str(x) for x in z["feature_names"]] == names
                     and sch.read_bytes() == schema)
    if fresh:
        tmp = out_dir / "deploy_input.tmp.npz"
        np.savez(tmp, X=X, feature_names=np.asarray(names))
        tmp.replace(npz)
        sch.write_bytes(schema)
    z = np.load(npz, allow_pickle=False)
    ok = bool(np.array_equal(z["X"], X) and z["X"].shape == (QD.N_ROWS, QD.N_PERMITTED)
              and hashlib.sha256("\n".join(names).encode()).hexdigest() == QD.DD.FEATURE_NAMES_SHA)
    return {"what": "deploy input (X n x 83 float, feature_names) + schema.json, from the pinned loader",
            "declared_source": QD.INPUT_PUBLIC + " via qpc.data.load (sealed)",
            "destination": "<PRIVATE_CACHE>/qpc_v1/inputs/{deploy_input.npz, schema.json}",
            "rewritten_now": bool(fresh), "X_sha256": xsha, "X_shape": list(X.shape),
            "schema_sha256": hashlib.sha256(schema).hexdigest(), "feature_names_sha256_equals_pin": ok, "pass": ok}


# ------------------------------------------------------------------ full admission (lead, through qpc.run)
def run(D=None, write=True):
    t0, c0 = time.time(), time.process_time()
    D = _data(D)
    if write:
        ADMITTED.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "qpc-admission-result-v1", "written_at": now(), "source_evidence_commit": SOURCE_SHA,
           "teacher_provenance_commit": TEACHER_PIN,
           "pins_used": {f"{DPC_REL}/{f}": hashlib.sha256(pinned_bytes(SOURCE_SHA, f"{DPC_REL}/{f}")).hexdigest()
                         for f in ("ADMISSION.json", "EVALUATION_LOCK.json", "ROLE_MANIFEST.json")},
           "code_sha256": {f: sha(WT / f) for f in ("qpc/admit.py", "qpc/data.py") + REUSED_CODE},
           "reused_code_blob_equals_pin": reused_code_check(),
           "labels_read": "none (no task label and no SEX value is read; assessment labels sealed (-1))",
           "teachers": {}, "references": {}}
    sums = {}
    for t in TEACHERS:
        for k in SEEDS:
            _, r = admit_teacher(t, k, D)
            rec["teachers"][f"{t}|{k}"] = r
            sums.update({f"{tea_unit(t, k)}/{f}": h for f, h in r["complete_files_sha256"].items()})
    for lab in REFERENCES:
        for k in SEEDS:
            _, r = admit_reference(lab, k)
            rec["references"][f"{lab}|{k}"] = r
            sums.update({f"{ref_unit(lab, k)}/{f}": h for f, h in r["dpc_reference_unit_files_sha256"].items()})
    rec["inputs"] = admit_inputs(D)
    sums["inputs/adult_jcv.npz"] = QD.INPUT_SHA
    failed = [f"teacher:{x}" for x, v in rec["teachers"].items() if not v["pass"]]
    failed += [f"reference:{x}" for x, v in rec["references"].items() if not v["pass"]]
    failed += [] if rec["inputs"]["pass"] else ["inputs"]
    failed += [f"reused_code:{f}" for f, ok in rec["reused_code_blob_equals_pin"].items() if not ok]
    if write:
        (ADMITTED / "SHA256SUMS").write_text("".join(f"{h}  {r}\n" for r, h in sorted(sums.items())))
        rec["SHA256SUMS_sha256"] = sha(ADMITTED / "SHA256SUMS")
    rec["files_admitted"] = len(sums)
    rec["verdict"] = "ADMITTED" if not failed else "REFUSED"
    rec["checks_failed"] = failed
    rec["summary"] = {"teachers_admitted_bitwise": int(sum(v["pass"] for v in rec["teachers"].values())),
                      "references_provenance_valid": int(sum(v["provenance_valid"]
                                                             for v in rec["references"].values())),
                      "teacher_p_sha256": {x: v["p_sha256"] for x, v in rec["teachers"].items()}}
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["cpu_s"] = round(time.process_time() - c0, 1)
    rec["privacy"] = "aggregates, hashes, booleans and placeholders only; no row ids, labels or per-person values"
    if write:
        write_json(RECEIPT, rec, public=True)
        write_json(RESULT, rec, public=True)
    if failed:
        raise SystemExit(f"REFUSED: admission failed: {failed}")
    return rec


def admission_record():
    if RECEIPT.exists():
        return jload(RECEIPT)
    if RESULT.exists():
        return jload(RESULT)
    return {}


def reused_code_check():
    pins = {"dpc/admit.py": SOURCE_SHA, "dpc/data.py": SOURCE_SHA, "osf/data.py": TEACHER_PIN}
    return {f: (WT / f).read_bytes() == pinned_bytes(pin, f) for f, pin in pins.items()}


# ------------------------------------------------------------------ public plan (pre-lock, hash-only)
def remote_head(branch):
    r = subprocess.run(["git", "-C", str(WT), "ls-remote", "origin", f"refs/heads/{branch}"], capture_output=True,
                       text=True)
    return r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None


def plan():
    """SOURCE_ADMISSION.json: pins, file hashes at the pinned SHAs, roles, raw-input custody and the admission plan
    with every pinned hash; plus a hash-only pre-check of the private sources (no forward pass, no copy)."""
    man = jload(PKG / "ROLE_MANIFEST.json") if (PKG / "ROLE_MANIFEST.json").exists() else QD.write_role_manifest()
    pin_files = {SOURCE_SHA: [f"{DPC_REL}/{f}" for f in ("ADMISSION.json", "EVALUATION_LOCK.json", "ROLE_MANIFEST.json",
                                                         "MODEL_MANIFEST.json", "EXPOSURE_LEDGER.md",
                                                         "SOURCE_INDEX.json", "BACKUP_VERIFICATION.json",
                                                         "RESTORE_INDEX.json", "QUICKSTART.md",
                                                         "provenance/predecessor_custody/STATUS.json")]
                 + list(REUSED_CODE[:2]) + ["dpc/closeout.py", "dpc/run.py"],
                 TEACHER_PIN: [f"{OSF_REL}/{f}" for f in ("MODEL_MANIFEST.json", "ADMISSION.json",
                                                          "ROLE_MANIFEST.json")] + ["osf/data.py"]}
    files = {}
    for pin, rels in pin_files.items():
        for rel in rels:
            b = pinned_bytes(pin, rel)
            wt = WT / rel
            files[rel] = {"pin": pin, "sha256_at_pin": hashlib.sha256(b).hexdigest(), "bytes": len(b),
                          "working_tree_equals_pin": wt.exists() and wt.read_bytes() == b}
    teachers, refs = {}, {}
    for t in TEACHERS:
        for k in SEEDS:
            e = pinned_teacher(t, k)
            src_ok = {w: files_match(root / tea_unit(t, k), e["complete_files_sha256"])[0]
                      for w, root in (("dpc_admitted", DPC_ADMITTED), ("osf_store", OSF_UNITS))}
            upin = pinned_unit_files(dpc_tea_unit(t, k), k)
            teachers[f"{t}|{k}"] = {
                "artifact_unit": tea_unit(t, k), "epoch": e["epoch"], "model_sha256": e["model_sha256"],
                "complete_files_sha256": e["complete_files_sha256"], "admitted_from_osf": e["admitted_from_osf"],
                "pinned_dpc_admission_pass": bool(e["pass"]),
                "parity_target": {"unit": dpc_tea_unit(t, k), "files_sha256": upin,
                                  "present_and_matching_now": files_match(DPC_UNITS / dpc_tea_unit(t, k), upin)[0]},
                "sources_present_and_matching_now": src_ok}
    for lab in REFERENCES:
        for k in SEEDS:
            u = ref_unit(lab, k)
            upin = pinned_unit_files(u, k)
            under = pinned_reference_entries(lab, k)
            refs[f"{lab}|{k}"] = {"unit": u, "description": REFERENCES[lab].format(k=k), "files_sha256": upin,
                                  "present_and_matching_now": files_match(DPC_UNITS / u, upin)[0],
                                  "underlying_admitted_units": {
                                      x: {"complete_files_sha256": v["complete_files_sha256"],
                                          "pinned_dpc_admission_pass": bool(v["pass"]),
                                          "alias": v.get("checks", {}).get("alias"),
                                          "present_and_matching_now": files_match(DPC_ADMITTED / x,
                                                                                  v["complete_files_sha256"])[0]}
                                      for x, v in under.items()}}
    dsum = DPC_ADMITTED / "SHA256SUMS"
    obj = {
        "schema": "qpc-source-admission-v1", "written_at": now(), "study": "pcrl_confidence_capacity_v1",
        "branch": QD.BRANCH,
        "source_pins": {
            "source_evidence": {"branch": QD.SOURCE_BRANCH, "commit": SOURCE_SHA,
                                "remote_head_now": remote_head(QD.SOURCE_BRANCH)},
            "teacher_provenance": {"branch": "research/pcrl-online-strength-frontier-v1", "commit": TEACHER_PIN,
                                   "remote_head_now": remote_head("research/pcrl-online-strength-frontier-v1")},
            "study_branch_created_from": SOURCE_SHA,
            "study_branch_parent_is_source": subprocess.run(
                ["git", "-C", str(WT), "merge-base", "--is-ancestor", SOURCE_SHA, "HEAD"]).returncode == 0},
        "files_at_pins": files,
        "code": {"qpc/data.py": sha(WT / "qpc/data.py"), "qpc/admit.py": sha(WT / "qpc/admit.py"),
                 "reused_code_blob_equals_pin": reused_code_check(),
                 "reuse": "dpc.admit.forward / head_outputs / state_sha / copy_verified / reference and dpc.data / "
                          "osf.data are imported unchanged at their pins (provenance: blobs above)"},
        "roles": {"loader": man["loader"], "counts": man["counts"], "group_isolation_pass": man["group_isolation"]["pass"],
                  "defense_fit_groups_shared_with": man["group_isolation"]["defense_fit_groups_shared_with"],
                  "all_equal_to_pinned_manifests": man["equality_with_pinned_manifests"]["all_equal"],
                  "role_manifest_sha256": sha(PKG / "ROLE_MANIFEST.json"),
                  "assessment_labels_sealed": man["label_sealing"]["assessment_labels_minus_one"],
                  "unseal_gate": man["label_sealing"]["unseal_gate"]},
        "raw_input_custody": {
            "file": QD.INPUT_PUBLIC, "sha256_pinned": QD.INPUT_SHA, "present": QD.SRC.exists(),
            "sha256_recomputed_matches": QD.SRC.exists() and sha(QD.SRC) == QD.INPUT_SHA,
            "bytes": QD.SRC.stat().st_size if QD.SRC.exists() else None,
            "dpc_admitted_copy_matches": (DPC_ADMITTED / "inputs" / "adult_jcv.npz").exists()
            and sha(DPC_ADMITTED / "inputs" / "adult_jcv.npz") == QD.INPUT_SHA,
            "plan": "verified copy to <PRIVATE_CACHE>/qpc_v1/admitted/inputs/adult_jcv.npz at admission so a qpc "
                    "backup restores from the copy alone; nothing downloaded",
            "removed_scratch_inputs": {
                "what": "the dpc closeout removed the session-scratch deploy inputs (83-column X + feature_names npz "
                        "and schema.json; dpc COST_AND_CLOSEOUT.md, closeout step 4); they were never in git",
                "action": "reconstructed at admission from the declared private source through the pinned loader into "
                          "<PRIVATE_CACHE>/qpc_v1/inputs/ (qpc.admit.reconstruct_deploy_input); no replacement data "
                          "is downloaded and no other derived input is authorised"}},
        "dpc_admitted_store": {"root": "<PRIVATE_CACHE>/dpc_v1/admitted (read-only for qpc)",
                               "SHA256SUMS_sha256_pinned": DPC_ADMITTED_SUMS_SHA,
                               "SHA256SUMS_sha256_now": sha(dsum) if dsum.exists() else None},
        "teacher_admission_plan": {
            "rule": "verified copy from the dpc admitted copy (fallback: osf store) to <PRIVATE_CACHE>/qpc_v1/admitted/"
                    "rel__s{k}__{t}; own forward application (pinned dpc.admit functions) on all 39,170 rows; BITWISE "
                    "parity (values, dtype, shape) of row_id, p1, p2, d1, d2, c1, c2, r1, r2 with the dpc teacher unit "
                    "and of r, c, p, hard with the admitted release, before any code is fitted",
            "teachers": teachers},
        "reference_admission_plan": {
            "rule": "verified copy of the dpc reference unit (hash = dpc EVALUATION_LOCK pin) to <PRIVATE_CACHE>/qpc_v1/"
                    "admitted/ref__s{k}__{label}; bitwise array parity with dpc.admit.reference on the dpc admitted "
                    "artifacts; provenance valid iff every underlying pinned dpc admission passed and hashes and "
                    "arrays match now",
            "references": refs},
        "precheck_all_sources_present_and_matching": all(
            v["parity_target"]["present_and_matching_now"] and any(v["sources_present_and_matching_now"].values())
            for v in teachers.values()) and all(
            v["present_and_matching_now"] and all(u["present_and_matching_now"]
                                                  for u in v["underlying_admitted_units"].values())
            for v in refs.values()),
        "result_file": "provenance/ADMISSION_RESULT.json (written by the admit stage; not part of this plan)",
        "labels_read": "none",
        "privacy": "hashes, booleans and placeholders only"}
    write_json(PLAN, obj)
    return obj


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    if argv == ["plan"]:
        o = plan()
        print(json.dumps({"precheck": o["precheck_all_sources_present_and_matching"],
                          "remote": o["source_pins"]}, indent=1))
    elif argv == ["run"]:
        r = run()
        print(json.dumps({"verdict": r["verdict"], "failed": r["checks_failed"], **r["summary"],
                          "wall_s": r["wall_s"]}, indent=1))
    else:
        raise SystemExit("usage: python -m qpc.admit plan|run")


if __name__ == "__main__":
    main()
