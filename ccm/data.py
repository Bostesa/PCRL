"""Access guard, label allowlist, frozen reference vectors and the toy-law construction check of the
confidence-constrained mechanism sprint (ccm; role F; PROTOCOL.md sections 2, 3 and 6).

LOAD (load()). The pinned hcal loader hcal.data.load(verify=True, unseal=False) is called unchanged (sealed: assessment
labels are -1; it returns the lra / qpc / dpc / osf D with every pinned-blob, input-hash and role-manifest check, plus
D["hcal"]["roles"]). Its result is RESTRICTED before anything is returned. The returned dict W holds ONLY the rows of
    OSF_DEFENSE_FIT       15,434 rows   reference vectors for geometry and (pilot) defense fitting
    CALIBRATION_HELDOUT    2,000 rows   the hcal representatives; held-out geometry (label-free)
    ATTACK_FIT_NEW         4,064 rows   new attackers (pilot only)
    INNER_SELECTION        2,235 rows   bounded exploratory pilot (after the pushed PILOT_LOCK)
    HEAD_VALIDATION        1,500 rows   features only; its labels are never readable
in source row order: row_id, unit (exact-record group), role, X (the 83 permitted columns), feature_names, and
idx[role] = positions into W (plus CALIBRATION_TRAIN_MATCHED, a label-free position subset of OSF_DEFENSE_FIT). W
contains NO label array of any role. Every OSF_DEVELOPMENT_ASSESSMENT row (13,936) is dropped before W exists, and so is
the one AUDIT_FIT row that is a non-representative member of a CALIBRATION_HELDOUT group (it is in no ccm role). The
guard asserts: no assessment row id survives; no returned array has the assessment length (13,936) or the full source
length (39,170); every role has its pinned row / group count; the role row-id and group-id hashes equal the committed
hcal ROLE_MANIFEST.json (read with git show at the hcal tip); the five roles are disjoint.

LABELS (labels(procedure, role)). The ONLY way to read a label. A request passes (1) the ccm allowlist ALLOW below, then
(2) the pushed PILOT_LOCK.json (present, committed at HEAD and byte-identical on origin/<branch>, fetched now), then
(3) the hcal allowlist of the pinned loader (hcal procedure in brackets):
    geometry      no labels at all (every request refused)
    defense_fit   SEX of OSF_DEFENSE_FIT          privacy-aware assignment fitting; pilot only   [attack_diagnostic]
    attack        SEX of ATTACK_FIT_NEW and of INNER_SELECTION                ; pilot only   [attack]
    utility       income and occupation labels of INNER_SELECTION             ; pilot only   [selection]
Everything else is refused, before any data is touched: HEAD_VALIDATION labels always, CALIBRATION_HELDOUT labels (the
frozen hcal temperature is reused, never refitted), AUDIT_FIT, OSF_DEVELOPMENT_ASSESSMENT, unknown procedures. Labels
are never stored in W or in this module: an allowed call re-runs the sealed loader, extracts only the requested rows of
the requested role, aligns them to W by row id and refuses any sealed (-1) value.

REFERENCES (reference(k, role)). For seed k in {0, 1, 2} and role in REFERENCE_ROLES (OSF_DEFENSE_FIT,
CALIBRATION_HELDOUT, ATTACK_FIT_NEW, INNER_SELECTION):
    U0[i]   = dpc.deploy.teacher_probs(model, heads, X) of the admitted frozen U teacher <HCAL_PRIV>/admitted/rel__s{k}__U
              (dpc.deploy.load_teacher(unit_dir, seed=k): COMPLETE.json hash-complete; model.pt must equal the hcal
              SOURCE_ADMISSION.json teacher_parity["U|k"].model_sha256 and the heads its admitted_teacher_dirs hashes),
              run ONCE on the permitted rows of the four reference roles (one batch; no assessment row is ever passed
              to the teacher, so no assessment prediction is materialised). Batches of 15,434 and 25,233 rows were
              bitwise equal to the admitted all-row outputs; single-row forwards are NOT (a different GEMM path), so
              never recompute U0 row by row: use reference()
    d[i]    = numpy argmax of U0[i] (first index; the source tie rule)
    Ucal[i] = hcal.calib.apply_u(U0[i], tab, d[i]) with tab = the FROZEN H-GLOBAL-TEMP record of
              <HCAL_PRIV>/run/units/calU__s{k}/record.json (tables["H-GLOBAL-TEMP|r1" / "|r2"], alpha), whose sha256 must
              equal the hcal EVALUATION_LOCK.json unit_file_sha256 (git show at the hcal tip). Never refitted.
              argmax(Ucal) == d is asserted on every row (apply_u checks it too).
Results are cached in <PRIVATE_CACHE>/ccm_v1/ref/ref__s{k}.npz (PCRL_CCM_PRIVATE_CACHE) with a sha256 receipt
ref__s{k}.json; a cached file is used only if its sha256, every array hash, the W row-id token, the pinned teacher /
record hashes and the sha256 of this file match (otherwise it is recomputed, with parity, in about a second).
PARITY (bitwise) at every compute: U0 and d of every reference row equal the admitted hcal teacher.npz
(<HCAL_PRIV>/admitted/units/tea__s{k}__U, sha256-checked against SOURCE_ADMISSION.json). teacher.npz holds all 39,170
source rows; it is opened, its arrays are indexed to the permitted reference rows IMMEDIATELY (the full arrays are
dropped in the same expression, never stored, never compared, never written) and only those rows are compared. The
hcal calU u.npz files (calibrated predictions on all rows) are NEVER opened.

TOY LAWS (toy_law_check(law), check_toy_laws()). Role F's own exact construction check of TOY_LAWS.json, independent of
ccm.guard / ccm.geometry / ccm.oracle (none is imported): every subset of the distinct reference values of a recipient
is classified CERTIFIED (a q found and checked against every member and label, exactly in rationals with a rigorous
bracket of exp(+-d), and again in float64 with no tolerance), INFEASIBLE_CLASS (members disagree on the decision) or
INFEASIBLE_NLL (sum_k max p_k > exp(d), the closed-form certificate); an UNDECIDED subset (necessary condition holds but
no certified q) invalidates the law. Admissible partitions are enumerated exhaustively and plug-in MI (nats) / Bayes
accuracy of SEX are reported for identity, decision-only, CLASS, task-only (fewest tokens), local, sequential and joint
partitions (all ties listed as ranges). The stochastic LP arm is not computed here.

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m ccm.sema --label F:manifest -- <python> -m ccm.data manifest
    PYTHONPATH=. <python> -m ccm.data toy-laws-check
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from ccm import ids as I

# ----------------------------------------------------------------------------------------------- roles
FIT = "OSF_DEFENSE_FIT"
HEAD = "HEAD_VALIDATION"
HELD = "CALIBRATION_HELDOUT"
ATT = "ATTACK_FIT_NEW"
INNER = "INNER_SELECTION"
AUDIT = "AUDIT_FIT"
ASSESS = "OSF_DEVELOPMENT_ASSESSMENT"
TRAIN_MATCHED = "CALIBRATION_TRAIN_MATCHED"
ROLES = (FIT, HELD, ATT, INNER, HEAD)                 # rows kept in W (HEAD_VALIDATION: features only)
REFERENCE_ROLES = (FIT, HELD, ATT, INNER)             # reference vectors computed and served
SUBROLES = {TRAIN_MATCHED: FIT}                       # label-free position subsets (no reference, no labels)
EXPECTED = {FIT: (15434, 15428), HELD: (2000, 2000), ATT: (4064, 4061), INNER: (2235, 2234), HEAD: (1500, 1499)}
N_SOURCE = 39170
N_ASSESS = 13936
FORBIDDEN_LENGTHS = (N_SOURCE, N_ASSESS)
N_PERMITTED = 83
KS = {1: 2, 2: 6}                                     # recipient 1 income, recipient 2 occupation

# ----------------------------------------------------------------------------------------------- label allowlist
# procedure -> {role: (label kinds,)}; hcal procedure used for the second gate
ALLOW = {"geometry": {},
         "defense_fit": {FIT: ("sex",)},
         "attack": {ATT: ("sex",), INNER: ("sex",)},
         "utility": {INNER: ("income", "occupation")}}
HCAL_PROCEDURE = {("defense_fit", FIT): "attack_diagnostic", ("attack", ATT): "attack", ("attack", INNER): "attack",
                  ("utility", INNER): "selection"}
PILOT_ONLY = ("defense_fit", "attack", "utility")
ALWAYS_REFUSED_ROLES = (HEAD, ASSESS)
FORBIDDEN = (
    "every label of OSF_DEVELOPMENT_ASSESSMENT (old assessment labels)",
    "every old assessment prediction and every person-level assessment array (hcal / lra outer__ and assessment "
    "units, and the hcal calU__s{k}/u.npz calibrated predictions, which cover all rows: never opened)",
    "teacher.npz rows outside the permitted reference rows (indexed away at load; never stored, compared or written)",
    "HEAD_VALIDATION labels (it selected the deployed heads)",
    "CALIBRATION_HELDOUT labels (the hcal temperature is frozen; no refit)",
    "AUDIT_FIT labels (legacy attacker role; ccm attackers use ATTACK_FIT_NEW)",
    "any label before a pushed PILOT_LOCK.json; any label for the geometry procedure",
    "load(unseal=True) of any pinned loader; any *.assess module")

PILOT_LOCK = I.PKG / "PILOT_LOCK.json"
HCAL_REL = "results/pcrl_heldout_calibration_v1"
REF_DIR_NAME = "ref"
REF_SCHEMA = "ccm-reference-v1"


class AccessRefused(PermissionError):
    """A request outside the ccm access guard or allowlist (never silently narrowed)."""


# ----------------------------------------------------------------------------------------------- helpers
def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def sha_arr(a):
    """hcal.data.sha_arr convention: sha256(dtype + shape + C-order bytes)."""
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def git_show(rev, rel):
    r = subprocess.run(["git", "-C", str(I.WT), "show", f"{rev}:{rel}"], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"REFUSED: {rel} is absent at {rev}")
    return r.stdout


def pinned_json(rel, rev=I.SOURCE_TIP):
    return json.loads(git_show(rev, rel))


def priv():
    """<PRIVATE_CACHE>/ccm_v1 (read at call time so tests may redirect PCRL_CCM_PRIVATE_CACHE)."""
    return Path(os.environ.get("PCRL_CCM_PRIVATE_CACHE") or (Path.home() / "PCRL_eval_cache_private" / "ccm_v1"))


def ref_dir():
    return priv() / REF_DIR_NAME


def _atomic_write(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    tmp.replace(path)


def assess_modules_loaded():
    return sorted(m for m in sys.modules if m.split(".")[-1] == "assess")


# ----------------------------------------------------------------------------------------------- the access guard
def _source():
    """The pinned, sealed hcal D (lra -> qpc -> dpc -> osf). Never unsealed here."""
    from hcal import data as HD
    return HD.load(verify=True, unseal=False)


def _role_positions(D):
    """{ccm role: source positions} from the pinned D (hcal subroles for HELD / ATT / TRAIN_MATCHED)."""
    R = D["hcal"]["roles"]
    pos = {FIT: D["idx"][FIT], HEAD: D["idx"][HEAD], INNER: D["idx"][INNER], HELD: R[HELD], ATT: R[ATT],
           TRAIN_MATCHED: R[TRAIN_MATCHED]}
    return {r: np.sort(np.asarray(p, dtype=np.int64)) for r, p in pos.items()}


def _walk_arrays(obj, path="W"):
    if isinstance(obj, np.ndarray):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_arrays(v, f"{path}[{k!r}]")


def guard_check(W, assess_row_ids, forbidden_lengths=FORBIDDEN_LENGTHS):
    """Raise unless W carries no assessment row and no array of a forbidden length. Returns the check record."""
    rid = np.asarray(W["row_id"], dtype=np.int64)
    survivors = np.intersect1d(rid, np.asarray(assess_row_ids, dtype=np.int64))
    if survivors.size:
        raise AccessRefused(f"REFUSED: {survivors.size} assessment row id(s) survive the guard")
    bad = [p for p, a in _walk_arrays(W) if a.ndim and a.shape[0] in forbidden_lengths]
    if bad:
        raise AccessRefused(f"REFUSED: arrays with an assessment / full-source length: {bad}")
    n = rid.size
    wrong = [p for p, a in _walk_arrays({k: W[k] for k in ("row_id", "unit", "role", "X")}) if a.shape[0] != n]
    if wrong:
        raise AccessRefused(f"REFUSED: misaligned arrays {wrong}")
    lab = [k for k in W if re.search(r"(^|_)(sex|race|y|label|labels|y_income|y_occupation_group)$", str(k))]
    if lab:
        raise AccessRefused(f"REFUSED: label keys in the returned data: {lab}")
    return {"assessment_row_ids_surviving": 0, "arrays_of_forbidden_length": 0, "label_keys": 0,
            "forbidden_lengths": list(forbidden_lengths), "rows": int(n)}


def restrict(D, expected=EXPECTED, forbidden_lengths=FORBIDDEN_LENGTHS, n_columns=N_PERMITTED):
    """W = the permitted rows of the pinned D (module docstring). Pure function of D; reads no label."""
    if not D.get("sealed", False):
        raise AccessRefused("REFUSED: ccm reads only the sealed loader (assessment labels -1)")
    pos = _role_positions(D)
    keep = np.sort(np.concatenate([pos[r] for r in ROLES]))
    if np.unique(keep).size != keep.size:
        raise AccessRefused("REFUSED: the ccm roles overlap")
    a = np.asarray(D["idx"][ASSESS], dtype=np.int64)
    if np.intersect1d(keep, a).size:
        raise AccessRefused("REFUSED: an assessment row lies in a ccm role")
    role = np.full(keep.size, "", dtype="<U32")
    idx = {}
    for r in ROLES + tuple(SUBROLES):
        p = np.searchsorted(keep, pos[r])
        if not np.array_equal(keep[p], pos[r]):
            raise AccessRefused(f"REFUSED: {r} rows are not inside the kept rows")
        idx[r] = p.astype(np.int64)
        if r in ROLES:
            role[p] = r
    for sub, parent in SUBROLES.items():
        if not np.isin(idx[sub], idx[parent]).all():
            raise AccessRefused(f"REFUSED: {sub} is not inside {parent}")
    unit = np.asarray(D["unit"], dtype=np.int64)
    for r, (nr, ng) in expected.items():
        got = (int(idx[r].size), int(np.unique(unit[keep][idx[r]]).size))
        if got != (nr, ng):
            raise AccessRefused(f"REFUSED: {r} has {got} rows/groups; pinned {(nr, ng)}")
    X = np.array(np.asarray(D["X"])[keep], copy=True)
    if X.ndim != 2 or X.shape[1] != n_columns:
        raise AccessRefused(f"REFUSED: X has shape {X.shape}; the pinned schema has {n_columns} columns")
    rid = np.asarray(D["row_id"], dtype=np.int64)[keep].copy()
    W = {"schema": "ccm-data-v1", "row_id": rid, "unit": unit[keep].copy(), "role": role, "X": X,
         "feature_names": np.asarray(D["feature_names"]).copy(), "idx": idx,
         "label_access": "none in this dict: use ccm.data.labels(procedure, role) (allowlist + pushed PILOT_LOCK)",
         "token": sha_arr(rid)}
    W["guard"] = guard_check(W, np.asarray(D["row_id"], dtype=np.int64)[a], forbidden_lengths)
    W["guard"]["dropped"] = {"assessment_rows": int(a.size),
                             "other_rows_in_no_ccm_role": int(len(D["row_id"]) - keep.size - a.size)}
    return W


_W_CACHE = {}


def load(fresh=False):
    """The restricted ccm data dict W (module docstring; cached per process). Sealed source; no label in W."""
    if not fresh and "W" in _W_CACHE:
        return _W_CACHE["W"]
    D = _source()
    W = restrict(D)
    W["guard"]["source_sealed"] = bool(D["sealed"])
    W["guard"]["assess_modules_loaded"] = assess_modules_loaded()
    if W["guard"]["assess_modules_loaded"]:
        raise AccessRefused(f"REFUSED: assess modules loaded: {W['guard']['assess_modules_loaded']}")
    W["receipt"] = role_receipt(W)
    del D
    _W_CACHE["W"] = W
    return W


def role_receipt(W, against_hcal=True):
    """Counts and hashes of every ccm role (hcal.data.sha_arr convention on sorted row ids / unique group ids)."""
    out = {}
    hm = pinned_json(f"{HCAL_REL}/ROLE_MANIFEST.json") if against_hcal else None
    for r in ROLES + tuple(SUBROLES):
        p = W["idx"][r]
        rec = {"rows": int(p.size), "groups": int(np.unique(W["unit"][p]).size),
               "row_id_sha256": sha_arr(np.sort(W["row_id"][p])), "group_id_sha256": sha_arr(np.unique(W["unit"][p]))}
        if hm is not None:
            src = hm["roles"].get(r) or hm["inherited"].get(r)
            rec["equal_to_hcal_role_manifest"] = bool(src is not None and all(
                rec[x] == src[x] for x in ("rows", "groups", "row_id_sha256", "group_id_sha256")))
        out[r] = rec
    if hm is not None and not all(v["equal_to_hcal_role_manifest"] for v in out.values()):
        raise AccessRefused(f"REFUSED: ccm roles differ from the hcal ROLE_MANIFEST: {out}")
    return out


# ----------------------------------------------------------------------------------------------- labels
def pilot_lock_pushed(lock=None, branch=I.BRANCH):
    """(ok, reason): PILOT_LOCK.json exists, is committed at HEAD and is byte-identical on origin/<branch> (fetched)."""
    lock = Path(lock or PILOT_LOCK)
    rel = f"{I.REL}/PILOT_LOCK.json"
    if not lock.exists():
        return False, "PILOT_LOCK.json does not exist"
    local = lock.read_bytes()
    r = subprocess.run(["git", "-C", str(I.WT), "show", f"HEAD:{rel}"], capture_output=True)
    if r.returncode != 0 or r.stdout != local:
        return False, "PILOT_LOCK.json is not committed at HEAD as it is on disk"
    subprocess.run(["git", "-C", str(I.WT), "fetch", "-q", "origin", branch], capture_output=True)
    r = subprocess.run(["git", "-C", str(I.WT), "show", f"origin/{branch}:{rel}"], capture_output=True)
    if r.returncode != 0:
        return False, "PILOT_LOCK.json is not on origin"
    if r.stdout != local:
        return False, "local PILOT_LOCK.json differs from origin"
    try:
        if json.loads(local).get("name") != "PILOT_LOCK":
            return False, "PILOT_LOCK.json does not name PILOT_LOCK"
    except ValueError:
        return False, "PILOT_LOCK.json is not JSON"
    return True, "PILOT_LOCK.json committed and byte-identical on origin"


def allowed_kinds(procedure, role):
    """Label kinds `procedure` may read on `role`, or AccessRefused. No data is touched."""
    if role in ALWAYS_REFUSED_ROLES:
        raise AccessRefused(f"REFUSED: no ccm procedure may read labels of {role}")
    if procedure not in ALLOW:
        raise AccessRefused(f"REFUSED: unknown procedure {procedure!r}")
    if role not in ALLOW[procedure]:
        raise AccessRefused(f"REFUSED: procedure {procedure!r} may not read labels of {role}")
    return ALLOW[procedure][role]


def labels(procedure, role, data=None):
    """{"procedure", "role", "positions" (into W), "row_id", <kind>: int64 labels} through the three gates."""
    kinds = allowed_kinds(procedure, role)
    if procedure in PILOT_ONLY:
        ok, why = pilot_lock_pushed()
        if not ok:
            raise AccessRefused(f"REFUSED: {procedure!r} is pilot-only and {why}")
    from hcal import data as HD
    D = _source()
    W = restrict(D)
    if data is not None and data.get("token") != W["token"]:
        raise AccessRefused("REFUSED: the given data dict is not the restriction of the pinned loader")
    hp = HCAL_PROCEDURE[(procedure, role)]
    pos = W["idx"][role]
    want = W["row_id"][pos]
    out = {"procedure": procedure, "role": role, "positions": pos.copy(), "row_id": want.copy()}
    for kind in kinds:
        if kind == "sex":
            rows, val = HD.sex_labels(D, hp, role)
        else:
            rows, val = HD.task_labels(D, kind, hp, role)
        rid = np.asarray(D["row_id"], dtype=np.int64)[rows]
        o = np.argsort(rid)
        j = np.searchsorted(rid[o], want)
        if j.max(initial=0) >= rid.size or not np.array_equal(rid[o][j], want) or rid.size != want.size:
            raise AccessRefused(f"REFUSED: {role} labels do not align with the ccm rows")
        v = np.asarray(val, dtype=np.int64)[o][j]
        if np.any(v < 0):
            raise AccessRefused(f"REFUSED: sealed labels on {role}")
        out[kind] = v
    del D
    return out


# ----------------------------------------------------------------------------------------------- reference vectors
def teacher_dir(k):
    return I.HCAL_PRIV / "admitted" / f"rel__s{k}__U"


def teacher_npz(k):
    return I.HCAL_PRIV / "admitted" / "units" / f"tea__s{k}__U" / "teacher.npz"


def calu_record(k):
    return I.HCAL_PRIV / "run" / "units" / f"calU__s{k}" / "record.json"


def pins(k):
    """The hcal-committed hashes every reference input must match (git show at the hcal tip)."""
    sa = pinned_json(f"{HCAL_REL}/SOURCE_ADMISSION.json")
    el = pinned_json(f"{HCAL_REL}/EVALUATION_LOCK.json")
    tdir = sa["admitted_teacher_dirs"][f"rel__s{k}__U"]
    return {"model_sha256": sa["teacher_parity"][f"U|{k}"]["model_sha256"], "model_dir_sha256": tdir["model.pt"],
            "head_0_sha256": tdir["head_0.joblib"], "head_1_sha256": tdir["head_1.joblib"],
            "teacher_npz_sha256": sa["units"][f"tea__s{k}__U"]["files_sha256"]["teacher.npz"],
            "calU_record_sha256": el["seeds"][str(k)]["unit_file_sha256"][f"calU__s{k}"]["record.json"],
            "teacher_parity_ok_at_admission": bool(sa["teacher_parity"][f"U|{k}"]["ok"])}


def frozen_tables(k, pin=None):
    """{1: H-GLOBAL-TEMP|r1 record, 2: ...|r2 record} of calU__s{k}, hash-checked; never refitted."""
    pin = pin or pins(k)
    p = calu_record(k)
    b = p.read_bytes()
    if sha_bytes(b) != pin["calU_record_sha256"]:
        raise AccessRefused(f"REFUSED: calU__s{k}/record.json differs from the hcal EVALUATION_LOCK hash")
    rec = json.loads(b)
    out = {}
    for i in (1, 2):
        t = rec["tables"][f"H-GLOBAL-TEMP|r{i}"]
        if (t["kind"], t["target"], t["K"], t["alphas"]) != ("H-GLOBAL-TEMP", "U", KS[i], None):
            raise AccessRefused(f"REFUSED: calU__s{k} r{i} is not a global U temperature record")
        out[i] = t
    return out


def apply_frozen(P, tab, d):
    """Ucal = hcal.calib.apply_u(P, frozen tab, d); asserts argmax(Ucal) == d on every row."""
    from hcal import calib as C
    q = C.apply_u(np.asarray(P, dtype=np.float64), tab, np.asarray(d, dtype=np.int64))
    mis = int(np.sum(np.argmax(q, axis=1) != d))
    assert mis == 0, f"Ucal changed the decision on {mis} row(s)"
    return q


def _reference_rows(W):
    return np.sort(np.concatenate([W["idx"][r] for r in REFERENCE_ROLES]))


def teacher_parity(k, rows_rid, U0, d, pin):
    """Bitwise U0 / d parity against the admitted teacher.npz on the permitted reference rows only (module docstring)."""
    p = teacher_npz(k)
    if sha_file(p) != pin["teacher_npz_sha256"]:
        raise AccessRefused(f"REFUSED: tea__s{k}__U/teacher.npz differs from the admitted hash")
    with np.load(p, allow_pickle=False) as z:
        trid = np.asarray(z["row_id"], dtype=np.int64)
        o = np.argsort(trid)
        j = o[np.searchsorted(trid[o], rows_rid)]
        if not np.array_equal(trid[j], rows_rid):
            raise AccessRefused("REFUSED: reference rows missing from teacher.npz")
        ref = {}
        for i in (1, 2):                       # index the permitted rows immediately; the full array is not kept
            ref[f"p{i}"] = z[f"p{i}"][j]
            ref[f"d{i}"] = z[f"d{i}"][j]
        del trid, o
    out = {}
    for i in (1, 2):
        out[f"p{i}_bitwise"] = bool(np.array_equal(U0[i], ref[f"p{i}"]))
        out[f"d{i}_bitwise"] = bool(np.array_equal(d[i], ref[f"d{i}"]))
    out["rows_compared"] = int(rows_rid.size)
    out["ok"] = all(v for x, v in out.items() if x.endswith("_bitwise"))
    return out


def compute_reference(k, W):
    """Forward pass of the admitted teacher on the permitted reference rows + frozen temperature (no caching)."""
    from dpc import deploy as DD
    pin = pins(k)
    tdir = teacher_dir(k)
    have = {"model": sha_file(tdir / "model.pt"), "h0": sha_file(tdir / "head_0.joblib"),
            "h1": sha_file(tdir / "head_1.joblib")}
    if (have["model"] != pin["model_sha256"] or have["model"] != pin["model_dir_sha256"]
            or have["h0"] != pin["head_0_sha256"] or have["h1"] != pin["head_1_sha256"]):
        raise AccessRefused(f"REFUSED: admitted teacher rel__s{k}__U differs from the hcal SOURCE_ADMISSION hashes")
    model, heads, msha = DD.load_teacher(tdir, seed=k)
    if msha != pin["model_sha256"]:
        raise AccessRefused("REFUSED: loaded model.pt hash differs from the pin")
    rows = _reference_rows(W)
    X = W["X"][rows]
    P = DD.teacher_probs(model, heads, X)
    U0 = {i: np.asarray(P[i - 1], dtype=np.float64) for i in (1, 2)}
    d = {i: np.argmax(U0[i], axis=1).astype(np.int64) for i in (1, 2)}
    tabs = frozen_tables(k, pin)
    Ucal = {i: apply_frozen(U0[i], tabs[i], d[i]) for i in (1, 2)}
    parity = teacher_parity(k, W["row_id"][rows], U0, d, pin)
    if not parity["ok"]:
        raise AccessRefused(f"REFUSED: U0 parity with the admitted teacher.npz fails for seed {k}: {parity}")
    strict = {i: int(np.sum(np.sort(Ucal[i], axis=1)[:, -1] > np.sort(Ucal[i], axis=1)[:, -2])) for i in (1, 2)}
    arrays = {"row_id": W["row_id"][rows], "positions": rows}
    for i in (1, 2):
        arrays[f"U0_{i}"], arrays[f"Ucal_{i}"], arrays[f"d_{i}"] = U0[i], Ucal[i], d[i]
    rec = {"schema": REF_SCHEMA, "seed": k, "token": W["token"], "roles": list(REFERENCE_ROLES),
           "rows": int(rows.size), "pins": pin, "observed": have,
           "alpha": {str(i): float(tabs[i]["alpha"]) for i in (1, 2)},
           "calU_content_sha256": {str(i): tabs[i]["content_sha256"] for i in (1, 2)},
           "decision_rule": "numpy argmax of U0, first index", "parity": parity,
           "argmax_preserved_rows": {str(i): int(np.sum(np.argmax(Ucal[i], 1) == d[i])) for i in (1, 2)},
           "strict_argmax_rows": {str(i): strict[i] for i in (1, 2)},
           "array_sha256": {x: sha_arr(v) for x, v in arrays.items()},
           "code_sha256": sha_file(Path(__file__)),
           "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return arrays, rec


def _ref_paths(k):
    return ref_dir() / f"ref__s{k}.npz", ref_dir() / f"ref__s{k}.json"


def _cached(k, W):
    pz, pj = _ref_paths(k)
    if not (pz.exists() and pj.exists()):
        return None
    rec = json.loads(pj.read_text())
    if rec.get("schema") != REF_SCHEMA or rec.get("token") != W["token"] or sha_file(pz) != rec.get("npz_sha256"):
        return None
    if rec.get("pins") != pins(k) or rec.get("code_sha256") != sha_file(Path(__file__)):
        return None                                   # recomputed by the code in this tree (cheap; bitwise parity)
    with np.load(pz, allow_pickle=False) as z:
        arrays = {x: z[x] for x in z.files}
    if any(sha_arr(arrays[x]) != h for x, h in rec["array_sha256"].items()):
        return None
    for i in (1, 2):
        assert np.array_equal(np.argmax(arrays[f"Ucal_{i}"], 1), arrays[f"d_{i}"]), "cached Ucal changes a decision"
    return arrays, rec


def reference_all(k, W=None, use_cache=True):
    """(arrays over the union of reference rows, receipt) for seed k, from the verified cache or computed + cached."""
    if k not in I.SEEDS:
        raise ValueError(f"seed {k} not in {I.SEEDS}")
    W = W if W is not None else load()
    got = _cached(k, W) if use_cache else None
    if got is not None:
        return got
    arrays, rec = compute_reference(k, W)
    pz, pj = _ref_paths(k)
    pz.parent.mkdir(parents=True, exist_ok=True)
    tmp = pz.with_name(pz.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    rec["npz_sha256"] = sha_file(tmp)
    tmp.replace(pz)
    _atomic_write(pj, (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode())
    return arrays, rec


def reference(k, role, W=None, use_cache=True):
    """Reference vectors of seed k on one permitted role:
    {"U0": {1: (n, 2), 2: (n, 6)}, "Ucal": {...}, "d": {1, 2}, "row_id", "positions" (into W), "alpha", "seed",
     "role"}."""
    if role not in REFERENCE_ROLES:
        raise AccessRefused(f"REFUSED: reference vectors are served only for {REFERENCE_ROLES}; not {role}")
    W = W if W is not None else load()
    arrays, rec = reference_all(k, W, use_cache)
    pos = W["idx"][role]
    j = np.searchsorted(arrays["positions"], pos)
    if not np.array_equal(arrays["positions"][j], pos):
        raise AccessRefused(f"REFUSED: cached reference rows do not cover {role}")
    out = {"seed": k, "role": role, "positions": pos.copy(), "row_id": arrays["row_id"][j].copy(),
           "U0": {i: arrays[f"U0_{i}"][j].copy() for i in (1, 2)},
           "Ucal": {i: arrays[f"Ucal_{i}"][j].copy() for i in (1, 2)},
           "d": {i: arrays[f"d_{i}"][j].copy() for i in (1, 2)},
           "alpha": {i: rec["alpha"][str(i)] for i in (1, 2)}}
    assert np.array_equal(out["row_id"], W["row_id"][pos])
    return out


# ----------------------------------------------------------------------------------------------- toy laws
TOY_LAWS = I.PKG / "TOY_LAWS.json"
LAW_NAMES = ("NULL", "POSITIVE", "COMPLEMENTARY", "NO_COALITION", "JOINT_HEADROOM")
G_D = Fraction(1, 200)              # d = 0.005
G_B = Fraction(1, 400)              # b = 0.0025
CAPACITY = {1: 8, 2: 64}            # tokens per predicted class (i8o64); laws have <= 6 inputs, so it never binds
MI_TIE = 1e-12                      # argmin sets of float MI values (ties are reported as ranges)


def exp_bracket(x: Fraction, terms=24):
    """Rational (lo, hi) with lo < exp(x) < hi for |x| <= 1/2: Taylor partial sum +- 2 |x|^n / n! (Lagrange)."""
    s, t = Fraction(0), Fraction(1)
    for n in range(terms):
        s += t
        t = t * x / (n + 1)
    r = 2 * abs(t)
    return s - r, s + r


EXP_D = exp_bracket(G_D)
EXP_NEG_D = exp_bracket(-G_D)


def fr(x):
    return Fraction(str(x))


def decision(p):
    """First-index argmax."""
    k = 0
    for j in range(1, len(p)):
        if p[j] > p[k]:
            k = j
    return k


def _certify(q, members, dec):
    """Exact (rational, rigorous exp bracket) and float64 (no tolerance) check of q against every member and label."""
    K = len(q)
    if sum(q) != 1 or min(q) < 0 or any(q[dec] <= q[k] for k in range(K) if k != dec):
        return False
    qf = [float(x) for x in q]
    if any(qf[dec] <= qf[k] for k in range(K) if k != dec) or min(qf) < 0 or abs(math.fsum(qf) - 1.0) > 1e-12:
        return False
    emd = math.exp(-float(G_D))
    for p in members:
        pf = [float(x) for x in p]
        if any(not q[k] >= EXP_NEG_D[1] * p[k] for k in range(K)):
            return False
        if any(not qf[k] >= emd * pf[k] for k in range(K)):
            return False
        ex = sum(x * x for x in q) - sum(x * x for x in p)
        exf = sum(x * x for x in qf) - sum(x * x for x in pf)
        for y in range(K):
            if ex - 2 * (q[y] - p[y]) > G_B or exf - 2 * (qf[y] - pf[y]) > float(G_B):
                return False
    return True


def _slack(q, members):
    K = len(q)
    nll = min(q[k] - EXP_NEG_D[1] * p[k] for p in members for k in range(K))
    br = min(G_B - (sum(x * x for x in q) - sum(x * x for x in p) - 2 * (q[y] - p[y]))
             for p in members for y in range(K))
    return float(nll), float(br)


def _candidates(members, dec):
    K = len(members[0])
    M = [max(p[k] for p in members) for k in range(K)]
    S = sum(M)
    mean = [sum(p[k] for p in members) / len(members) for k in range(K)]
    nm = [x / S for x in M]
    out = [list(p) for p in members] + [nm, mean, [(a + b) / 2 for a, b in zip(nm, mean)]]
    out += [[(a + b) / 2 for a, b in zip(p, mean)] for p in members]
    for p, r in itertools.combinations(members, 2):
        out.append([(a + b) / 2 for a, b in zip(p, r)])
    # a fine 1-parameter family between the normalised maximum and each member
    for p in members:
        for w in range(1, 10):
            t = Fraction(w, 10)
            out.append([t * a + (1 - t) * b for a, b in zip(nm, p)])
    return out


def bin_status(members):
    """('CERTIFIED', q) | ('INFEASIBLE_CLASS', None) | ('INFEASIBLE_NLL', None) | ('UNDECIDED', None)."""
    decs = {decision(p) for p in members}
    if len(decs) > 1:
        return "INFEASIBLE_CLASS", None
    dec = decs.pop()
    S = sum(max(p[k] for p in members) for k in range(len(members[0])))
    if S > EXP_D[1]:
        return "INFEASIBLE_NLL", None
    if S < EXP_D[0]:
        for q in _candidates(members, dec):
            if _certify(q, members, dec):
                return "CERTIFIED", q
    return "UNDECIDED", None


def _law_inputs(law):
    xs = law["inputs"]
    P = [fr(x["P"]) for x in xs]
    s = [fr(x["sex1"]) for x in xs]
    if sum(P) != 1 or min(P) <= 0 or not all(0 <= v <= 1 for v in s):
        raise ValueError(f"{law['name']}: P must be a positive law and P(SEX=1|x) in [0, 1]")
    vec = {}
    for i in (1, 2):
        vec[i] = [tuple(fr(v) for v in x[f"p{i}"]) for x in xs]
        K = law["recipients"][str(i)]["K"]
        for x, p in zip(xs, vec[i]):
            if len(p) != K or sum(p) != 1 or min(p) <= 0:
                raise ValueError(f"{law['name']} {x['id']} p{i}: positive probability vector of length {K} required")
            if sorted(p)[-1] == sorted(p)[-2]:
                raise ValueError(f"{law['name']} {x['id']} p{i}: the decision must be a strict argmax")
        for x in xs:                                        # float columns must equal the exact rationals
            if [float(fr(v)) for v in x[f"p{i}"]] != list(x[f"p{i}_float"]):
                raise ValueError(f"{law['name']} {x['id']} p{i}_float differs from float(p{i})")
    for x in xs:
        if float(fr(x["P"])) != x["P_float"] or float(fr(x["sex1"])) != x["sex1_float"]:
            raise ValueError(f"{law['name']} {x['id']}: float columns differ from the rationals")
    return P, s, vec


def _partitions(n, admissible):
    full = (1 << n) - 1

    def rec(rest):
        if rest == 0:
            yield ()
            return
        first = rest & -rest
        sub = rest
        while sub:
            if sub & first and sub in admissible:
                for tail in rec(rest & ~sub):
                    yield (sub,) + tail
            sub = (sub - 1) & rest
    yield from rec(full)


def _kl_bern(a, b):
    out = 0.0
    if a > 0:
        out += float(a) * math.log(float(a) / float(b))
    if a < 1:
        out += float(1 - a) * math.log(float(1 - a) / float(1 - b))
    return out


def measures(cells, sbar):
    """cells: {key: [P(t), P(t, S=1)]} -> (plug-in MI in nats, exact Bayes accuracy)."""
    mi = math.fsum(float(pt) * _kl_bern(ps / pt, sbar) for pt, ps in cells.values())
    bayes = sum(max(ps, pt - ps) for pt, ps in cells.values())
    return max(mi, 0.0), bayes


def _cells(P, s, keys):
    c = {}
    for p, v, key in zip(P, s, keys):
        t = c.setdefault(key, [Fraction(0), Fraction(0)])
        t[0] += p
        t[1] += p * v
    return c


def _rng(vals):
    return [min(vals), max(vals)]


def toy_law_check(law):
    """Exhaustive construction check of one law (module docstring). Returns a JSON-safe record."""
    P, s, vec = _law_inputs(law)
    n_in = len(P)
    sbar = sum(p * v for p, v in zip(P, s))
    out = {"inputs": n_in, "P_sex1": str(sbar), "I_S_X_nats": measures(_cells(P, s, range(n_in)), sbar)[0],
           "recipients": {}}
    parts, value_of, decs_of = {}, {}, {}
    for i in (1, 2):
        vals = list(dict.fromkeys(vec[i]))
        value_of[i] = [vals.index(p) for p in vec[i]]
        decs_of[i] = [decision(v) for v in vals]
        n = len(vals)
        subsets, admissible, undecided = [], set(), []
        for m in range(1, 1 << n):
            mem = [vals[j] for j in range(n) if m >> j & 1]
            st, q = bin_status(mem)
            if st == "CERTIFIED":
                admissible.add(m)
            if st == "UNDECIDED":
                undecided.append(m)
            if bin(m).count("1") >= 2:
                idx = [j for j in range(n) if m >> j & 1]
                inputs = [law["inputs"][x]["id"] for x in range(n_in) if value_of[i][x] in idx]
                rec = {"values": idx, "inputs": inputs, "status": st,
                       "sum_max": float(sum(max(v[k] for v in mem) for k in range(len(mem[0])))),
                       "sex_homogeneous": len({s[x] for x in range(n_in) if value_of[i][x] in idx}) == 1}
                if q is not None:
                    rec["q"] = [str(x) for x in q]
                    rec["q_float"] = [float(x) for x in q]
                    rec["min_nll_slack"], rec["min_brier_slack"] = _slack(q, mem)
                subsets.append(rec)
        cap_ok = []
        for pt in _partitions(n, admissible):
            per_class = {}
            for blk in pt:
                c = decs_of[i][(blk & -blk).bit_length() - 1]
                per_class[c] = per_class.get(c, 0) + 1
            if max(per_class.values()) <= CAPACITY[i]:
                cap_ok.append(pt)
        blk_of = []
        for pt in cap_ok:
            m = [0] * n
            for b_id, blk in enumerate(pt):
                for j in range(n):
                    if blk >> j & 1:
                        m[j] = b_id
            blk_of.append(m)
        meas = [measures(_cells(P, s, [m[value_of[i][x]] for x in range(n_in)]), sbar) for m in blk_of]
        ntok = [len(pt) for pt in cap_ok]
        ident = measures(_cells(P, s, value_of[i]), sbar)
        dec_only = measures(_cells(P, s, [decs_of[i][value_of[i][x]] for x in range(n_in)]), sbar)
        class_sets = {}
        for j, c in enumerate(decs_of[i]):
            class_sets[c] = class_sets.get(c, 0) | (1 << j)
        class_ok = all(m in admissible for m in class_sets.values())
        tmin = min(ntok)
        targ = [a for a, t in enumerate(ntok) if t == tmin]
        mimin = min(m[0] for m in meas)
        larg = [a for a, m in enumerate(meas) if m[0] <= mimin + MI_TIE]

        def named(a):
            return [[law["inputs"][x]["id"] for x in range(n_in) if blk_of[a][value_of[i][x]] == blk]
                    for blk in sorted(set(blk_of[a]))]
        out["recipients"][str(i)] = {
            "K": law["recipients"][str(i)]["K"], "distinct_values": n,
            "value_inputs": [[law["inputs"][x]["id"] for x in range(n_in) if value_of[i][x] == j] for j in range(n)],
            "value_decisions": decs_of[i], "undecided_subsets": len(undecided),
            "admissible_multi_value_bins": sum(1 for r in subsets if r["status"] == "CERTIFIED"),
            "admissible_bins_all_sex_homogeneous": all(r["sex_homogeneous"] for r in subsets
                                                       if r["status"] == "CERTIFIED"),
            "subsets": subsets, "admissible_partitions": len(cap_ok),
            "identity": {"tokens": n, "mi": ident[0], "bayes": str(ident[1])},
            "decision_only": {"confidence_eligible": False, "mi": dec_only[0], "bayes": str(dec_only[1])},
            "class": {"eligible": class_ok, **({"mi": dec_only[0], "bayes": str(dec_only[1])} if class_ok else {})},
            "task_only_fewest_tokens": {"tokens": tmin, "argmin_partitions": len(targ),
                                        "partitions": [named(a) for a in targ[:12]],
                                        "mi_range": _rng([meas[a][0] for a in targ]),
                                        "bayes_range": [str(x) for x in _rng([meas[a][1] for a in targ])]},
            "local_min_mi": {"mi": mimin, "argmin_partitions": len(larg), "partitions": [named(a) for a in larg[:12]],
                             "bayes_range": [str(x) for x in _rng([meas[a][1] for a in larg])]},
            "max_mi_over_admissible": max(m[0] for m in meas)}
        parts[i] = {"blk": blk_of, "meas": meas, "targ": targ, "larg": larg,
                    "named": [named(a) for a in range(len(cap_ok))]}
        if undecided:
            raise ValueError(f"{law['name']} r{i}: {len(undecided)} UNDECIDED subset(s); the law is ambiguous")

    def pair_mi(a, b):
        keys = [(parts[1]["blk"][a][value_of[1][x]], parts[2]["blk"][b][value_of[2][x]]) for x in range(n_in)]
        return measures(_cells(P, s, keys), sbar)

    n1, n2 = len(parts[1]["blk"]), len(parts[2]["blk"])
    grid = [[pair_mi(a, b) for b in range(n2)] for a in range(n1)]
    ident_pair = measures(_cells(P, s, list(zip(value_of[1], value_of[2]))), sbar)
    gain = [[grid[a][b][0] - max(parts[1]["meas"][a][0], parts[2]["meas"][b][0]) for b in range(n2)]
            for a in range(n1)]
    jmin = min(grid[a][b][0] for a in range(n1) for b in range(n2))
    jarg = [(a, b) for a in range(n1) for b in range(n2) if grid[a][b][0] <= jmin + MI_TIE]

    def seq(first):
        vals = []
        for a in parts[first]["larg"]:
            row = [grid[a][b][0] if first == 1 else grid[b][a][0] for b in range(n2 if first == 1 else n1)]
            vals.append(min(row))
        return _rng(vals)

    loc = [grid[a][b] for a in parts[1]["larg"] for b in parts[2]["larg"]]
    tsk = [grid[a][b] for a in parts[1]["targ"] for b in parts[2]["targ"]]
    out["coalition"] = {
        "identity_pair": {"mi": ident_pair[0], "bayes": str(ident_pair[1]),
                          "gain_over_better_single": ident_pair[0] - max(out["recipients"]["1"]["identity"]["mi"],
                                                                         out["recipients"]["2"]["identity"]["mi"])},
        "local_pair_mi_range": _rng([g[0] for g in loc]),
        "local_pair_bayes_range": [str(x) for x in _rng([g[1] for g in loc])],
        "local_pair_gain_range": _rng([gain[a][b] for a in parts[1]["larg"] for b in parts[2]["larg"]]),
        "task_only_pair_mi_range": _rng([g[0] for g in tsk]),
        "sequential_1_to_2_mi_range": seq(1), "sequential_2_to_1_mi_range": seq(2),
        "joint_min_mi": jmin, "joint_argmin_pairs": len(jarg),
        "joint_bayes_range": [str(x) for x in _rng([grid[a][b][1] for a, b in jarg])],
        "max_pair_mi_over_admissible": max(grid[a][b][0] for a in range(n1) for b in range(n2)),
        "gain_range_over_all_admissible_pairs": _rng([g for row in gain for g in row]),
        "admissible_pairs": n1 * n2}
    if "joint_argmin_partitions" in law.get("report", ()):          # extra witnesses, only when a law asks for them
        nm = {i: parts[i]["named"] for i in (1, 2)}

        def best_second(a, first):
            row = [grid[a][b][0] for b in range(n2)] if first == 1 else [grid[b][a][0] for b in range(n1)]
            v = min(row)
            return {"first": nm[first][a], "second_best": [nm[3 - first][b] for b, x in enumerate(row)
                                                            if x <= v + MI_TIE], "coalition_mi": v}
        out["coalition"]["witnesses"] = {
            "joint_argmin": [{"recipient_1": nm[1][a], "recipient_2": nm[2][b], "mi": grid[a][b][0],
                              "bayes": str(grid[a][b][1])} for a, b in jarg],
            "local_pairs": [{"recipient_1": nm[1][a], "recipient_2": nm[2][b], "mi": grid[a][b][0],
                             "bayes": str(grid[a][b][1])} for a in parts[1]["larg"] for b in parts[2]["larg"]],
            "sequential_1_to_2": [best_second(a, 1) for a in parts[1]["larg"]],
            "sequential_2_to_1": [best_second(a, 2) for a in parts[2]["larg"]],
            "tie_rule": "every own-MI argmin of the first mover (ties within 1e-12) is listed; the second mover's "
                        "coalition-MI minimum is a single value whatever its tie-break"}
    out["intended_property"] = intended_property(law["name"], out)
    return out


def intended_property(name, c):
    """The registered intended property of each law as numeric predicates on the construction check."""
    r1, r2, co = c["recipients"]["1"], c["recipients"]["2"], c["coalition"]
    if name == "NULL":
        preds = {"every admissible bin is SEX-homogeneous (both recipients)":
                 r1["admissible_bins_all_sex_homogeneous"] and r2["admissible_bins_all_sex_homogeneous"],
                 "some admissible multi-value bin exists (both recipients)":
                 r1["admissible_multi_value_bins"] > 0 and r2["admissible_multi_value_bins"] > 0,
                 "local min MI == identity MI (both)": all(abs(r["local_min_mi"]["mi"] - r["identity"]["mi"]) <= MI_TIE
                                                           for r in (r1, r2)),
                 "joint min coalition MI == identity pair MI": abs(co["joint_min_mi"] - co["identity_pair"]["mi"])
                 <= MI_TIE,
                 "SEX is informative (I(S;X) > 0.05 nats)": c["I_S_X_nats"] > 0.05}
    elif name == "POSITIVE":
        preds = {"local min MI < identity MI - 0.05 (both)": all(r["local_min_mi"]["mi"] < r["identity"]["mi"] - 0.05
                                                                for r in (r1, r2)),
                 "task-only fewest-token partition is unique and keeps identity MI (both)": all(
                     r["task_only_fewest_tokens"]["argmin_partitions"] == 1
                     and abs(r["task_only_fewest_tokens"]["mi_range"][1] - r["identity"]["mi"]) <= MI_TIE
                     for r in (r1, r2)),
                 "joint min coalition MI < identity pair MI - 0.05": co["joint_min_mi"] < co["identity_pair"]["mi"]
                 - 0.05,
                 "CLASS ineligible for recipient 1": not r1["class"]["eligible"]}
    elif name == "COMPLEMENTARY":
        preds = {"each identity single release is informative (MI > 0.05)": r1["identity"]["mi"] > 0.05
                 and r2["identity"]["mi"] > 0.05,
                 "each local single release is uninformative (MI <= 1e-12)": r1["local_min_mi"]["mi"] <= MI_TIE
                 and r2["local_min_mi"]["mi"] <= MI_TIE,
                 "local pair reveals SEX (coalition MI > 0.3)": co["local_pair_mi_range"][0] > 0.3,
                 "coalition gain over the better single local release > 0.3": co["local_pair_gain_range"][0] > 0.3,
                 "the clue is forced: joint min == local pair MI": abs(co["joint_min_mi"] - co["local_pair_mi_range"][0])
                 <= MI_TIE}
    elif name == "NO_COALITION":
        preds = {"pair MI == better single MI for every admissible pair": max(
                     abs(x) for x in co["gain_range_over_all_admissible_pairs"]) <= MI_TIE,
                 "recipient 1 has headroom (local < identity - 0.05)": r1["local_min_mi"]["mi"]
                 < r1["identity"]["mi"] - 0.05,
                 "recipient 2 alone is uninformative (max MI <= 1e-12)": r2["max_mi_over_admissible"] <= MI_TIE,
                 "joint min == local pair MI": abs(co["joint_min_mi"] - co["local_pair_mi_range"][0]) <= MI_TIE}
    elif name == "JOINT_HEADROOM":
        w = 0.05
        preds = {"joint min < sequential 1->2 - 0.05 (every tie choice)": co["joint_min_mi"]
                 < co["sequential_1_to_2_mi_range"][0] - w,
                 "joint min < sequential 2->1 - 0.05 (every tie choice)": co["joint_min_mi"]
                 < co["sequential_2_to_1_mi_range"][0] - w,
                 "joint min < local pair - 0.05 (every tie choice)": co["joint_min_mi"]
                 < co["local_pair_mi_range"][0] - w,
                 "each recipient's own-MI argmin is unique (no tie-break involved)":
                 r1["local_min_mi"]["argmin_partitions"] == 1 and r2["local_min_mi"]["argmin_partitions"] == 1,
                 "joint argmin pair is unique": co["joint_argmin_pairs"] == 1,
                 "joint Bayes accuracy < local pair Bayes accuracy": Fraction(co["joint_bayes_range"][1])
                 < Fraction(co["local_pair_bayes_range"][0])}
    else:
        raise ValueError(name)
    return {"predicates": preds, "holds": all(preds.values())}


def construction_check(laws_doc):
    return {name: toy_law_check(laws_doc["laws"][name]) for name in LAW_NAMES}


def check_toy_laws(path=None):
    """Recompute the construction check of TOY_LAWS.json and compare it with the stored block. Returns a summary."""
    path = Path(path or TOY_LAWS)
    doc = json.loads(path.read_text())
    fresh = construction_check(doc)
    same = json.loads(json.dumps(fresh, sort_keys=True)) == doc.get("construction_check")
    return {"file_sha256": sha_file(path), "construction_check_reproduced": bool(same),
            "intended_properties_hold": {n: fresh[n]["intended_property"]["holds"] for n in LAW_NAMES}}


# ----------------------------------------------------------------------------------------------- manifest
def scrub(text):
    if re.search(r"/Users/|/Volumes/|/private/|/home/|" + re.escape(Path.home().name), text):
        raise SystemExit("REFUSED: identifying path or user name in a public file")
    return text


def guard_probe():
    """What labels() answers for every (procedure, role) without touching data (allowlist + current lock state)."""
    ok, why = pilot_lock_pushed()
    out = {}
    for proc in list(ALLOW) + ["calibration", "assessment", "fitting"]:
        for role in ROLES + (AUDIT, ASSESS):
            try:
                kinds = allowed_kinds(proc, role)
                out[f"{proc}:{role}"] = (f"allowed after the pushed PILOT_LOCK ({', '.join(kinds)})" if not ok
                                         else f"allowed ({', '.join(kinds)})")
                if not ok:
                    out[f"{proc}:{role}"] += f"; now refused: {why}"
            except AccessRefused as e:
                out[f"{proc}:{role}"] = "refused (" + str(e).replace("REFUSED: ", "") + ")"
    return out


def build_manifest(W=None, seeds=I.SEEDS):
    W = W if W is not None else load()
    refs = {}
    for k in seeds:
        _arr, rec = reference_all(k, W)
        refs[str(k)] = {"cache": f"<PRIVATE_CACHE>/ccm_v1/{REF_DIR_NAME}/ref__s{k}.npz", "npz_sha256": rec["npz_sha256"],
                        "array_sha256": rec["array_sha256"], "rows": rec["rows"], "pins": rec["pins"],
                        "alpha_H_GLOBAL_TEMP": rec["alpha"], "calU_content_sha256": rec["calU_content_sha256"],
                        "parity_vs_admitted_teacher_npz": rec["parity"],
                        "argmax_preserved_rows": rec["argmax_preserved_rows"],
                        "strict_argmax_rows_Ucal": rec["strict_argmax_rows"], "computed_at": rec["computed_at"]}
    ok, why = pilot_lock_pushed()
    names = [str(x) for x in W["feature_names"]]
    use = {FIT: "reference vectors (U0, Ucal) for Stage D geometry F1-F3u and engineering; SEX only via "
                "labels('defense_fit') in the pilot (privacy-aware assignment fitting)",
           HELD: "Ucal reference vectors for held-out geometry F4 (fallback rate); label-free; no temperature refit",
           ATT: "new attacker fitting (pilot only): reference vectors and SEX via labels('attack')",
           INNER: "bounded exploratory pilot only (after the pushed PILOT_LOCK): reference vectors, SEX via "
                  "labels('attack') and task labels via labels('utility'); scored once",
           HEAD: "rows kept for features only; labels never readable; no reference vectors served",
           TRAIN_MATCHED: "label-free position subset of OSF_DEFENSE_FIT (hcal diagnostic subrole); no use planned"}
    roles = {r: {**W["receipt"][r], "ccm_use": use[r],
                 "labels": {p: list(ALLOW[p][r]) for p in ALLOW if r in ALLOW[p]} or "none readable"}
             for r in ROLES + tuple(SUBROLES)}
    toy = {"file": "TOY_LAWS.json", "sha256": sha_file(TOY_LAWS) if TOY_LAWS.exists() else None,
           "construction_notes": "TOY_LAWS_CONSTRUCTION.md",
           "construction_notes_sha256": sha_file(I.PKG / "TOY_LAWS_CONSTRUCTION.md")
           if (I.PKG / "TOY_LAWS_CONSTRUCTION.md").exists() else None}
    if TOY_LAWS.exists():
        toy.update(check_toy_laws())
        tl = json.loads(TOY_LAWS.read_text())
        toy["laws"] = list(tl["laws"])
        toy["history"] = [{"previous_sha256": a["previous_file_sha256"], "amendment": a["id"], "at": a["at"],
                           "requested_by": a["requested_by"], "change": a["change"], "reason": a["reason"],
                           "unchanged_laws_canonical_sha256": a["unchanged_laws_canonical_sha256"]}
                          for a in tl.get("amendments", [])]
    return {
        "schema": "ccm-data-access-manifest-v1", "study": I.STUDY, "owner": "role F",
        "source": {"hcal_tip": I.SOURCE_TIP, "hcal_evidence": I.SOURCE_EVIDENCE,
                   "loader": "hcal.data.load(verify=True, unseal=False) -> lra.data -> qpc.data -> dpc.data -> "
                             "osf.data (pinned blobs, input sha256, role manifests verified by the pinned chain); "
                             "restricted by ccm.data.restrict before anything is returned",
                   "input": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz",
                   "input_sha256": "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12",
                   "permitted_columns": {"n": len(names), "feature_names_sha256":
                                         hashlib.sha256("\n".join(names).encode()).hexdigest(),
                                         "convention": "sha256 of '\\n'.join(names)"}},
        "rows": {"source_rows": N_SOURCE, "kept_rows": int(W["row_id"].size), "token_row_id_sha256": W["token"],
                 "dropped": W["guard"]["dropped"],
                 "dropped_note": "all OSF_DEVELOPMENT_ASSESSMENT rows; the one AUDIT_FIT row that is a "
                                 "non-representative member of a CALIBRATION_HELDOUT group (in no ccm role)"},
        "roles": roles,
        "role_hash_convention": "hcal.data.sha_arr (sha256 of dtype + shape + bytes) of the sorted int64 row ids / "
                                "unique int64 group ids; equal_to_hcal_role_manifest compares with "
                                f"{HCAL_REL}/ROLE_MANIFEST.json at {I.SOURCE_TIP}",
        "guard": {**W["guard"], "rule": "assessment rows dropped before any array is returned; no assessment row id "
                                       "survives; no array of the assessment (13,936) or full-source (39,170) "
                                       "length; no label key in the returned dict; *.assess never imported"},
        "allowlist": {"procedures": {p: {r: list(v) for r, v in ALLOW[p].items()} for p in ALLOW},
                      "hcal_procedure_behind_each_grant": {f"{p}:{r}": h for (p, r), h in HCAL_PROCEDURE.items()},
                      "pilot_only": list(PILOT_ONLY),
                      "pilot_lock_rule": f"{I.REL}/PILOT_LOCK.json present, committed at HEAD and byte-identical on "
                                         f"origin/{I.BRANCH} (fetched at every labels() call)",
                      "pilot_lock_state_at_writing": why, "always_refused_roles": list(ALWAYS_REFUSED_ROLES),
                      "refused_everything_else": True, "probe": guard_probe()},
        "forbidden": list(FORBIDDEN),
        "reference_vectors": {
            "roles_served": list(REFERENCE_ROLES),
            "U0": "dpc.deploy.load_teacher(<HCAL_PRIV>/admitted/rel__s{k}__U, seed=k) + teacher_probs on the permitted "
                  "reference rows only (one batch of the four roles)",
            "Ucal": "hcal.calib.apply_u(U0, frozen calU__s{k} H-GLOBAL-TEMP record, d); never refitted",
            "decision": "numpy argmax of U0 (first index); argmax(Ucal) == d asserted on every row",
            "parity": "U0 and d bitwise equal to the admitted hcal teacher.npz on every reference row (teacher.npz is "
                      "indexed to the permitted rows immediately at load; other rows are never kept)",
            "seeds": refs},
        "stages": {
            "A triage": "committed aggregates only; no data access",
            "D geometry (F1-F3u, F5)": "reference(k, OSF_DEFENSE_FIT): Ucal (and U0 for the secondary note); no labels",
            "D geometry (F4)": "reference(k, CALIBRATION_HELDOUT): Ucal; no labels",
            "D finite-law oracles": "TOY_LAWS.json only (sha256-pinned); no real data",
            "E verification (role E)": "the same reference() outputs and TOY_LAWS.json; no labels",
            "F pilot (only after the pushed PILOT_LOCK)": "reference(k, role) for FIT / ATT / INNER (and HELD for "
                                                          "fallback); labels('defense_fit', FIT), labels('attack', "
                                                          "ATT | INNER), labels('utility', INNER)",
            "never": "OSF_DEVELOPMENT_ASSESSMENT rows, labels or predictions; HEAD_VALIDATION labels"},
        "toy_laws": toy,
        "assess_modules_loaded": assess_modules_loaded(),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def write_manifest(path=None):
    path = Path(path or (I.PKG / "DATA_ACCESS_MANIFEST.json"))
    man = build_manifest()
    _atomic_write(path, scrub(json.dumps(man, indent=1, sort_keys=True, default=str) + "\n").encode())
    return man


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["manifest"]:
        m = write_manifest()
        print(json.dumps({"kept_rows": m["rows"]["kept_rows"], "roles": {r: [v["rows"], v["groups"],
                                                                            v["equal_to_hcal_role_manifest"]]
                                                                        for r, v in m["roles"].items()},
                          "parity": {k: v["parity_vs_admitted_teacher_npz"]["ok"]
                                     for k, v in m["reference_vectors"]["seeds"].items()},
                          "toy_laws": m["toy_laws"]}, indent=1))
    elif argv == ["toy-laws-check"]:
        print(json.dumps(check_toy_laws(), indent=1))
    else:
        raise SystemExit("usage: python -m ccm.data manifest | toy-laws-check")


if __name__ == "__main__":
    main()
