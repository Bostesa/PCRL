"""Role loader for the confidence-capacity study (qpc): the pinned osf/dpc roles, reused EXACTLY.

Owner: data/custody role (E). No new role rule exists in this study. ``load`` returns the D of the pinned loader chain
unchanged: dpc.data (blob at the source evidence commit 0a7b05a52746544213742f50efd0a48167efffb1) -> osf.data (blob at
the teacher provenance commit 925e0fddfcb666116c6179575339728a324ed78e) -> <PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz
(sha256 e0d9e54a...2f12). Same rows, order, roles, subroles, pools, numeric refit and 83 permitted columns.

Verified at every ``load(verify=True)`` (the default):
  * dpc/data.py and osf/data.py in this tree are byte-identical to their blobs at the pins;
  * the input file hashes to the pinned sha256;
  * per role: rows, exact-record groups, sorted-row-id sha256 and group-id-set sha256 equal BOTH the osf
    ROLE_MANIFEST.json at 925e0fd and the dpc ROLE_MANIFEST.json at 0a7b05a (subroles and the four assessment pools
    against osf, as dpc.data does);
        OSF_DEFENSE_FIT              15,434 rows / 15,428 groups
        HEAD_VALIDATION               1,500 rows /  1,499 groups
        AUDIT_FIT                     6,065 rows /  6,061 groups
        INNER_SELECTION               2,235 rows /  2,234 groups
        OSF_DEVELOPMENT_ASSESSMENT   13,936 rows / 13,929 groups
  * group isolation: no OSF_DEFENSE_FIT exact-record group occurs in AUDIT_FIT (attackers), INNER_SELECTION
    (selection) or OSF_DEVELOPMENT_ASSESSMENT; the full role x role shared-group matrix is published and must be zero
    off the diagonal (every group lies in exactly one role).

Assessment labels (sex, race, y_income, y_occupation_group and the y dict) are -1 at load. ``load(unseal=True)``
refuses unless (a) the calling module is qpc.assess and (b) results/pcrl_confidence_capacity_v1/EVALUATION_LOCK.json
is committed and byte-identical on origin/research/pcrl-confidence-capacity-v1 (git fetch + git show, now). dpc.data's
own gate only opens for dpc.assess with the dpc lock, so the unsealed path here calls the pinned osf.data.load directly
after this gate and re-runs every role check. There is no bypass.

``labels_for(D, procedure, role)`` is this study's label allowlist (ALLOW below).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m qpc.data manifest   -> ROLE_MANIFEST.json
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from dpc import data as DD
from osf import data as OD

WT = Path(__file__).resolve().parents[1]
REL = "results/pcrl_confidence_capacity_v1"
PKG = WT / REL
BRANCH = "research/pcrl-confidence-capacity-v1"
SOURCE_SHA = "0a7b05a52746544213742f50efd0a48167efffb1"          # dpc evidence commit (code + roles reused)
SOURCE_BRANCH = "research/pcrl-decision-preserving-compression-v1"
DPC_REL = "results/pcrl_decision_preserving_compression_v1"
TEACHER_PIN = "925e0fddfcb666116c6179575339728a324ed78e"         # osf (teacher provenance)
OSF_REL = "results/pcrl_online_strength_frontier_v1"
INPUT_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
INPUT_PUBLIC = "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz"
SRC = DD.SRC
ROLES = OD.ROLES
SUBROLES = OD.SUBROLES
FIT_ROLES = OD.FIT_ROLES
POOLS = OD.POOLS
LABEL_KEYS = OD.LABEL_KEYS
N_PERMITTED = OD.N_PERMITTED
FIT = "OSF_DEFENSE_FIT"
ASSESS = "OSF_DEVELOPMENT_ASSESSMENT"
EXPECTED = {FIT: (15434, 15428), "HEAD_VALIDATION": (1500, 1499), "AUDIT_FIT": (6065, 6061),
            "INNER_SELECTION": (2235, 2234), ASSESS: (13936, 13929)}
N_ROWS = 39170
ISOLATED_FROM_FIT = ("AUDIT_FIT", "INNER_SELECTION", ASSESS)   # attacker, selection, assessment
EVALUATION_LOCK = PKG / "EVALUATION_LOCK.json"
UNSEAL_CALLER = "qpc.assess"
PINNED_BLOBS = {"dpc/data.py": SOURCE_SHA, "osf/data.py": TEACHER_PIN}
# Label allowlist: which procedure may read the labels (task labels, SEX) of which role. The dpc vocabulary
# (dpc.data.ALLOW: fitting, head_validation_calibration, inner_audit, selection, assessment) is accepted unchanged, so
# code written against dpc.data.labels_for works as is. qpc refits/recalibrates no head, so
# "head_validation_calibration" is a known procedure that may read NO role. qpc adds three narrower procedures.
ALLOW = {"fitting": (FIT,),                                  # partitions, prototypes, privacy objectives (SEX MI)
         "head_validation_calibration": (),                  # dpc name; no recalibration in qpc (prompt sec. 5)
         "inner_audit": ("AUDIT_FIT", "INNER_SELECTION"),    # attackers fitted on AUDIT_FIT, selected on INNER
         "selection": ("INNER_SELECTION",),                  # Stage A gate, rates, nominees, comparators
         "assessment": (FIT, "AUDIT_FIT", "INNER_SELECTION", ASSESS),
         "permutation_diagnostic": (FIT,),                   # DEFENSE_FIT SEX label-permutation MI-bias diagnostic
         "stage_a_utility": ("INNER_SELECTION",),            # A3 capacity gate (true-label utility)
         "controls": (FIT, "AUDIT_FIT", "INNER_SELECTION")}  # real-data null/positive controls (authorised roles)
PROCEDURE_ALIASES = {"attack": "inner_audit", "audit": "inner_audit"}
assert all(set(DD.ALLOW[p]) >= set(ALLOW[p]) for p in DD.ALLOW), "qpc must not widen a dpc procedure"
assert DD.INPUT_SHA == INPUT_SHA and OD.SRC_SHA == INPUT_SHA, "the pinned loaders pin a different input"


# ------------------------------------------------------------------ helpers
def sha_file(p) -> str:
    return DD.sha_file(p)


def rowid_hash(r) -> str:
    return DD.rowid_hash(r)


def group_set_hash(u) -> str:
    return DD.group_set_hash(u)


def git(*a, cwd=WT):
    r = subprocess.run(["git", "-C", str(cwd), *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def pinned_bytes(pin, rel_path):
    b = git("show", f"{pin}:{rel_path}")
    if b is None:
        raise SystemExit(f"REFUSED: {rel_path} absent at {pin}")
    return b


def pinned_json(pin, rel_path):
    return json.loads(pinned_bytes(pin, rel_path))


def loader_blob_checks():
    """{file: True iff the working-tree file is byte-identical to its blob at its pin}."""
    return {f: (WT / f).read_bytes() == pinned_bytes(pin, f) for f, pin in PINNED_BLOBS.items()}


# ------------------------------------------------------------------ unseal gate
def _caller_module(depth):
    f = sys._getframe(depth)
    name = f.f_globals.get("__name__")
    if name == "__main__":
        spec = f.f_globals.get("__spec__")
        name = getattr(spec, "name", None) or "__main__"
    return name


def evaluation_lock_pushed(lock=None, branch=BRANCH):
    """(ok, reason): EVALUATION_LOCK.json committed and byte-identical on origin/<branch> (fetched now)."""
    lock = Path(lock or EVALUATION_LOCK)
    rel = f"{REL}/EVALUATION_LOCK.json"
    if not lock.exists():
        return False, "EVALUATION_LOCK.json does not exist"
    subprocess.run(["git", "-C", str(WT), "fetch", "-q", "origin", branch], capture_output=True)
    remote = git("show", f"origin/{branch}:{rel}")
    if remote is None:
        return False, "EVALUATION_LOCK.json is not on origin"
    if remote != lock.read_bytes():
        return False, "local EVALUATION_LOCK.json differs from origin"
    return True, "EVALUATION_LOCK.json byte-identical on origin"


def unseal_gate(caller):
    if caller != UNSEAL_CALLER:
        raise PermissionError(f"REFUSED: only {UNSEAL_CALLER} may unseal assessment labels (caller: {caller})")
    ok, why = evaluation_lock_pushed()
    if not ok:
        raise PermissionError(f"REFUSED: assessment labels stay sealed: {why}")
    return why


# ------------------------------------------------------------------ checks
def dpc_role_manifest():
    return pinned_json(SOURCE_SHA, f"{DPC_REL}/ROLE_MANIFEST.json")


def check_against_dpc(D, src=None):
    """Counts, groups, sorted-row-id and group-set hashes of every role and pool equal dpc's ROLE_MANIFEST at 0a7b05a."""
    src = src or dpc_role_manifest()
    out, bad = {}, []
    for r in ROLES:
        ix = D["idx"][r]
        ref = src["roles"][r]
        eq = {"rows": int(len(ix)) == ref["rows"], "groups": int(len(np.unique(D["unit"][ix]))) == ref["groups"],
              "row_id_sha256": rowid_hash(D["row_id"][ix]) == ref["row_id_sha256"],
              "group_id_set_sha256": group_set_hash(D["unit"][ix]) == ref["group_id_set_sha256"]}
        out[r] = eq
        bad += [f"dpc:{r}.{k}" for k, v in eq.items() if not v]
    a = D["idx"][ASSESS]
    for p in POOLS:
        ref = src["roles"][ASSESS]["by_pool"][p]
        ix = a[D["pool"][a] == p]
        eq = (len(ix) == ref["rows"] and len(np.unique(D["unit"][ix])) == ref["groups"]
              and rowid_hash(D["row_id"][ix]) == ref["row_id_sha256"])
        out[f"pool:{p}"] = bool(eq)
        bad += [] if eq else [f"dpc:pool:{p}"]
    names = "\n".join(str(x) for x in D["feature_names"]).encode()
    out["feature_names_sha256"] = hashlib.sha256(names).hexdigest() == src["permitted_columns"]["feature_names_sha256"]
    bad += [] if out["feature_names_sha256"] else ["dpc:feature_names_sha256"]
    return out, bad


def group_overlap(D):
    """Role x role count of shared exact-record groups (diagonal = the role's own group count)."""
    sets = {r: set(np.unique(D["unit"][D["idx"][r]]).tolist()) for r in ROLES}
    return {a: {b: int(len(sets[a] & sets[b])) for b in ROLES} for a in ROLES}


def check_counts_and_isolation(D):
    out, bad = {}, []
    for r, (n, g) in EXPECTED.items():
        ix = D["idx"][r]
        got = (int(len(ix)), int(len(np.unique(D["unit"][ix]))))
        out[r] = {"rows": got[0], "groups": got[1], "expected": [n, g], "equal": got == (n, g)}
        bad += [] if got == (n, g) else [f"count:{r}"]
    ov = group_overlap(D)
    iso = {r: ov[FIT][r] for r in ISOLATED_FROM_FIT}
    off = {f"{a}|{b}": v for a in ROLES for b in ROLES if a < b and (v := ov[a][b])}
    bad += [f"defense_fit_group_overlap:{r}" for r, v in iso.items() if v]
    bad += [f"group_overlap:{k}" for k in off]
    out["n_rows"] = int(len(D["row_id"]))
    bad += [] if out["n_rows"] == N_ROWS else ["n_rows"]
    out["defense_fit_groups_shared_with"] = iso
    out["nonzero_offdiagonal_group_overlaps"] = off
    out["role_partition_complete"] = bool(np.all(np.isin(D["role"], ROLES)))
    bad += [] if out["role_partition_complete"] else ["role_partition"]
    return out, bad


def verify_D(D):
    """All role checks; returns (report, mismatches)."""
    blobs = loader_blob_checks()
    bad = [f"blob:{f}" for f, ok in blobs.items() if not ok]
    if sha_file(SRC) != INPUT_SHA:
        bad.append("input_sha256")
    eq_osf, b1 = DD.check_against_source(D)
    eq_dpc, b2 = check_against_dpc(D)
    cnt, b3 = check_counts_and_isolation(D)
    return {"loader_blobs_equal_pins": blobs, "input_sha256_matches": "input_sha256" not in bad,
            "vs_osf_role_manifest": eq_osf, "vs_dpc_role_manifest": eq_dpc, "counts_and_isolation": cnt}, \
        bad + list(b1) + b2 + b3


# ------------------------------------------------------------------ loader
def load(verify=True, unseal=False):
    """The pinned osf/dpc D (all 39,170 rows, osf order). Assessment labels are -1 unless unseal=True is called from
    qpc.assess after the pushed qpc EVALUATION_LOCK (unseal_gate)."""
    gate = unseal_gate(_caller_module(2)) if unseal else None
    if verify:
        bad = [f for f, ok in loader_blob_checks().items() if not ok]
        if bad:
            raise SystemExit(f"REFUSED: pinned loader differs from its blob at the pin: {bad}")
    if unseal:
        D = OD.load(verify=verify, unseal=True)      # dpc.data's gate is dpc-only; this gate has run above
    else:
        D = DD.load(verify=verify, unseal=False)
    if verify:
        _, bad = verify_D(D)
        if bad:
            raise SystemExit(f"REFUSED: roles differ from the pinned manifests: {bad}")
    D["study"] = "pcrl_confidence_capacity_v1"
    D["unseal_gate"] = gate
    a = D["idx"][ASSESS]
    if not unseal:
        assert D["sealed"] and all(np.all(D[k][a] == -1) for k in LABEL_KEYS), "assessment labels not sealed"
        assert all(np.all(D["y"][t][a] == -1) for t in D["y"]), "assessment task labels not sealed"
    return D


def labels_for(D, procedure, role):
    """Allowlist guard: row indices of `role` whose labels `procedure` may read (sealed labels never)."""
    role = OD.ALIASES.get(role, role)
    procedure = PROCEDURE_ALIASES.get(procedure, procedure)
    if procedure not in ALLOW or role not in ALLOW[procedure]:
        raise PermissionError(f"{procedure} may not read labels of {role}")
    ix = D["idx"][role]
    if D["sealed"] and np.any(D["role"][ix] == ASSESS):
        raise PermissionError("assessment labels are sealed until the pushed qpc EVALUATION_LOCK")
    return ix


def row_index(D):
    return {int(r): j for j, r in enumerate(D["row_id"])}


# ------------------------------------------------------------------ public manifest
def scrub(text):
    if re.search(r"/Users/|/Volumes/|/private/|" + re.escape(Path.home().name), text):
        raise SystemExit("REFUSED: identifying path or user name in a public file")
    return text


def role_manifest(D=None):
    D = D if D is not None else load()
    rep, bad = verify_D(D)
    man = OD.manifest(D)
    roles = {}
    use = {FIT: "Stage A k-means, Stage B fine partitions, prototypes, compression policies and privacy objectives "
                "(fit here only; teacher probabilities and, for privacy terms, SEX)",
           "HEAD_VALIDATION": "historical deployed-head C selection only; no head is refitted, recalibrated or "
                              "selected in qpc; no label read",
           "AUDIT_FIT": "attackers fitted here only",
           "INNER_SELECTION": "Stage A capacity gate (true-label utility), rate selection, attacker and nominee "
                              "selection here only",
           ASSESS: "withheld from every qpc fit and selection; labels -1 until the pushed qpc EVALUATION_LOCK; scored "
                   "once after it (historically exposed)"}
    for r in ROLES:
        ix = D["idx"][r]
        roles[r] = {"rows": man[r]["rows"], "groups": man[r]["groups"], "row_id_sha256": man[r]["row_id_sha256"],
                    "group_id_set_sha256": group_set_hash(D["unit"][ix]), "qpc_use": use[r]}
    a = D["idx"][ASSESS]
    roles[ASSESS]["by_pool"] = {p: {"rows": int((D["pool"][a] == p).sum()),
                                    "groups": int(len(np.unique(D["unit"][a][D["pool"][a] == p]))),
                                    "row_id_sha256": rowid_hash(D["row_id"][a][D["pool"][a] == p])} for p in POOLS}
    sealed = {k: bool(np.all(D[k][a] == -1)) for k in LABEL_KEYS}
    sealed["y_dict"] = bool(all(np.all(D["y"][t][a] == -1) for t in D["y"]))
    sealed["loader_flag_sealed"] = bool(D["sealed"])
    outside = np.setdiff1d(np.arange(len(D["row_id"])), a)
    names = [str(x) for x in D["feature_names"]]
    guard = {}
    for proc, role in (("fitting", ASSESS), ("inner_audit", ASSESS), ("selection", ASSESS),
                       ("stage_a_utility", ASSESS), ("controls", ASSESS), ("fitting", "AUDIT_FIT"),
                       ("fitting", "HEAD_VALIDATION"), ("head_validation_calibration", "HEAD_VALIDATION"),
                       ("selection", FIT), ("assessment", ASSESS)):
        try:
            labels_for(D, proc, role)
            guard[f"{proc}:{role}"] = "allowed"
        except PermissionError as e:
            guard[f"{proc}:{role}"] = "refused (" + ("sealed" if "sealed" in str(e) else "allowlist") + ")"
    ov = group_overlap(D)
    return {
        "loader": "qpc.data.load(verify=True, unseal=False) -> dpc.data.load (blob at 0a7b05a) -> osf.data.load "
                  "(blob at 925e0fd); D returned unchanged; no new role rule",
        "loader_blobs_equal_pins": rep["loader_blobs_equal_pins"],
        "code_sha256": {"qpc/data.py": sha_file(Path(__file__)), "dpc/data.py": sha_file(WT / "dpc/data.py"),
                        "osf/data.py": sha_file(WT / "osf/data.py")},
        "input": {"file": INPUT_PUBLIC, "sha256": INPUT_SHA, "sha256_recomputed_matches": rep["input_sha256_matches"],
                  "bytes": int(SRC.stat().st_size)},
        "rows_loaded": int(len(D["row_id"])),
        "roles": roles,
        "defense_fit_subroles": {r: {k: man[r][k] for k in ("rows", "groups", "row_id_sha256")} for r in SUBROLES},
        "subrole_note": "OSF_DEFENSE_FIT subroles are inherited from the source studies; qpc fits on all "
                        "OSF_DEFENSE_FIT rows and uses no subrole.",
        "equality_with_pinned_manifests": {
            "osf": {"source": f"{OSF_REL}/ROLE_MANIFEST.json at {TEACHER_PIN}", "checks": rep["vs_osf_role_manifest"]},
            "dpc": {"source": f"{DPC_REL}/ROLE_MANIFEST.json at {SOURCE_SHA}", "checks": rep["vs_dpc_role_manifest"]},
            "all_equal": not bad, "mismatches": bad},
        "counts": {r: rep["counts_and_isolation"][r] for r in EXPECTED},
        "group_isolation": {
            "rule": "no OSF_DEFENSE_FIT exact-record group occurs in an attacker (AUDIT_FIT), selection "
                    "(INNER_SELECTION) or assessment (OSF_DEVELOPMENT_ASSESSMENT) role; every group lies in one role",
            "defense_fit_groups_shared_with": rep["counts_and_isolation"]["defense_fit_groups_shared_with"],
            "shared_group_matrix": ov,
            "nonzero_offdiagonal": rep["counts_and_isolation"]["nonzero_offdiagonal_group_overlaps"],
            "pass": not [b for b in bad if "overlap" in b],
            "analysis_unit": "exact-record group; never split; feature collisions between different groups are "
                             "different people and are never pooled"},
        "exclusions": {"dropped_rows": int(39205 - len(D["row_id"])),
                       "rule": "osf rule, unchanged (excluded_exposure / excluded_dup and pool rows whose group also "
                               "occurs in a fitting role or an exclusion)", "pool_admission": D["pool_info"]},
        "permitted_columns": {"n_columns": len(names),
                              "feature_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                              "equals_pinned": hashlib.sha256("\n".join(names).encode()).hexdigest()
                              == DD.FEATURE_NAMES_SHA,
                              "convention": "sha256 of '\\n'.join(names)", "order": names,
                              "X_dtype": str(D["X"].dtype), "X_shape": list(D["X"].shape),
                              "proxy_note": "relationship categories (incl. Husband/Wife) are retained permitted "
                                            "proxies, as in the source; preprocessing and order are the pinned osf "
                                            "ones (numeric columns re-standardised on OSF_DEFENSE_FIT)."},
        "label_sealing": {"assessment_labels_minus_one": sealed,
                          "no_minus_one_outside_assessment": {k: bool(np.all(D[k][outside] != -1)) for k in LABEL_KEYS},
                          "unseal_gate": f"load(unseal=True) only from {UNSEAL_CALLER}, and only when "
                                         f"{REL}/EVALUATION_LOCK.json is byte-identical on origin/{BRANCH}",
                          "allowlist": {k: list(v) for k, v in ALLOW.items()},
                          "procedure_aliases": PROCEDURE_ALIASES,
                          "dpc_vocabulary_accepted": sorted(DD.ALLOW), "guard_probe": guard},
        "labels_read": "none (manifest uses role, group and row ids only; label arrays are checked for the -1 seal)",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_role_manifest(D=None):
    out = PKG / "ROLE_MANIFEST.json"
    man = {"schema": "qpc-role-manifest-v1", "study": "pcrl_confidence_capacity_v1",
           "source_study": "pcrl_decision_preserving_compression_v1 (dpc)", "source_evidence_commit": SOURCE_SHA,
           "teacher_provenance_commit": TEACHER_PIN,
           "source_role_manifests_sha256": {
               "dpc_at_source_commit": hashlib.sha256(pinned_bytes(SOURCE_SHA, f"{DPC_REL}/ROLE_MANIFEST.json"))
               .hexdigest(),
               "osf_at_teacher_pin": hashlib.sha256(pinned_bytes(TEACHER_PIN, f"{OSF_REL}/ROLE_MANIFEST.json"))
               .hexdigest()},
           "all_rows_historically_exposed": True,
           "exposure_note": "Every row is a previously used UCI Adult row and every assessment row has been used "
                            "historically, including by dpc (EXPOSURE_LEDGER.md). Masking keeps qpc's fitting and "
                            "selection separate from the assessment rows; it does not make them unseen.",
           **role_manifest(D)}
    txt = scrub(json.dumps(man, indent=1, default=str) + "\n")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(txt)
    tmp.replace(out)
    return man


if __name__ == "__main__":
    if sys.argv[1:] == ["manifest"]:
        m = write_role_manifest()
        print(json.dumps({"rows": m["rows_loaded"], "all_equal": m["equality_with_pinned_manifests"]["all_equal"],
                          "mismatches": m["equality_with_pinned_manifests"]["mismatches"],
                          "isolation_pass": m["group_isolation"]["pass"]}, indent=1))
    else:
        raise SystemExit("usage: python -m qpc.data manifest")
