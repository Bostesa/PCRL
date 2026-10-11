"""Role loader for the decision-preserving compression study (dpc): the pinned OSF roles, reused EXACTLY.

Owner: data/custody role. No new role rule exists in this study. ``load`` calls the pinned ``osf.data.load`` (source
commit 925e0fddfcb666116c6179575339728a324ed78e, osf/data.py sha256 389c0633...3600) and returns its D unchanged
(same rows, order, roles, subroles, pools, numeric refit and 83 permitted columns), after checking every role's
row count, exact-record group count and sorted-row-id sha256 against the pinned osf ROLE_MANIFEST.json:

    OSF_DEFENSE_FIT              15,434 rows  score partitions, prototypes, compression policies, privacy objectives
    HEAD_VALIDATION               1,500 rows  historical deployed heads were selected here (no new heads in dpc)
    AUDIT_FIT                     6,065 rows  independent attackers only
    INNER_SELECTION               2,235 rows  configurations and attackers only
    OSF_DEVELOPMENT_ASSESSMENT   13,936 rows / 13,929 exact-record groups (four historically used pools)

Assessment labels (sex, race, y_income, y_occupation_group and the y dict) are masked to -1 at load. Only dpc.assess
may unseal, and only after the pushed EVALUATION_LOCK: ``load(unseal=True)`` refuses unless (a) the calling module is
dpc.assess and (b) results/pcrl_decision_preserving_compression_v1/EVALUATION_LOCK.json is committed and byte-identical
on origin/research/pcrl-decision-preserving-compression-v1 (git fetch + show). There is no bypass.

``labels_for(D, procedure, role)`` is the label allowlist for this study's procedures (fitting reads OSF_DEFENSE_FIT
only; attackers AUDIT_FIT; selection INNER_SELECTION; the assessment role only under "assessment").

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.data manifest   -> ROLE_MANIFEST.json (loader part)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from osf import data as OD

WT = Path(__file__).resolve().parents[1]
REL = "results/pcrl_decision_preserving_compression_v1"
PKG = WT / REL
OSF_REL = "results/pcrl_online_strength_frontier_v1"
OSF_PKG = WT / OSF_REL
BRANCH = "research/pcrl-decision-preserving-compression-v1"
SOURCE_BRANCH = "research/pcrl-online-strength-frontier-v1"
SOURCE_PIN = "925e0fddfcb666116c6179575339728a324ed78e"
INPUT_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
OSF_DATA_SHA = "389c0633b2802fe8ada7bc77b3f29fabcbcac52c5093b1c1f16d9e53c9f33600"
FEATURE_NAMES_SHA = "7101f2fb37b623a08221c32b33eedf132cf0c4c376bb4042ab9cf16da08ad975"   # sha256("\n".join(names))
SRC = OD.SRC
ROLES = OD.ROLES
SUBROLES = OD.SUBROLES
FIT_ROLES = OD.FIT_ROLES
POOLS = OD.POOLS
EXCLUSIONS = OD.EXCLUSIONS
LABEL_KEYS = OD.LABEL_KEYS
N_PERMITTED = OD.N_PERMITTED
ASSESS = "OSF_DEVELOPMENT_ASSESSMENT"
EXPECTED = {"OSF_DEFENSE_FIT": (15434, 15428), "HEAD_VALIDATION": (1500, 1499), "AUDIT_FIT": (6065, 6061),
            "INNER_SELECTION": (2235, 2234), ASSESS: (13936, 13929)}
N_ROWS = 39170
EVALUATION_LOCK = PKG / "EVALUATION_LOCK.json"
UNSEAL_CALLER = "dpc.assess"
# label allowlist (which procedure may read labels of which role). No head refitting or selection exists in dpc;
# "head_validation_calibration" is the optional secondary temperature control (HEAD_VALIDATION only, prompt sec. 12).
ALLOW = {"fitting": ("OSF_DEFENSE_FIT",),
         "head_validation_calibration": ("HEAD_VALIDATION",),
         "inner_audit": ("AUDIT_FIT", "INNER_SELECTION"),
         "selection": ("INNER_SELECTION",),
         "assessment": ("OSF_DEFENSE_FIT", "AUDIT_FIT", "INNER_SELECTION", ASSESS)}
assert OD.SRC_SHA == INPUT_SHA, "osf.data pins a different input"


# ------------------------------------------------------------------ helpers
def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rowid_hash(r) -> str:
    """sha256 of the sorted row ids as int64 bytes (jcv/rgj/smf/osf manifest convention)."""
    return hashlib.sha256(np.ascontiguousarray(np.sort(np.asarray(r)).astype(np.int64)).tobytes()).hexdigest()


def group_set_hash(u) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.unique(np.asarray(u)).astype(np.int64)).tobytes()).hexdigest()


def git(*a, cwd=WT):
    r = subprocess.run(["git", "-C", str(cwd), *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def pinned_json(rel_path):
    """A source file as committed at the pinned evidence commit (not the working tree)."""
    b = git("show", f"{SOURCE_PIN}:{rel_path}")
    if b is None:
        raise SystemExit(f"REFUSED: {rel_path} absent at {SOURCE_PIN}")
    return json.loads(b)


def source_role_manifest():
    return pinned_json(f"{OSF_REL}/ROLE_MANIFEST.json")


# ------------------------------------------------------------------ unseal gate
def _caller_module(depth):
    f = sys._getframe(depth)
    name = f.f_globals.get("__name__")
    if name == "__main__":
        spec = f.f_globals.get("__spec__")
        name = getattr(spec, "name", None) or "__main__"
    return name


def evaluation_lock_pushed():
    """(ok, reason): EVALUATION_LOCK.json committed and byte-identical on origin/<BRANCH> (fetched now)."""
    rel = f"{REL}/EVALUATION_LOCK.json"
    if not EVALUATION_LOCK.exists():
        return False, "EVALUATION_LOCK.json does not exist"
    subprocess.run(["git", "-C", str(WT), "fetch", "-q", "origin", BRANCH], capture_output=True)
    remote = git("show", f"origin/{BRANCH}:{rel}")
    if remote is None:
        return False, "EVALUATION_LOCK.json is not on origin"
    if remote != EVALUATION_LOCK.read_bytes():
        return False, "local EVALUATION_LOCK.json differs from origin"
    return True, "EVALUATION_LOCK.json byte-identical on origin"


def unseal_gate(caller):
    if caller != UNSEAL_CALLER:
        raise PermissionError(f"REFUSED: only {UNSEAL_CALLER} may unseal assessment labels (caller: {caller})")
    ok, why = evaluation_lock_pushed()
    if not ok:
        raise PermissionError(f"REFUSED: assessment labels stay sealed: {why}")
    return why


# ------------------------------------------------------------------ loader
def check_against_source(D, src=None):
    """Counts, groups and sorted row-id hashes of every role/subrole/pool equal the pinned osf ROLE_MANIFEST."""
    src = src or source_role_manifest()
    man = OD.manifest(D)
    out, bad = {}, []
    for r in ROLES:
        ref = src["roles"][r]
        got = man[r]
        eq = {"rows": got["rows"] == ref["rows"], "groups": got["groups"] == ref["groups"],
              "row_id_sha256": got["row_id_sha256"] == ref["row_id_sha256"],
              "group_id_set_sha256": group_set_hash(D["unit"][D["idx"][r]]) == ref["group_id_set_sha256"],
              "expected_counts": (got["rows"], got["groups"]) == EXPECTED[r]}
        out[r] = eq
        bad += [f"{r}.{k}" for k, v in eq.items() if not v]
    for r in SUBROLES:
        ref = src["defense_fit_subroles"][r]
        got = man[r]
        eq = all(got[k] == ref[k] for k in ("rows", "groups", "row_id_sha256"))
        out[r] = eq
        bad += [] if eq else [r]
    a = D["idx"][ASSESS]
    for p in POOLS:
        ref = src["roles"][ASSESS]["by_pool"][p]
        ix = a[D["pool"][a] == p]
        eq = (len(ix) == ref["rows"] and len(np.unique(D["unit"][ix])) == ref["groups"]
              and rowid_hash(D["row_id"][ix]) == ref["row_id_sha256"])
        out[f"pool:{p}"] = eq
        bad += [] if eq else [f"pool:{p}"]
    names = "\n".join(str(x) for x in D["feature_names"]).encode()
    out["feature_names_sha256"] = hashlib.sha256(names).hexdigest() == src["permitted_columns"]["feature_names_sha256"]
    out["n_rows"] = len(D["row_id"]) == N_ROWS
    bad += [k for k in ("feature_names_sha256", "n_rows") if not out[k]]
    return out, bad


def load(verify=True, unseal=False):
    """The pinned osf D (all 39,170 rows of the five OSF roles, osf order). Assessment labels are -1 unless
    unseal=True is called from dpc.assess after the pushed EVALUATION_LOCK (unseal_gate)."""
    gate = unseal_gate(_caller_module(2)) if unseal else None
    if verify and sha_file(Path(OD.__file__)) != OSF_DATA_SHA:
        raise SystemExit("REFUSED: osf/data.py differs from the pinned role loader")
    D = OD.load(verify=verify, unseal=unseal)
    if verify:
        _, bad = check_against_source(D)
        if bad:
            raise SystemExit(f"REFUSED: roles differ from the pinned osf ROLE_MANIFEST: {bad}")
    D["study"] = "pcrl_decision_preserving_compression_v1"
    D["unseal_gate"] = gate
    if not unseal:
        a = D["idx"][ASSESS]
        assert all(np.all(D[k][a] == -1) for k in LABEL_KEYS), "assessment labels not sealed"
    return D


def labels_for(D, procedure, role):
    """Allowlist guard: row indices of `role` whose labels `procedure` may read (sealed labels never)."""
    role = OD.ALIASES.get(role, role)
    if procedure not in ALLOW or role not in ALLOW[procedure]:
        raise PermissionError(f"{procedure} may not read labels of {role}")
    ix = D["idx"][role]
    if D["sealed"] and np.any(D["role"][ix] == ASSESS):
        raise PermissionError("assessment labels are sealed until the pushed EVALUATION_LOCK")
    return ix


def row_index(D):
    """row_id -> position in D."""
    return {int(r): j for j, r in enumerate(D["row_id"])}


# ------------------------------------------------------------------ public manifest (loader part)
def role_manifest(D=None):
    D = D if D is not None else load()
    src = source_role_manifest()
    eq, bad = check_against_source(D, src)
    man = OD.manifest(D)
    roles = {}
    for r in ROLES:
        ix = D["idx"][r]
        roles[r] = {"rows": man[r]["rows"], "groups": man[r]["groups"], "row_id_sha256": man[r]["row_id_sha256"],
                    "group_id_set_sha256": group_set_hash(D["unit"][ix]),
                    "permitted_use": src["roles"][r]["permitted_use"] if r != ASSESS else
                    "withheld from every dpc fitting and selection; labels -1 until the pushed EVALUATION_LOCK; "
                    "scored once after it (historically exposed benchmark)",
                    "dpc_use": {"OSF_DEFENSE_FIT": "score partitions, prototypes, compression policies, privacy "
                                                   "objectives (fit here only)",
                                "HEAD_VALIDATION": "historical deployed-head C selection only; no head is refitted or "
                                                   "selected in dpc",
                                "AUDIT_FIT": "fresh attackers fitted here only",
                                "INNER_SELECTION": "configurations and attackers chosen here only",
                                ASSESS: "single locked exploratory assessment"}[r]}
    a = D["idx"][ASSESS]
    roles[ASSESS]["by_pool"] = {p: {"rows": int((D["pool"][a] == p).sum()),
                                    "groups": int(len(np.unique(D["unit"][a][D["pool"][a] == p]))),
                                    "row_id_sha256": rowid_hash(D["row_id"][a][D["pool"][a] == p])} for p in POOLS}
    ag = set(np.unique(D["unit"][a]).tolist())
    shared = {r: int(len(ag & set(np.unique(D["unit"][D["idx"][r]]).tolist()))) for r in FIT_ROLES}
    groups_role = {}
    for r, u in zip(D["role"], D["unit"]):
        groups_role.setdefault(int(u), set()).add(str(r))
    span = int(sum(1 for s in groups_role.values() if len(s) > 1))
    sealed = {k: bool(np.all(D[k][a] == -1)) for k in LABEL_KEYS}
    sealed["y_dict"] = bool(all(np.all(D["y"][t][a] == -1) for t in D["y"]))
    sealed["loader_flag_sealed"] = bool(D["sealed"])
    outside = np.setdiff1d(np.arange(len(D["row_id"])), a)
    names = [str(x) for x in D["feature_names"]]
    low = [n.lower().split("=")[0] for n in names]
    guard = {}
    for proc, role in (("fitting", ASSESS), ("inner_audit", ASSESS), ("selection", ASSESS), ("fitting", "AUDIT_FIT"),
                       ("selection", "OSF_DEFENSE_FIT"), ("assessment", ASSESS)):
        try:
            labels_for(D, proc, role)
            guard[f"{proc}:{role}"] = "allowed"
        except PermissionError as e:
            guard[f"{proc}:{role}"] = "refused (" + ("sealed" if "sealed" in str(e) else "allowlist") + ")"
    return {
        "loader": "dpc.data.load(verify=True, unseal=False) -> osf.data.load (pinned, unchanged; no new rule)",
        "osf_data_py_sha256": sha_file(Path(OD.__file__)),
        "osf_data_py_equals_pinned": sha_file(Path(OD.__file__)) == OSF_DATA_SHA,
        "osf_data_py_blob_equals_source_pin": git("show", f"{SOURCE_PIN}:osf/data.py") == Path(OD.__file__).read_bytes(),
        "dpc_data_py_sha256": sha_file(Path(__file__)),
        "input": {"file": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz", "sha256": INPUT_SHA,
                  "sha256_recomputed_matches": sha_file(SRC) == INPUT_SHA},
        "rows_loaded": int(len(D["row_id"])),
        "roles": roles,
        "defense_fit_subroles": {r: {k: man[r][k] for k in ("rows", "groups", "row_id_sha256")} for r in SUBROLES},
        "subrole_note": "OSF_DEFENSE_FIT subroles are inherited (critic/diagnostic splits of the source studies); dpc "
                        "fits on all OSF_DEFENSE_FIT rows and uses no subrole.",
        "equality_with_source_role_manifest": {"source": f"{OSF_REL}/ROLE_MANIFEST.json at {SOURCE_PIN}",
                                               "checks": eq, "all_equal": not bad, "mismatches": bad},
        "exclusions": {"dropped_rows": int(39205 - len(D["row_id"])),
                       "dropped_by_old_role": src["consolidated_assessment_variants"]["with_CERT"][
                           "dropped_rows_by_old_role"],
                       "rule": "rows of excluded_exposure / excluded_dup and every pool row whose exact-record group "
                               "also occurs in a fitting role or an exclusion are dropped (osf rule, unchanged)",
                       "pool_admission": D["pool_info"]},
        "assessment_isolation": {"assessment_groups_shared_with_role": shared,
                                 "all_zero": all(v == 0 for v in shared.values()),
                                 "groups_spanning_two_roles": span,
                                 "analysis_unit": "exact-record group; never split; feature collisions are different "
                                                  "people and are never pooled"},
        "permitted_columns": {"n_columns": len(names), "feature_names_sha256": hashlib.sha256(
            "\n".join(names).encode()).hexdigest(), "feature_names_sha256_convention": "sha256 of '\\n'.join(names)",
            "equals_source": hashlib.sha256("\n".join(names).encode()).hexdigest() == FEATURE_NAMES_SHA,
            "order": names, "X_dtype": str(D["X"].dtype), "X_shape": list(D["X"].shape),
            "forbidden_names_present": [n for n, l in zip(names, low) if l in ("sex", "race", "income", "occupation",
                                                                                "fnlwgt", "row_id", "unit")],
            "excluded_source_columns": src["permitted_columns"]["excluded_source_columns"],
            "proxy_note": "relationship categories (incl. Husband/Wife) are retained permitted proxies; preprocessing "
                          "and order are the pinned osf ones (numeric columns re-standardised on OSF_DEFENSE_FIT)."},
        "numeric_refit": {c: {"mean_osf_fit": v["mean_new_fit"], "sd_osf_fit": v["sd_new_fit"],
                              "equals_source": (v["mean_new_fit"] == src["numeric_refit"]["per_column"][c]["mean_osf_fit"]
                                                and v["sd_new_fit"] == src["numeric_refit"]["per_column"][c]["sd_osf_fit"])}
                          for c, v in D["numeric_refit"].items()},
        "label_sealing": {"assessment_labels_minus_one": sealed,
                          "no_minus_one_outside_assessment": {k: bool(np.all(D[k][outside] != -1)) for k in LABEL_KEYS},
                          "unseal_gate": f"load(unseal=True) only from {UNSEAL_CALLER}, and only when "
                                         f"{REL}/EVALUATION_LOCK.json is byte-identical on origin/{BRANCH}",
                          "allowlist": {k: list(v) for k, v in ALLOW.items()}, "guard_probe": guard},
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_role_manifest():
    """Writes the loader part of ROLE_MANIFEST.json; keeps the independent recomputation written by role_check.py."""
    out = PKG / "ROLE_MANIFEST.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    src = source_role_manifest()
    man = {"schema": "dpc-role-manifest-v1", "study": "pcrl_decision_preserving_compression_v1",
           "source_study": "pcrl_online_strength_frontier_v1 (osf)", "source_pin": SOURCE_PIN,
           "source_role_manifest_sha256_at_pin": hashlib.sha256(git("show", f"{SOURCE_PIN}:{OSF_REL}/ROLE_MANIFEST.json"))
           .hexdigest(),
           "rule": "The OSF roles are reused EXACTLY (osf.data at the pin, imported unchanged). " + src["rule"]["text"],
           "cert_eligibility_verdict_at_source": src["cert_eligibility"]["verdict"],
           "cert_eligible": OD.CERT_ELIGIBLE,
           "all_rows_historically_exposed": True,
           "exposure_note": "Every row is a previously used UCI Adult row; every assessment row has been used "
                            "historically (EXPOSURE_LEDGER.md). Assessment labels are masked during dpc fitting and "
                            "selection; this preserves the new procedure's separation, it does not make the rows unseen.",
           **role_manifest()}
    if "independent_recomputation" in prev:
        man["independent_recomputation"] = prev["independent_recomputation"]
    txt = json.dumps(man, indent=1, default=str) + "\n"
    if "/Users/" in txt or "/Volumes/" in txt:
        raise SystemExit("REFUSED: identifying path in a public file")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(txt)
    tmp.replace(out)
    return man


if __name__ == "__main__":
    if sys.argv[1:] == ["manifest"]:
        m = write_role_manifest()
        print(json.dumps({"rows": m["rows_loaded"], "all_equal": m["equality_with_source_role_manifest"]["all_equal"],
                          "mismatches": m["equality_with_source_role_manifest"]["mismatches"]}, indent=1))
    else:
        raise SystemExit("usage: python -m dpc.data manifest")
