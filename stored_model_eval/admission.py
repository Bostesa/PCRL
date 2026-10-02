"""Admission of stored arrays: hashes, explicit row-ID alignment, class support, units, role disjointness.

Manifest (JSON, schema stored_model_eval.manifest/v1):
{
  "schema": "stored_model_eval.manifest/v1",
  "synthetic": false,
  "files":  {"<name>": {"path": "<abs or relative to manifest>", "sha256": "<hex>"}},
  "arrays": {"<role-of-array>": {"file": "<name>", "key": "<npz key>", "ids": "<npz key of row ids>"}},
      required array names: "labels" (sensitive attribute), "units", "roles";
      optional: "representations", "outputs", "task_labels", "record_keys", "features"
  "reference_checks": [{"array": "labels", "file": "<name>", "key": "...", "ids": "..."}],
  "pending": [{"name": "<file name>", "archive_member": "<drive archive member>", "sha256": "..."}],
  "allowed_roles": ["defense_fit", "attacker_fit", "attacker_val", "evaluation"],
  "min_class_support": 100
}
Rules: every array carries an explicit ID array; no array is aligned implicitly by position. Arrays whose
IDs are the same set in a different order are REJECTED (no silent re-indexing). Duplicated IDs are
REJECTED. A reference check joins by ID and rejects value disagreements (catches values shuffled under
intact IDs). A unit or record key present in two roles is REJECTED.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import SCHEMA_MANIFEST

REQUIRED = ("labels", "units", "roles")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _resolve(base: Path, p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else (base / q)


def _load_member(path: Path, key: str | None):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if key not in z.files:
                raise KeyError(f"{path.name} has no key {key!r} (keys: {sorted(z.files)})")
            return z[key]
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    raise ValueError(f"unsupported array file type {path.suffix}")


def admit(manifest: dict | str | Path, base_dir: Path | None = None) -> dict:
    if not isinstance(manifest, dict):
        mpath = Path(manifest)
        base_dir = mpath.parent
        manifest = json.loads(mpath.read_text())
    base_dir = Path(base_dir or ".")
    errors, warnings, checks = [], [], []
    res = {"schema": SCHEMA_MANIFEST, "status": None, "errors": errors, "warnings": warnings,
           "checks": checks, "files": {}, "pending": []}
    if manifest.get("schema") != SCHEMA_MANIFEST:
        errors.append(f"manifest schema must be {SCHEMA_MANIFEST}")

    # 1. files and hashes ------------------------------------------------------------------------
    pending_names = {p["name"]: p for p in manifest.get("pending", [])}
    paths = {}
    for name, f in manifest.get("files", {}).items():
        p = _resolve(base_dir, f["path"])
        if not p.exists():
            if name in pending_names:
                res["pending"].append({"name": name, **pending_names[name], "status": "PENDING"})
                continue
            errors.append(f"file {name} missing: {p}")
            continue
        got = sha256_file(p)
        ok = (got == f.get("sha256"))
        res["files"][name] = {"path": str(p), "sha256": got, "expected": f.get("sha256"), "match": ok}
        checks.append(("sha256", name, ok))
        if not ok:
            errors.append(f"sha256 mismatch for {name}: expected {f.get('sha256')}, got {got}")
        paths[name] = p
    if res["pending"] and not errors:
        res["status"] = "PENDING"
        return res

    # 2. arrays with explicit ids -------------------------------------------------------------------
    arrays, ids = {}, {}
    spec = manifest.get("arrays", {})
    required = tuple(manifest.get("required_arrays", REQUIRED))
    for req in required:
        if req not in spec:
            errors.append(f"required array {req!r} not declared")
    for aname, a in spec.items():
        if a.get("file") not in paths:
            errors.append(f"array {aname}: file {a.get('file')} unavailable")
            continue
        if not a.get("ids"):
            errors.append(f"array {aname}: no explicit ID array declared (positional alignment refused)")
            continue
        try:
            v = _load_member(paths[a["file"]], a.get("key"))
            i = _load_member(paths[a["file"]] if not a.get("ids_file") else paths[a["ids_file"]], a["ids"])
        except (KeyError, ValueError) as e:
            errors.append(f"array {aname}: {e}")
            continue
        if len(i) != len(v):
            errors.append(f"array {aname}: {len(v)} rows but {len(i)} ids")
            continue
        u, c = np.unique(i, return_counts=True)
        if (c > 1).any():
            errors.append(f"array {aname}: {int((c > 1).sum())} duplicated ids "
                          f"(e.g. {u[c > 1][:3].tolist()}); duplicates must be declared as record keys, "
                          "not repeated row ids")
            continue
        arrays[aname], ids[aname] = v, i

    # 3. alignment ---------------------------------------------------------------------------------
    anchor = next((a for a in ("representations", "features", "labels") if a in ids), None)
    if anchor is not None:
        a_ids = ids[anchor]
        for aname, i in ids.items():
            if aname == anchor:
                continue
            if len(i) == len(a_ids) and np.array_equal(i, a_ids):
                checks.append(("aligned", aname, True))
                continue
            s_a, s_i = set(a_ids.tolist()), set(i.tolist())
            if s_a == s_i:
                errors.append(f"array {aname}: same IDs as {anchor} in a different order "
                              "(equal-length reordered arrays are rejected; re-export in anchor order)")
            else:
                errors.append(f"array {aname}: ID sets differ from {anchor} "
                              f"(missing {len(s_a - s_i)}, extra {len(s_i - s_a)})")
            checks.append(("aligned", aname, False))

    # 4. reference value checks (join by id) --------------------------------------------------------
    for rc in manifest.get("reference_checks", []):
        an = rc["array"]
        if an not in arrays or rc.get("file") not in paths:
            errors.append(f"reference check for {an}: inputs unavailable")
            continue
        rv = _load_member(paths[rc["file"]], rc["key"])
        ri = _load_member(paths[rc["file"]], rc["ids"])
        pos = {int(k) if np.issubdtype(np.asarray(ri).dtype, np.integer) else k: j
               for j, k in enumerate(ri.tolist())}
        missing = [k for k in ids[an].tolist() if k not in pos]
        if missing:
            errors.append(f"reference check {an}: {len(missing)} ids absent from reference")
            continue
        ref_vals = rv[[pos[k] for k in ids[an].tolist()]]
        n_bad = int((np.asarray(ref_vals) != np.asarray(arrays[an])).reshape(len(ref_vals), -1).any(1).sum())
        checks.append(("reference_values", an, n_bad == 0))
        if n_bad:
            errors.append(f"reference check {an}: {n_bad} rows disagree with the reference joined by id "
                          "(values reordered or corrupted under intact ids)")

    # 5. roles, units, support ----------------------------------------------------------------------
    allowed = manifest.get("allowed_roles", ["defense_fit", "attacker_fit", "attacker_val", "evaluation"])
    min_sup = int(manifest.get("min_class_support", 100))
    summary = {}
    if not errors and all(k in arrays for k in REQUIRED):  # role/unit checks need labels+units+roles
        roles = arrays["roles"].astype(str)
        units = arrays["units"]
        y = arrays["labels"].astype(np.int64)
        bad_roles = sorted(set(roles.tolist()) - set(allowed))
        if bad_roles:
            errors.append(f"undeclared roles {bad_roles}")
        unit_roles: dict = {}
        for u, r in zip(units.tolist(), roles.tolist()):
            unit_roles.setdefault(u, set()).add(r)
        crossing = [u for u, rs in unit_roles.items() if len(rs) > 1]
        if crossing:
            errors.append(f"{len(crossing)} grouping units appear in more than one role (role disjointness "
                          f"by unit violated; e.g. {crossing[:3]})")
        if "record_keys" in arrays:
            rk_roles: dict = {}
            for k, r in zip(arrays["record_keys"].tolist(), roles.tolist()):
                rk_roles.setdefault(k, set()).add(r)
            rk_cross = [k for k, rs in rk_roles.items() if len(rs) > 1]
            if rk_cross:
                errors.append(f"{len(rk_cross)} duplicated records span more than one role")
            n_dup = len(roles) - len(rk_roles)
            if n_dup:
                warnings.append(f"{n_dup} rows are duplicate records; they collapse to one unit in inference")
        K = int(y.max()) + 1
        for r in allowed:
            m = roles == r
            cnt = np.bincount(y[m], minlength=K).tolist()
            uns = [k for k, c in enumerate(cnt) if c < min_sup]
            summary[r] = {"n_rows": int(m.sum()), "n_units": int(len(set(units[m].tolist()))),
                          "class_counts": cnt, "unsupported_classes": uns}
            if m.sum() and uns:
                warnings.append(f"role {r}: classes {uns} below min support {min_sup} -> NOT_ESTIMABLE "
                                "for per-class / pair quantities involving them")
    res["role_summary"] = summary
    res["n_rows"] = int(len(ids[anchor])) if anchor else 0
    res["status"] = "REJECTED" if errors else "ADMITTED"
    res["checks"] = [list(c) for c in checks]
    return res


def load_admitted(manifest_path: str | Path) -> tuple[dict, dict]:
    """Admit, then return (arrays aligned in anchor order, admission record). Raises if not ADMITTED."""
    mpath = Path(manifest_path)
    manifest = json.loads(mpath.read_text())
    rec = admit(manifest, mpath.parent)
    if rec["status"] != "ADMITTED":
        raise ValueError(f"manifest not admitted ({rec['status']}): {rec['errors']}")
    out = {}
    for aname, a in manifest["arrays"].items():
        out[aname] = _load_member(Path(rec["files"][a["file"]]["path"]), a.get("key"))
        if "row_ids" not in out:
            out["row_ids"] = _load_member(Path(rec["files"][a["file"]]["path"]), a["ids"])
    out["_synthetic"] = bool(manifest.get("synthetic", False))
    return out, rec
