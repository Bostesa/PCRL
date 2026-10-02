"""PILOT_LOCK_v2.json: build and verify the execution lock (the preparation PILOT_LOCK.json is never modified).

The lock pins:
  * the EFFECTIVE_PROTOCOL hash (and that it validates);
  * sha256 of every consumed code file: stored_model_eval/*.py (tests excluded) and
    results/combined_stored_model_pilot_v1/scripts/* (the runner and input builder); a file added to or removed
    from either set is a mismatch;
  * the applicable access table sha256;
  * the exact 26 registered unit IDs and each v2 manifest's sha256;
  * every data file those manifests reference (labels, task labels, forward cache, releases, checkpoint) plus
    features.npz and INPUTS_INDEX.json;
  * the primary family (16 IDs, statistics, bars), bootstrap B / seeds / alpha / quantile rule, support rule, bars.
`verify_lock` recomputes everything and returns every mismatch; `pilot --execute-scientific-fits` refuses on any.
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

from .admission import sha256_file
from .effective import (EFFECTIVE_PROTOCOL, PRIMARY_FAMILY, REGISTERED_UNITS, effective_hash, validate_effective)
from .pilot import select_units

LOCK_SCHEMA = "stored_model_eval.pilot_lock/v2"
CODE_DIRS = (("stored_model_eval", "*.py"), ("results/combined_stored_model_pilot_v1/scripts", "*"))
DEFAULT_ACCESS_TABLE = "results/combined_evaluation_preparation_v1/ATTACKER_ACCESS_TABLE.csv"
PREP_LOCK = "results/combined_evaluation_preparation_v1/PILOT_LOCK.json"


def code_files(root: Path) -> dict:
    root = Path(root)
    out = {}
    for d, pat in CODE_DIRS:
        for p in sorted((root / d).glob(pat)):
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith((".pyc", "~")):
                out[str(p.relative_to(root))] = sha256_file(p)
    return out


def _inference_pins(eff: dict) -> dict:
    I = eff["inference"]
    return {"exploratory": I["exploratory"], "primary": I["primary"], "quantile_method": I["quantile_method"],
            "bars": I["bars"], "tau": I["tau"], "tau_sensitivity": I["tau_sensitivity"], "chunk": I["chunk"],
            "sampling_unit": I["sampling_unit"]}


def _data_files(manifests: dict) -> dict:
    files = {}
    for u, mp in manifests.items():
        man = json.loads(Path(mp).read_text())
        for name, f in man["files"].items():
            p = Path(f["path"]).expanduser()
            p = p if p.is_absolute() else Path(mp).parent / p
            files[str(p.resolve())] = None
    return {p: sha256_file(Path(p)) for p in sorted(files)}


def _git_head(root: Path) -> str | None:
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def compute_state(root: Path, inputs_dir: Path, access_table: str, features: str | None) -> dict:
    root, inputs_dir = Path(root), Path(inputs_dir)
    errs = validate_effective(EFFECTIVE_PROTOCOL)
    sel = select_units(inputs_dir)
    at = root / access_table
    state = {"effective_protocol_sha256": effective_hash(), "effective_protocol_errors": errs,
             "code_files": code_files(root),
             "access_table": {"path": access_table, "sha256": sha256_file(at) if at.exists() else None},
             "units": {"ids": sel["selected"], "expected_ids": list(REGISTERED_UNITS), "match": sel["match"],
                       "manifests_sha256": {u: sha256_file(Path(p)) for u, p in sel["manifests"].items()}},
             "data_files_sha256": _data_files(sel["manifests"]),
             "inputs_index_sha256": (sha256_file(inputs_dir / "INPUTS_INDEX.json")
                                     if (inputs_dir / "INPUTS_INDEX.json").exists() else None),
             "primary_family": [dict(e) for e in PRIMARY_FAMILY],
             "inference": _inference_pins(EFFECTIVE_PROTOCOL),
             "support_rule": {k: EFFECTIVE_PROTOCOL["support"][k] for k in
                              ("min_attacker_fit", "min_attacker_val", "min_assessment", "min_supported_classes")}}
    if features:
        state["features"] = {"path": str(Path(features).resolve()), "sha256": sha256_file(Path(features))}
    pl = root / PREP_LOCK
    state["preparation_lock_sha256"] = sha256_file(pl) if pl.exists() else None
    return state


def build_lock(root: Path, inputs_dir: Path, out: Path, access_table: str = DEFAULT_ACCESS_TABLE,
               features: str | None = None, write_effective: Path | None = None) -> dict:
    st = compute_state(root, inputs_dir, access_table, features)
    if st["effective_protocol_errors"]:
        raise ValueError(f"effective protocol invalid: {st['effective_protocol_errors']}")
    if not st["units"]["match"]:
        raise ValueError("inputs dir does not hold exactly the registered units")
    if st["access_table"]["sha256"] is None:
        raise ValueError(f"access table {access_table} missing")
    lock = {"schema": LOCK_SCHEMA, "built_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "LOCKED before any real-data attacker/probe fit (development data; not a confirmation)",
            "worktree_head_at_build_informational": _git_head(root), "inputs_dir": str(Path(inputs_dir)),
            **{k: v for k, v in st.items() if k != "effective_protocol_errors"}}
    lock["units"].pop("match", None)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=1))
    if write_effective:
        Path(write_effective).write_text(json.dumps(EFFECTIVE_PROTOCOL, indent=1))
    return lock


def verify_lock(lock_path: Path, root: Path, inputs_dir: Path | None = None) -> dict:
    lock = json.loads(Path(lock_path).read_text())
    if lock.get("schema") != LOCK_SCHEMA:
        return {"ok": False, "mismatches": [f"lock schema {lock.get('schema')} != {LOCK_SCHEMA}"]}
    inputs_dir = Path(inputs_dir or lock["inputs_dir"])
    st = compute_state(root, inputs_dir, lock["access_table"]["path"],
                       lock.get("features", {}).get("path"))
    mm = []
    if st["effective_protocol_errors"]:
        mm.append(f"effective protocol invalid: {st['effective_protocol_errors']}")
    if st["effective_protocol_sha256"] != lock["effective_protocol_sha256"]:
        mm.append("effective protocol hash changed")
    lc, sc = lock["code_files"], st["code_files"]
    for f in sorted(set(lc) | set(sc)):
        if f not in sc:
            mm.append(f"code file removed: {f}")
        elif f not in lc:
            mm.append(f"code file added (not in lock): {f}")
        elif lc[f] != sc[f]:
            mm.append(f"code file changed: {f}")
    if st["access_table"] != lock["access_table"]:
        mm.append("access table changed")
    if st["units"]["ids"] != lock["units"]["ids"] or lock["units"]["ids"] != list(REGISTERED_UNITS):
        mm.append(f"unit IDs differ from the lock / registered list (actual {len(st['units']['ids'])})")
    for u, h in lock["units"]["manifests_sha256"].items():
        if st["units"]["manifests_sha256"].get(u) != h:
            mm.append(f"manifest changed or missing: {u}")
    ld, sd = lock["data_files_sha256"], st["data_files_sha256"]
    for f in sorted(set(ld) | set(sd)):
        if ld.get(f) != sd.get(f):
            mm.append(f"data file changed/added/removed: {f}")
    for k in ("inputs_index_sha256", "primary_family", "inference", "support_rule", "preparation_lock_sha256"):
        if st.get(k) != lock.get(k):
            mm.append(f"{k} changed")
    if "features" in lock and st.get("features") != lock["features"]:
        mm.append("features.npz changed")
    return {"ok": not mm, "mismatches": mm, "lock": str(lock_path), "n_code_files": len(sc),
            "n_data_files": len(sd), "n_units": len(st["units"]["ids"])}
