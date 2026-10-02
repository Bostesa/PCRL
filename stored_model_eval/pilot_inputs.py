"""Build the run_v1 inputs: task_labels_v1.npz (separate file; labels.npz is never rewritten) and v2 manifests for
exactly the 26 registered units, selected by ID from the preparation's 152 manifests.

Alignment is verified by IDs, not lengths: the task-label source arrays carry their own row ids; they are joined to
labels.npz by row_id (ID sets must be equal, no duplicates) and every reference column supplied with them (record
keys recomputed from the raw rows, the sensitive attributes) must agree row by row after the join. The joined
arrays are written in labels.npz row order together with `ref_<name>` copies, and every v2 manifest declares a
reference check so admission re-verifies the join on every load.

v2 manifest additions (Addendum D1 #5): task_labels (y_task_<purpose>), clean_representations (rep_p<k> from the
forward cache; used by the A4 LRT on noise units), the hash-pinned checkpoint (U1 frozen head), unit_id / purpose /
attribute, derived_from (original manifest path + sha256), the per-role support rule.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .admission import admit, sha256_file
from .effective import PURPOSES, REGISTERED_UNITS, unit_info
from .guards import require_outside_git
from .pilot import select_units

TASK_FILE = "task_labels_v1.npz"


def join_by_id(ids_target: np.ndarray, ids_source: np.ndarray) -> np.ndarray:
    """Positions in source for each target id. Refuses duplicates and unequal ID sets."""
    ids_target, ids_source = np.asarray(ids_target), np.asarray(ids_source)
    for name, ids in (("target", ids_target), ("source", ids_source)):
        u, c = np.unique(ids, return_counts=True)
        if (c > 1).any():
            raise ValueError(f"{name} ids contain {int((c > 1).sum())} duplicates")
    st, ss = set(ids_target.tolist()), set(ids_source.tolist())
    if st != ss:
        raise ValueError(f"ID sets differ: {len(st - ss)} target ids missing from source, {len(ss - st)} extra")
    pos = {k: i for i, k in enumerate(ids_source.tolist())}
    return np.array([pos[k] for k in ids_target.tolist()], dtype=np.int64)


def build_inputs(orig_dir: Path, out_dir: Path, source: dict, checkpoint: dict, *, reference_keys=None,
                 log=print) -> dict:
    """source: {"row_id": ids, "y_task_<purpose>": ..., "ref_<name>": ...} from the dataset object.
    checkpoint: {"path": ..., "sha256": ...} (verified here)."""
    orig_dir, out_dir = Path(orig_dir), Path(out_dir)
    require_outside_git(out_dir, "pilot inputs")
    sel = select_units(orig_dir)
    if not sel["match"]:
        raise ValueError(f"registered units missing from {orig_dir}: {sel['missing']}")
    if sha256_file(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("checkpoint sha256 mismatch")
    labels_path = orig_dir / "labels.npz"
    labels_sha_before = sha256_file(labels_path)
    with np.load(labels_path, allow_pickle=False) as z:
        lab = {k: z[k] for k in z.files}
    pos = join_by_id(lab["row_id"], source["row_id"])
    checks = {}
    ref_names = reference_keys or sorted(k[len("ref_"):] for k in source if k.startswith("ref_"))
    for name in ref_names:
        a, b = lab[name], np.asarray(source[f"ref_{name}"])[pos]
        n_bad = int((np.asarray(a).astype(str) != np.asarray(b).astype(str)).sum())
        checks[name] = {"n_rows": int(len(a)), "n_disagree": n_bad}
        if n_bad:
            raise ValueError(f"reference column {name}: {n_bad} rows disagree after joining by row_id")
    out = {"row_id": lab["row_id"]}
    for p in PURPOSES:
        y = np.asarray(source[f"y_task_{p}"]).astype(np.int64)[pos]
        if y.min() < 0 or y.max() >= PURPOSES[p]["n_task_classes"]:
            raise ValueError(f"y_task_{p} outside 0..{PURPOSES[p]['n_task_classes'] - 1}")
        out[f"y_task_{p}"] = y
    for name in ref_names:
        out[f"ref_{name}"] = np.asarray(source[f"ref_{name}"])[pos]
    out_dir.mkdir(parents=True, exist_ok=True)
    tpath = out_dir / TASK_FILE
    if tpath.exists():
        raise ValueError(f"{tpath} exists; inputs are written once (remove deliberately to rebuild)")
    np.savez(tpath, **out)
    tsha = sha256_file(tpath)
    if sha256_file(labels_path) != labels_sha_before:
        raise RuntimeError("labels.npz changed during the build")
    units = {}
    for u in REGISTERED_UNITS:
        info = unit_info(u)
        src = Path(sel["manifests"][u])
        man = json.loads(src.read_text())
        man["unit_id"] = u
        man["purpose"], man["attribute"] = info["purpose"], info["attribute"]
        man["purpose_head"] = info["purpose"]
        man["output_object"] = "logit_vector"
        man["derived_from"] = {"path": str(src), "sha256": sha256_file(src)}
        man["support_rule"] = {"attacker_fit": 100, "attacker_val": 30, "assessment": 100}
        man["files"]["task"] = {"path": str(tpath), "sha256": tsha}
        man["files"]["checkpoint"] = {"path": str(checkpoint["path"]), "sha256": checkpoint["sha256"]}
        man["arrays"]["task_labels"] = {"file": "task", "key": f"y_task_{info['purpose']}", "ids": "row_id"}
        if info["kind"] == "noise":
            man["arrays"]["clean_representations"] = {"file": "fwd", "key": f"rep_p{info['purpose_index']}",
                                                      "ids": "row_id"}
        if info["attribute"] in ref_names:
            man["reference_checks"] = [{"array": "labels", "file": "task", "key": f"ref_{info['attribute']}",
                                        "ids": "row_id"}]
        mp = out_dir / f"manifest_{u}.json"
        if mp.exists():
            raise ValueError(f"{mp} exists; inputs are written once")
        mp.write_text(json.dumps(man, indent=1))
        rec = admit(man, out_dir)
        if rec["status"] != "ADMITTED":
            raise ValueError(f"v2 manifest {u} not admitted: {rec['errors']}")
        units[u] = {"manifest": str(mp), "sha256": sha256_file(mp), "derived_from": man["derived_from"],
                    "admission": rec["status"]}
    index = {"built_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "orig_dir": str(orig_dir), "n_orig_manifests": sel["n_available"], "selected": sel["selected"],
             "labels_npz": {"path": str(labels_path), "sha256": labels_sha_before, "rewritten": False},
             "task_labels": {"path": str(tpath), "sha256": tsha, "keys": sorted(out)},
             "alignment": {"method": "join by row_id; reference columns compared after the join",
                           "reference_checks": checks},
             "checkpoint": checkpoint, "units": units}
    (out_dir / "INPUTS_INDEX.json").write_text(json.dumps(index, indent=1))
    log(f"[inputs] wrote {len(units)} v2 manifests + {TASK_FILE} to {out_dir}")
    return index
