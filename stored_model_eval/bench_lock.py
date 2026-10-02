"""LOCK.json for the matched removal benchmark: build and verify (refuse on any mismatch).

Pins:
  * the bench EFFECTIVE_PROTOCOL sha256 (and that it validates);
  * sha256 of every consumed code file: stored_model_eval/*.py (incl. defenses.py; tests excluded) and every file
    under results/combined_matched_removal_benchmark_v1/scripts/ (runner, plots, admission scripts); a file added to
    or removed from either set is a mismatch;
  * dependency versions (python, numpy, scipy, scikit-learn, torch, joblib) and concept-erasure 0.2.4 with the
    installed .py tree sha256 (must equal the pinned upstream-identical tree);
  * INPUTS_INDEX.json sha256 and every input file it references (recomputed);
  * SUPPORT_FROZEN.json sha256 (frozen from labels and roles before any fit);
  * the historical native git blobs (dominant_axis_audit.json, Adult and HMDA Round 4);
  * the exact Tier-1 and Tier-2 (E1, E2, E3) unit lists, the primary family, inference and support pins.
Fitted LEACE maps are pinned separately in <private>/defenses/MAP_PINS.json as they are created (bench.MapStore);
`verify_bench_lock` also verifies every pinned map and refuses an unpinned map directory.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from .admission import sha256_file
from .bench_effective import (BENCH_EFFECTIVE, CONCEPT_ERASURE, HISTORICAL_NATIVE, bench_effective_hash,
                              primary_family, tier1_units, tier2_units, validate_bench_effective)

LOCK_SCHEMA = "stored_model_eval.bench_lock/v1"
CODE_GLOBS = (("stored_model_eval", "*.py", False), ("results/combined_matched_removal_benchmark_v1/scripts", "*", True))


def code_files(root: Path) -> dict:
    root = Path(root)
    out = {}
    for d, pat, rec in CODE_GLOBS:
        base = root / d
        it = base.rglob(pat) if rec else base.glob(pat)
        for p in sorted(it):
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith((".pyc", "~")) \
                    and p.name != ".DS_Store":
                out[str(p.relative_to(root))] = sha256_file(p)
    return out


def concept_erasure_tree() -> dict:
    """Independent recomputation of the installed concept-erasure tree hash (rule in CONCEPT_ERASURE)."""
    try:
        import concept_erasure
        from importlib.metadata import version
    except ImportError as e:  # pragma: no cover
        return {"error": repr(e)}
    pkg = Path(concept_erasure.__file__).resolve().parent
    h = hashlib.sha256()
    files = sorted(p for p in pkg.rglob("*.py") if "__pycache__" not in p.parts)
    for p in files:
        h.update(("concept_erasure/" + p.relative_to(pkg).as_posix()).encode())
        h.update(p.read_bytes())
    return {"version": version("concept-erasure"), "tree_sha256": h.hexdigest(), "n_files": len(files),
            "path": str(pkg)}


def dependencies() -> dict:
    import joblib
    import numpy
    import scipy
    import sklearn
    import torch
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__, "joblib": joblib.__version__,
            "concept-erasure": concept_erasure_tree(), "machine": platform.machine()}


def historical_blobs(root: Path) -> dict:
    out = {}
    for d, b in HISTORICAL_NATIVE["blobs"].items():
        r = subprocess.run(["git", "-C", str(root), "cat-file", "-t", b["blob"]], capture_output=True, text=True)
        out[d] = {"path": b["path"], "blob": b["blob"], "present": r.returncode == 0 and r.stdout.strip() == "blob"}
    return out


def _inference_pins(eff: dict) -> dict:
    I = eff["inference"]
    return {k: I[k] for k in ("exploratory", "primary", "quantile_method", "bars", "tau", "ni_margin", "chunk",
                              "sampling_unit", "replicate_summary", "worst_rule")}


def home_relative(obj):
    """Rewrite absolute paths under $HOME as '~/...' (keys and values) so the committed lock holds no private paths."""
    home = str(Path.home())
    if isinstance(obj, str):
        return "~" + obj[len(home):] if obj == home or obj.startswith(home + "/") else obj
    if isinstance(obj, dict):
        return {home_relative(k): home_relative(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [home_relative(v) for v in obj]
    return obj


def compute_state(root: Path, index: Path, private_root: Path, eff: dict) -> dict:
    from .bench import Inputs
    inputs = Inputs(index)
    files = inputs.all_files()
    data = {p: (sha256_file(Path(p)) if Path(p).exists() else None) for p in sorted(files)}
    sf = Path(private_root) / "support" / "SUPPORT_FROZEN.json"
    t2 = tier2_units()
    if eff == BENCH_EFFECTIVE:
        errs = validate_bench_effective(eff)
    elif inputs.synthetic:
        errs = []
    else:
        errs = ["effective protocol differs from the frozen BENCH_EFFECTIVE (override allowed for synthetic only)"]
    return home_relative({"effective_protocol_sha256": bench_effective_hash(eff), "effective_protocol_errors": errs,
            "code_files": code_files(root), "dependencies": dependencies(),
            "inputs": {"index_path": str(Path(index)), "index_sha256": inputs.sha256, "synthetic": inputs.synthetic,
                       "files_expected": files, "files_actual": data},
            "support_frozen_sha256": sha256_file(sf) if sf.exists() else None,
            "historical_native": historical_blobs(root),
            "units": {"tier1": tier1_units(), "tier2": t2},
            "primary_family": primary_family(),
            "inference": _inference_pins(eff),
            "support_rule": {k: eff["support"][k] for k in ("min_attacker_fit", "min_attacker_val", "min_assessment",
                                                            "min_supported_classes", "min_defense_fit_concept")}})


def build_bench_lock(root: Path, index: Path, private_root: Path, out: Path, eff: dict | None = None,
                     effective_out: Path | None = None) -> dict:
    from .bench import Inputs, freeze_all_support, _write_json
    eff = BENCH_EFFECTIVE if eff is None else eff
    inputs = Inputs(index)
    mm = inputs.verify_hashes()
    if mm:
        raise SystemExit("REFUSED: input files do not match INPUTS_INDEX: " + "; ".join(mm[:10]))
    sf = Path(private_root) / "support" / "SUPPORT_FROZEN.json"
    if not sf.exists():
        _write_json(sf, freeze_all_support(inputs, eff))
    st = compute_state(root, index, private_root, eff)
    if st["effective_protocol_errors"]:
        raise SystemExit(f"REFUSED: effective protocol invalid: {st['effective_protocol_errors']}")
    ce = st["dependencies"]["concept-erasure"]
    if ce.get("version") != CONCEPT_ERASURE["version"] or ce.get("tree_sha256") != CONCEPT_ERASURE["tree_sha256"]:
        raise SystemExit(f"REFUSED: installed concept-erasure {ce} is not the pinned official 0.2.4 tree")
    if not all(v["present"] for v in st["historical_native"].values()):
        raise SystemExit(f"REFUSED: historical native blobs missing: {st['historical_native']}")
    lock = {"schema": LOCK_SCHEMA, "built_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "LOCKED before any benchmark eraser/attacker/probe fit (development data; not a confirmation)",
            "worktree_head_at_build_informational": subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "private_root": home_relative(str(private_root)), "support_frozen_path": home_relative(str(sf)),
            **{k: v for k, v in st.items() if k != "effective_protocol_errors"}}
    lock["inputs"].pop("files_actual")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=1))
    if effective_out:
        Path(effective_out).parent.mkdir(parents=True, exist_ok=True)
        Path(effective_out).write_text(json.dumps(eff, indent=1))
    return {"lock": str(out), "n_code_files": len(lock["code_files"]), "n_input_files": len(lock["inputs"]
                                                                                            ["files_expected"]),
            "n_tier1": len(lock["units"]["tier1"]), "effective_protocol_sha256": lock["effective_protocol_sha256"],
            "support_frozen_sha256": lock["support_frozen_sha256"]}


def verify_bench_lock(lock_path: Path, root: Path, index: Path, private_root: Path, eff: dict | None = None) -> dict:
    from .bench import MapStore
    eff = BENCH_EFFECTIVE if eff is None else eff
    lock = json.loads(Path(lock_path).read_text())
    if lock.get("schema") != LOCK_SCHEMA:
        return {"ok": False, "mismatches": [f"lock schema {lock.get('schema')} != {LOCK_SCHEMA}"]}
    st = compute_state(root, index, private_root, eff)
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
    ld, sd = lock["dependencies"], st["dependencies"]
    for k in sorted(set(ld) | set(sd)):
        if ld.get(k) != sd.get(k):
            mm.append(f"dependency changed: {k} ({ld.get(k)} -> {sd.get(k)})")
    li, si = lock["inputs"], st["inputs"]
    if li["index_sha256"] != si["index_sha256"]:
        mm.append("INPUTS_INDEX.json changed")
    if li["files_expected"] != si["files_expected"]:
        mm.append("set of input files referenced by the index changed")
    for p, h in li["files_expected"].items():
        if si["files_actual"].get(p) != h:
            mm.append(f"input file changed or missing: {p}")
    for k in ("support_frozen_sha256", "historical_native", "units", "primary_family", "inference", "support_rule"):
        if st[k] != lock.get(k):
            mm.append(f"{k} changed")
    mp = MapStore(Path(private_root)).verify()
    mm += [f"MAP_PINS: {m}" for m in mp]
    return {"ok": not mm, "mismatches": mm, "lock": str(lock_path), "n_code_files": len(sc),
            "n_input_files": len(si["files_expected"]), "n_tier1": len(st["units"]["tier1"]),
            "n_maps_pinned": len(MapStore(Path(private_root)).pins()["maps"])}


__all__ = ["build_bench_lock", "verify_bench_lock", "home_relative", "code_files", "concept_erasure_tree", "LOCK_SCHEMA"]
