"""Pre-lock host pipeline (inner roles only; the outer role is never loaded here).

    restore-prior   fetch the predecessor's a{0,1,2}_{bank,NM4_U,DET_SEL4} archives by
                    key/version, verify SHA-256 against the committed MODEL_MANIFEST
                    pins, extract (safe members only) into private/prior_units
    basis           BASIS_MANIFEST.json: SHA-256 of every basis file, before any solve
    solve           every arm/control per anchor (solve.solve_anchor)
    variation       within-T32 decision variation on coefficient_split,
                    inner_selection and inner_check (label-free law statistics)
    audit           SC inner audit panel for one anchor (fresh attackers)
    inner-report    three-anchor inner_check contrasts vs D17 (descriptive; gates nothing)
    archive         tar a private directory to s3 (SSE, versioned, read-back verified)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

import numpy as np

from experiments.pcrl_shared_context_release_v1 import audit_panel, channel, fit_nm, laws, selection
from experiments.pcrl_task_aligned_cuts_v1 import data

from . import solve

ROOT = Path(__file__).resolve().parents[2]
STUDY = solve.STUDY
RESULTS = ROOT / "results" / STUDY
PRIVATE = RESULTS / "private"
PRIOR = PRIVATE / "prior_units"
UNITS = PRIVATE / "units"
PANELS = PRIVATE / "inner_panels"
SC_RESULTS = ROOT / "results" / "pcrl_shared_context_release_v1"
BUCKET = "pcrl-ux-archive-ed9d21fd"
ANCHORS = (0, 1, 2)
PRIOR_UNITS = tuple(f"a{a}_{u}" for a in ANCHORS for u in ("bank", "NM4_U", "DET_SEL4"))
DECLARED = ("D17", "DET_SEL4", "TASK_SEL4", "D4", "R4", "NM4PF", "D1", "R1")   # declaration order
VARIATION_ROLES = ("coefficient_split", "inner_selection", "inner_check")
INDEX = ROOT / data.INDEX_RELATIVE


def _sha(path) -> str:
    return laws.sha256_file(path)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _aws(*args: str) -> dict:
    out = subprocess.run(["aws", "--region", "us-east-1", *args, "--output", "json"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out) if out.strip() else {}


def restore_prior(fetch=None) -> dict:
    manifest = json.loads((SC_RESULTS / "MODEL_MANIFEST.json").read_text())
    PRIOR.mkdir(parents=True, exist_ok=True)
    records = []
    for unit in PRIOR_UNITS:
        pin = manifest["unit_archives"][unit]
        target = PRIOR / unit
        if (PRIOR / f"{unit}.restored").is_file():
            records.append({"unit": unit, "action": "already_present", **pin})
            continue
        with tempfile.TemporaryDirectory(dir=PRIVATE) as temp:
            tar_path = Path(temp) / f"{unit}.tar.gz"
            (fetch or _fetch)(pin["key"], pin["version_id"], tar_path)
            digest = _sha(tar_path)
            if digest != pin["archive_sha256"] or tar_path.stat().st_size != pin["bytes"]:
                raise ValueError(f"{unit}: archive differs from its MODEL_MANIFEST pin")
            with tarfile.open(tar_path, "r:gz") as stream:
                names = stream.getnames()
                if not all(not Path(n).is_absolute() and ".." not in Path(n).parts and
                           (n == unit or n.startswith(unit + "/")) for n in names):
                    raise ValueError(f"{unit}: unsafe archive member")
                stream.extractall(PRIOR, filter="data")
            files = sum(1 for p in target.rglob("*") if p.is_file())
            if files != pin["file_count"]:
                raise ValueError(f"{unit}: file count {files} differs from pin {pin['file_count']}")
        (PRIOR / f"{unit}.restored").write_text(digest + "\n")
        records.append({"unit": unit, "action": "restored_verified", **pin})
    receipt = {"schema": "pcrl-pfs-prior-restore-v1", "utc": _utc(),
               "model_manifest_sha256": _sha(SC_RESULTS / "MODEL_MANIFEST.json"),
               "records": records, "all_verified": True}
    (PRIVATE / "PRIOR_RESTORE.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _fetch(key, version, target):
    _aws("s3api", "get-object", "--bucket", BUCKET, "--key", key, "--version-id", version, str(target))


def basis_manifest() -> dict:
    out = {"schema": solve.SCHEMA_BASIS, "utc": _utc(), "study": STUDY,
           "written_before_any_solve": not any(UNITS.glob("a*_D4")), "anchors": {}}
    if not out["written_before_any_solve"]:
        raise FileExistsError("basis manifest must precede every solve")
    for anchor in ANCHORS:
        b = solve.load_basis(PRIOR / f"a{anchor}_NM4_U", PRIOR / f"a{anchor}_bank")
        det = PRIOR / f"a{anchor}_DET_SEL4"
        out["anchors"][str(anchor)] = {
            "nm4u_files_sha256": b["files_sha256"], "bank_sha256": b["bank_sha256"],
            "bank_manifest_sha256": b["bank_manifest_sha256"],
            "policy_bank_sha256": b["bank_manifest"]["policy_bank_sha256"],
            "contexts_sha256": b["bank_manifest"]["contexts_sha256"],
            "policy_columns": b["names"], "K": b["K"], "cut_count": b["cut_count"],
            "cut_count_by_role": b["cut_count_by_role"],
            "det_sel4_files_sha256": {n: _sha(det / n) for n in
                                      ("DET_SEL.json", "PARAMS.npz", "RELEASE_SPEC.json", "COMPLETE.json")},
            "d17_map_array_sha256": laws.law_identity(data.load_map(data.index(INDEX), anchor, "D17"))}
    out["prior_restore_sha256"] = _sha(PRIVATE / "PRIOR_RESTORE.json")
    path = RESULTS / "BASIS_MANIFEST.json"
    if path.exists():
        raise FileExistsError("BASIS_MANIFEST.json is write-once")
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def solve_all() -> dict:
    if not (RESULTS / "BASIS_MANIFEST.json").is_file():
        raise FileNotFoundError("write BASIS_MANIFEST.json before solving")
    value = data.index(INDEX)
    summary, reports = {}, {}
    for anchor in ANCHORS:
        report = solve.solve_anchor(PRIOR / f"a{anchor}_NM4_U", PRIOR / f"a{anchor}_bank",
                                    PRIOR / f"a{anchor}_DET_SEL4", data.load_map(value, anchor, "D17"), UNITS)
        reports[str(anchor)] = report
        summary[str(anchor)] = {r: {k: report["releases"][r].get(k) for k in
                                    ("t", "t_by_weighting", "task_U", "task_W", "assignment",
                                     "feasible", "fractional_contexts")}
                                for r in report["releases"]}
    path = RESULTS / "SOLVE_REPORTS.json"
    if path.exists():
        raise FileExistsError("SOLVE_REPORTS.json is write-once")
    path.write_text(json.dumps({"schema": "pcrl-pfs-solve-reports-v1", "anchors": reports,
                                "content": "coefficient-block aggregates only; no person rows"},
                               indent=2, sort_keys=True, default=solve._default) + "\n")
    return summary


def sources(anchor: int) -> dict:
    value = {"D17": {"kind": "historical_map", "map_name": "D17", "anchor": anchor,
                     "index_path": str(INDEX.resolve())},
             "DET_SEL4": str((PRIOR / f"a{anchor}_DET_SEL4").resolve())}
    for name in DECLARED[2:]:
        value[name] = str((UNITS / f"a{anchor}_{name}").resolve())
    if list(value) != list(DECLARED):
        raise AssertionError("declaration order changed")
    return value


def variation() -> dict:
    """Label-free law statistics on coefficient_split, inner_selection, inner_check."""
    out = {"schema": "pcrl-pfs-decision-variation-v1", "utc": _utc(), "roles": list(VARIATION_ROLES),
           "tolerance": 1e-9, "anchors": {}}
    for anchor in ANCHORS:
        role_dict, q_ref, _, _ = fit_nm.load_roles(anchor, INDEX)
        src = sources(anchor)
        per = {}
        for name in DECLARED:
            law_fn = laws.load(src[name])
            per[name] = {}
            for role in VARIATION_ROLES:
                rows = role_dict[role]
                q = law_fn(laws.legal_inputs(rows))
                diag, _ = channel.nonalias_diagnostic(q, rows["token_codes"], q_ref, rows["weights"],
                                                      rows["households"])
                within = {k: v for k, v in diag["within_t32"].items() if k != "per_state"}
                within["per_state"] = [{k: s[k] for k in ("state", "people", "V_unweighted", "V_weighted",
                                                          "varying_households")}
                                       for s in diag["within_t32"]["per_state"]]
                per[name][role] = {"people": diag["people"], "households": diag["households"],
                                   "tv_to_d17": diag["tv_to_d17"], "within_t32": within,
                                   "deterministic_emission": diag["deterministic_emission"],
                                   "stochastic_weight_fraction": diag["stochastic_weight_fraction"],
                                   "law_sha256": laws.law_identity(q)}
        out["anchors"][str(anchor)] = per
    path = RESULTS / "DECISION_VARIATION_INNER.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def audit(anchor: int) -> dict:
    return audit_panel.audit_panel(anchor, sources(anchor), INDEX, PANELS / f"a{anchor}")


def inner_report() -> dict:
    """Descriptive three-anchor inner_check contrasts vs D17 (P-route signs); gates nothing."""
    reports = {a: audit_panel.verify_complete(PANELS / f"a{a}") for a in ANCHORS}
    scores = {a: selection.check_scores_from_report(reports[a]) for a in ANCHORS}
    roles, weights = selection.ROLES, ("U", "PWGTP")
    table = {}
    for name in DECLARED:
        table[name] = {}
        for role in roles:
            for w in weights:
                per = []
                for a in ANCHORS:
                    c, d = scores[a][name][role][w], scores[a]["D17"][role][w]
                    per.append(c - d if role == selection.TASK_ROLE else d - c)
                table[name][f"{role}|{w}"] = {"three_anchor_mean": float(np.mean(per)), "anchors": per}
    out = {"schema": "pcrl-pfs-inner-report-v1", "utc": _utc(), "score_role": "inner_check",
           "signs": "task = CE_c - CE_D17; recovery = CE_D17 - CE_c (negative = less recovery)",
           "gates_nothing": True,
           "logical_to_canonical": {str(a): reports[a]["logical_to_canonical"] for a in ANCHORS},
           "panel_sha256": {str(a): {"complete": _sha(PANELS / f"a{a}" / "COMPLETE.json"),
                                     "inner_audit": _sha(PANELS / f"a{a}" / "INNER_AUDIT.json")}
                            for a in ANCHORS},
           "hist_gb_min_leaf": {str(a): reports[a]["hist_gb_min_leaf"] for a in ANCHORS},
           "contrasts_vs_D17": table}
    (RESULTS / "INNER_REPORT.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def archive(directory: str, name: str) -> dict:
    """tar.gz -> put (SSE) -> head -> get by version -> SHA compare. Originals kept."""
    source = Path(directory).resolve()
    key = f"{STUDY}/archive/{name}.tar.gz"
    with tempfile.TemporaryDirectory(dir="/opt/pcrl" if Path("/opt/pcrl").is_dir() else None) as temp:
        tar_path = Path(temp) / "a.tar.gz"
        with tarfile.open(tar_path, "w:gz") as stream:
            stream.add(source, arcname=source.name)
        digest = _sha(tar_path)
        put = _aws("s3api", "put-object", "--bucket", BUCKET, "--key", key, "--body", str(tar_path),
                   "--server-side-encryption", "AES256")
        back = Path(temp) / "b.tar.gz"
        _aws("s3api", "get-object", "--bucket", BUCKET, "--key", key, "--version-id", put["VersionId"], str(back))
        if _sha(back) != digest:
            raise ValueError("archive read-back SHA mismatch")
        record = {"key": key, "version_id": put["VersionId"], "archive_sha256": digest,
                  "bytes": tar_path.stat().st_size, "readback_sha256_verified": True, "utc": _utc(),
                  "source": str(source)}
    ledger = PRIVATE / "ARCHIVES.jsonl"
    with ledger.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("restore-prior", "basis", "solve", "variation", "inner-report"):
        sub.add_parser(name)
    item = sub.add_parser("audit"); item.add_argument("--anchor", type=int, choices=ANCHORS, required=True)
    item = sub.add_parser("archive"); item.add_argument("--dir", required=True); item.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    if args.action == "restore-prior":
        value = {k: v for k, v in restore_prior().items() if k != "records"}
    elif args.action == "basis":
        value = basis_manifest()
    elif args.action == "solve":
        value = solve_all()
    elif args.action == "variation":
        value = {"written": str(RESULTS / "DECISION_VARIATION_INNER.json")}
        variation()
    elif args.action == "audit":
        report = audit(args.anchor)
        value = {"anchor": args.anchor, "canonical": sorted(report["releases"]),
                 "aliases": report["logical_to_canonical"]}
    elif args.action == "inner-report":
        value = {"written": str(RESULTS / "INNER_REPORT.json")}
        inner_report()
    else:
        value = archive(args.dir, args.name)
    print(json.dumps(value, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
