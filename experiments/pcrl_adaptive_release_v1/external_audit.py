"""Separate same-host 2018 development audit for frozen continuous A releases.

J and the five historical erasure/OptNet maps emit 16 continuous coordinates,
not one of the adaptive study's 17 tokens. This module uses the inherited
continuous-wire auditor with a one-state bookkeeping law; no token is added.
Only globally assigned inner roles can be loaded here. Results are contextual
frontiers, never matched finite-token algorithmic controls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import audit as inherited, data
from . import evaluate, roles


METHODS = ("J", "leace_A0", "splince_A0", "optnet16_C1", "optnet16_L1", "optnet16_L2")
EXTERNAL_INDEX_SCHEMA = "pcrl-external-release-inputs-private-v1"
HISTORICAL_COMPETITIVE_COMMIT = "7f961d5c7f6f0562efcb25a27a77bb5221c279a7"
POOLS = ("representation_fit", "downstream_fit", "downstream_validation", "attacker_fit")
OUTER_POOLS = (*POOLS, "attacker_validation")
# The J inner slates were fitted and archived before the five-pool outer-only
# correction. Keep their actual source pin; the outer scorer records its new SHA.
FROZEN_J_INNER_SOURCE_SHA256 = "36b2bdd3692d296364d52aff12453d5c14d7b4bb66a461556548dc3fe5c17cba"
INNER_ROLES = ("audit_fit", "inner_selection", "inner_check")
H_ROLES = (*inherited.ROLES, "attack:B/SEX", "attack:B/RAC1P")


def _sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _unit_law(rows: Mapping[str, Any]) -> np.ndarray:
    return np.ones((len(rows["ids"]), 1), dtype=np.float64)


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def pinned_external_record(index_path: str | Path, expected_sha256: str,
                           anchor: int, method: str) -> tuple[dict, dict]:
    """Pin the private 15-file index before locating one declared release."""
    source = Path(index_path)
    if (not source.is_file() or len(expected_sha256) != 64 or
            _sha(source) != expected_sha256):
        raise ValueError("external private index SHA-256 mismatch")
    index = json.loads(source.read_text())
    if (index.get("schema") != EXTERNAL_INDEX_SCHEMA or
            index.get("historical_competitive_commit") != HISTORICAL_COMPETITIVE_COMMIT or
            index.get("verified_release_file_count") != 15 or
            index.get("service_view_mismatch_count") != 0):
        raise ValueError("historical external index provenance is incomplete")
    if method not in METHODS or method == "J" or anchor not in (0, 1, 2):
        raise ValueError("undeclared external continuous release")
    found = [record for record in index.get("release_files", [])
             if record.get("anchor") == anchor and record.get("method") == method]
    if len(found) != 1:
        raise ValueError("undeclared or duplicated external continuous release")
    record = found[0]
    member = Path(record["archive_member_path"])
    if (member.is_absolute() or ".." in member.parts or
            member.name != "releases.npz" or
            f"seed_{anchor}" not in member.parts or method not in member.parts):
        raise ValueError("unsafe or misidentified archive member")
    return index, record


def archived_ab_wire(pool: Mapping[str, Any], aux: np.ndarray) -> np.ndarray:
    """The stored external coalition order is H_A, aux16, H_B."""
    return np.column_stack((pool["ha"], aux, pool["hb"]))


def verified_external_aux(path: str | Path, expected_sha256: str,
                          prepared: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Verify source hash and all three original service views before filtering."""
    source = Path(path)
    if not source.is_file() or _sha(source) != expected_sha256:
        raise ValueError("external release SHA-256 mismatch")
    result = {}
    with np.load(source, allow_pickle=False) as stored:
        for name in POOLS:
            pool = prepared["ctx"]["pools"][name]
            a = np.asarray(stored[f"wire/A/{name}"])
            b = np.asarray(stored[f"wire/B/{name}"])
            ab = np.asarray(stored[f"wire/AB/{name}"])
            n = len(pool["ids"])
            if (a.shape != (n, 20) or b.shape != (n, 2) or
                    ab.shape != (n, 22) or
                    a.dtype != np.float64 or b.dtype != np.float64 or
                    ab.dtype != np.float64 or not np.isfinite(a).all() or
                    not np.isfinite(b).all() or not np.isfinite(ab).all()):
                raise ValueError("external release width or original-row alignment changed")
            aux = a[:, 4:].copy()
            if (not _same_bytes(a[:, :4], pool["ha"]) or
                    not _same_bytes(b, pool["hb"]) or
                    not _same_bytes(ab, archived_ab_wire(pool, aux))):
                raise ValueError("external release service byte parity failed")
            result[name] = aux
    return result


def continuous_role_rows(prepared: Mapping[str, Any], role: str, method: str,
                         *, external_aux: Mapping[str, np.ndarray] | None = None) -> dict:
    """Attach the frozen continuous A aux to the same global household rows."""
    if role not in INNER_ROLES or method not in METHODS:
        raise ValueError("undeclared inner role or continuous release")
    if method != "J" and external_aux is None:
        raise ValueError("external release requires verified per-pool aux arrays")
    rows = roles.pooled_role(prepared, role)
    pieces = []
    for name in POOLS:
        pool = prepared["ctx"]["pools"][name]
        mask = np.asarray([roles.role_of(h) == role for h in pool["households"]], dtype=bool)
        aux = np.asarray(pool["J"] if method == "J" else external_aux[name])
        if (aux.shape != (len(pool["ids"]), 16) or
                not np.issubdtype(aux.dtype, np.floating) or
                not np.isfinite(aux).all()):
            raise ValueError("historical continuous auxiliary width/alignment changed")
        pieces.append(np.asarray(aux[mask], dtype=np.float64))
    attached = dict(rows)
    attached["aux"] = np.concatenate(pieces)
    if attached["aux"].shape != (len(rows["ids"]), 16):
        raise ValueError("continuous auxiliary rows differ from global role")
    return attached


def _no_aux(rows: Mapping[str, Any]) -> dict:
    return {key: value for key, value in rows.items() if key != "aux"}


def _route_bank(role: str, release_id: str, own_by_role: Mapping[str, dict],
                h_by_role: Mapping[str, dict]) -> dict:
    _, view, target = inherited.parse_role(role)
    return inherited.build_role_route_bank(
        role, release_id, own_by_role[role], h_by_role[role],
        a_same=own_by_role.get(f"attack:A/{target}") if view == "AB" else None,
        b_h_only=h_by_role.get(f"attack:B/{target}") if view == "AB" else None)


def audit_external_panel(anchor: int, methods: tuple[str, ...],
                         input_index_path: str | Path,
                         external_index_path: str | Path,
                         external_index_sha256: str,
                         external_root: str | Path,
                         output_dir: str | Path, *,
                         slate: str = "catchup", resume: bool = False) -> dict:
    """Fit/lock equal-slate inner routes; score only inner_check households."""
    if anchor not in (0, 1, 2) or slate not in ("standard", "catchup"):
        raise ValueError("undeclared anchor or slate")
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("duplicate or undeclared continuous release")
    root = Path(output_dir)
    if "private" not in root.parts:
        raise ValueError("continuous audit weights and contributions must remain private")
    if (root / "COMPLETE.json").exists():
        raise FileExistsError("completed external audit is immutable")
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("partial external audit needs explicit technical resume")
    if (root / "INNER_AUDIT.json").exists():
        raise FileExistsError("partial external report retained for incident review")
    input_path = Path(input_index_path).resolve()
    value = data.index(input_path)
    if data._sanitized_record(value, anchor) is None:
        raise FileNotFoundError("verified label-stripped 2018 prepared receipt required")
    external_index_file = Path(external_index_path).resolve()
    if _sha(external_index_file) != external_index_sha256:
        raise ValueError("external private index SHA-256 mismatch")
    external_index = json.loads(external_index_file.read_text())
    if (external_index.get("schema") != EXTERNAL_INDEX_SCHEMA or
            external_index.get("historical_competitive_commit") != HISTORICAL_COMPETITIVE_COMMIT or
            external_index.get("verified_release_file_count") != 15 or
            external_index.get("service_view_mismatch_count") != 0):
        raise ValueError("historical external release index is incomplete")
    j_records = [entry for entry in external_index.get("J", [])
                 if entry.get("anchor") == anchor]
    if (len(j_records) != 1 or
            j_records[0]["prepared_source_sha256"] !=
            data.member_record(value, anchor, "prepared")["sha256"]):
        raise ValueError("J source and prepared anchor pin differ")
    prepared = data.load_prepared(value, anchor)
    if "labels" in prepared["ctx"]["pools"]["attacker_validation"]:
        raise ValueError("historical final labels unexpectedly present")
    h_rows = {role: roles.pooled_role(prepared, role) for role in INNER_ROLES}
    inherited.assert_household_disjoint(*(h_rows[role]["households"] for role in INNER_ROLES))
    release_rows = {}
    descriptors = {}
    for method in methods:
        if method == "J":
            release_rows[method] = {role: continuous_role_rows(prepared, role, "J")
                                    for role in INNER_ROLES}
            descriptors[method] = {
                "kind": "archived_continuous_J16",
                "prepared_source_sha256": j_records[0]["prepared_source_sha256"],
                "service_order_archived": "A=H_A4,J16; AB=H_A4,J16,H_B2",
            }
        else:
            _, record = pinned_external_record(
                external_index_file, external_index_sha256, anchor, method)
            source = (Path(external_root) / record["archive_member_path"]).resolve()
            if not source.is_relative_to(Path(external_root).resolve()):
                raise ValueError("external release escaped private root")
            aux = verified_external_aux(source, record["sha256"], prepared)
            release_rows[method] = {
                role: continuous_role_rows(prepared, role, method, external_aux=aux)
                for role in INNER_ROLES}
            descriptors[method] = {
                "kind": "archived_continuous_aux16",
                "release_source_member": record["archive_member_path"],
                "release_source_sha256": record["sha256"],
                "service_byte_parity_four_pools": True,
                "service_order_archived": "A=H_A4,aux16; AB=H_A4,aux16,H_B2",
            }
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    h_by_role = {}
    for role_index, role in enumerate(H_ROLES):
        h_by_role[role] = inherited.fit_role_slate(
            h_rows["audit_fit"], _unit_law(h_rows["audit_fit"]),
            h_rows["inner_selection"], _unit_law(h_rows["inner_selection"]),
            role, root / "H" / role.replace(":", "_").replace("/", "_"),
            26000 + 1000 * anchor + role_index,
            release_id="H", slate=slate)
    h_locks, h_scores = {}, {}
    for role in inherited.ROLES:
        bank = _route_bank(role, "H", h_by_role, h_by_role)
        h_locks[role] = inherited.select_frozen_routes(
            h_rows["inner_selection"], _unit_law(h_rows["inner_selection"]),
            role, bank, release_id="H")
        h_scores[role] = inherited.score_frozen_route(
            h_rows["inner_check"], _unit_law(h_rows["inner_check"]), h_locks[role])
    report_releases, contributions = {}, {}
    for method in methods:
        release_id = method
        own = {}
        for role_index, role in enumerate(inherited.ROLES):
            own[role] = inherited.fit_role_slate(
                release_rows[method]["audit_fit"], _unit_law(h_rows["audit_fit"]),
                release_rows[method]["inner_selection"], _unit_law(h_rows["inner_selection"]),
                role, root / release_id / role.replace(":", "_").replace("/", "_"),
                26000 + 1000 * anchor + role_index,
                release_id=release_id, slate=slate)
        role_results = {}
        prefix = hashlib.sha256(release_id.encode()).hexdigest()[:12]
        for role in inherited.ROLES:
            bank = _route_bank(role, release_id, own, h_by_role)
            lock = inherited.select_frozen_routes(
                release_rows[method]["inner_selection"], _unit_law(h_rows["inner_selection"]),
                role, bank, release_id=release_id)
            score = inherited.score_frozen_route(
                release_rows[method]["inner_check"], _unit_law(h_rows["inner_check"]), lock)
            h_score = h_scores[role]
            contributions.update(evaluate._private_contributions(prefix, role, score, h_score))
            role_results[role] = {
                "n_people": len(score["loss"]),
                "n_households": len(set(score["households"])),
                "candidate": score["scores"], "H": h_score["scores"],
                "H_minus_candidate": {w: h_score["scores"][w] - score["scores"][w]
                                      for w in ("U", "PWGTP")},
                "selected_candidate": lock["selected"],
                "candidate_count": lock["candidate_count"],
                "candidate_validation_scores": lock["scores"],
                "candidate_selection_rule": lock["rule"],
                "selected_route": {k: lock["route"].get(k) for k in
                                   ("source_view", "wire", "model_sha256")},
                "H_selected_candidate": h_locks[role]["selected"],
                "H_validation_scores": h_locks[role]["scores"],
                "H_selected_route": {k: h_locks[role]["route"].get(k) for k in
                                     ("source_view", "wire", "model_sha256")},
                "fit_missing_classes": own[role]["fit_missing_classes"],
                "selection_missing_classes": own[role]["validation_missing_classes"],
            }
        report_releases[release_id] = {
            "roles": role_results, "source": descriptors[method],
            "private_contribution_prefix": prefix,
        }
    contributions_path = root / "PANEL_CONTRIBUTIONS.npz"
    if contributions_path.exists():
        raise FileExistsError("existing private contributions retained")
    np.savez_compressed(contributions_path, **contributions)
    report = {
        "schema": "pcrl-adaptive-continuous-inner-audit-v1", "anchor": anchor,
        "not_confirmation": True, "outer_pool_opened": False,
        "fit_role": "audit_fit", "selection_role": "inner_selection",
        "score_role": "inner_check", "slate": slate,
        "input_index_sha256": _sha(input_path),
        "external_index_sha256": external_index_sha256,
        "source_code_sha256": {"external_audit.py": _sha(__file__),
                               "inherited_audit.py": _sha(inherited.__file__),
                               "roles.py": _sha(roles.__file__)},
        "internal_coalition_feature_order": "H_A4,H_B2,aux16 consistently for all fresh AB fits and scores; a fixed column permutation of archived H_A4,aux16,H_B2",
        "release_contract": "continuous 16-coordinate auxiliary beside unchanged H; no token emitted by this contextual comparator",
        "releases": report_releases,
        "contributions_relative_path": contributions_path.name,
        "contributions_sha256": _sha(contributions_path),
    }
    _write_atomic(root / "INNER_AUDIT.json", report)
    evaluate._seal_private_permissions(root)
    receipt = {
        "schema": "pcrl-adaptive-continuous-inner-complete-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "anchor": anchor, "release_ids": sorted(methods),
        "input_index_sha256": report["input_index_sha256"],
        "external_index_sha256": external_index_sha256,
        "artifacts": evaluate._inventory(root),
    }
    _write_atomic(root / "COMPLETE.json", receipt)
    (root / "COMPLETE.json").chmod(0o600)
    return report


def attach_locked_outer_j(prepared: Mapping[str, Any],
                          unlocked_rows: Mapping[str, Any]) -> dict:
    """Align J16 to already-unlocked outer rows, preserving their H and labels."""
    # The ordinary prepared loader strips fifth-pool labels. The locked outer
    # gate supplies those from the verified original; compare every nonlabel
    # source coordinate here before attaching the continuous J auxiliary.
    for key in ("ids", "households", "weights", "ha", "hb"):
        pieces = []
        for name in OUTER_POOLS:
            pool = prepared["ctx"]["pools"][name]
            mask = np.asarray([roles.role_of(h) == "outer_assessment"
                               for h in pool["households"]], dtype=bool)
            pieces.append(np.asarray(pool[key])[mask])
        if not _same_bytes(np.concatenate(pieces), unlocked_rows[key]):
            raise ValueError("J source differs from verified outer gate rows")
    pieces = []
    for name in OUTER_POOLS:
        pool = prepared["ctx"]["pools"][name]
        mask = np.asarray([roles.role_of(h) == "outer_assessment"
                           for h in pool["households"]], dtype=bool)
        aux = np.asarray(pool["J"])
        if (aux.shape != (len(pool["ids"]), 16) or
                not np.issubdtype(aux.dtype, np.floating) or
                not np.isfinite(aux).all()):
            raise ValueError("frozen outer J auxiliary schema changed")
        pieces.append(np.asarray(aux[mask], dtype=np.float64))
    attached = dict(unlocked_rows)
    attached["aux"] = np.concatenate(pieces)
    if attached["aux"].shape != (len(unlocked_rows["ids"]), 16):
        raise ValueError("J auxiliary differs from verified outer gate rows")
    return attached


def score_locked_j_outer(anchor: int, input_index_path: str | Path,
                         inner_panel_dir: str | Path,
                         selection_lock_path: str | Path,
                         expected_lock_sha256: str,
                         output_dir: str | Path) -> dict:
    """Score saved J/H predictors only, after remote-verified selection unlock."""
    from . import outer_access, outer_audit

    # This gate is first: even the prepared/index object is not opened before
    # the coordinator's committed lock and remote-verification receipt pass.
    lock = outer_access.verify_outer_unlock(
        selection_lock_path, expected_lock_sha256)
    if anchor not in (0, 1, 2):
        raise ValueError("registered anchor required")
    output = Path(output_dir).resolve()
    if "private" not in output.parts:
        raise ValueError("continuous outer contributions must remain private")
    if output.exists():
        raise FileExistsError("outer contextual score already exists; preserve first record")
    contextual = lock.get("external_context", {}).get("J", {}).get(str(anchor))
    if (not isinstance(contextual, dict) or
            not isinstance(contextual.get("inner_complete_sha256"), str) or
            not isinstance(contextual.get("inner_audit_sha256"), str)):
        raise PermissionError("three-anchor J context was not pinned in selection lock")
    inner_root = Path(inner_panel_dir).resolve()
    if "private" not in inner_root.parts:
        raise ValueError("completed J inner panel must remain private")
    complete_path = inner_root / "COMPLETE.json"
    report_path = inner_root / "INNER_AUDIT.json"
    if (_sha(complete_path) != contextual["inner_complete_sha256"] or
            _sha(report_path) != contextual["inner_audit_sha256"]):
        raise ValueError("J inner artifacts differ from committed selection lock")
    complete = json.loads(complete_path.read_text())
    inner = json.loads(report_path.read_text())
    if (complete.get("schema") != "pcrl-adaptive-continuous-inner-complete-v1" or
            complete.get("anchor") != anchor or
            complete.get("artifacts") != evaluate._inventory(inner_root) or
            inner.get("schema") != "pcrl-adaptive-continuous-inner-audit-v1" or
            inner.get("anchor") != anchor or
            inner.get("outer_pool_opened") is not False or
            "J" not in complete.get("release_ids", []) or
            "J" not in inner.get("releases", {}) or
            set(complete.get("release_ids", [])) != set(inner.get("releases", {})) or
            inner.get("source_code_sha256") != {
                "external_audit.py": FROZEN_J_INNER_SOURCE_SHA256,
                "inherited_audit.py": _sha(inherited.__file__),
                "roles.py": _sha(roles.__file__) }):
        raise ValueError("frozen J inner receipt/source/selection differs")
    index_file = Path(input_index_path).resolve()
    if (not index_file.is_file() or _sha(index_file) != lock["input_index_sha256"] or
            _sha(index_file) != inner.get("input_index_sha256")):
        raise ValueError("prepared input index differs from J inner/selection lock")
    index = data.index(index_file)
    j_source = inner["releases"]["J"].get("source", {})
    if (j_source.get("kind") != "archived_continuous_J16" or
            j_source.get("prepared_source_sha256") !=
            data.member_record(index, anchor, "prepared")["sha256"]):
        raise ValueError("J prepared source differs from frozen inner release")
    # Reconstruct exact saved route IDs and verify all fitted-model hashes.
    # This function does not fit predictors or reselect on any outer scores.
    own_locks, h_locks = outer_audit.reconstruct_selected_routes(
        inner_root, inner, "J")
    unlocked = outer_access.load_locked_outer_role(
        index_file, anchor, selection_lock_path, expected_lock_sha256)
    if data._sanitized_record(index, anchor) is None:
        raise FileNotFoundError("verified label-stripped prepared receipt required")
    prepared = data.load_prepared(index, anchor)
    if "labels" in prepared["ctx"]["pools"]["attacker_validation"]:
        raise ValueError("historical final labels unexpectedly present")
    j_rows = attach_locked_outer_j(prepared, unlocked)
    law = _unit_law(unlocked)
    scores, contributions = {}, {}
    prefix = inner["releases"]["J"]["private_contribution_prefix"]
    for role in inherited.ROLES:
        score = inherited.score_frozen_route(j_rows, law, own_locks[role])
        h_score = inherited.score_frozen_route(unlocked, law, h_locks[role])
        contributions.update(evaluate._private_contributions(
            prefix, role, score, h_score))
        scores[role] = {
            "n_people": len(score["loss"]),
            "n_households": len(set(score["households"])),
            "candidate": score["scores"], "H": h_score["scores"],
            "H_minus_candidate": {w: h_score["scores"][w] - score["scores"][w]
                                  for w in ("U", "PWGTP")},
            "selected_candidate": own_locks[role]["selected"],
            "H_selected_candidate": h_locks[role]["selected"],
        }
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    contributions_path = output / "OUTER_CONTRIBUTIONS.npz"
    np.savez_compressed(contributions_path, **contributions)
    report = {
        "schema": "pcrl-adaptive-continuous-outer-J-v1",
        "anchor": anchor, "assessment_year": 2018,
        "development_only": True, "contextual_not_matched_token_comparator": True,
        "no_outer_fit_or_selection": True,
        "selection_lock_sha256": expected_lock_sha256,
        "inner_complete_sha256": contextual["inner_complete_sha256"],
        "inner_audit_sha256": contextual["inner_audit_sha256"],
        "index_sha256": _sha(index_file),
        "source_code_sha256": {"external_audit.py": _sha(__file__),
                               "outer_audit.py": _sha(outer_audit.__file__),
                               "inherited_audit.py": _sha(inherited.__file__)},
        "releases": {"J": {"source": j_source, "roles": scores,
                            "private_contribution_prefix": prefix}},
        "contributions_relative_path": contributions_path.name,
        "contributions_sha256": _sha(contributions_path),
    }
    _write_atomic(output / "OUTER_AUDIT.json", report)
    evaluate._seal_private_permissions(output)
    receipt = {
        "schema": "pcrl-adaptive-continuous-outer-complete-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "anchor": anchor, "release_ids": ["J"],
        "selection_lock_sha256": expected_lock_sha256,
        "artifacts": evaluate._inventory(output),
    }
    _write_atomic(output / "COMPLETE.json", receipt)
    (output / "COMPLETE.json").chmod(0o600)
    return report


def outer_j_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=score_locked_j_outer.__doc__)
    parser.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--inner-panel-dir", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = score_locked_j_outer(
        args.anchor, args.index, args.inner_panel_dir, args.selection_lock,
        args.lock_sha256, args.out)
    print(json.dumps({"schema": report["schema"], "anchor": args.anchor,
                      "complete_sha256": _sha(args.out / "COMPLETE.json")},
                     sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=int, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--external-index", type=Path, required=True)
    parser.add_argument("--external-index-sha256", required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slate", choices=("standard", "catchup"), default="catchup")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    report = audit_external_panel(
        args.anchor, tuple(args.methods), args.index, args.external_index,
        args.external_index_sha256, args.external_root, args.out,
        slate=args.slate, resume=args.resume)
    print(json.dumps({"schema": report["schema"], "anchor": args.anchor,
                      "methods": sorted(report["releases"]),
                      "complete_sha256": _sha(args.out / "COMPLETE.json")},
                     sort_keys=True))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "outer-j":
        outer_j_main(sys.argv[2:])
    else:
        main()
