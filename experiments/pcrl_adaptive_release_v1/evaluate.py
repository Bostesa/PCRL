"""Independent inner-development audit of saved 17-token adaptive releases.

The caller supplies frozen channels and private routers. This module opens only
the globally assigned audit_fit, inner_selection and inner_check roles from the
hash-verified 2018 prepared object. It never opens outer-assessment labels.
All fitted predictors and person/household contributions remain private.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import audit as inherited_audit, data
from . import audit, roles


H_ROLES = (*audit.ROLES, "attack:B/SEX", "attack:B/RAC1P")
POOL_ROLES = ("audit_fit", "inner_selection", "inner_check")


def _sha(path: str | Path) -> str:
    return data.sha256_file(path)


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _pin(path: str | Path, expected: str) -> Path:
    source = Path(path).resolve()
    if (not source.is_file() or not isinstance(expected, str) or
            len(expected) != 64 or _sha(source) != expected):
        raise ValueError("frozen artifact is missing or differs from SHA-256 pin")
    return source


def _artifact_record(path: Path, index_dir: Path) -> dict[str, str]:
    return {"relative_to_index_directory": os.path.relpath(path, index_dir),
            "sha256": _sha(path)}


def _slug(label: str) -> str:
    if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError("release ID must be a safe nonempty label")
    return label


def _json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def _channel_artifact(spec: Mapping[str, Any]) -> tuple[np.ndarray, Path]:
    q = np.asarray(spec["Q"], dtype=np.float64)
    if (q.ndim != 2 or len(q) < 1 or q.shape[1] != 17 or
            not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 1) or
            np.max(np.abs(q.sum(axis=1) - 1)) > 1e-10):
        raise ValueError("invalid saved 17-token channel")
    source = _pin(spec["channel_artifact_path"], spec["channel_artifact_sha256"])
    with np.load(source, allow_pickle=False) as saved:
        if list(saved) != ["Q"]:
            raise ValueError("channel artifact must contain only Q")
        frozen = np.asarray(saved["Q"], dtype=np.float64)
    if not np.array_equal(q, frozen):
        raise ValueError("in-memory channel differs from pinned artifact")
    return q, source


def token_law_for_release(spec: Mapping[str, Any], rows: dict) -> np.ndarray:
    """Evaluate a pinned channel/router or task-only original-person law privately."""
    if "task_only_model_dir" in spec:
        from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
        from . import task_baselines

        if "Q" in spec or "router" in spec:
            raise ValueError("task-only person law cannot be represented as a T0 channel")
        directory = Path(spec["task_only_model_dir"]).resolve()
        if "private" not in directory.parts:
            raise ValueError("task-only frozen model must remain private")
        _pin(directory / "TASK_ONLY.json", spec["task_only_receipt_sha256"])
        model, _ = task_baselines.load_task_only(directory)
        mode = spec.get("mode")
        publish = float(spec.get("publish"))
        if (mode not in ("unmodified", "constant_replacement", "randomized_response") or
                publish not in (0.5, 0.75, 0.9, 1.0) or
                (mode == "unmodified" and publish != 1.0) or
                spec.get("constant_token", 0) != 0):
            raise ValueError("task-only law requires a registered mode/rate and existing token 0")
        local = RuntimeInputs(np.asarray(rows["x"]), np.asarray(rows["ha"]))
        law = task_baselines.task_only_control_law(
            model, local, mode=mode, publish=publish, constant_token=0)
        if (law.shape != (len(local.x_a), 17) or not np.isfinite(law).all() or
                np.any(law < 0) or np.max(np.abs(law.sum(axis=1)-1)) > 1e-10):
            raise ValueError("invalid private original-person token law")
        return law
    q, _ = _channel_artifact(spec)
    router = spec["router"]
    if router == "T0":
        if len(q) != 32:
            raise ValueError("stored T0 router requires 32 parent rows")
        leaf = np.asarray(rows["token_codes"])
    elif callable(router):
        _pin(spec["router_artifact_path"], spec["router_sha256"])
        if not isinstance(spec.get("router_entrypoint"), str) or not spec["router_entrypoint"]:
            raise ValueError("callable router needs a replay entrypoint")
        # The router receives local X_A/H_A and the verified historical code,
        # teacher posterior and residual derived from those allowed inputs.
        # On 2018 these stored Linux x86 values avoid Mac float32 re-encoding.
        # B's H, Y/S labels, household role, weights and IDs stay inaccessible.
        legal = {name: np.asarray(rows[name]) for name in
                 ("x", "ha", "token_codes", "teacher_p", "residual")}
        leaf = np.asarray(router(legal))
    else:
        raise ValueError("router must be stored T0 or a pinned private callable")
    return audit.person_token_law(q, leaf)


def _registry_routes(role: str, release_id: str, own: dict, h_only: dict,
                     own_by_role: dict, h_by_role: dict) -> dict:
    _, view, target = audit.parse_role(role)
    kwargs = ({"a_same": own_by_role[f"attack:A/{target}"],
               "b_h_only": h_by_role[f"attack:B/{target}"]}
              if view == "AB" else {})
    return audit.candidate_route_bank(role, release_id, own, h_only, **kwargs)


def _private_contributions(prefix: str, role: str, score: dict, h_score: dict) -> dict[str, np.ndarray]:
    if (not np.array_equal(score["ids"], h_score["ids"]) or
            not np.array_equal(score["households"], h_score["households"]) or
            not np.array_equal(score["weights"], h_score["weights"])):
        raise ValueError(f"H/release score rows differ for {prefix}/{role}")
    household = np.asarray(score["households"]).astype(str)
    weights = np.asarray(score["weights"], dtype=np.float64)
    candidate = np.asarray(score["loss"], dtype=np.float64)
    h_loss = np.asarray(h_score["loss"], dtype=np.float64)
    labels, inverse = np.unique(household, return_inverse=True)
    count = np.bincount(inverse, minlength=len(labels)).astype(np.int64)
    w_sum = np.bincount(inverse, weights=weights, minlength=len(labels))
    key = f"{prefix}__{role.replace(':', '_').replace('/', '_')}"
    return {
        f"{key}_ids": np.asarray(score["ids"]).astype(str),
        f"{key}_households": household,
        f"{key}_weights": weights,
        f"{key}_candidate_loss": candidate,
        f"{key}_H_loss": h_loss,
        f"{key}_household": labels,
        f"{key}_household_count": count,
        f"{key}_household_weight": w_sum,
        f"{key}_household_candidate_U_num": np.bincount(inverse, weights=candidate, minlength=len(labels)),
        f"{key}_household_H_U_num": np.bincount(inverse, weights=h_loss, minlength=len(labels)),
        f"{key}_household_candidate_W_num": np.bincount(inverse, weights=weights*candidate, minlength=len(labels)),
        f"{key}_household_H_W_num": np.bincount(inverse, weights=weights*h_loss, minlength=len(labels)),
    }


def _inventory(root: Path) -> dict[str, str]:
    artifacts = {}
    for file in sorted(root.rglob("*")):
        if file.is_symlink():
            raise ValueError("symlink in private audit output")
        if file.is_file() and file.name != "COMPLETE.json":
            artifacts[str(file.relative_to(root))] = _sha(file)
    return artifacts


def _seal_private_permissions(root: Path) -> None:
    """Keep fitted models and original-person contributions owner-only."""
    for path in [root, *sorted(root.rglob("*"))]:
        if path.is_symlink():
            raise ValueError("symlink in private audit output")
        path.chmod(0o700 if path.is_dir() else 0o600)


def audit_panel(anchor: int, releases: Mapping[str, Mapping[str, Any]],
                index_path: str | Path, output_dir: str | Path, *,
                resume: bool = False, slate: str = "catchup") -> dict:
    """Fit equal-slate routes, select on inner_selection, score inner_check.

    Only a coordinator-dispatched job should invoke this on ACS data. A
    completed receipt is immutable; `resume=True` permits verified per-slate
    recovery after a technical interruption, never a fresh scientific choice.
    """
    if anchor not in (0, 1, 2) or slate not in ("standard", "catchup"):
        raise ValueError("undeclared anchor or audit slate")
    if not isinstance(releases, Mapping) or not releases:
        raise ValueError("at least one named release required")
    root = Path(output_dir)
    if "private" not in root.parts:
        raise ValueError("fitted models and private contributions need private output")
    if (root / "COMPLETE.json").exists():
        raise FileExistsError("completed inner panel is immutable")
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("partial inner panel retained; use explicit technical resume")
    if (root / "INNER_AUDIT.json").exists():
        raise FileExistsError("incomplete final audit report retained for incident review")
    index_file = Path(index_path).resolve()
    index_value = data.index(index_file)
    # The inherited loader has a raw-object fallback. Require the previously
    # verified label-stripped receipt before any prepared object deserialization.
    if data._sanitized_record(index_value, anchor) is None:
        raise FileNotFoundError("verified sanitized 2018 prepared receipt required")
    prepared = data.load_prepared(index_value, anchor)
    pools = {name: roles.pooled_role(prepared, name) for name in POOL_ROLES}
    audit.assert_household_disjoint(*(pools[name]["households"] for name in POOL_ROLES))
    descriptors = {}
    laws = {}
    for release_id, spec in releases.items():
        _slug(release_id)
        if release_id == "H":
            raise ValueError("H-only is a shared ancestor, not a candidate release")
        if "task_only_model_dir" in spec:
            from . import task_baselines

            receipt_path = _pin(Path(spec["task_only_model_dir"]) / "TASK_ONLY.json",
                                spec["task_only_receipt_sha256"])
            _, model_receipt = task_baselines.load_task_only(receipt_path.parent)
            descriptors[release_id] = {
                "law_kind": "task_only_original_person_17_token",
                "task_only_receipt_artifact": _artifact_record(receipt_path, index_file.parent),
                "task_only_model_artifact": _artifact_record(receipt_path.parent / "task_only.joblib", index_file.parent),
                "task_only_model_sha256": model_receipt["model_sha256"],
                "mode": spec["mode"], "publish": spec["publish"],
                "constant_token": 0 if spec["mode"] == "constant_replacement" else None,
                "recipient_observes_person_law": False,
            }
        else:
            q, source = _channel_artifact(spec)
            descriptors[release_id] = {
                "channel_artifact": _artifact_record(source, index_file.parent),
                "channel_array_sha256": _array_sha(q),
                "router_kind": "T0" if spec["router"] == "T0" else "callable",
            }
            if callable(spec["router"]):
                route_source = _pin(spec["router_artifact_path"], spec["router_sha256"])
                descriptors[release_id]["router_artifact"] = _artifact_record(route_source, index_file.parent)
                descriptors[release_id]["router_entrypoint"] = spec["router_entrypoint"]
            else:
                descriptors[release_id]["router_source"] = "stored Linux x86 T0 codes in pinned prepared object"
        laws[release_id] = {name: token_law_for_release(spec, pools[name])
                            for name in POOL_ROLES}
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    h_laws = {name: np.ones((len(pools[name]["ids"]), 1), dtype=np.float64)
              for name in POOL_ROLES}
    h_by_role = {}
    for role_index, role in enumerate(H_ROLES):
        h_by_role[role] = audit.fit_role_slate(
            pools["audit_fit"], h_laws["audit_fit"],
            pools["inner_selection"], h_laws["inner_selection"],
            role, root / "H" / role.replace(":", "_").replace("/", "_"),
            26000 + 1000*anchor + role_index, release_id="H", slate=slate)
    h_scores = {}
    h_locks = {}
    for role in audit.ROLES:
        routes = _registry_routes(role, "H", h_by_role[role], h_by_role[role],
                                  h_by_role, h_by_role)
        lock = audit.select_frozen_routes(
            pools["inner_selection"], h_laws["inner_selection"],
            role, routes, release_id="H")
        h_locks[role] = lock
        h_scores[role] = audit.score_frozen_route(
            pools["inner_check"], h_laws["inner_check"], lock)
    report_releases = {}
    contributions = {}
    for release_id in sorted(releases):
        own_by_role = {}
        for role_index, role in enumerate(audit.ROLES):
            own_by_role[role] = audit.fit_role_slate(
                pools["audit_fit"], laws[release_id]["audit_fit"],
                pools["inner_selection"], laws[release_id]["inner_selection"],
                role, root / release_id / role.replace(":", "_").replace("/", "_"),
                26000 + 1000*anchor + role_index,
                release_id=release_id, slate=slate)
        role_results = {}
        prefix = hashlib.sha256(release_id.encode()).hexdigest()[:12]
        for role in audit.ROLES:
            routes = _registry_routes(role, release_id, own_by_role[role],
                                      h_by_role[role], own_by_role, h_by_role)
            lock = audit.select_frozen_routes(
                pools["inner_selection"], laws[release_id]["inner_selection"],
                role, routes, release_id=release_id)
            score = audit.score_frozen_route(
                pools["inner_check"], laws[release_id]["inner_check"], lock)
            h_score = h_scores[role]
            contributions.update(_private_contributions(prefix, role, score, h_score))
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
                "fit_missing_classes": own_by_role[role]["fit_missing_classes"],
                "selection_missing_classes": own_by_role[role]["validation_missing_classes"],
            }
        report_releases[release_id] = {"roles": role_results,
                                       "source": descriptors[release_id],
                                       "private_contribution_prefix": prefix}
    contributions_path = root / "PANEL_CONTRIBUTIONS.npz"
    if contributions_path.exists():
        raise FileExistsError("existing private contributions retained")
    np.savez_compressed(contributions_path, **contributions)
    source_code_sha256 = {"evaluate.py": _sha(__file__),
                          "audit.py": _sha(audit.__file__),
                          "roles.py": _sha(roles.__file__),
                          "inherited_audit.py": _sha(inherited_audit.__file__),
                          "data.py": _sha(data.__file__)}
    if any("task_only_model_dir" in spec for spec in releases.values()):
        from . import task_baselines
        source_code_sha256["task_baselines.py"] = _sha(task_baselines.__file__)
    report = {
        "schema": "pcrl-adaptive-inner-audit-v1", "anchor": anchor,
        "not_confirmation": True, "outer_pool_opened": False,
        "fit_role": "audit_fit", "selection_role": "inner_selection",
        "score_role": "inner_check", "slate": slate,
        "probability_floor": audit.FLOOR,
        "index_sha256": _sha(index_file),
        "source_code_sha256": source_code_sha256,
        "releases": report_releases,
        "contributions_relative_path": contributions_path.name,
        "contributions_sha256": _sha(contributions_path),
    }
    _json_atomic(root / "INNER_AUDIT.json", report)
    _seal_private_permissions(root)
    receipt = {
        "schema": "pcrl-adaptive-inner-audit-complete-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "anchor": anchor, "release_ids": sorted(releases),
        "index_sha256": report["index_sha256"],
        "artifacts": _inventory(root),
    }
    _json_atomic(root / "COMPLETE.json", receipt)
    (root / "COMPLETE.json").chmod(0o600)
    return report
