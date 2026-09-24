"""Nondeployable, label-derived positive controls for inner audit power.

This module is deliberately separate from release artifacts and APIs. Its
token law reads protected labels and must never be used as a candidate or
control release, as a deployable mapping, or on outer assessment rows.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np


N_TOKENS = 17
CLASS_COUNT = {"SEX": 2, "RAC1P": 9}


def revealing_token_law(labels: np.ndarray, *, target: str) -> np.ndarray:
    """One-hot token equal to S, solely for an inner-only audit-power check."""
    if target not in CLASS_COUNT:
        raise ValueError("only the registered protected targets are diagnostic")
    s = np.asarray(labels)
    if (s.ndim != 1 or s.size == 0 or not np.issubdtype(s.dtype, np.integer)
            or np.any(s < 0) or np.any(s >= CLASS_COUNT[target])):
        raise ValueError("diagnostic labels outside the full class schema")
    law = np.zeros((len(s), N_TOKENS), dtype=np.float64)
    law[np.arange(len(s)), s] = 1.0
    return law


def xor_fixture(view: str) -> dict[str, np.ndarray]:
    """Exact local or coalition interaction whose one-variable marginals hide S."""
    if view not in ("A", "AB"):
        raise ValueError("XOR fixture must use A or AB released view")
    h = np.array([0, 0, 1, 1], dtype=np.int64)
    token = np.array([0, 1, 0, 1], dtype=np.int64)
    sensitive = h ^ token
    ha = np.zeros((4, 4), dtype=np.float64)
    hb = np.zeros((4, 2), dtype=np.float64)
    if view == "A":
        ha[:, 0] = h
    else:
        hb[:, 0] = h
    law = np.zeros((4, N_TOKENS), dtype=np.float64)
    law[np.arange(4), token] = 1.0
    full = np.full((4, N_TOKENS, 2), .5, dtype=np.float64)
    for i in range(4):
        for z in (0, 1):
            p = h[i] ^ z
            full[i, z, p] = .99
            full[i, z, 1-p] = .01
    return {"view": view, "ha": ha, "hb": hb, "h_bit": h,
            "token": token, "sensitive": sensitive, "law": law,
            "full_prediction": full,
            "marginal_prediction": np.full_like(full, .5)}


def _digest_rows(rows: Mapping, target: str) -> str:
    digest = hashlib.sha256()
    for key in ("ha", "hb", "weights", "ids", "households"):
        value = np.asarray(rows[key])
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        if value.dtype.kind in "OUS":
            for entry in value.ravel():
                raw = str(entry).encode()
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(target.encode())
    digest.update(np.ascontiguousarray(rows["labels"][target]).tobytes())
    return digest.hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def run_inner_diagnostic(inner_roles: Mapping[str, Mapping], *, role: str,
                         anchor: int, output_dir: str | Path,
                         slate: str = "standard") -> dict:
    """Fit and check one label-oracle audit diagnostic on inner 2018 roles only.

    The caller supplies already admitted role dictionaries. This routine has
    no ACS loader or outer-assessment path, writes fitted models privately,
    and returns aggregate-only losses. The label-derived law is NEVER a
    deployable or competitive release, even if an auditor detects it.
    """
    from . import audit, roles

    allowed = {"audit_fit", "inner_selection", "inner_check"}
    if not isinstance(inner_roles, Mapping) or set(inner_roles) != allowed:
        raise ValueError("only the three inner audit roles are accepted")
    if anchor not in (0, 1, 2) or slate not in ("standard", "catchup"):
        raise ValueError("undeclared anchor or audit slate")
    kind, view, target = audit.parse_role(role)
    if kind != "attack" or view not in ("A", "AB") or target not in CLASS_COUNT:
        raise ValueError("diagnostic requires a registered A or AB protected role")
    root = Path(output_dir)
    if "private" not in root.parts:
        raise ValueError("diagnostic models and labels require a private output path")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("existing diagnostic output is immutable or partial")
    households = []
    for name in ("audit_fit", "inner_selection", "inner_check"):
        rows = inner_roles[name]
        houses = np.asarray(rows["households"])
        if houses.ndim != 1 or not len(houses) or any(roles.role_of(h) != name for h in houses):
            raise ValueError("household does not belong to its declared inner role")
        households.append(houses)
        labels = np.asarray(rows["labels"][target])
        if labels.shape != houses.shape:
            raise ValueError("protected labels and households differ in length")
        revealing_token_law(labels, target=target)  # full-schema admission before fitting
    audit.assert_household_disjoint(*households)
    leak_laws = {name: revealing_token_law(inner_roles[name]["labels"][target],
                                         target=target)
                 for name in allowed}
    h_laws = {name: np.ones((len(inner_roles[name]["households"]), 1), dtype=float)
              for name in allowed}
    for name in allowed:
        audit.role_arrays(inner_roles[name], leak_laws[name], role)
        audit.role_arrays(inner_roles[name], h_laws[name], role)
    root.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)

    release_id = "POSITIVE_CONTROL_LABEL_ORACLE_NONDEPLOYABLE"
    needed_h = [role]
    needed_release = [role]
    if view == "AB":
        needed_h += [f"attack:A/{target}", f"attack:B/{target}"]
        needed_release += [f"attack:A/{target}"]
    registries = {}
    for wire, needed, laws, identity in (("H", needed_h, h_laws, "H"),
                                         ("release", needed_release, leak_laws, release_id)):
        for offset, scored_role in enumerate(needed):
            directory = root / "models" / wire / scored_role.replace(":", "_").replace("/", "_")
            registries[(wire, scored_role)] = audit.fit_role_slate(
                inner_roles["audit_fit"], laws["audit_fit"],
                inner_roles["inner_selection"], laws["inner_selection"],
                scored_role, directory, 27000 + 1000*anchor + offset,
                release_id=identity, slate=slate)

    def routes(wire: str) -> dict:
        identity = "H" if wire == "H" else release_id
        own = registries[(wire, role)]
        h_own = registries[("H", role)]
        kwargs = {}
        if view == "AB":
            kwargs = {"a_same": registries[(wire, f"attack:A/{target}")],
                      "b_h_only": registries[("H", f"attack:B/{target}")]}
        return audit.candidate_route_bank(role, identity, own, h_own, **kwargs)

    selection = {}
    check = {}
    for wire, laws, identity in (("H", h_laws, "H"),
                                  ("release", leak_laws, release_id)):
        selection[wire] = audit.select_frozen_routes(
            inner_roles["inner_selection"], laws["inner_selection"],
            role, routes(wire), release_id=identity)
        check[wire] = audit.score_frozen_route(
            inner_roles["inner_check"], laws["inner_check"], selection[wire])
    h_score, leak_score = check["H"], check["release"]
    if (not np.array_equal(h_score["ids"], leak_score["ids"]) or
            not np.array_equal(h_score["households"], leak_score["households"]) or
            not np.array_equal(h_score["weights"], leak_score["weights"])):
        raise ValueError("H and positive-control check rows differ")
    improvement = {weighting: float(h_score["scores"][weighting]
                                    - leak_score["scores"][weighting])
                   for weighting in ("U", "PWGTP")}
    threshold = .01  # descriptive audit-power threshold, not a claim margin
    result = {"schema": "pcrl-adaptive-inner-positive-control-v1",
              "anchor": anchor, "role": role, "slate": slate,
              "nondeployable_label_oracle": True,
              "outer_labels_accessed": False,
              "fit_role": "audit_fit", "selection_role": "inner_selection",
              "score_role": "inner_check",
              "original_people_scored": int(len(leak_score["ids"])),
              "households_scored": int(len(set(map(str, leak_score["households"])))),
              "role_input_sha256": {name: _digest_rows(inner_roles[name], target)
                                    for name in sorted(allowed)},
              "source_sha256": _sha_file(Path(__file__)),
              "selected_routes": {wire: selection[wire]["selected"] for wire in selection},
              "scores": {wire: check[wire]["scores"] for wire in check},
              "improvement_nats": improvement,
              "descriptive_detection_threshold_nats": threshold,
              "detected_both_weightings": all(x >= threshold for x in improvement.values())}
    _write_json(root / "POSITIVE_CONTROL.json", result)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in private positive-control output")
        path.chmod(0o700 if path.is_dir() else 0o600)
    inventory = {str(path.relative_to(root)): _sha_file(path)
                 for path in sorted(root.rglob("*")) if path.is_file()}
    _write_json(root / "COMPLETE.json", {"schema": 1,
        "status": "COMPLETE_NONDEPLOYABLE_DIAGNOSTIC", "artifact_sha256": inventory})
    return result
