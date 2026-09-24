"""Independent inner audit of per-person shared-context releases on one anchor.

This is the predecessor slate (`AR/evaluate.audit_panel`) with only the release
plumbing replaced:

* laws come from `laws.person_law` (per-person q(z|x) on legal inputs), not a
  32x17 matrix routed by T0;
* de-duplication hashes each release's per-person law on the admitted rows of
  every audit role (audit_fit, inner_selection, inner_check) plus the identity
  of those rows. Releases with identical laws share one audit; every declared
  name is kept in an alias ledger (`ALIAS_LEDGER.json`);
* the inherited hist_gb min-leaf rule (TDR/audits.py:366-368) is recorded per
  release and role; the rule itself is unchanged and identical for every
  candidate (see `HIST_GB_TREATMENT`).

Everything else is imported unchanged: the standard slate, roles, H-only and
legal ancestor routes (`AR/evaluate._registry_routes`), validation selection
on inner_selection, scoring on inner_check, private contributions and the
receipt format. Seeds are common across releases (common random numbers).

Outer access: inner mode loads only `roles.pooled_role` (which refuses outer).
`score_outer` exists, but it runs only after `verify_outer_gate` accepts the
coordinator's selection lock and remote-verified unlock receipt (AR
`outer_access` pattern) and then reuses the frozen inner routes without any
refit or reselection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import audit as inherited_audit, data
from experiments.pcrl_adaptive_release_v1 import audit as ar_audit, evaluate, roles
from . import laws

ROOT = Path(__file__).resolve().parents[2]
STUDY = "pcrl_shared_context_release_v1"
RESULTS = ROOT / "results" / STUDY
LOCK_PATH = RESULTS / "SELECTION_LOCK.json"
UNLOCK_PATH = RESULTS / "private" / "OUTER_UNLOCK.json"
ORIGINAL_RESTORE_ROOT = RESULTS / "private" / "original_2018_restore"
CENSUS_PATH = ROOT / "results/pcrl_adaptive_release_v1/DATA_ROLE_COUNTS.json"
BRANCH = "research/pcrl-shared-context-release-v1"
LOCK_SCHEMA = "pcrl-sc-selection-lock-v1"
UNLOCK_SCHEMA = "pcrl-sc-outer-unlock-v1"
POOL_ROLES = evaluate.POOL_ROLES  # audit_fit, inner_selection, inner_check
H_ROLES = evaluate.H_ROLES
# seed = SEED_BASE + 1000*anchor + role index. Deliberately the AR value: the
# J continuity panel runs AR/external_audit.py unchanged, which hard-codes
# 26000, so the same-host H-only slates of the main and J panels are fitted
# with identical rows, seeds and code (one H reference across panels).
SEED_BASE = 26000
REPORT_SCHEMA = "pcrl-sc-inner-audit-v1"
COMPLETE_SCHEMA = "pcrl-sc-inner-audit-complete-v1"
LEAVES = (20, 5)
HIST_GB_TREATMENT = (
    "Inherited TDR rule, one code path for every candidate and for H: exact "
    "hist_gb_{20,5} set min_samples_leaf = leaf x (max per-person token support "
    "on that release's fit rows); sampled_hist_gb_{20,5} fit one fixed draw per "
    "person (seed 20260921+seed) with min_samples_leaf = leaf. All six families "
    "enter every candidate's validation selection. Stochastic per-person laws "
    "therefore get a coarser exact-tree leaf (recorded below); the sampled trees "
    "keep the nominal leaf. No mass floor or law truncation is applied.")


def _sha(path: str | Path) -> str:
    return laws.sha256_file(path)


def _role_dir(role: str) -> str:
    return role.replace(":", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# per-person law identity and aliases
# ---------------------------------------------------------------------------

def canonicalize(sources: Mapping[str, Any], pools: Mapping[str, Mapping]) -> tuple[dict, dict, dict]:
    """Evaluate each declared release on the admitted rows of every audit role.

    Returns (canonical laws {id: {role: law}}, alias ledger, descriptors). The
    first declared name of each distinct per-person law is canonical.
    """
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("at least one named release required")
    inputs = {role: laws.legal_inputs(pools[role]) for role in POOL_ROLES}
    row_ids = {role: laws.rows_identity(pools[role]["ids"]) for role in POOL_ROLES}
    canonical: dict[str, dict] = {}
    descriptors: dict[str, dict] = {}
    by_identity: dict[str, str] = {}
    entries = {}
    for release_id, source in sources.items():
        evaluate._slug(release_id)
        if release_id == "H":
            raise ValueError("H-only is the shared ancestor, not a candidate release")
        pinned = laws.load(source)
        if pinned.kind == "h_only":
            raise ValueError("H_ONLY is reported from the shared H ancestor; do not audit it as a token release")
        values = {role: pinned(inputs[role]) for role in POOL_ROLES}
        per_role = {role: laws.law_identity(values[role]) for role in POOL_ROLES}
        digest = hashlib.sha256()
        for role in POOL_ROLES:
            digest.update(f"{role}|{row_ids[role]}|{per_role[role]}\n".encode())
        identity = digest.hexdigest()
        first = by_identity.setdefault(identity, release_id)
        entry = {"canonical": first, "law_identity": identity,
                 "law_sha256_by_role": per_role, "kind": pinned.kind,
                 "source": _public_descriptor(pinned.descriptor)}
        if first == release_id:
            canonical[release_id] = values
            descriptors[release_id] = entry["source"]
        else:
            # Diagnostic only: aliases are exact; near-identity is never merged.
            entry["max_abs_difference_to_canonical"] = max(
                float(np.max(np.abs(values[r] - canonical[first][r]))) for r in POOL_ROLES)
        entries[release_id] = entry
    for release_id, entry in entries.items():
        if entry["canonical"] != release_id:
            continue
        nearest = [(max(float(np.max(np.abs(canonical[release_id][r] - canonical[other][r])))
                        for r in POOL_ROLES), other)
                   for other in canonical if other != release_id
                   and all(canonical[other][r].shape == canonical[release_id][r].shape
                           for r in POOL_ROLES)]
        if nearest:
            gap, other = min(nearest)
            entry["nearest_distinct_canonical"] = {"release_id": other,
                                                   "max_abs_difference": gap}
    ledger = {"schema": "pcrl-sc-alias-ledger-v1",
              "identity_rule": ("sha256 over (role, sha256 of ordered original-person ids, "
                                "sha256 of float64 law bytes+shape) for audit_fit, "
                                "inner_selection, inner_check; exact equality only"),
              "row_identity_sha256": row_ids,
              "declared": entries,
              "canonical_ids": list(canonical),
              "declared_name_count": len(entries),
              "canonical_release_count": len(canonical)}
    return canonical, ledger, descriptors


def _public_descriptor(descriptor: Mapping) -> dict:
    keep = ("kind", "release_id", "anchor", "map_name", "map_file_sha256",
            "map_array_sha256", "descriptor_sha256", "pins", "wire")
    return {key: descriptor[key] for key in keep if key in descriptor}


def hist_gb_record(pools: Mapping[str, Mapping], law_by_role: Mapping[str, np.ndarray]) -> dict:
    """Effective exact-tree leaf per audit role on the fit rows (audit_fit)."""
    out = {}
    for role in ar_audit.ROLES:
        fit = inherited_audit.role_arrays(pools["audit_fit"], law_by_role["audit_fit"], role)
        support = laws.support_record(fit["token_probs"])
        out[role] = {**support,
                     "effective_min_samples_leaf": {f"hist_gb_{leaf}": leaf * support["max_support"]
                                                    for leaf in LEAVES},
                     "sampled_min_samples_leaf": {f"sampled_hist_gb_{leaf}": leaf for leaf in LEAVES},
                     "coarsened": support["max_support"] > 1}
    return out


def _observed_leaf(slate_dir: Path) -> dict | None:
    path = slate_dir / "models" / "slate.json"
    if not path.is_file():
        return None
    meta = json.loads(path.read_text())
    return {f"hist_gb_{leaf}": meta["candidates"][f"hist_gb_{leaf}"]["parameters"].get("min_samples_leaf")
            for leaf in LEAVES if f"hist_gb_{leaf}" in meta.get("candidates", {})}


# ---------------------------------------------------------------------------
# inner panel
# ---------------------------------------------------------------------------

def load_inner_pools(index_path: str | Path, anchor: int) -> tuple[dict, str]:
    """Admit only the three inner audit roles from the sanitized prepared object."""
    index_file = Path(index_path).resolve()
    value = data.index(index_file)
    if data._sanitized_record(value, anchor) is None:
        raise FileNotFoundError("verified sanitized 2018 prepared receipt required")
    prepared = data.load_prepared(value, anchor)
    if "labels" in prepared["ctx"]["pools"]["attacker_validation"]:
        raise PermissionError("outer labels present in inner-mode prepared object")
    pools = {name: roles.pooled_role(prepared, name) for name in POOL_ROLES}
    del prepared
    return pools, _sha(index_file)


def run_panel(anchor: int, pools: Mapping[str, Mapping], sources: Mapping[str, Any],
              output_dir: str | Path, *, index_sha256: str, slate: str = "standard",
              resume: bool = False) -> dict:
    """Fit/select/score with already admitted inner pools (inner mode core)."""
    if anchor not in (0, 1, 2) or slate not in ("standard", "catchup"):
        raise ValueError("undeclared anchor or audit slate")
    if set(pools) != set(POOL_ROLES):
        raise PermissionError("inner audit accepts exactly audit_fit, inner_selection, inner_check")
    root = Path(output_dir)
    if "private" not in root.parts:
        raise ValueError("fitted models and private contributions need a private output path")
    if (root / "COMPLETE.json").exists():
        raise FileExistsError("completed inner panel is immutable")
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError("partial inner panel retained; use explicit technical resume")
    if (root / "INNER_AUDIT.json").exists():
        raise FileExistsError("incomplete final audit report retained for incident review")
    for name in POOL_ROLES:
        houses = np.asarray(pools[name]["households"])
        if any(roles.role_of(h) != name for h in houses):
            raise PermissionError(f"household outside declared role {name}")
    ar_audit.assert_household_disjoint(*(pools[name]["households"] for name in POOL_ROLES))
    canonical, ledger, descriptors = canonicalize(sources, pools)
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    evaluate._json_atomic(root / "ALIAS_LEDGER.json", ledger)
    h_laws = {name: np.ones((len(pools[name]["ids"]), 1), dtype=np.float64)
              for name in POOL_ROLES}
    h_by_role = {}
    for role_index, role in enumerate(H_ROLES):
        h_by_role[role] = ar_audit.fit_role_slate(
            pools["audit_fit"], h_laws["audit_fit"],
            pools["inner_selection"], h_laws["inner_selection"],
            role, root / "H" / _role_dir(role),
            SEED_BASE + 1000*anchor + role_index, release_id="H", slate=slate)
    h_scores, h_locks = {}, {}
    for role in ar_audit.ROLES:
        routes = evaluate._registry_routes(role, "H", h_by_role[role], h_by_role[role],
                                           h_by_role, h_by_role)
        h_locks[role] = ar_audit.select_frozen_routes(
            pools["inner_selection"], h_laws["inner_selection"], role, routes, release_id="H")
        h_scores[role] = ar_audit.score_frozen_route(
            pools["inner_check"], h_laws["inner_check"], h_locks[role])
    report_releases, contributions, leaf_records = {}, {}, {}
    for release_id in canonical:
        law = canonical[release_id]
        leaf_records[release_id] = hist_gb_record(pools, law)
        own_by_role = {}
        for role_index, role in enumerate(ar_audit.ROLES):
            directory = root / release_id / _role_dir(role)
            own_by_role[role] = ar_audit.fit_role_slate(
                pools["audit_fit"], law["audit_fit"],
                pools["inner_selection"], law["inner_selection"],
                role, directory, SEED_BASE + 1000*anchor + role_index,
                release_id=release_id, slate=slate)
            observed = _observed_leaf(directory)
            if observed is not None:
                expected = leaf_records[release_id][role]["effective_min_samples_leaf"]
                if observed != expected:
                    raise AssertionError(f"hist_gb leaf rule differs for {release_id}/{role}")
                leaf_records[release_id][role]["observed_min_samples_leaf"] = observed
        prefix = hashlib.sha256(release_id.encode()).hexdigest()[:12]
        role_results = {}
        for role in ar_audit.ROLES:
            routes = evaluate._registry_routes(role, release_id, own_by_role[role],
                                               h_by_role[role], own_by_role, h_by_role)
            lock = ar_audit.select_frozen_routes(
                pools["inner_selection"], law["inner_selection"], role, routes,
                release_id=release_id)
            score = ar_audit.score_frozen_route(pools["inner_check"], law["inner_check"], lock)
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
                "fit_missing_classes": own_by_role[role]["fit_missing_classes"],
                "selection_missing_classes": own_by_role[role]["validation_missing_classes"],
            }
        report_releases[release_id] = {
            "roles": role_results, "source": descriptors[release_id],
            "aliases": sorted(name for name, entry in ledger["declared"].items()
                              if entry["canonical"] == release_id),
            "law_identity": ledger["declared"][release_id]["law_identity"],
            "private_contribution_prefix": prefix}
    contributions_path = root / "PANEL_CONTRIBUTIONS.npz"
    if contributions_path.exists():
        raise FileExistsError("existing private contributions retained")
    np.savez_compressed(contributions_path, **contributions)
    here = Path(__file__).resolve().parent
    report = {
        "schema": REPORT_SCHEMA, "study": STUDY, "anchor": anchor,
        "not_confirmation": True, "outer_pool_opened": False, "mode": "inner",
        "fit_role": "audit_fit", "selection_role": "inner_selection",
        "score_role": "inner_check", "slate": slate, "seed_base": SEED_BASE,
        "probability_floor": ar_audit.FLOOR, "index_sha256": index_sha256,
        "source_code_sha256": {
            "audit_panel.py": _sha(__file__), "laws.py": _sha(here / "laws.py"),
            "AR/evaluate.py": _sha(evaluate.__file__), "AR/audit.py": _sha(ar_audit.__file__),
            "AR/roles.py": _sha(roles.__file__),
            "TAC/audit.py": _sha(inherited_audit.__file__), "TAC/data.py": _sha(data.__file__)},
        "hist_gb_treatment": HIST_GB_TREATMENT,
        "hist_gb_min_leaf": leaf_records,
        "alias_ledger_sha256": _sha(root / "ALIAS_LEDGER.json"),
        "logical_to_canonical": {name: entry["canonical"]
                                 for name, entry in ledger["declared"].items()},
        "releases": report_releases,
        "contributions_relative_path": contributions_path.name,
        "contributions_sha256": _sha(contributions_path),
    }
    evaluate._json_atomic(root / "INNER_AUDIT.json", report)
    evaluate._seal_private_permissions(root)
    receipt = {"schema": COMPLETE_SCHEMA,
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "anchor": anchor, "release_ids": sorted(canonical),
               "declared_ids": sorted(ledger["declared"]),
               "index_sha256": index_sha256,
               "artifacts": evaluate._inventory(root)}
    evaluate._json_atomic(root / "COMPLETE.json", receipt)
    (root / "COMPLETE.json").chmod(0o600)
    return report


def audit_panel(anchor: int, sources: Mapping[str, Any], index_path: str | Path,
                output_dir: str | Path, *, slate: str = "standard",
                resume: bool = False) -> dict:
    """Inner mode on the sanitized 2018 object: never touches outer rows."""
    pools, index_sha = load_inner_pools(index_path, anchor)
    return run_panel(anchor, pools, sources, output_dir, index_sha256=index_sha,
                     slate=slate, resume=resume)


def verify_complete(output_dir: str | Path) -> dict:
    """Re-verify a completed panel's receipt and full artifact inventory."""
    root = Path(output_dir)
    receipt = json.loads((root / "COMPLETE.json").read_text())
    report = json.loads((root / "INNER_AUDIT.json").read_text())
    if (receipt.get("schema") != COMPLETE_SCHEMA or report.get("schema") != REPORT_SCHEMA or
            receipt.get("artifacts") != evaluate._inventory(root) or
            report.get("outer_pool_opened") is not False or
            sorted(report.get("releases", {})) != receipt.get("release_ids")):
        raise ValueError("completed inner panel identity or artifact inventory differs")
    return report


# ---------------------------------------------------------------------------
# outer mode (coordinator lock required)
# ---------------------------------------------------------------------------

def _hex(value: object, size: int) -> bool:
    return (isinstance(value, str) and len(value) == size and
            all(char in "0123456789abcdef" for char in value))


LOCK_RELATIVE = f"results/{STUDY}/SELECTION_LOCK.json"


def _git(*args: str, repo: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True,
                                   stderr=subprocess.PIPE)


def remote_lock_check(commit: str, expected_lock_sha256: str, *, repo: Path = ROOT) -> dict:
    """Amendment M5.5: the lock commit is on origin/<branch> and holds these lock bytes."""
    if not _hex(commit, 40) or not _hex(expected_lock_sha256, 64):
        raise PermissionError("40-hex commit and 64-hex lock SHA-256 required")
    listing = _git("ls-remote", "origin", f"refs/heads/{BRANCH}", repo=repo).split()
    if len(listing) < 1 or not _hex(listing[0], 40):
        raise PermissionError(f"origin/{BRANCH} is not visible")
    tip = listing[0]
    _git("fetch", "-q", "origin", f"refs/heads/{BRANCH}", repo=repo)
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, tip], cwd=repo,
                      capture_output=True).returncode:
        raise PermissionError("lock commit is not on the pushed study branch")
    blob = subprocess.check_output(["git", "show", f"{commit}:{LOCK_RELATIVE}"], cwd=repo)
    if hashlib.sha256(blob).hexdigest() != expected_lock_sha256:
        raise PermissionError("SELECTION_LOCK.json at the pushed commit differs from the expected SHA-256")
    return {"branch": BRANCH, "remote_tip": tip, "remote_commit_sha": commit,
            "lock_relative_path": LOCK_RELATIVE, "lock_sha256_at_commit": expected_lock_sha256}


def write_unlock(expected_lock_sha256: str, commit: str, *, unlock_path: Path | None = None) -> dict:
    """Coordinator command: verify on origin, then write the unlock receipt once."""
    target = unlock_path or UNLOCK_PATH
    if target.exists():
        raise FileExistsError("outer unlock receipt is write-once")
    if _sha(LOCK_PATH) != expected_lock_sha256:
        raise PermissionError("local SELECTION_LOCK.json differs from the expected SHA-256")
    evidence = remote_lock_check(commit, expected_lock_sha256)
    receipt = {"schema": UNLOCK_SCHEMA, "lock_sha256": expected_lock_sha256,
               "remote_verified": True, "branch": BRANCH, "remote_commit_sha": commit,
               "verified_utc": datetime.now(timezone.utc).isoformat(), "evidence": evidence}
    target.parent.mkdir(parents=True, exist_ok=True)
    evaluate._json_atomic(target, receipt)
    return receipt


def verify_outer_gate(selection_lock_path: str | Path, expected_lock_sha256: str, *,
                      lock_path: Path | None = None, unlock_path: Path | None = None) -> dict:
    """AR outer_access pattern with this study's constants; raise unless unlocked.

    The receipt's `remote_verified` flag is not trusted: the commit is re-checked
    on origin (`remote_lock_check`) before any outer row can be opened.
    """
    registered = (lock_path or LOCK_PATH).resolve()
    unlock_file = unlock_path or UNLOCK_PATH
    path = Path(selection_lock_path).resolve()
    if path != registered or not _hex(expected_lock_sha256, 64):
        raise PermissionError("registered selection lock path and SHA-256 required")
    if not path.is_file() or _sha(path) != expected_lock_sha256:
        raise PermissionError("selection lock is absent or differs from expected SHA-256")
    lock = json.loads(path.read_text())
    if (lock.get("schema") != LOCK_SCHEMA or lock.get("status") != "LOCKED" or
            lock.get("study") != STUDY or lock.get("outer_assessment_authorized") is not True or
            not isinstance(lock.get("anchors"), dict)):
        raise PermissionError("outer development panel was not frozen by the coordinator")
    if not unlock_file.is_file():
        raise PermissionError("remote-verified outer unlock receipt is absent")
    unlock = json.loads(unlock_file.read_text())
    if (unlock.get("schema") != UNLOCK_SCHEMA or
            unlock.get("lock_sha256") != expected_lock_sha256 or
            unlock.get("remote_verified") is not True or unlock.get("branch") != BRANCH or
            not _hex(unlock.get("remote_commit_sha"), 40)):
        raise PermissionError("outer unlock was not remotely verified")
    remote_lock_check(unlock["remote_commit_sha"], expected_lock_sha256)
    return lock


def _score_release(rows: Mapping, law: np.ndarray, route_locks: tuple, prefix: str,
                   contributions: dict) -> dict:
    own_locks, h_locks = route_locks
    h_law = np.ones((len(rows["ids"]), 1), dtype=np.float64)
    role_results = {}
    for role in ar_audit.ROLES:
        score = ar_audit.score_frozen_route(rows, law, own_locks[role])
        h_score = ar_audit.score_frozen_route(rows, h_law, h_locks[role])
        contributions.update(evaluate._private_contributions(prefix, role, score, h_score))
        role_results[role] = {"candidate": score["scores"], "H": h_score["scores"],
                              "n_people": len(score["loss"]),
                              "selected_candidate": own_locks[role]["selected"],
                              "H_selected_candidate": h_locks[role]["selected"]}
    return {"roles": role_results, "private_contribution_prefix": prefix}


def score_outer(anchor: int, sources: Mapping[str, Any], index_path: str | Path,
                inner_panel_dir: str | Path, selection_lock_path: str | Path,
                expected_lock_sha256: str, output_dir: str | Path, *, _gate=None,
                _load_outer_rows=None) -> dict:
    """Score frozen inner routes on outer rows; no refit, no reselection.

    `sources` must declare every logical release of the inner panel (the same
    source file). Amendment M5.3: each logical law is evaluated on the outer rows
    and must be byte-identical to its canonical law; an alias that breaks on
    outer rows is scored separately (with its canonical's frozen routes, since it
    shared them on inner rows) and recorded under `outer_alias_breaks`.
    """
    from experiments.pcrl_adaptive_release_v1 import outer_audit, outer_pool

    root = Path(output_dir).resolve()
    if "private" not in root.parts or root.exists():
        raise FileExistsError("outer output must be a new private directory")
    lock = (_gate or verify_outer_gate)(selection_lock_path, expected_lock_sha256)
    pinned = lock["anchors"].get(str(anchor), {})
    inner_root = Path(inner_panel_dir)
    inner = verify_complete(inner_root)
    if (pinned.get("inner_panel_complete_sha256") != _sha(inner_root / "COMPLETE.json") or
            pinned.get("inner_audit_sha256") != _sha(inner_root / "INNER_AUDIT.json")):
        raise PermissionError("inner panel differs from the locked anchor record")
    ledger = json.loads((inner_root / "ALIAS_LEDGER.json").read_text())
    if _sha(inner_root / "ALIAS_LEDGER.json") != inner["alias_ledger_sha256"]:
        raise ValueError("alias ledger differs from the inner report")
    declared = ledger["declared"]
    if set(sources) != set(declared):
        raise ValueError("outer scoring must declare exactly the inner panel's logical releases")
    index_file = Path(index_path).resolve()
    if _sha(index_file) != inner["index_sha256"]:
        raise ValueError("input index differs from the inner panel")
    pinned_laws, route_locks = {}, {}
    for name, source in sources.items():
        pinned_laws[name] = laws.load(source)
        if _public_descriptor(pinned_laws[name].descriptor) != declared[name]["source"]:
            raise ValueError(f"{name}: supplied release differs from its locked inner descriptor")
        canonical = declared[name]["canonical"]
        if canonical not in route_locks:
            route_locks[canonical] = outer_audit.reconstruct_selected_routes(inner_root, inner, canonical)
    # Sole outer-label access, after every route and artifact check above.
    if _load_outer_rows is None:
        census = json.loads(CENSUS_PATH.read_text())
        rows = outer_pool.load_verified_outer(
            data.index(index_file), anchor,
            census["anchors"][str(anchor)]["outer_assessment"], ORIGINAL_RESTORE_ROOT)
    else:
        rows = _load_outer_rows()
    legal = laws.legal_inputs(rows)
    outer_laws = {name: pinned_laws[name](legal) for name in sorted(sources)}
    results, contributions, breaks, confirmed = {}, {}, {}, {}
    for name in sorted(sources):
        canonical = declared[name]["canonical"]
        if name == canonical:
            results[name] = _score_release(rows, outer_laws[name], route_locks[name],
                                           inner["releases"][name]["private_contribution_prefix"],
                                           contributions)
            continue
        same = laws.law_identity(outer_laws[name]) == laws.law_identity(outer_laws[canonical])
        if same:
            confirmed[name] = canonical
            continue
        differing = np.any(outer_laws[name] != outer_laws[canonical], axis=1)
        breaks[name] = {"canonical": canonical, "outer_people_differing": int(differing.sum()),
                        "max_abs_difference": float(np.max(np.abs(outer_laws[name] - outer_laws[canonical]))),
                        "routes": "canonical's frozen inner routes (shared on inner rows)"}
        prefix = hashlib.sha256(f"outer_alias_break|{name}".encode()).hexdigest()[:12]
        results[name] = {**_score_release(rows, outer_laws[name], route_locks[canonical], prefix,
                                          contributions),
                         "scored_separately_from": canonical}
    root.mkdir(parents=True, mode=0o700)
    np.savez_compressed(root / "OUTER_CONTRIBUTIONS.npz", **contributions)
    report = {"schema": "pcrl-sc-outer-audit-v1", "anchor": anchor, "mode": "outer",
              "no_outer_fit_or_selection": True, "selection_lock_sha256": expected_lock_sha256,
              "releases": results, "outer_alias_confirmed": confirmed,
              "outer_alias_breaks": breaks,
              "contributions_sha256": _sha(root / "OUTER_CONTRIBUTIONS.npz")}
    evaluate._json_atomic(root / "OUTER_AUDIT.json", report)
    evaluate._seal_private_permissions(root)
    evaluate._json_atomic(root / "COMPLETE.json", {
        "schema": "pcrl-sc-outer-audit-complete-v1", "anchor": anchor,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": evaluate._inventory(root)})
    return report


def score_outer_j(anchor: int, index_path: str | Path, j_inner_panel_dir: str | Path,
                  selection_lock_path: str | Path, expected_lock_sha256: str,
                  output_dir: str | Path) -> dict:
    """Outer J continuity score behind THIS study's gate (never AR's outer_access).

    The J inner panel is AR/external_audit's (continuous J16 wire, no token);
    its artifacts must be pinned in the lock under external_context.J[anchor].
    Saved routes are replayed; nothing is fitted or reselected on outer rows.
    """
    from experiments.pcrl_adaptive_release_v1 import external_audit, outer_audit, outer_pool

    lock = verify_outer_gate(selection_lock_path, expected_lock_sha256)  # first, before any data
    root = Path(output_dir).resolve()
    if "private" not in root.parts or root.exists():
        raise FileExistsError("outer J output must be a new private directory")
    pinned = (lock.get("external_context") or {}).get("J", {}).get(str(anchor))
    inner_root = Path(j_inner_panel_dir).resolve()
    complete_path, report_path = inner_root / "COMPLETE.json", inner_root / "INNER_AUDIT.json"
    if (not isinstance(pinned, dict) or
            pinned.get("inner_complete_sha256") != _sha(complete_path) or
            pinned.get("inner_audit_sha256") != _sha(report_path)):
        raise PermissionError("J inner panel is not the one pinned in the selection lock")
    complete, inner = json.loads(complete_path.read_text()), json.loads(report_path.read_text())
    if (complete.get("schema") != "pcrl-adaptive-continuous-inner-complete-v1" or
            complete.get("artifacts") != evaluate._inventory(inner_root) or
            inner.get("schema") != "pcrl-adaptive-continuous-inner-audit-v1" or
            inner.get("anchor") != anchor or inner.get("outer_pool_opened") is not False or
            "J" not in inner.get("releases", {}) or
            inner.get("source_code_sha256", {}).get("external_audit.py") != _sha(external_audit.__file__)):
        raise ValueError("J inner receipt, anchor or source differs")
    index_file = Path(index_path).resolve()
    if _sha(index_file) != inner.get("input_index_sha256"):
        raise ValueError("input index differs from the J inner panel")
    index = data.index(index_file)
    own_locks, h_locks = outer_audit.reconstruct_selected_routes(inner_root, inner, "J")
    census = json.loads(CENSUS_PATH.read_text())
    rows = outer_pool.load_verified_outer(
        index, anchor, census["anchors"][str(anchor)]["outer_assessment"], ORIGINAL_RESTORE_ROOT)
    prepared = data.load_prepared(index, anchor)  # sanitized: J16 source, no outer labels
    j_rows = external_audit.attach_locked_outer_j(prepared, rows)
    law = np.ones((len(rows["ids"]), 1), dtype=np.float64)
    prefix = inner["releases"]["J"]["private_contribution_prefix"]
    results, contributions = {}, {}
    for role in ar_audit.ROLES:
        score = inherited_audit.score_frozen_route(j_rows, law, own_locks[role])
        h_score = inherited_audit.score_frozen_route(rows, law, h_locks[role])
        contributions.update(evaluate._private_contributions(prefix, role, score, h_score))
        results[role] = {"candidate": score["scores"], "H": h_score["scores"],
                         "n_people": len(score["loss"]),
                         "selected_candidate": own_locks[role]["selected"],
                         "H_selected_candidate": h_locks[role]["selected"]}
    root.mkdir(parents=True, mode=0o700)
    np.savez_compressed(root / "OUTER_CONTRIBUTIONS.npz", **contributions)
    report = {"schema": "pcrl-sc-outer-J-v1", "anchor": anchor, "mode": "outer",
              "contextual_not_matched_token_comparator": True,
              "no_outer_fit_or_selection": True, "selection_lock_sha256": expected_lock_sha256,
              "releases": {"J": {"roles": results, "private_contribution_prefix": prefix}},
              "contributions_sha256": _sha(root / "OUTER_CONTRIBUTIONS.npz")}
    evaluate._json_atomic(root / "OUTER_AUDIT.json", report)
    evaluate._seal_private_permissions(root)
    evaluate._json_atomic(root / "COMPLETE.json", {
        "schema": "pcrl-sc-outer-J-complete-v1", "anchor": anchor,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": evaluate._inventory(root)})
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def read_sources(path: str | Path) -> dict[str, Any]:
    """JSON {release_id: unit_dir | in-memory spec}; order is the declaration order."""
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or not value:
        raise ValueError("release source file must be a nonempty JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    inner = sub.add_parser("inner")
    inner.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    inner.add_argument("--index", required=True)
    inner.add_argument("--sources", required=True, help="JSON {release_id: unit_dir|spec}")
    inner.add_argument("--out", required=True)
    inner.add_argument("--slate", choices=("standard", "catchup"), default="standard")
    inner.add_argument("--resume", action="store_true")
    outer = sub.add_parser("outer")
    outer.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    outer.add_argument("--index", required=True)
    outer.add_argument("--sources", required=True)
    outer.add_argument("--inner-panel", required=True)
    outer.add_argument("--lock", required=True)
    outer.add_argument("--lock-sha256", required=True)
    outer.add_argument("--out", required=True)
    unlock = sub.add_parser("unlock", help="coordinator: verify the pushed lock, write OUTER_UNLOCK.json")
    unlock.add_argument("--lock-sha256", required=True)
    unlock.add_argument("--commit", required=True)
    outer_j = sub.add_parser("outer-j")
    outer_j.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    outer_j.add_argument("--index", required=True)
    outer_j.add_argument("--j-inner-panel", required=True)
    outer_j.add_argument("--lock", required=True)
    outer_j.add_argument("--lock-sha256", required=True)
    outer_j.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.mode == "unlock":
        print(json.dumps(write_unlock(args.lock_sha256, args.commit), indent=2, sort_keys=True))
        return
    if args.mode == "outer-j":
        report = score_outer_j(args.anchor, args.index, args.j_inner_panel, args.lock,
                               args.lock_sha256, args.out)
        print(json.dumps({"anchor": args.anchor, "scored": ["J"]}, sort_keys=True))
        return
    sources = read_sources(args.sources)
    if args.mode == "inner":
        report = audit_panel(args.anchor, sources, args.index, args.out,
                             slate=args.slate, resume=args.resume)
        print(json.dumps({"anchor": args.anchor, "canonical": sorted(report["releases"]),
                          "aliases": report["logical_to_canonical"],
                          "outer_pool_opened": report["outer_pool_opened"]}, sort_keys=True))
    else:
        report = score_outer(args.anchor, sources, args.index, args.inner_panel,
                             args.lock, args.lock_sha256, args.out)
        print(json.dumps({"anchor": args.anchor, "scored": sorted(report["releases"])},
                         sort_keys=True))


if __name__ == "__main__":
    main()
