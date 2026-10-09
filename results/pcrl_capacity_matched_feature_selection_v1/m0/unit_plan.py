"""Mechanical prelock enumeration of every possible focused science unit.

This reads public mask geometry and role-file hashes only. It never opens an
NPZ member or trains a model. The private registry contains absolute admitted
role paths; only its SHA and aggregate count are published in the lock.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .queue import sha256


ROLE_FIELDS = {
    "fit_bank": lambda s: (
        "OSF_DEFENSE_FIT:features", "OSF_DEFENSE_FIT:income",
        "OSF_DEFENSE_FIT:occupation", f"OSF_DEFENSE_FIT:references_s{s}",
        "OSF_DEFENSE_FIT:sex", "CALIBRATION_HELDOUT:features",
        "CALIBRATION_HELDOUT:income", "CALIBRATION_HELDOUT:occupation"),
    "release": lambda s: ("ATTACK_FIT_NEW:features", "INNER_SELECTION:features"),
    "raw_union": lambda s: ("ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
                            "INNER_SELECTION:features"),
    "attack_view": lambda s: ("ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
                              "INNER_SELECTION:features", "INNER_SELECTION:sex"),
    "v2_probe": lambda s: ("ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
                           "INNER_SELECTION:features", "INNER_SELECTION:sex"),
    "independent_replay": lambda s: (
        "OSF_DEFENSE_FIT:features", "OSF_DEFENSE_FIT:income",
        "OSF_DEFENSE_FIT:occupation", f"OSF_DEFENSE_FIT:references_s{s}",
        "ATTACK_FIT_NEW:features", "ATTACK_FIT_NEW:sex",
        "INNER_SELECTION:features", "INNER_SELECTION:income",
        "INNER_SELECTION:occupation", "INNER_SELECTION:sex",
        f"INNER_SELECTION:references_s{s}"),
}


def _interpreter_path(python: Path | None) -> str:
    """Keep a venv launcher path; resolving its symlink escapes the venv."""
    path = Path(os.path.abspath(python or sys.executable))
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)
VIEW_TAGS = {"income": "INCOME", "occupation_group": "OCCUPATION",
             "coalition": "COALITION"}


def _role_inputs(keys: tuple[str, ...], prior: Path, manifest: dict) -> tuple[dict, dict]:
    declared = {}
    inputs = {}
    for key in keys:
        path = (prior / manifest["role_file_relative_path"][key]).resolve(strict=True)
        expected = manifest["role_file_sha256"][key]
        if sha256(path) != expected:
            raise ValueError("admitted role file hash differs")
        declared[key] = {"path": str(path), "sha256": expected}
        inputs[str(path)] = expected
    return declared, inputs


def _base(*, uid: str, stage: str, argv: list[str], script: Path,
          worktree: Path, output: Path | list[Path], deps: list[str], seed: int,
          prior: Path, manifest: dict, source_map_sha: str,
          caps: dict[str, dict], role_keys: tuple[str, ...] | None = None) -> dict:
    keys = ROLE_FIELDS[stage](seed) if role_keys is None else role_keys
    role_inputs, inputs = _role_inputs(keys, prior, manifest)
    cap = caps[stage]
    role_access = list(dict.fromkeys(key.split(":", 1)[0] for key in role_inputs))
    return {
        "unit_id": uid, "mode": "real", "stage": stage,
        "data_access": (stage.upper() + "_ADMITTED_ROLES_ONLY" if keys
                        else "NONE_PRIVATE_DERIVED_ONLY"),
        "argv": argv, "cwd": str(worktree), "script": str(script),
        "script_sha256": sha256(script), "inputs": inputs,
        "role_inputs": role_inputs, "role_access": role_access,
        "outputs": [str(path) for path in (output if isinstance(output, list)
                                           else [output])], "deps": deps,
        "source_map_sha256": source_map_sha,
        "seal_sha256": "$LOCAL_SEAL_SHA256",
        "allow_child_processes": False,
        "cpu_reserve_s": cap["cpu_reserve_s"],
        "wall_limit_s": cap["wall_limit_s"],
        "max_rss_gib": cap["max_rss_gib"],
    }


def build_core_unit_specs(*, worktree: Path, study: Path, prior: Path,
                          manifest: dict, source_map_sha: str,
                          geometry: dict, caps: dict[str, dict],
                          seeds: tuple[int, ...] = (0, 1, 2),
                          python: Path | None = None) -> dict[str, dict]:
    """Enumerate FIT, six RAW, 25 RELEASE and 75 ATT/75 V2 per seed."""
    worktree = worktree.resolve(strict=True)
    study = study.resolve(strict=True)
    prior = prior.resolve(strict=True)
    interpreter = _interpreter_path(python)
    m2_script = (study / "m2/fit_worker.py").resolve(strict=True)
    m4_script = (study / "m4/worker.py").resolve(strict=True)
    queue = study / ".private/queue"
    private = study / ".private"
    lock = study / "CAMPAIGN_LOCK.json"
    pairs = geometry["pairs"]
    masks = {mask["mask_id"]: mask for mask in geometry["masks"]}
    diagonal_union = {
        left: pair["union_sha256"]
        for pair in pairs
        for left, right in [pair["pair_id"].split("__")]
        if left == right
    }
    union_shas = sorted({pair["union_sha256"] for pair in pairs})
    if (len(masks) != 5 or len(pairs) != 25 or len(union_shas) != 6 or
            len({pair["pair_id"] for pair in pairs}) != 25 or
            set(diagonal_union) != set(masks) or
            set(seeds) != {0, 1, 2}):
        raise ValueError("focused finite bank or seeds changed")
    specs: dict[str, dict] = {}

    def add(spec: dict) -> None:
        uid = spec["unit_id"]
        if uid in specs:
            raise ValueError("duplicate science unit ID")
        specs[uid] = spec

    for seed in seeds:
        fit_id = f"M2-FIT-S{seed}"
        fit_output = private / "m2/fit" / f"seed_{seed}" / "FIT_RECEIPT.json"
        add(_base(
            uid=fit_id, stage="fit_bank", script=m2_script, worktree=worktree,
            argv=[interpreter, str(m2_script), "--unit-id", fit_id,
                  "--seed", str(seed), "--lock", str(lock),
                  "--queue-root", str(queue), "--private-root", str(private)],
            output=fit_output, deps=[], seed=seed, prior=prior,
            manifest=manifest, source_map_sha=source_map_sha, caps=caps))
        for union_sha in union_shas:
            uid = f"M4-RAW-S{seed}-{union_sha[:16].upper()}"
            output = private / "m4/raw_union" / f"seed_{seed}" / union_sha / "RAW_UNION_INDEX.json"
            add(_base(
                uid=uid, stage="raw_union", script=m4_script, worktree=worktree,
                argv=[interpreter, str(m4_script), "raw_union", "--unit-id", uid,
                      "--seed", str(seed), "--union-sha", union_sha,
                      "--queue-root", str(queue), "--output-index", str(output)],
                output=output, deps=[fit_id], seed=seed, prior=prior,
                manifest=manifest, source_map_sha=source_map_sha, caps=caps))
        for pair in pairs:
            pair_id = pair["pair_id"]
            income_mask_id, occupation_mask_id = pair_id.split("__")
            release_id = f"M4-REL-S{seed}-{pair_id}"
            release_output = private / "m4/releases" / f"seed_{seed}" / pair_id / "RELEASE_RECEIPT.json"
            add(_base(
                uid=release_id, stage="release", script=m4_script, worktree=worktree,
                argv=[interpreter, str(m4_script), "release", "--unit-id", release_id,
                      "--seed", str(seed), "--pair-id", pair_id,
                      "--queue-root", str(queue), "--output-index", str(release_output)],
                output=release_output, deps=[fit_id], seed=seed, prior=prior,
                manifest=manifest, source_map_sha=source_map_sha, caps=caps))
            local_ids = {}
            for view in VIEW_TAGS:
                tag = VIEW_TAGS[view]
                uid = f"M4-ATT-S{seed}-{pair_id}-{tag}"
                output = private / "m4/attacks" / f"seed_{seed}" / pair_id / view / "ATTACK_INDEX.json"
                union_sha = (pair["union_sha256"] if view == "coalition" else
                             diagonal_union[income_mask_id if view == "income"
                                            else occupation_mask_id])
                raw_id = f"M4-RAW-S{seed}-{union_sha[:16].upper()}"
                deps = [fit_id, release_id, raw_id]
                if view == "coalition":
                    deps += [local_ids["income"], local_ids["occupation_group"]]
                add(_base(
                    uid=uid, stage="attack_view", script=m4_script, worktree=worktree,
                    argv=[interpreter, str(m4_script), "attack", "--unit-id", uid,
                          "--seed", str(seed), "--pair-id", pair_id,
                          "--view", view, "--queue-root", str(queue),
                          "--output-index", str(output)],
                    output=output, deps=deps, seed=seed, prior=prior,
                    manifest=manifest, source_map_sha=source_map_sha, caps=caps))
                local_ids[view] = uid
                v2_id = f"M4-V2-S{seed}-{pair_id}-{tag}"
                v2_output = private / "m4/v2" / f"seed_{seed}" / pair_id / view / "V2_INDEX.json"
                add(_base(
                    uid=v2_id, stage="v2_probe", script=m4_script, worktree=worktree,
                    argv=[interpreter, str(m4_script), "v2", "--unit-id", v2_id,
                          "--seed", str(seed), "--pair-id", pair_id,
                          "--view", view, "--queue-root", str(queue),
                          "--output-index", str(v2_output)],
                    output=v2_output, deps=[fit_id, release_id], seed=seed,
                    prior=prior, manifest=manifest, source_map_sha=source_map_sha,
                    caps=caps))
    expected = len(seeds) * (1 + 6 + 25 + 75 + 75)
    if len(specs) != expected:
        raise AssertionError("science unit enumeration is incomplete")
    return specs


def build_endpoint_inference_specs(*, worktree: Path, study: Path, prior: Path,
                                   manifest: dict, source_map_sha: str,
                                   caps: dict[str, dict],
                                   python: Path | None = None) -> dict[str, dict]:
    """Register FIRST, SCREEN and conditional FINALIST aggregate workers."""
    worktree = worktree.resolve(strict=True)
    study = study.resolve(strict=True)
    prior = prior.resolve(strict=True)
    interpreter = _interpreter_path(python)
    endpoint_script = (study / "m4/endpoint_worker.py").resolve(strict=True)
    inference_script = (study / "m3/inference_cli.py").resolve(strict=True)
    private = study / ".private"
    queue = private / "queue"
    lock = study / "CAMPAIGN_LOCK.json"
    specs = {}
    for phase, seeds in (("FIRST", (0,)), ("SCREEN", (0, 1)),
                         ("FINALIST", (0, 1, 2))):
        endpoint_id = "M4-ENDPOINTS-" + phase
        endpoint_path = private / "m4/endpoints" / phase / "INFERENCE_INDEX.json"
        role_keys = (
            "INNER_SELECTION:features", "INNER_SELECTION:income",
            "INNER_SELECTION:occupation", "INNER_SELECTION:sex",
            "OSF_DEFENSE_FIT:income", "OSF_DEFENSE_FIT:occupation",
            *(f"INNER_SELECTION:references_s{seed}" for seed in seeds),
        )
        specs[endpoint_id] = _base(
            uid=endpoint_id, stage="endpoint_index", script=endpoint_script,
            worktree=worktree,
            argv=[interpreter, str(endpoint_script), "--phase", phase,
                  "--unit-id", endpoint_id, "--queue-root", str(queue),
                  "--output-index", str(endpoint_path)],
            output=endpoint_path, deps=[f"M2-FIT-S{seed}" for seed in seeds],
            seed=seeds[0], prior=prior, manifest=manifest,
            source_map_sha=source_map_sha, caps=caps, role_keys=role_keys)
        if phase == "FIRST":
            continue
        inference_id = "M3-INFERENCE-" + phase
        inference_path = private / "m3/inference" / (phase + ".json")
        specs[inference_id] = _base(
            uid=inference_id, stage="simultaneous_inference",
            script=inference_script, worktree=worktree,
            argv=[interpreter, str(inference_script), "--unit-id", inference_id,
                  "--queue-root", str(queue), "--lock-path", str(lock),
                  "--index-path", str(endpoint_path), "--output",
                  str(inference_path)],
            output=inference_path, deps=[endpoint_id], seed=seeds[0],
            prior=prior, manifest=manifest, source_map_sha=source_map_sha,
            caps=caps, role_keys=())
    return specs


def build_replay_specs(*, worktree: Path, study: Path, prior: Path,
                       manifest: dict, source_map_sha: str, geometry: dict,
                       caps: dict[str, dict], python: Path | None = None) -> dict[str, dict]:
    """Enumerate 75 potential independent pair replays and two aggregates."""
    worktree = worktree.resolve(strict=True)
    study = study.resolve(strict=True)
    prior = prior.resolve(strict=True)
    interpreter = _interpreter_path(python)
    pair_script = (study / "m5/replay_worker.py").resolve(strict=True)
    aggregate_script = (study / "m5/replay_aggregate.py").resolve(strict=True)
    private = study / ".private"
    queue = private / "queue"
    pairs = geometry["pairs"]
    diagonal_union = {
        left: pair["union_sha256"]
        for pair in pairs
        for left, right in [pair["pair_id"].split("__")]
        if left == right
    }
    if len(pairs) != 25 or len(diagonal_union) != 5:
        raise ValueError("independent replay geometry is not the frozen bank")
    specs = {}
    for seed in (0, 1, 2):
        for pair in pairs:
            pair_id = pair["pair_id"]
            left, right = pair_id.split("__")
            uid = f"M5-REPLAY-S{seed}-{pair_id}"
            folder = private / "m5/replays" / f"seed_{seed}" / pair_id
            report = folder / "REPLAY_REPORT.json"
            values = folder / "REPLAY_VALUES.npz"
            raw_shas = sorted({diagonal_union[left], diagonal_union[right],
                               pair["union_sha256"]})
            deps = ([f"M2-FIT-S{seed}", f"M4-REL-S{seed}-{pair_id}"] +
                    [f"M4-RAW-S{seed}-{sha[:16].upper()}" for sha in raw_shas] +
                    [f"M4-ATT-S{seed}-{pair_id}-{tag}" for tag in VIEW_TAGS.values()] +
                    [f"M4-V2-S{seed}-{pair_id}-{tag}" for tag in VIEW_TAGS.values()])
            specs[uid] = _base(
                uid=uid, stage="independent_replay", script=pair_script,
                worktree=worktree,
                argv=[interpreter, str(pair_script), "--unit-id", uid,
                      "--seed", str(seed), "--pair-id", pair_id,
                      "--queue-root", str(queue), "--output-report", str(report),
                      "--output-values", str(values)],
                output=[report, values], deps=deps, seed=seed, prior=prior,
                manifest=manifest, source_map_sha=source_map_sha, caps=caps)
    for phase in ("SCREEN", "FINALIST"):
        uid = f"M5-REPLAY-AGGREGATE-{phase}"
        report = private / "m5/aggregates" / phase / "AGGREGATE_REPORT.json"
        specs[uid] = _base(
            uid=uid, stage="independent_aggregate", script=aggregate_script,
            worktree=worktree,
            argv=[interpreter, str(aggregate_script), "--phase", phase,
                  "--unit-id", uid, "--queue-root", str(queue),
                  "--output-report", str(report)],
            output=report,
            deps=[f"M4-ENDPOINTS-{phase}", f"M3-INFERENCE-{phase}"],
            seed=0, prior=prior, manifest=manifest,
            source_map_sha=source_map_sha, caps=caps, role_keys=())
    if len(specs) != 77:
        raise AssertionError("independent replay potential registry incomplete")
    return specs
