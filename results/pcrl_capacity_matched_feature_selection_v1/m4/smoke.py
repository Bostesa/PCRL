"""Role-sized synthetic timing for M4 fresh readers and V2 gates.

Run only as a registered S3 synthetic heavy unit. No Adult role file is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


REPO = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
for directory in (REPO, REPO / "results/pcrl_method_portfolio_v1/shared",
                  REPO / "results/pcrl_method_portfolio_v1/teams/c", STUDY / "m2"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from attack_eval import AttackView
from bank import build_shared_bank
from prototype import AdultGroupedSchema, IdentityCalibrator, PurposeRelease
from release import release_selected_pair
from s2_bridge import to_s2_attack_views
from results.pcrl_capacity_matched_feature_selection_v1.m1.capacity import capacity_geometry
from attack_runner import fit_complete_view_readers
from raw_union import canonical_raw_union, fit_raw_union_readers
from v2_runner import run_real_view_controls
from views import build_complete_attack_views


class SyntheticHead:
    def __init__(self, classes: int):
        self.classes = classes

    def predict_proba(self, values):
        x = np.asarray(values, dtype=np.float64)
        signal = x[:, 0:1] * np.linspace(0.1, 0.5, self.classes)[None, :]
        logits = signal + np.linspace(0.15, -0.15, self.classes)[None, :]
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)


def _role(n: int, first: int, groups: int, rng) -> tuple[dict, dict]:
    x = np.asarray(rng.normal(size=(n, 83)), dtype=np.float32)
    # Keep the dimensional/layout burden, with stable balanced sensitive labels.
    row = np.arange(first, first + n, dtype=np.int64)
    unit = np.arange(first + 100000, first + 100000 + n, dtype=np.int64)
    unit[-(n - groups):] = unit[:n - groups]
    sex = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(np.int64)
    return ({"X": x, "feature_names": AdultGroupedSchema().feature_names,
             "row_id": row, "unit": unit},
            {"row_id": row.copy(), "unit": unit.copy(), "sex": sex})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_smoke(output: Path) -> dict:
    if output.exists():
        raise FileExistsError("synthetic smoke output is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64DXSM(73101))
    att, sex_att = _role(4064, 1, 4061, rng)
    inner, sex_inner = _role(2235, 10001, 2234, rng)
    schema = AdultGroupedSchema()
    bank = build_shared_bank(capacity_geometry(schema.feature_names)["masks"])
    pair = next(item for item in bank.pairs
                if item.income_mask_id != item.occupation_mask_id)
    income = PurposeRelease(
        schema, bank.mask(pair.income_mask_id).atoms,
        SyntheticHead(2), "income", "synthetic-i", "identity-i",
        IdentityCalibrator())
    occupation = PurposeRelease(
        schema, bank.mask(pair.occupation_mask_id).atoms,
        SyntheticHead(6), "occupation_group", "synthetic-o", "identity-o",
        IdentityCalibrator())
    releases = {"income": income, "occupation_group": occupation}
    start = time.perf_counter()
    views = {}
    for role, source in (("att", att), ("inner", inner)):
        wire = release_selected_pair(
            source["X"], tuple(source["feature_names"]),
            source["row_id"], source["unit"],
            "ATTACK_FIT_NEW" if role == "att" else "INNER_SELECTION",
            pair, bank, income, occupation)["wire"]
        views[role], _ = build_complete_attack_views(
            source, wire, releases, schema, bridge=to_s2_attack_views,
            attack_view_type=AttackView)
    forward_wall = time.perf_counter() - start
    findings = {"schema": "pcrl-focused-m4-role-sized-synthetic-smoke-v1",
                "adult_role_values_accessed": False,
                "rows": {"ATTACK_FIT_NEW": len(att["row_id"]),
                         "INNER_SELECTION": len(inner["row_id"])},
                "pair_id": pair.pair_id, "forward_wall_s": forward_wall,
                "views": {}}
    with tempfile.TemporaryDirectory(prefix="pcrl-m4-smoke-") as temporary:
        private = Path(temporary) / ".private"
        private.mkdir()
        raw_indices = {}
        for view in ("income", "occupation_group", "coalition"):
            if view == "coalition":
                geometry = pair
            else:
                mask = pair.income_mask_id if view == "income" else pair.occupation_mask_id
                geometry = next(item for item in bank.pairs
                                if item.income_mask_id == item.occupation_mask_id == mask)
            columns = geometry.union_columns
            names = tuple(schema.feature_names[j] for j in columns)
            au = canonical_raw_union(att, source_columns=columns, feature_names=names)
            iu = canonical_raw_union(inner, source_columns=columns, feature_names=names)
            stage = time.perf_counter()
            raw_dir = private / f"raw_{view}"
            fit_raw_union_readers(
                au, sex_att, iu, seed=0, union_sha256=geometry.union_sha256,
                union_atoms=geometry.union_atoms,
                source_schema_sha256=bank.source_schema_sha256,
                lock_sha256="a" * 64, manifest_sha256="b" * 64,
                private_output=raw_dir)
            raw_indices[view] = (raw_dir / "RAW_UNION_INDEX.json", columns, names)
            findings["views"][view] = {
                "raw_columns": len(columns),
                "complete_columns": int(views["att"][view].x.shape[1]),
                "raw_fit_wall_s": time.perf_counter() - stage}
        locals_for_coalition = {}
        for view in ("income", "occupation_group", "coalition"):
            raw_index, columns, names = raw_indices[view]
            stage = time.perf_counter()
            out = private / f"attack_{view}"
            fit_complete_view_readers(
                att_view=views["att"][view], sex_att=sex_att,
                inner_view=views["inner"][view], sex_inner=sex_inner,
                att_source=att, inner_source=inner,
                source_columns=columns, feature_names=names,
                raw_index_path=raw_index,
                expected_raw_index_sha256=_sha(raw_index),
                release_index_sha256="c" * 64,
                lock_sha256="a" * 64, manifest_sha256="b" * 64,
                seed=0, pair_id=pair.pair_id, view_name=view,
                private_output=out,
                coalition_local_dependencies=(locals_for_coalition
                                              if view == "coalition" else None),
            )
            attack_path = out / "ATTACK_INDEX.json"
            findings["views"][view]["complete_attack_wall_s"] = \
                time.perf_counter() - stage
            findings["views"][view]["reader_count"] = \
                len(json.loads(attack_path.read_text())["reader_order"])
            if view != "coalition":
                locals_for_coalition[view] = (
                    attack_path, _sha(attack_path),
                    views["att"][view], views["inner"][view],
                )
            stage = time.perf_counter()
            control = run_real_view_controls(
                att_view=views["att"][view], sex_att=sex_att,
                inner_view=views["inner"][view], sex_inner=sex_inner,
                seed=0, pair_id=pair.pair_id, view_name=view,
                release_index_sha256="c" * 64, lock_sha256="a" * 64,
                manifest_sha256="b" * 64,
                private_output=private / f"v2_{view}",
            )
            findings["views"][view]["v2_wall_s"] = time.perf_counter() - stage
            findings["views"][view]["v2_status"] = control["status"]
    findings["fresh_raw_reader_fits"] = 6
    findings["fresh_complete_reader_fits"] = 12
    findings["fresh_v2_fits"] = 6
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(findings, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_smoke(Path(args.output))
    print(json.dumps({"schema": result["schema"],
                      "adult_role_values_accessed": False,
                      "output_sha256": _sha(Path(args.output))}, sort_keys=True))


if __name__ == "__main__":
    main()
