"""Bind every Team C release field to S2's actual shared AttackView.

The ordinary matrix contains exactly the disclosed per-person raw values,
calibrated probabilities, and decisions. The defense-aware matrix additionally
recomputes pre-calibration probabilities from those disclosed raw values and
the restored public heads. It never reads provider-only columns or labels.
"""

from __future__ import annotations

import json
from typing import Mapping

import numpy as np

from evaluation import attack_views_from_release
from prototype import PurposeRelease


_NUMERIC_FIELDS = {
    "income": ("selected_values", "probabilities", "decisions"),
    "occupation_group": ("selected_values", "probabilities", "decisions"),
    "coalition": (
        "selected_values", "income_probabilities",
        "occupation_group_probabilities", "income_decisions",
        "occupation_group_decisions",
    ),
}


def _matrix_and_columns(
    name: str, wire: Mapping[str, object], raw: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[str, int, int], ...]]:
    blocks = []
    spans = []
    start = 0
    for field in _NUMERIC_FIELDS[name]:
        if field == "selected_values":
            block = raw
        elif field.endswith("decisions") or field == "decisions":
            classes = 2 if field in ("decisions", "income_decisions") and name != "occupation_group" else 6
            decision = np.asarray(wire[field], dtype=np.int64)
            block = np.eye(classes, dtype=np.float64)[decision]
        else:
            block = np.asarray(wire[field], dtype=np.float64)
        blocks.append(block)
        end = start + block.shape[1]
        spans.append((field, start, end))
        start = end
    return np.concatenate(blocks, axis=1), tuple(spans)


def _public_precalibration(
    wire: Mapping[str, object], purpose: PurposeRelease, *, name: str,
) -> np.ndarray:
    if (purpose.purpose != name
            or purpose.selected_atoms != tuple(wire["selected_groups"])
            or purpose.head_id != wire["head_id"]
            or purpose.calibrator_id != wire["calibrator_id"]):
        raise ValueError("public head/calibrator identity differs from transmitted wire")
    raw = np.asarray(wire["selected_values"], dtype=np.float64)
    probability = np.asarray(purpose.head.predict_proba(raw), dtype=np.float64)
    classes = 2 if name == "income" else 6
    if (probability.shape != (len(raw), classes)
            or not np.all(np.isfinite(probability))
            or np.any(probability < 0) or np.any(probability > 1)
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-8, rtol=0)):
        raise ValueError("public head recomputation returned invalid probabilities")
    calibrated = np.asarray(purpose.calibrator.transform(probability.copy()), dtype=np.float64)
    if not np.array_equal(calibrated, np.asarray(wire["probabilities"], dtype=np.float64)):
        raise ValueError("public head/calibrator recomputation differs from transmitted output")
    return probability


def to_s2_attack_views(
    release: Mapping[str, Mapping[str, object]],
    private_alignment: Mapping[str, Mapping[str, object]],
    public_releases: Mapping[str, PurposeRelease],
    attack_view_type: type,
) -> dict[str, object]:
    """Return validated S2 AttackView instances for all three actual views.

    ``attack_view_type`` is S2's imported ``attack_eval.AttackView``. Keeping
    the class injected avoids another data-loader path in this team module.
    """
    if set(public_releases) != {"income", "occupation_group"}:
        raise ValueError("both restored public recipient heads are required")
    validated = attack_views_from_release(release, private_alignment)
    precal_i = _public_precalibration(
        release["income"], public_releases["income"], name="income",
    )
    precal_o = _public_precalibration(
        release["occupation_group"], public_releases["occupation_group"],
        name="occupation_group",
    )
    outputs = {}
    for name in ("income", "occupation_group", "coalition"):
        wire = release[name]
        prior = validated[name]
        x, spans = _matrix_and_columns(name, wire, prior.raw_matrix)
        if not np.array_equal(prior.complete_matrix[:, :x.shape[1]], x):
            raise ValueError("Team C and S2 numeric wire columns disagree")
        numeric = _NUMERIC_FIELDS[name]
        constants = tuple(
            (field, json.dumps(wire[field], sort_keys=True, separators=(",", ":")))
            for field in sorted(set(wire) - set(numeric))
        )
        defense_probabilities = (
            (precal_i,) if name == "income" else
            (precal_o,) if name == "occupation_group" else
            (precal_i, precal_o)
        )
        defense_aware = np.concatenate((x, *defense_probabilities), axis=1)
        aligned = private_alignment[name]
        view = attack_view_type(
            row_id=np.asarray(aligned["row_id"], dtype=np.int64).copy(),
            unit=np.asarray(aligned["unit"], dtype=np.int64).copy(),
            x=x, components=numeric,
            wire_components=numeric + tuple(key for key, _ in constants),
            component_columns=spans, wire_constants=constants,
            x_defense_aware=defense_aware,
            cell_key=None, x_metadata=None, x_code_bits=None,
            x_raw_only=prior.raw_matrix.copy(),
        )
        outputs[name] = view.validate()
    return outputs
