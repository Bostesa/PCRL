"""Aggregate-only fixed-versus-fresh task-decoder check on inner 2018 pilot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import audit, data, method, release
from experiments.pcrl_task_directed_release_v1.audits import load_candidate

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "results/pcrl_task_aligned_cuts_v1"
PRIVATE = STUDY / "private/run"


def main() -> None:
    value = data.index(STUDY / "REUSABLE_INPUTS_PINNED.json")
    prepared = {anchor: data.load_prepared(value, anchor) for anchor in (0, 1, 2)}
    split = data.global_validation_split(prepared)
    anchor = 0
    item = prepared[anchor]
    indices = split["anchors"][anchor]["rows"]["inner_pilot"]
    pool = item["ctx"]["pools"]["downstream_validation"]
    y = np.asarray(pool["labels"]["same_residence"])[indices]
    w = np.asarray(pool["weights"])[indices]
    ha = np.asarray(pool["ha"])[indices]
    code = np.asarray(item["encoded"]["downstream_validation"]["codes"]["T0"])[indices]
    if not np.isin(y, [0, 1]).all():
        raise ValueError("pilot task label support changed")
    fixed_positive = np.asarray(item["encoded"]["downstream_validation"]["actions"][17])[indices]
    fixed_predictions = method.binary_predictions(fixed_positive)
    u1_fit = json.loads((PRIVATE / "a0_u1_decoder/FIT_U1.json").read_text())
    model_dir = PRIVATE / "a0_u1_decoder" / u1_fit["selected_model_relative"]
    decoder = load_candidate(model_dir)
    u1_predictions = decoder.predict_token_proba(ha, 17)
    maps = {
        "historical_Q": data.load_map(value, anchor, "Q"),
        "task_aligned_center": release.ChannelArtifact.load(
            PRIVATE / "a0_u1p1_z000/channel").Q,
    }
    audit_files = {"historical_Q": PRIVATE / "a0_u0p0_b010_audit/INNER_PANEL.json",
                   "task_aligned_center": PRIVATE / "a0_u1p1_z000_audit/INNER_PANEL.json"}
    result = {"schema": 1, "scope": "2018 inner_pilot, anchor 0, aggregate only",
              "split_assignment_sha256": split["assignment_sha256"],
              "U1_model_sha256": u1_fit["selected_model_sha256"],
              "n_original_people": len(y), "releases": {}}
    for name, q in maps.items():
        law = q[code]
        fixed_loss = audit.expected_token_loss(fixed_predictions, law, y)
        common_loss = audit.expected_token_loss(u1_predictions, law, y)
        fresh = json.loads(audit_files[name].read_text())["roles"]["utility:A/same_residence"]["candidate"]
        result["releases"][name] = {
            "U0_fixed": audit.score_weightings(fixed_loss, w),
            "U1_common": audit.score_weightings(common_loss, w),
            "fresh_validation_selected_probe": fresh,
        }
    contrasts = {}
    for family in ("U0_fixed", "U1_common", "fresh_validation_selected_probe"):
        contrasts[family] = {
            key: result["releases"]["task_aligned_center"][family][key]
            - result["releases"]["historical_Q"][family][key]
            for key in ("U", "PWGTP")}
    result["center_minus_historical_Q"] = contrasts
    destination = STUDY / "SURROGATE_GAP.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(contrasts, sort_keys=True))


if __name__ == "__main__":
    main()
