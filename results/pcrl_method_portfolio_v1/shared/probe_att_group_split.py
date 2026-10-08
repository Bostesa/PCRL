"""Prefit aggregate-only check of ATTACK_FIT_NEW class presence by fixed group split.

This reads only the S4-admitted ATTACK_FIT_NEW SEX role file and its row/group
identity. It fits no model, exports no person array and must run via S3's
prefit audited role-check queue after S4's role-export admission.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from attack_eval import attack_fit_tune_split
from safe_data_guard import SafeRoleStore


CAMPAIGN = Path(__file__).resolve().parent.parent


def aggregate_probe() -> dict:
    store = SafeRoleStore(CAMPAIGN / "DATA_ACCESS_MANIFEST.json")
    if store.manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise PermissionError("ATT role export is not independently admitted")
    role = store.read("ATTACK_FIT_NEW", "sex", "attack")
    sex = np.asarray(role["sex"], dtype=np.int64)
    unit = np.asarray(role["unit"], dtype=np.int64)
    split = attack_fit_tune_split(unit)
    counts = {part: [int(np.sum(sex[idx] == c)) for c in (0, 1)]
              for part, idx in (("fit", split["fit"]), ("tune", split["tune"]))}
    if any(v == 0 for values in counts.values() for v in values):
        raise ValueError("ATT fixed group split lacks a sensitive class")
    return {"schema": "pcrl-att-group-split-aggregate-v1", "status": "PASS",
            "rows": int(sex.size), "groups": int(np.unique(unit).size),
            "fit_groups": split["fit_groups"], "tune_groups": split["tune_groups"],
            "fit_tune_class_counts": counts,
            "tune_group_id_sha256": split["tune_group_id_sha256"],
            "scope": "role-only ATT SEX and group identities; no model fit"}


if __name__ == "__main__":
    print(json.dumps(aggregate_probe(), sort_keys=True))
