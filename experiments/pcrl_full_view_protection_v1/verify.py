"""Independent replay of stored finite-channel laws, costs and privacy."""

import argparse
import datetime
import hashlib
import json
import math
import os
import tempfile

import numpy as np

from .finite import conditional_mi
from .synthetic import fixtures, task_cost, task_information


def _check_channel(nominal, laws, value, errors, path):
    q = np.asarray(value["channel"], dtype=float)
    if q.shape != (nominal.shape[-2], 2) or np.min(q) < -1e-10 or not np.allclose(q.sum(axis=1), 1, atol=1e-8):
        errors.append((path, "invalid channel"))
        return
    if abs(task_cost(nominal, q) - value["cost"]) > 1e-9:
        errors.append((path, "cost replay"))
    if "task_information" in value and abs(task_information(nominal, q) - value["task_information"]) > 1e-9:
        errors.append((path, "task information replay"))
    for i, law in enumerate(laws):
        for role, axes in (("A", (1,)), ("AB", (1, 2))):
            actual = conditional_mi(law, q, axes)
            if abs(actual - value["cmi_by_law"][i][role]) > 1e-9:
                errors.append((path, i, role, "CMI replay"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    errors = []
    fit_count = 0
    control_count = 0
    map_count = 0
    for name, case in fixtures().items():
        path = os.path.join(args.input_dir, name + ".json")
        stored = json.load(open(path))
        laws = case["laws"]
        if stored["law_sha256"] != [hashlib.sha256(p.tobytes()).hexdigest() for p in laws]:
            errors.append((name, "law hash"))
        nominal = laws[0]
        for key, value in stored["controls"].items():
            _check_channel(nominal, laws, value, errors, f"{name}/control/{key}")
            control_count += 1
        for family in ("randomized_response", "withholding"):
            for key, value in stored[family].items():
                _check_channel(nominal, laws, value, errors, f"{name}/{family}/{key}")
                control_count += 1
        if len(stored["deterministic"]) != 2 ** nominal.shape[-2]:
            errors.append((name, "deterministic enumeration size"))
        for entry in stored["deterministic"]:
            q = np.eye(2)[entry["assignment"]]
            if abs(task_cost(nominal, q) - entry["cost"]) > 1e-9:
                errors.append((name, "deterministic cost"))
            map_count += 1
        for budget, entry in stored["budgets"].items():
            feasible_d = next((d for d in stored["deterministic"]
                               if d["max_full_view_cmi"] <= float(budget) + 1e-7), None)
            if feasible_d != entry["selected_deterministic"]:
                errors.append((name, budget, "deterministic selector"))
            for family, selector in (("randomized_response", "selected_randomized_response"),
                                     ("withholding", "selected_withholding")):
                feasible = [(key, value) for key, value in stored[family].items()
                            if value["max_full_view_cmi"] <= float(budget) + 1e-7]
                chosen = min(feasible, key=lambda pair: (pair[1]["cost"], float(pair[0])))[0] if feasible else None
                if chosen != entry[selector]:
                    errors.append((name, budget, selector))
            for method, value in entry["methods"].items():
                if "alias" in value:
                    continue
                fit_count += 1
                if value.get("status") == "quarantined":
                    errors.append((name, budget, method, "quarantined"))
                    continue
                _check_channel(nominal, laws, value, errors, f"{name}/{budget}/{method}")
                if value["objective_lower_bound"] > value["cost"] + 1e-8:
                    errors.append((name, budget, method, "lower bound orientation"))
                if method == "robust" and value["max_full_view_cmi"] > float(budget) + 1e-7:
                    errors.append((name, budget, method, "privacy feasibility"))
                if method == "capacity" and value["radius"]["upper"] > float(budget) + 1e-7:
                    errors.append((name, budget, method, "radius feasibility"))
                if "outside_cmi" in value:
                    q = np.asarray(value["channel"])
                    for i, law in enumerate(case["outside"]):
                        for role, axes in (("A", (1,)), ("AB", (1, 2))):
                            actual = conditional_mi(law, q, axes)
                            if abs(actual - value["outside_cmi"][i][role]) > 1e-9:
                                errors.append((name, budget, method, i, role, "outside replay"))
    rational = json.load(open(os.path.join(args.input_dir, "rational_separation.json")))
    channel = rational["budgets"]["0.0"]["methods"]["robust"]
    if abs(channel["task_information"] - 0.0863046217355342) > 1e-9 or abs(channel["cost"] - 0.3) > 1e-9:
        errors.append(("rational", "independent known value"))
    report = {"schema": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "fixture_count": len(fixtures()), "fit_count": fit_count,
              "control_channels_replayed": control_count, "deterministic_maps_replayed": map_count,
              "errors": errors, "status": "pass" if not errors else "fail"}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(args.output), prefix=".verify-",
                                 delete=False) as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        temp = f.name
    os.replace(temp, args.output)
    if errors:
        raise SystemExit(f"verification failed: {errors}")


if __name__ == "__main__":
    main()
