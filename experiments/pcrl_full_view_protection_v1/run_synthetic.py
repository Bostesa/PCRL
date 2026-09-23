"""Resumable registered finite suite with atomic per-case checkpoints."""

import argparse
import datetime
import hashlib
import json
import os
import resource
import tempfile
import time

import numpy as np

from .finite import conditional_mi, information_radius_bracket
from .optimize import deterministic_controls, optimize_channel
from .synthetic import fixtures, task_cost, task_information


BUDGETS = (0.0, 0.001, 0.01, 0.05)
LEVELS = (0.25, 0.5, 0.75)


def _atomic_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path), prefix=".checkpoint-",
                                 delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp = handle.name
    os.replace(temp, path)


def _evaluate(law, laws, q):
    cmi = [{role: conditional_mi(p, q, axes)
            for role, axes in (("A", (1,)), ("AB", (1, 2)))} for p in laws]
    return {"channel": q.tolist(), "cost": task_cost(law, q),
            "task_information": task_information(law, q),
            "cmi_by_law": cmi,
            "max_full_view_cmi": max(v for row in cmi for v in row.values())}


def _family_selection(members, budget):
    feasible = [(key, v) for key, v in members.items()
                if v["max_full_view_cmi"] <= budget + 1e-7]
    if not feasible:
        return None
    return min(feasible, key=lambda pair: (pair[1]["cost"], float(pair[0])))[0]


def run_case(name, case):
    started = time.monotonic()
    laws = case["laws"]
    nominal = laws[0]
    deterministic = deterministic_controls(nominal, laws)
    best = deterministic[0]
    unprotected = np.eye(2)[list(best["assignment"])]
    const_costs = [(task_cost(nominal, np.tile(np.eye(2)[z], (nominal.shape[-2], 1))), z)
                   for z in range(2)]
    best_constant = min(const_costs)[1]
    constant = np.tile(np.eye(2)[best_constant], (nominal.shape[-2], 1))
    controls = {"constant": _evaluate(nominal, laws, constant),
                "unprotected_deterministic": _evaluate(nominal, laws, unprotected),
                "null_zero_column": _evaluate(nominal, laws, np.tile([1.0, 0.0],
                                                                     (nominal.shape[-2], 1)))}
    rr = {}
    withholding = {}
    for level in LEVELS:
        rr[str(level)] = _evaluate(nominal, laws, (1 - level) * unprotected +
                                   level * unprotected[:, ::-1])
        withholding[str(level)] = _evaluate(nominal, laws, (1 - level) * unprotected +
                                            level * constant)
    result = {"name": name, "law_sha256": [hashlib.sha256(p.tobytes()).hexdigest()
                                           for p in laws],
              "law_shapes": [list(p.shape) for p in laws],
              "outside_law_sha256": [hashlib.sha256(p.tobytes()).hexdigest()
                                     for p in case["outside"]],
              "deterministic": deterministic,
              "controls": controls, "randomized_response": rr,
              "withholding": withholding, "budgets": {},
              "published_robust_adaptation": "same finite-list Shannon-CMI programme as robust; identical channel, no duplicate fit"}
    for budget in BUDGETS:
        key = str(budget)
        selected_d = next((d for d in deterministic
                           if d["max_full_view_cmi"] <= budget + 1e-7), None)
        entry = {"selected_deterministic": selected_d,
                 "selected_randomized_response": _family_selection(rr, budget),
                 "selected_withholding": _family_selection(withholding, budget),
                 "methods": {}}
        for method in ("robust", "nominal", "capacity"):
            if method == "nominal" and len(laws) == 1:
                entry["methods"][method] = {"alias": "robust; U has one law"}
                continue
            try:
                solution = optimize_channel(nominal, laws, budget, method)
                solution["task_information"] = task_information(nominal,
                                                                  np.array(solution["channel"]))
                if case["outside"]:
                    solution["outside_cmi"] = [_evaluate(nominal, [p],
                                                          np.array(solution["channel"]))["cmi_by_law"][0]
                                               for p in case["outside"]]
                entry["methods"][method] = solution
            except Exception as exc:
                entry["methods"][method] = {"status": "quarantined", "cause": repr(exc),
                                            "attempts": 1}
        result["budgets"][key] = entry
    result["elapsed_seconds"] = time.monotonic() - started
    result["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for name, case in fixtures().items():
        path = os.path.join(args.output_dir, name + ".json")
        if os.path.exists(path):
            continue
        result = run_case(name, case)
        result["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _atomic_json(path, result)
        print(name, round(result["elapsed_seconds"], 3), flush=True)


if __name__ == "__main__":
    main()
