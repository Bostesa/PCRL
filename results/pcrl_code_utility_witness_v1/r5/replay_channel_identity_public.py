"""Independent public-aggregate check of the prior finite-bank identity costs.

No old or current person-level file is opened. This checks arithmetic and
cross-source consistency, not the underlying saved release or parameterized
channel search. The latter need a separately admitted private replay.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "pcrl_latent_and_sampled_methods_v1"
CURRENT = ROOT / "pcrl_code_utility_witness_v1"
SOURCE = OLD / "r04/A_PREFIT_INDEPENDENT_REPLAY.json"
PRODUCER = CURRENT / "r2/CHANNEL_IDENTITY_AUDIT.csv"
OUTPUT = CURRENT / "r5/CHANNEL_IDENTITY_PUBLIC_REPLAY.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run() -> dict:
    old = json.loads(SOURCE.read_text())
    if set(old["seeds"]) != {"0", "1"}:
        raise ValueError("two old model seeds required")
    expected = {}
    for seed in (0, 1):
        for role in ("OSF_DEFENSE_FIT", "INNER_SELECTION"):
            entry = old["seeds"][str(seed)]["roles"][role]
            if not entry["identity_token_exact"] or not entry["source_cell_exact"]:
                raise ValueError("old bank identity/source-cell contract missing")
            for task in ("income", "occupation"):
                actual = entry["identity_task"][task]
                for reference in ("U0", "Ucal"):
                    baseline = entry["references"][task][reference]
                    for metric, cap in (("accuracy", .010), ("nll", .010), ("brier", .005)):
                        cost = (baseline[metric] - actual[metric] if metric == "accuracy"
                                else actual[metric] - baseline[metric])
                        key = (seed, role, task, reference, metric)
                        expected[key] = (actual[metric], baseline[metric], cost, cap, cap - cost,
                                         "PASS" if cost <= cap else "FAIL")
    seen = set()
    maximum_difference = 0.0
    with PRODUCER.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["seed"]), row["role"], row["task"], row["reference"], row["metric"])
            if key not in expected or key in seen:
                raise ValueError("unexpected or repeated producer identity row")
            seen.add(key)
            for observed, target in zip((float(row[name]) for name in
                                          ("identity_value", "reference_value", "cost", "cap",
                                           "margin_cap_minus_cost")), expected[key][:5]):
                maximum_difference = max(maximum_difference, abs(observed - target))
            if row["point_status"] != expected[key][5]:
                raise ValueError("point gate classification differs")
    if seen != set(expected) or maximum_difference > 1e-12:
        raise ValueError("producer identity table does not match inherited aggregates")
    failures = {str(seed): sorted({f"{task}:{reference}:{metric}" for
                (s, role, task, reference, metric), item in expected.items()
                if s == seed and role == "OSF_DEFENSE_FIT" and item[5] == "FAIL"})
                for seed in (0, 1)}
    inner_failures = [key for key, item in expected.items()
                      if key[1] == "INNER_SELECTION" and item[5] == "FAIL"]
    receipt = {
        "schema": "pcrl-code-utility-r5-channel-public-aggregate-replay-v1",
        "source_aggregate_sha256": sha(SOURCE),
        "r2_identity_table_sha256": sha(PRODUCER),
        "aggregate_cells_recomputed": len(expected),
        "maximum_abs_difference": maximum_difference,
        "fitting_domain_failed_cells_by_seed": failures,
        "common_inner_failed_cell_count": len(inner_failures),
        "scope": "Historical verified public aggregates only; no bank arrays, channel checkpoint or person-level role input",
        "conclusion": "Identity misses old fitting-domain utility in occupation even though common INNER point utility passes; this is not a global channel infeasibility proof.",
    }
    OUTPUT.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
