"""Exact Team D checks on S1's five frozen symbolic synthetic laws.

This script performs no fitting, opens no Adult artifact, and writes only
aggregate synthetic quantities. It binds the source law file by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from prototype import FrozenLatentRelease, ProductVQ, PurposeNormalizer, SoftmaxHead


LAW_SHA256 = "b2e68f5dec7dd690afa9389c44c0c2f98a78b30cde25c20a48751665f35fdc53"
LAW_IDS = (
    "L1_FEASIBLE_NULL",
    "L2_REMOVABLE_DISTINCTION",
    "L3_COMPLEMENTARY_XOR",
    "L4_NO_COMPLEMENTARITY",
    "L5_CONFIDENCE_VS_CLASS",
)


def load_law_lock(path: Path) -> dict:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != LAW_SHA256:
        raise ValueError(f"frozen law SHA-256 mismatch: {digest}")
    law = json.loads(raw)
    if tuple(item["id"] for item in law["laws"]) != LAW_IDS:
        raise ValueError("frozen law IDs or order changed")
    return law


def enumerate_law(law_id: str) -> list[dict]:
    """Exact rational masses implementing the pinned law's declared draws."""
    rows = []
    if law_id in (LAW_IDS[0], LAW_IDS[1]):
        for s, b1, b2 in itertools.product((0, 1), repeat=3):
            rows.append(dict(mass=Fraction(1, 8), s=s, b1=b1, b2=b2))
    elif law_id == LAW_IDS[2]:
        for s, b1, b2, u in itertools.product((0, 1), repeat=4):
            rows.append(dict(mass=Fraction(1, 16), s=s, b1=b1, b2=b2,
                             u=u, v=u ^ s))
    elif law_id == LAW_IDS[3]:
        for s, b1, b2 in itertools.product((0, 1), repeat=3):
            for e, p_e in ((0, Fraction(4, 5)), (1, Fraction(1, 5))):
                rows.append(dict(mass=Fraction(1, 8) * p_e, s=s, b1=b1, b2=b2,
                                 e=e, c=s ^ e))
    elif law_id == LAW_IDS[4]:
        for h, y2 in itertools.product((0, 1), repeat=2):
            q = Fraction(3, 5) if h == 0 else Fraction(19, 20)
            for y1 in (0, 1):
                rows.append(dict(mass=Fraction(1, 4) * (q if y1 else 1 - q),
                                 s=h, h=h, y1=y1, y2=y2, q=q))
    else:
        raise ValueError("unknown frozen law")
    if not rows or sum((r["mass"] for r in rows), Fraction(0)) != 1:
        raise AssertionError(f"{law_id}: masses do not normalize")
    if any(r["mass"] < 0 for r in rows):
        raise AssertionError(f"{law_id}: negative mass")
    return rows


def mutual_information(rows: list[dict], views: list[tuple]) -> float:
    """I(S;view) in bits, from exact rational joint/marginal masses."""
    if len(rows) != len(views):
        raise ValueError("view and law rows disagree")
    joint = defaultdict(Fraction)
    ps = defaultdict(Fraction)
    pv = defaultdict(Fraction)
    for row, view in zip(rows, views):
        mass = row["mass"]
        key = (row["s"], view)
        joint[key] += mass
        ps[row["s"]] += mass
        pv[view] += mass
    return sum(float(p) * math.log2(float(p / (ps[s] * pv[v])))
               for (s, v), p in joint.items() if p > 0)


def _fixed_release(codebook: list[list[float]]) -> FrozenLatentRelease:
    centers = np.asarray(codebook, dtype=np.float64)[None, :, :]
    d = centers.shape[2]
    normalizer = PurposeNormalizer(np.zeros((2, d)), np.ones((2, d)))
    # For the two-coordinate laws, class decision equals the first coordinate.
    head = SoftmaxHead(np.array([[-1.0, 0.0], [1.0, 0.0]]), np.array([1.0, -1.0]))
    return FrozenLatentRelease(ProductVQ(centers, normalizer), (head, head))


def _complete_keys(release: dict) -> list[tuple]:
    """Use every transmitted array and fixed metadata in the symbolic view."""
    metadata = tuple(sorted(release["metadata"].items()))
    return [
        (tuple(map(int, release["code_ids"][i])),
         tuple(map(float, release["decoded"][i])),
         tuple(map(float, release["probabilities"][i])),
         int(release["decision"][i]), metadata)
        for i in range(len(release["decision"]))
    ]


def _emit(release: FrozenLatentRelease, h0: np.ndarray, h1: np.ndarray) -> tuple[list[tuple], list[tuple], dict, dict]:
    a, b = release.recipient(0, h0), release.recipient(1, h1)
    if a["metadata"]["purpose"] != 0 or b["metadata"]["purpose"] != 1:
        raise AssertionError("purpose metadata inconsistent")
    if set(a) != {"code_ids", "decoded", "probabilities", "decision", "metadata"} or set(b) != set(a):
        raise AssertionError("incomplete or extra release field")
    return _complete_keys(a), _complete_keys(b), a, b


def _nll(q: float, p: float) -> float:
    return -q * math.log(p) - (1 - q) * math.log(1 - p)


def run(law_path: Path) -> dict:
    load_law_lock(law_path)
    laws = {law_id: enumerate_law(law_id) for law_id in LAW_IDS}
    metrics = {}
    checks = {}

    l1 = laws[LAW_IDS[0]]
    null_info = [mutual_information(l1, [(r["b1"],) for r in l1]),
                 mutual_information(l1, [(r["b2"],) for r in l1]),
                 mutual_information(l1, [(r["b1"], r["b2"]) for r in l1])]
    metrics[LAW_IDS[0]] = {"local0_bits": null_info[0], "local1_bits": null_info[1],
                           "coalition_bits": null_info[2]}
    checks["L1_null_floor"] = all(abs(x) < 1e-12 for x in null_info)

    drop_clue = _fixed_release([[0, 0], [2, 0]])
    l2 = laws[LAW_IDS[1]]
    h0 = np.array([[2 * r["b1"], 2 * r["s"]] for r in l2], dtype=float)
    h1 = np.array([[2 * r["b2"], 2 * r["s"]] for r in l2], dtype=float)
    v0, v1, a, b = _emit(drop_clue, h0, h1)
    before_l2 = mutual_information(l2, [(r["b1"], r["s"], r["b2"]) for r in l2])
    after_l2 = mutual_information(l2, list(zip(v0, v1)))
    accuracy_l2 = (float(np.mean(a["decision"] == [r["b1"] for r in l2])),
                   float(np.mean(b["decision"] == [r["b2"] for r in l2])))
    metrics[LAW_IDS[1]] = {"raw_coalition_bits": before_l2, "repaired_full_view_bits": after_l2,
                           "task_accuracies": accuracy_l2}
    checks["L2_removal_preserves_tasks"] = (abs(before_l2 - 1) < 1e-12 and
                                            abs(after_l2) < 1e-12 and accuracy_l2 == (1.0, 1.0))

    l3 = laws[LAW_IDS[2]]
    preserve_clue = _fixed_release([[0, 0], [0, 2], [2, 0], [2, 2]])
    h0 = np.array([[2 * r["b1"], 2 * r["u"]] for r in l3], dtype=float)
    h1 = np.array([[2 * r["b2"], 2 * r["v"]] for r in l3], dtype=float)
    v0, v1, _, _ = _emit(preserve_clue, h0, h1)
    l3_local = (mutual_information(l3, v0), mutual_information(l3, v1))
    l3_pair = mutual_information(l3, list(zip(v0, v1)))
    dropped0, dropped1, da, db = _emit(drop_clue, h0, h1)
    l3_dropped_pair = mutual_information(l3, list(zip(dropped0, dropped1)))
    l3_task = (float(np.mean(da["decision"] == [r["b1"] for r in l3])),
               float(np.mean(db["decision"] == [r["b2"] for r in l3])))
    metrics[LAW_IDS[2]] = {"retained_local_bits": l3_local,
                           "retained_pair_bits": l3_pair,
                           "clue_dropped_pair_bits": l3_dropped_pair,
                           "clue_dropped_task_accuracies": l3_task}
    checks["L3_complementarity"] = (all(abs(x) < 1e-12 for x in l3_local)
                                     and abs(l3_pair - 1) < 1e-12
                                     and abs(l3_dropped_pair) < 1e-12
                                     and l3_task == (1.0, 1.0))

    l4 = laws[LAW_IDS[3]]
    h0 = np.array([[2 * r["b1"], 2 * r["c"]] for r in l4], dtype=float)
    h1 = np.array([[2 * r["b2"], 2 * r["c"]] for r in l4], dtype=float)
    v0, v1, _, _ = _emit(preserve_clue, h0, h1)
    l4_local = (mutual_information(l4, v0), mutual_information(l4, v1))
    l4_pair = mutual_information(l4, list(zip(v0, v1)))
    expected_l4 = 1 + Fraction(4, 5) * math.log2(Fraction(4, 5)) + Fraction(1, 5) * math.log2(Fraction(1, 5))
    metrics[LAW_IDS[3]] = {"local_bits": l4_local, "pair_bits": l4_pair,
                           "binary_symmetric_channel_bits": expected_l4}
    checks["L4_no_complementarity"] = (all(abs(x - expected_l4) < 1e-12 for x in l4_local)
                                        and abs(l4_pair - expected_l4) < 1e-12)

    l5 = laws[LAW_IDS[4]]
    qbar = Fraction(31, 40)  # (3/5 + 19/20) / 2 = 0.775
    fine_nll = sum(float(r["mass"]) * _nll(float(r["q"]), float(r["q"])) for r in l5)
    class_nll = sum(float(r["mass"]) * _nll(float(r["q"]), float(qbar)) for r in l5)
    # Two-class Brier excess conditional on the true q is 2*(p-q)^2.
    brier_excess = sum(float(r["mass"]) * 2 * (float(qbar - r["q"])) ** 2 for r in l5)
    fine_info = mutual_information(l5, [(float(r["q"]),) for r in l5])
    class_info = mutual_information(l5, [(float(qbar),) for _ in l5])
    metrics[LAW_IDS[4]] = {"fine_nll": fine_nll, "class_only_nll": class_nll,
                           "class_only_nll_excess": class_nll - fine_nll,
                           "class_only_brier_excess": brier_excess,
                           "fine_confidence_sex_bits": fine_info,
                           "class_only_sex_bits": class_info,
                           "same_decision_expected_accuracy": float(qbar)}
    checks["L5_confidence_cost"] = (class_nll > fine_nll and brier_excess > 0
                                    and abs(fine_info - 1) < 1e-12
                                    and abs(class_info) < 1e-12)

    return {"schema": "team-d-locked-law-check-v1", "law_sha256": LAW_SHA256,
            "checks": checks, "all_pass": all(checks.values()), "metrics": metrics,
            "scope": "exact symbolic and hand-coded-code capability checks; no fitted model or Adult result"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--laws", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.laws)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        print(output, end="")
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
