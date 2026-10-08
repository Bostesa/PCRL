"""Independent exact premise checks for S1's frozen five finite laws.

This is an analytic law checker, not a Team B neural-fit result. It reads only
the declared JSON law specification and never opens Adult data. Run under the
global numerical queue with the frozen file path and SHA-256 as arguments.
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


LAW_IDS = (
    "L1_FEASIBLE_NULL",
    "L2_REMOVABLE_DISTINCTION",
    "L3_COMPLEMENTARY_XOR",
    "L4_NO_COMPLEMENTARITY",
    "L5_CONFIDENCE_VS_CLASS",
)
EXPECTED_FORMULAS = {
    LAW_IDS[0]: {
        "latent": "S,B1,B2 independent Bernoulli(1/2)",
        "x1": "B1", "x2": "B2", "y1": "B1", "y2": "B2",
        "score1": "0.9 if B1=1 else 0.1",
        "score2": "0.9 if B2=1 else 0.1",
    },
    LAW_IDS[1]: {
        "latent": "S,B1,B2 independent Bernoulli(1/2)",
        "x1": "(B1,S)", "x2": "(B2,S)", "y1": "B1", "y2": "B2",
        "score1": "0.80+0.10*S if B1=1 else 0.20-0.10*S",
        "score2": "0.80+0.10*S if B2=1 else 0.20-0.10*S",
    },
    LAW_IDS[2]: {
        "latent": "S,B1,B2,U independent Bernoulli(1/2)",
        "x1": "(B1,U)", "x2": "(B2,U XOR S)", "y1": "B1", "y2": "B2",
        "score1": "0.80+0.10*U if B1=1 else 0.20-0.10*U",
        "score2": "0.80+0.10*(U XOR S) if B2=1 else 0.20-0.10*(U XOR S)",
    },
    LAW_IDS[3]: {
        "latent": "S,B1,B2 independent Bernoulli(1/2); E independent Bernoulli(1/5); C=S XOR E",
        "x1": "(B1,C)", "x2": "(B2,C)", "y1": "B1", "y2": "B2",
        "score1": "0.80+0.10*C if B1=1 else 0.20-0.10*C",
        "score2": "0.80+0.10*C if B2=1 else 0.20-0.10*C",
    },
    LAW_IDS[4]: {
        "latent": "H independent Bernoulli(1/2); S=H; Y1 conditional Bernoulli(0.60 if H=0 else 0.95); Y2 independent Bernoulli(1/2)",
        "x1": "H", "x2": "Y2", "y1": "Y1", "y2": "Y2",
        "score1": "0.60 if H=0 else 0.95",
        "score2": "0.90 if Y2=1 else 0.10",
    },
}


def worlds(law_id: str) -> list[tuple[Fraction, dict[str, int]]]:
    """Enumerate the variables and masses stated in S1's fixed law file."""
    result = []
    if law_id in LAW_IDS[:3]:
        fields = ("S", "B1", "B2", "U") if law_id == LAW_IDS[2] else ("S", "B1", "B2")
        for values in itertools.product((0, 1), repeat=len(fields)):
            result.append((Fraction(1, 2 ** len(fields)), dict(zip(fields, values))))
    elif law_id == LAW_IDS[3]:
        for s, b1, b2, e in itertools.product((0, 1), repeat=4):
            mass = Fraction(1, 8) * (Fraction(1, 5) if e else Fraction(4, 5))
            result.append((mass, {"S": s, "B1": b1, "B2": b2, "E": e, "C": s ^ e}))
    elif law_id == LAW_IDS[4]:
        for h, y1, y2 in itertools.product((0, 1), repeat=3):
            q = Fraction(3, 5) if h == 0 else Fraction(19, 20)
            mass = Fraction(1, 4) * (q if y1 else 1 - q)
            result.append((mass, {"H": h, "S": h, "Y1": y1, "Y2": y2}))
    else:
        raise ValueError(f"Unknown frozen law {law_id}")
    if any(mass < 0 for mass, _ in result) or sum((mass for mass, _ in result), Fraction(0)) != 1:
        raise AssertionError(f"Invalid finite probability law: {law_id}")
    return result


def is_independent_of_sex(entries, view) -> bool:
    joint, sex_mass, view_mass = defaultdict(Fraction), defaultdict(Fraction), defaultdict(Fraction)
    for mass, world in entries:
        s, v = world["S"], view(world)
        joint[s, v] += mass
        sex_mass[s] += mass
        view_mass[v] += mass
    return all(joint[s, v] == sex_mass[s] * view_mass[v]
               for s in sex_mass for v in view_mass)


def source_score(law_id: str, world: dict[str, int], purpose: int) -> Fraction:
    bit = world[f"B{purpose}"]
    if law_id == LAW_IDS[0]:
        return Fraction(9, 10) if bit else Fraction(1, 10)
    if law_id == LAW_IDS[1]:
        clue = world["S"]
    elif law_id == LAW_IDS[2]:
        clue = world["U"] if purpose == 1 else world["U"] ^ world["S"]
    elif law_id == LAW_IDS[3]:
        clue = world["C"]
    else:
        raise ValueError(law_id)
    return (Fraction(4, 5) + Fraction(1, 10) * clue) if bit else (
        Fraction(1, 5) - Fraction(1, 10) * clue
    )


def check_laws(path: Path, expected_sha256: str) -> dict:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Frozen S1 synthetic-law SHA-256 mismatch")
    spec = json.loads(raw)
    if spec["schema"] != "pcrl-method-portfolio-synthetic-laws-v1" or tuple(
        law["id"] for law in spec["laws"]
    ) != LAW_IDS:
        raise ValueError("Unexpected frozen synthetic-law schema or ordering")
    for law in spec["laws"]:
        if any(law.get(field) != formula for field, formula in EXPECTED_FORMULAS[law["id"]].items()):
            raise ValueError(f"Frozen law formula differs from enumerator: {law['id']}")
    enumerated = {law_id: worlds(law_id) for law_id in LAW_IDS}
    for law_id in LAW_IDS[:4]:
        for _, world in enumerated[law_id]:
            for purpose in (1, 2):
                score = source_score(law_id, world, purpose)
                assert 0 < score < 1
                assert int(score > Fraction(1, 2)) == world[f"B{purpose}"]

    null = enumerated[LAW_IDS[0]]
    assert is_independent_of_sex(null, lambda w: (w["B1"],))
    assert is_independent_of_sex(null, lambda w: (w["B2"],))
    assert is_independent_of_sex(null, lambda w: (w["B1"], w["B2"]))

    removable = enumerated[LAW_IDS[1]]
    assert not is_independent_of_sex(removable, lambda w: (w["B1"], w["S"]))
    assert not is_independent_of_sex(removable, lambda w: (w["B2"], w["S"]))
    assert is_independent_of_sex(removable, lambda w: (w["B1"],))
    assert is_independent_of_sex(removable, lambda w: (w["B2"],))
    assert is_independent_of_sex(removable, lambda w: (w["B1"], w["B2"]))

    complementary = enumerated[LAW_IDS[2]]
    view1 = lambda w: (w["B1"], w["U"])
    view2 = lambda w: (w["B2"], w["U"] ^ w["S"])
    assert is_independent_of_sex(complementary, view1)
    assert is_independent_of_sex(complementary, view2)
    assert all((view1(w)[1] ^ view2(w)[1]) == w["S"] for _, w in complementary)

    redundant = enumerated[LAW_IDS[3]]
    posterior_by_c = defaultdict(Fraction)
    mass_by_c = defaultdict(Fraction)
    posterior_by_pair = defaultdict(Fraction)
    mass_by_pair = defaultdict(Fraction)
    for mass, world in redundant:
        c = world["C"]
        pair = (world["B1"], c, world["B2"], c)
        posterior_by_c[c, world["S"]] += mass
        mass_by_c[c] += mass
        posterior_by_pair[pair, world["S"]] += mass
        mass_by_pair[pair] += mass
    assert all(posterior_by_pair[pair, s] / mass_by_pair[pair]
               == posterior_by_c[pair[1], s] / mass_by_c[pair[1]]
               for pair in mass_by_pair for s in (0, 1))

    confidence = enumerated[LAW_IDS[4]]
    p_class_only = Fraction(31, 40)  # E[Y1] = (.60 + .95)/2 = .775
    nll_precise = nll_class = 0.0
    brier_precise = brier_class = Fraction(0)
    for mass, world in confidence:
        p_precise = Fraction(3, 5) if world["H"] == 0 else Fraction(19, 20)
        y = world["Y1"]
        nll_precise += float(mass) * -math.log(float(p_precise if y else 1 - p_precise))
        nll_class += float(mass) * -math.log(float(p_class_only if y else 1 - p_class_only))
        brier_precise += mass * 2 * (p_precise - y) ** 2
        brier_class += mass * 2 * (p_class_only - y) ** 2
    assert nll_precise < nll_class and brier_precise < brier_class
    assert all((Fraction(3, 5) if world["H"] == 0 else Fraction(19, 20)) > Fraction(1, 2)
               for _, world in confidence)
    assert is_independent_of_sex(confidence, lambda w: (1,))

    return {
        "status": "PASS",
        "source_law_sha256": digest,
        "checked_laws": list(LAW_IDS),
        "scope": "exact finite-law premises; no fitted mechanism or Adult evidence",
        "confidence_nll_advantage": nll_class - nll_precise,
        "confidence_multiclass_brier_advantage": float(brier_class - brier_precise),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--law-file", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(check_laws(args.law_file, args.expected_sha256), sort_keys=True, indent=2))
