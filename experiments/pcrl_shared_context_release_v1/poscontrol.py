"""Revealing-channel positive controls for the shared-context audit (inner only).

Wraps ``experiments/pcrl_adaptive_release_v1/positive_control.run_inner_diagnostic``
UNCHANGED, for the registered coalition roles AB/SEX and AB/RAC1P on every
anchor (0, 1, 2). The predecessor ran only a0 AB/RAC1P; this study runs all six
cells before the lock.

The control law is nondeployable and label-derived: token = S (one-hot on the
first 2 or 9 of the 17 tokens). It exists only to show that the independent
audit slate (standard slate, audit_fit fit / inner_selection route selection /
inner_check score, legal ignore-channel ancestors) *can* see a leak through the
17-token wire.

"Detected" (registered, identical to AR): on inner_check, the selected H-only
route loss minus the selected revealing-release route loss is >= 0.01 nats in
BOTH weightings (U = unweighted mean, PWGTP = person-weighted mean). 0.01 nats
is a descriptive audit-power threshold, not a claim margin. A slate that fails
to detect this maximal leak makes every "no recovery" statement for that
role/anchor uninformative (reported as such, not as survived).

Rows with a missing protected label for the target are excluded from all three
roles before fitting (the predecessor admission refuses negative codes); the
excluded count is recorded. Outer rows are never loaded.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import rd

ROLES = ("AB/SEX", "AB/RAC1P")
INNER = ("audit_fit", "inner_selection", "inner_check")
THRESHOLD_NATS = 0.01


def admitted_inner_roles(role_dict, target):
    """Three inner roles restricted to rows with an observed ``target`` label."""
    out, excluded = {}, {}
    for name in INNER:
        rows = role_dict[name]
        keep = rd.valid_mask(rows, target)
        excluded[name] = int((~keep).sum())
        out[name] = rd.subset_rows({k: rows[k] for k in ("ha", "hb", "weights", "ids", "households", "labels")},
                                   keep)
    return out, excluded


def run_control(anchor, role, out_dir, *, role_dict=None, smoke=False, slate="standard"):
    from experiments.pcrl_adaptive_release_v1 import positive_control

    if role not in ROLES and role not in ("A/SEX", "A/RAC1P"):
        raise ValueError("registered positive-control roles are AB/SEX and AB/RAC1P (A-view allowed as extra)")
    if role_dict is None:
        role_dict, _, _, _ = rd.load_roles(anchor, smoke=smoke)
    target = role.split("/")[1]
    inner, excluded = admitted_inner_roles(role_dict, target)
    out = Path(out_dir)
    private = out if "private" in out.parts else out / "private"
    unit = private / f"a{anchor}_{role.replace('/', '_')}_positive_control"
    t0 = time.perf_counter()
    result = positive_control.run_inner_diagnostic(inner, role=f"attack:{role}", anchor=anchor,
                                                   output_dir=unit, slate=slate)
    summary = {"schema": "pcrl-sc-positive-control-v1", "anchor": anchor, "role": role,
               "smoke": bool(smoke), "slate": slate,
               "improvement_nats": result["improvement_nats"],
               "threshold_nats": THRESHOLD_NATS,
               "detected": bool(all(v >= THRESHOLD_NATS for v in result["improvement_nats"].values())),
               "predecessor_detected_flag": result["detected_both_weightings"],
               "selected_routes": result["selected_routes"], "scores": result["scores"],
               "excluded_missing_label_rows": excluded,
               "people_scored": result["original_people_scored"],
               "wall_seconds": time.perf_counter() - t0,
               "unit_dir": str(unit), "nondeployable_label_oracle": True,
               "outer_labels_accessed": False}
    if summary["detected"] != summary["predecessor_detected_flag"]:
        raise AssertionError("detection rule drifted from the predecessor definition")
    rd.write_json(out / f"a{anchor}_{role.replace('/', '_')}_SUMMARY.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--anchor", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--role", choices=ROLES + tuple(f"attack:{r}" for r in ROLES) + ("all",), default="all")
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    rd._torch_threads()
    role_dict, _, _, _ = rd.load_roles(args.anchor, smoke=args.smoke)
    for role in (ROLES if args.role == "all" else (args.role.removeprefix("attack:"),)):
        s = run_control(args.anchor, role, args.out, role_dict=role_dict, smoke=args.smoke)
        print(f"a{args.anchor} {role}: improvement U={s['improvement_nats']['U']:.4f} "
              f"PWGTP={s['improvement_nats']['PWGTP']:.4f} detected={s['detected']} ({s['wall_seconds']:.1f}s)")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
