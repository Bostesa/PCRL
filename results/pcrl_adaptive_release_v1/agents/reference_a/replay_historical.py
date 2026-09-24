"""Read-only aggregate replay of the two closed P1 reference contradictions.

Example:
  python results/pcrl_adaptive_release_v1/agents/reference_a/replay_historical.py \
    --old-run-dir /path/to/completed/results/pcrl_task_aligned_cuts_v1/private/run \
    --output results/pcrl_adaptive_release_v1/agents/reference_a/HISTORICAL_REPLAY.json

No individual record, coefficient matrix, or private path is written to the
aggregate receipt.  Old bank and map bytes are hash-checked before loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.pcrl_adaptive_release_v1.reference import calibrate_reference


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replay(old_run_dir, index_path, certificate_path):
    root = Path(old_run_dir)
    index = json.loads(Path(index_path).read_text())
    certificate = json.loads(Path(certificate_path).read_text())
    anchors = {}
    for anchor in (1, 2):
        named = f"a{anchor}_reference_bank"
        bank_file = root / named / "bank.json"
        coeff_file = root / named / "coefficients.npz"
        pinned = certificate["anchors"][str(anchor)]
        if (digest(bank_file) != pinned["bank_manifest"]["sha256"] or
                digest(coeff_file) != pinned["coefficient_archive"]["sha256"]):
            raise ValueError("closed historical bank differs from its registered certificate")
        map_record = index["anchors"][str(anchor)]["maps"]["D17"]["Q.npz"]
        map_file = Path(map_record["path"])
        if digest(map_file) != map_record["sha256"]:
            raise ValueError("D17 map differs from pinned archived object")
        with np.load(map_file, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError("unexpected D17 archive schema")
            q_ref = archive[archive.files[0]].copy()
        bank = json.loads(bank_file.read_text())
        with np.load(coeff_file, allow_pickle=False) as archive:
            cuts = [{**item, "coeff": archive[f"cut_{i:04d}"].copy()}
                    for i, item in enumerate(bank["cuts"])]
        if len(cuts) != pinned["cut_count"]:
            raise ValueError("historical cut count differs from certificate")
        old_max = max(max(0., float(c["floor"] - np.sum(c["coeff"] * q_ref))) for c in cuts)
        calibrated = calibrate_reference(q_ref, cuts, 0.0)
        offending = pinned["largest_channel_invariant_H_cut"]
        old_cut = next(c for c in cuts if c["id"] == offending["id"])
        fixed_loss = float(np.sum(old_cut["coeff"] * q_ref))
        if not np.all(old_cut["coeff"] == old_cut["coeff"][:, :1]):
            raise ValueError("claimed historical H-only cut is not action invariant")
        if abs(old_cut["floor"] - fixed_loss - offending["floor_minus_loss_nats"]) > 1e-10:
            raise ValueError("historical H-only contradiction differs from certificate")
        anchors[str(anchor)] = {
            "closed_bank_sha256": bank["bank_sha256"],
            "closed_bank_file_sha256": digest(bank_file),
            "closed_coefficient_file_sha256": digest(coeff_file),
            "archived_D17_map_file_sha256": digest(map_file),
            "attack_cuts": len(cuts),
            "old_max_D17_cut_violation_nats": old_max,
            "old_H_only_single_cut_unavoidable_gap_nats": float(old_cut["floor"] - fixed_loss),
            "new_calibrated_D17_max_cut_violation_nats": calibrated["witness"]["maximum_cut_violation"],
            "new_bank_sha256": calibrated["bank_sha256"],
            "group_reference_risks": calibrated["rho"],
        }
    return {"schema": "pcrl-adaptive-reference-historical-replay-v1",
            "scope": "closed 2018 coefficient arrays and D17 maps; no person rows or new fit",
            "anchors": anchors,
            "interpretation": "Same-row min-bank rebasing makes D17 a constructive witness at delta >= 0; this does not retroactively change the closed study."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run-dir", required=True)
    parser.add_argument("--index", default="results/pcrl_task_aligned_cuts_v1/REUSABLE_INPUTS_PINNED.json")
    parser.add_argument("--certificate", default="results/pcrl_task_aligned_cuts_v1/CROSS_ANCHOR_INFEASIBILITY_CERTIFICATE.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = replay(args.old_run_dir, args.index, args.certificate)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "anchors": len(result["anchors"])}, sort_keys=True))


if __name__ == "__main__":
    main()
