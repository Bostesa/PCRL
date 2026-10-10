"""Receipt-only builder for independent FIT nomination verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .r5_selection_math import V_ARMS


GATE_FILE = {
    "P0": "P0_TWO_SEED_INNER_GATE.json",
    "O_J": "O_J_TWO_SEED_INNER_GATE_V2.json",
    **{a: f"{a}_TWO_SEED_INNER_GATE.json" for a in V_ARMS},
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path, value, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode), "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return sha(path)


def build(repo, amendment):
    repo = Path(repo).resolve(strict=True)
    amendment = Path(amendment).resolve(strict=True)
    root = repo / "results/pcrl_incumbent_view_risk_v1"
    r3path = root / "r3/R3_INTERNAL_READER_NOMINATION_V1.json"
    r3 = json.loads(r3path.read_text())
    if r3.get("schema") != "pcrl-incumbent-r3-internal-nomination-v1":
        raise PermissionError("FIT-only R3 nomination source differs")
    r2_receipts = {}
    for arm in V_ARMS:
        for seed in (0, 1):
            key = f"{arm}:s{seed}"
            path = root / "r2" / f"R2_{arm}_S{seed}_FIT_UNIT_RECEIPT.json"
            if r3["source_r2_units"][key]["receipt_sha256"] != sha(path):
                raise PermissionError("R2 FIT source not bound to nomination")
            r2_receipts[key] = {"path": str(path), "sha256": sha(path)}
    gates = {}
    for arm, filename in GATE_FILE.items():
        path = root / "r3" / filename
        if r3["source_gate_sha256"].get(arm) != sha(path):
            raise PermissionError("point gate not bound to nomination")
        gates[arm] = {"path": str(path), "sha256": sha(path)}
    admission = root / "r0/ADMISSION_LOCK.json"
    science = root / "r0/CAMPAIGN_LOCK.json"
    unit = "R5_R3_NOMINATION_REPLAY_V1"
    output = root / "r5" / f"{unit}.json"
    cfgpath = root / "r5/.private" / f"{unit}_CONFIG.json"
    cfg = {"schema": "IVR1-r5-nomination-replay-config-v1", "unit_id": unit,
           "r3_nomination": {"path": str(r3path), "sha256": sha(r3path)},
           "r2_receipts": r2_receipts, "gates": gates,
           "queue_seal": {"path": str(admission), "sha256": sha(admission)},
           "campaign_seal": {"path": str(science), "sha256": sha(science)},
           "verification_amendment": {"path": str(amendment), "sha256": sha(amendment)},
           "output": str(output)}
    cfgsha = write_once(cfgpath, cfg, 0o600)
    paths = [cfgpath, r3path, admission, science, amendment,
             repo / "incumbent_view/r5_nomination_replay.py",
             repo / "incumbent_view/r5_selection_math.py",
             repo / "incumbent_view/r5_attack_math.py",
             repo / "incumbent_view/r5_build_nomination_spec.py",
             root / "r0/compute_queue.py",
             *(Path(x["path"]) for x in r2_receipts.values()),
             *(Path(x["path"]) for x in gates.values())]
    sources = {str(path.resolve(strict=True)): sha(path) for path in paths}
    venv = Path(os.path.abspath(repo.parents[1] / ".venv/bin/python"))
    if not venv.is_file() or not os.access(venv, os.X_OK):
        raise FileNotFoundError("scientific venv entrypoint unavailable")
    spec = {"unit_id": unit, "stage": "verification", "cwd": str(repo),
            "argv": [str(venv), "-m", "incumbent_view.r5_nomination_replay",
                     "--config", str(cfgpath)],
            "max_wall_s": 60, "cpu_reserve_s": 60, "max_rss_gib": 1.0,
            "projected_output_bytes": 100000, "outputs": [str(output)],
            "deps": ["R3_INTERNAL_READER_NOMINATION_V1"],
            "sources": [{"path": path, "sha256": digest} for path, digest in sorted(sources.items())],
            "env": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "amendment_sha256": sha(amendment)}
    specpath = root / "r5/.private" / f"{unit}_QUEUE_SPEC.json"
    specsha = write_once(specpath, spec, 0o600)
    index = {"schema": "IVR1-r5-nomination-spec-index-v1", "unit_id": unit,
             "source_count": len(sources), "config_sha256": cfgsha,
             "spec_sha256": specsha, "amendment_sha256": sha(amendment),
             "private_spec_relative": str(specpath.relative_to(repo)),
             "role_arrays_deserialized_by_builder": False}
    write_once(root / "r5" / f"{unit}_SPEC_INDEX.json", index, 0o644)
    return index


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--amendment", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.amendment), sort_keys=True))


if __name__ == "__main__":
    main()
