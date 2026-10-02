"""Build the CELL-A run_v1 inputs (NON-FITTING step): task_labels_v1.npz + 26 v2 manifests.

Reads the PCRL Adult test split through the archived PCRL b96c412 dataset object (exactly as the preparation's
prepare_pilot_adult_s0.py did), takes `task_labels` (income, occupation_group, education_level) and recomputes the
record keys and sensitive attributes as reference columns. stored_model_eval.pilot_inputs.build_inputs then joins
them to the untouched labels.npz BY row_id and verifies every reference column row by row before writing anything
new under ~/PCRL_eval_cache_private/pilot_adult_s0/run_v1/inputs/. labels.npz and the 152 original manifests are
never modified. No model is fitted; no assessment outcome is computed.

Run: /Users/nathansamson/PCRL/.venv/bin/python results/combined_stored_model_pilot_v1/scripts/build_inputs_v1.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from stored_model_eval.guards import install_network_guard  # noqa: E402
from stored_model_eval.pilot import resolve_worktree  # noqa: E402
from stored_model_eval.pilot_inputs import build_inputs  # noqa: E402

install_network_guard()
PCRL_REPO = Path("/Users/nathansamson/PCRL")
EXPORT = Path("/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/"
              "pcrl_b96c412")
ORIG = Path.home() / "PCRL_eval_cache_private" / "pilot_adult_s0"
OUT = ORIG / "run_v1" / "inputs"
TASKS = {"income_prediction": "income", "employment_analysis": "occupation_group",
         "education_assessment": "education_level"}


def ensure_export(path: Path) -> Path:
    if not (path / "pcrl" / "data" / "adult.py").exists():
        path.mkdir(parents=True, exist_ok=True)
        arc = subprocess.run(["git", "-C", str(PCRL_REPO), "archive", "b96c412", "pcrl"], check=True,
                             capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(path)], input=arc, check=True)
    return path


def main():
    resolve_worktree(None)  # refuses unless this worktree is on the pilot branch
    exp = ensure_export(EXPORT)
    sys.path.insert(0, str(exp))
    from pcrl.data.adult import AdultDataset, get_adult_purposes
    root = PCRL_REPO / "data"
    for f in ("adult.data", "adult.test"):
        if not (root / "adult" / f).exists():
            raise SystemExit(f"missing {f}: the historical loader would silently synthesise data; refusing")
    purposes = get_adult_purposes()
    train = AdultDataset(purposes=purposes, root=str(root), split="train", download=False)
    test = AdultDataset(purposes=purposes, root=str(root), split="test", download=False, norm_stats=train.norm_stats)
    n = len(test.features)
    rec = np.array([hashlib.sha256("|".join(map(str, r)).encode()).hexdigest()[:20]
                    for r in test.raw_df.itertuples(index=False)])
    source = {"row_id": np.arange(n, dtype=np.int64), "ref_record_key": rec}
    for p, task in TASKS.items():
        source[f"y_task_{p}"] = test.task_labels[task].numpy().astype(np.int64)
    for k, v in test.sensitive_attrs.items():
        source[f"ref_{k}"] = v.numpy().astype(np.int64)
    fman = json.loads((ORIG / "forward_manifest.json").read_text())
    ck = fman["files"]["checkpoint"]
    idx = build_inputs(ORIG, OUT, source, {"path": ck["path"], "sha256": ck["sha256"]})
    print(json.dumps({"inputs_dir": str(OUT), "n_units": len(idx["units"]), "task_labels": idx["task_labels"],
                      "alignment": idx["alignment"], "labels_npz_rewritten": idx["labels_npz"]["rewritten"],
                      "fits_performed": 0}, indent=1))


if __name__ == "__main__":
    main()
