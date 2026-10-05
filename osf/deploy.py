"""Deployment: the two recipient releases from PERMITTED inputs only (no labels, row keys or clean outputs).

    ~/PCRL/.venv/bin/python -m osf.deploy --unit <rel__s1__RAW-J_b0.3> --X permitted_inputs.npy --out release.npz

A neural unit is model.pt (encoders g_1, g_2; identity release map, no erasure) plus the deployed affine heads
(head_0.joblib, head_1.joblib). Returns per recipient r_i (16 features), centred logits c_i (computed from r_i only),
probabilities p_i and hard decisions. The input must have exactly the 83 permitted, preprocessed columns; anything else
(e.g. an appended protected attribute or a row key) is refused. LEACE units (lc__*) also load their official map.
The study label travels with the output (an EXPERIMENTAL label unless a development criterion was met). FARE
references are served by their own official encoder (osf.baselines), not by this CLI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from rgj import finalize as FN
from osf import run as R
from osf import train as T

N_PERMITTED = 83


def check_inputs(X):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != N_PERMITTED:
        raise ValueError(f"deployment accepts exactly the {N_PERMITTED} permitted input columns; got shape {X.shape}")
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError("permitted inputs must be the preprocessed float matrix")
    return X.astype(np.float32)


def release(unit_dir: Path, X):
    X = check_inputs(X)
    if not FN.unit_complete(unit_dir):
        raise RuntimeError(f"{unit_dir.name}: unit not hash-complete")
    rec = json.loads((unit_dir / "record.json").read_text())
    model = T.Model(N_PERMITTED, T.KS, rec["seed"])
    model.load_state_dict(torch.load(unit_dir / "model.pt"))
    maps = {}
    for i in (0, 1):
        if (unit_dir / f"leace_{i}").exists():
            from stored_model_eval.defenses import LeaceMap
            maps[i] = LeaceMap.load(unit_dir / f"leace_{i}", verify_package=True)
    out = {}
    with torch.no_grad():
        for i in (0, 1):
            H = model.encode(i, torch.from_numpy(X)).double().numpy()
            r = maps[i].transform(H) if i in maps else H
            c, P, hard = FN.outputs(joblib.load(unit_dir / f"head_{i}.joblib"), r)
            out.update({f"r{i + 1}": r, f"c{i + 1}": c, f"p{i + 1}": P, f"hard{i + 1}": hard})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True)
    ap.add_argument("--X", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="EXPERIMENTAL_NO_ADVANTAGE")
    a = ap.parse_args(argv)
    out = release(R.U(a.unit), np.load(a.X))
    np.savez_compressed(a.out, **out)
    print(f"released {len(out['r1'])} rows from {a.unit} ({a.label}): recipient views "
          f"{out['r1'].shape[1] + out['c1'].shape[1]} and {out['r2'].shape[1] + out['c2'].shape[1]} columns")


if __name__ == "__main__":
    main()
