"""Deployment: release the two recipient views from PERMITTED inputs only (no labels, no row keys, no clean outputs).

    ~/PCRL/.venv/bin/python -m jcv.deploy --unit nn__s0__J__b1 --X permitted_inputs.npy --out release.npz

Neural units: g_i (model.pt) -> official LEACE map (leace_i/, float64) -> deployed affine head (head_i.joblib);
returns r_i, centred logits c_i, probabilities p_i, hard decisions. FARE units need the FARE environment (oar wrapper).
Traps: the input must have exactly the admitted permitted columns (83); a matrix with extra columns (e.g. an appended
protected attribute or a row key) is refused; nothing else is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from jcv import finalize as FN
from jcv import run as R
from jcv import train as T

N_PERMITTED = 83


def check_inputs(X):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != N_PERMITTED:
        raise ValueError(f"deployment accepts exactly the {N_PERMITTED} permitted input columns; got shape {X.shape}")
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError("permitted inputs must be the preprocessed float matrix")
    return X.astype(np.float32)


def load_neural(unit_dir: Path):
    from stored_model_eval.defenses import LeaceMap
    if not FN.unit_complete(unit_dir):
        raise RuntimeError(f"{unit_dir.name}: unit not hash-complete")
    rec = json.loads((unit_dir / "record.json").read_text())
    model = T.Model(N_PERMITTED, R.KS, rec["seed"])
    model.load_state_dict(torch.load(unit_dir / "model.pt"))
    maps = {i: LeaceMap.load(unit_dir / f"leace_{i}", verify_package=True) for i in (0, 1) if (unit_dir / f"leace_{i}").exists()}
    heads = {i: joblib.load(unit_dir / f"head_{i}.joblib") for i in (0, 1)}
    return model, maps, heads, rec


def release(unit_dir: Path, X):
    X = check_inputs(X)
    model, maps, heads, rec = load_neural(unit_dir)
    out = {}
    with torch.no_grad():
        for i in (0, 1):
            H = model.encode(i, torch.from_numpy(X)).double().numpy()
            r = maps[i].transform(H) if i in maps else H
            c, P, hard = FN.outputs(heads[i], r)
            out.update({f"r{i + 1}": r, f"c{i + 1}": c, f"p{i + 1}": P, f"hard{i + 1}": hard})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True)
    ap.add_argument("--X", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = release(R.U(a.unit), np.load(a.X))
    np.savez_compressed(a.out, **out)
    print(f"released {len(out['r1'])} rows: recipient 1 view dim {out['r1'].shape[1] + out['c1'].shape[1]}, "
          f"recipient 2 view dim {out['r2'].shape[1] + out['c2'].shape[1]}")


if __name__ == "__main__":
    main()
