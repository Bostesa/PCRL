"""Deployment for PN/LN (identity map) and reused units: permitted inputs only -> [features, centred logits] per recipient.
    ~/PCRL/.venv/bin/python -m pnx.deploy --unit pn__s0__PN__b1 --X permitted_inputs.npy --out release.npz
Uses jcv.deploy.release (no labels, no row keys, no clean outputs; exactly 83 permitted columns). PN/LN units carry no
LEACE directory, so the map is the identity; the function asserts that.
"""
import argparse

import numpy as np

from jcv import deploy as JD
from pnx import run as R


def release(name, X):
    d = R.udir(name)
    if name.startswith("pn__"):
        assert not any(d.glob("leace_*")), "a PN/LN unit must not carry an erasure map"
    return JD.release(d, X)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True)
    ap.add_argument("--X", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = release(a.unit, np.load(a.X))
    np.savez_compressed(a.out, **out)
    print(f"released {len(out['r1'])} rows from {a.unit} (EXPERIMENTAL)")


if __name__ == "__main__":
    main()
