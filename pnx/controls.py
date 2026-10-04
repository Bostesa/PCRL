"""Real-data audit controls on attacker_fit / attacker_val only (no assessment rows): shuffled-label null (<= 0.55)
and planted leak (> 0.75) on the local and coalition views of the selected PN and LN releases and of U (seed 0).
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m pnx.controls --selection-lock <SELECTION_LOCK.json>"""
import json
from pathlib import Path

from jcv import controls as JC
from pnx import outer as PO
from pnx import run as R


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-lock", required=True)
    a = ap.parse_args(argv)
    SL = json.loads(Path(a.selection_lock).read_text())
    D = R.load_D()
    arms = SL["seeds"]["0"]["arms"]
    out = {}
    for arm in ("PN", "LN", "U"):
        spec = {"unit": arms[arm]["unit"]} if arm != "U" else {"unit": "nn__s0__U"}
        V, _, finite, _ = PO.release_for(spec)
        for w in ("v1", "v2", "pair"):
            out[f"{arm}({spec['unit']})/{w}"] = JC.control(V[w], D["sex"], D, finite)
    (R.RUN / "controls.json").write_text(json.dumps(out, indent=1, default=bool))
    R.PKG.mkdir(parents=True, exist_ok=True)
    (R.PKG / "AUDIT_CONTROLS.json").write_text(json.dumps(out, indent=1, default=bool))
    return out


if __name__ == "__main__":
    main()
