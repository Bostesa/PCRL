"""Planned unit list (frozen in LOCK.json): python -m cap.plan  -> UNITS_PLANNED.json + COVERAGE_AND_UNITS.csv"""
import csv
import json
from pathlib import Path

from cap.run import ARMS, uid

PKG = Path(__file__).resolve().parents[1] / "results" / "combined_analysis_paper_v1"


def plan():
    rows = []
    for k in (0, 1, 2):
        for arm, tag in ARMS.items():
            base = {"dataset": "adult", "purpose": "income_prediction", "attribute": "sex", "encoder_round": "Round-4", "seed": k, "arm": arm}
            rows += [
                {**base, "unit": uid(k, arm, "feat", "none"), "kind": "alias", "view": "features-only", "surface": "-", "source": f"output-aware {tag} features-only unit"},
                {**base, "unit": uid(k, arm, "out", "full"), "kind": "alias", "view": "output-only", "surface": "full", "source": f"output-aware O_head{tag}"},
                {**base, "unit": uid(k, arm, "feat+out", "full"), "kind": "alias", "view": "features+own-output", "surface": "full", "source": f"output-aware {tag}__rep+head"},
                {**base, "unit": uid(k, arm, "feat+clean", "full"), "kind": "alias", "view": "features+historical-clean-output (adverse control)", "surface": "full", "source": f"output-aware {tag}__rep+clean"}]
            for sname in ("dc", "centred", "prob", "hard"):
                rows.append({**base, "unit": uid(k, arm, "out", sname), "kind": "attack (new)", "view": "output-only", "surface": sname, "source": f"surfaces of saved HEAD__{tag} log-probabilities"})
            for sname in ("centred", "prob", "hard"):
                rows.append({**base, "unit": uid(k, arm, "feat+out", sname), "kind": "attack (new, plus rule)", "view": "features+own-output", "surface": sname, "source": f"{arm} features + saved HEAD__{tag} surface"})
            for view, b in (("out", "iobank"), ("out", "fullbank"), ("feat+out", "iobank"), ("feat+out", "fullbank")):
                rows.append({**base, "unit": uid(k, arm, view, b), "kind": "bank (validation selection, no fit)", "view": {"out": "output-only", "feat+out": "features+own-output"}[view], "surface": b, "source": "attacker_val log loss"})
    for name in ("A__out__centred", "F__out__hard", "F__feat+out__hard", "B__out__prob"):
        rows.append({"dataset": "adult", "purpose": "income_prediction", "attribute": "sex", "encoder_round": "Round-4", "seed": 0, "arm": name.split("__")[0],
                     "unit": f"cap__CTL__{name}", "kind": "control (null + planted, fit/val only)", "view": name.split("__")[1], "surface": name.split("__")[2], "source": "attacker_fit / attacker_val only"})
    return rows


if __name__ == "__main__":
    rows = plan()
    (PKG / "UNITS_PLANNED.json").write_text(json.dumps([r["unit"] for r in rows], indent=1) + "\n")
    with open(PKG / "COVERAGE_AND_UNITS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["status"], lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "status": "planned"})
    from collections import Counter
    print(len(rows), Counter(r["kind"] for r in rows))
