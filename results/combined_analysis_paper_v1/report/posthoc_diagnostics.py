"""POST HOC, descriptive only (not in any registered family): (1) linear-attacker recovery on output-only centred
releases (does the LEACE head's score carry sex only nonlinearly?); (2) R(A, hard) - R(F, centred) and the matching
accuracy contrast. 90% normal intervals from the same paired bootstrap.
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/combined_analysis_paper_v1/report/posthoc_diagnostics.py"""
import json, sys
from pathlib import Path
import numpy as np
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
import cap.infer as CI
import odx.infer as I
from cap import family as F
from cap.run import uid
W, a, maj = CI.world()
D = I.DS(W["row_id"][a], W["unit"][a])
t = W["t"][a]
ids = {}
for arm in ("A", "B", "F", "F0"):
    ids[f"R_L|{arm}|out|centred"] = D.mean(f"L{arm}", [D.recovery(uid(k, arm, "out", "centred"), CI.CLASSES, recipe="L") for k in (0, 1, 2)])
    ids[f"R_NL|{arm}|out|centred"] = D.mean(f"N{arm}", [D.recovery(uid(k, arm, "out", "centred"), CI.CLASSES) for k in (0, 1, 2)])
rAh = D.mean("rah", [D.recovery(uid(k, "A", "out", "hard"), CI.CLASSES) for k in (0, 1, 2)])
rFc = ids["R_NL|F|out|centred"]
ids["R(A,hard) - R(F,centred)"] = D.diff("d1", rAh, rFc)
cor = {(k, arm): (CI.head_probs(k, arm, W, a).argmax(1) == t).astype(float) for k in (0, 1, 2) for arm in ("A", "F")}
ids["Acc(A) - Acc(F)"] = D.mean("d2", [D.accuracy(f"dacc{k}", cor[k, "A"] - cor[k, "F"]) for k in (0, 1, 2)])
est = I.estimate(D, list(ids.values()), F.B_SE, F.SEED_SE, 1.6448536269514722)
out = {k: {"point": est[v]["point"], "lower90": est[v]["lower"], "upper90": est[v]["upper"]} for k, v in ids.items()}
out["_status"] = "POST HOC, descriptive; not part of the registered families"
(Path(__file__).parent / "posthoc_diagnostics.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
