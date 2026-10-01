"""ORIGINAL-IMPLEMENTATION comparison step for F1, F2, F3, F9 (separate from the
independent fixtures). Run with the PCRL venv (torch) against a READ-ONLY export:

  git -C /Users/nathansamson/PCRL archive <ref> pcrl | tar -x -C <export_root>
  /Users/nathansamson/PCRL/.venv/bin/python orig_compare.py --root <export_root> --label <ref>

The synthetic data come from the independent fixture modules (numpy only); their
sha256 prefixes are printed so they can be matched against outputs/F0*.json.
No repo code is modified.
"""
import argparse
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import F01_onehot_identity as F1  # noqa: E402
import F02_multiclass_contrast as F2  # noqa: E402
import F03_unsupported_classes as F3  # noqa: E402
import F09_accuracy_guarantee as F9  # noqa: E402


def clean(x):
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating,)):
        return clean(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def safe(fn):
    try:
        return fn()
    except Exception as e:  # report, never hide
        return {"exception": type(e).__name__, "message": str(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.root)
    from pcrl.evaluation.certificates import compute_dominant_axis_r2 as DA
    from pcrl.purposes import verification as V
    LCC = V.LinearComplianceCertificate
    has_nc = "num_classes" in inspect.signature(LCC.check).parameters
    da_has_nc = "num_classes" in inspect.signature(DA).parameters
    out = {"label": a.label, "root": a.root, "module_files": {"verification": V.__file__},
           "check_accepts_num_classes": has_nc, "dominant_axis_accepts_num_classes": da_has_nc}

    def cert(H, y, K=None):
        kw = {"num_classes": K} if (has_nc and K is not None) else {}
        r = LCC(epsilon=0.05, regularization=1e-6).check(H, y, **kw)
        return {k: getattr(r, k) for k in ("r_squared", "certified", "class_support", "valid_mask",
                                           "coverage_complete", "score_defined") if hasattr(r, k)}

    def da(H, y, K=None):
        kw = {"num_classes": K} if (da_has_nc and K is not None) else {}
        return DA(H, y, **kw)

    # ---- F1
    H, y = F1.make_data()
    d = da(H, y)
    pc = np.asarray(d["per_class_r2"], float)
    pi = np.bincount(y, minlength=4) / len(y)
    w = pi * (1 - pi) / (pi * (1 - pi)).sum()
    c64 = cert(H, y)
    c32 = cert(H.astype(np.float32), y)
    out["F1"] = {"sha_H": F1.sha(H), "sha_y": F1.sha(y), "per_class_r2": pc.tolist(),
                 "cert_r2_float64_input": c64["r_squared"], "cert_r2_float32_input": c32["r_squared"],
                 "convex_combo": float(w @ pc), "residual_float64": float(w @ pc - c64["r_squared"]),
                 "residual_float32": float(w @ pc - c32["r_squared"]), "stress": []}
    for tail_sd, offset in F1.STRESS:
        Hs, yb = F1.make_stress(tail_sd, offset)
        ds = da(Hs, yb)
        out["F1"]["stress"].append({"sha_H": F1.sha(Hs), "tail_sd": tail_sd, "offset": offset,
                                    "cert_r2_on_float32_H": cert(Hs, yb)["r_squared"],
                                    "cert_certified_at_0.05": cert(Hs, yb)["certified"],
                                    "dominant_axis_float64_per_class": ds["per_class_r2"]})
    # ---- F2
    out["F2"] = {}
    for name in F2.CASES:
        Z, yy, K = F2.make_case(name)
        dd = da(Z, yy, K)
        out["F2"][name] = {"sha_Z": F2.sha(Z), "r2_da": dd["r2_da"], "argmax": dd["argmax_class"],
                           "per_class_r2": dd["per_class_r2"], "cert_onehot_r2": cert(Z, yy, K)["r_squared"]}
    # ---- F3
    out["F3"] = {}
    for name in F3.CASES:
        Hh, yy, K = F3.make_case(name)
        out["F3"][name] = {
            "sha_H": F3.sha(Hh), "sha_y": F3.sha(yy),
            "cert_default_schema": safe(lambda: cert(Hh, yy)),
            "cert_declared_K4": safe(lambda: cert(Hh, yy, 4)) if has_nc else "num_classes not supported",
            "da_default_schema": safe(lambda: da(Hh, yy)),
            "da_declared_K4": safe(lambda: da(Hh, yy, 4)) if da_has_nc else "num_classes not supported",
        }
    # ---- F9
    h, A = F9.doc_example()
    out["F9"] = {"sha_h": F9.sha(h),
                 "certified_accuracy_bound_r2_0_pi_0.5": safe(lambda: V.certified_accuracy_bound(0.0, 0.5, 2)),
                 "cert_r2_on_counterexample": safe(lambda: cert(h[:, None], A, 2))}
    print(json.dumps(clean(out), indent=1, default=str))


if __name__ == "__main__":
    main()
