#!/usr/bin/env python3
"""Fixture G - status of the PCRL 'Linear Compliance Guarantee' (R2 -> accuracy bound). Standalone, exact.

1. Reads the bound's formula text from origin/main and from fix/retire-accuracy-guarantee via `git show`
   (no import) and records whether each ref still returns a number or raises.
2. Re-implements the origin/main formula  min(max(pi + sqrt(eps*k*pi*(1-pi)), pi), 1)  and the paper's
   Prop. 3 form  pi + k*sqrt(eps*pi*(1-pi))  and evaluates both on the 20-row counterexample
   (A,h) = (1,1)x9, (1,-9)x1, (0,-1)x9, (0,9)x1  in exact rational arithmetic.
3. Scope statement: what a zero-covariance (LEACE) fact does and does not imply.
Run: python3 G_accuracy_guarantee.py > outputs/G_accuracy_guarantee.json
"""
import hashlib
import json
import math
import re
import subprocess
from fractions import Fraction as F

REPO = "/Users/nathansamson/PCRL"
refs = {}
for ref in ["origin/main", "origin/fix/retire-accuracy-guarantee"]:
    raw = subprocess.check_output(["git", "-C", REPO, "show", f"{ref}:pcrl/purposes/verification.py"]).decode()
    sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", ref]).decode().strip()
    body = raw[raw.index("def certified_accuracy_bound"):]
    body = body[:body.index("\ndef ", 10)] if "\ndef " in body[10:] else body
    code = "\n".join(l for l in body.splitlines() if not l.strip().startswith(("#",)))
    refs[ref] = {"commit": sha, "file_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                 "returns_numeric_bound": bool(re.search(r"return min\(max\(bound, pi_maj\), 1\.0\)", code)),
                 "raises_NotImplementedError": "raise NotImplementedError" in code,
                 "docstring_marks_retired": "RETIRED" in body[:400],
                 "numeric_return_unreachable_after_raise": ("raise NotImplementedError" in code and
                     code.index("raise NotImplementedError") < code.index("return min(max(bound"))}
merged = subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", "origin/fix/retire-accuracy-guarantee",
                         "origin/main"]).returncode == 0

rows = [(1, F(1))] * 9 + [(1, F(-9))] + [(0, F(-1))] * 9 + [(0, F(9))]
n = len(rows)
mA = sum(F(a) for a, _ in rows) / n
mh = sum(h for _, h in rows) / n
cov = sum((a - mA) * (h - mh) for a, h in rows) / n
acc_thresh = F(sum((h > 0) == (a == 1) for a, h in rows), n)
pi, k, eps = 0.5, 2, 0.0
code_bound = min(max(pi + math.sqrt(eps * k * pi * (1 - pi)), pi), 1.0)
paper_bound = pi + k * math.sqrt(eps * pi * (1 - pi))
out = {
    "refs": refs,
    "fix_branch_merged_into_main": merged,
    "counterexample": {"n": n, "cov_h_A_exact": str(cov), "affine_LS_R2": 0.0 if cov == 0 else None,
                       "threshold_rule_h_gt_0_accuracy": str(acc_thresh), "majority": "1/2",
                       "origin_main_formula_bound_at_R2_0": code_bound, "paper_prop3_form_bound_at_R2_0": paper_bound,
                       "bound_violated": float(acc_thresh) > code_bound},
    "scope": {
        "valid_linear_fact": "Cov(h,A)=0 on a distribution/sample implies the best AFFINE least-squares predictor of one-hot A is "
                             "constant there (and LEACE guarantees this for every linear map of the erased features on the fit law).",
        "not_implied": "any bound on the accuracy of a linear THRESHOLD/argmax classifier (piecewise-constant, nonlinear in h), "
                       "any bound on nonlinear auditors, any bound off the fit distribution (held-out/test drift).",
    },
}
print(json.dumps(out, indent=1))
