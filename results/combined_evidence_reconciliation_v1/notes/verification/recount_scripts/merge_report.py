#!/usr/bin/env python3
"""Merge report_parts/*.json into ../verification_report.json.
Fills sha256 for inputs given as 'Bostesa/PCRL@<ref>:<path>' (single path, no spaces) by hashing `git show` output.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

V = Path(__file__).resolve().parent.parent
REPO = "/Users/nathansamson/PCRL"
VOCAB = {"independently recomputed", "checked against code and recorded aggregates",
         "reported but not independently reproduced", "unsupported by located artifacts"}
ORDER = ["ABCG_pcrl.json", "D.json", "EFG_aaai.json"]
items = []
for name in ORDER:
    p = V / "report_parts" / name
    if p.exists():
        items += json.load(open(p))
pat = re.compile(r"^Bostesa/PCRL@([^:\s]+):([^\s:]+)$")
for it in items:
    for inp in it.get("inputs", []):
        if inp.get("sha256"):
            continue
        m = pat.match(inp["location"])
        if m:
            try:
                raw = subprocess.check_output(["git", "-C", REPO, "show", f"{m.group(1)}:{m.group(2)}"], stderr=subprocess.DEVNULL)
                inp["sha256"] = hashlib.sha256(raw).hexdigest()
            except subprocess.CalledProcessError:
                inp["sha256"] = None
bad = [it["id"] for it in items if it.get("status") not in VOCAB]
out = {"generated": "2026-10-01", "role": "C (independent verification)",
       "status_vocabulary": sorted(VOCAB), "n_items": len(items), "items_with_nonstandard_status": bad,
       "items": items}
json.dump(out, open(V / "verification_report.json", "w"), indent=1)
print(len(items), "items; nonstandard status:", bad)
