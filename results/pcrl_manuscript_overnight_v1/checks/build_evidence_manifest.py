#!/usr/bin/env python3
"""Evidence manifest: every source this session read, by commit and SHA-256, plus what was verified."""
import hashlib, json, os, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PINS = {
    "manuscript_baseline": "eb4aa96685568426c098effb00b857f1ae934b0d",
    "diagnostic_audited": "cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a",
    "replacement_registration_reviewed": "0af175c35164ef224b4dc8149b5a3aa309ac6280",
    "study6_evidence": "ba531ab424c593fe8573cd320d2bcef379907cf1",
    "study5_evidence": "a56bcc7ff7a122a6411ac3072b11b00b75f4cdad",
}
FILES = {
    "cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a": [
        "results/pcrl_stochastic_channel_v1/STAGE_B_RESULTS.json",
        "results/pcrl_stochastic_channel_v1/gates/G2.json",
        "results/pcrl_stochastic_channel_v1/gates/G0.json",
        "results/pcrl_stochastic_channel_v1/AMENDMENT_1.md",
        "results/pcrl_stochastic_channel_v1/RESEARCH_DECISION.md",
        "results/pcrl_stochastic_channel_v1/REGISTRATION.md",
        "experiments/pcrl_stochastic_channel_v1/stage_b.py",
    ],
    "0af175c35164ef224b4dc8149b5a3aa309ac6280": [
        "results/pcrl_stochastic_replacement_overnight_v1/PROTOCOL.md",
        "results/pcrl_stochastic_replacement_overnight_v1/METHOD.md",
        "results/pcrl_stochastic_replacement_overnight_v1/STATISTICAL_PLAN.md",
        "results/pcrl_stochastic_replacement_overnight_v1/RELEASE_CONTRACT.md",
        "results/pcrl_stochastic_replacement_overnight_v1/RUN_MATRIX.json",
    ],
}
out = {"pins": PINS, "sources": {}, "verified_locally": [
    "Stage B B2 advantages, both weightings, all anchors, recomputed from STAGE_B_RESULTS.json",
    "estimate/SE ratios recomputed from the stored candidate-specific intervals",
    "J-only and J+T subsumption columns cross-checked against STAGE_B_RESULTS.json (exact match)",
    "B1 label path traced in stage_b.py source and reproduced in a synthetic fixture",
    "nominal Q-fit cap 144 reconciled from RUN_MATRIX stage_Q_primary_matrix factors",
], "not_verified": [
    "prior and T-only probe values (no machine-readable backing or committed generator at the pin)",
    "the Stage B bootstrap itself (per-row losses were not serialised; SEs are read, not recomputed)",
    "any replacement-study outcome (none exists; zero units fitted at review time)",
    "Terminal 1's cloud/instance state (reported by its status file, not directly verified)",
]}
for sha, paths in FILES.items():
    for p in paths:
        raw = subprocess.run(["git", "-C", ROOT, "show", f"{sha}:{p}"], capture_output=True).stdout
        out["sources"][f"{sha[:9]}:{p}"] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
local = {}
for d, _, fs in os.walk(os.path.join(ROOT, "results/pcrl_manuscript_overnight_v1")):
    for f in sorted(fs):
        if f.endswith((".py", ".md", ".json")):
            p = os.path.join(d, f)
            local[os.path.relpath(p, ROOT)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
for d, _, fs in os.walk(os.path.join(ROOT, "papers/pcrl_satml_v1")):
    for f in sorted(fs):
        if not f.endswith((".aux", ".log", ".out", ".bbl", ".blg", ".fls", ".fdb_latexmk")):
            p = os.path.join(d, f)
            local[os.path.relpath(p, ROOT)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
out["session_artifacts_sha256"] = local
json.dump(out, open(os.path.join(ROOT, "results/pcrl_manuscript_overnight_v1/EVIDENCE_MANIFEST.json"), "w"), indent=1)
print(f"{len(out['sources'])} pinned sources, {len(local)} session artifacts")
