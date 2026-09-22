#!/usr/bin/env python3
"""Independent recomputation of the task-directed study's headline claims.

Reads committed evidence at the pinned commit through the git object store. Arithmetic only: no ACS
model is fitted, no pool is read, 2016 is not scored. Every number the manuscript quotes is either
recomputed here from the smallest machine-readable source, or explicitly marked source-stated with a
pointer when the evidence package does not expose the resolved mapping.
"""
import hashlib, json, os, subprocess

SHA = "f4bdf4cd5bf74c634feeec50aef78bff249667e4"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RD = "results/pcrl_task_directed_release_v1/"
Q, D17, D33 = "T0_L_0.01_a17", "T0_U_unconstrained_a17", "T0_U_unconstrained_a33"
sources, checks = {}, []


def show(name):
    raw = subprocess.run(["git", "-C", ROOT, "show", f"{SHA}:{RD}{name}"], check=True,
                         capture_output=True).stdout
    sources[RD + name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode())


def check(item, got, want, status="CONFIRMED", note=""):
    ok = got == want if not isinstance(want, float) else abs(got - want) <= 5e-5
    checks.append({"item": item, "recomputed": got, "expected": want, "ok": bool(ok),
                   "status": status if ok else "MISMATCH", "note": note})
    if not ok:
        raise SystemExit(f"MISMATCH {item}: {got} != {want}")


head = show("HEADLINE_EVIDENCE.json")
narr = show("NARRATIVE_MANIFEST.json")

# ---------------------------------------------------------------- 1. the nominated release Q
uf = head["nominees"]["utility_first"]
check("utility-route nominee is Q", uf["configuration"], Q)
task = head["fixed_task_path"][Q]["loss_delta_J"]
check("Q task improvement over J, unweighted (nats)", round(-task["unweighted"], 5), 0.01625)
check("Q task improvement over J, person-weighted (nats)", round(-task["PWGTP"], 5), 0.01639)

ub = uf["J_utility_bounds"]
check("both task bounds exclude zero (material improvement passes)",
      all(b["passed"] and b["upper"] < 0 for b in ub), True)

pb = uf["J_privacy_bounds"]
check("eight primary sensitive endpoints", len(pb), 8)
check("all eight sensitive point estimates favour Q over J",
      all(b["estimate"] < 0 for b in pb), True,
      note="negative = Q's audited attacker recovers less than J's")
passed = [b for b in pb if b["passed"]]
check("adjusted non-inferiority bounds that fail", 8 - len(passed), 7,
      note="only AB/SEX unweighted resolves (upper %.5f)" % passed[0]["upper"] if passed else "")
check("registered competitiveness conjunction not established", uf["competitive_passed"], False)
check("historical J conjunction not established", uf["historical_J_passed"], False)

# ---------------------------------------------------------------- 2. attribution earned nothing
attr_u = uf["attribution"]
attr_p = head["nominees"]["protection_first"]["attribution"]
check("no attribution claim obtained (either route)",
      any(attr_u.values()) or any(attr_p.values()), False,
      note="randomization, coalition and task-refinement attributions all unmet; the nominee is local")

# ---------------------------------------------------------------- 3. the deterministic comparators
ftp = head["fixed_task_path"]
for name, label in ((D17, "D17"), (D33, "D33")):
    d_j = -ftp[name]["loss_delta_J"]["unweighted"]
    q_j = -ftp[Q]["loss_delta_J"]["unweighted"]
    check(f"{label} task point estimate beats Q's (unweighted)", d_j > q_j, True,
          note=f"{label} {d_j:.5f} vs Q {q_j:.5f} nats over J")

pts = {}
for row in narr["primary_sensitive_points"]:
    pts.setdefault(row["configuration"], {})[(row["role"], row["weighting"])] = row
keys = sorted(pts[Q])
lower = sum(pts[D17][k]["recovery_increment_over_J"] < pts[Q][k]["recovery_increment_over_J"]
            for k in keys)
check("D17 sensitive point estimates lower than Q in 7 of 8 cells", (lower, len(keys)), (7, 8),
      note="point estimates only; the corresponding adjusted differences are unresolved")

mixed_lower = sum(pts[D33][k]["recovery_increment_over_J"] < pts[Q][k]["recovery_increment_over_J"]
                  for k in keys)
checks.append({"item": "D33 sensitive comparison is mixed", "recomputed": f"{mixed_lower} of 8 lower",
               "expected": "mixed", "ok": 0 < mixed_lower < 8, "status": "CONFIRMED",
               "note": "D33 is not an overall dominating method and no cap feasibility was established"})

# ---------------------------------------------------------------- 4. execution counts
c = head["counts"]
kg = head["kernel_groups"]
check("scheduled/completed maps of the nominal total", (c["accepted_map_units"], c["nominal_map_units"]),
      (126, 162))
check("reduced exact kernel groups", kg["unique_reduced_kernel_groups"], 125)
check("prospectively unscheduled optional maps", c["resource_unscheduled_release_anchor_units"], 36)
check("current audits / evaluations", (c["accepted_audit_units"], c["accepted_evaluation_units"]),
      (297, 297))
check("126 + 36 = 162 reconciles the nominal map total",
      c["accepted_map_units"] + c["resource_unscheduled_release_anchor_units"], c["nominal_map_units"])

# 81 primary maps is a NOMINAL protocol count, not an execution count; verify its arithmetic in source.
proto = subprocess.run(["git", "-C", ROOT, "show", f"{SHA}:{RD}PROTOCOL.md"], check=True,
                       capture_output=True).stdout.decode()
sources[RD + "PROTOCOL.md"] = hashlib.sha256(proto.encode()).hexdigest()
import re
m = re.search(r"Primary matrix: (\d+) positive maps,\s*(\d+) zero maps,\s*(\d+) unconstrained maps\s*=\s*(\d+) nominal maps", proto)
check("81 primary maps = 54 positive + 18 zero + 9 unconstrained (nominal, from PROTOCOL)",
      (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
       int(m.group(1)) + int(m.group(2)) + int(m.group(3))), (54, 18, 9, 81, 81),
      note="nominal primary-matrix count; distinct from the 126 accepted map units")
mv = subprocess.run(["git", "-C", ROOT, "show", f"{SHA}:{RD}MATH_VALIDATION.md"], check=True,
                    capture_output=True).stdout.decode()
sources[RD + "MATH_VALIDATION.md"] = hashlib.sha256(mv.encode()).hexdigest()
check("independent replay covered all 81 primary maps", "all 81 primary ACS maps" in mv, True,
      note="replay evidence: MATH_REPLAY_PRIMARY_anchor*.json, before the registered numerical recovery")

# ---------------------------------------------------------------- 5. scope statements
check("evidence is development data, conditional on fitted models",
      "development" in head["scope"], True)
check("family size used for the simultaneous correction", head["family_size"], 706)
check("bootstrap households", head["bootstrap_households"], 5473)

out = {"pinned_commit": SHA, "sources_sha256": sources, "checks": checks,
       "n_checks": len(checks), "all_ok": all(c["ok"] for c in checks),
       "scope": "arithmetic on committed evidence; no ACS model fitted; 2016 not scored"}
if __name__ == "__main__":
    json.dump(out, open(os.path.join(ROOT, "results/pcrl_submission_review_v1/TASK_DIRECTED_VERIFICATION.json"), "w"), indent=1)
    for c_ in checks:
        print(f"[{c_['status']:12s}] {c_['item']}")
    print(f"\n{len(checks)} checks, all_ok={out['all_ok']}")
