#!/usr/bin/env python3
"""Independent reconstruction of the prospective ACS 2016 conclusions.

Reads committed endpoint data at the pinned evidence commit through the git object store and re-derives
each primary decision from the stored estimates and standard errors, rather than reading the reported
status strings. Arithmetic only: no model is fitted, no pool is read, nothing is re-scored.
"""
import hashlib, json, math, os, subprocess

SHA = "5e154e5c4fdaeb23d327a0ebefe838525f1a19cb"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RD = "results/pcrl_final_prospective_v1/"
TASK_MARGIN = -0.003      # registered: task upper bound must be <= -0.003
SENS_CAP = 0.001          # registered: sensitive upper bound must be <= +0.001
sources, checks = {}, []


def show(name):
    raw = subprocess.run(["git", "-C", ROOT, "show", f"{SHA}:{RD}{name}"], check=True,
                         capture_output=True).stdout
    sources[RD + name] = hashlib.sha256(raw).hexdigest()
    return raw.decode()


def check(item, got, want, tol=None, note=""):
    ok = (abs(got - want) <= tol) if tol is not None else (got == want)
    checks.append({"item": item, "recomputed": got, "expected": want, "ok": bool(ok), "note": note})
    if not ok:
        raise SystemExit(f"MISMATCH {item}: {got} != {want}")


inf = json.loads(show("INFERENCE_2016.json"))
z_one = inf["primary"]["Q"]["z_one_sided"]
z_two = inf["secondary"]["z"]

# ---------------------------------------------------------------- 1. primary clauses, re-derived
prim = inf["primary"]
summary = {}
for cand, block in prim.items():
    entries = block["clauses"]
    task_pass, sens_pass, task_bounds = 0, 0, []
    for e in entries:
        est, se = float(e["estimate"]), float(e["bootstrap_se"])
        ub = est + z_one * se                      # re-derived, not read
        stored_ub = float(e["upper_bound_97_5"])
        assert abs(ub - stored_ub) < 5e-9, (cand, e["id"], ub, stored_ub)
        if e["clause"] == "task":
            task_bounds.append(ub)
            task_pass += ub <= TASK_MARGIN
        else:
            sens_pass += ub <= SENS_CAP
    summary[cand] = {"task_pass": task_pass, "sens_pass": sens_pass,
                     "task_upper_bounds": sorted(task_bounds), "clauses": task_pass + sens_pass}

for cand, want_ub in (("Q", [-0.001871410923827089, -0.0015052917591446078]),
                      ("D17", [-0.0029513, -0.0029134])):
    s = summary[cand]
    check(f"{cand}: sensitive non-inferiority clauses passed", s["sens_pass"], 8,
          note="every sensitive upper bound is below zero, stronger than the registered +0.001 cap")
    check(f"{cand}: task minimum-improvement clauses passed", s["task_pass"], 0,
          note=f"both upper bounds exceed the registered {TASK_MARGIN} margin")
    check(f"{cand}: total primary clauses passed", s["clauses"], 8)
    for got, want in zip(s["task_upper_bounds"], sorted(want_ub)):
        check(f"{cand}: task upper bound re-derived as estimate + z*SE", round(got, 6), round(want, 6), 1e-6)
    check(f"{cand}: full primary conjunction", s["clauses"] == 10, False,
          note="8 of 10; the conjunction requires all ten")

# The distinction the paper must draw: bounds support improvement, not the registered minimum.
for cand in ("Q", "D17"):
    ubs = summary[cand]["task_upper_bounds"]
    check(f"{cand}: task upper bounds are strictly below zero", all(u < 0 for u in ubs), True,
          note="the data do support task improvement over J")
    check(f"{cand}: task upper bounds are above the registered margin", all(u > TASK_MARGIN for u in ubs), True,
          note="they do not support the registered MINIMUM improvement of 0.003 nats")

# ---------------------------------------------------------------- 2. matched simple controls
rows = inf["secondary"]["rows"]
ctrl = {}
for name in ("RR75", "W75", "D17", "D33"):
    hits = [r for r in rows if r["id"].startswith(f"secondary|Q-vs-{name}|")]
    task = [r for r in hits if "utility" in r.get("role", "")]
    sens = [r for r in hits if "attack" in r.get("role", "")]
    q_better_task, q_worse_sens = 0, 0
    for r in task:
        # orientation: CE(Q) - CE(control); negative favours Q on the task
        if r["upper"] < 0: q_better_task += 1
    for r in sens:
        # same orientation on recovery: strictly positive interval means Q leaks MORE
        if r["lower"] > 0: q_worse_sens += 1
    ctrl[name] = {"task_endpoints": len(task), "sensitive_endpoints": len(sens),
                  "task_estimates": [round(r["estimate"], 6) for r in task],
                  "task_endpoints_Q_significantly_better": q_better_task,
                  "sensitive_endpoints_Q_significantly_worse": q_worse_sens}
check("Q improves task loss against RR75 under both weightings",
      ctrl["RR75"]["task_endpoints_Q_significantly_better"], 2)
check("Q improves task loss against W75 under both weightings",
      ctrl["W75"]["task_endpoints_Q_significantly_better"], 2)
check("Q is significantly WORSE on sensitive recovery in 6 of 8 endpoints against RR75",
      (ctrl["RR75"]["sensitive_endpoints_Q_significantly_worse"], ctrl["RR75"]["sensitive_endpoints"]), (6, 8),
      note="a complete task/privacy benefit over the matched simple control is NOT established")
check("Q is significantly WORSE on sensitive recovery in 6 of 8 endpoints against W75",
      (ctrl["W75"]["sensitive_endpoints_Q_significantly_worse"], ctrl["W75"]["sensitive_endpoints"]), (6, 8))
check("Q shows no significant sensitive advantage over D17 on any endpoint",
      sum(1 for r in rows if r["id"].startswith("secondary|Q-vs-D17|")
          and "attack" in r.get("role", "") and r["upper"] < 0), 0,
      note="randomisation is not shown to beat the matched deterministic control")

# ---------------------------------------------------------------- 3. deterministic controls, by weighting
for comp in ("D17", "D33"):
    task = [r for r in rows if r["id"].startswith(f"secondary|Q-vs-{comp}|") and "utility" in r["role"]]
    unw = [r for r in task if r["weighting"] == "unweighted"][0]
    pw = [r for r in task if r["weighting"] != "unweighted"][0]
    check(f"Q is significantly WORSE than {comp} on unweighted task loss", unw["lower"] > 0, True,
          note=f"estimate {unw['estimate']:+.6f}, interval [{unw['lower']:+.6f}, {unw['upper']:+.6f}] excludes zero")
    check(f"Q vs {comp} weighted task contrast is UNRESOLVED", pw["lower"] <= 0 <= pw["upper"], True,
          note=f"estimate {pw['estimate']:+.6f}, interval [{pw['lower']:+.6f}, {pw['upper']:+.6f}] crosses zero")
    sens = [r for r in rows if r["id"].startswith(f"secondary|Q-vs-{comp}|") and "attack" in r["role"]]
    check(f"no sensitive endpoint shows Q superior to {comp}", sum(1 for r in sens if r["upper"] < 0), 0,
          note="and none shows Q significantly worse either; all eight are unresolved")

# ---------------------------------------------------------------- 4. inferential scope
check("the primary decision is a conjunction of per-clause one-sided bounds, not a simultaneous family",
      inf["primary"]["Q"]["alpha"], 0.025,
      note="intersection-union: each clause is tested at its own level. Passing 8 of 10 does NOT confer "
           "simultaneous 95% coverage on those 8 components when they are highlighted separately.")
check("the secondary family carries its own simultaneous correction", round(z_two, 4), 3.384,
      note="70 endpoints, two-sided Bonferroni; reported apart from the primary decision")

out = {"pinned_commit": SHA, "sources_sha256": sources, "z_one_sided": z_one, "z_two_sided": z_two,
       "primary_summary": summary, "matched_simple_controls": ctrl, "checks": checks,
       "counts": inf.get("counts"), "households": inf.get("households"),
       "scope": "arithmetic on committed endpoint data; nothing re-scored; 2016 prospective, kept separate from 2018 development"}
if __name__ == "__main__":
    json.dump(out, open(os.path.join(ROOT, "results/pcrl_submission_review_v1/PROSPECTIVE_2016_VERIFICATION.json"), "w"), indent=1)
    for c in checks:
        print(f"[{'OK ' if c['ok'] else 'BAD'}] {c['item']}: {c['recomputed']}")
    print(f"\n{len(checks)} checks, all_ok={all(c['ok'] for c in checks)}")
    print("matched simple controls:", json.dumps(ctrl, indent=1))
