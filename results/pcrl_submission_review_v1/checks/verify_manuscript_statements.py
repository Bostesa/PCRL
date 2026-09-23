#!/usr/bin/env python3
"""Check that the manuscript still states each required scope fact, and states it correctly.

Text checks against the compiled source plus, where the fact is about the world rather than the paper,
a source check against the repository. No scientific refitting.
"""
import json, os, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TEX = open(os.path.join(ROOT, "papers/pcrl_satml_final_v1/main.tex")).read()
checks = []


def want(item, ok, detail):
    checks.append({"item": item, "ok": bool(ok), "detail": detail})


def has(*frags):
    return all(f in TEX for f in frags)


# 1. separate implementations, no evaluated combined path
want("separate implementations; no evaluated combined LoRA/stochastic path",
     has("no combined encoder, adapter and stochastic-release pipeline",
         "separate implementations"),
     "lineage appendix states no combined pipeline has been built or evaluated")

# 2. Q is residence-supervised and uses local policy L; coalition audited not constrained
want("Q uses residence-supervised teachers",
     has("residence-supervised teachers") and "label-free code $T=g(X_A)$" not in TEX,
     "construction paragraph; the single remaining 'label-free' refers to the predecessor study")
want("Q's policy is local; coalition audited but not constrained",
     has("selected utility release is local", "audited but not constrained"),
     "method section distinguishes available policies from the selected candidate's policy")

# 3. coarse conditioning cells, and what they are not
want("two local and four joint conditioning cells, not the recipient's continuous side information",
     has("two cells of $H_A$", "four of $(H_A,H_B)$") and "not a population" in TEX,
     "stated with the limit that a fitted-model budget is not a guarantee on continuous H")

# 4. increments are not MI bounds
want("fitted recovery increments are not bounds on conditional mutual information",
     "bounds $I(S;Z\\mid H)$ in neither" in TEX and "floor on what is recoverable" not in TEX,
     "absolute recovery is the lower bound; the increment is a difference of two lower bounds")

# 5. byte preservation vs accuracy drift
want("freezing the service preserves bytes; accuracy can change across years",
     has("byte-identical", "That is a statement\nabout bytes, not about accuracy", "$0.874$ for income"),
     "preservation is architectural and checked bitwise; the same service has year-specific accuracy")

# 6. established mathematics
want("the convex finite programme is established mathematics",
     has("an established", "calmon2012privacy", "We claim no new\noptimisation result"),
     "attribution to Calmon & Fawaz 2012 / Salamatian 2015 and the nullspace criterion")

# 7. dominant-axis vs all-directions
want("dominant single class is not all sensitive directions",
     has("largest per-class\none-versus-rest score, itself only a lower bound on the best linear sensitive direction"),
     "the dominant axis is a lower bound on the best linear direction, not a maximum over directions")

# 8. checkpoint counts not interchangeable
want("best-checkpoint and final-checkpoint counts are distinguished",
     has("final iterate gives $56/60$", "best-checkpoint rule gives $54/60$", "under one stated rule"),
     "counts are reported separately and under one rule")

# 9. the main-branch guarantee is not retired until merged
merged = subprocess.run(["git", "-C", ROOT, "branch", "-r", "--contains",
                         "5d4eda04639aae10733e4d72c2ceaae0e849ede5"],
                        capture_output=True, text=True).stdout
on_main = any(b.strip() in ("origin/main",) for b in merged.splitlines())
want("the repair is NOT merged into main",
     not on_main, f"origin/main does not contain the repair commit; branches containing it: {merged.split() or 'none'}")
want("the paper does not claim the public API is fixed",
     has("refuses to run in the evaluated code") and "API now refuses to run" not in TEX,
     "claim is scoped to the evaluated code, not to the public default branch")

# 10. the two years are separated
want("2016 prospective and 2018 development are kept separate",
     "evidence are never pooled" in TEX and "fresh-year test of a supervised fixed family" in TEX,
     "no pooling, and the 2016 result is not presented as unseen-task transfer")

out = {"checks": checks, "n": len(checks), "all_ok": all(c["ok"] for c in checks)}
if __name__ == "__main__":
    json.dump(out, open(os.path.join(ROOT, "results/pcrl_submission_review_v1/MANUSCRIPT_STATEMENTS.json"), "w"), indent=1)
    for c in checks:
        print(f"[{'OK ' if c['ok'] else 'BAD'}] {c['item']}")
    print(f"\n{len(checks)} statements, all_ok={out['all_ok']}")
