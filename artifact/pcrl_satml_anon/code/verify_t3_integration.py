#!/usr/bin/env python3
"""Source-pinned verification of the claims-foundation integration.

Re-checks, from committed sources, each claim the integration relies on. Arithmetic and source reads
only: no model is fitted, no ACS pool is read, 2016 is not scored.
"""
import hashlib, json, os, subprocess

T3 = "33124f965861ccfcaa710aff4a56880c5ab3957e"
TD = "f4bdf4cd5bf74c634feeec50aef78bff249667e4"
OW = "0176f149e91c02b8e2d202eb25ea9cba8ae019dc"
REG = "ecaeba46fcb921802087e792c5dbac2073e25347"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sources, checks = {}, []


def show(sha, path):
    raw = subprocess.run(["git", "-C", ROOT, "show", f"{sha}:{path}"], check=True,
                         capture_output=True).stdout
    sources[f"{sha[:9]}:{path}"] = hashlib.sha256(raw).hexdigest()
    return raw.decode(errors="replace")


def check(item, ok, detail):
    checks.append({"item": item, "ok": bool(ok), "detail": detail})
    if not ok:
        raise SystemExit(f"FAILED: {item}")


enc = show(TD, "experiments/pcrl_task_directed_release_v1/encoding.py")
check("the input code is residence-supervised, not label-free",
      "same_residence" in enc.split("def fit_encoder")[1][:1200],
      "fit_encoder fits its teachers on rf['labels']['same_residence']")
check("conditioning partition is 2 local cells", "n_local=2" in enc,
      "ServicePartitions.fit(..., n_local=2); the coalition adds a median split of H_B -> 4 cells")

head = json.loads(show(TD, "results/pcrl_task_directed_release_v1/HEADLINE_EVIDENCE.json"))
cfg = head["nominees"]["utility_first"]["configuration"]
check("the selected utility release uses the local policy", "_L_" in cfg,
      f"nominee {cfg}: policy L imposes no AB constraint, so the coalition view is audited, not constrained")

ver = show("origin/main", "pcrl/purposes/verification.py")
check("public main still ships the retired guarantee",
      "def certified_accuracy_bound" in ver and "NotImplementedError" not in ver.split("def certified_accuracy_bound")[1][:3000],
      "origin/main defines certified_accuracy_bound with its Theorem docstring and does not raise")

pred = show(REG, "results/rebuttal/ablations_facct/PREDICTIONS.md")
check("the earlier submission was reviewed (handles on record)",
      "AkJK" in pred and "NY7k" in pred,
      "PREDICTIONS.md registers ablations against 'reviewers AC, AkJK Q1, NY7k Q2' and 'AkJK Q2'")

for d in ("erase_layer_vicreg_sweep_aws", "erase_rank8_diabetes_cpu"):
    h = show(OW, f"results/rebuttal/{d}/HEADLINE.md")
    check(f"results/rebuttal/{d} is this line's rebuttal work",
          ("per_dim_std" in h or "LoRA" in h) and "AAAI" not in h[:400],
          "headline concerns per-dim std / the LoRA erasure floor on the published backbone")

tex = open(os.path.join(ROOT, "papers/pcrl_satml_final_v1/main.tex")).read()
# The task-directed study's code is residence-supervised. The PREDECESSOR study's k-means quantizer
# genuinely was label-free (stage_b.fit_code sees no labels), so that one sentence stays as written.
check("the task-directed construction is not called label-free",
      "residence-supervised teachers" in tex and "label-free code $T=g(X_A)$" not in tex,
      "construction paragraph describes two residence-supervised teachers and a residual logit")
check("the baselines paragraph says residence-supervised",
      "same residence-supervised code" in tex, "deterministic action releases use that same code")
check("the one surviving 'label-free' refers to the predecessor study only",
      tex.count("label-free") == 1 and "A predecessor experiment found that a label-free code" in tex,
      "predecessor quantizer is label-free by construction (stage_b.fit_code); verified separately")
check("the paper states the selected release is local",
      "selected utility release is local" in tex, "local vs coalition scope is explicit")
check("the paper states the conditioning cell counts",
      "two cells of $H_A$" in tex, "2 cells for A, 4 for AB, with the guarantee limit stated")
check("the paper does not call the increment a floor",
      "floor on what is recoverable" not in tex,
      "absolute recovery is the lower bound; the increment bounds I(S;Z|H) in neither direction")
check("the paper states the two lines are separate implementations",
      "no combined encoder, adapter and stochastic-release pipeline" in tex, "lineage appendix")
check("the paper qualifies the API claim to the evaluated code",
      "refuses to run in the evaluated code" in tex, "public main is not claimed fixed")

out = {"audit_commit": T3, "sources_sha256": sources, "checks": checks,
       "n_checks": len(checks), "all_ok": all(c["ok"] for c in checks)}
if __name__ == "__main__":
    json.dump(out, open(os.path.join(ROOT, "results/pcrl_submission_review_v1/T3_VERIFICATION.json"), "w"), indent=1)
    for c in checks:
        print(f"[{'OK ' if c['ok'] else 'BAD'}] {c['item']}")
    print(f"\n{len(checks)} checks, all_ok={out['all_ok']}")
