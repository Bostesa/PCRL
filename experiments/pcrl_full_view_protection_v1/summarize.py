"""Render all registered finite outcomes and numerical checks from checkpoints."""

import argparse
import datetime
import glob
import json
import os
import tempfile


def _atomic(path, payload):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=directory, prefix=".summary-", delete=False) as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
        temp = f.name
    os.replace(temp, path)


def _pair(item):
    if item is None:
        return "—"
    return f'{item["cost"]:.5f} / {item["max_full_view_cmi"]:.5f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    items = {os.path.splitext(os.path.basename(path))[0]: json.load(open(path))
             for path in glob.glob(os.path.join(args.input_dir, "*.json"))}
    expected = ["xor", "safe_independent", "safe_correlated", "partly_coupled",
                "rational_separation", "finite_uncertainty", "unsupported", "rare",
                "prior_sensitivity"]
    if set(items) != set(expected):
        raise ValueError(f"missing or extra fixture: {set(expected) ^ set(items)}")
    lines = ["# Registered finite synthetic outcomes", "",
             "The first eight fixtures were registered before outcomes. `prior_sensitivity` is a post-suite exploratory extension registered in Amendment 2 before its own outcomes. All entries are `nominal Hamming cost / maximum I(S;Z|H_r)` in nats; the maximum is over both roles and every declared law in U. An em dash means no feasible member at that budget. The matched adaptation of Diaz-style uniform robust optimization to this Shannon-CMI target is mathematically the same robust programme and was not counted as a second fit. `synthetic/*.json` retains every channel, deterministic map, control level, law hash, CMI and primal/dual bracket.", "",
             "| Fixture | Budget | Robust full view | Nominal full view | Radius control | Best deterministic | Best RR | Best withholding |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fit_count = 0
    failed = []
    inaccurate = []
    wide = []
    max_gap = (0.0, None)
    max_residual = 0.0
    for name in expected:
        item = items[name]
        for budget, entry in item["budgets"].items():
            solutions = entry["methods"]
            fields = []
            for method in ("robust", "nominal", "capacity"):
                value = solutions[method]
                if "alias" in value:
                    fields.append("same as robust")
                    continue
                fit_count += 1
                if value.get("status") == "quarantined":
                    failed.append((name, budget, method, value["cause"]))
                    fields.append("quarantined")
                    continue
                if value["status"] == "optimal_inaccurate":
                    inaccurate.append((name, budget, method))
                gap = value["cost"] - value["objective_lower_bound"]
                if gap > max_gap[0]:
                    max_gap = (gap, (name, budget, method))
                if gap > 1e-5:
                    wide.append((name, budget, method, gap))
                if gap < -1e-8:
                    failed.append((name, budget, method, "invalid objective orientation"))
                max_residual = max(max_residual, value["simplex_residual"])
                if method == "robust" and value["max_full_view_cmi"] > float(budget) + 1e-7:
                    failed.append((name, budget, method, "CMI violation"))
                fields.append(_pair(value))
            d = entry["selected_deterministic"]
            rr_key = entry["selected_randomized_response"]
            w_key = entry["selected_withholding"]
            lines.append("| " + " | ".join((name, budget, *fields,
                                          _pair(d), _pair(item["randomized_response"].get(rr_key)),
                                          _pair(item["withholding"].get(w_key)))) + " |")
    lines += ["", "## Interpretation", "",
              "- The XOR deterministic release has coarse leakage 0 and full-H leakage log 2. The full-view budget correctly forces a privacy/utility tradeoff.",
              "- In the independent safe-task fixture, the deterministic and robust channels release the task perfectly with zero sensitive CMI, while the capacity control limits task information at its numerical budget. With H_A=S xor Y, that safe label becomes fully sensitive given H_A.",
              "- The inherited rational three-state fixture has a zero-CMI stochastic channel of cost 0.30000 and task information 0.0863046 nats; its best zero-CMI deterministic control costs 0.40000. This is a known perfect-privacy separation, not new theory.",
              "- In the partly coupled fixture at .01, robust full-view cost is about .37855, versus .45764 for capacity and .50000 for the best feasible deterministic map. These are same-target finite-law comparisons. In the finite perturbation list, robust cost is about .37865; the nominal solution has lower nominal cost but exceeds the list's .01 target and is not a feasible competitor.",
              "- The robust solution for the finite perturbation list has an outside-U local CMI of about .010054 at the .01 budget. Protection is therefore specifically scoped to U. In the unsupported and rare fixtures, a guarantee for the displayed exact law gives no uniform claim for unobserved population classes or continuous H.",
              "- The post-suite nine-law sensitivity varies both protected priors and conditional input laws in one coherent joint distribution per scenario. It remains a finite-scenario certificate and cannot establish ACS conditional-law coverage.",
              "- No new algorithmic method separates from the matched published robust-information adaptation: under the same U, full views, channel access and cost, its programme is identical. The suite therefore does not trigger the 2018 pilot under the registered gate.", "",
              "## Numerical status", "",
              f"{fit_count} distinct convex fit units; {len(failed)} quarantined; {len(inaccurate)} CLARABEL `optimal_inaccurate`; {len(wide)} objective brackets wider than 1e-5. Maximum guarded primal-minus-dual gap: {max_gap[0]:.8g} at {max_gap[1]}. Maximum row-simplex residual after output normalization: {max_residual:.3g}. The guarded supporting-hyperplane bounds are numerical certificates, not formal interval-arithmetic proofs. Wide cells are reported as optimization-precision limits, not as exact optima. Constant/null and deterministic channels are evaluated directly; every deterministic binary map is enumerated (4 for two inputs, 8 for three).", ""]
    checks = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "fixture_count": len(items), "convex_fit_units": fit_count,
              "quarantined": failed, "optimal_inaccurate": inaccurate,
              "wide_objective_brackets": wide,
              "maximum_guarded_objective_gap": max_gap,
              "maximum_row_simplex_residual": max_residual,
              "synthetic_pass_for_acs": False,
              "gate_reason": "matched published finite-list robust Shannon-CMI adaptation is the same optimization; no continuous-H envelope coverage"}
    _atomic(os.path.join(args.results_dir, "SYNTHETIC_RESULTS.md"), "\n".join(lines))
    _atomic(os.path.join(args.results_dir, "SYNTHETIC_CHECKS.json"),
            json.dumps(checks, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
