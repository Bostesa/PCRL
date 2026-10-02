"""A6: class support per role, representation scale on defense_fit rows, and the PANEL draft. No fitting.

Outputs (repo, aggregate only):
  notes/admission/support_counts.csv   one row per (dataset, variable, role, class): rows, units, floor, supported
  notes/admission/support_pairs.csv    one row per (tier, dataset, purpose, attribute): supported classes / class
                                       pairs per role and overall (class must meet the floor in every role where it
                                       is used: defense_fit, attacker_fit, attacker_val, assessment), NE classes
  notes/admission/scale.csv            per (dataset, seed, purpose, sigma): per-dim std and ||h|| quantiles on
                                       defense_fit rows, sigma / scale ratios
  notes/admission/PANEL_draft.csv      Tier-1 units (status admitted/missing) + Tier-2 registered extension units
  notes/admission/extension_inventory.csv  Round-5/7 headline checkpoints (hash only; never in Round-4 rows)
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admission_common import (DATASETS, INPUTS, NOTES, RELEASE_SEEDS, SEEDS, SIGMAS, SUPPORT_MIN, TIER1,  # noqa
                              TIER2_E1)

ROLES = ("defense_fit", "attacker_fit", "attacker_val", "assessment")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    idx = json.loads((INPUTS / "INPUTS_INDEX.json").read_text())
    sc_rows, sp_rows, scale_rows, panel = [], [], [], []
    supported = {}
    for ds in DATASETS:
        D = idx["datasets"][ds]
        L = np.load(D["labels_npz"])
        role, unit = L["role"], L["unit"]
        dims = {}
        for p in D["purposes"].values():
            dims.update(p["disallowed_attr_dims"])
            dims[p["task"]] = p["task_dim"]
        variables = [(a, "sensitive", a) for a in D["sensitive_label_keys"]] + \
                    [(k[5:], "task", k) for k in D["task_label_keys"]]
        for name, kind, key in variables:
            y = L[key]
            for r in ROLES:
                m = role == r
                for c in range(dims[name]):
                    mc = m & (y == c)
                    n = int(mc.sum())
                    sc_rows.append({"dataset": ds, "variable": name, "kind": kind, "role": r, "class": c,
                                    "rows": n, "units": int(len(np.unique(unit[mc]))), "floor": SUPPORT_MIN[r],
                                    "supported": n >= SUPPORT_MIN[r]})
                    supported[(ds, name, r, c)] = n >= SUPPORT_MIN[r]
        # pair-level support (Tier-1 target, Tier-1 policy-set attrs, Tier-2 E1 attrs)
        t1 = TIER1[ds]
        pairs = [("T1", t1["purpose"], t1["target"])] + [("T2-E1", p, a) for p, a in TIER2_E1[ds]]
        for tier, pur, att in pairs:
            K = D["purposes"][pur]["disallowed_attr_dims"][att]
            task = D["purposes"][pur]["task"]
            ok = [c for c in range(K) if all(supported[(ds, att, r, c)] for r in ROLES)]
            ne = {c: [r for r in ROLES if not supported[(ds, att, r, c)]] for c in range(K) if c not in ok}
            tdim = D["purposes"][pur]["task_dim"]
            tok = [c for c in range(tdim) if all(supported[(ds, task, r, c)] for r in ROLES[1:])]
            sp_rows.append({"tier": tier, "dataset": ds, "purpose": pur, "attribute": att, "n_classes": K,
                            "supported_classes": ";".join(map(str, ok)), "n_supported": len(ok),
                            "supported_class_pairs": len(list(itertools.combinations(ok, 2))),
                            "NE_classes(role failing floor)": ";".join(f"{c}:{'+'.join(v)}" for c, v in ne.items()),
                            "estimable_macro_auc": len(ok) >= 2,
                            "task": task, "task_classes_supported_attacker_roles": ";".join(map(str, tok)),
                            "task_NE_classes": ";".join(str(c) for c in range(tdim) if c not in tok)})
        # policy-set support in defense_fit for C (Tier 1 and E2 purposes)
        for pur in sorted({t1["purpose"]} | {p for p, _ in TIER2_E1[ds]}):
            for att in D["purposes"][pur]["disallowed_attrs"]:
                K = D["purposes"][pur]["disallowed_attr_dims"][att]
                bad = [c for c in range(K) if not supported[(ds, att, "defense_fit", c)]]
                sp_rows.append({"tier": "C-fit(defense_fit)", "dataset": ds, "purpose": pur, "attribute": att,
                                "n_classes": K, "supported_classes": ";".join(str(c) for c in range(K) if c not in bad),
                                "n_supported": K - len(bad), "supported_class_pairs": "",
                                "NE_classes(role failing floor)": ";".join(f"{c}:defense_fit" for c in bad),
                                "estimable_macro_auc": "", "task": "", "task_classes_supported_attacker_roles": "",
                                "task_NE_classes": ""})
        # scale on defense_fit rows
        dfm = role == "defense_fit"
        for s in SEEDS:
            E = D["encoders"][str(s)]
            if E["lineage_status"] != "ADMITTED":
                continue
            F = np.load(E["forward_npz"])
            assert np.array_equal(F["row_id"], L["row_id"])
            for pur, p in D["purposes"].items():
                H = F[p["rep_key"]][dfm]
                std = H.std(axis=0, ddof=0)
                rms = float(np.sqrt(np.mean(std ** 2)))
                nrm = np.linalg.norm(H, axis=1)
                cn = np.linalg.norm(H - H.mean(0), axis=1)
                q = np.quantile(nrm, [0.05, 0.25, 0.5, 0.75, 0.95])
                d = H.shape[1]
                base = {"dataset": ds, "encoder_seed": s, "purpose": pur, "purpose_index": p["index"],
                        "n_defense_fit_rows": int(dfm.sum()), "dim": d,
                        "dim_std_mean": float(std.mean()), "dim_std_min": float(std.min()),
                        "dim_std_median": float(np.median(std)), "dim_std_max": float(std.max()),
                        "dim_std_rms": rms, "norm_mean": float(nrm.mean()), "norm_q05": float(q[0]),
                        "norm_q25": float(q[1]), "norm_q50": float(q[2]), "norm_q75": float(q[3]),
                        "norm_q95": float(q[4]), "centered_norm_median": float(np.median(cn))}
                for sg in SIGMAS:
                    scale_rows.append({**base, "sigma_abs": sg, "sigma_over_dim_std_rms": sg / rms,
                                       "sigma_over_dim_std_mean": sg / float(std.mean()),
                                       "noise_norm_over_median_norm(sigma*sqrt(d)/norm_q50)": sg * np.sqrt(d) / q[2],
                                       "noise_norm_over_median_centered_norm": sg * np.sqrt(d) / float(np.median(cn))})
        # PANEL
        t1p = D["tier1"]

        def add(tier, s, pur, att, arm, sigma="", rs="", scope_target="", scope_policy="", note=""):
            E = D["encoders"][str(s)]
            okenc = E["lineage_status"] == "ADMITTED" and E["forward_sha256"] is not None
            pr = next(r for r in sp_rows if r["dataset"] == ds and r["purpose"] == pur and r["attribute"] == att
                      and r["tier"] in ("T1", "T2-E1"))
            status = ("admitted" if tier == "T1" else "registered_inputs_admitted") if okenc else "missing"
            reason = "" if okenc else f"encoder {ds} s{s}: {E['lineage_status']}"
            if okenc and not pr["estimable_macro_auc"]:
                status, reason = "NE", "fewer than 2 supported classes"
            panel.append({"tier": tier, "unit_id": f"{ds}__s{s}__{pur}__{att}__{arm}", "dataset": ds,
                          "encoder_seed": s, "purpose": pur, "purpose_index": D["purposes"][pur]["index"],
                          "task": D["purposes"][pur]["task"], "attribute": att, "arm": arm.split("_sigma")[0],
                          "sigma_abs": sigma, "release_seed": rs, "eraser_target_scope": scope_target,
                          "eraser_policy_scope": scope_policy, "status": status, "reason": reason,
                          "supported_classes": pr["supported_classes"],
                          "NE_classes": pr["NE_classes(role failing floor)"],
                          "checkpoint_sha256": E["checkpoint_sha256"], "forward_sha256": E["forward_sha256"],
                          "labels_sha256": D["labels_sha256"], "defense_route": D["defense_route"], "note": note})

        pur, att = t1p["purpose"], t1p["target"]
        pol = "+".join(t1p["policy"])
        for s in SEEDS:
            add("T1", s, pur, att, "A")
            add("T1", s, pur, att, "B", scope_target=att)
            add("T1", s, pur, att, "C", scope_target=att, scope_policy=pol,
                note="C fitted on concatenated marginal one-hots of the policy set; alias of B only if maps equal")
            for sg in SIGMAS:
                for rs in RELEASE_SEEDS:
                    add("T1", s, pur, att, f"D_sigma{sg:g}_rs{rs}", sigma=sg, rs=rs)
        for s in SEEDS:
            for p2, a2 in TIER2_E1[ds]:
                add("T2-E1", s, p2, a2, "A")
                add("T2-E1", s, p2, a2, "B", scope_target=a2)
        for s in SEEDS:
            for p2, a2 in TIER2_E1[ds]:
                polset = D["purposes"][p2]["disallowed_attrs"]
                shared = " (same fitted C map as the Tier-1 C unit of this purpose)" if p2 == pur else ""
                add("T2-E2", s, p2, a2, "C", scope_target=a2, scope_policy="+".join(polset),
                    note="one C map per (seed, purpose), scored per attribute" + shared)
        for s in SEEDS:
            for p2, a2 in TIER2_E1[ds]:
                for sg in SIGMAS:
                    for rs in RELEASE_SEEDS:
                        add("T2-E3", s, p2, a2, f"D_sigma{sg:g}_rs{rs}", sigma=sg, rs=rs)
    write_csv(NOTES / "support_counts.csv", sc_rows)
    write_csv(NOTES / "support_pairs.csv", sp_rows)
    write_csv(NOTES / "scale.csv", scale_rows)
    write_csv(NOTES / "PANEL_draft.csv", panel)
    # extension inventory (hash only)
    ext = json.loads((NOTES / "checkpoint_extraction.json").read_text())["extension_inventory_round5_round7"]
    write_csv(NOTES / "extension_inventory.csv",
              [{"member": f"fl-PCRL-main-checkpoints.tar::{m}", "sha256": v["sha256"], "size": v["size"],
                "lineage_label": "Round-7 (Diabetes headline)" if "ROUND7" in m else "Round-5 (NeurIPS headline)",
                "status": "inventoried (hash only, not extracted); separate labelled extension; never a Round-4 row"}
               for m, v in ext.items()])
    from collections import Counter
    print(Counter((r["tier"], r["status"]) for r in panel))


if __name__ == "__main__":
    main()
