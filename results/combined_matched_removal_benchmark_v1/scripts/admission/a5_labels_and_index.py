"""A5: freeze purposes, decide the defense_fit route from provenance only, write labels/roles and INPUTS_INDEX.json.

No model is fitted and no outcome is read: the route decision uses only rows_provenance.json (regeneration
evidence) and lineage.json (checkpoint lineage + the best.pt test-row fingerprint), both written before any
benchmark fit.

Route rule (BENCH_DESIGN "Roles"):
  PREFERRED  defense_fit = PCRL train-split rows (b96c412 loader) whose canon_key does not occur in any test row;
             the others are 'excluded_dup'. Requires: train split regenerated deterministically AND verified against
             what training used (Adult: raw-row index == preparation, raw_df == raw file rows; HMDA: regenerated
             arrays == local processed arrays incl. norm/label stats; both: every seed's best.pt test fingerprint
             matches per_seed_results.json and global_step == 205*ceil(n_train/256)).
  FALLBACK   (only if any requirement fails) attacker_fit units with sha256('bench-defense-fallback-v1|'+key)
             < 0.30 become defense_fit; train rows become 'unused_train'.
Outputs (private): <ds>_labels.npz, <ds>_roles.npz, INPUTS_INDEX.json. Repo: notes/admission/purposes.json,
notes/admission/defense_route.json, notes/admission/pilot_roles_check.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admission_common import (B96, CKPT_DIR, DATASETS, FALLBACK_SALT, INPUTS, NOTES, PILOT, SEEDS, TIER1,  # noqa
                              ensure_export, sha256_array, sha256_file, u01, write_json)

ROLES = ("defense_fit", "attacker_fit", "attacker_val", "assessment")


def purposes(ds):
    ensure_export()
    if ds == "adult":
        from pcrl.data.adult import get_adult_purposes
        P = get_adult_purposes()
    else:
        from pcrl.data.hmda import get_hmda_purposes
        P = get_hmda_purposes()
    return P


def route_decision(prov, lin):
    dec = {}
    for ds in DATASETS:
        p = prov[ds]
        seeds = lin["datasets"][ds]["seeds"]
        req = {"all_seed_fingerprints_match": all(v["best_fingerprint"]["health_match(rel<1e-4)"]
                                                  and v["best_fingerprint"]["task_acc_match(abs<=1/n)"]
                                                  for v in seeds.values()),
               "global_step_consistent_all_seeds": all(v["final"]["checks"]["global_step_eq_205_x_ceil(n_train/256)"]
                                                       for v in seeds.values())}
        if ds == "adult":
            req["train_raw_row_index_matches_preparation"] = p["train_raw_row_index_matches_preparation"]
            req["loader_raw_df_equals_raw_file_rows"] = p["loader_raw_df_equals_raw_file_rows"]
            req["input_files_match_preparation"] = all(v for k, v in prov["input_sha256_matches_preparation"].items()
                                                       if k.startswith("adult"))
        else:
            req["regenerated_equals_local_processed_all_arrays"] = all(
                all(v for k, v in sp.items() if k != "n") for sp in p["regenerated_vs_local_processed"].values())
            req["metadata_stats_equal_regenerated"] = (p["metadata_norm_stats_equal_regenerated"]
                                                       and p["metadata_label_stats_equal_regenerated"])
            req["input_files_match_preparation"] = all(v for k, v in prov["input_sha256_matches_preparation"].items()
                                                       if k.startswith("hmda"))
        dec[ds] = {"requirements": req, "route": "PREFERRED" if all(req.values()) else "FALLBACK"}
    return dec


def main():
    prov = json.loads((NOTES / "rows_provenance.json").read_text())
    lin = json.loads((NOTES / "lineage.json").read_text())
    fwd = json.loads((NOTES / "forward.json").read_text())
    dec = route_decision(prov, lin)
    write_json(NOTES / "defense_route.json", {"decided_from": ["notes/admission/rows_provenance.json",
                                                               "notes/admission/lineage.json"],
                                              "decided_before_any_benchmark_fit": True, "datasets": dec})
    index = {"schema": "bench_v1.inputs_index/v1", "generated_by": "scripts/admission/a5_labels_and_index.py",
             "pcrl_loader_commit": B96, "row_id_convention": "test rows 0..n_test-1 (test-split position; = pilot "
             "row_id for Adult); train rows n_test + train-split position",
             "unit_convention": "de-duplicated record by canon_key, shared within and across splits; Adult canon_key "
             "strips the trailing '.' of adult.test income labels; HMDA canon_key == record_key; no households",
             "role_values": list(ROLES) + ["excluded_dup", "unused_train"],
             "note_excluded_dup": "train rows whose canon_key occurs in a test role; they share a unit with that "
             "test row and must be dropped before any admission/fit (never defense_fit)",
             "datasets": {}}
    pur_rec = {}
    pilot_chk = {}
    for ds in DATASETS:
        P = purposes(ds)
        R = np.load(INPUTS / f"{ds}_rows.npz")
        split, ck, rk = R["split"], R["canon_key"], R["record_key"]
        role = np.where(split == "test", R["test_role"], "").astype("<U12")
        te_keys = set(ck[split == "test"].tolist())
        tr = split == "train"
        if dec[ds]["route"] == "PREFERRED":
            dup = np.array([k in te_keys for k in ck]) & tr
            role[tr & ~dup] = "defense_fit"
            role[dup] = "excluded_dup"
        else:
            move = (role == "attacker_fit") & np.array([u01(FALLBACK_SALT + k) < 0.30 for k in rk])
            role[move] = "defense_fit"
            role[tr] = "unused_train"
        assert not (role == "").any()
        # a unit never spans two scored roles
        unit = R["unit"]
        for r1 in ROLES:
            for r2 in ROLES:
                if r1 < r2:
                    assert not set(unit[role == r1].tolist()) & set(unit[role == r2].tolist()), (ds, r1, r2)
        attrs = sorted({a for p in P for a in p.disallowed_attrs})
        tasks = [p.allowed_tasks[0] for p in P]
        lab = {"row_id": R["row_id"], "split": split, "unit": unit, "record_key": rk, "canon_key": ck, "role": role}
        for a in attrs:
            lab[a] = R[a]
        for t in tasks:
            lab[f"task_{t}"] = R[f"task_{t}"]
        np.savez(INPUTS / f"{ds}_labels.npz", **lab)
        roles_npz = {}
        for r in ROLES:
            roles_npz[f"{r}__row_id"] = R["row_id"][role == r]
            roles_npz[f"{r}__unit"] = unit[role == r]
        roles_npz["excluded_dup__row_id"] = R["row_id"][role == "excluded_dup"]
        np.savez(INPUTS / f"{ds}_roles.npz", **roles_npz)
        # purposes frozen from b96c412 + checkpoint
        pmap = {}
        for i, p in enumerate(P):
            pmap[p.name] = {"index": i, "task": p.allowed_tasks[0], "task_dim": p.allowed_task_dims[p.allowed_tasks[0]],
                            "disallowed_attrs": list(p.disallowed_attrs),
                            "disallowed_attr_dims": dict(p.disallowed_attr_dims),
                            "rep_key": f"rep_p{i}", "logits_key": f"logits_{p.name}",
                            "labels_task_key": f"task_{p.allowed_tasks[0]}"}
        seeds_lin = lin["datasets"][ds]["seeds"]
        ck_agree = all(v["final"]["checks"]["head_order_eq_purposes"]
                       and v["final"]["checks"]["lambda_key_order_eq_b96c412_purpose_pairs"] for v in seeds_lin.values())
        t1 = TIER1[ds]
        pur_rec[ds] = {"source": f"PCRL {B96[:7]} pcrl/data/{ds}.py get_{ds}_purposes() order == checkpoint "
                                 "task_heads order == checkpoint lambdas key order == LoRA adapter index",
                       "checkpoint_order_agrees_all_seeds": ck_agree, "purposes": pmap,
                       "tier1": {**t1, "purpose_index": pmap[t1["purpose"]]["index"],
                                 "policy_equals_disallowed": sorted(t1["policy"]) ==
                                 sorted(pmap[t1["purpose"]]["disallowed_attrs"]),
                                 "target_dims": pmap[t1["purpose"]]["disallowed_attr_dims"][t1["target"]]}}
        if not ck_agree or not pur_rec[ds]["tier1"]["policy_equals_disallowed"]:
            raise SystemExit(f"{ds}: purpose freeze disagreement")
        # pilot check (Adult)
        if ds == "adult":
            pl = np.load(PILOT / "labels.npz")
            pf = np.load(PILOT / "features.npz")
            F = np.load(INPUTS / "adult_features.npz")
            pos = {int(x): i for i, x in enumerate(R["row_id"])}
            ix = np.array([pos[int(x)] for x in pl["row_id"]])
            # pilot units are within-test np.unique inverses: compare partitions
            pu, ou = pl["unit"], unit[ix]
            part_eq = len(set(zip(pu.tolist(), ou.tolist()))) == len(set(pu.tolist())) == len(set(ou.tolist()))
            pilot_chk = {"pilot_labels_sha256": sha256_file(PILOT / "labels.npz"), "n_pilot_rows": int(len(ix)),
                         "all_pilot_rows_are_test": bool((split[ix] == "test").all()),
                         "roles_equal_by_row_id": bool(np.array_equal(pl["role"], role[ix])),
                         "record_keys_equal_by_row_id": bool(np.array_equal(pl["record_key"], rk[ix])),
                         "unit_partition_equal": bool(part_eq),
                         "sensitive_attrs_equal": {a: bool(np.array_equal(pl[a], R[a][ix])) for a in attrs
                                                   if a in pl.files},
                         "features_equal": bool(np.array_equal(pf["features"], F["features"][ix])
                                                and np.array_equal(pf["row_id"], F["row_id"][ix])),
                         "pilot_role_counts": {r: int((pl["role"] == r).sum()) for r in ROLES[1:]}}
            write_json(NOTES / "pilot_roles_check.json", pilot_chk)
            if not (pilot_chk["roles_equal_by_row_id"] and pilot_chk["record_keys_equal_by_row_id"]):
                raise SystemExit("Adult roles differ from the pilot")
        # index entry
        lf = INPUTS / f"{ds}_labels.npz"
        enc = {}
        for s in SEEDS:
            L = seeds_lin[str(s)]["final"]
            fr = fwd["runs"].get(f"{ds}_s{s}", {})
            fpath = INPUTS / f"{ds}_s{s}_forward.npz"
            enc[str(s)] = {"checkpoint": str(CKPT_DIR / f"v2_{ds}_s{s}_final.pt"), "checkpoint_sha256": L["sha256"],
                           "lineage_status": "ADMITTED" if L["admitted"] else "MISSING",
                           "lineage_record": "results/combined_matched_removal_benchmark_v1/notes/admission/lineage.json",
                           "forward_npz": str(fpath), "forward_sha256": sha256_file(fpath) if fpath.exists() else None,
                           "forward_sha256_matches_forward_json": (fr.get("sha256") == sha256_file(fpath))
                           if fpath.exists() else None,
                           "forward_keys": ["row_id", "split"] + [f"rep_p{i}" for i in range(len(P))]
                           + [f"logits_{p.name}" for p in P],
                           "forward_dtype": "float64 (exact upcast of the float32 frozen forward)"}
        role_info = {}
        for r in ROLES + ("excluded_dup",):
            m = role == r
            role_info[r] = {"n_rows": int(m.sum()), "n_units": int(len(np.unique(unit[m]))),
                            "row_id_sha256": sha256_array(R["row_id"][m].astype(np.int64)),
                            "unit_sha256": sha256_array(unit[m].astype(np.int64)),
                            "splits": sorted(set(split[m].tolist()))}
        index["datasets"][ds] = {
            "defense_route": dec[ds]["route"],
            "labels_npz": str(lf), "labels_sha256": sha256_file(lf),
            "labels_keys": list(lab.keys()),
            "roles_npz": str(INPUTS / f"{ds}_roles.npz"), "roles_sha256": sha256_file(INPUTS / f"{ds}_roles.npz"),
            "record_key_array": {"file": str(lf), "key": "record_key", "canon_key": "canon_key",
                                 "sha256_record_key": sha256_array(rk), "sha256_canon_key": sha256_array(ck)},
            "role_array": {"file": str(lf), "key": "role", "unit_key": "unit", "sha256": sha256_array(role)},
            "roles": role_info,
            "features_npz": str(INPUTS / f"{ds}_features.npz"),
            "features_sha256": sha256_file(INPUTS / f"{ds}_features.npz"),
            "rows_npz": str(INPUTS / f"{ds}_rows.npz"), "rows_sha256": sha256_file(INPUTS / f"{ds}_rows.npz"),
            "purposes": pmap, "tier1": pur_rec[ds]["tier1"],
            "sensitive_label_keys": attrs, "task_label_keys": [f"task_{t}" for t in tasks],
            "encoders": enc,
        }
        print(ds, dec[ds]["route"], {r: v["n_rows"] for r, v in role_info.items()})
    write_json(NOTES / "purposes.json", pur_rec)
    write_json(INPUTS / "INPUTS_INDEX.json", index)
    print("pilot check", {k: v for k, v in pilot_chk.items() if k != "pilot_labels_sha256"})


if __name__ == "__main__":
    main()
