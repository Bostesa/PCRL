"""Private configuration preparation after coordinator stage freezes.

No optimizer/reader is run. INNER SEX is not even hashed before the exact
global ALL_READERS_FROZEN reporting lock has been checked. Queue execution
remains the coordinator's responsibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from .source import load_manifest, write_once

BASE = Path(__file__).resolve().parents[1]
PRIVATE = BASE / ".private/conditional_release/r5"
PUBLIC = BASE / "results/pcrl_conditional_budgeted_release_v1/r5"
FIT_CODE = ("conditional_release/attacks.py", "conditional_release/metrics.py", "conditional_release/attack_worker.py",
            "conditional_release/source.py", "conditional_release/scheduler.py", "conditional_release/fit.py",
            "conditional_release/contexts.py", "conditional_release/preparation.py", "conditional_release/templates.py",
            "conditional_release/secondary.py", "conditional_release/witness.py", "conditional_release/controls.py", "conditional_release/lp.py",
            "conditional_release/control_wide_policy.py",
            "incumbent_view/r2_package.py", "incumbent_view/r2_policy.py", "incumbent_view/r2_space.py")
REPORT_CODE = ("conditional_release/attacks.py", "conditional_release/metrics.py", "conditional_release/attack_worker.py",
               "conditional_release/source.py", "conditional_release/scheduler.py")
INFERENCE_CODE = (*REPORT_CODE, "conditional_release/inference.py", "conditional_release/inference_worker.py")


def pin(path):
    value = Path(path)
    if value.is_symlink():
        raise PermissionError("source cannot be a symlink")
    value = value.resolve(strict=True)
    with value.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path":str(value), "sha256":digest}


def check_pin(item):
    actual = pin(item["path"])
    if actual["sha256"] != item["sha256"]:
        raise PermissionError("registered artifact bytes differ")
    return actual


def unique_sources(items):
    by_path = {}
    for item in items:
        val = check_pin(item)
        if val["path"] in by_path and by_path[val["path"]] != val["sha256"]:
            raise PermissionError("two source identities for the same path")
        by_path[val["path"]] = val["sha256"]
    return [{"path":path, "sha256":digest} for path, digest in sorted(by_path.items())]


def amendment(stage):
    state = json.loads((BASE / ".private/conditional_release/compute/state.json").read_text())
    matches = [item["sha256"] for item in state["amendments"] if stage in item["stages"]]
    return matches[-1] if matches else None


def unit(config, *, phase, code, extra_sources, deps=(), rss=2, cpu=600, wall=600, output_bytes=128*1024**2):
    uid = config["unit_id"]
    path = PRIVATE / "configs" / (uid + ".json")
    write_once(path, config)
    sources = [pin(BASE / name) for name in code] + [pin(path), *extra_sources]
    spec = {"unit_id":uid, "stage":"attacks", "cwd":str(BASE),
            "argv":[sys.executable, "-m", "conditional_release.attack_worker",
                    "--phase",phase,"--config",str(path)], "max_wall_s":wall,"cpu_reserve_s":cpu,
            "max_rss_gib":rss, "projected_output_bytes":output_bytes, "sources":unique_sources(sources),
            "deps":list(deps), "outputs":[config["public_output"]]}
    latest = amendment("attacks")
    if latest:
        spec["amendment_sha256"] = latest
    spec_path = PRIVATE / "specs" / (uid + ".json")
    write_once(spec_path, spec)
    return {"unit_id":uid,"config":pin(path),"spec":pin(spec_path),"private_output":config["private_output"],
            "public_output":config["public_output"]}


def prepare_fit(index_path, output):
    index = json.loads(Path(index_path).read_text())
    source = check_pin(index["source_manifest"])
    manifest = load_manifest(source["path"], source["sha256"])
    science, attack = check_pin(index["science_lock"]), check_pin(index["attack_lock"])
    lock = json.loads(Path(attack["path"]).read_text())
    if lock.get("status") != "RELEASES_AND_ENDPOINTS_FROZEN" or lock.get("K") != 552:
        raise PermissionError("fit preparation requires release/endpoint lock first")
    grouped = {}
    for item in index["instances"]:
        if type(item["seed"]) is not int or item["seed"] not in (0,1) or type(item["point_eligible"]) is not bool:
            raise PermissionError("paired seed and point-gate types differ")
        pair = grouped.setdefault(item["arm_id"], {})
        if item["seed"] in pair:
            raise PermissionError("duplicate arm/seed identity")
        pair[item["seed"]] = item
    prepared = []
    for arm in sorted(grouped):
        pair = grouped[arm]
        if set(pair) != {0, 1} or not all(item["point_eligible"] for item in pair.values()):
            prepared.append({"arm_id":arm,"status":"UNTRIGGERED_POINT_GATE","reason":"both source seeds must pass original point utility"})
            continue
        for seed in (0, 1):
            item = pair[seed]
            uid = item.get("reader_unit_id", f"R5_FIT_{arm}_S{seed}_V1")
            cfg = {"unit_id":uid,"arm_id":arm,"seed":seed,"source_manifest":source["path"],
                "source_manifest_sha256":source["sha256"],"science_lock":science["path"],"science_lock_sha256":science["sha256"],
                "attack_lock":attack["path"],"attack_lock_sha256":attack["sha256"],"law":check_pin(item["attack_law"]),
                "policy":item["policy"],"policy_pins":item.get("policy_pins",[]),
                "private_output":str(PRIVATE / "readers" / uid),"public_output":str(PUBLIC / (uid + ".json"))}
            seed_sources = manifest["seeds"][str(seed)]
            extra = [source,science,attack,cfg["law"],seed_sources["public_bank"],
                     seed_sources["roles"]["ATTACK_FIT_NEW"]["release"],manifest["roles"]["ATTACK_FIT_NEW"]["sex"],
                     *cfg["policy_pins"]]
            # Restored source PolicySpace/donor dependencies are public fitted
            # parameters, not barred role arrays or a private input prior.
            for original in seed_sources["policies"].values():
                extra.extend((original["receipt"],original["package"]))
            entry = unit(cfg,phase="fit",code=FIT_CODE,extra_sources=extra,deps=item.get("deps",[]))
            prepared.append({"arm_id":arm,"seed":seed,"status":"PREPARED_FIT", "inner_law":item["inner_law"],
                             "utility_rows":item["utility_rows"],"policy":item["policy"],
                             "policy_pins":item.get("policy_pins",[]),"attack_law":cfg["law"], **entry})
    result = {"schema":"CBR1-r5-prepared-reader-index-v1","source_manifest":source,"science_lock":science,
              "attack_lock":attack,"instances":prepared,"source_index_sha256":pin(index_path)["sha256"],
              "additional_utility_inputs":index.get("additional_utility_inputs",[])}
    write_once(output,result)
    return result


def prepare_report(index_path, reporting_lock, output, *, unit_version=1):
    if type(unit_version) is not int or unit_version not in (1,2):
        raise PermissionError("report metadata version must be one or two")
    index = json.loads(Path(index_path).read_text())
    reporting = check_pin(pin(reporting_lock))
    lock = json.loads(Path(reporting["path"]).read_text())
    if lock.get("status") != "ALL_READERS_FROZEN" or lock.get("K") != 552:
        raise PermissionError("do not hash INNER labels before every global reader is frozen")
    required = {str((Path(item["private_output"])/"FROZEN_COMPLETE.json").resolve())
                for item in index["instances"] if item["status"] == "PREPARED_FIT"}
    admitted = {str(Path(item["path"]).resolve()) for item in lock["reader_freezes"]}
    if not required or not required.issubset(admitted):
        raise PermissionError("global freeze omits a prepared reader bank")
    # Frozen receipts are model/role metadata only. Check ALL before constructing
    # a queue spec that will hash the separately authorized INNER SEX file.
    for frozen in lock["reader_freezes"]:
        check_pin(frozen)
        if json.loads(Path(frozen["path"]).read_text()).get("status") != "FROZEN_COMPLETE":
            raise PermissionError("a registered final reader remains unfrozen")
    sex = lock["inner_sex_pin"]
    if sex["sha256"] != lock["inner_sex_sha256"]:
        raise PermissionError("reporting authorization has inconsistent one-role label pin")
    # The exact role-only archive is copied only after every frozen receipt
    # above passes. Preserve source bytes, so restored replay keeps the original
    # label binding without walking any predecessor's private directories.
    sex = check_pin(sex)
    custody = PRIVATE / "reporting_label_custody"
    custody.mkdir(parents=True,exist_ok=True,mode=0o700)
    archived = custody / "INNER_SELECTION_SEX.npz"
    if not archived.exists():
        with Path(sex["path"]).open("rb") as source_stream, archived.open("xb") as target:
            os.chmod(archived,0o600)
            while block := source_stream.read(1024**2):
                target.write(block)
            target.flush();os.fsync(target.fileno())
    copied = pin(archived)
    if copied["sha256"] != sex["sha256"]:
        raise PermissionError("reporting label custody copy differs from authorized bytes")
    custody_receipt = custody / "CUSTODY_RECEIPT.json"
    if not custody_receipt.exists():
        write_once(custody_receipt,{"schema":"CBR1-post-reader-freeze-label-custody-v1",
            "source_pin":sex,"copy_pin":copied,"reporting_lock":reporting,
            "authorized_only_after_all_reader_freezes":True})
    sex = copied
    source = index["source_manifest"]
    manifest = load_manifest(source["path"], source["sha256"])
    prepared = []
    for item in index["instances"]:
        if item["status"] != "PREPARED_FIT":
            prepared.append(item);continue
        arm, seed = item["arm_id"], item["seed"]
        uid = f"R5_REPORT_{arm}_S{seed}_V{unit_version}"
        root = Path(item["private_output"])
        cfg = {"unit_id":uid,"arm_id":arm,"seed":seed,"source_manifest":source["path"],
            "source_manifest_sha256":source["sha256"],"reporting_lock":reporting["path"],"reporting_lock_sha256":reporting["sha256"],
            "law":check_pin(item["inner_law"]),"frozen_receipt":pin(root / "FROZEN_COMPLETE.json"),
            "score_tables":pin(root / "FROZEN_SCORE_TABLES.npz"),"inner_sex":sex,
            "private_output":str(PRIVATE / "reports" / uid),"public_output":str(PUBLIC / (uid + ".json"))}
        extra = [source,reporting,cfg["law"],cfg["frozen_receipt"],cfg["score_tables"],sex,*lock["reader_freezes"],
            manifest["seeds"][str(seed)]["public_bank"],manifest["seeds"][str(seed)]["roles"]["INNER_SELECTION"]["release"]]
        entry = unit(cfg,phase="report",code=REPORT_CODE,extra_sources=extra,deps=[item["unit_id"]],
                     cpu=240,wall=240,output_bytes=256*1024**2)
        prepared.append({"arm_id":arm,"seed":seed,"status":"PREPARED_REPORT","reader_private_output":str(root),
                         "utility_rows":item["utility_rows"],"policy":item["policy"],
                         "policy_pins":item.get("policy_pins",[]),"attack_law":item["attack_law"],**entry})
    result = {**index,"reporting_lock":reporting,"instances":prepared}
    write_once(output,result)
    return result


def prepare_inference(index_path, endpoint_family, output):
    index = json.loads(Path(index_path).read_text())
    family = pin(endpoint_family)
    if json.loads(Path(family["path"]).read_text()).get("K") != 552:
        raise PermissionError("exact K552 endpoint family required")
    recovery, utility, deps = [], [], []
    for item in index["instances"]:
        if item["status"] != "PREPARED_REPORT":
            continue
        key = {"arm_id":item["arm_id"],"seed":item["seed"]}
        recovery.append({**key,"pin":pin(Path(item["private_output"]) / "EXPECTED_SCORE_LAWS.npz")})
        utility.append({**key,"pin":check_pin(item["utility_rows"])})
        deps.append(item["unit_id"])
    # Utility inputs from ineligible arms still belong to the registered
    # uncertainty family even when their final-reader dependency was not run.
    for item in index.get("additional_utility_inputs", []):
        utility.append({**item,"pin":check_pin(item["pin"])})
    uid = "R5_PRIMARY_INFERENCE_V1"
    cfg = {"unit_id":uid,"endpoint_family":family,"reporting_lock":index["reporting_lock"]["path"],
        "reporting_lock_sha256":index["reporting_lock"]["sha256"],"recovery_inputs":recovery,"utility_inputs":utility,
        "private_output":str(PRIVATE / "inference" / uid),
        "public_csv":str(BASE / "results/pcrl_conditional_budgeted_release_v1/PRIMARY_ENDPOINTS.csv"),
        "public_receipt":str(PUBLIC / (uid + ".json"))}
    cfg_path = PRIVATE / "configs" / (uid + ".json");write_once(cfg_path,cfg)
    spec = {"unit_id":uid,"stage":"attacks","cwd":str(BASE),
        "argv":[sys.executable,"-m","conditional_release.inference_worker","--config",str(cfg_path)],
        "max_wall_s":1800,"cpu_reserve_s":1800,"max_rss_gib":2,"projected_output_bytes":64*1024**2,"deps":deps,
        "sources":unique_sources([*[pin(BASE / name) for name in INFERENCE_CODE],pin(cfg_path),family,index["reporting_lock"],
                                  *[item["pin"] for item in recovery],*[item["pin"] for item in utility]]),
        "outputs":[cfg["public_csv"],cfg["public_receipt"]],"amendment_sha256":amendment("attacks")}
    spec_path = PRIVATE / "specs" / (uid + ".json");write_once(spec_path,spec)
    result = {"unit_id":uid,"config":pin(cfg_path),"spec":pin(spec_path)}
    write_once(output,result)
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--phase",choices=("fit","report","inference"),required=True)
    parser.add_argument("--index",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--reporting-lock")
    parser.add_argument("--endpoint-family")
    parser.add_argument("--unit-version",type=int,choices=(1,2),default=1)
    args = parser.parse_args()
    if args.phase == "fit":
        result = prepare_fit(args.index,args.output)
    elif args.phase == "report":
        result = prepare_report(args.index,args.reporting_lock,args.output,unit_version=args.unit_version)
    else:
        result = prepare_inference(args.index,args.endpoint_family,args.output)
    print(json.dumps({"status":"PREPARED","instances":len(result.get("instances",[])),"unit_id":result.get("unit_id")}))


if __name__ == "__main__":
    main()
