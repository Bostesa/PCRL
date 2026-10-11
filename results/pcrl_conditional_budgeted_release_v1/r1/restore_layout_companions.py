"""Restore-layout engineering supplement; no scientific source or law changes."""
from pathlib import Path
import argparse,json
from conditional_release.scheduler import assert_active_from_env
from conditional_release.custody import uncached_hash,copy_pinned
from conditional_release.source import write_once

def run(config):
    active=assert_active_from_env(stage="custody")
    cfg=json.loads(Path(config).read_text())
    if cfg["unit_id"]!=active["unit_id"]:raise PermissionError("unit differs")
    root=Path(cfg["restore_root"]).resolve()
    manifest=root/"runtime/SOURCE_PATHS.json"
    if uncached_hash(manifest)!=cfg["manifest_sha256"]:raise PermissionError("runtime manifest differs")
    m=json.loads(manifest.read_text());added=[]
    for seed in (0,1):
        for arm in ("V_J","V_S21"):
            policy=m["seeds"][str(seed)]["policies"][arm]
            package=Path(policy["package"]["path"]).resolve();receipt=Path(policy["receipt"]["path"]).resolve()
            if not package.is_relative_to(root) or not receipt.is_relative_to(root):raise PermissionError("original path refused")
            if uncached_hash(package)!=policy["package"]["sha256"] or uncached_hash(receipt)!=policy["receipt"]["sha256"]:raise PermissionError("restored policy pin differs")
            r=json.loads(receipt.read_text())
            if r["policy_package_sha256"]!=policy["package"]["sha256"]:raise PermissionError("receipt package binding differs")
            target=receipt.parent/"policy.npz"
            if target.exists():raise FileExistsError("preserve existing supplement")
            copy_pinned(package,target,policy["package"]["sha256"],package.stat().st_size)
            added.append({"seed":seed,"arm":arm,"source_alias":str(package.relative_to(root)),"added_alias":str(target.relative_to(root)),"sha256":uncached_hash(target),"size_bytes":target.stat().st_size})
    if uncached_hash(manifest)!=cfg["manifest_sha256"]:raise PermissionError("runtime manifest changed")
    result={"schema":"CBR1-restored-policy-colocation-supplement-v1","status":"PASS","unit_id":active["unit_id"],"amendment_sha256":cfg["amendment_sha256"],"source_manifest_unchanged":True,"source_manifest_sha256":cfg["manifest_sha256"],"original_private_artifacts_read":False,"scientific_source_changed":False,"policy_laws_changed":False,"uncached_full_readback":True,"added_companions":added}
    write_once(cfg["public_output"],result,False)
    print(json.dumps({"status":"PASS","companions":len(added)}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args();run(a.config)
